#!/usr/bin/env python3
"""camd.py - the camera owner on the Jetson (docs/scripted-demos.md, phase 2). Stdlib only.

TODAY (dashboard/server.py jetson_cam_loop): the Mac dashboard starts ONE ffmpeg per camera over ssh, ffmpeg owns the V4L2 device and pipes 15 fps MJPEG
(q:v 6) back to the Mac. A UVC device has one streaming owner, so no second process on the Jetson can read the same camera while that ffmpeg runs.
SO camd becomes the owner and fans out:

    camd.py serve                    one ffmpeg per camera (the same command line the dashboard runs, at 30 fps), newest JPEG + time.monotonic() per camera,
                                     a unix socket (default /dev/shm/so101_camd.sock) that any number of readers subscribe to
    camd.py tap NAME [--fps 15]      a reader that writes the camera's JPEGs to stdout, back to back: the exact stream the dashboard's ffmpeg gave it.
                                     Exit 3 = no camd answers (the dashboard's command then falls back to its own ffmpeg, so a dead camd costs one reconnect)
    demo_cams.CamdCameraSource       the recorder's reader (30 fps, with the capture time)

Wire format of a subscription: the client sends one JSON line {"cam": "top", "fps": 30}; the server then sends, per frame, a 12-byte header
(">dI": capture time as time.monotonic() seconds, JPEG length) and the JPEG. {"status": 1} gets one JSON line back instead.
A slow reader loses frames, it never slows the capture or another reader (each reader thread takes the newest frame when it is ready).
camd yields the devices while ~/so101/.cams_busy exists, like the dashboard's own command (so101-record, cam_lock and cam_health set it).
It never opens the arm's serial port and never writes anything but its socket.
"""
import json, os, socket, struct, subprocess, sys, threading, time

SOCK = os.environ.get("SO101_CAMD_SOCK", "/dev/shm/so101_camd.sock")
BUSY = os.path.expanduser("~/so101/.cams_busy")
HDR = struct.Struct(">dI")
# same devices and formats as dashboard/server.py JETSON_CAMS (usb by-id paths); the sizes are what the recorder's features are built from
CAMS = {"top": ("usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0", "mjpeg", "640x360"),
        "front": ("usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0", "mjpeg", "640x360"),
        "wrist": ("usb-icSpring_icspring_camera-video-index0", "yuyv422", "640x480")}
STALE_S = 5.0   # a reader gets disconnected when its camera produced nothing for this long (the dashboard's loop then reconnects, or falls back)
# why: the line the dashboard command gains in front of its ffmpeg, 2026-11-11. tests/jetson/test_camd.py runs this snippet in sh with stubs for both outcomes.
DASH_PREFIX = "python3 ~/so101/camd.py tap {name}; rc=$?; [ $rc -eq 3 ] || exit $rc; "


def ffmpeg_cmd(dev, fmt, size, fps, q): return ["ffmpeg", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay", "-f", "v4l2", "-input_format", fmt, "-video_size", size, "-framerate", str(fps),
                                                  "-i", f"/dev/v4l/by-id/{dev}", "-f", "mjpeg", "-q:v", str(q), "-"]


def split_jpegs(buf):
    """(frames, rest): complete JPEGs found in `buf` (SOI ff d8 ... EOI ff d9), as the dashboard's reader finds them."""
    out = []
    while True:
        i = buf.find(b"\xff\xd8"); j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
        if i < 0 or j < 0: return out, (buf[-2:] if i < 0 else buf[i:])
        out.append(buf[i:j + 2]); buf = buf[j + 2:]


class Camera:
    """One camera: a worker thread keeps ffmpeg running and `latest` = (seq, t_mono, jpeg). `cmd_fn(name)` -> argv, or None when the device is missing (a test passes a fake)."""
    def __init__(self, name, cmd_fn, busy=BUSY, latency_s=0.0, clock=time.monotonic, retry_s=2.0):
        self.name, self.cmd_fn, self.busy, self.latency, self.clock, self.retry = name, cmd_fn, busy, latency_s, clock, retry_s
        self.latest = (0, 0.0, b""); self.cond = threading.Condition(); self.stop = False; self.proc = None; self.restarts = 0; self.err = None

    def publish(self, jpg):
        with self.cond: self.latest = (self.latest[0] + 1, self.clock() - self.latency, jpg); self.cond.notify_all()

    def run(self):
        while not self.stop:
            if os.path.exists(self.busy): time.sleep(0.2); continue
            try:
                argv = self.cmd_fn(self.name)
                if argv is None: self.err = "no device"; time.sleep(self.retry); continue
                self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL); self.err = None
                threading.Thread(target=self.watch, args=(self.proc,), daemon=True).start(); buf = b""
                while not self.stop:
                    chunk = os.read(self.proc.stdout.fileno(), 65536)
                    if not chunk: break
                    frames, buf = split_jpegs(buf + chunk)
                    for jpg in frames: self.publish(jpg)
                self.proc.wait()
            except Exception as e: self.err = str(e)[:120]
            self.restarts += 1; time.sleep(self.retry)

    def watch(self, proc):
        # why: yield the device within half a second when a lerobot job or cam_lock sets .cams_busy, like the dashboard's own shell loop does
        while proc.poll() is None and not self.stop:
            if os.path.exists(self.busy): proc.kill(); return
            time.sleep(0.1)

    def close(self):
        self.stop = True
        try: self.proc and self.proc.kill()
        except Exception: pass


def dev_cmd(fps=30, q=4):
    def fn(name):
        dev, fmt, size = CAMS[name]
        return ffmpeg_cmd(dev, fmt, size, fps, q) if os.path.exists(f"/dev/v4l/by-id/{dev}") else None
    return fn


class Server:
    def __init__(self, cams, path=SOCK, stale_s=STALE_S):
        self.cams, self.path, self.stale_s, self.sock = cams, path, stale_s, None; self.threads = []

    def start(self):
        try: os.unlink(self.path)
        except OSError: pass
        self.sock = socket.socket(socket.AF_UNIX); self.sock.bind(self.path); self.sock.listen(8); os.chmod(self.path, 0o600)
        for c in self.cams.values(): t = threading.Thread(target=c.run, daemon=True); t.start(); self.threads.append(t)
        threading.Thread(target=self.accept, daemon=True).start(); return self

    def stop(self):
        for c in self.cams.values(): c.close()
        try: self.sock.close()
        except Exception: pass

    def accept(self):
        while True:
            try: conn, _ = self.sock.accept()
            except OSError: return
            threading.Thread(target=self.serve_one, args=(conn,), daemon=True).start()

    def status(self):
        now = time.monotonic()
        return {"pid": os.getpid(), "cams": {n: {"seq": c.latest[0], "age_s": round(now - c.latest[1], 3) if c.latest[0] else None, "restarts": c.restarts, "err": c.err} for n, c in self.cams.items()}}

    def serve_one(self, conn):
        try:
            conn.settimeout(5.0); line = b""
            while not line.endswith(b"\n") and len(line) < 512: line += conn.recv(64)
            req = json.loads(line)
            if req.get("status"): conn.sendall((json.dumps(self.status()) + "\n").encode()); return
            cam = self.cams[req["cam"]]; fps = float(req.get("fps") or 30.0); period = 1.0 / fps; last_seq, last_t = 0, -1e9; conn.settimeout(10.0); t_wait = time.monotonic()
            while True:
                with cam.cond:
                    cam.cond.wait_for(lambda: cam.latest[0] != last_seq, timeout=1.0); seq, t, jpg = cam.latest
                if seq == last_seq:
                    if time.monotonic() - t_wait > self.stale_s: return
                    continue
                t_wait = time.monotonic(); last_seq = seq
                if t - last_t < period * 0.8: continue   # why: a 15 fps reader of a 30 fps camera takes every other frame, not a random 4 of 5
                last_t = t; conn.sendall(HDR.pack(t, len(jpg)) + jpg)
        except Exception: pass
        finally:
            try: conn.close()
            except Exception: pass


def read_frames(conn):
    """Generator of (t_mono, jpeg) from a subscribed socket; ends at EOF."""
    f = conn.makefile("rb")
    while True:
        h = f.read(HDR.size)
        if len(h) < HDR.size: return
        t, n = HDR.unpack(h); jpg = f.read(n)
        if len(jpg) < n: return
        yield t, jpg


def subscribe(cam, fps, path=SOCK, timeout=5.0):
    s = socket.socket(socket.AF_UNIX); s.settimeout(timeout); s.connect(path); s.sendall((json.dumps({"cam": cam, "fps": fps}) + "\n").encode()); s.settimeout(15.0); return s


def tap(name, fps=15.0, path=SOCK, out=None):
    """Write the camera's JPEGs to `out` until camd goes away. 3 = no camd (fall back to ffmpeg), 1 = the stream ended (the caller reconnects)."""
    out = out or sys.stdout.buffer
    try: s = subscribe(name, fps, path)
    except OSError: return 3
    try:
        for _, jpg in read_frames(s): out.write(jpg); out.flush()
    except (BrokenPipeError, OSError): return 1
    return 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve"); sv.add_argument("--fps", type=int, default=30); sv.add_argument("--quality", type=int, default=4, help="ffmpeg -q:v (2 best, 31 worst); the dashboard used 6 at 15 fps")
    sv.add_argument("--latency-ms", type=float, default=0.0, help="subtracted from each frame's time (capture to pipe delay; measure it, see the doc)")
    tp = sub.add_parser("tap"); tp.add_argument("name", choices=sorted(CAMS)); tp.add_argument("--fps", type=float, default=15.0)
    sub.add_parser("status"); a = ap.parse_args(argv)
    if a.cmd == "tap": return tap(a.name, a.fps)
    if a.cmd == "status":
        try: s = socket.socket(socket.AF_UNIX); s.settimeout(3); s.connect(SOCK); s.sendall(b'{"status": 1}\n'); print(s.makefile().readline().strip()); return 0
        except OSError as e: print(json.dumps({"up": False, "err": str(e)})); return 1
    fn = dev_cmd(a.fps, a.quality); srv = Server({n: Camera(n, fn, latency_s=a.latency_ms / 1000.0) for n in CAMS}).start(); print(f"camd up on {SOCK}", flush=True)
    try:
        while True: time.sleep(3600)
    except KeyboardInterrupt: pass
    finally:
        srv.stop()
        try: os.unlink(SOCK)
        except OSError: pass
    return 0


if __name__ == "__main__": sys.exit(main())
