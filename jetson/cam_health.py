#!/usr/bin/env python3
"""cam_health.py - frame-rate and dropped-frame check for the three recording cameras.

Opens top, front and wrist SIMULTANEOUSLY at the exact settings lerobot-record uses (SO101_CAMS in env.sh:
C920s 640x360 MJPG 30 fps, icSpring wrist 640x480 YUYV 30 fps; fourcc, then size, then fps, like lerobot's
OpenCVCamera), reads each in its own thread for N seconds and reports per camera:

  achieved fps, frame-interval p50 / p95 / max (ms), duplicate frames (consecutive frames whose strided
  sample is identical), stalls (intervals > --stall-ms, default 100), the negotiated size / fourcc / fps,
  and the Jetson's CPU load during the run (loadavg before and after, psutil cpu% when psutil is present).

Exit status: 0 healthy; 1 when any camera's fps < --min-fps (28), any stall > --max-stall-ms (250) or the
negotiated size differs from the requested one; 2 when a camera could not be opened at all.

Run on the Jetson with the dashboard's feeds paused for the duration:   so101-cams-health [seconds]
(it holds ~/so101/.cams_busy itself: the Mac dashboard's ffmpeg lets go within a second and comes back after).
`--json` prints a machine-readable report instead of the table. `--cams top,front` limits the set,
`--device wrist=/dev/video2` overrides a path.
"""
import argparse, json, os, platform, sys, threading, time

import numpy as np

try:
    import cv2
except ImportError:  # the statistics and verdicts still work without OpenCV (tests on a bare host); capture does not
    cv2 = None

BUSY_FLAG = os.path.expanduser("~/so101/.cams_busy")

# the recording configuration, kept in step with SO101_CAMS in env.sh
DEFAULT_CAMS = {
    "top":   {"path": os.environ.get("CAM_TOP", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0"), "width": 640, "height": 360, "fps": 30, "fourcc": "MJPG"},
    "front": {"path": os.environ.get("CAM_FRONT", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0"), "width": 640, "height": 360, "fps": 30, "fourcc": "MJPG"},
    "wrist": {"path": os.environ.get("CAM_WRIST", "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"), "width": 640, "height": 480, "fps": 30, "fourcc": None},
}
THRESHOLDS = {"min_fps": 28.0, "stall_ms": 100.0, "max_stall_ms": 250.0}


def log(*a): print(*a, file=sys.stderr, flush=True)


def fourcc_str(code):
    """OpenCV's float fourcc -> 'MJPG' (unprintable bytes shown as '?')."""
    try: n = int(code)
    except (TypeError, ValueError): return "????"
    if n <= 0: return "????"
    return "".join(chr(b) if 32 <= b < 127 else "?" for b in ((n >> (8 * i)) & 0xFF for i in range(4)))


# ---------------------------------------------------------------- capture sources
class CvSource:
    """A V4L device through cv2.VideoCapture, configured the way lerobot's OpenCVCamera does it.
    read() -> (frame or None, monotonic timestamp taken right after the frame arrived)."""
    def __init__(self, cfg, open_timeout=6.0):
        if cv2 is None: raise RuntimeError("OpenCV (cv2) is not installed")
        path = cfg["path"]; deadline = time.monotonic() + open_timeout
        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        while True:   # the dashboard's ffmpeg can take a moment to let go after the busy flag appears
            self.cap = cv2.VideoCapture(int(path) if str(path).isdigit() else path, backend)
            if self.cap.isOpened(): break
            self.cap.release(); self.cap = None
            if time.monotonic() >= deadline: raise RuntimeError(f"cannot open {path} (busy or unplugged)")
            time.sleep(0.5)
        if cfg.get("fourcc"): self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg["width"])); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg["height"]))
        self.cap.set(cv2.CAP_PROP_FPS, float(cfg["fps"]))

    def negotiated(self):
        g = self.cap.get
        return {"width": int(round(g(cv2.CAP_PROP_FRAME_WIDTH))), "height": int(round(g(cv2.CAP_PROP_FRAME_HEIGHT))),
                "fps": round(float(g(cv2.CAP_PROP_FPS)), 2), "fourcc": fourcc_str(g(cv2.CAP_PROP_FOURCC))}

    def read(self):
        ok, frame = self.cap.read()
        return (frame if ok else None), time.monotonic()

    def now(self): return time.monotonic()

    def close(self):
        if self.cap is not None: self.cap.release(); self.cap = None


def frame_signature(frame):
    """Cheap fingerprint for duplicate detection: a strided sample of the frame (every 8th row and column)."""
    return np.ascontiguousarray(frame[::8, ::8]).tobytes()


def sample_mad(a, b):
    """Mean absolute difference of two strided samples (the same stride as frame_signature)."""
    return float(np.mean(np.abs(a[::8, ::8].astype(np.int16) - b[::8, ::8].astype(np.int16))))


# ---------------------------------------------------------------- statistics
def interval_stats(timestamps, stall_ms=THRESHOLDS["stall_ms"]):
    """Frame-timing statistics from a sequence of arrival timestamps (seconds). Pure; drives both the real
    capture and the tests. Returns fps, interval percentiles in ms, stall count and the longest interval."""
    ts = np.asarray(timestamps, np.float64)
    n = int(ts.size)
    if n < 2:
        return {"frames": n, "fps": 0.0, "span_s": 0.0, "p50_ms": None, "p95_ms": None, "max_ms": None, "stalls": 0, "longest_stall_ms": None, "stall_ms": stall_ms}
    iv = np.diff(ts) * 1000.0; span = float(ts[-1] - ts[0])
    return {"frames": n, "fps": round((n - 1) / span, 2) if span > 0 else 0.0, "span_s": round(span, 3),
            "p50_ms": round(float(np.percentile(iv, 50)), 2), "p95_ms": round(float(np.percentile(iv, 95)), 2), "max_ms": round(float(iv.max()), 2),
            "stalls": int(np.count_nonzero(iv > stall_ms)), "longest_stall_ms": round(float(iv.max()), 2), "stall_ms": stall_ms}


def verdicts(cam, thresholds=THRESHOLDS):
    """Threshold failures for one camera's report, as human strings ([] = healthy)."""
    bad = []
    if cam.get("error"): return [f"could not capture: {cam['error']}"]
    st = cam["stats"]; req = cam["requested"]; neg = cam.get("negotiated") or {}; fs = cam.get("frame_size")
    if st["frames"] < 2: bad.append("no frames")
    elif st["fps"] < thresholds["min_fps"]: bad.append(f"fps {st['fps']} < {thresholds['min_fps']}")
    if st["longest_stall_ms"] is not None and st["longest_stall_ms"] > thresholds["max_stall_ms"]: bad.append(f"stall {st['longest_stall_ms']} ms > {thresholds['max_stall_ms']} ms")
    got = fs or ((neg["width"], neg["height"]) if neg else None)
    if got and tuple(got) != (req["width"], req["height"]): bad.append(f"size {got[0]}x{got[1]} != requested {req['width']}x{req['height']}")
    if req.get("fourcc") and neg.get("fourcc") and neg["fourcc"] != req["fourcc"]: bad.append(f"fourcc {neg['fourcc']} != requested {req['fourcc']}")
    return bad


def exit_code(report):
    if any(c.get("error") for c in report["cameras"].values()): return 2
    return 1 if any(c["problems"] for c in report["cameras"].values()) else 0


# ---------------------------------------------------------------- capture
def capture(source, seconds, warmup=1.0, stall_ms=THRESHOLDS["stall_ms"], dup_mad=0.0):
    """Read `source` for warmup+seconds, keeping timestamps and duplicate flags from the measurement window only.
    `source.read()` returns (frame|None, t) and `source.now()` the same clock, so a fake source can script time."""
    t_start = source.now(); t_meas = t_start + warmup; t_end = t_meas + seconds
    ts, dups, failed, frame_size = [], 0, 0, None
    prev_sig = prev_sample = None
    while True:
        frame, t = source.read()
        if frame is None:
            failed += 1
            if source.now() >= t_end or failed > 1000: break
            time.sleep(0.001); continue
        if frame_size is None: frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        if t >= t_meas:
            ts.append(t)
            sig = frame_signature(frame)
            if prev_sig is not None:
                if sig == prev_sig: dups += 1
                elif dup_mad > 0 and sample_mad(prev_sample, frame) <= dup_mad: dups += 1
            prev_sig, prev_sample = sig, (frame if dup_mad > 0 else None)
        if t >= t_end: break
    return {"timestamps": ts, "duplicates": dups, "failed_reads": failed, "frame_size": frame_size}


def cpu_snapshot():
    """loadavg (1/5/15) and, with psutil, the CPU busy percent since the previous call."""
    out = {}
    try: out["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError): out["loadavg"] = None
    try:
        import psutil
        out["cpu_percent"] = psutil.cpu_percent(interval=None); out["cpu_count"] = psutil.cpu_count()
    except ImportError: pass
    return out


def run(cams, seconds=10.0, warmup=1.0, opener=CvSource, thresholds=THRESHOLDS, dup_mad=0.0, cpu=cpu_snapshot):
    """Open every camera in `cams` ({name: cfg}) at once, capture them in parallel threads, build the report."""
    results = {}
    def worker(name, cfg):
        src = None
        try:
            src = opener(cfg)
            neg = src.negotiated() if hasattr(src, "negotiated") else None
            cap = capture(src, seconds, warmup, thresholds["stall_ms"], dup_mad)
            results[name] = {"negotiated": neg, **cap}
        except Exception as e:
            results[name] = {"error": str(e)}
        finally:
            if src is not None:
                try: src.close()
                except Exception: pass
    before = cpu()
    threads = [threading.Thread(target=worker, args=(n, c), daemon=True, name=f"cam-{n}") for n, c in cams.items()]
    for t in threads: t.start()
    for t in threads: t.join()
    after = cpu()
    report = {"seconds": seconds, "warmup": warmup, "thresholds": dict(thresholds), "host": platform.node(), "cpu": {"before": before, "after": after}, "cameras": {}}
    for name, cfg in cams.items():
        r = results.get(name, {"error": "no result"})
        req = {k: cfg[k] for k in ("width", "height", "fps", "fourcc")}
        if "error" in r:
            cam = {"requested": req, "path": cfg["path"], "error": r["error"]}
        else:
            cam = {"requested": req, "path": cfg["path"], "negotiated": r["negotiated"], "frame_size": r["frame_size"],
                   "stats": interval_stats(r["timestamps"], thresholds["stall_ms"]), "duplicates": r["duplicates"], "failed_reads": r["failed_reads"]}
        cam["problems"] = verdicts(cam, thresholds)
        report["cameras"][name] = cam
    report["ok"] = exit_code(report) == 0
    return report


# ---------------------------------------------------------------- output
def format_text(report):
    lines = [f"camera health on {report['host']}: {report['seconds']:.0f} s capture after {report['warmup']:.1f} s warm-up, all cameras open together"]
    hdr = f"{'cam':<6} {'size/fourcc':<16} {'fps':>6} {'p50':>7} {'p95':>7} {'max':>7} {'dups':>5} {'stalls':>6}  verdict"
    lines.append(hdr); lines.append("-" * len(hdr))
    for name, c in report["cameras"].items():
        if c.get("error"):
            lines.append(f"{name:<6} {'-':<16} {'-':>6} {'-':>7} {'-':>7} {'-':>7} {'-':>5} {'-':>6}  FAIL {c['error']}"); continue
        n = c["negotiated"] or {}; s = c["stats"]; fs = c.get("frame_size") or (n.get("width"), n.get("height"))
        fmt = f"{fs[0]}x{fs[1]} {n.get('fourcc', '????')}"
        p = lambda v: "-" if v is None else f"{v:.1f}"
        verdict = "OK" if not c["problems"] else "FAIL " + "; ".join(c["problems"])
        lines.append(f"{name:<6} {fmt:<16} {s['fps']:>6.2f} {p(s['p50_ms']):>7} {p(s['p95_ms']):>7} {p(s['max_ms']):>7} {c['duplicates']:>5} {s['stalls']:>6}  {verdict}")
        extra = f"       requested {c['requested']['width']}x{c['requested']['height']} {c['requested']['fourcc'] or 'default'} @ {c['requested']['fps']}; driver says {n.get('fps', '?')} fps; {s['frames']} frames in {s['span_s']} s"
        if c["failed_reads"]: extra += f"; {c['failed_reads']} failed reads"
        lines.append(extra)
    cpu = report["cpu"]; b, a = cpu.get("before", {}), cpu.get("after", {})
    cpu_line = f"cpu: loadavg {b.get('loadavg')} -> {a.get('loadavg')}"
    if "cpu_percent" in a: cpu_line += f", {a['cpu_percent']}% busy of {a.get('cpu_count')} cores during the run"
    lines.append(cpu_line)
    th = report["thresholds"]
    lines.append(f"thresholds: fps >= {th['min_fps']}, stall = interval > {th['stall_ms']:.0f} ms, fail when any stall > {th['max_stall_ms']:.0f} ms or the size differs")
    lines.append("RESULT: " + ("all cameras healthy" if report["ok"] else "PROBLEMS: " + "; ".join(f"{n}: {', '.join(c['problems'])}" for n, c in report["cameras"].items() if c["problems"])))
    return "\n".join(lines)


class BusyFlag:
    """Hold ~/so101/.cams_busy while the devices are open so the Mac dashboard's ffmpeg lets go. Leaves a pre-existing flag alone."""
    def __init__(self, path, enabled, settle=1.5): self.path, self.enabled, self.settle, self.made = path, enabled, settle, False
    def __enter__(self):
        if self.enabled and not os.path.exists(self.path):
            open(self.path, "w").close(); self.made = True; time.sleep(self.settle)
        return self
    def __exit__(self, *exc):
        if self.made:
            try: os.remove(self.path)
            except FileNotFoundError: pass
        return False


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seconds", nargs="?", type=float, default=10.0, help="measurement length (default 10)")
    p.add_argument("--warmup", type=float, default=1.0, help="seconds discarded before measuring (auto-exposure settles; default 1)")
    p.add_argument("--cams", default="top,front,wrist", help="comma list of cameras to test (default all three)")
    p.add_argument("--device", action="append", default=[], metavar="NAME=PATH", help="override a camera path, e.g. wrist=/dev/video2")
    p.add_argument("--min-fps", type=float, default=THRESHOLDS["min_fps"]); p.add_argument("--stall-ms", type=float, default=THRESHOLDS["stall_ms"])
    p.add_argument("--max-stall-ms", type=float, default=THRESHOLDS["max_stall_ms"])
    p.add_argument("--dup-mad", type=float, default=0.0, help="also count near-identical frames (strided mean abs diff <= this; default 0 = bit-identical only)")
    p.add_argument("--json", action="store_true", help="print the JSON report instead of the table")
    p.add_argument("--no-busy", action="store_true", help="do not touch ~/so101/.cams_busy")
    p.add_argument("--busy-flag", default=BUSY_FLAG)
    return p


def main(argv=None, opener=CvSource):
    a = build_parser().parse_args(argv)
    cams = {}
    for name in [c.strip() for c in a.cams.split(",") if c.strip()]:
        if name not in DEFAULT_CAMS: log(f"unknown camera {name!r}; choose from {', '.join(DEFAULT_CAMS)}"); return 2
        cams[name] = dict(DEFAULT_CAMS[name])
    for ov in a.device:
        name, _, path = ov.partition("=")
        if name not in cams or not path: log(f"bad --device {ov!r}"); return 2
        cams[name]["path"] = path
    th = {"min_fps": a.min_fps, "stall_ms": a.stall_ms, "max_stall_ms": a.max_stall_ms}
    with BusyFlag(a.busy_flag, not a.no_busy):
        report = run(cams, a.seconds, a.warmup, opener, th, a.dup_mad)
    print(json.dumps(report, indent=1) if a.json else format_text(report))
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
