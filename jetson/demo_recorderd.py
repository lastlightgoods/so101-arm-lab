#!/usr/bin/env python3
"""demo_recorderd.py - the resident demo recorder on the Jetson (docs/scripted-demos.md, phase 2).

    demo_recorderd.py serve [--vcodec libsvtav1|h264] [--ticks PATH] [--sock PATH] [--cams-sock PATH]
    demo_recorderd.py ctl OP [JSON]        one control request (JSON from the argument or stdin); prints the one-line JSON reply; exit 0 ok, 1 refused, 2 no daemon

It owns a jetson/demo_recorder.DemoRecorder. Inputs: the pick_seq tick sink file (TickBuffer.poll_file, every 1/30 s) and the camera frames from camd
(demo_cams.CamdCameraSource: a reader, never the device owner). Control channel: a unix socket, one JSON line in and one JSON line out per connection.

    ping                          -> {"ok": true, "pid", "configured", "episode"}
    status                        -> ping + camera ages, tick file state, counts, disk free
    configure {repo, root, rejects_root?, rejects_repo?, lab: {LabConfig fields}, vcodec?}   start a dataset session (closes the previous one; refused with an episode open)
    open {meta}                   start an episode (never refused for a missing camera: that episode just fails its health check and goes to the rejects)
    end                           stop sampling; reply carries the health dict
    commit {keep, verdict, meta?} write the ended episode to the main (keep) or the rejects dataset; `meta` is merged into the episode's meta first
    abort {reason}                drop the open episode, nothing saved
    link {kind, row}              append a row to links.jsonl;   summary / finish  -> the dataset-version numbers (finish also finalizes the datasets)

SAFETY: this process never commands the arm and never opens a serial port (it imports no robot code; tests/jetson/test_demo_recorderd.py greps for it).
Any exception in a handler or in the 30 Hz loop is answered or logged and the loop goes on; an exception while sampling aborts the open episode (so a crash
or a bad frame rejects the episode, never the arm). If the process itself dies the Mac sees `alive` false (autoloop trips its `recorder` guard before the next
cycle) and the next start deletes the half-written spool. It refuses to run an episode while ~/so101/.cams_busy exists (a lerobot job owns the cameras).
"""
import json, os, select, shutil, signal, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import demo_recorder as dr  # noqa: E402
import lab_config as lc  # noqa: E402

SOCK = os.environ.get("SO101_DEMOREC_SOCK", "/dev/shm/so101_demorec.sock")
TICKS = os.environ.get("SO101_TICK_SINK") or "/dev/shm/so101_ticks.jsonl"
SPOOL = os.path.expanduser("~/so101/outputs/run/demo_spool")
BUSY_FLAGS = (os.path.expanduser("~/so101/.cams_busy"),)   # why: not .arms_busy: pick_seq --serve holds that flag for the whole session it records (12 Nov 2026); a lerobot job sets .cams_busy too
IDLE_KEEP_S = 3.0   # ticks older than this are dropped while no episode is open


def deep_merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict): deep_merge(a[k], v)
        else: a[k] = v
    return a


class Daemon:
    """The recorder plus its control handlers. `cams` = a CameraSource (demo_cams.CamdCameraSource, or the FakeCameraSource in tests), `clock` = time.monotonic."""
    def __init__(self, cams, ticks_path=TICKS, spool=SPOOL, vcodec=dr.DEFAULT_VCODEC, clock=time.monotonic, busy=BUSY_FLAGS, calib_dir=None, log=None):
        self.cams, self.ticks_path, self.spool, self.vcodec, self.clock, self.busy = cams, ticks_path, spool, vcodec, clock, busy
        self.calib_dir = calib_dir or os.path.join(HERE, "calib"); self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.ticks = dr.TickBuffer(); self.rec = None; self.last_err = None; self.stopping = False; self.last_abort = None
        shutil.rmtree(self.spool, ignore_errors=True); os.makedirs(self.spool, exist_ok=True)   # why: a crash leaves ep_* folders of raw frames; nothing in them can be committed any more

    # -- the 30 Hz work
    def step(self, now=None):
        """Poll the tick file and sample. Never raises: an error aborts the open episode and is remembered."""
        try:
            self.ticks.poll_file(self.ticks_path)
            now = self.clock() if now is None else now
            if self.rec is not None and self.rec.ep is not None and self.rec.ep.state == "open": self.rec.pump(now)
            else: self.ticks.trim(now - IDLE_KEEP_S)
        except Exception as e:
            self.last_err = f"{type(e).__name__}: {str(e)[:160]}"; self.log(f"sampling error: {self.last_err}")
            try: self.rec and self.rec.abort("sampling_error")
            except Exception: pass

    # -- control
    def episode(self):
        ep = self.rec.ep if self.rec is not None else None
        return None if ep is None or ep.state not in ("open", "ended") else {"state": ep.state, "frames": len(ep.frames), "dropped_idle": ep.dropped_idle}

    def handle(self, req):
        op = req.get("op")
        try:
            fn = getattr(self, f"op_{op}", None)
            if fn is None: return {"ok": False, "err": f"unknown op {op!r}"}
            return fn(req)
        except Exception as e: return {"ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}

    def need_rec(self):
        if self.rec is None: raise RuntimeError("not configured")
        return self.rec

    def op_ping(self, req): return {"ok": True, "pid": os.getpid(), "configured": self.rec is not None, "episode": self.episode()}

    def op_status(self, req):
        ages = self.cams.ages() if hasattr(self.cams, "ages") else {}
        try: free = shutil.disk_usage(self.spool).free // 2**20
        except OSError: free = None
        return {**self.op_ping(req), "cam_age_s": {c: None if a is None else round(a, 3) for c, a in ages.items()}, "cams_missing": self.cams.missing() if hasattr(self.cams, "missing") else [],
                "ticks": len(self.ticks.ticks), "tick_file": os.path.exists(self.ticks_path), "busy": [p for p in self.busy if os.path.exists(p)], "counts": dict(self.rec.counts) if self.rec else None, "disk_free_mb": free, "last_err": self.last_err}

    def op_configure(self, req):
        if self.rec is not None and self.episode(): return {"ok": False, "err": "an episode is open"}
        if self.rec is not None: self.rec.close()
        lab = lc.LabConfig.from_dict(req["lab"]) if req.get("lab") else lc.LabConfig(); root = os.path.expanduser(req["root"]); repo = req["repo"]
        self.rec = dr.DemoRecorder(root, repo, self.cams, self.ticks, lab, rejects_root=os.path.expanduser(req["rejects_root"]) if req.get("rejects_root") else None, rejects_repo=req.get("rejects_repo"),
                                   calib_hashes=lc.calib_hashes(self.calib_dir), spool_dir=self.spool, vcodec=req.get("vcodec") or self.vcodec)
        return {"ok": True, "root": self.rec.root, "rejects_root": self.rec.rejects_root, "repo": self.rec.repo_id, "rejects_repo": self.rec.rejects_repo, "vcodec": self.rec.vcodec, "config_id": lab.config_id, "fps": self.rec.fps}

    def op_open(self, req):
        rec = self.need_rec(); busy = [p for p in self.busy if os.path.exists(p)]
        if busy: return {"ok": False, "err": "hardware_busy:" + ",".join(os.path.basename(p) for p in busy)}
        self.ticks.poll_file(self.ticks_path); rec.open(req.get("meta") or {}, self.clock())
        return {"ok": True, "cams_missing": self.cams.missing() if hasattr(self.cams, "missing") else []}   # why: a missing camera does not refuse the episode; its health check rejects it (the arm never waits for the recorder)

    def op_end(self, req):
        rec = self.need_rec(); self.step(); return {"ok": True, "health": rec.end(self.clock())}

    def op_commit(self, req):
        rec = self.need_rec()
        if req.get("meta") and rec.ep is not None: deep_merge(rec.ep.meta, req["meta"])
        res = rec.commit(bool(req.get("keep")), req.get("verdict") or {}); res.pop("row", None); return {"ok": True, **res}

    def op_abort(self, req):
        was = self.rec.abort(req.get("reason") or "aborted") if self.rec is not None else False; return {"ok": True, "was_open": bool(was)}

    def op_link(self, req): self.need_rec().append_link(req["kind"], req["row"]); return {"ok": True}

    def op_summary(self, req): return {"ok": True, **self.need_rec().summary()}

    def op_finish(self, req):
        rec = self.need_rec(); rec.abort("session_end"); rec.close(); return {"ok": True, **rec.summary()}

    def close(self):
        try:
            if self.rec is not None: self.rec.close()
        except Exception as e: self.log(f"close: {e}")

    # -- the socket
    def listen(self, path=SOCK):
        try: os.unlink(path)
        except OSError: pass
        s = socket.socket(socket.AF_UNIX); s.bind(path); s.listen(4); os.chmod(path, 0o600); self.path = path; self.lsock = s; return s

    def serve_one(self, conn):
        try:
            conn.settimeout(2.0); line = b""
            while not line.endswith(b"\n") and len(line) < 1 << 20:
                chunk = conn.recv(65536)
                if not chunk: break
                line += chunk
            reply = self.handle(json.loads(line))
        except Exception as e: reply = {"ok": False, "err": f"bad request: {str(e)[:100]}"}
        try: conn.sendall((json.dumps(reply, default=str) + "\n").encode())
        except OSError: pass
        finally: conn.close()

    def run(self, fps=dr.FPS):
        """The loop: wake every 1/fps (deadline-based), step(), answer the control socket in between. Single thread, so no locks."""
        period = 1.0 / fps; nxt = self.clock()
        while not self.stopping:
            self.step(); nxt += period; now = self.clock()
            if nxt < now - 1.0: nxt = now   # why: after a long commit (encode) do not run a second of catch-up wakes
            while not self.stopping:
                left = nxt - self.clock()
                if left <= 0: break
                r, _, _ = select.select([self.lsock], [], [], left)
                if r:
                    conn, _ = self.lsock.accept(); self.serve_one(conn)
        self.close()
        try: os.unlink(self.path)
        except OSError: pass


def ctl(op, payload, sock=SOCK, timeout=600.0):
    """One request to the daemon; returns (exit code, reply dict)."""
    s = socket.socket(socket.AF_UNIX); s.settimeout(timeout)
    try: s.connect(sock)
    except OSError as e: return 2, {"ok": False, "err": f"no_daemon: {e}"}
    try:
        s.sendall((json.dumps({**payload, "op": op}) + "\n").encode()); data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk: break
            data += chunk
        reply = json.loads(data)
    except (OSError, ValueError) as e: return 2, {"ok": False, "err": f"no_reply: {str(e)[:100]}"}
    finally: s.close()
    return (0 if reply.get("ok") else 1), reply


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve"); sv.add_argument("--vcodec", default=dr.DEFAULT_VCODEC, help="libsvtav1 (lerobot's default) or h264 if the Orin's CPU cannot keep up"); sv.add_argument("--ticks", default=TICKS)
    sv.add_argument("--sock", default=SOCK); sv.add_argument("--cams-sock", default=None)
    c = sub.add_parser("ctl"); c.add_argument("op"); c.add_argument("payload", nargs="?", default=None); c.add_argument("--sock", default=SOCK); c.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    if a.cmd == "ctl":
        raw = a.payload if a.payload is not None else (sys.stdin.read() if not sys.stdin.isatty() else "")
        code, reply = ctl(a.op, json.loads(raw) if raw.strip() else {}, a.sock, a.timeout); print(json.dumps(reply, default=str)); return code
    import demo_cams
    cams = demo_cams.CamdCameraSource(**({"path": a.cams_sock} if a.cams_sock else {})).start(); d = Daemon(cams, a.ticks, vcodec=a.vcodec); d.listen(a.sock)
    def bye(*_): d.stopping = True
    signal.signal(signal.SIGTERM, bye); signal.signal(signal.SIGINT, bye)   # so101-stop sends TERM: the open episode is aborted by close()
    print(f"demo_recorderd up on {a.sock}, ticks {a.ticks}, vcodec {a.vcodec}", flush=True); d.run(); cams.close(); return 0


if __name__ == "__main__": sys.exit(main())
