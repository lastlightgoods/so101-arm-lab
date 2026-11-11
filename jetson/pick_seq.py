#!/usr/bin/env python3
"""pick_seq.py - run a list of joint waypoints on the follower in ONE bus session (goto.py reconnects per move, ~2 s each).

  python pick_seq.py --plan plan.json [--correct 0.8] [--max-deg-s 80]

plan.json: {"waypoints": [{"name": "hover", "joints": {...LeRobot deg...}, "gripper": 35, "seconds": 2.0, "settle": 0.3}, ...]}
Each waypoint is an eased move (half-cosine) over `seconds`, stretched if any joint would exceed --max-deg-s. After each
move the joints are read back; with --correct g > 0 the NEXT waypoint's command is biased by g times the error of the
LAST waypoint (target - reached) for the five arm joints. That cancels most of the servos' load-dependent sag, which was
measured at 1-3.5 deg on 9 Nov. Prints one JSON line per waypoint with target / reached / max error. Refuses to run
while a lerobot job holds the arm; holds ~/so101/.arms_busy so the dashboard poller lets go of the port.
--tick-sink [PATH] (default OFF) also appends one JSON line per command tick (action sent + state read just before it) for jetson/demo_recorder.py.
"""
import argparse, json, math, os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # why: so101-stop sends TERM; turn it into the same clean unwind as Ctrl-C (torque stays on, the arm freezes where it is)

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = J[:5]
BUSY = os.path.expanduser("~/so101/.arms_busy")


def lerobot_running():
    return subprocess.run(["pgrep", "-f", "lerobot-(record|teleoperate|rollout|calibrate|replay)"], capture_output=True).returncode == 0


def connect():
    import lerobot.robots.so_follower as _sf
    Cfg = getattr(_sf, "SO101FollowerConfig", None) or _sf.SOFollowerConfig; Rob = getattr(_sf, "SO101Follower", None) or _sf.SOFollower
    for i in range(4):
        r = Rob(Cfg(port=os.environ.get("FOLLOWER", "/dev/so101_follower"), id=os.environ.get("FOLLOWER_ID", "ll_follower"), cameras={}, disable_torque_on_disconnect=False))
        try: r.connect(); return r
        except Exception as e:
            print(f"connect attempt {i + 1}/4 failed: {str(e).splitlines()[0][:80]}", file=sys.stderr, flush=True)
            try: r.bus.port_handler.closePort()
            except Exception: pass
            time.sleep(1.5)
    raise SystemExit("bus not responding; check the servo cables / USB")


def read(robot):
    obs = robot.get_observation(); return {j: float(obs[f"{j}.pos"]) for j in J}


def move_secs(cur, tgt, seconds, max_deg_s):
    """Duration of one eased move: `seconds`, stretched so no joint exceeds max_deg_s."""
    biggest = max(abs(tgt[j] - cur[j]) for j in J); return max(seconds, biggest / max_deg_s)


def ease_n(secs): return max(12, int(secs * 40))   # ticks in a move, about 40 Hz


def ease_cmd(cur, tgt, i, n):
    """The i-th (1-based) of n half-cosine command ticks from cur to tgt. Shared with the fake arm (jetson/demo_recorder.py) so both send the same sequence."""
    e = (1 - math.cos(math.pi * i / n)) / 2
    return {f"{j}.pos": cur[j] + (tgt[j] - cur[j]) * e for j in J}


def move(robot, cur, tgt, seconds, max_deg_s, sink=None, wp=None):
    secs = move_secs(cur, tgt, seconds, max_deg_s); n = ease_n(secs)
    for i in range(1, n + 1):
        st = sink.before(robot) if sink is not None else None   # why: lerobot-record reads the state BEFORE it sends the action of the same tick
        sent = robot.send_action(ease_cmd(cur, tgt, i, n))
        if sink is not None: sink.after(st, sent, wp)
        time.sleep(secs / n)
    return secs


DEFAULT_TICK_SINK = "/dev/shm/so101_ticks.jsonl"   # --tick-sink with no value; $SO101_TICK_SINK overrides it (tests, a second arm)
SINK_MAX_LINES = 20000   # about 8 minutes of ticks per file; at the limit the file moves to <path>.1 (one generation kept)


class TickSink:
    """OFF unless --tick-sink is given. One JSON line per command tick: {seq, t_mono, t_wall, wp, action, state}, `action` = what send_action
    returned (the command actually sent), `state` = the servo readback taken just BEFORE that command. For the demo recorder (jetson/demo_recorder.py),
    which merges these with camera frames. Never raises into the motion code: the first I/O error turns the sink off for the rest of the run."""
    def __init__(self, path, max_lines=SINK_MAX_LINES):
        self.path, self.max_lines, self.seq, self.lines, self.f, self.err = path, max_lines, 0, 0, None, None
        self._open("w")

    def _open(self, mode):
        try:
            self.f = open(self.path, mode); self.lines = 0
            self.f.write(json.dumps({"header": 1, "version": 1, "pid": os.getpid(), "t_mono": time.monotonic(), "t_wall": time.time()}) + "\n"); self.f.flush()
        except OSError as e: self._fail(e)

    def _fail(self, e):
        self.err = str(e); self.f = None
        print(f"tick sink off: {str(e)[:80]}", file=sys.stderr, flush=True)

    def before(self, robot):
        if self.f is None: return None
        try: return read(robot)
        except Exception as e: self._fail(e); return None

    def after(self, st, sent, wp):
        if self.f is None: return
        try:
            if self.lines >= self.max_lines: self.f.close(); os.replace(self.path, self.path + ".1"); self._open("w")
            if self.f is None: return
            self.seq += 1
            self.f.write(json.dumps({"seq": self.seq, "t_mono": time.monotonic(), "t_wall": time.time(), "wp": wp,
                                     "action": {f"{j}.pos": float(sent[f"{j}.pos"]) for j in J}, "state": None if st is None else {f"{j}.pos": st[j] for j in J}}) + "\n")
            self.f.flush(); self.lines += 1
        except Exception as e: self._fail(e)

    def close(self):
        try:
            if self.f: self.f.close()
        except OSError: pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True); ap.add_argument("--correct", type=float, default=0.8, help="gain on the last waypoint's error fed into the next command (0 = off)")
    ap.add_argument("--max-deg-s", type=float, default=80.0); ap.add_argument("--force", action="store_true")
    ap.add_argument("--tick-sink", nargs="?", const="", default=None, metavar="PATH", help=f"write per-tick action and state here for the demo recorder (default off; no value = $SO101_TICK_SINK or {DEFAULT_TICK_SINK})")
    a = ap.parse_args()
    if lerobot_running() and not a.force: raise SystemExit("a lerobot job (teleop/record/rollout) is holding the arm; so101-stop first")
    plan = json.load(open(a.plan))["waypoints"]
    made = not os.path.exists(BUSY)
    if made: open(BUSY, "w").close(); time.sleep(1.0)
    robot = connect(); t0 = time.time(); sink = TickSink(a.tick_sink or os.environ.get("SO101_TICK_SINK") or DEFAULT_TICK_SINK) if a.tick_sink is not None else None
    try:
        cur = read(robot); print(json.dumps({"start": {j: round(v, 2) for j, v in cur.items()}}), flush=True)
        bias = {j: 0.0 for j in ARM}
        for wp in plan:
            tgt = {j: float(wp["joints"].get(j, cur[j])) for j in ARM}; tgt["gripper"] = float(wp.get("gripper", cur["gripper"]))
            cmd = {j: tgt[j] + a.correct * bias[j] for j in ARM}; cmd["gripper"] = tgt["gripper"]
            cmd["wrist_flex"] = max(-93.0, min(93.0, cmd["wrist_flex"]))
            secs = move(robot, cur, cmd, float(wp.get("seconds", 1.5)), a.max_deg_s, sink, wp.get("name")); time.sleep(float(wp.get("settle", 0.3)))
            got = read(robot); err = {j: tgt[j] - got[j] for j in ARM}
            if wp.get("gripper_only"): err = {j: 0.0 for j in ARM}   # a pure gripper move says nothing new about arm sag
            bias = ({j: err[j] for j in ARM} if not wp.get("gripper_only") else bias) if a.correct > 0 else bias   # why: feed forward the LAST error only; summing them acted like an integrator and overshot (9 Nov: 24 mm miss)
            print(json.dumps({"wp": wp.get("name"), "seconds": round(secs, 2), "t": round(time.time() - t0, 1), "target": {j: round(tgt[j], 2) for j in J},
                              "reached": {j: round(got[j], 2) for j in J}, "max_err_deg": round(max(abs(err[j]) for j in ARM), 2)}), flush=True)
            cur = got
        print(json.dumps({"done": True, "total_s": round(time.time() - t0, 1)}), flush=True)
    finally:
        if sink is not None: sink.close()
        robot.disconnect()
        if made:
            try: os.remove(BUSY)
            except FileNotFoundError: pass


if __name__ == "__main__": main()
