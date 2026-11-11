#!/usr/bin/env python3
"""pick_seq.py - run a list of joint waypoints on the follower in ONE bus session (goto.py reconnects per move, ~2 s each).

  python pick_seq.py --plan plan.json [--correct 0.8] [--max-deg-s 80]

plan.json: {"waypoints": [{"name": "hover", "joints": {...LeRobot deg...}, "gripper": 35, "seconds": 2.0, "settle": 0.3}, ...]}
Each waypoint is an eased move (half-cosine) over `seconds`, stretched if any joint would exceed --max-deg-s. After each
move the joints are read back; with --correct g > 0 the NEXT waypoint's command is biased by g times the error of the
LAST waypoint (target - reached) for the five arm joints. That cancels most of the servos' load-dependent sag, which was
measured at 1-3.5 deg on 9 Nov. Prints one JSON line per waypoint with target / reached / max error. Refuses to run
while a lerobot job holds the arm; holds ~/so101/.arms_busy so the dashboard poller lets go of the port.
"""
import argparse, json, math, os, signal, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: import servo_telemetry   # read-only servo health block; absent = the sequence runs exactly as before
except ImportError: servo_telemetry = None
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # why: so101-stop sends TERM; turn it into the same clean unwind as Ctrl-C (torque stays on, the arm freezes where it is)

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = J[:5]
BUSY = os.path.expanduser("~/so101/.arms_busy")
SHM = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"   # the dashboard's arm_state.py forwards so101_follower_tele.json from here while we hold .arms_busy
TELE_S = float(os.environ.get("SO101_TELE_S", "0.25"))      # seconds between health reads while moving (0 = off); one read is ~2.4 ms of bus time


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


class Tele:
    """Servo health samples during the sequence (docs/telemetry.md). sample() is rate-limited and returns the seconds it spent so the move loop can
    take them out of its own sleep: the move takes as long as it did without telemetry. take() hands back the waypoint's summary and starts a new one.
    Each sample is also written to /dev/shm/so101_follower_tele.json for the dashboard (the busheal plugin does the same inside lerobot-record)."""
    def __init__(self, robot):
        self.bus = getattr(robot, "bus", None); self.on = bool(servo_telemetry and TELE_S > 0 and self.bus is not None)
        self.last = 0.0; self.limits = None; self.all = servo_telemetry.Summary() if self.on else None; self.cur = servo_telemetry.Summary() if self.on else None
    def sample(self, force=False):
        if not self.on: return 0.0
        t0 = time.time()
        if not force and t0 - self.last < TELE_S: return 0.0
        self.last = t0
        try:
            s = servo_telemetry.read_sample(self.bus, t0)
            if s:
                if self.limits is None: self.limits = servo_telemetry.read_limits(self.bus); s["limits"] = self.limits
                self.cur.add(s); self.all.add(s); self.publish(s)
        except Exception: pass   # why: telemetry never costs the pick
        return time.time() - t0
    def publish(self, s):
        try:
            path = os.path.join(SHM, "so101_follower_tele.json"); tmp = path + ".tmp"
            with open(tmp, "w") as f: json.dump({**s, "role": "follower", "src": "pick_seq", "mode": "deg"}, f)
            os.replace(tmp, path)
        except OSError: pass
    def take(self):
        if not self.on: return None
        out = self.cur.out(); self.cur = servo_telemetry.Summary(); return out if out["n"] else None


def read(robot):
    obs = robot.get_observation(); return {j: float(obs[f"{j}.pos"]) for j in J}


def move(robot, cur, tgt, seconds, max_deg_s, tele=None):
    biggest = max(abs(tgt[j] - cur[j]) for j in J); secs = max(seconds, biggest / max_deg_s); n = max(12, int(secs * 40))
    for i in range(1, n + 1):
        e = (1 - math.cos(math.pi * i / n)) / 2
        robot.send_action({f"{j}.pos": cur[j] + (tgt[j] - cur[j]) * e for j in J}); time.sleep(max(0.0, secs / n - (tele.sample() if tele else 0.0)))
    return secs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True); ap.add_argument("--correct", type=float, default=0.8, help="gain on the last waypoint's error fed into the next command (0 = off)")
    ap.add_argument("--max-deg-s", type=float, default=80.0); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if lerobot_running() and not a.force: raise SystemExit("a lerobot job (teleop/record/rollout) is holding the arm; so101-stop first")
    plan = json.load(open(a.plan))["waypoints"]
    made = not os.path.exists(BUSY)
    if made: open(BUSY, "w").close(); time.sleep(1.0)
    robot = connect(); t0 = time.time(); tele = Tele(robot)
    try:
        cur = read(robot); print(json.dumps({"start": {j: round(v, 2) for j, v in cur.items()}}), flush=True)
        bias = {j: 0.0 for j in ARM}
        for wp in plan:
            tgt = {j: float(wp["joints"].get(j, cur[j])) for j in ARM}; tgt["gripper"] = float(wp.get("gripper", cur["gripper"]))
            cmd = {j: tgt[j] + a.correct * bias[j] for j in ARM}; cmd["gripper"] = tgt["gripper"]
            cmd["wrist_flex"] = max(-93.0, min(93.0, cmd["wrist_flex"]))
            secs = move(robot, cur, cmd, float(wp.get("seconds", 1.5)), a.max_deg_s, tele); time.sleep(float(wp.get("settle", 0.3)))
            tele.sample(force=True); got = read(robot); err = {j: tgt[j] - got[j] for j in ARM}
            if wp.get("gripper_only"): err = {j: 0.0 for j in ARM}   # a pure gripper move says nothing new about arm sag
            bias = ({j: err[j] for j in ARM} if not wp.get("gripper_only") else bias) if a.correct > 0 else bias   # why: feed forward the LAST error only; summing them acted like an integrator and overshot (9 Nov: 24 mm miss)
            print(json.dumps({"wp": wp.get("name"), "seconds": round(secs, 2), "t": round(time.time() - t0, 1), "target": {j: round(tgt[j], 2) for j in J},
                              "reached": {j: round(got[j], 2) for j in J}, "max_err_deg": round(max(abs(err[j]) for j in ARM), 2),
                              **({"tele": tw} if (tw := tele.take()) else {})}), flush=True)
            cur = got
        print(json.dumps({"done": True, "total_s": round(time.time() - t0, 1), **({"tele": tele.all.out()} if tele.on and tele.all.n else {})}), flush=True)
    finally:
        robot.disconnect()
        if made:
            try: os.remove(BUSY)
            except FileNotFoundError: pass


if __name__ == "__main__": main()
