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


def move(robot, cur, tgt, seconds, max_deg_s):
    biggest = max(abs(tgt[j] - cur[j]) for j in J); secs = max(seconds, biggest / max_deg_s); n = max(12, int(secs * 40))
    for i in range(1, n + 1):
        e = (1 - math.cos(math.pi * i / n)) / 2
        robot.send_action({f"{j}.pos": cur[j] + (tgt[j] - cur[j]) * e for j in J}); time.sleep(secs / n)
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
    robot = connect(); t0 = time.time()
    try:
        cur = read(robot); print(json.dumps({"start": {j: round(v, 2) for j, v in cur.items()}}), flush=True)
        bias = {j: 0.0 for j in ARM}
        for wp in plan:
            tgt = {j: float(wp["joints"].get(j, cur[j])) for j in ARM}; tgt["gripper"] = float(wp.get("gripper", cur["gripper"]))
            cmd = {j: tgt[j] + a.correct * bias[j] for j in ARM}; cmd["gripper"] = tgt["gripper"]
            cmd["wrist_flex"] = max(-93.0, min(93.0, cmd["wrist_flex"]))
            secs = move(robot, cur, cmd, float(wp.get("seconds", 1.5)), a.max_deg_s); time.sleep(float(wp.get("settle", 0.3)))
            got = read(robot); err = {j: tgt[j] - got[j] for j in ARM}
            if wp.get("gripper_only"): err = {j: 0.0 for j in ARM}   # a pure gripper move says nothing new about arm sag
            bias = ({j: err[j] for j in ARM} if not wp.get("gripper_only") else bias) if a.correct > 0 else bias   # why: feed forward the LAST error only; summing them acted like an integrator and overshot (9 Nov: 24 mm miss)
            print(json.dumps({"wp": wp.get("name"), "seconds": round(secs, 2), "t": round(time.time() - t0, 1), "target": {j: round(tgt[j], 2) for j in J},
                              "reached": {j: round(got[j], 2) for j in J}, "max_err_deg": round(max(abs(err[j]) for j in ARM), 2)}), flush=True)
            cur = got
        print(json.dumps({"done": True, "total_s": round(time.time() - t0, 1)}), flush=True)
    finally:
        robot.disconnect()
        if made:
            try: os.remove(BUSY)
            except FileNotFoundError: pass


if __name__ == "__main__": main()
