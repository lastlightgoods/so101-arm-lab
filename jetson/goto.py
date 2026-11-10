#!/usr/bin/env python3
"""goto.py - move the follower to a joint target slowly and smoothly (eased, like nudge.py), for calibration tests.

  python goto.py --deg PAN LIFT ELBOW WRIST [ROLL] [--gripper G] [--seconds S]
  python goto.py --json '{"shoulder_pan": .., "shoulder_lift": .., ...}'
  python goto.py --pose NAME                      a pose from ~/so101/poses.json (so101-mark)
  python goto.py --read                           print the current joints and exit

Safety: refuses to run while a lerobot job (teleop/record/rollout) holds the arm, caps the move at 40 deg/s per
joint (stretching --seconds if needed), holds the arm's busy flag so the dashboard poller lets go of the port,
and prints the start pose first so you can send the arm back with --json.
"""
import argparse, json, math, os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # why: so101-stop sends TERM; turn it into the same clean unwind as Ctrl-C (torque stays on, the arm freezes where it is)

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BUSY = os.path.expanduser("~/so101/.arms_busy")
MAX_DEG_PER_S = 40.0


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--deg", type=float, nargs="+"); g.add_argument("--json"); g.add_argument("--pose"); g.add_argument("--read", action="store_true")
    ap.add_argument("--gripper", type=float, help="gripper 0-100 (default: leave as is)"); ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--poses", default=os.path.expanduser("~/so101/poses.json")); ap.add_argument("--force", action="store_true", help="move even if a lerobot job is running (it will fight you)")
    a = ap.parse_args()
    if lerobot_running() and not a.force: raise SystemExit("a lerobot job (teleop/record/rollout) is holding the arm; so101-stop first")
    target = None
    if a.deg:
        if len(a.deg) < 4: raise SystemExit("--deg needs pan lift elbow wrist [roll]")
        target = dict(zip(J[:5], a.deg))
    elif a.json: target = json.loads(a.json)
    elif a.pose:
        poses = json.load(open(a.poses))
        if a.pose not in poses: raise SystemExit(f"pose {a.pose!r} not in {a.poses}; have {sorted(poses)}")
        target = poses[a.pose]
    made = not os.path.exists(BUSY)
    if made: open(BUSY, "w").close(); time.sleep(1.0)
    robot = connect()
    try:
        obs = robot.get_observation(); cur = {j: float(obs[f"{j}.pos"]) for j in J}
        print(json.dumps({"start": {j: round(v, 2) for j, v in cur.items()}}), flush=True)
        if a.read: return
        tgt = {j: float(target.get(j, cur[j])) for j in J}
        if a.gripper is not None: tgt["gripper"] = a.gripper
        biggest = max(abs(tgt[j] - cur[j]) for j in J)
        secs = max(a.seconds, biggest / MAX_DEG_PER_S); n = max(20, int(secs * 30))
        for i in range(1, n + 1):
            e = (1 - math.cos(math.pi * i / n)) / 2
            robot.send_action({f"{j}.pos": cur[j] + (tgt[j] - cur[j]) * e for j in J}); time.sleep(secs / n)
        time.sleep(0.3); obs = robot.get_observation(); end = {j: float(obs[f"{j}.pos"]) for j in J}
        print(json.dumps({"target": {j: round(v, 2) for j, v in tgt.items()}, "reached": {j: round(v, 2) for j, v in end.items()},
                          "max_err_deg": round(max(abs(end[j] - tgt[j]) for j in J[:5]), 2), "seconds": round(secs, 1)}), flush=True)
    finally:
        robot.disconnect()
        if made:
            try: os.remove(BUSY)
            except FileNotFoundError: pass


if __name__ == "__main__": main()
