"""Scripted demo routine for the SO-101 follower: rise from rest, wave, look around, snap the gripper, return home.
Moves are slow linear interpolations in calibrated degrees (same units LeRobot uses), well inside the joint ranges.
    python demo_wave.py [--dry-run] [--speed 1.0]"""
import argparse, math, os, time
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--speed", type=float, default=1.0); a = ap.parse_args()
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
def pose(pan=0, lift=0, elbow=0, wrist=0, roll=0, grip=10): return dict(zip(J, [pan, lift, elbow, wrist, roll, grip]))
UP     = pose(0, -20, 20, 0, 0, 10)      # upright, forearm forward
SCRIPT = [("rise", UP, 3.0), ("look left", pose(45, -20, 20, 0, 0, 10), 1.5), ("look right", pose(-45, -20, 20, 0, 0, 10), 2.5), ("center", UP, 1.5),
          ("wave", pose(0, -20, 20, 40, 0, 10), 0.6), ("wave", pose(0, -20, 20, -30, 0, 10), 0.6), ("wave", pose(0, -20, 20, 40, 0, 10), 0.6), ("wave", pose(0, -20, 20, -30, 0, 10), 0.6), ("center", UP, 0.8),
          ("roll", pose(0, -20, 20, 0, 60, 10), 1.0), ("roll", pose(0, -20, 20, 0, -60, 10), 1.5), ("center", UP, 1.0),
          ("gripper open", pose(0, -20, 20, 0, 0, 80), 0.8), ("gripper close", pose(0, -20, 20, 0, 0, 5), 0.8), ("gripper open", pose(0, -20, 20, 0, 0, 80), 0.8), ("gripper close", UP, 0.8),
          ("reach", pose(0, 30, -20, 30, 0, 10), 2.5), ("back up", UP, 2.5)]
if a.dry_run:
    for name, p, dur in SCRIPT: print(f"{name:14s} {dur/a.speed:4.1f}s  {p}")
    raise SystemExit
from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
robot = SOFollower(SOFollowerConfig(port=os.environ.get("FOLLOWER", "/dev/so101_follower"), id=os.environ.get("FOLLOWER_ID", "ll_follower"), cameras={}))
robot.connect()
try:
    obs = robot.get_observation(); cur = {j: float(obs[f"{j}.pos"]) for j in J}; rest = dict(cur)
    def move(name, target, dur):
        start = dict(cur); n = max(2, int(dur / a.speed * 50))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)   # ease in/out
            for j in J: cur[j] = start[j] + (target[j] - start[j]) * s
            robot.send_action({f"{j}.pos": cur[j] for j in J}); time.sleep(dur / a.speed / n)
        print(f"{name:14s} done", flush=True)
    for name, p, dur in SCRIPT: move(name, p, dur)
    move("home", rest, 3.0)
finally:
    robot.disconnect()
