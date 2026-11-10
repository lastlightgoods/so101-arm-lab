#!/usr/bin/env python3
"""free_arm.py · let go of the follower arm so it can be moved by hand (torque off on all six servos).
    python free_arm.py          -> torque OFF (arm goes free; it will sag, support it)
    python free_arm.py hold     -> torque ON at the current pose (arm holds where you left it)
Stops nothing itself: run so101-stop first if teleop/policy is holding the bus (so101-free does both)."""
import os, sys, time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
port = os.environ.get("FOLLOWER", "/dev/so101_follower"); hold = len(sys.argv) > 1 and sys.argv[1] == "hold"
bus = FeetechMotorsBus(port=port, motors={n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100) for i, n in enumerate(NAMES)})
for attempt in range(4):
    try: bus.connect(handshake=False); break
    except Exception as e:
        print(f"connect attempt {attempt + 1}/4 failed: {str(e).splitlines()[0][:80]}", flush=True); time.sleep(1.0)
else: raise SystemExit("bus not responding; is the arm powered?")
if hold:
    for n in NAMES:   # park the goal where the joint is, then engage, so nothing jumps
        bus.write("Goal_Position", n, bus.read("Present_Position", n, normalize=False), normalize=False)
        bus.write("Torque_Enable", n, 1, normalize=False)
    print("follower: torque ON, holding the current pose")
else:
    for n in NAMES: bus.write("Torque_Enable", n, 0, normalize=False)
    print("follower: torque OFF, arm is free to move by hand")
bus.disconnect(disable_torque=False)
