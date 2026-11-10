"""so101-mark <name> : save the follower's current joint pose into ~/so101/poses.json (used by scripted_record.py).
Reads the fresh /dev/shm state the busheal plugin publishes while teleop runs; otherwise reads the bus once."""
import json, os, sys, time
name = sys.argv[1] if len(sys.argv) > 1 else "pose"; P = os.path.expanduser("~/so101/poses.json")
pos = None
try:
    d = json.load(open("/dev/shm/so101_follower.json"))
    if time.time() - d["t"] < 2 and d.get("mode") == "deg": pos = d["pos"]
except Exception: pass
if pos is None:
    from arm_state import ARMS, MOTORS, CAL_DIR, MotorCalibration, FeetechMotorsBus  # noqa: F401  (reuses the idle reader's definitions)
    cal = {n: MotorCalibration(**v) for n, v in json.load(open(os.path.join(CAL_DIR, ARMS["follower"][1]))).items()}
    b = FeetechMotorsBus(ARMS["follower"][0], MOTORS, calibration=cal); b.connect(handshake=False)
    pos = {k: float(v) for k, v in b.sync_read("Present_Position").items()}; b.disconnect(disable_torque=False)
poses = json.load(open(P)) if os.path.exists(P) else {}
poses[name] = {k: round(v, 2) for k, v in pos.items()}; json.dump(poses, open(P, "w"), indent=1)
print(f"saved '{name}':", {k: round(v, 1) for k, v in pos.items()}); print("poses on file:", ", ".join(poses))
