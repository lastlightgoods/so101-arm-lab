"""Voice-guided gripper range calibration for one arm:  python calib_gripper.py follower|leader
Torque off on the gripper servo, you close the fingers fully (6 s), then open them fully (6 s); the raw tick medians become
range_min (closed) / range_max (open) in the LeRobot calibration JSON, keeping homing_offset. Prompts go to the Mac speaker via
the dashboard's say relay (/dev/shm/so101_say.jsonl). Other joints are untouched."""
import os, sys, json, time, glob, statistics
role = sys.argv[1] if len(sys.argv) > 1 else "follower"
port = os.environ["FOLLOWER" if role == "follower" else "LEADER"]
cal = glob.glob(os.path.expanduser(f"~/.cache/huggingface/lerobot/calibration/*/*/ll_{role}.json"))[0]
def say(t):
    print(t, flush=True)
    try: open("/dev/shm/so101_say.jsonl", "a").write(json.dumps({"t": time.time(), "say": t}) + "\n")
    except Exception: pass
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode
bus = FeetechMotorsBus(port=port, motors={"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)}); bus.connect(handshake=False)
def sample(secs=1.2):
    v = []; t0 = time.time()
    while time.time() - t0 < secs: v.append(int(bus.read("Present_Position", "gripper", normalize=False))); time.sleep(0.05)
    return int(statistics.median(v))
try:
    bus.disable_torque("gripper")
    say(f"Calibrating the {role} gripper. Torque is off. Squeeze the fingers fully closed and hold them."); time.sleep(float(os.environ.get("CALIB_HOLD", "6")))
    closed = sample(); say(f"Closed reads {closed}. Now open the fingers all the way and hold."); time.sleep(float(os.environ.get("CALIB_HOLD", "6")))
    opened = sample(); say(f"Open reads {opened}.")
    if abs(opened - closed) < 300: say("That range is too small. Nothing was changed."); raise SystemExit(1)
    c = json.load(open(cal)); g = c["gripper"]; old = (g["range_min"], g["range_max"])
    lo, hi = sorted((closed, opened)); g["range_min"], g["range_max"] = lo, hi
    if closed > opened: g["drive_mode"] = 1 - int(g.get("drive_mode", 0))   # keep 0 % = closed if the direction is reversed
    json.dump(c, open(cal, "w"), indent=4)
    from lerobot.motors import MotorCalibration
    bus.write_calibration({"gripper": MotorCalibration(**g)})   # keep the servo EEPROM in step with the file, or LeRobot asks to recalibrate interactively
    bus.enable_torque("gripper"); time.sleep(0.3)
    say(f"Saved. {role} gripper range {lo} to {hi}, was {old[0]} to {old[1]}. Torque is back on.")
    print("calibration file:", cal)
finally:
    try: bus.disconnect(disable_torque=False)
    except Exception: pass
