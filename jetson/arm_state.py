"""Stream SO-101 joint states as JSON lines (one per arm per tick) for the Mac dashboard.
Idle: reads both arms directly over serial (torque untouched, read-only, ~15 Hz).
Busy (a lerobot process is running or ~/so101/.arms_busy exists): releases the ports and forwards the
state files the busheal plugin writes to /dev/shm from inside the lerobot process instead."""
import json, os, subprocess, sys, time
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
ARMS = {"follower": ("/dev/so101_follower", "robots/so_follower/ll_follower.json"),
        "leader": ("/dev/so101_leader", "teleoperators/so_leader/ll_leader.json")}
CAL_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/calibration"); BUSY = os.path.expanduser("~/so101/.arms_busy")
MOTORS = {n: Motor(i, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"), 1)}
MOTORS["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)
def emit(d): sys.stdout.write(json.dumps(d) + "\n"); sys.stdout.flush()
def lerobot_running():
    try: return subprocess.run(["pgrep", "-f", "lerobot-(record|teleoperate|rollout|calibrate|replay|setup-motors)"], capture_output=True).returncode == 0
    except Exception: return False
buses = {}; cals = {}
for role, (_, rel) in ARMS.items():
    try: cals[role] = {n: MotorCalibration(**v) for n, v in json.load(open(os.path.join(CAL_DIR, rel))).items()}
    except Exception as e: cals[role] = None; emit({"role": role, "err": f"no calibration: {e}"})
def close(role):
    b = buses.pop(role, None)
    if b:
        try: b.disconnect(disable_torque=False)
        except Exception: pass
SHM = "/dev/shm"   # state files written by the busheal plugin; a module constant so tests can point it elsewhere
SAY = f"{SHM}/so101_say.jsonl"; say_pos = os.path.getsize(SAY) if os.path.exists(SAY) else 0
def forward_say():
    global say_pos
    try:
        size = os.path.getsize(SAY)
        if size < say_pos: say_pos = 0
        if size > say_pos:
            with open(SAY, "rb") as f:
                f.seek(say_pos); chunk = f.read()
            chunk = chunk[:chunk.rfind(b"\n") + 1]; say_pos += len(chunk)   # why: a read can land mid-write; leave the half line for the next tick instead of dropping it
            for l in chunk.decode(errors="ignore").splitlines():
                try: emit({"role": "say", **json.loads(l)})
                except (ValueError, TypeError): pass   # why: a valid-JSON non-object line ("42") raised TypeError and killed the whole streamer
    except FileNotFoundError: say_pos = 0
def main():
    last_check = 0; busy = True
    while True:
        now = time.time(); forward_say()
        if now - last_check > 0.5:
            busy = os.path.exists(BUSY) or lerobot_running(); last_check = now
            if busy:
                for r in list(buses): close(r)
        for role, (port, _) in ARMS.items():
            if busy:
                try:
                    d = json.load(open(f"{SHM}/so101_{role}.json"))
                    if now - d.get("t", 0) < 2: emit(d)
                    else: emit({"t": now, "role": role, "src": "busy", "pos": {}})
                except Exception: emit({"t": now, "role": role, "src": "busy", "pos": {}})
                continue
            try:
                if role not in buses:
                    if not os.path.exists(port): emit({"t": now, "role": role, "src": "unplugged", "pos": {}}); continue
                    b = FeetechMotorsBus(port, MOTORS, calibration=cals[role]); b.connect(handshake=False); buses[role] = b
                pos = buses[role].sync_read("Present_Position", normalize=cals[role] is not None)
                emit({"t": now, "role": role, "mode": "deg" if cals[role] else "raw", "src": "idle-poll", "pos": {k: float(v) for k, v in pos.items()},
                      "cal": {n: [c.range_min, c.range_max] for n, c in (cals[role] or {}).items()}})
            except Exception as e:
                emit({"t": now, "role": role, "src": "error", "err": str(e)[:100], "pos": {}}); close(role); time.sleep(1)
        time.sleep(1 / 15)
if __name__ == "__main__": main()
