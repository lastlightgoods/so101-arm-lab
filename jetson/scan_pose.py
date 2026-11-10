"""Raise the follower so the WRIST camera faces up toward whoever stands at the arm (retinal-scan prop for the dashboard),
holding torque; `python scan_pose.py rest` eases back to rest and releases torque. Pose overridable:
SO101_SCAN_POSE="pan,lift,elbow,wrist,roll,grip" (degrees). Skips silently if another process owns the arm."""
import os, sys, math, time
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SCAN = dict(zip(J, [float(x) for x in os.environ.get("SO101_SCAN_POSE", "0,-30,0,-40,0,8").split(",")]))
REST = dict(shoulder_pan=0, shoulder_lift=-100, elbow_flex=95, wrist_flex=72, wrist_roll=0, gripper=1)
rest = len(sys.argv) > 1 and sys.argv[1] == "rest"
import lerobot.robots.so_follower as sf
Cfg = getattr(sf, "SO101FollowerConfig", None) or sf.SOFollowerConfig; Rob = getattr(sf, "SO101Follower", None) or sf.SOFollower
_mk = lambda: Rob(Cfg(port=os.environ.get("FOLLOWER", "/dev/so101_follower"), id=os.environ.get("FOLLOWER_ID", "ll_follower"), cameras={}, disable_torque_on_disconnect=rest))
def _connect_retry(mk, n=4):
    for i in range(n):
        r = mk()
        try: r.connect(); return r
        except Exception as e:
            print(f"connect attempt {i+1}/{n} failed: {str(e).splitlines()[0][:80]}", flush=True)
            try: r.bus.port_handler.closePort()
            except Exception: pass
            time.sleep(1.5)
    raise SystemExit("bus not responding; check the servo cables / USB")
robot = _connect_retry(_mk); HZ = 30
obs = robot.get_observation(); cur = {j: float(obs[f"{j}.pos"]) for j in J}; tgt = REST if rest else SCAN
n = int(2.5 * HZ)
for i in range(1, n + 1):
    a = (1 - math.cos(math.pi * i / n)) / 2
    robot.send_action({f"{j}.pos": cur[j] + (tgt[j] - cur[j]) * a for j in J}); time.sleep(1 / HZ)
robot.disconnect(); print("rest" if rest else "scan pose", {j: round(tgt[j]) for j in J})
