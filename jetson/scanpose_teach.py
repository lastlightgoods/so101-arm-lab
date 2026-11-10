"""so101-scanpose-teach : set the retinal-scan pose by hand. Torque goes off so you can pose the follower (or drive it with
the leader in another window), press Enter, and the pose is saved as SO101_SCAN_POSE in ~/so101/env.sh and held."""
import os, re, time
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
import lerobot.robots.so_follower as sf
Cfg = getattr(sf, "SO101FollowerConfig", None) or sf.SOFollowerConfig; Rob = getattr(sf, "SO101Follower", None) or sf.SOFollower
_mk = lambda: Rob(Cfg(port=os.environ.get("FOLLOWER", "/dev/so101_follower"), id=os.environ.get("FOLLOWER_ID", "ll_follower"), cameras={}, disable_torque_on_disconnect=False))
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
robot = _connect_retry(_mk)
robot.bus.disable_torque(); print("\nTorque OFF. Pose the arm where the wrist camera should look at you, then press Enter (Ctrl-C aborts).", flush=True)
try: input()
except KeyboardInterrupt: robot.bus.enable_torque(); robot.disconnect(); raise SystemExit("\naborted, pose unchanged") from None
obs = robot.get_observation(); pose = [round(float(obs[f"{j}.pos"]), 1) for j in J]
robot.bus.enable_torque(); robot.send_action({f"{j}.pos": p for j, p in zip(J, pose)}); time.sleep(0.3); robot.disconnect()
env = os.path.expanduser("~/so101/env.sh"); s = open(env).read(); line = 'export SO101_SCAN_POSE="' + ",".join(str(p) for p in pose) + '"'
s2, n = re.subn(r'^export SO101_SCAN_POSE="[^"]*"', line, s, flags=re.M)
if not n: s2 = s + "\n" + line + "   # retinal-scan pose (taught by hand)\n"
open(env, "w").write(s2); print("saved:", dict(zip(J, pose)), "\n-> SO101_SCAN_POSE in env.sh; the arm is holding it now. Open a new terminal (or source env.sh) to use it.")
