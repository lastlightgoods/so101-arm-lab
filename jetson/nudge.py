"""Move the follower a little off the rest pose so an ACT policy trained on demos with idle starts
does not freeze (from the exact rest pose it predicts "stay" forever). Eased 1.5 s move, no cameras."""
import os, sys, time, math
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
delta = dict(shoulder_lift=float(sys.argv[1]) if len(sys.argv) > 1 else 18.0, elbow_flex=-14.0, wrist_flex=-4.0)
import lerobot.robots.so_follower as _sf
Cfg = getattr(_sf, "SO101FollowerConfig", None) or _sf.SOFollowerConfig; Rob = getattr(_sf, "SO101Follower", None) or _sf.SOFollower
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
try:
    obs = robot.get_observation(); cur = {j: float(obs[f"{j}.pos"]) for j in J}
    tgt = {j: cur[j] + delta.get(j, 0.0) for j in J}
    n = 45
    for i in range(1, n + 1):
        a = (1 - math.cos(math.pi * i / n)) / 2
        robot.send_action({f"{j}.pos": cur[j] + (tgt[j] - cur[j]) * a for j in J}); time.sleep(1.5 / n)
    print("nudged:", {j: round(tgt[j], 1) for j in J})
finally:
    robot.disconnect()
