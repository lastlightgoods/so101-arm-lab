"""SO-101 dance: joints ride phase-shifted sinusoids locked to a tempo, around a safe pose above the mat.
    python dance.py [--bpm 100] [--seconds 30] [--energy 1.0]
Eases in from wherever the arm is (2 s), dances, eases back to rest. Ctrl-C returns to rest too."""
import argparse, math, os, time
ap = argparse.ArgumentParser(); ap.add_argument("--bpm", type=float, default=100); ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--energy", type=float, default=1.0); a = ap.parse_args()
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BASE = dict(shoulder_pan=0, shoulder_lift=-45, elbow_flex=45, wrist_flex=40, wrist_roll=0, gripper=15)
REST = dict(shoulder_pan=0, shoulder_lift=-100, elbow_flex=95, wrist_flex=72, wrist_roll=0, gripper=1)
# amplitude (deg), beats per cycle, phase (beats)   -> a bob every beat, a sway every 4, a roll every 8, chomps on the 2 and 4
MOVES = dict(shoulder_pan=(28, 4, 0.0), shoulder_lift=(10, 1, 0.0), elbow_flex=(14, 1, 0.5), wrist_flex=(22, 2, 0.25), wrist_roll=(50, 8, 0.0), gripper=(0, 1, 0))
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
robot = _connect_retry(_mk); HZ = 30
def send(p): robot.send_action({f"{j}.pos": float(p[j]) for j in J})
def ease_to(target, dur):
    obs = robot.get_observation(); cur = {j: float(obs[f"{j}.pos"]) for j in J}; n = max(1, int(dur * HZ))
    for i in range(1, n + 1):
        s = (1 - math.cos(math.pi * i / n)) / 2; send({j: cur[j] + (target[j] - cur[j]) * s for j in J}); time.sleep(1 / HZ)
try:
    ease_to(BASE, 2.0); beat = 60.0 / a.bpm; t0 = time.perf_counter(); print(f"dancing at {a.bpm:.0f} bpm for {a.seconds:.0f}s", flush=True)
    while (t := time.perf_counter() - t0) < a.seconds:
        b = t / beat; ramp = min(1.0, t / 2.0, (a.seconds - t) / 2.0) * a.energy   # fade in/out over 2 s
        p = {j: BASE[j] + ramp * amp * math.sin(2 * math.pi * (b / per + ph)) for j, (amp, per, ph) in MOVES.items()}
        bar = b % 4; p["gripper"] = BASE["gripper"] + (45 * ramp if (1.0 <= bar < 1.5 or 3.0 <= bar < 3.5) else 0)   # chomp on 2 and 4
        send(p); time.sleep(max(0, (t0 + (int(t * HZ) + 1) / HZ) - time.perf_counter()))
finally:
    ease_to(REST, 2.5); robot.disconnect(); print("back at rest")
