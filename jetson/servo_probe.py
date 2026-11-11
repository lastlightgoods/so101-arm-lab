#!/usr/bin/env python3
"""servo_probe.py - the Jetson side of tools/servo_play.py: measure the play (backlash and compliance) of the shoulder lift and the elbow, and
optionally try the shoulder lift's P gain, in ONE bus session. Same shape as pick_seq.py: run over ssh, holds ~/so101/.arms_busy so the dashboard
poller lets go of the port, refuses while a lerobot job holds the arm, torque stays on at the end. After a NORMAL end (done) the arm goes back to the REST pose (after P is restored); after an abort it holds where it is.

  python servo_probe.py --plan plan.json [--max-deg-s 50] [--force]
  python servo_probe.py --plan plan.json --fake        # the same code against FakeRig (no bus, no lerobot): the --dry-run of tools/servo_play.py

plan.json (written by tools/servo_play.py, which has already put every commanded pose through the workspace box):
  {"poses": [{"name", "joints": {5 arm joints, LeRobot deg}, "gripper", "approach_deg": {joint: deg}}], "test_joints": [...], "reps": 3, "settle_s": 1.0,
   "limits": {"min_volts", "max_temp_c", "max_load", "max_osc_deg"}, "p_test": null | {"joint": "shoulder_lift", "scales": [1, 1.5, 2], "pose": "far", "move_to": "near"}}

One JSON line per event on stdout: settings, point (one settled reading of one joint at one target, approached from "below" or "above"), p_value,
p_write_ignored, abort, restore, done, rest (the move back to REST after done; or rest_failed). Exit 0 = done, 2 = aborted (restored), 3 = a P value could NOT be verified as restored (read the restore line).

What it writes to the servos: NOTHING, except P_Coefficient of the one joint named in p_test, and only with Lock == 1 (the state LeRobot leaves a torque-enabled
arm in: feetech.py enable_torque writes Lock=1). It never writes Lock, Torque_Enable or any other register, so nothing goes through the unlocked EEPROM
path that LeRobot's configure() uses (disable_torque writes Lock=0 first). Whether the firmware then keeps the value in RAM only or ignores the write is
checked by reading it back; if it is ignored the P test says so and stops. The original P is written back in a finally block and read back to verify.
"""
import argparse, json, math, os, random, signal, statistics, subprocess, sys, time
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # why: so101-stop sends TERM; same clean unwind as Ctrl-C (and the finally below still restores P)

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = J[:5]
BUSY = os.path.expanduser("~/so101/.arms_busy")
MAX_DEG_S = 50.0                      # the pick loop's closed-loop speed cap (tools/pick_loop.py SPEED["closed"]); never exceeded here
EEPROM_BELOW = 40                     # lerobot/motors/feetech/tables.py: addresses 5..39 are the "EPROM" block, 40.. the "SRAM" block (P_Coefficient is 21)
READ_REGS = ("Present_Load", "Present_Current", "Present_Voltage", "Present_Temperature")
SETTING_REGS = ("P_Coefficient", "I_Coefficient", "D_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone", "Max_Torque_Limit", "Protection_Current", "Acceleration", "Lock", "Torque_Enable", "Max_Temperature_Limit")
P_MAX = 64                            # hard cap on any P value written (LeRobot default 16; a byte register, but nothing near 254 is ever tried)
REST = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 72.0, "wrist_roll": 0.5}   # same as tools/pick_loop.py REST (this script is shipped alone, so the copy lives here)
LIMITS = {"min_volts": 9.0, "max_temp_c": 60.0, "max_load": 800, "max_osc_deg": 0.5}
# why 9.0 V: the 12 V STS3215 is specified for 6-12.6 V (docs/stronger-arm.md, Feetech page); 6 V is where it stops working, a 12 V supply that sags
# to 9 V under a 4.2 A-class load is already in brown-out trouble for the Waveshare board and the other five servos. Override with --min-volts.
# why 60 C: the servo's own shutdown is Max_Temperature_Limit (70 C factory, busheal plugin comment); STS3215 held 48 C for an hour at 1 kg x 15 cm and
# overheated at 71 C after ~90 min of oscillation (docs/stronger-arm.md); 60 C leaves a margin and is TC's limit.


class Abort(Exception):
    def __init__(self, reason, detail=None): super().__init__(reason); self.reason = reason; self.detail = detail or {}


# ---------------------------------------------------------------- rigs: the only things that touch a bus
class Rig:
    """What the measurement needs from a servo bus. Subclasses give pos / send / _read / _write; write_p is the one guarded way to change a register."""
    def pos(self): raise NotImplementedError
    def send(self, cmd): raise NotImplementedError
    def _read(self, name, joint): raise NotImplementedError
    def _write(self, name, joint, value): raise NotImplementedError
    def close(self): pass

    def reg(self, name, joint): return int(self._read(name, joint))

    def write_p(self, joint, value, allowed_joint, hi=P_MAX):
        """The ONLY register write in this module. P_Coefficient of `allowed_joint`, 1..P_MAX, and only while Lock reads 1 (EEPROM locked)."""
        if joint != allowed_joint: raise ValueError(f"refusing to write P to {joint}: only {allowed_joint} is under test")
        if not (1 <= int(value) <= hi): raise ValueError(f"refusing P={value}: outside 1..{hi}")
        if self.reg("Lock", joint) != 1: raise Abort("lock_not_set", {"joint": joint, "why": "Lock != 1; a write now would go through the unlocked EEPROM path, which this tool never uses"})
        self._write("P_Coefficient", joint, int(value))


class RealRig(Rig):
    def __init__(self, robot): self.robot = robot
    def pos(self): obs = self.robot.get_observation(); return {j: float(obs[f"{j}.pos"]) for j in J}
    def send(self, cmd): self.robot.send_action({f"{j}.pos": v for j, v in cmd.items()})
    def _read(self, name, joint): return self.robot.bus.read(name, joint, normalize=False)
    def _write(self, name, joint, value): self.robot.bus.write(name, joint, value, normalize=False)   # lerobot motors_bus.write: no Lock / torque side effects
    def close(self): self.robot.disconnect()   # disable_torque_on_disconnect=False below, so the arm holds


class VClock:
    """Virtual time for the fake: sleep() advances it, nothing really waits."""
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, s): self.t += max(0.0, s)


class FakeRig(Rig):
    """A servo bus with a known model, so the analysis can be checked against numbers it must recover (tests/tools/test_servo_play.py).
    Per joint: an encoder-visible play `backlash_deg` (play operator: the read-back sits half the play short when approached from below, half over when approached
    from above), a gravity deflection of `compliance[j]` deg per Present_Load unit (load = load_k[j] x the joint's horizontal lever in mm), all divided by
    P/16 (a stiffer gain sags less: the fake's assumption, not a measurement of the real servo). Current ~ load + P x speed; voltage dips with current; position
    chatters above P = p_osc. `lock_mode`: "ram" (a P write with Lock=1 takes effect), "ignore" (it is dropped). Every write is kept in .writes with the Lock value then."""
    def __init__(self, clock=None, start=None, backlash_deg=0.8, compliance=None, load_k=None, lever_fn=None, p0=16, lock=1, lock_mode="ram", p_osc=40, noise_deg=0.01,
                 temp0=32, temp_slope=0.0, volt0=120, volt_per_current=0.0, seed=1):
        self.clock = clock or VClock(); self.rng = random.Random(seed); self.b = dict(backlash_deg) if isinstance(backlash_deg, dict) else {j: float(backlash_deg) for j in J}
        self.comp = compliance or {"shoulder_lift": 0.006, "elbow_flex": 0.004}; self.load_k = load_k or {"shoulder_lift": 1.4, "elbow_flex": 1.0}
        self.lever_fn = lever_fn or self._lever; self.regs = {j: {"P_Coefficient": p0, "I_Coefficient": 0, "D_Coefficient": 32, "CW_Dead_Zone": 1, "CCW_Dead_Zone": 1, "Max_Torque_Limit": 1000,
                                                              "Protection_Current": 500, "Acceleration": 254, "Lock": lock, "Torque_Enable": 1, "Max_Temperature_Limit": 70} for j in J}
        self.lock_mode, self.p_osc, self.noise, self.temp0, self.temp_slope, self.volt0, self.vpc = lock_mode, p_osc, noise_deg, temp0, temp_slope, volt0, volt_per_current
        self.writes = []; self.eeprom_writes = []; self.reads = 0
        self.cmd = dict(start or {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.0}); self.y = dict(self.cmd)
        self.speed = {j: 0.0 for j in J}; self.t_last = self.clock.t; self.load_now = {j: 0.0 for j in J}; self._settle(self.cmd)

    @staticmethod
    def _lever(cmd):   # a planar stand-in: horizontal mm from the joint to the tip (only used when a test does not pass a lever_fn built on the real FK)
        a, b = math.radians(cmd["shoulder_lift"]), math.radians(cmd["elbow_flex"])
        return {"shoulder_lift": 116.0 * math.sin(a) + 135.0 * math.sin(a + b) + 100.0, "elbow_flex": 135.0 * math.sin(a + b) + 100.0}

    def _pscale(self, j): return self.regs[j]["P_Coefficient"] / 16.0
    def _settle(self, cmd):
        lev = self.lever_fn(cmd)
        for j in J:
            self.load_now[j] = self.load_k.get(j, 0.0) * lev.get(j, 0.0)
            u = cmd[j] - self.comp.get(j, 0.0) * self.load_now[j] / self._pscale(j)
            half = self.b[j] / 2.0; self.y[j] = max(min(self.y[j], u + half), u - half)   # the play operator

    def send(self, cmd):
        now = self.clock.now(); dt = max(now - self.t_last, 1e-3); self.t_last = now
        for j in J: self.speed[j] = abs(cmd[j] - self.cmd[j]) / dt
        self.cmd = dict(cmd); self._settle(self.cmd)

    def pos(self):
        osc = {j: (0.12 * max(0, self.regs[j]["P_Coefficient"] - self.p_osc) if self.regs[j]["P_Coefficient"] > self.p_osc else 0.0) for j in J}
        return {j: self.y[j] + self.rng.gauss(0, self.noise) + osc[j] * math.sin(2 * math.pi * 7.0 * self.clock.now() + J.index(j)) for j in J}

    def _current(self, j): return 0.28 * abs(self.load_now[j]) + 0.9 * self.regs[j]["P_Coefficient"] * min(self.speed[j], 60.0) / 10.0

    def _read(self, name, joint):
        self.reads += 1
        if name == "Present_Load": return int(self.load_now[joint])
        if name == "Present_Current": return int(self._current(joint))
        if name == "Present_Voltage": return int(self.volt0 - self.vpc * sum(self._current(j) for j in J))
        if name == "Present_Temperature": return int(self.temp0 + self.temp_slope * self.clock.now())
        return self.regs[joint][name]

    def _write(self, name, joint, value):
        lock = self.regs[joint]["Lock"]; self.writes.append((name, joint, int(value), lock))
        addr_eeprom = name in ("P_Coefficient", "I_Coefficient", "D_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone", "Max_Torque_Limit", "Protection_Current", "Lock")   # in the table's EPROM block
        if addr_eeprom and lock == 0: self.eeprom_writes.append((name, joint, int(value))); self.regs[joint][name] = int(value); return
        if addr_eeprom and self.lock_mode == "ignore": return
        self.regs[joint][name] = int(value)


# ---------------------------------------------------------------- measuring
def volts(raw): return raw / 10.0   # Present_Voltage is in 0.1 V


def check(vals, lim, where=""):
    """Raise Abort when a reading is outside the limits. vals: {volts, temp_c, load}."""
    if vals.get("volts") is not None and vals["volts"] < lim["min_volts"]: raise Abort("undervoltage", {"volts": vals["volts"], "limit": lim["min_volts"], "where": where})
    if vals.get("temp_c") is not None and vals["temp_c"] > lim["max_temp_c"]: raise Abort("overtemperature", {"temp_c": vals["temp_c"], "limit": lim["max_temp_c"], "where": where})
    if vals.get("load") is not None and abs(vals["load"]) > lim["max_load"]: raise Abort("overload", {"load": vals["load"], "limit": lim["max_load"], "where": where})


def read_vitals(rig, joint, lim, where=""):
    v = {"load": rig.reg("Present_Load", joint), "current": rig.reg("Present_Current", joint), "volts": volts(rig.reg("Present_Voltage", joint)), "temp_c": rig.reg("Present_Temperature", joint)}
    check(v, lim, where); return v


def sample(rig, joint, lim, sleep, n=8, dt=0.06, where=""):
    """n readings of `joint` over ~n*dt s: median position (deg) and std, median load / current, max current, min volts, max temp."""
    P, L, C, V, T = [], [], [], [], []
    for _ in range(n):
        P.append(rig.pos()[joint]); v = read_vitals(rig, joint, lim, where); L.append(v["load"]); C.append(v["current"]); V.append(v["volts"]); T.append(v["temp_c"]); sleep(dt)
    return {"pos": statistics.median(P), "pos_std": statistics.pstdev(P), "load": statistics.median(L), "current": statistics.median(C), "current_max": max(C), "volts_min": min(V), "temp_c": max(T)}


def ease_move(rig, cmd, tgt, max_deg_s, sleep, min_s=0.6, tick=None):
    """Half-cosine move of the commanded joints from `cmd` to `tgt` at <= max_deg_s (pick_seq.move, 40 Hz); returns the new command. tick(i, n) runs after every step."""
    max_deg_s = min(float(max_deg_s), MAX_DEG_S); biggest = max(abs(tgt[j] - cmd[j]) for j in J)
    secs = max(min_s, biggest / max_deg_s * (math.pi / 2)); n = max(12, int(secs * 40))   # why pi/2: the half-cosine's peak speed is pi/2 x the mean, and the cap is on the peak
    for i in range(1, n + 1):
        e = (1 - math.cos(math.pi * i / n)) / 2; rig.send({j: cmd[j] + (tgt[j] - cmd[j]) * e for j in J}); sleep(secs / n)
        if tick: tick(i, n)
    return dict(tgt)


def pose_cmd(pose, grip): c = {j: float(pose["joints"][j]) for j in ARM}; c["gripper"] = float(pose.get("gripper", grip)); return c


def settings_of(rig, joints):
    out = {}
    for j in joints:
        out[j] = {}
        for r in SETTING_REGS:
            try: out[j][r] = rig.reg(r, j)
            except Exception as e: out[j][r] = f"err: {type(e).__name__}"
    return out


def run_play(plan, rig, emit, clock_now=time.time, sleep=time.sleep, max_deg_s=MAX_DEG_S):
    """Backlash / compliance sweep, then the optional P test. Returns the exit code (0 done, 2 aborted, 3 P restore unverified). Always restores P."""
    lim = {**LIMITS, **(plan.get("limits") or {})}; joints = plan.get("test_joints") or ["shoulder_lift", "elbow_flex"]; ptest = plan.get("p_test")
    poses = {p["name"]: p for p in plan["poses"]}; reps = int(plan.get("reps", 3)); settle = float(plan.get("settle_s", 1.0)); t0 = clock_now()
    p_joint = ptest["joint"] if ptest else None; p_orig = None; p_now = None; code = 0; normal = False
    cur = rig.pos(); cmd = {j: cur[j] for j in J}
    try:
        emit({"event": "start", "t": 0.0, "pos": {j: round(v, 2) for j, v in cur.items()}, "limits": lim, "max_deg_s": min(max_deg_s, MAX_DEG_S)})
        emit({"event": "settings", "regs": settings_of(rig, sorted(set(joints + ([p_joint] if p_joint else []))))})
        if p_joint: p_orig = p_now = rig.reg("P_Coefficient", p_joint)
        for name, pose in poses.items():
            cmd = ease_move(rig, cmd, pose_cmd(pose, cmd["gripper"]), max_deg_s, sleep, min_s=1.5); sleep(settle)
            for jn in joints:
                c = float(pose["joints"][jn]); d = float((pose.get("approach_deg") or {}).get(jn, 3.0))
                for rep in range(reps):
                    for direction in ("below", "above"):
                        pre = dict(cmd); pre[jn] = c - d if direction == "below" else c + d
                        cmd = ease_move(rig, cmd, pre, max_deg_s, sleep, min_s=0.5); sleep(0.3)
                        goal = dict(cmd); goal[jn] = c; cmd = ease_move(rig, cmd, goal, max_deg_s, sleep, min_s=0.5); sleep(settle)
                        s = sample(rig, jn, lim, sleep, where=f"{name}/{jn}/{direction}")
                        emit({"event": "point", "t": round(clock_now() - t0, 1), "pose": name, "joint": jn, "dir": direction, "rep": rep, "target": round(c, 3), "pos": round(s["pos"], 3),
                              "err": round(s["pos"] - c, 3), "pos_std": round(s["pos_std"], 4), "load": s["load"], "current": s["current"], "volts": s["volts_min"], "temp_c": s["temp_c"]})
        if ptest:
            cmd = p_test(rig, ptest, poses, cmd, lim, emit, sleep, clock_now, t0, max_deg_s, settle, p_orig)
            p_now = rig.reg("P_Coefficient", p_joint)
        emit({"event": "done", "t": round(clock_now() - t0, 1)}); normal = True
    except Abort as a:
        emit({"event": "abort", "reason": a.reason, "detail": a.detail}); code = 2
    except KeyboardInterrupt:
        emit({"event": "abort", "reason": "interrupted", "detail": {}}); code = 2
    finally:
        if p_joint and p_orig is not None:
            try: p_now = rig.reg("P_Coefficient", p_joint)
            except Exception: p_now = None
            if p_now != p_orig: code = restore_p(rig, p_joint, p_orig, emit) or code
            else: emit({"event": "restore", "joint": p_joint, "original": p_orig, "readback": p_now, "ok": True, "wrote": False})
        if normal: go_rest(rig, cmd, emit, sleep, max_deg_s)   # why: 11 Nov 2026 the arm held the far pose over the mat for ~2 h after a normal end and the next camera nut search took the gripper for the nut. Only after done, and after P is back (so it moves at the default gain); an abort never starts new motion
    return code


def go_rest(rig, cmd, emit, sleep, max_deg_s):
    """Move the arm to REST with the same capped half-cosine as every other move here (the gripper stays as it is). Never raises: a failure is an event, the arm just holds."""
    try:
        tgt = {**{j: float(REST[j]) for j in ARM}, "gripper": cmd["gripper"]}; emit({"event": "rest", "from": {j: round(cmd[j], 2) for j in ARM}, "to": REST})
        ease_move(rig, cmd, tgt, max_deg_s, sleep, min_s=1.5)
    except Exception as e: emit({"event": "rest_failed", "why": f"{type(e).__name__}: {e}"[:160]})


def restore_p(rig, joint, orig, emit, tries=3):
    """Write the original P back and read it back; 3 tries. Returns 0 when verified, 3 when not (the next LeRobot connect rewrites P=16 anyway: so_follower.py configure())."""
    rb = None
    for i in range(tries):
        try:
            rig.write_p(joint, orig, joint, hi=254); rb = rig.reg("P_Coefficient", joint)
            if rb == orig: emit({"event": "restore", "joint": joint, "original": orig, "readback": rb, "ok": True, "wrote": True, "try": i + 1}); return 0
        except Abort as a: emit({"event": "restore", "joint": joint, "original": orig, "ok": False, "why": a.reason, "try": i + 1}); return 3
        except Exception as e: emit({"event": "restore", "joint": joint, "original": orig, "ok": False, "why": f"{type(e).__name__}: {e}"[:160], "try": i + 1})
    emit({"event": "restore", "joint": joint, "original": orig, "readback": rb, "ok": False, "why": "readback differs after retries", "try": tries}); return 3


def p_test(rig, pt, poses, cmd, lim, emit, sleep, clock_now, t0, max_deg_s, settle, p_orig):
    """For each P scale: write (RAM only), hold at pt['pose'] and measure sag / oscillation over 2 s, then move to pt['move_to'] and back sampling current and voltage on every
    tick. The first scale is the original value (no write). Stops climbing when the oscillation limit is hit or a write is ignored. Restoring is run_play's finally."""
    jn = pt["joint"]; far, near = poses[pt["pose"]], poses[pt.get("move_to") or pt["pose"]]; seen = set()
    for scale in pt.get("scales") or [1.0]:
        val = p_orig if abs(scale - 1.0) < 1e-9 else max(1, min(P_MAX, int(round(p_orig * scale))))
        if val in seen: continue
        seen.add(val)
        if val != rig.reg("P_Coefficient", jn):
            rig.write_p(jn, val, jn); rb = rig.reg("P_Coefficient", jn)
            if rb != val: emit({"event": "p_write_ignored", "joint": jn, "wanted": val, "readback": rb, "note": "Lock=1 write was not accepted; nothing more to test without unlocking, which needs TC's approval"}); return cmd
        cmd = ease_move(rig, cmd, pose_cmd(near, cmd["gripper"]), max_deg_s, sleep, min_s=1.5)   # why: always arrive at the hold pose from the same side, or the play (hysteresis) decides the sag reading instead of P
        cmd = ease_move(rig, cmd, pose_cmd(far, cmd["gripper"]), max_deg_s, sleep, min_s=1.5); sleep(settle)
        c = float(far["joints"][jn]); idle = read_vitals(rig, jn, lim, f"p{val}/idle")
        osc = sample(rig, jn, lim, sleep, n=40, dt=0.05, where=f"p{val}/hold")   # 2 s of readings at 20 Hz
        vmin, imax, peak_load = [idle["volts"]], [idle["current"]], [0]
        def tick(i, n, jn=jn):
            v = read_vitals(rig, jn, lim, f"p{val}/move"); vmin.append(v["volts"]); imax.append(v["current"]); peak_load.append(abs(v["load"]))
        cmd = ease_move(rig, cmd, pose_cmd(near, cmd["gripper"]), max_deg_s, sleep, min_s=1.5, tick=tick); sleep(0.5)
        cmd = ease_move(rig, cmd, pose_cmd(far, cmd["gripper"]), max_deg_s, sleep, min_s=1.5, tick=tick); sleep(settle)
        emit({"event": "p_value", "t": round(clock_now() - t0, 1), "joint": jn, "p": val, "scale": round(val / p_orig, 3), "sag_deg": round(osc["pos"] - c, 3), "osc_std_deg": round(osc["pos_std"], 4),
              "load": osc["load"], "current_hold": osc["current"], "current_peak_move": max(imax), "volts_idle": idle["volts"], "volts_min_move": min(vmin), "volts_dip": round(idle["volts"] - min(vmin), 2),
              "peak_load_move": max(peak_load), "temp_c": osc["temp_c"]})
        if osc["pos_std"] > lim["max_osc_deg"]: emit({"event": "p_stop", "reason": "oscillation", "p": val, "osc_std_deg": round(osc["pos_std"], 4), "limit": lim["max_osc_deg"]}); break
    return cmd


# ---------------------------------------------------------------- the real thing
def lerobot_running(): return subprocess.run(["pgrep", "-f", "lerobot-(record|teleoperate|rollout|calibrate|replay)"], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True); ap.add_argument("--max-deg-s", type=float, default=MAX_DEG_S); ap.add_argument("--force", action="store_true")
    ap.add_argument("--fake", action="store_true", help="FakeRig instead of the bus (no lerobot, no .arms_busy)")
    a = ap.parse_args(); plan = json.load(open(a.plan)); emit = lambda d: print(json.dumps(d), flush=True)
    if a.fake:
        clk = VClock(); first = plan["poses"][0]["joints"]; rig = FakeRig(clock=clk, start={**{j: first[j] for j in ARM}, "gripper": 0.0}); raise SystemExit(run_play(plan, rig, emit, clk.now, clk.sleep, a.max_deg_s))
    if lerobot_running() and not a.force: raise SystemExit("a lerobot job (teleop/record/rollout) is holding the arm; so101-stop first")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pick_seq   # why: the same connect (4 tries, busheal plugin, torque stays on) as every other bus session here
    made = not os.path.exists(BUSY)
    if made: open(BUSY, "w").close(); time.sleep(1.0)
    robot = pick_seq.connect(); code = 3
    try: code = run_play(plan, RealRig(robot), emit, time.time, time.sleep, a.max_deg_s)
    finally:
        robot.disconnect()
        if made:
            try: os.remove(BUSY)
            except FileNotFoundError: pass
    raise SystemExit(code)


if __name__ == "__main__": main()
