#!/usr/bin/env python3
"""fake_servo_bus.py - a pretend SO-101 follower bus for building and testing the telemetry without the arm.

FakeServoBus quacks like a connected lerobot FeetechMotorsBus for jetson/servo_telemetry.py (motors, _setup_sync_reader, sync_reader,
_is_comm_success, _normalize) and answers a block read with bytes produced by servo_telemetry.encode_block, so the REAL decode path runs.
A little physics sits behind the bytes: the goal follows a scripted pick, the position lags it, load comes from gravity (shoulder lift and
elbow carry the arm) plus acceleration, current follows load, temperature integrates current squared and cools, and the supply voltage sags
with total current. Faults can be injected (a hot servo, a brown-out, an overload bit). Time is explicit (`bus.t`), so tests and demos can run
minutes of arm time in milliseconds. Nothing here opens a port.

    bus = FakeServoBus(seed=1); bus.advance(60.0)            # one minute of picking, instantly
    from servo_telemetry import read_sample; s = read_sample(bus, now=bus.t)
"""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jetson"))
import servo_telemetry as st

JOINTS = st.JOINTS
# a pick as joint goals in LeRobot-style degrees (gripper %): rest, hover, grasp, lift, carry, place
POSES = [
    ("rest",  {"shoulder_pan": 0, "shoulder_lift": -100, "elbow_flex": 95, "wrist_flex": 72, "wrist_roll": 0, "gripper": 1}),
    ("hover", {"shoulder_pan": 18, "shoulder_lift": -35, "elbow_flex": 55, "wrist_flex": 60, "wrist_roll": 0, "gripper": 40}),
    ("grasp", {"shoulder_pan": 18, "shoulder_lift": -12, "elbow_flex": 40, "wrist_flex": 55, "wrist_roll": 0, "gripper": 40}),
    ("close", {"shoulder_pan": 18, "shoulder_lift": -12, "elbow_flex": 40, "wrist_flex": 55, "wrist_roll": 0, "gripper": 12}),
    ("lift",  {"shoulder_pan": 18, "shoulder_lift": -40, "elbow_flex": 60, "wrist_flex": 62, "wrist_roll": 0, "gripper": 12}),
    ("place", {"shoulder_pan": -22, "shoulder_lift": -20, "elbow_flex": 45, "wrist_flex": 58, "wrist_roll": 0, "gripper": 12}),
    ("open",  {"shoulder_pan": -22, "shoulder_lift": -20, "elbow_flex": 45, "wrist_flex": 58, "wrist_roll": 0, "gripper": 40}),
]
HOLD_S, MOVE_S = 1.6, 1.4
CYCLE_S = len(POSES) * (HOLD_S + MOVE_S)
GRAVITY = {"shoulder_pan": 0.0, "shoulder_lift": 1.0, "elbow_flex": 0.55, "wrist_flex": 0.18, "wrist_roll": 0.02, "gripper": 0.05}   # relative torque from the arm's weight
CAL = {j: (800, 3200) for j in JOINTS}; CAL["gripper"] = (1400, 2800)


class _Motor:
    def __init__(self, id_): self.id = id_; self.model = "sts3215"


class _Reader:
    def __init__(self, bus): self.bus = bus; self.ids = []; self.start_address = 0; self.data_length = 0; self.data_dict = {}
    def clearParam(self): self.ids = []; self.data_dict = {}
    def addParam(self, i): self.ids.append(i)
    def txRxPacket(self):
        if self.bus.offline: return -6   # COMM_RX_TIMEOUT
        self.data_dict = {}
        for i in self.ids:
            if i in self.bus.silent: continue
            full = self.bus.block(i)
            a = self.start_address - st.BLOCK[0]
            if a >= 0: self.data_dict[i] = list(full[a:a + self.data_length])
            else: self.data_dict[i] = list(self.bus.limits_block(i))   # the EEPROM limits block (13..38)
        return 0
    def isAvailable(self, i, addr, length): return i in self.data_dict
    def getData(self, i, addr, length): d = self.data_dict[i]; return d[addr - self.start_address] if length == 1 else d[addr - self.start_address] | (d[addr - self.start_address + 1] << 8)


class FakeServoBus:
    def __init__(self, seed=0, volts=12.3, t0=1_000_000.0, ambient=27.0):
        self.rng = random.Random(seed); self.motors = {n: _Motor(i) for i, n in enumerate(JOINTS, 1)}
        self.t = t0; self.port = "/dev/so101_follower"; self.sync_reader = _Reader(self)
        self.supply_v = volts; self.offline = False; self.silent = set(); self.reads = 0
        self.temp = {n: ambient + self.rng.random() for n in JOINTS}; self.pos = dict(POSES[0][1]); self.goal = dict(POSES[0][1]); self.vel = {n: 0.0 for n in JOINTS}
        self.load = {n: 0.0 for n in JOINTS}; self.cur = {n: 0.0 for n in JOINTS}; self.status = {n: 0 for n in JOINTS}
        self.extra_heat = {}; self.extra_drop_v = 0.0; self.sag_extra = {}; self.torque_on = True; self.stall = {}; self.script = True

    # -- the arm's motion and body --------------------------------------------------------------------
    def goal_at(self, t):
        k = (t % CYCLE_S); i = int(k // (HOLD_S + MOVE_S)); r = k - i * (HOLD_S + MOVE_S)
        a, b = POSES[i - 1][1], POSES[i][1]
        f = 1.0 if r >= MOVE_S else (1 - math.cos(math.pi * r / MOVE_S)) / 2
        return {n: a[n] + (b[n] - a[n]) * f for n in JOINTS}

    def advance(self, seconds, dt=0.05):
        for _ in range(int(round(seconds / dt))): self._step(dt)

    def _step(self, dt):
        self.t += dt
        if self.script: self.goal = self.goal_at(self.t)
        total = 0.0
        for n in JOINTS:
            err = self.goal[n] - self.pos[n]
            sag = (0.8 + self.sag_extra.get(n, 0.0)) * GRAVITY[n] * 2.5   # degrees the arm droops below the commanded angle while holding
            target = self.goal[n] - sag if self.torque_on else self.pos[n]
            vel = (target - self.pos[n]) / 0.12; vel = max(-220.0, min(220.0, vel))
            if self.stall.get(n): vel = 0.0
            acc = (vel - self.vel[n]) / dt; self.vel[n] = vel; self.pos[n] += vel * dt
            torque = GRAVITY[n] * 0.4 + abs(acc) / 4000.0 + abs(err) / 120.0 + (0.7 if self.stall.get(n) else 0.0)
            self.load[n] = (1 if (err + sag) >= 0 else -1) * min(100.0, 100 * torque) if self.torque_on else 0.0
            self.cur[n] = (0.08 + 1.7 * min(1.0, torque)) if self.torque_on else 0.02
            total += self.cur[n]
            heat = self.cur[n] ** 2 * 0.012 + self.extra_heat.get(n, 0.0)
            self.temp[n] = min(72.0, self.temp[n] + (heat - (self.temp[n] - 27.0) * 0.006) * dt)   # the servo's own cut holds it near 70
        self.v = self.supply_v - 0.15 * total - self.extra_drop_v
        for n in JOINTS:
            self.status[n] = (1 if self.v < 9.5 else 0) | (4 if self.temp[n] >= 70 else 0) | (32 if self.stall.get(n) and self.cur[n] > 2.0 else 0)

    # -- what the bus returns -------------------------------------------------------------------------
    def _raw(self, n, v):
        lo, hi = CAL[n]; mid = (lo + hi) / 2
        if n == "gripper": return int(round(lo + (v / 100.0) * (hi - lo)))
        return int(round(mid + v * 4095 / 360.0))

    def block(self, i):
        n = JOINTS[i - 1]; self.reads += 1
        return st.encode_block(on=int(self.torque_on), goal=self._raw(n, self.goal[n]), pos=self._raw(n, self.pos[n]), vel=self.vel[n] * 4096 / 360.0,
                               load=self.load[n] * 10, volt=getattr(self, "v", self.supply_v), temp=int(self.temp[n]), status=self.status[n],
                               moving=int(abs(self.vel[n]) > 3), cur=self.cur[n] * 1000 / st.MA_PER_COUNT)

    def limits_block(self, i):
        b = bytearray(st.LIMITS_BLOCK[1]); o = st.LIMITS_BLOCK[0]
        b[13 - o] = 70; b[14 - o] = 140; b[15 - o] = 40; b[16 - o] = 1000 & 0xFF; b[17 - o] = 1000 >> 8; b[28 - o] = 250 & 0xFF; b[29 - o] = 250 >> 8; b[36 - o] = 90; b[38 - o] = 20
        return bytes(b)

    def image(self, i):
        """The servo's registers 0..90 as bytes: identity, the EEPROM limits and the live block, so single reads and block reads agree."""
        b = bytearray(91); b[0] = 3; b[1] = 15; b[3] = 777 & 0xFF; b[4] = 777 >> 8; b[5] = i
        b[13:39] = self.limits_block(i); b[40:71] = self.block(i); return bytes(b)

    def _read(self, address, length, motor_id, *, num_retry=0, raise_on_error=True, err_msg=""):
        """lerobot's single-register read: (value, comm, error)."""
        if self.offline or motor_id in self.silent:
            if raise_on_error: raise ConnectionError(err_msg or "no status packet")
            return 0, -6, 0
        img = self.image(motor_id); self.reads += 1
        return int.from_bytes(img[address:address + length], "little"), 0, 0

    # -- the slice of FeetechMotorsBus that servo_telemetry uses ---------------------------------------
    def _setup_sync_reader(self, ids, addr, length):
        r = self.sync_reader; r.clearParam(); r.start_address = addr; r.data_length = length
        for i in ids: r.addParam(i)
    def _is_comm_success(self, comm): return comm == 0
    def _normalize(self, ids_values):
        out = {}
        for i, raw in ids_values.items():
            n = JOINTS[i - 1]; lo, hi = CAL[n]; raw = max(lo, min(hi, raw))
            out[i] = (raw - lo) / (hi - lo) * 100.0 if n == "gripper" else (raw - (lo + hi) / 2) * 360.0 / 4095
        return out


def demo_stream(bus, seconds, hz=5.0, hot=None):
    """Yield one telemetry sample per 1/hz of arm time for `seconds`; `hot` = {joint: extra heating} applies from the start."""
    if hot: bus.extra_heat.update(hot)
    for _ in range(int(seconds * hz)):
        bus.advance(1.0 / hz); yield st.read_sample(bus, now=bus.t)


def waypoint_tele(bus, goal, seconds, hz=4.0):
    """Drive the pretend arm to `goal` ({joint: deg, gripper %}) over `seconds` of arm time, sampling at `hz`; the Summary.out() pick_seq attaches to a waypoint row."""
    summary = st.Summary(); bus.goal.update({k: float(v) for k, v in goal.items() if k in bus.goal})
    for _ in range(max(1, int(seconds * hz))):
        bus.advance(1.0 / hz); summary.add(st.read_sample(bus, now=bus.t))
    return summary.out()
