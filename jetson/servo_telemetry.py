#!/usr/bin/env python3
"""servo_telemetry.py - read-only health telemetry for the Feetech STS servos on the SO-101 bus (docs/telemetry.md).

One block read per sample: SRAM addresses 40..70 of every servo in ONE sync-read packet (Torque_Enable, goal, torque limit,
present position / speed / load / voltage / temperature, status bits, moving flag, present current). Nothing is ever written to
a servo: this module only builds read packets (a sync read of a register block), so no setting can change.

    from servo_telemetry import read_sample
    s = read_sample(bus, now=time.time())        # bus = a connected lerobot FeetechMotorsBus; None when the bus did not answer
    s["j"]["shoulder_lift"] -> {"temp_c": 41, "volt_v": 11.9, "load_pct": -12.4, "cur_a": 0.83, "sag": 1.9, "moving": 0, ...}
    s["total_a"], s["volt_min"], s["temp_max"], s["flags"], s["dt_ms"]

It lives in jetson/ because that folder is what gets copied (flat) to the Jetson; arm_state.py, pick_seq.py and the busheal plugin
import it from there, Mac tools (pick_loop, autoloop, the dashboard) load it by path. Pure Python apart from the bus object it is handed.

Units (Feetech STS memory table, http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0, and
lerobot/motors/feetech/tables.py in LeRobot 0.6.1; STS3215 and STS3250 share one table, so one decoder serves both):
  position 0..4095 = one turn; speed in steps/s, sign bit 15; load 0.1 % of the rated PWM, sign bit 10; voltage 0.1 V;
  temperature 1 degC; current 6.5 mA per count (the scale is on the "to verify" list: tools/verify_telemetry.py checks it against a meter).
"""
import math, time

# ---------------------------------------------------------------- the register list (what a servo can report over the bus)
# (name, address, bytes, kind, unit/meaning, in the block we poll, status on TC's servos)
# status: "seen" = the repo already reads it successfully on the real arm (cited), "to verify" = nothing in the repo proves it reads back
# on these servos; tools/verify_telemetry.py reads every row below with the arm idle and prints which ones answered.
REGISTERS = [
    # name,                     addr, n, kind,      meaning,                                             polled, status
    ("Model_Number",              3, 2, "id",      "777 = STS3215, 2825 = STS3250 (LeRobot MODEL_NUMBER_TABLE)", False, "seen"),   # lerobot's connect() handshake pings it
    ("Firmware_Major_Version",    0, 1, "id",      "firmware (3.9 / 3.15 reported on this arm, NEXT_STEPS.md)", False, "seen"),
    ("ID",                        5, 1, "config",  "bus id",                                           False, "seen"),
    ("Max_Temperature_Limit",    13, 1, "limit",   "degC at which the servo cuts torque (factory 70)",   False, "to verify"),
    ("Max_Voltage_Limit",        14, 1, "limit",   "0.1 V; over this the servo flags a voltage error",   False, "to verify"),
    ("Min_Voltage_Limit",        15, 1, "limit",   "0.1 V; under this the servo flags a voltage error",  False, "to verify"),
    ("Max_Torque_Limit",         16, 2, "limit",   "0..1000 permille of rated torque (busheal sets the gripper's)", False, "seen"),
    ("Protection_Current",       28, 2, "limit",   "counts; over this for Over_Current_Protection_Time the servo trips", False, "seen"),
    ("Protective_Torque",        34, 1, "limit",   "torque kept after an overload trip (busheal writes 0 on the gripper)", False, "seen"),
    ("Protection_Time",          35, 1, "limit",   "overload timer (busheal writes 254 on the gripper)", False, "seen"),
    ("Overload_Torque",          36, 1, "limit",   "load threshold that starts the overload timer",      False, "to verify"),
    ("Over_Current_Protection_Time", 38, 1, "limit", "over-current timer",                                False, "to verify"),
    ("LED_Alarm_Condition",      20, 1, "limit",   "which Status bits blink the servo LED",              False, "to verify"),
    ("Unloading_Condition",      19, 1, "limit",   "which Status bits release the torque",               False, "to verify"),
    ("Torque_Enable",            40, 1, "live",    "1 = torque on, 0 = free",                            True,  "seen"),    # lerobot reads and writes it every connect
    ("Acceleration",             41, 1, "live",    "commanded acceleration ramp",                        True,  "to verify"),
    ("Goal_Position",            42, 2, "live",    "commanded position, steps (sign bit 15)",            True,  "seen"),    # busheal clear_overload() parks it from the present position
    ("Goal_Time",                44, 2, "live",    "commanded move time",                                True,  "to verify"),
    ("Goal_Velocity",            46, 2, "live",    "commanded speed, steps/s",                           True,  "to verify"),
    ("Torque_Limit",             48, 2, "live",    "active torque cap, permille",                        True,  "seen"),    # busheal writes it for the gripper
    ("Lock",                     55, 1, "live",    "EEPROM write lock",                                  True,  "to verify"),
    ("Present_Position",         56, 2, "live",    "steps, sign bit 15",                                 True,  "seen"),
    ("Present_Velocity",         58, 2, "live",    "steps/s, sign bit 15 (LeRobot ENCODINGS table)",      True,  "to verify"),   # tests read it on a fake bus only
    ("Present_Load",             60, 2, "live",    "0.1 % of rated PWM, sign bit 10 (LeRobot ENCODINGS table)", True, "to verify"),
    ("Present_Voltage",          62, 1, "live",    "0.1 V",                                              True,  "to verify"),
    ("Present_Temperature",      63, 1, "live",    "degC",                                               True,  "to verify"),
    ("Status",                   65, 1, "live",    "error bits: 0 voltage, 1 sensor, 2 temperature, 3 current, 5 overload", True, "to verify"),
    ("Moving",                   66, 1, "live",    "1 while the servo is moving to a goal",              True,  "to verify"),
    ("Present_Current",          69, 2, "live",    "counts, 6.5 mA each on the Feetech sheet",           True,  "seen"),    # busheal: gripper stall read 541 counts (6 Oct 2026); scale unverified
    ("Goal_Position_2",          71, 2, "live",    "second goal register (read-only)",                   False, "to verify"),
]
BLOCK = (40, 31)         # (first address, byte count): 40..70 covers every polled row above in one packet
LIMITS_BLOCK = (13, 26)  # 13..38: temperature, voltage, torque, protection settings; read once per connect, not per sample
STATUS_BITS = {0: "voltage", 1: "sensor", 2: "temp", 3: "current", 4: "bit4", 5: "overload"}   # Feetech sheet; bit 4 undocumented, kept as is
MODELS = {   # per model, what the dashboard marks: stall current A (datasheet, docs/stronger-arm.md) and the factory temperature cut
    "sts3215": {"number": 777, "stall_a": 2.7, "temp_cut_c": 70},
    "sts3250": {"number": 2825, "stall_a": 4.2, "temp_cut_c": 70},
}
MA_PER_COUNT = 6.5
TEMP_CUT_C = 70            # the overheat cut (Max_Temperature_Limit factory value); the real one is read once and replaces this
TEMP_WARN_C = 60           # lab choice: amber from here, red at TEMP_HOT_C
TEMP_HOT_C = 65
VOLT_WARN_V = 11.0         # lab choice for the 12 V 5 A supply: amber under this, red under VOLT_LOW_V (servos 12 V C018: 6..12.6 V)
VOLT_LOW_V = 10.0
BUS_A_MARK = 5.0           # Waveshare bus board continuous rating (Robonine load test, docs/stronger-arm.md)
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIELDS = ("pos", "goal", "sag", "vel_dps", "load_pct", "cur_a", "volt_v", "temp_c")   # the numeric ones summarised per pick


def _word(b, i): return b[i] | (b[i + 1] << 8)
def _signed(v, bit):   # sign-magnitude exactly as LeRobot's decode_sign_magnitude: bits above the sign bit are dropped
    m = v & ((1 << bit) - 1); return -m if (v >> bit) & 1 else m


def decode_block(b):
    """The 31 bytes of addresses 40..70 for one servo -> a dict of numbers in physical units (raw counts kept as *_raw)."""
    o = BLOCK[0]
    a = lambda addr: addr - o
    status = b[a(65)]
    cur = _word(b, a(69)) & 0x7FFF   # bit 15 would be a direction flag on some firmware: magnitude only
    return {"on": b[a(40)], "acc": b[a(41)], "goal_raw": _signed(_word(b, a(42)), 15), "tq_lim": _word(b, a(48)),
            "pos_raw": _signed(_word(b, a(56)), 15), "vel_raw": _signed(_word(b, a(58)), 15), "load_raw": _signed(_word(b, a(60)), 10),
            "volt_v": b[a(62)] / 10.0, "temp_c": b[a(63)], "status": status, "moving": 1 if b[a(66)] else 0,
            "cur_raw": cur, "cur_a": round(cur * MA_PER_COUNT / 1000.0, 3),
            "vel_dps": round(_signed(_word(b, a(58)), 15) * 360.0 / 4096, 1), "load_pct": round(_signed(_word(b, a(60)), 10) / 10.0, 1),
            "err": [n for bit, n in STATUS_BITS.items() if status & (1 << bit)]}


def encode_block(on=1, goal=2048, tq_lim=1000, pos=2048, vel=0, load=0, volt=11.9, temp=30, status=0, moving=0, cur=0, acc=0):
    """The inverse of decode_block (counts in, 31 bytes out): the fake servos in the tests and the demo feed use it."""
    b = bytearray(BLOCK[1]); o = BLOCK[0]
    def put(addr, v, n=1):
        for k in range(n): b[addr - o + k] = (v >> (8 * k)) & 0xFF
    def sm(v, bit): return (abs(int(v)) | (1 << bit)) if v < 0 else int(v)   # sign-magnitude
    put(40, on); put(41, acc); put(42, sm(goal, 15), 2); put(48, tq_lim, 2); put(56, sm(pos, 15), 2); put(58, sm(vel, 15), 2)
    put(60, sm(load, 10), 2); put(62, int(round(volt * 10))); put(63, int(temp)); put(65, status); put(66, moving); put(69, int(cur), 2)
    return bytes(b)


def read_blocks(bus, ids, addr=BLOCK[0], length=BLOCK[1]):
    """{id: bytes} for one sync-read packet over `ids`, or None when the bus did not answer. Read-only: a sync READ packet. Uses the
    SDK's group reader directly because LeRobot's sync_read decodes one register at a time (three round trips for position, load, current)."""
    bus._setup_sync_reader(list(ids), addr, length)
    comm = bus.sync_reader.txRxPacket()
    if not bus._is_comm_success(comm): return None
    out = {}
    for i in ids:
        avail = getattr(bus.sync_reader, "isAvailable", None)
        if avail is not None and not avail(i, addr, length): continue   # this servo did not answer in the packet
        data = getattr(bus.sync_reader, "data_dict", {}).get(i)
        if data is None or len(data) < length:   # fakes without a data_dict: one getData per byte
            data = [bus.sync_reader.getData(i, addr + k, 1) & 0xFF for k in range(length)]
        out[i] = bytes(data[:length])
    return out


def _norm(bus, name, raw):
    """LeRobot's own normalisation of a raw position for one motor (degrees, or 0..100 for the gripper); None if the bus has no calibration."""
    try: return float(bus._normalize({bus.motors[name].id: raw})[bus.motors[name].id])
    except Exception: return None


def read_sample(bus, now=None, names=None):
    """One telemetry sample for every motor on `bus` (or `names`), or None if the bus did not answer. Never raises on a bad read."""
    t0 = time.perf_counter()
    names = list(names or bus.motors); ids = [bus.motors[n].id for n in names]
    try: blocks = read_blocks(bus, ids)
    except Exception: return None
    if not blocks: return None
    j = {}
    for n in names:
        raw = blocks.get(bus.motors[n].id)
        if raw is None: continue
        d = decode_block(raw)
        pos, goal = _norm(bus, n, d["pos_raw"]), _norm(bus, n, d["goal_raw"])
        d["pos"], d["goal"] = (None if pos is None else round(pos, 2)), (None if goal is None else round(goal, 2))
        d["sag"] = None if pos is None or goal is None else round(goal - pos, 2)   # commanded minus actual, in the joint's own units
        for k in ("goal_raw", "pos_raw", "vel_raw", "load_raw", "cur_raw", "acc"): d.pop(k)
        j[n] = d
    if not j: return None
    return {"t": time.time() if now is None else now, "j": j, **summarise_now(j), "dt_ms": round((time.perf_counter() - t0) * 1000, 2)}


def summarise_now(j):
    """Bus-level numbers for one sample: total current, lowest voltage, hottest servo, and every error flag as 'joint:flag'."""
    return {"total_a": round(sum(d["cur_a"] for d in j.values()), 3), "volt_min": min(d["volt_v"] for d in j.values()),
            "temp_max": max(d["temp_c"] for d in j.values()), "flags": [f"{n}:{e}" for n, d in j.items() for e in d["err"]]}


def read_limits(bus, names=None):
    """Once per connect: {joint: {"temp_cut_c", "volt_max_v", "volt_min_v", "torque_max", "protect_a", ...}} from addresses 13..38. {} on failure."""
    names = list(names or bus.motors); ids = [bus.motors[n].id for n in names]
    try: blocks = read_blocks(bus, ids, *LIMITS_BLOCK)
    except Exception: return {}
    o, out = LIMITS_BLOCK[0], {}
    for n in names:
        b = blocks and blocks.get(bus.motors[n].id)
        if not b: continue
        out[n] = {"temp_cut_c": b[13 - o], "volt_max_v": b[14 - o] / 10.0, "volt_min_v": b[15 - o] / 10.0, "torque_max": _word(b, 16 - o),
                  "led_alarm": b[20 - o], "protect_a": round(_word(b, 28 - o) * MA_PER_COUNT / 1000.0, 2), "protect_counts": _word(b, 28 - o),
                  "overload_torque": b[36 - o], "over_current_s": b[38 - o]}
    return out


# ---------------------------------------------------------------- flags the dashboard raises (shared so the tests and the page agree)
def level_temp(c, cut=TEMP_CUT_C):
    if c is None: return 0
    return 2 if c >= min(TEMP_HOT_C, cut - 5) else 1 if c >= min(TEMP_WARN_C, cut - 10) else 0

def level_volt(v): return 0 if v is None else 2 if v < VOLT_LOW_V else 1 if v < VOLT_WARN_V else 0
def level_bus(a): return 0 if a is None else 2 if a >= BUS_A_MARK else 1 if a >= 0.8 * BUS_A_MARK else 0


def sag_flag(history, window=20, factor=1.5, floor_deg=1.0):
    """True when a joint's recent sag is above its usual level. `history` = signed sag values (oldest first) taken while the servo was
    holding (moving == 0, torque on). Usual level = median |sag| over everything but the last `window`; flag when the mean |sag| of the last
    `window` exceeds factor x usual AND usual + floor_deg. Needs 3 x window samples; fewer never flags."""
    h = [abs(v) for v in history if v is not None]
    if len(h) < 3 * window: return False
    base = sorted(h[:-window]); usual = base[len(base) // 2]; recent = sum(h[-window:]) / window
    return recent > factor * usual and recent > usual + floor_deg


# ---------------------------------------------------------------- per-run summaries (one pick, one autoloop cycle) for MLflow
class Summary:
    """Accumulates samples and reports the extremes: add(sample) per read, .out() -> a plain dict, merge() joins several, metrics() flattens."""
    def __init__(self): self.n = 0; self.d = {}; self.total_a_max = 0.0; self.volt_min = None; self.flags = {}; self.dt_ms_max = 0.0; self.temp0 = {}

    def add(self, s):
        if not s: return
        self.n += 1
        self.total_a_max = max(self.total_a_max, s.get("total_a") or 0.0); v = s.get("volt_min")
        if v is not None: self.volt_min = v if self.volt_min is None else min(self.volt_min, v)
        self.dt_ms_max = max(self.dt_ms_max, s.get("dt_ms") or 0.0)
        for f in s.get("flags") or []: self.flags[f] = self.flags.get(f, 0) + 1
        for n, x in s["j"].items():
            e = self.d.setdefault(n, {"temp_max_c": 0, "temp_last_c": 0, "load_abs_max_pct": 0.0, "cur_max_a": 0.0, "sag_abs_max": 0.0, "sag_hold_abs_max": 0.0, "volt_min_v": None})
            self.temp0.setdefault(n, x["temp_c"])
            e["temp_max_c"] = max(e["temp_max_c"], x["temp_c"]); e["temp_last_c"] = x["temp_c"]
            e["load_abs_max_pct"] = max(e["load_abs_max_pct"], abs(x["load_pct"])); e["cur_max_a"] = max(e["cur_max_a"], x["cur_a"])
            if x.get("sag") is not None:
                e["sag_abs_max"] = max(e["sag_abs_max"], abs(x["sag"]))
                if not x["moving"] and x["on"]: e["sag_hold_abs_max"] = max(e["sag_hold_abs_max"], abs(x["sag"]))   # holding, not tracking a move: the gravity sag
            e["volt_min_v"] = x["volt_v"] if e["volt_min_v"] is None else min(e["volt_min_v"], x["volt_v"])

    def out(self):
        return {"n": self.n, "total_a_max": round(self.total_a_max, 3), "volt_min": self.volt_min, "flags": dict(self.flags), "dt_ms_max": round(self.dt_ms_max, 2),
                "j": {n: {**e, "temp_rise_c": e["temp_last_c"] - self.temp0[n]} for n, e in self.d.items()}}


def merge(summaries):
    """Join several Summary.out() dicts (the waypoints of a pick, the picks of a cycle) into one with the same shape."""
    out = {"n": 0, "total_a_max": 0.0, "volt_min": None, "flags": {}, "dt_ms_max": 0.0, "j": {}}
    for s in summaries:
        if not s or not s.get("n"): continue
        out["n"] += s["n"]; out["total_a_max"] = max(out["total_a_max"], s["total_a_max"]); out["dt_ms_max"] = max(out["dt_ms_max"], s.get("dt_ms_max") or 0.0)
        if s["volt_min"] is not None: out["volt_min"] = s["volt_min"] if out["volt_min"] is None else min(out["volt_min"], s["volt_min"])
        for f, c in (s.get("flags") or {}).items(): out["flags"][f] = out["flags"].get(f, 0) + c
        for n, e in s["j"].items():
            m = out["j"].get(n)
            if m is None: out["j"][n] = dict(e); continue
            for k in ("temp_max_c", "load_abs_max_pct", "cur_max_a", "sag_abs_max", "sag_hold_abs_max"): m[k] = max(m[k], e[k])
            if e["volt_min_v"] is not None: m["volt_min_v"] = e["volt_min_v"] if m["volt_min_v"] is None else min(m["volt_min_v"], e["volt_min_v"])
            m["temp_last_c"] = e["temp_last_c"]; m["temp_rise_c"] = m.get("temp_rise_c", 0) + e.get("temp_rise_c", 0)
    return out


def metrics(s, prefix="tele"):
    """A merged/out() summary -> flat {name: number} for LabTrack.metrics (None when no sample was taken: nothing is logged)."""
    if not s or not s.get("n"): return {}
    m = {f"{prefix}_samples": s["n"], f"{prefix}_total_a_max": s["total_a_max"], f"{prefix}_volt_min_v": s["volt_min"], f"{prefix}_error_flags": sum((s.get("flags") or {}).values())}
    for n, e in s["j"].items():
        m.update({f"{prefix}_temp_max_c_{n}": e["temp_max_c"], f"{prefix}_temp_rise_c_{n}": e.get("temp_rise_c"), f"{prefix}_load_abs_max_pct_{n}": e["load_abs_max_pct"],
                  f"{prefix}_cur_max_a_{n}": e["cur_max_a"], f"{prefix}_sag_abs_max_{n}": e["sag_abs_max"], f"{prefix}_sag_hold_max_{n}": e["sag_hold_abs_max"]})
    return {k: v for k, v in m.items() if isinstance(v, (int, float)) and math.isfinite(v)}
