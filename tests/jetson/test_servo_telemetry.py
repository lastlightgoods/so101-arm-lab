"""jetson/servo_telemetry.py: the register list, the block decode, one read against the pretend bus, flags and per-run summaries. No hardware."""
import pytest
from telehelpers import fake_bus, fake_bus_module, servo_telemetry

st = servo_telemetry()


# ---------- the register list matches LeRobot's own control table ----------
def test_register_addresses_match_lerobot_tables():
    tables = pytest.importorskip("lerobot.motors.feetech.tables")
    ctl = tables.STS_SMS_SERIES_CONTROL_TABLE
    seen = {}
    for name, addr, size, kind, meaning, polled, status in st.REGISTERS:
        assert name in ctl, name
        assert ctl[name] == (addr, size), f"{name}: ours {(addr, size)}, lerobot {ctl[name]}"
        seen[name] = addr
    assert len(seen) == len(st.REGISTERS), "a register is listed twice"
    # sts3215 and sts3250 share the one table in LeRobot 0.6.1, which is why one decoder serves both
    assert tables.MODEL_CONTROL_TABLE["sts3215"] is tables.MODEL_CONTROL_TABLE["sts3250"]
    assert tables.MODEL_NUMBER_TABLE["sts3215"] == st.MODELS["sts3215"]["number"] and tables.MODEL_NUMBER_TABLE["sts3250"] == st.MODELS["sts3250"]["number"]


def test_block_covers_every_polled_register_and_nothing_else_is_polled():
    lo, n = st.BLOCK
    for name, addr, size, kind, meaning, polled, status in st.REGISTERS:
        inside = lo <= addr and addr + size <= lo + n
        assert (not polled) or inside, f"{name} is marked polled but lies outside the block"
        if inside and kind == "live": assert polled, f"{name} lies in the block and should be marked polled"


def test_sign_magnitude_matches_lerobot_decode():
    tables = pytest.importorskip("lerobot.motors.feetech.tables")
    enc = tables.STS_SMS_SERIES_ENCODINGS_TABLE
    assert enc["Present_Velocity"] == 15 and enc["Present_Load"] == 10 and enc["Present_Position"] == 15
    from lerobot.motors.encoding_utils import decode_sign_magnitude
    for raw in (0, 5, 1000, (1 << 10) | 250, (1 << 15) | 300):
        assert st._signed(raw, 10) == decode_sign_magnitude(raw, 10)
        assert st._signed(raw, 15) == decode_sign_magnitude(raw, 15)


# ---------- decode / encode ----------
def test_decode_block_units():
    b = st.encode_block(on=1, goal=2100, tq_lim=800, pos=2048, vel=-409.6, load=-250, volt=11.9, temp=41, status=0b100100, moving=1, cur=100)
    d = st.decode_block(b)
    assert d["on"] == 1 and d["tq_lim"] == 800 and d["goal_raw"] == 2100 and d["pos_raw"] == 2048
    assert d["volt_v"] == 11.9 and d["temp_c"] == 41 and d["moving"] == 1
    assert d["load_pct"] == -25.0                     # 0.1 % units, sign bit 10
    assert d["vel_dps"] == pytest.approx(-36.0, abs=0.2)   # 409.6 steps/s = 36 deg/s
    assert d["cur_a"] == pytest.approx(0.65)          # 100 counts x 6.5 mA
    assert d["err"] == ["temp", "overload"]


def test_current_ignores_the_direction_bit():
    b = bytearray(st.encode_block(cur=100)); b[69 - 40 + 1] |= 0x80
    assert st.decode_block(bytes(b))["cur_raw"] == 100


# ---------- one read against the pretend bus ----------
def test_read_sample_shape_and_values():
    bus = fake_bus(warm=20.0)
    s = st.read_sample(bus, now=123.0)
    assert s["t"] == 123.0 and list(s["j"]) == list(st.JOINTS)
    d = s["j"]["shoulder_lift"]
    assert set(d) >= {"on", "pos", "goal", "sag", "vel_dps", "load_pct", "cur_a", "volt_v", "temp_c", "status", "moving", "err", "tq_lim"}
    assert d["sag"] == pytest.approx(d["goal"] - d["pos"], abs=0.02)
    assert s["total_a"] == pytest.approx(sum(x["cur_a"] for x in s["j"].values()), abs=0.01)
    assert s["volt_min"] == min(x["volt_v"] for x in s["j"].values()) and s["temp_max"] == max(x["temp_c"] for x in s["j"].values())
    assert s["dt_ms"] >= 0


def test_read_sample_is_one_packet_and_writes_nothing():
    bus = fake_bus()
    calls = []
    real = bus.sync_reader.txRxPacket
    bus.sync_reader.txRxPacket = lambda: (calls.append(1), real())[1]
    st.read_sample(bus)
    assert len(calls) == 1, "all six servos and every register in one sync-read packet"
    assert not [a for a in dir(bus) if "write" in a.lower()], "the pretend bus has no write path, so any write would have raised"


def test_dead_bus_gives_none_not_an_exception():
    bus = fake_bus(); bus.offline = True
    assert st.read_sample(bus) is None
    bus.offline = False; bus.silent = set(range(1, 7))
    assert st.read_sample(bus) is None


def test_one_silent_servo_is_left_out_of_the_sample():
    bus = fake_bus(); bus.silent = {3}
    s = st.read_sample(bus)
    assert "elbow_flex" not in s["j"] and len(s["j"]) == 5


def test_read_limits_once_gives_the_cut_and_protection():
    lim = st.read_limits(fake_bus())
    assert lim["shoulder_lift"]["temp_cut_c"] == 70 and lim["shoulder_lift"]["volt_max_v"] == 14.0
    assert lim["gripper"]["protect_counts"] == 250 and lim["gripper"]["protect_a"] == pytest.approx(1.62)


def test_positions_agree_with_lerobots_own_normalisation():
    """The pretend bus normalises like lerobot would; the sag sign convention is goal minus present in the same units as the dashboard's joints."""
    bus = fake_bus(warm=5.0)
    s = st.read_sample(bus)
    for n, d in s["j"].items():
        assert d["pos"] == pytest.approx(bus.pos[n], abs=0.1)


# ---------- flags ----------
def test_temperature_levels_follow_the_cut():
    assert [st.level_temp(c) for c in (30, 59, 60, 64, 65, 70)] == [0, 0, 1, 1, 2, 2]
    assert (st.level_temp(52, cut=60), st.level_temp(56, cut=60), st.level_temp(None)) == (1, 2, 0)   # a lower cut read from the servo pulls the thresholds down with it


def test_voltage_and_bus_levels():
    assert [st.level_volt(v) for v in (12.2, 11.0, 10.9, 10.0, 9.9)] == [0, 0, 1, 1, 2]
    assert [st.level_bus(a) for a in (1.0, 3.9, 4.0, 4.9, 5.0, 6.2)] == [0, 0, 1, 1, 2, 2]


def test_sag_flag_needs_history_and_a_real_rise():
    usual = [1.0, 1.1, 0.9] * 20
    assert st.sag_flag(usual) is False
    assert st.sag_flag(usual + [3.5] * 20) is True
    assert st.sag_flag(usual + [1.3] * 20) is False          # above usual but inside the margin
    assert st.sag_flag([3.5] * 10) is False                  # not enough history to know what usual is
    assert st.sag_flag([None] * 80) is False


# ---------- summaries for MLflow ----------
def _samples(bus, secs, hot=None):
    out = []
    if hot: bus.extra_heat.update(hot)
    for _ in range(int(secs * 4)):
        bus.advance(0.25); out.append(st.read_sample(bus, now=bus.t))
    return out


def test_summary_extremes_and_rise():
    bus = fake_bus(warm=1.0); s = st.Summary()
    for x in _samples(bus, 60, hot={"shoulder_lift": 0.2}): s.add(x)
    o = s.out()
    assert o["n"] == 240 and o["total_a_max"] > 0.5 and 9.0 < o["volt_min"] < 12.4
    lift = o["j"]["shoulder_lift"]
    assert lift["temp_max_c"] >= lift["temp_last_c"] > 30 and lift["temp_rise_c"] > 5
    assert o["j"]["wrist_roll"]["temp_rise_c"] <= 2
    assert lift["load_abs_max_pct"] > 10 and lift["cur_max_a"] > 0.3 and lift["sag_abs_max"] > 0.2


def test_summary_separates_holding_sag_from_tracking_lag():
    s = st.Summary()
    base = {"on": 1, "temp_c": 30, "load_pct": 5.0, "cur_a": 0.3, "volt_v": 12.0, "err": [], "status": 0, "pos": 0.0, "goal": 4.0}
    s.add({"t": 0, "j": {"shoulder_lift": {**base, "sag": 4.0, "moving": 1}}, "total_a": 0.3, "volt_min": 12.0, "flags": []})
    s.add({"t": 1, "j": {"shoulder_lift": {**base, "sag": 1.5, "moving": 0}}, "total_a": 0.3, "volt_min": 12.0, "flags": []})
    e = s.out()["j"]["shoulder_lift"]
    assert e["sag_abs_max"] == 4.0 and e["sag_hold_abs_max"] == 1.5


def test_merge_joins_waypoint_summaries():
    bus = fake_bus(warm=1.0); a, b = st.Summary(), st.Summary()
    for x in _samples(bus, 10): a.add(x)
    for x in _samples(bus, 10, hot={"gripper": 0.3}): b.add(x)
    m = st.merge([a.out(), None, b.out()])
    assert m["n"] == a.n + b.n
    assert m["total_a_max"] == max(a.total_a_max, b.total_a_max)
    assert m["j"]["gripper"]["temp_max_c"] == max(a.out()["j"]["gripper"]["temp_max_c"], b.out()["j"]["gripper"]["temp_max_c"])
    assert st.merge([None, {"n": 0}])["n"] == 0


def test_metrics_are_flat_finite_numbers_named_per_joint():
    bus = fake_bus(warm=1.0); s = st.Summary()
    for x in _samples(bus, 20): s.add(x)
    m = st.metrics(s.out())
    assert m["tele_samples"] == 80 and "tele_total_a_max" in m and "tele_volt_min_v" in m
    for j in st.JOINTS:
        for k in ("temp_max_c", "temp_rise_c", "load_abs_max_pct", "cur_max_a", "sag_abs_max", "sag_hold_max"): assert f"tele_{k}_{j}" in m, (k, j)
    assert all(isinstance(v, (int, float)) for v in m.values())
    assert st.metrics(None) == {} and st.metrics({"n": 0}) == {}
    assert all(k.startswith("x_") for k in st.metrics(s.out(), "x"))


def test_pretend_bus_reproduces_a_hot_servo_and_a_brownout():
    fb = fake_bus_module(); bus = fb.FakeServoBus(seed=2)
    bus.extra_heat["shoulder_lift"] = 0.3
    flags = []
    for s in fb.demo_stream(bus, 600):
        flags += s["flags"]
    assert any(f == "shoulder_lift:temp" for f in flags)
    bus2 = fb.FakeServoBus(seed=2); bus2.extra_drop_v = 2.5; bus2.advance(5)
    assert st.level_volt(st.read_sample(bus2)["volt_min"]) == 2
