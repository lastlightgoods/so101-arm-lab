"""jetson/verify_telemetry.py against the pretend bus: the register sweep, the verdicts, the meter checks, and that it refuses to run over a lerobot job."""
import subprocess
import sys

import pytest
from telehelpers import ROOT, fake_bus, load, servo_telemetry

st = servo_telemetry()


@pytest.fixture(scope="module")
def vt():
    return load(f"{ROOT}/jetson/verify_telemetry.py", "verify_telemetry_for_tests")


def sweep(vt, bus, seconds=2, meter=(None, None), hand=None):
    adv = lambda s: bus.advance(s)
    singles = vt.single_reads(bus); samples = vt.idle_samples(bus, seconds, sleep=adv)
    return vt.evaluate(singles, samples, None, *meter, hand=hand), singles


def test_every_register_reads_on_a_healthy_bus(vt):
    bus = fake_bus(warm=5.0)
    (rows, checks, ok), singles = sweep(vt, bus)
    assert ok and len(rows) == len(st.REGISTERS) and all(r["verdict"] == "reads back" and r["reads_on"] == "6/6 servos" for r in rows)
    assert all(g for _, g, _ in checks)
    assert singles["shoulder_lift"]["Model_Number"]["value"] == 777 and singles["gripper"]["Present_Voltage"]["value"] > 100


def test_a_silent_servo_fails_the_sweep_and_names_the_registers(vt):
    bus = fake_bus(warm=5.0); bus.silent = {2}
    (rows, checks, ok), _ = sweep(vt, bus)
    assert not ok
    r = next(r for r in rows if r["register"] == "Present_Temperature")
    assert r["verdict"] == "NO ANSWER" and r["reads_on"] == "5/6 servos"


def test_a_register_that_is_not_polled_does_not_fail_the_run_but_is_reported(vt, monkeypatch):
    bus = fake_bus(warm=5.0)
    orig = bus._read
    monkeypatch.setattr(bus, "_read", lambda addr, n, i, **k: (0, -6, 0) if addr == 71 else orig(addr, n, i, **k))   # Goal_Position_2 (not polled) does not answer
    (rows, checks, ok), _ = sweep(vt, bus)
    assert ok and next(r for r in rows if r["register"] == "Goal_Position_2")["verdict"] == "NO ANSWER"


def test_meter_checks_pass_and_fail(vt):
    bus = fake_bus(warm=5.0)
    s = st.read_sample(bus)
    volts = s["volt_min"]; amps = s["total_a"]
    (_, checks, ok), _ = sweep(vt, fake_bus(warm=5.0), meter=(volts, amps))
    assert ok and any(n == "supply voltage vs meter" and g for n, g, _ in checks) and any(n == "current scale vs meter" and g for n, g, _ in checks)
    (_, checks, ok), _ = sweep(vt, fake_bus(warm=5.0), meter=(volts + 1.5, amps * 2.2))
    bad = {n: d for n, g, d in checks if not g}
    assert not ok and "supply voltage vs meter" in bad and "suggested scale" in bad["current scale vs meter"]


def test_status_bits_at_rest_must_be_clear(vt):
    bus = fake_bus(warm=5.0); step = bus._step
    bus._step = lambda dt: (step(dt), bus.status.__setitem__("gripper", 4))   # the pretend servos recompute Status every step; pin a temperature bit on the gripper
    (_, checks, ok), _ = sweep(vt, bus)
    assert not ok and any(n == "status bits at rest" and not g for n, g, _ in checks)


def test_hand_move_passes_for_joints_that_responded_and_skips_the_rest(vt):
    bus = fake_bus(warm=2.0); bus.script = False; bus.torque_on = False
    steps = {"n": 0}
    def sleep(s):
        steps["n"] += 1; bus.pos["elbow_flex"] += 1.2 * (1 if steps["n"] % 20 < 10 else -1); bus.goal["elbow_flex"] = bus.pos["elbow_flex"]; bus.vel["elbow_flex"] = 60 if steps["n"] % 20 < 10 else -60
    hm = vt.hand_move(bus, 3, sleep=sleep)
    assert hm["elbow_flex"]["responds"] and not hm["gripper"]["responds"]
    (rows, checks, ok), _ = sweep(vt, fake_bus(warm=5.0), hand=hm)
    assert ok
    got = {n: g for n, g, _ in checks if n.startswith("hand move")}
    assert got["hand move: elbow_flex"] is True and got["hand move: gripper"] is None


def test_timing_reports_block_and_position_reads(vt):
    bus = fake_bus(warm=1.0); bus.sync_read = lambda name, normalize=True: {}
    t = vt.time_block(bus, 10)
    assert t["n"] == 10 and set(t["block_ms"]) == {"mean", "p95", "max"}


def test_refuses_while_a_lerobot_job_has_the_arm(vt, monkeypatch, tmp_path):
    monkeypatch.setattr(vt, "BUSY", str(tmp_path / ".arms_busy"))
    class R: returncode = 0
    monkeypatch.setattr(vt.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(SystemExit, match="lerobot job holds the arm"):
        vt.main(["--seconds", "1"])
    assert not (tmp_path / ".arms_busy").exists()


def test_the_script_only_calls_read_paths(vt):
    """verify_telemetry.py must never write to a servo: no write, sync_write, enable/disable_torque, configure or calibration call anywhere in it."""
    src = open(f"{ROOT}/jetson/verify_telemetry.py").read() + open(f"{ROOT}/jetson/servo_telemetry.py").read()
    for banned in ("._write(", ".write(\"", "sync_write", "enable_torque", "disable_torque(", "write_calibration", "configure(", "torque_disabled", "setTorque"):
        assert banned not in src.replace("disconnect(disable_torque=False)", ""), banned


def test_demo_refuses_the_live_dashboards_ports():
    for port in ("8765", "8766", "9333"):
        r = subprocess.run([sys.executable, f"{ROOT}/tools/telemetry_demo.py", port], capture_output=True, text=True, timeout=60)
        assert r.returncode != 0 and "live dashboard" in (r.stdout + r.stderr)
