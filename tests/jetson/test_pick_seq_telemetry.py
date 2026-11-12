"""jetson/pick_seq.py with servo telemetry: a `tele` summary on every waypoint row, the /dev/shm file for the dashboard, and no change to move timing. Faked arm and clock."""
import json
import signal
import sys

import pytest
from telehelpers import ROOT, fake_bus, load, servo_telemetry

st = servo_telemetry()
J = list(st.JOINTS)
PLAN = {"waypoints": [
    {"name": "hover", "joints": {"shoulder_pan": 18, "shoulder_lift": -35, "elbow_flex": 55, "wrist_flex": 60, "wrist_roll": 0}, "gripper": 40, "seconds": 1.5, "settle": 0.3},
    {"name": "grasp", "joints": {"shoulder_pan": 18, "shoulder_lift": -12, "elbow_flex": 40, "wrist_flex": 55, "wrist_roll": 0}, "gripper": 40, "seconds": 1.5, "settle": 0.3},
]}


class Clock:
    def __init__(self): self.now = 1000.0
    def time(self): return self.now
    def sleep(self, s): self.now += s


class FakeRobot:
    """What pick_seq needs: .bus (the pretend servo bus), send_action, get_observation, disconnect. The bus follows the commanded goal."""
    def __init__(self, bus, clock):
        self.bus, self.clock, self.actions, self.disconnected = bus, clock, 0, False
        bus.script = False
        for j in J: bus.goal[j] = bus.pos[j]
    def send_action(self, a):
        self.actions += 1
        for k, v in a.items(): self.bus.goal[k.split(".")[0]] = v
        self.bus.advance(0.0)   # no time passes in send_action; the sleeps drive the bus clock
    def get_observation(self): return {f"{j}.pos": self.bus.pos[j] for j in J}
    def disconnect(self): self.disconnected = True


@pytest.fixture
def seq(tmp_path, monkeypatch):
    monkeypatch.setattr(signal, "signal", lambda *a, **k: None)   # the module installs a SIGTERM handler at import
    m = load(f"{ROOT}/jetson/pick_seq.py", "pick_seq_for_tests")
    clock = Clock(); bus = fake_bus(warm=2.0)
    monkeypatch.setattr(m, "time", clock)
    monkeypatch.setattr(m, "BUSY", str(tmp_path / ".arms_busy")); monkeypatch.setattr(m, "SHM", str(tmp_path))
    monkeypatch.setattr(m, "lerobot_running", lambda: False)
    robot = FakeRobot(bus, clock)
    monkeypatch.setattr(m, "connect", lambda: robot)
    # sleeping advances the arm's own clock too, so the pretend servos move while pick_seq "waits"
    real_sleep = clock.sleep
    clock.sleep = lambda s: (real_sleep(s), bus.advance(s, dt=0.05))[0]
    plan = tmp_path / "plan.json"; plan.write_text(json.dumps(PLAN))

    class H:
        mod, rob, bus_, clk, tmp = m, robot, bus, clock, tmp_path
        def run(self, *extra, capsys=None):
            monkeypatch.setattr(sys, "argv", ["pick_seq.py", "--plan", str(plan), "--correct", "0", *extra])
            m.main()
    return H()


def lines(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]


def test_every_waypoint_row_carries_a_tele_summary(seq, capsys):
    seq.run()
    out = lines(capsys)
    rows = [d for d in out if "wp" in d]
    assert [r["wp"] for r in rows] == ["hover", "grasp"]
    for r in rows:
        t = r["tele"]
        assert t["n"] >= 3 and set(t["j"]) == set(J) and t["j"]["shoulder_lift"]["cur_max_a"] > 0 and t["volt_min"] < 12.4
        assert r["max_err_deg"] >= 0 and "reached" in r   # the rows keep every field they had
    done = out[-1]
    assert done["done"] is True and done["tele"]["n"] == sum(r["tele"]["n"] for r in rows)


def test_dashboard_file_is_written_while_moving(seq, capsys):
    seq.run()
    f = json.loads((seq.tmp / "so101_follower_tele.json").read_text())
    assert f["role"] == "follower" and f["src"] == "pick_seq" and set(f["j"]) == set(J) and f["j"]["shoulder_lift"]["pos"] is not None


def test_first_sample_carries_the_limits_once(seq, capsys, monkeypatch):
    seen = []
    orig = seq.mod.Tele.publish
    monkeypatch.setattr(seq.mod.Tele, "publish", lambda self, s: (seen.append("limits" in s), orig(self, s))[1])
    seq.run()
    assert seen[0] is True and not any(seen[1:])


def test_telemetry_does_not_change_how_long_a_move_takes(seq, capsys, monkeypatch):
    """Each health read costs bus time (faked here as 3 ms of clock); move() takes it out of its own sleep, so the sequence lasts as long as without telemetry."""
    real = st.read_sample
    def costly(bus, now=None, names=None):
        seq.clk.now += 0.003; return real(bus, now, names)
    monkeypatch.setattr(st, "read_sample", costly)
    t0 = seq.clk.now; seq.run(); with_tele = seq.clk.now - t0; capsys.readouterr()
    monkeypatch.setattr(seq.mod, "TELE_S", 0.0)
    t1 = seq.clk.now; seq.run(); without = seq.clk.now - t1
    # the forced read after each settle is the only unmasked cost: 2 waypoints x 3 ms
    assert with_tele - without == pytest.approx(0.006, abs=0.0015), (with_tele, without)


def test_off_switch_and_missing_module_leave_rows_as_before(seq, capsys, monkeypatch):
    monkeypatch.setattr(seq.mod, "TELE_S", 0.0)
    seq.run()
    out = lines(capsys)
    assert all("tele" not in d for d in out) and not (seq.tmp / "so101_follower_tele.json").exists()
    monkeypatch.setattr(seq.mod, "TELE_S", 0.25); monkeypatch.setattr(seq.mod, "servo_telemetry", None)
    seq.run()
    assert all("tele" not in d for d in lines(capsys))


def test_a_failing_health_read_never_stops_the_sequence(seq, capsys, monkeypatch):
    monkeypatch.setattr(st, "read_sample", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bus glitch")))
    seq.run()
    out = lines(capsys)
    assert [d["wp"] for d in out if "wp" in d] == ["hover", "grasp"] and all("tele" not in d for d in out) and seq.rob.disconnected


def test_arms_busy_flag_is_held_and_released(seq, capsys):
    seq.run()
    assert not (seq.tmp / ".arms_busy").exists() and seq.rob.disconnected


def test_rows_carry_the_stall_seconds_and_a_stall_run_survives_the_waypoint_boundary(seq, capsys):
    """Overnight guard: tools/guardrails.servo_health_trip reads stall_s_max; a stall that spans two waypoints must count as one."""
    seq.run(); rows = [d for d in lines(capsys) if "wp" in d]
    assert all("stall_s_max" in e for r in rows for e in r["tele"]["j"].values()) and all(e["stall_s_max"] < 3.0 for r in rows for e in r["tele"]["j"].values()), "the pretend arm's sagging lift is over the stall line for under a second, far under the 3 s trip"
    tele = seq.mod.Tele(seq.rob); tele.cur.run0 = {"shoulder_lift": 990.0}; tele.sample(force=True); first = tele.cur.run0
    assert tele.take() is not None and tele.cur.run0 is first, "the new summary keeps the run starts"
