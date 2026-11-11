"""arm_state.py with servo telemetry: idle it rides on the follower's line every 3rd tick; busy it forwards the plugin's / pick_seq's /dev/shm file. All faked."""
import json

import pytest
from jhelpers import FakeBus
from telehelpers import fake_bus, servo_telemetry

st = servo_telemetry()


class TeleBus(FakeBus):
    """arm_state's FakeBus (position reads) plus the pretend servo bus behind the block-read calls servo_telemetry makes."""
    servo = None

    def __init__(self, port, motors, calibration=None):
        super().__init__(port, motors, calibration)
        self.sync_reader = TeleBus.servo.sync_reader
        self._setup_sync_reader = TeleBus.servo._setup_sync_reader
        self._is_comm_success = TeleBus.servo._is_comm_success
        self._normalize = TeleBus.servo._normalize
        self._read = TeleBus.servo._read


@pytest.fixture
def tele_arm(arm, monkeypatch):
    TeleBus.servo = fake_bus(warm=10.0)
    monkeypatch.setattr(arm.mod, "FeetechMotorsBus", TeleBus)
    monkeypatch.setattr(arm.mod, "servo_telemetry", st)
    monkeypatch.setattr(arm.mod, "TELE_EVERY", 3)
    monkeypatch.setattr(arm.mod, "tele_sent", {})       # module state: reset so a test's fake clock (same start every time) does not look like a repeat
    monkeypatch.setattr(arm.mod, "limits_sent", set())
    return arm


def test_idle_follower_line_carries_telemetry_every_third_tick(tele_arm):
    tele_arm.run(sleeps=7)
    fol = tele_arm.by_role("follower")
    assert [("tele" in d) for d in fol] == [True, False, False, True, False, False, True]
    t = fol[0]["tele"]
    assert set(t["j"]) == set(st.JOINTS) and t["total_a"] > 0 and "limits" in t and t["limits"]["shoulder_lift"]["temp_cut_c"] == 70
    assert "limits" not in fol[3]["tele"], "the limits are read once per connect, not every sample"
    assert fol[0]["src"] == "idle-poll" and fol[0]["pos"] == {"shoulder_pan": 1.0, "gripper": 2.0}   # the position stream is unchanged


def test_leader_gets_no_telemetry_by_default(tele_arm):
    tele_arm.run(sleeps=4)
    assert not any("tele" in d for d in tele_arm.by_role("leader"))


def test_telemetry_reads_cost_the_position_stream_nothing_when_they_fail(tele_arm):
    TeleBus.servo.offline = True      # block reads time out
    tele_arm.run(sleeps=4)
    fol = tele_arm.by_role("follower")
    assert len(fol) == 4 and all(d["src"] == "idle-poll" and d["pos"] for d in fol) and not any("tele" in d for d in fol)


def test_telemetry_exception_is_swallowed(tele_arm, monkeypatch):
    monkeypatch.setattr(st, "read_sample", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    tele_arm.run(sleeps=3)
    assert all(d["src"] == "idle-poll" for d in tele_arm.by_role("follower"))


def test_every_zero_switches_it_off(tele_arm, monkeypatch):
    monkeypatch.setattr(tele_arm.mod, "TELE_EVERY", 0)
    tele_arm.run(sleeps=4)
    assert not any("tele" in d for d in tele_arm.out)


def test_missing_module_runs_as_before(tele_arm, monkeypatch):
    monkeypatch.setattr(tele_arm.mod, "servo_telemetry", None)
    tele_arm.run(sleeps=3)
    assert len(tele_arm.by_role("follower")) == 3 and not any("tele" in d for d in tele_arm.out)


# ---------- busy: forward the file ----------
def _tele_file(arm, role="follower", age=0.3, **over):
    s = st.read_sample(fake_bus(warm=3.0), now=arm.clock.now - age)
    s.update(role=role, src="pick_seq", mode="deg", **over)
    (arm.shm / f"so101_{role}_tele.json").write_text(json.dumps(s))
    return s


def test_busy_forwards_fresh_tele_file_once_and_keeps_the_state_line(tele_arm):
    tele_arm.mark_busy()
    fresh = {"t": tele_arm.clock.now - 0.5, "role": "follower", "src": "lerobot", "pos": {"gripper": 42.0}}
    (tele_arm.shm / "so101_follower.json").write_text(json.dumps(fresh))
    tele = _tele_file(tele_arm)
    tele_arm.run(sleeps=3)
    fol = tele_arm.by_role("follower")
    assert fol[0]["pos"] == {"gripper": 42.0} and fol[0]["src"] == "lerobot" and fol[0]["tele"]["t"] == tele["t"]
    assert all("tele" not in d for d in fol[1:]), "the same telemetry file is forwarded once, not every tick"


def test_busy_without_a_state_file_borrows_positions_from_the_tele_file(tele_arm):
    """pick_seq holds .arms_busy and writes only the tele file: the joints strip used to go blank here."""
    tele_arm.mark_busy()
    tele = _tele_file(tele_arm)
    tele_arm.run(sleeps=1)
    (d,) = tele_arm.by_role("follower")
    assert d["src"] == "busy-tele" and set(d["pos"]) == set(st.JOINTS) and d["tele"]["t"] == tele["t"]
    assert d["pos"]["shoulder_lift"] == pytest.approx(tele["j"]["shoulder_lift"]["pos"])


def test_busy_stale_tele_file_is_ignored(tele_arm):
    tele_arm.mark_busy()
    _tele_file(tele_arm, age=3.0)
    tele_arm.run(sleeps=1)
    (d,) = tele_arm.by_role("follower")
    assert d["src"] == "busy" and d["pos"] == {} and "tele" not in d


def test_busy_never_opens_the_bus_for_telemetry(tele_arm):
    tele_arm.mark_busy()
    tele_arm.run(sleeps=6)
    from jhelpers import FakeBus as FB
    assert FB.instances == [] and TeleBus.servo.reads == 0
