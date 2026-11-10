"""Bus self-heal around read/write, overload clearing, and the joint-state publisher."""
import json

import pytest

from lerobot.utils.errors import DeviceNotConnectedError

from busheal_fakes import GOAL_POSITION, PRESENT_POSITION, Sim, make_bus


# --- joint-state publisher -----------------------------------------------------------------------------------------

def test_sync_read_present_position_publishes_state(plugin, state_dir, clock):
    bus, sim = make_bus()
    out = bus.sync_read("Present_Position", normalize=False)
    assert out["gripper"] == 2006
    data = json.loads((state_dir / "so101_follower.json").read_text())
    assert data["role"] == "follower" and data["src"] == "lerobot" and data["mode"] == "raw"
    assert data["pos"] == {k: float(v) for k, v in out.items()}
    assert data["cal"]["gripper"] == [0, 4095]
    assert data["t"] == clock.time()
    assert not (state_dir / "so101_follower.json.tmp").exists()


def test_publish_mode_label_follows_normalize(plugin, state_dir, clock):
    bus, _ = make_bus()
    bus.sync_read("Present_Position")
    assert json.loads((state_dir / "so101_follower.json").read_text())["mode"] == "deg"


def test_other_sync_reads_do_not_publish(plugin, state_dir, clock):
    bus, _ = make_bus()
    bus.sync_read("Present_Velocity", normalize=False)
    assert not list(state_dir.glob("so101_*.json"))


def test_publish_is_rate_limited_per_role(plugin, state_dir, clock):
    bus, sim = make_bus()
    bus.sync_read("Present_Position", normalize=False)
    sim.reg[(6, PRESENT_POSITION)] = 1234
    clock.advance(0.01)
    bus.sync_read("Present_Position", normalize=False)
    assert json.loads((state_dir / "so101_follower.json").read_text())["pos"]["gripper"] == 2006.0
    clock.advance(0.05)
    bus.sync_read("Present_Position", normalize=False)
    assert json.loads((state_dir / "so101_follower.json").read_text())["pos"]["gripper"] == 1234.0


@pytest.mark.parametrize("port,role", [
    ("/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107034-if00", "follower"),
    ("/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107128-if00", "leader"),
    ("/dev/tty.so101_leader", "leader"),
    ("/dev/ttyACM0", "arm"),
])
def test_role_from_port(plugin, port, role):
    assert plugin._role(port) == role


def test_publish_failure_never_breaks_the_read(plugin, monkeypatch, state_dir, clock):
    monkeypatch.setattr(plugin, "STATE_DIR", str(state_dir / "does" / "not" / "exist"))
    bus, _ = make_bus()
    assert bus.sync_read("Present_Position", normalize=False)["gripper"] == 2006


# --- heal on a stalled adapter --------------------------------------------------------------------------------------

def test_sync_read_heals_stalled_bus_and_retries(plugin, clock, busheal_log):
    bus, sim = make_bus()
    sim.stalled = True
    out = bus.sync_read("Present_Position", normalize=False)
    assert out["shoulder_pan"] == 2001
    assert bus._heal_count == 1 and bus._healing is False
    assert sim.port.log == ["close", "open"]
    assert sim.port.timeouts, "set_timeout() must be re-applied after reopening"
    msgs = [r.getMessage() for r in busheal_log.records if r.levelname == "WARNING"]
    assert any("bus stalled during sync_read" in m and "attempt 1" in m for m in msgs)


def test_write_and_read_heal_too(plugin, clock):
    bus, sim = make_bus()
    sim.stalled = True
    bus.write("Goal_Position", "gripper", 1500, normalize=False)
    assert sim.reg[(6, GOAL_POSITION)] == 1500
    sim.stalled = True
    assert bus.read("Present_Position", "gripper", normalize=False) == 2006
    assert bus._heal_count == 2


def test_heal_keeps_reopening_until_a_motor_answers(plugin, clock, monkeypatch):
    bus, sim = make_bus()
    sim.stalled = True
    sim.heal_on_reopen = False
    opens = {"n": 0}
    real_open = sim.port.openPort

    def flaky_open():
        opens["n"] += 1
        ok = real_open()
        if opens["n"] >= 3:
            sim.stalled = False
        return ok

    monkeypatch.setattr(sim.port, "openPort", flaky_open)
    bus.sync_read("Present_Position", normalize=False)
    assert opens["n"] == 3 and bus._heal_count == 1


def test_heal_gives_up_after_max_heal_s(plugin, clock, monkeypatch):
    monkeypatch.setattr(plugin, "MAX_HEAL_S", 1.0)
    bus, sim = make_bus()
    sim.stalled = True
    sim.heal_on_reopen = False
    with pytest.raises(ConnectionError, match="could not revive the bus within 1.0s"):
        bus.sync_read("Present_Position", normalize=False)
    assert bus._healing is False
    assert sim.port.opens >= 3
    assert clock.now - 1000.0 > 1.0


def test_heal_retry_failure_propagates_without_looping(plugin, clock):
    bus, sim = make_bus()
    sim.stalled = True
    sim.sync_fail_count = 5        # the bus revives (ping answers) but the sync read keeps timing out
    with pytest.raises(ConnectionError, match="Failed to sync read"):
        bus.sync_read("Present_Position", normalize=False)
    assert bus._heal_count == 1
    assert sim.port.opens == 1


def test_no_heal_when_disabled(plugin, clock, monkeypatch):
    monkeypatch.setenv("LEROBOT_BUSHEAL", "0")
    bus, sim = make_bus()
    sim.stalled = True
    with pytest.raises(ConnectionError):
        bus.sync_read("Present_Position", normalize=False)
    assert sim.port.log == [] and not hasattr(bus, "_heal_count")


def test_no_heal_when_port_is_closed(plugin, clock):
    bus, sim = make_bus(connected=False)
    assert issubclass(DeviceNotConnectedError, ConnectionError)
    with pytest.raises(DeviceNotConnectedError):
        bus.sync_read("Present_Position", normalize=False)
    assert sim.port.log == []


def test_no_recursive_heal_while_healing(plugin, clock):
    bus, sim = make_bus()
    bus._healing = True
    sim.stalled = True
    with pytest.raises(ConnectionError):
        bus.sync_read("Present_Position", normalize=False)
    assert sim.port.log == []


# --- overload on the gripper ----------------------------------------------------------------------------------------

def test_write_clears_overload_then_retries(plugin, clock, busheal_log):
    bus, sim = make_bus()
    sim.overload_ids = {6}
    bus.write("Goal_Position", "gripper", 1000, normalize=False)
    assert bus._overload_count == 1
    assert sim.torque_writes(6) == [0, 1, 0]
    # the errored first attempt reaches the motor; the goal is then parked at the measured present position
    # before torque is cycled, and the retried write lands last
    assert sim.goal_writes(6) == [1000, 2006, 1000]
    assert 6 not in sim.overload_ids
    assert any("OVERLOAD on motor id [6] cleared" in r.getMessage() for r in busheal_log.records)


def test_overload_clear_targets_only_the_motor_in_the_error(plugin, clock):
    bus, sim = make_bus()
    sim.overload_ids = {6}
    bus.write("Goal_Position", "gripper", 1000, normalize=False)
    assert all(i == 6 for i, _, _ in sim.writes)


def test_overload_clear_does_not_park_goal_when_the_position_read_fails(plugin, clock):
    # The SCS SDK returns value 0 (not None) on a timed-out read; parking the goal at 0 would slam the gripper.
    bus, sim = make_bus()
    sim.overload_ids = {6}
    sim.read_fail_addrs = {PRESENT_POSITION}
    bus.write("Goal_Position", "gripper", 1000, normalize=False)
    assert sim.goal_writes(6) == [1000, 1000], "errored attempt + retry only: no park at a made-up position 0"
    assert sim.torque_writes(6) == [0, 1, 0]


def test_overload_not_cleared_when_disabled(plugin, clock, monkeypatch):
    monkeypatch.setenv("LEROBOT_BUSHEAL", "0")
    bus, sim = make_bus()
    sim.overload_ids = {6}
    with pytest.raises(RuntimeError, match="Overload error"):
        bus.write("Goal_Position", "gripper", 1000, normalize=False)
    assert sim.torque_writes(6) == []


def test_other_motor_errors_propagate_untouched(plugin, clock):
    bus, sim = make_bus()
    sim.overload_ids = {6}
    sim.error_text = "[RxPacketError] Input voltage error!"
    with pytest.raises(RuntimeError, match="Input voltage error"):
        bus.write("Goal_Position", "gripper", 1000, normalize=False)
    assert sim.torque_writes(6) == [] and not hasattr(bus, "_overload_count")
