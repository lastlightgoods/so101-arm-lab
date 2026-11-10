"""connect(): the handshake is retried while the motors do not answer, up to CONNECT_MAX_S."""
import pytest

from busheal_fakes import make_bus


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_connect_retries_motor_check_until_motors_answer(plugin, clock, busheal_log):
    bus, sim = make_bus(connected=False)
    sim.ping_fail_rounds = 2
    bus.connect()
    assert bus.is_connected
    assert sim.port.opens == 3 and sim.port.log.count("close") == 2
    assert sim.port.timeouts, "set_timeout() runs after the successful handshake"
    assert [m for m in _warnings(busheal_log) if "motors not answering (attempt 1)" in m]
    assert clock.sleeps == [0.6, 0.6]


def test_connect_warning_cadence_is_first_then_every_fifth(plugin, clock, busheal_log):
    bus, sim = make_bus(connected=False)
    sim.ping_fail_rounds = 11
    bus.connect()
    attempts = []
    for m in _warnings(busheal_log):
        if "motors not answering (attempt" in m:
            attempts.append(int(m.split("attempt ")[1].split(")")[0]))
    assert attempts == [1, 5, 10]


def test_connect_gives_up_after_connect_max_s(plugin, clock, monkeypatch, busheal_log):
    monkeypatch.setattr(plugin, "CONNECT_MAX_S", 2.0)
    bus, sim = make_bus(connected=False)
    sim.ping_fail_rounds = 1000
    with pytest.raises(RuntimeError, match="motor check failed"):
        bus.connect()
    assert not bus.is_connected, "the port is left closed after giving up"
    assert any("still not answering after 2s" in r.getMessage() and r.levelname == "ERROR" for r in busheal_log.records)
    assert 2.0 < clock.now - 1000.0 < 3.5


def test_connect_open_failure_is_retried_as_connection_error(plugin, clock):
    bus, sim = make_bus(connected=False)
    sim.port.open_ok = False
    real_open = sim.port.openPort

    def open_after_two():
        if sim.port.log.count("open") >= 2:
            sim.port.open_ok = True
        return real_open()

    sim.port.openPort = open_after_two
    bus.connect()
    assert bus.is_connected and sim.port.opens == 3


def test_connect_firmware_mismatch_is_not_retried(plugin, clock):
    bus, sim = make_bus(connected=False)
    sim.firmware[3] = (3, 10)
    with pytest.raises(RuntimeError, match="different firmware"):
        bus.connect()
    assert sim.port.opens == 1 and clock.sleeps == []


def test_connect_disabled_raises_immediately(plugin, clock, monkeypatch):
    monkeypatch.setenv("LEROBOT_BUSHEAL", "0")
    bus, sim = make_bus(connected=False)
    sim.ping_fail_rounds = 1
    with pytest.raises(RuntimeError, match="motor check failed"):
        bus.connect()
    assert sim.port.opens == 1 and clock.sleeps == []


def test_connect_without_handshake_passes_flag_through(plugin, clock):
    bus, sim = make_bus(connected=False)
    sim.ping_fail_rounds = 5     # would fail a handshake, but none is asked for
    bus.connect(handshake=False)
    assert bus.is_connected and sim.port.opens == 1
