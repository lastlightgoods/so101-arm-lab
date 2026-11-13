"""disconnect(): torque off motor by motor, never abort on one refusing servo.
12 Nov 2026: lerobot-record core-dumped on disconnect ("Failed to write 'Torque_Enable' on id_=2 with '0' after 6 tries") after the
episode was saved; LeRobot's one-call disable_torque raised at servo 2 and the motors after it kept their torque."""
import logging

from busheal_fakes import TORQUE_ENABLE, Sim, make_bus


def test_a_servo_that_refuses_torque_off_does_not_abort_and_the_others_still_go_off(plugin, caplog, clock):
    caplog.set_level(logging.ERROR, logger="lerobot_robot_busheal")
    bus, sim = make_bus()
    sim.sticky_error_ids = {2}; sim.error_text = ""   # the real refusal carried an error status with no text
    bus.disconnect(True)                               # must not raise
    for i in (1, 3, 4, 5, 6):
        assert sim.reg[(i, TORQUE_ENABLE)] == 0, f"motor {i} still has torque"
    assert len(sim.torque_writes(2)) >= 2, "the refusing servo is retried"
    assert "shoulder_lift" in caplog.text and "refused" in caplog.text
    assert not sim.port.is_open, "the port is closed anyway"


def test_a_clean_disconnect_turns_every_motor_off_and_closes(plugin):
    bus, sim = make_bus()
    bus.disconnect(True)
    assert all(sim.reg[(i, TORQUE_ENABLE)] == 0 for i in range(1, 7)) and not sim.port.is_open


def test_disconnect_without_torque_off_leaves_torque_alone(plugin):
    bus, sim = make_bus()
    for i in range(1, 7): sim.reg[(i, TORQUE_ENABLE)] = 1
    bus.disconnect(False)
    assert all(sim.reg[(i, TORQUE_ENABLE)] == 1 for i in range(1, 7)) and not sim.port.is_open
