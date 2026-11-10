"""SOFollower.configure(): the gripper torque cap written after lerobot's own configuration."""
import pytest

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

from busheal_fakes import RecordingBus


@pytest.fixture
def robot(tmp_path):
    r = SOFollower(SOFollowerRobotConfig(port="/dev/tty.not_a_real_port", id="busheal_test", calibration_dir=tmp_path))
    r.bus = RecordingBus()
    return r


def test_configure_is_wrapped_once(plugin):
    assert SOFollower.configure._busheal and SOFollower.configure.__name__ == "configure"


def test_configure_caps_gripper_after_stock_values(plugin, robot):
    robot.configure()
    plug = robot.bus.plugin_writes("gripper")
    names = [n for n, _, _ in plug]
    assert names == ["Max_Torque_Limit", "Torque_Limit", "Protective_Torque", "Protection_Time", "Protection_Current"]
    assert dict((n, v) for n, v, _ in plug) == {
        "Max_Torque_Limit": 250, "Torque_Limit": 250, "Protective_Torque": 0, "Protection_Time": 254,
        "Protection_Current": 337,
    }
    assert all(inblk for _, _, inblk in plug), "plugin writes happen with torque disabled"
    assert robot.bus.torque_blocks == 2
    # lerobot's own Max_Torque_Limit=500 is written first and the plugin's 250 is the last word
    mtl = [v for n, m, v, _, _ in robot.bus.writes if n == "Max_Torque_Limit" and m == "gripper"]
    assert mtl == [500, 250]


def test_default_current_guard_formula(plugin):
    assert plugin.GRIPPER_TORQUE == 250
    assert plugin.GRIPPER_CURRENT == max(250, int(plugin.GRIPPER_TORQUE * 1.35)) == 337
    assert plugin.GRIPPER_P == 0


def test_configure_p_gain_only_when_overridden(plugin, robot, monkeypatch):
    robot.configure()
    assert "P_Coefficient" not in [n for n, _, _ in robot.bus.plugin_writes()]
    robot.bus = RecordingBus()
    monkeypatch.setattr(plugin, "GRIPPER_P", 20)
    robot.configure()
    assert ("P_Coefficient", 20, True) in robot.bus.plugin_writes()


def test_configure_uses_module_constants(plugin, robot, monkeypatch):
    monkeypatch.setattr(plugin, "GRIPPER_TORQUE", 300)
    monkeypatch.setattr(plugin, "GRIPPER_CURRENT", 405)
    robot.configure()
    vals = dict((n, v) for n, v, _ in robot.bus.plugin_writes())
    assert vals["Max_Torque_Limit"] == 300 and vals["Torque_Limit"] == 300 and vals["Protection_Current"] == 405


def test_configure_skipped_when_disabled(plugin, robot, monkeypatch):
    monkeypatch.setenv("LEROBOT_BUSHEAL", "0")
    robot.configure()
    assert robot.bus.plugin_writes() == []
    assert robot.bus.torque_blocks == 1, "lerobot's own configure still ran"


def test_configure_skipped_without_gripper(plugin, robot):
    robot.bus = RecordingBus(motors=("shoulder_pan", "wrist_roll"))
    robot.configure()
    assert robot.bus.plugin_writes() == [] and robot.bus.torque_blocks == 1
