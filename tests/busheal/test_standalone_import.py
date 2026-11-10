"""Importing the plugin in a fresh interpreter without lerobot-record (teleop, calibrate) must install everything but
the pause section. One subprocess (~1.5 s); the module is loaded by path so the venv's editable install cannot win."""
import json
import subprocess
import sys

from bhelpers import PLUGIN_INIT

SCRIPT = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("lerobot_robot_busheal", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
from lerobot.motors.feetech import FeetechMotorsBus
import lerobot.utils.utils as lu, lerobot.utils.keyboard_input as ki
print(json.dumps({
    "record_loaded": "lerobot.scripts.lerobot_record" in sys.modules,
    "pause": hasattr(m, "PAUSE"),
    "say_is_relay": lu.say is m._say,
    "apply_patched": getattr(ki.apply_recording_control, "_busheal", False),
    "wrapped": all(getattr(getattr(FeetechMotorsBus, n), "_busheal", False) for n in m._WRAPPED),
    "connect": FeetechMotorsBus.connect._busheal,
    "torque": m.GRIPPER_TORQUE, "current": m.GRIPPER_CURRENT, "p": m.GRIPPER_P,
}))
"""


def test_import_without_record_installs_all_but_pause():
    env = {"LEROBOT_GRIPPER_TORQUE": "300", "LEROBOT_GRIPPER_P": "20", "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT, str(PLUGIN_INIT)], env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["record_loaded"] is False and got["pause"] is False
    assert got["say_is_relay"] and not got["apply_patched"]
    assert got["wrapped"] and got["connect"]
    # env parsing at import: current guard follows the torque limit (300 * 1.35)
    assert (got["torque"], got["current"], got["p"]) == (300, 405, 20)
