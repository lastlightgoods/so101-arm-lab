"""Test harness for the lerobot_robot_busheal plugin.

How the plugin is loaded (read this before changing anything here):

* The plugin monkeypatches lerobot classes *at import time* and guards each patch with a ``_busheal`` marker, so it
  is imported exactly once per pytest session, at collection, from THIS worktree. Tests never reload it; they reset
  its mutable state through the autouse fixtures below.
* The venv has an *editable* install of the plugin whose meta-path finder maps ``lerobot_robot_busheal`` to the
  main checkout (``~/so101/plugins/...``). A plain ``import lerobot_robot_busheal`` would therefore test the wrong
  copy, so the module is loaded from the worktree file path under that name and registered in ``sys.modules``
  (the plugin dir is also put on ``sys.path`` as requested, but the explicit load is what makes it authoritative).
  ``test_loaded_from_worktree`` asserts the path.
* The record-pause section of the plugin only installs when ``lerobot.scripts.lerobot_record`` is already in
  ``sys.modules`` (that is how lerobot-record imports plugins: from inside ``record()``). That module is imported
  here first on purpose; it costs ~4.5 s once and is the bulk of the suite's run time.
* Env vars the plugin reads at import are cleared before the import so the defaults under test are deterministic.

Hermeticity: ``time.sleep`` raises unless a test requests the virtual ``clock`` fixture; the plugin's ``STATE_DIR``
is redirected to ``tmp_path`` for every test (``so101_say.jsonl`` would otherwise land in /tmp); ``_orig_say`` (the
macOS ``say`` subprocess) is replaced by a recorder; no serial port, camera or socket is ever opened.
"""
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = ROOT / "plugins" / "lerobot-robot-busheal"
PLUGIN_INIT = PLUGIN_DIR / "lerobot_robot_busheal" / "__init__.py"

sys.path.insert(0, str(PLUGIN_DIR))
for _v in (
    "LEROBOT_BUSHEAL", "LEROBOT_LOCAL_SAY", "LEROBOT_GRIPPER_TORQUE", "LEROBOT_GRIPPER_CURRENT",
    "LEROBOT_GRIPPER_P", "LEROBOT_BUSHEAL_MAX_S", "LEROBOT_BUSHEAL_CONNECT_S", "LEROBOT_CAM_STALE_S",
    "LEROBOT_CAM_REOPEN_S",
):
    os.environ.pop(_v, None)

import lerobot.scripts.lerobot_record  # noqa: E402,F401  (must precede the plugin import, see docstring)


def _load_plugin():
    if "lerobot_robot_busheal" in sys.modules:
        return sys.modules["lerobot_robot_busheal"]
    spec = importlib.util.spec_from_file_location(
        "lerobot_robot_busheal", PLUGIN_INIT, submodule_search_locations=[str(PLUGIN_INIT.parent)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


PLUGIN = _load_plugin()


class Clock:
    """Virtual clock: sleep() advances it instead of waiting."""

    def __init__(self, start=1000.0):
        self.start = start
        self.now = start
        self.sleeps = []

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += max(float(s), 0.0)

    def advance(self, s):
        self.now += s

    def perf_counter(self):
        return self.now

    def monotonic(self):
        return self.now

    def time(self):
        return 1.7e9 + (self.now - self.start)


class SayRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, text, blocking=False):
        self.calls.append((text, blocking))


@pytest.fixture
def plugin():
    return PLUGIN


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    def _refuse(s):
        raise AssertionError(f"time.sleep({s}) called in a test; use the `clock` fixture")

    monkeypatch.setattr(time, "sleep", _refuse)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    for name in ("sleep", "perf_counter", "monotonic", "time"):
        monkeypatch.setattr(time, name, getattr(c, name))
    return c


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(PLUGIN, "STATE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def local_say(monkeypatch):
    rec = SayRecorder()
    monkeypatch.setattr(PLUGIN, "_orig_say", rec)
    monkeypatch.delenv("LEROBOT_LOCAL_SAY", raising=False)
    return rec


@pytest.fixture(autouse=True)
def _reset_plugin_state(monkeypatch):
    monkeypatch.delenv("LEROBOT_BUSHEAL", raising=False)
    PLUGIN._last_pub.clear()
    if hasattr(PLUGIN, "PAUSE"):
        for k in list(PLUGIN.PAUSE):
            PLUGIN.PAUSE[k] = False if k == "on" else 0.0
    if hasattr(PLUGIN, "LAST_SAY"):
        PLUGIN.LAST_SAY["text"] = ""
    yield
    if hasattr(PLUGIN, "PAUSE"):
        PLUGIN.PAUSE["on"] = False


@pytest.fixture
def say_lines(state_dir):
    """Lines appended to so101_say.jsonl by the relay, parsed."""
    import json

    def _read():
        p = state_dir / "so101_say.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    return _read


@pytest.fixture
def busheal_log(caplog):
    caplog.set_level(logging.DEBUG, logger="lerobot_robot_busheal")
    return caplog
