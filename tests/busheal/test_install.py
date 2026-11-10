"""The plugin is loaded from this worktree, installs each patch exactly once, and stays Python 3.10 syntax."""
import ast
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus

from bhelpers import PLUGIN_INIT, ROOT


def test_loaded_from_worktree(plugin):
    assert Path(plugin.__file__).resolve() == PLUGIN_INIT.resolve()
    assert str(ROOT) in plugin.__file__


def test_bus_methods_wrapped_and_marked(plugin):
    for name in plugin._WRAPPED:
        fn = getattr(FeetechMotorsBus, name)
        assert getattr(fn, "_busheal", False), name
        assert fn.__name__ == name
    assert FeetechMotorsBus.connect._busheal


def test_wrap_is_idempotent(plugin):
    before = {n: getattr(FeetechMotorsBus, n) for n in plugin._WRAPPED}
    for n in plugin._WRAPPED:
        plugin._wrap(n)
    assert {n: getattr(FeetechMotorsBus, n) for n in plugin._WRAPPED} == before


def test_plugin_parses_as_python_310():
    # The Jetson runs 3.10: no `match`, no `except*`, no PEP 695. feature_version rejects the newer grammar.
    src = PLUGIN_INIT.read_text()
    ast.parse(src, filename=str(PLUGIN_INIT), feature_version=(3, 10))
    assert "match " not in [ln.strip()[:6] for ln in src.splitlines()]


def test_enabled_flag_reads_env_at_call_time(plugin, monkeypatch):
    assert plugin._enabled()
    monkeypatch.setenv("LEROBOT_BUSHEAL", "0")
    assert not plugin._enabled()
    monkeypatch.setenv("LEROBOT_BUSHEAL", "1")
    assert plugin._enabled()
