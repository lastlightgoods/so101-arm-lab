"""Servo-telemetry fixtures for the UI tests: a TeleStore filled by the pretend servo bus (tools/telemetry_demo.build) and the /api/tele payload it serves."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _demo():
    if "telemetry_demo_for_ui" in sys.modules:
        return sys.modules["telemetry_demo_for_ui"]
    spec = importlib.util.spec_from_file_location("telemetry_demo_for_ui", os.path.join(ROOT, "tools", "telemetry_demo.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def build(history_s=300, hot=None, seed=3):
    """(store, bus, tick): the TeleStore holding `history_s` seconds of picking; hot='lift' runs the shoulder lift hot and sags the supply."""
    return _demo().build(history_s=history_s, hot=hot, seed=seed)


def payload(store, since=0.0):
    return store.api(since)
