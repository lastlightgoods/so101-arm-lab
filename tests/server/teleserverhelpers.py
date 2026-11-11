"""Servo-telemetry fixtures for the dashboard server tests: samples built by the pretend servo bus through the real decoder, and a TeleStore on a fake clock."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def modules():
    """(servo_telemetry, fake_servo_bus, telemetry_store): the Jetson module as a plain import (one shared instance), the others under test-only names."""
    jet = os.path.join(ROOT, "jetson")
    if jet not in sys.path:
        sys.path.insert(0, jet)
    import servo_telemetry as st
    fb = _load(os.path.join(ROOT, "tools", "fake_servo_bus.py"), "fake_servo_bus_srvtest")
    ts = _load(os.path.join(ROOT, "dashboard", "telemetry_store.py"), "telemetry_store_srvtest")
    return st, fb, ts


class Clock:
    def __init__(self, t=5000.0): self.t = t
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def sample(st, bus, **tweak):
    """One sample from the pretend bus; tweak = {joint: {field: value}} edits applied after decoding, to force a state."""
    s = st.read_sample(bus, now=bus.t)
    for n, edits in tweak.items():
        s["j"][n].update(edits)
    s.update(st.summarise_now(s["j"]))
    return s
