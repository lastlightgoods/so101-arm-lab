"""Helpers for the servo-telemetry tests: the Jetson-side module and the pretend servo bus, loaded by path under unique names."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def servo_telemetry():
    """jetson/servo_telemetry.py as a plain import (jetson/ on sys.path), so arm_state, pick_seq, tools/fake_servo_bus.py and the tests all share ONE module object
    (monkeypatching a function on it must reach the code under test)."""
    jet = os.path.join(ROOT, "jetson")
    if jet not in sys.path:
        sys.path.insert(0, jet)
    import servo_telemetry as m
    assert m.__file__.startswith(ROOT), m.__file__
    return m


def fake_bus_module():
    servo_telemetry()
    return load(os.path.join(ROOT, "tools", "fake_servo_bus.py"), "fake_servo_bus_for_tests")


def fake_bus(seed=1, warm=0.0, **kw):
    b = fake_bus_module().FakeServoBus(seed=seed, **kw)
    if warm:
        b.advance(warm)
    return b
