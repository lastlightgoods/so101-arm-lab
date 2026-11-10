"""Fixtures for the Jetson-side scripts. Nothing here touches hardware: no serial ports, no ssh, no /dev/shm."""
import contextlib, importlib.util, io, os, sys, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JETSON = os.path.join(ROOT, "jetson")
PY = sys.executable
RECORD_CTL = os.path.join(JETSON, "record_ctl.py")
FAKE_RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_record.py")


def wait_for(path, needle, timeout=5.0):
    """Poll a log file until `needle` appears; return the whole content. Fails loudly with the log on timeout."""
    needle = needle.encode() if isinstance(needle, str) else needle
    deadline = time.time() + timeout
    while True:
        data = open(path, "rb").read() if os.path.exists(path) else b""
        if needle in data: return data
        if time.time() > deadline: raise AssertionError(f"{needle!r} not found in log within {timeout}s:\n{data.decode(errors='replace')}")
        time.sleep(0.02)


@pytest.fixture(scope="session")
def arm_state_module():
    """Import jetson/arm_state.py once, without running its loop (import-time calibration chatter is swallowed)."""
    pytest.importorskip("lerobot", reason="arm_state.py imports lerobot.motors at the top")  # why: CI installs only requirements-dev.txt
    spec = importlib.util.spec_from_file_location("arm_state", os.path.join(JETSON, "arm_state.py"))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


class StopLoop(Exception):
    """Raised by the fake clock to leave arm_state.main() after N sleeps."""


class FakeClock:
    """Deterministic stand-in for arm_state's `time`: time() is controlled, sleep() advances it and stops the loop."""
    def __init__(self, start=1_000_000.0, max_sleeps=1):
        self.now = start; self.sleeps = []; self.max_sleeps = max_sleeps
    def time(self): return self.now
    def sleep(self, s):
        self.sleeps.append(s); self.now += s
        if len(self.sleeps) >= self.max_sleeps: raise StopLoop()


class FakeBus:
    """Mimics the slice of FeetechMotorsBus that arm_state uses. `values` is what sync_read returns; `fail` makes it raise."""
    instances = []
    values = {"shoulder_pan": 1, "gripper": 2}
    fail = None
    def __init__(self, port, motors, calibration=None):
        self.port, self.motors, self.calibration = port, motors, calibration; self.calls = []; FakeBus.instances.append(self)
    def connect(self, handshake=True): self.calls.append(("connect", handshake))
    def sync_read(self, name, normalize=True):
        self.calls.append(("sync_read", name, normalize))
        if FakeBus.fail: raise FakeBus.fail
        return dict(FakeBus.values)
    def disconnect(self, disable_torque=True): self.calls.append(("disconnect", disable_torque))


@pytest.fixture
def arm(arm_state_module, tmp_path, monkeypatch):
    """arm_state wired to a temp dir: fake ports, fake shm, fake bus, fake clock, emit() captured into `out`."""
    m = arm_state_module
    FakeBus.instances = []; FakeBus.fail = None; FakeBus.values = {"shoulder_pan": 1, "gripper": 2}
    ports = {role: tmp_path / f"{role}_port" for role in ("follower", "leader")}
    for p in ports.values(): p.write_text("")
    shm = tmp_path / "shm"; shm.mkdir()
    out = []
    clock = FakeClock()
    monkeypatch.setattr(m, "ARMS", {role: (str(p), f"{role}.json") for role, p in ports.items()})
    monkeypatch.setattr(m, "BUSY", str(tmp_path / ".arms_busy"))
    monkeypatch.setattr(m, "SHM", str(shm))
    monkeypatch.setattr(m, "SAY", str(shm / "so101_say.jsonl"))
    monkeypatch.setattr(m, "say_pos", 0)
    monkeypatch.setattr(m, "FeetechMotorsBus", FakeBus)
    monkeypatch.setattr(m, "time", clock)
    monkeypatch.setattr(m, "emit", out.append)
    monkeypatch.setattr(m, "lerobot_running", lambda: False)
    monkeypatch.setattr(m, "buses", {})
    monkeypatch.setattr(m, "cals", {"follower": None, "leader": None})

    class Harness:
        mod = m
        def __init__(self): self.out, self.clock, self.shm, self.ports, self.tmp = out, clock, shm, ports, tmp_path
        def run(self, sleeps=1):
            """Run main() until the clock has been slept `sleeps` times (one per loop tick, +1 per error backoff)."""
            clock.sleeps.clear(); clock.max_sleeps = sleeps
            with pytest.raises(StopLoop): m.main()
            return list(out)
        def by_role(self, role): return [d for d in out if d.get("role") == role]
        def mark_busy(self): (tmp_path / ".arms_busy").write_text("")
    return Harness()
