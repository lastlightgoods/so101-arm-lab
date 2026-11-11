"""Fixtures for tools/servo_play.py and jetson/servo_probe.py. Nothing reaches the Jetson, a bus or the arm:
  * `sp` / `sv` load jetson/servo_probe.py and tools/servo_play.py by path; `sv_cc` / `sv_table` are cam_calib and the 9 Nov arm fit;
  * `play(...)` runs the whole sweep (and optionally the P test) against probe.FakeRig on a virtual clock and returns (rig, events, exit code, plan),
    so a test recovers the backlash / compliance the fake was built with from the script's own events;
  * `Spy` wraps a FakeRig and records every commanded step with its virtual time (to check the 50 deg/s cap) and can raise on the Nth send."""
import importlib.util, json, os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_TABLE = os.path.join(ROOT, "tests", "calib", "fixtures", "top_table_real.json")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


@pytest.fixture(scope="session")
def sp(): return _load(os.path.join(ROOT, "jetson", "servo_probe.py"), "servo_probe_under_test")


@pytest.fixture(scope="session")
def sv():
    pytest.importorskip("cv2"); return _load(os.path.join(ROOT, "tools", "servo_play.py"), "servo_play_under_test")


@pytest.fixture(scope="session")
def sv_cc(sv): return sv.cam_calib()


@pytest.fixture(scope="session")
def sv_table(): return json.load(open(FIXTURE_TABLE))


class Spy:
    """Delegates to a FakeRig; .steps = [(virtual t, cmd dict)]; raise_on_send = N makes the Nth send raise RuntimeError (a bus dying mid-test)."""
    def __init__(self, rig, clock, raise_on_send=None, interrupt_on_send=None):
        self.rig, self.clock, self.steps, self.raise_on_send, self.interrupt_on_send = rig, clock, [], raise_on_send, interrupt_on_send
    def __getattr__(self, k): return getattr(self.rig, k)
    def send(self, cmd):
        self.steps.append((self.clock.now(), dict(cmd)))
        if self.raise_on_send and len(self.steps) == self.raise_on_send: raise RuntimeError("bus died")
        if self.interrupt_on_send and len(self.steps) == self.interrupt_on_send: raise KeyboardInterrupt()
        self.rig.send(cmd)


@pytest.fixture
def play(sp, sv, sv_cc, sv_table):
    def go(rig_kw=None, p_test=None, limits=None, reps=2, rig_wrap=None, **plan_kw):
        plan, _ = sv.build_plan(sv_cc, sv_table, reps=reps, p_test=p_test, limits=limits, **plan_kw)
        clk = sp.VClock(); first = plan["poses"][0]["joints"]
        rig = sp.FakeRig(clock=clk, start={**first, "gripper": 0.0}, lever_fn=sv.lever_fn(sv_cc, sv_table), **(rig_kw or {}))
        use = rig_wrap(rig, clk) if rig_wrap else rig; ev = []
        code = sp.run_play(plan, use, ev.append, clk.now, clk.sleep, 50.0) if rig_wrap is None else None
        if rig_wrap is not None:
            try: code = sp.run_play(plan, use, ev.append, clk.now, clk.sleep, 50.0)
            except BaseException as e: code = e
        return rig, ev, code, plan, use
    return go
