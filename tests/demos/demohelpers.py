"""Fixtures and fakes for the scripted-demo tests (jetson/pick_seq.py --tick-sink, jetson/demo_recorder.py, jetson/lab_config.py, tools/demo_session.py).
Nothing here opens a camera, a serial port, ssh or /dev/shm; datasets are written under tmp_path. Uniquely named so test files can import from it directly."""
import hashlib, importlib.util, json, math, os, signal, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JETSON = os.path.join(ROOT, "jetson")
TOOLS = os.path.join(ROOT, "tools")
SMALL = {"top": (36, 64, 3), "front": (36, 64, 3), "wrist": (48, 64, 3)}   # tiny cameras so a test episode encodes in milliseconds


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


@pytest.fixture(scope="session")
def dr(): return load(os.path.join(JETSON, "demo_recorder.py"), "demo_recorder_under_test")


@pytest.fixture(scope="session")
def lc(): return load(os.path.join(JETSON, "lab_config.py"), "lab_config_under_test")


@pytest.fixture(scope="session")
def dses(): return load(os.path.join(TOOLS, "demo_session.py"), "demo_session_under_test")


@pytest.fixture(scope="session")
def ps(dr): return dr.load_pick_seq()


# ---------------------------------------------------------------- pick_seq on a fake robot
class FakeRobot:
    """Records every send_action; the servos 'follow' the command exactly. Stands in for the SO-101 follower pick_seq connects to."""
    def __init__(self, J, start=None):
        self.J = J; self.pos = dict(start or {j: 0.0 for j in J}); self.sent = []; self.reads = 0; self.fail_reads = False; self.disconnected = False

    def get_observation(self):
        self.reads += 1
        if self.fail_reads: raise OSError("bus read failed")
        return {f"{j}.pos": self.pos[j] for j in self.J}

    def send_action(self, a): self.sent.append(dict(a)); self.pos = {k[:-4]: v for k, v in a.items()}; return dict(a)

    def disconnect(self): self.disconnected = True


PLAN = {"waypoints": [
    {"name": "hover", "joints": {"shoulder_pan": 12.5, "shoulder_lift": -40.0, "elbow_flex": 55.0, "wrist_flex": 70.0, "wrist_roll": 0.5}, "gripper": 35, "seconds": 1.2, "settle": 0.1},
    {"name": "grasp", "joints": {"shoulder_pan": 13.0, "shoulder_lift": -20.0, "elbow_flex": 60.0, "wrist_flex": 95.0, "wrist_roll": 0.5}, "gripper": 35, "seconds": 0.6, "settle": 0.1},
    {"name": "close", "joints": {}, "gripper": 10, "seconds": 0.3, "settle": 0.1, "gripper_only": True},
    {"name": "lift", "joints": {"shoulder_pan": 13.0, "shoulder_lift": -45.0, "elbow_flex": 40.0, "wrist_flex": 80.0, "wrist_roll": 0.5}, "gripper": 10, "seconds": 1.0, "settle": 0.0},
]}


def commands_hash(sent):
    """sha256 over the exact float reprs of every command (byte for byte)."""
    return hashlib.sha256(json.dumps([[repr(v) for v in (a[k] for k in sorted(a))] for a in sent]).encode()).hexdigest()


def run_pick_seq(mod, tmp_path, monkeypatch, capsys, argv=(), plan=PLAN, start=None, robot=None):
    """mod.main() with a FakeRobot, no sleeping, no busy flag on the real home. Returns (robot, stdout lines)."""
    rob = robot or FakeRobot(mod.J, start or {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 70.0, "wrist_roll": 0.5, "gripper": 0.0})
    pf = tmp_path / "plan.json"; pf.write_text(json.dumps(plan))
    monkeypatch.setattr(mod, "connect", lambda: rob); monkeypatch.setattr(mod, "lerobot_running", lambda: False); monkeypatch.setattr(mod, "BUSY", str(tmp_path / "arms_busy"))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None); monkeypatch.setattr(sys, "argv", ["pick_seq.py", "--plan", str(pf), "--correct", "0.8", *argv])
    mod.main(); out = capsys.readouterr().out
    return rob, [json.loads(l) for l in out.splitlines() if l.startswith("{")]


# ---------------------------------------------------------------- recorder fixtures
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def feed_ticks(buf, t0, n, hz=40.0, seq0=1, wp="a", base=0.0):
    """n ticks at `hz` from monotonic time t0: action[m] = base + seq (+ motor index / 10), state = the previous action. Returns the time of the last tick."""
    for k in range(n):
        s = seq0 + k; t = t0 + k / hz
        buf.add({"seq": s, "t_mono": t, "t_wall": t, "wp": wp, "action": {f"{m}.pos": base + s + i / 10 for i, m in enumerate(MOTORS)}, "state": {f"{m}.pos": base + s - 1 + i / 10 for i, m in enumerate(MOTORS)}})
    return t0 + (n - 1) / hz


@pytest.fixture
def rig(dr, lc, tmp_path):
    """make(**kw) -> (DemoRecorder, TickBuffer, FakeCameraSource); datasets under tmp_path/data, repo ids lab/demo and lab/demo-rejects."""
    made = []
    def make(cams=None, cfg=None, root=None, **kw):
        buf = dr.TickBuffer(); cams = cams or dr.FakeCameraSource(SMALL); cfg = cfg or lc.LabConfig(lighting_tag="test-lamp")
        rec = dr.DemoRecorder(str(root or tmp_path / "data" / "demo"), "lab/demo", cams, buf, cfg, calib_hashes={"top_table.json": {"sha12": "abc123def456", "mtime": 1}}, spool_dir=str(tmp_path / "spool"), **kw)
        os.makedirs(tmp_path / "spool", exist_ok=True); made.append(rec); return rec, buf, cams
    yield make
    for r in made:
        try: r.close()
        except Exception: pass


def record_episode(rec, buf, t0=1000.0, n_ticks=80, keep=True, verdict=None, meta=None, **feed):
    """open -> feed ticks -> pump along -> end -> commit; returns the commit result."""
    rec.open(meta or {"scene": {"target_in": [9.0, 6.0]}, "teacher": {"session": "s", "cycle": 1}, "links": {"mlflow_cycle_run_name": "autoloop-s-c1"}}, t0)
    last = feed_ticks(buf, t0 + 0.2, n_ticks, **feed)
    t = t0
    while t <= last + 0.1: rec.pump(t); t += 1 / 120
    rec.end(last + 0.1); return rec.commit(keep, verdict or {"tier": "A", "keep": keep})
