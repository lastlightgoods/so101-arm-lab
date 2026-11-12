"""The publishers of the virtual-arm channel (tools/virtual_stage.py, docs/virtual-stage.md): the simulator's state hook, autoloop's DryArm sink with
--stage, demo_session's FakeRig sink, and a hygiene check on the new files. Everything writes to tmp dirs; nothing reaches the dashboard, ssh or an arm."""
import importlib.util, json, os, re, sys, types
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VS_PY = os.path.join(ROOT, "tools", "virtual_stage.py")
JOINTS6 = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture
def vs(): return load_by_path(VS_PY, "virtual_stage_publishers_test")


def test_the_sim_state_hook_stamps_source_and_session_and_writes_the_per_session_file(tmp_path, monkeypatch):
    pytest.importorskip("so101_nexus.lerobot_dataset")
    before = list(sys.meta_path); sim = load_by_path(os.path.join(ROOT, "sim", "so101_simstate.py"), "so101_simstate_under_test")
    sys.meta_path[:] = before   # the module installs its import hook at import; this test must not leave it behind
    sim.PATH = str(tmp_path / "sim_state.json"); sim.VDIR = str(tmp_path)

    class Env:
        action_space = types.SimpleNamespace(low=np.array([-1.0] * 5 + [-0.17]), high=np.array([1.0] * 5 + [1.75]))
        def reset(self, *a, **k): return {"state": np.zeros(6)}, {}
        def step(self, action, *a, **k): return {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8])}, 0.0, False, False, {"success": True}
        def _get_object_pose(self): return np.array([0.2, 0.0, 0.01, 1, 0, 0, 0])
        def _get_target_pos(self): return np.array([0.25, 0.05, 0.0])
    sim._patch(types.SimpleNamespace(SO101NexusMuJoCoBaseEnv=Env))
    env = Env(); env.reset(); env.step(None)
    legacy = json.load(open(sim.PATH))
    assert legacy["source"] == "SIM" and legacy["session"] == f"{legacy['mode']}-{os.getpid()}" and set(legacy["pos"]) == set(JOINTS6)
    msg = json.load(open(tmp_path / f"so101_virtual_{legacy['session']}.json"))
    assert msg["source"] == "SIM" and msg["session"] == legacy["session"] and msg["joint_frame"] == "urdf"
    assert {(o["kind"], o["frame"]) for o in msg["objects"]} == {("nut", "sim_base"), ("target", "sim_base")}


def test_a_sink_publishes_nothing_by_default_and_frames_are_ordered_and_paced_with_one(al, cc, table, vs, tmp_path):
    plan = {"waypoints": [{"name": n, "joints": {k: v for k, v in zip(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"), j)}, "gripper": g, "gripper_only": False}
                          for n, j, g in (("hover", (10, 15, 5, 80, 0.5), 35), ("grasp", (12, 20, 8, 75, 0.5), 35), ("rest", (0, -100, 90, 70, 0, ), 0))]}
    plan["waypoints"][2].update(gripper_only=True); plan["waypoints"][2]["joints"]["wrist_roll"] = 0.0

    def run(sink):
        world = al.FakeWorld(cc, table, lambda *a, **k: None, nut_in=(6.0, 8.0)); arm = al.DryArm(cc, table, world, sink=sink); arm.target_in = (9.0, 6.0)
        return arm, arm.run_plan(plan, 30.0)
    arm0, rows0 = run(None)
    assert type(arm0.sink).__name__ == "NullSink" and arm0.sink.active is False and arm0.pose is None, "the default sink does nothing"
    frames, sleeps = [], []
    sink = vs.VirtualSink("FAKE", "t1", speed=2.0, writer=frames.append, sleeper=sleeps.append)
    arm1, rows1 = run(sink)
    assert [r["reached"] for r in rows1] == [r["reached"] for r in rows0], "staging changes nothing the run sees"
    assert [m["seq"] for m in frames] == list(range(1, len(frames) + 1)) and len(frames) > 30 and {m["source"] for m in frames} == {"FAKE"}
    assert pytest.approx(sum(sleeps)) == 3.0 / 2.0, "3 waypoints x 1 s of virtual motion, at --stage-speed 2"
    last = frames[-1]
    assert last["joints"]["gripper"] == 0.0 and abs(last["joints"]["shoulder_lift"] + 100) < 1e-6, "ends on the rest pose"
    assert {(o["kind"], o["frame"]) for o in last["objects"]} == {("nut", "mat_in"), ("target", "mat_in")}
    assert [o["x"] for o in frames[0]["objects"] if o["kind"] == "target"] == [9.0]
    # monotone glide: shoulder_lift never moves back toward the previous waypoint within a waypoint
    lift = [m["joints"]["shoulder_lift"] for m in frames]; assert lift[0] != lift[-1]


def test_autoloop_dry_run_stage_writes_live_frames_paced_in_real_time_and_removes_them_at_the_end(al, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SO101_VIRTUAL_DIR", str(tmp_path / "v")); (tmp_path / "v").mkdir()
    seen, slept = [], []
    def fake_sleep(s):
        slept.append(s)
        for n in os.listdir(tmp_path / "v"):
            if n.endswith(".json"): seen.append(json.load(open(tmp_path / "v" / n)))
    monkeypatch.setattr("time.sleep", fake_sleep)
    rc = al.main(["--dry-run", "--stage", "--cycles", "1", "--seed", "2", "--session", "stg", "--out-dir", str(tmp_path / "out"), "--dry-miss", "0", "--settle", "0", "--stop-file", str(tmp_path / "none.stop")])
    capsys.readouterr()
    assert rc == 0 and len(seen) > 50 and sum(slept) > 10, "real time: seconds of waiting, not a flash"
    assert {m["session"] for m in seen} == {"stg"} and {m["source"] for m in seen} == {"FAKE"} and {m["joint_frame"] for m in seen} == {"lerobot"}
    seqs = [m["seq"] for m in seen]; assert seqs == sorted(seqs)
    assert any(o["kind"] == "target" for m in seen for o in m["objects"]) and any(o["kind"] == "nut" for m in seen for o in m["objects"])
    assert os.listdir(tmp_path / "v") == [], "the ghost goes when the run ends"
    with pytest.raises(SystemExit): al.main(["--stage", "--cycles", "1", "--out-dir", str(tmp_path / "o2")])   # --stage is for fake runs only


def test_the_fake_rig_of_demo_session_can_use_the_same_sink(tmp_path, vs):
    ds = load_by_path(os.path.join(ROOT, "tools", "demo_session.py"), "demo_session_publishers_test")
    frames, sleeps = [], []
    sink = vs.VirtualSink("FAKE", "demo", speed=1.0, writer=frames.append, sleeper=sleeps.append)
    rig = ds.FakeRig(types.SimpleNamespace(J=JOINTS6), None, lambda *a: (1001.0, {}, []), sink=sink)
    rows = [{"reached": {k: float(i + n) for i, k in enumerate(JOINTS6)}, "seconds": 1.0} for n in (0, 5)]
    rig.run_plan(None, {"waypoints": []}, rows, 30.0)
    assert len(frames) > 20 and frames[-1]["joints"]["shoulder_pan"] == 5.0 and sum(sleeps) == pytest.approx(2.0)


def test_new_files_hold_no_ips_emails_or_home_paths():
    files = ["tools/virtual_stage.py", "sim/so101_simstate.py", "docs/virtual-stage.md", "tests/server/test_virtual_stage.py", "tests/tools/test_virtual_stage_publishers.py", "tests/ui/test_virtual_ghost.py"]
    bad = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|[\w.+-]+@[\w-]+\.[\w.]+|/Us" + "ers/|/ho" + "me/\\w")
    ok_ip = re.compile(r"\b127\.0\.0\.1\b")
    for f in files:
        p = os.path.join(ROOT, f)
        assert os.path.exists(p), f
        for n, line in enumerate(open(p), 1):
            hit = [m for m in bad.findall(ok_ip.sub("", line))]
            assert not hit, f"{f}:{n}: {hit}"
    # and the added lines of the edited files
    import subprocess
    diff = subprocess.run(["git", "diff", "main", "--", "dashboard/index.html", "dashboard/server.py", "tools/autoloop.py", "tools/demo_session.py"], cwd=ROOT, capture_output=True, text=True).stdout
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"): assert not bad.findall(ok_ip.sub("", line)), line[:120]
