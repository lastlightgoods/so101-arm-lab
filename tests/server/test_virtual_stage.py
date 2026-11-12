"""The display-only virtual arm channel (docs/virtual-stage.md): /tmp/so101_virtual_<session>.json -> /api/state["virtual"]. Files live in a tmp dir
(SO101_VIRTUAL_DIR); nothing here talks to the live dashboard, ssh or the arm."""
import copy, json, os, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOINTS = {"shoulder_pan": 10.0, "shoulder_lift": 20.0, "elbow_flex": -30.0, "wrist_flex": 15.0, "wrist_roll": 5.0, "gripper": 40.0}


@pytest.fixture
def vs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("virtual_stage_under_test", os.path.join(ROOT, "tools", "virtual_stage.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture
def vdir(tmp_path, monkeypatch):
    d = tmp_path / "virt"; d.mkdir(); monkeypatch.setenv("SO101_VIRTUAL_DIR", str(d)); return str(d)


def put(d, name, msg, age=0.0):
    p = os.path.join(d, f"so101_virtual_{name}.json"); open(p, "w").write(json.dumps(msg))
    t = time.time() - age; os.utime(p, (t, t)); return p


def virtual(http): return http.get_json("/api/state")[0]["virtual"]


def test_every_served_message_is_labelled_sim_or_fake_and_real_is_never_accepted(srv, http, vs, vdir):
    put(vdir, "a", vs.make_message("SIM", "a", JOINTS))
    put(vdir, "b", vs.make_message("FAKE", "b", JOINTS))
    put(vdir, "c", {"source": "REAL", "session": "c", "joints": JOINTS, "seq": 1})   # a hand-written file claiming to be the real arm
    put(vdir, "d", {"session": "d", "joints": JOINTS, "seq": 1})                     # no source at all
    got = {m["session"]: m["source"] for m in virtual(http)}
    assert got == {"a": "SIM", "b": "FAKE", "c": "UNVERIFIED", "d": "UNVERIFIED"}, "REAL and missing are relabelled, never served as REAL or blank"
    with pytest.raises(ValueError): vs.make_message("REAL", "x", JOINTS)
    with pytest.raises(ValueError): vs.VirtualSink("REAL", "x")


def test_a_stale_file_disappears_by_its_mtime_not_the_senders_clock(srv, http, vs, vdir):
    put(vdir, "live", vs.make_message("FAKE", "live", JOINTS), age=0.5)
    put(vdir, "dead", vs.make_message("FAKE", "dead", JOINTS, t=time.time() + 10_000), age=3.0)   # the sender's clock says "from the future"; the file is 3 s old
    assert [m["session"] for m in virtual(http)] == ["live"]
    assert [m["session"] for m in vs.read_fresh(vdir, now=time.time() + 5)] == [], "and the live one ages out too"


def test_a_virtual_message_never_touches_the_real_arm_or_calls_out(srv, http, vs, vdir, fake, monkeypatch):
    srv.state["arms"]["follower"] = {"role": "follower", "mode": "deg", "pos": dict(JOINTS), "src": "lerobot", "rx": time.time()}
    before = copy.deepcopy(srv.state["arms"]); real = srv.scene_joints(time.time())
    called = []
    for name in ("arm_mode", "set_lock", "record_start"): monkeypatch.setattr(srv, name, lambda *a, _n=name, **k: called.append(_n))
    put(vdir, "s", vs.make_message("SIM", "s", {**JOINTS, "shoulder_pan": -77.0}))
    assert virtual(http)[0]["joints"]["shoulder_pan"] == -77.0
    assert srv.state["arms"] == before and srv.scene_joints(time.time()) == real, "state['arms'] and the detector's joints are untouched"
    assert called == [] and fake.calls == [] and fake.popens == [], "no arm-mode, lock, record, ssh or subprocess call"


def test_real_and_virtual_are_both_present_under_separate_keys(srv, http, vs, vdir):
    srv.state["arms"]["follower"] = {"role": "follower", "pos": dict(JOINTS), "src": "lerobot", "rx": time.time()}
    put(vdir, "f", vs.make_message("FAKE", "f", {**JOINTS, "elbow_flex": 99.0}))
    s = http.get_json("/api/state")[0]
    assert s["arms"]["follower"]["pos"]["elbow_flex"] == -30.0 and s["virtual"][0]["joints"]["elbow_flex"] == 99.0
    assert "virtual" not in s["arms"] and all(m["source"] in ("SIM", "FAKE") for m in s["virtual"])


def test_the_legacy_sim_state_file_is_served_as_a_sim_ghost(srv, vs, vdir):
    legacy = {"t": time.time(), "mode": "collect", "pid": 4242, "step": 7, "pos": dict(JOINTS), "obj": [0.2, 0.0, 0.01], "target": [0.25, 0.05, 0.0]}
    m = srv.virtual_arms(legacy)[0]
    assert m["source"] == "SIM" and m["session"] == "legacy-4242" and m["joint_frame"] == "urdf"
    assert {(o["kind"], o["frame"]) for o in m["objects"]} == {("nut", "sim_base"), ("target", "sim_base")}
    put(vdir, "real-sim", vs.make_message("SIM", "collect-9", JOINTS))
    assert [x["session"] for x in srv.virtual_arms(legacy)] == ["collect-9"], "a live per-session SIM stream replaces the legacy copy"
