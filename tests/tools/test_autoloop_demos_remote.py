"""tools/autoloop.py --record-demos WITHOUT --dry-run (phase 2), against fakes: the Mac side talks to the resident recorder (jetson/demo_recorderd.py's Daemon, in this process, on a
virtual clock with the fake arm's ticks and synthetic cameras) through an injected transport. No ssh, no Jetson, no camera, no arm."""
import json, os
import pytest
from autohelpers import events
from pickhelpers import load, JETSON, TOOLS

SMALL = {"top": (36, 64, 3), "front": (36, 64, 3), "wrist": (48, 64, 3)}


@pytest.fixture(scope="module")
def mods():
    pytest.importorskip("lerobot")   # why: the recorder writes real LeRobot datasets; CI has no lerobot
    import sys; sys.path.insert(0, JETSON)
    dr = load(os.path.join(JETSON, "demo_recorder.py"), "demo_recorder_for_remote_tests"); drd = load(os.path.join(JETSON, "demo_recorderd.py"), "demo_recorderd_for_remote_tests")
    ds = load(os.path.join(TOOLS, "demo_session.py"), "demo_session_for_remote_tests"); lc = load(os.path.join(JETSON, "lab_config.py"), "lab_config_for_remote_tests"); return dr, drd, ds, lc


class FakeJetson:
    """The Jetson end: a Daemon on the rig's virtual clock and `transport(op, payload, timeout)` that JSON-round-trips every request like the ssh one does.
    `fail` = {op: reply} makes an op fail; `down` = True makes everything fail (the daemon is dead); `log` = every (op, payload)."""
    def __init__(self, drd, dr, rig_clock, tmp, cams=None, busy=()):
        self.d = drd.Daemon(cams or dr.FakeCameraSource(SMALL), str(tmp / "no_ticks.jsonl"), spool=str(tmp / "spool"), vcodec="h264", clock=rig_clock, busy=busy, calib_dir=str(tmp), log=lambda m: None)
        self.fail, self.down, self.log = {}, False, []

    def transport(self, op, payload, timeout=30.0):
        self.log.append((op, payload))
        if self.down: return {"ok": False, "err": "ssh: connection refused"}
        if op in self.fail: return self.fail[op]
        return json.loads(json.dumps(self.d.handle({**json.loads(json.dumps(payload)), "op": op}), default=str))


@pytest.fixture
def remote_rig(autorig, mods, tmp_path):
    dr, drd, ds, lc = mods; ps = dr.load_pick_seq(); made = []
    def make(script=(), cams=None, busy=(), **kw):
        r = autorig(script=script, **kw); r.world.in_jaws = True; jet = FakeJetson(drd, dr, r.clock.now, tmp_path, cams, busy)
        # the fake arm's ticks: the same synthesised sink the dry run uses, written into the daemon's tick buffer on the rig's clock
        rig = ds.FakeRig(ps, jet.d.ticks, dr.synth_plan_ticks, t0=r.clock.t); jet.rig = rig
        rec = ds.RemoteRecorder(jet.transport, str(tmp_path / "data" / "nut"), "lab/nut", lc.LabConfig(lighting_tag="lamp-a"), vcodec="h264"); rec.configure()
        demo = ds.DemoSession(rec, "t", dr, track=None, radius_mm=15.0, fake=None); orig = demo.on_plan
        def on_plan(plan, rows, max_deg_s):   # the real arm's pick_seq writes the ticks; here the fake rig does, then the daemon "wakes" at each of them
            orig(plan, rows, max_deg_s); rig.run_plan(jet.d.rec if demo.recording else None, plan, rows, max_deg_s); r.clock.t = max(r.clock.t, rig.t)
        demo.on_plan = on_plan; made.append(jet); r.jet = jet; return r, demo, jet
    yield make
    for j in made:
        try: j.d.close()
        except Exception: pass


def test_landed_picks_are_kept_the_rest_rejected_through_the_remote_recorder(remote_rig, capsys):
    r, demo, jet = remote_rig(script=["ok", "ok", "drop", "ok", "ok", "ok"]); s, out = r.run(capsys, cycles=3, max_retries=0, demo=demo); cyc = events(out, "cycle")
    assert [c["demo"]["kept"] for c in cyc] == [True, False, True] and [c["demo"]["dataset"] for c in cyc] == ["main", "rejects", "main"] and cyc[1]["demo"]["discard_reason"] == "outside_radius"
    d = s["demos"]; assert (d["kept"], d["rejected"], d["aborted"], d["episodes_total"]) == (2, 1, 0, 2) and d["config_counts"] == {"stock-v1": 2} and d["dataset_hash"] and d["rejects_hash"]
    ops = [o for o, _ in jet.log]; assert ops.count("open") == 3 and ops.count("end") == 3 and ops.count("commit") == 3 and ops[-1] in ("finish", "summary")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = str(r.tmp / "data" / "nut"); assert LeRobotDataset("lab/nut", root=root).num_episodes == 2 and LeRobotDataset("lab/nut-rejects", root=root + "-rejects").num_episodes == 1
    rows = [json.loads(l) for l in open(os.path.join(root, "meta", "lab", "episodes.jsonl"))]; assert [x["teacher"]["cycle"] for x in rows] == [1, 3] and rows[0]["teacher"]["n_attempts"] == 1 and rows[0]["quality"]["tier"] == "A"
    commit_meta = [p for o, p in jet.log if o == "commit"][0]["meta"]; assert commit_meta["teacher"]["landed_err_mm"] is not None   # the teacher fields DemoSession adds at the end travel with the commit


def test_an_unreachable_recorder_at_open_never_stops_the_arm(remote_rig, capsys):
    r, demo, jet = remote_rig(); jet.fail["open"] = {"ok": False, "err": "hardware_busy:cams_busy"}; s, out = r.run(capsys, cycles=2, max_retries=0, demo=demo); cyc = events(out, "cycle")
    assert s["cycles"] == 2 and all(c["ok"] for c in cyc) and r.arm.calls   # every pick ran
    assert all(c["demo"]["discard_reason"] == "recorder_open_failed:hardware_busy:cams_busy" and not c["demo"]["kept"] and c["demo"]["episode_index"] is None for c in cyc)
    assert s["demos"]["kept"] == 0 and not any(o in ("end", "commit") for o, _ in jet.log)


def test_the_recorder_dying_mid_try_rejects_that_episode_and_trips_before_the_next_cycle(remote_rig, capsys):
    r, demo, jet = remote_rig(); orig = demo.on_plan
    def die_after_first_plan(plan, rows, max_deg_s):
        orig(plan, rows, max_deg_s); jet.down = True   # the daemon crashes in the middle of cycle 1's try
    demo.on_plan = die_after_first_plan; s, out = r.run(capsys, cycles=3, max_retries=0, demo=demo); c1, = events(out, "cycle"); t, = events(out, "trip")
    assert c1["ok"] and not c1["demo"]["kept"] and c1["demo"]["episode_index"] is None   # the pick itself finished and was judged; only the episode was lost
    assert t["reason"] == "recorder" and s["cycles"] == 1 and s["end"] == "recorder"      # a session without a recorder stops BETWEEN cycles, never mid-pick


def test_a_trip_mid_try_aborts_on_the_recorder_and_saves_nothing(remote_rig, capsys):
    r, demo, jet = remote_rig(); r.row_hook = lambda row: {**row, "max_err_deg": 9.0} if row["wp"] == "grasp" else row
    s, out = r.run(capsys, cycles=3, demo=demo); t, = events(out, "trip")
    assert t["reason"] == "motion" and s["demos"]["aborted"] == 1 and s["demos"]["episodes_total"] == 0 and ("abort", {"reason": "trip:motion"}) in jet.log and not os.path.exists(str(r.tmp / "data" / "nut"))


def test_a_missing_camera_gives_a_rejected_episode_and_the_pick_goes_on(remote_rig, mods, capsys):
    dr = mods[0]; r, demo, jet = remote_rig(cams=dr.FakeCameraSource(SMALL, freeze=[("wrist", 0.0, 1e12)])); s, out = r.run(capsys, cycles=2, max_retries=0, demo=demo); cyc = events(out, "cycle")
    assert all(c["ok"] for c in cyc) and all(not c["demo"]["kept"] and c["demo"]["dataset"] == "rejects" and "camera_frozen:wrist" in c["demo"]["discard_reason"] for c in cyc) and s["demos"]["rejected"] == 2 and s["demos"]["kept"] == 0


def test_the_dataset_version_run_is_logged_from_the_remote_summary(remote_rig, capsys):
    from pickhelpers import Recorder
    r, demo, jet = remote_rig(); demo.track = Recorder(); s, out = r.run(capsys, cycles=2, max_retries=0, demo=demo); v, = demo.track.runs
    assert v.params_["episodes_total"] == 2 and v.params_["dataset_hash"] == s["demos"]["dataset_hash"] and v.params_["features_hash"] and {"info.json", "episodes.jsonl"} <= {f.split("/")[-1] for f in v.files}


# ---------------------------------------------------------------- make_real_demo: the preflight and the pick_seq flag
def parse(al, *extra): return al.build_parser().parse_args(["--record-demos", "--session", "pre", *extra])


class Cfg:
    session = "pre"; radius_mm = 15.0


class Lt:
    class LabTrack:
        def __init__(self, name): self.name = name


def ok_transport(status_extra=None, log=None):
    def t(op, payload, timeout=30.0):
        (log if log is not None else []).append((op, payload))
        if op == "ping": return {"ok": True}
        if op == "status": return {"ok": True, "cams_missing": [], "busy": [], **(status_extra or {})}
        if op == "configure": return {"ok": True, "root": payload["root"], "rejects_root": payload["rejects_root"], "fps": 30}
        return {"ok": True}
    return t


def test_preflight_passes_sets_the_tick_sink_flag_and_configures_the_remote_datasets(al, pl, mods, monkeypatch):
    monkeypatch.setattr(pl, "PICK_SEQ_ARGS", ""); log = []; demo = al.make_real_demo(parse(al, "--lighting-tag", "lamp-b", "--demo-vcodec", "h264"), Cfg, Lt, pl, ok_transport(log=log))
    assert pl.PICK_SEQ_ARGS == "--tick-sink /dev/shm/so101_ticks.jsonl" and demo.fake is None and demo.rec.vcodec == "h264"
    cfg = dict(log)["configure"]; assert cfg["root"] == "~/so101/outputs/demos/pre/so101-nut-pick" and cfg["rejects_root"] == cfg["root"] + "-rejects" and cfg["repo"] == "lastlight/so101-nut-pick" and cfg["lab"]["lighting_tag"] == "lamp-b"


@pytest.mark.parametrize("transport, needle", [
    (lambda op, p, t=30.0: {"ok": False, "err": "no_daemon"}, "does not answer"),
    (ok_transport({"cams_missing": ["wrist"]}), "['wrist']"),
    (ok_transport({"busy": ["/x/.cams_busy"]}), "hardware flags"),
])
def test_preflight_refuses_before_any_arm_motion(al, pl, mods, monkeypatch, transport, needle):
    monkeypatch.setattr(pl, "PICK_SEQ_ARGS", "")
    with pytest.raises(SystemExit) as e: al.make_real_demo(parse(al), Cfg, Lt, pl, transport)
    assert needle in str(e.value) and pl.PICK_SEQ_ARGS == ""   # the tick sink is not switched on for a session that does not start


def test_pick_seq_gets_the_tick_sink_flag_only_when_recording(pl, monkeypatch):
    cmds = []
    class R: stdout = '{"wp": "a", "target": {}, "reached": {}}\n'; stderr = ""; returncode = 0
    monkeypatch.setattr(pl.subprocess, "run", lambda argv, **kw: (cmds.append(argv), R())[1]); monkeypatch.setattr(pl, "open", lambda *a, **k: __import__("io").StringIO(), raising=False)
    plan = {"waypoints": []}
    pl.run_plan(plan, 60)
    ssh = [c for c in cmds if c[0] == "ssh"]; assert ssh and "--tick-sink" not in ssh[-1][-1] and ssh[-1][-1].endswith("--max-deg-s 60")
    monkeypatch.setattr(pl, "PICK_SEQ_ARGS", "--tick-sink /dev/shm/x.jsonl"); pl.run_plan(plan, 60); ssh = [c for c in cmds if c[0] == "ssh"]; assert ssh[-1][-1].endswith("--max-deg-s 60 --tick-sink /dev/shm/x.jsonl")


def test_the_flag_is_off_by_default_and_dry_run_still_works_without_ssh(al):
    a = al.build_parser().parse_args([]); assert a.record_demos is False and a.demo_host == "jetson" and a.demo_tick_sink == "/dev/shm/so101_ticks.jsonl"
