"""tools/autoloop.py --record-demos (tools/demo_session.py + jetson/demo_recorder.py) on the ShortArm / FakeWorld dry-run rig: each TRY becomes a LeRobot episode that is
kept or rejected by the judge's result, the dry-run command line writes real datasets, and with the flag OFF nothing changes. No arm, no ssh, no camera, no Hugging Face."""
import hashlib, json, os, sys
import pytest
from autohelpers import events
from pickhelpers import load, JETSON, TOOLS

SMALL = {"top": (36, 64, 3), "front": (36, 64, 3), "wrist": (48, 64, 3)}
GOLDEN_OFF = ("1ddd6e431d3d3b49cbb0fd7d93d31a7dc5437c9ab424b0aeeb6bda2c6e50157b", 63)   # sha256 of the normalised dry-run output of main's autoloop.py (6 cycles, seed 11, dry-miss 0.3), before --record-demos existed


@pytest.fixture(scope="module")
def dmods():
    dr = load(os.path.join(JETSON, "demo_recorder.py"), "demo_recorder_for_autotests"); lc = load(os.path.join(JETSON, "lab_config.py"), "lab_config_for_autotests")
    ds = load(os.path.join(TOOLS, "demo_session.py"), "demo_session_for_autotests"); return dr, lc, ds


@pytest.fixture
def demo_rig(autorig, dmods, tmp_path):
    """make(script=(), track=None, cams=None, **rig kwargs) -> (rig, DemoSession); the fake world says the jaws hold the nut."""
    dr, lc, ds = dmods; ps = dr.load_pick_seq(); made = []
    def make(script=(), track=None, cams=None, **kw):
        r = autorig(script=script, **kw); r.world.in_jaws = True; buf = dr.TickBuffer()
        rec = dr.DemoRecorder(str(tmp_path / "data" / "nut"), "lab/nut", cams or dr.FakeCameraSource(SMALL), buf, lc.LabConfig(lighting_tag="lamp-a"), spool_dir=str(tmp_path / "spool"), calib_hashes={"top_table.json": {"sha12": "0123456789ab", "mtime": 1}})
        os.makedirs(tmp_path / "spool", exist_ok=True); demo = ds.DemoSession(rec, "t", dr, track=track, radius_mm=15.0, fake=ds.FakeRig(ps, buf, dr.synth_plan_ticks)); made.append(demo); r.buf = buf; return r, demo
    yield make
    for d in made:
        try: d.rec.close()
        except Exception: pass


def load_back(root, repo="lab/nut"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo, root=root)


def script(n_cycles, bad=(), kind="drop"):
    s = ["ok"] * (2 * n_cycles)
    for c in bad: s[2 * (c - 1)] = kind
    return s


def test_kept_episodes_equal_landed_cycles_and_the_rest_go_to_the_rejects_dataset(demo_rig, capsys, tmp_path):
    r, demo = demo_rig(script=script(4, bad=(2,))); s, out = r.run(capsys, cycles=4, max_retries=0, demo=demo); cyc = events(out, "cycle")
    assert [c["ok"] for c in cyc] == [True, False, True, True] and s["landed"] == 3
    assert [c["demo"]["kept"] for c in cyc] == [True, False, True, True] and [c["demo"]["dataset"] for c in cyc] == ["main", "rejects", "main", "main"]
    assert cyc[1]["demo"]["discard_reason"] == "outside_radius" and all(c["demo"]["tier"] == "A" for c in cyc if c["demo"]["kept"]) and cyc[1]["demo"]["tier"] is None
    d = s["demos"]; assert (d["kept"], d["rejected"], d["aborted"], d["episodes_total"]) == (3, 1, 0, 3) and d["config_counts"] == {"stock-v1": 3} and d["dataset_hash"] and d["fps"] == 30
    main = load_back(demo.rec.root); rej = load_back(demo.rec.rejects_root, "lab/nut-rejects")
    assert main.num_episodes == 3 and rej.num_episodes == 1 and main.fps == 30 and rej.fps == 30 and set(main.features) == set(rej.features)
    assert [x["teacher"]["cycle"] for x in demo.dr.read_sidecar(demo.rec.root)] == [1, 3, 4] and [x["teacher"]["cycle"] for x in demo.dr.read_sidecar(demo.rec.rejects_root)] == [2]


def test_success_after_the_retry_is_tier_R_and_spans_both_attempts(demo_rig, capsys):
    r, demo = demo_rig(script=["miss", "ok", "ok", "ok"]); s, out = r.run(capsys, cycles=2, max_retries=1, demo=demo); c1, c2 = events(out, "cycle")
    assert c1["retries"] == 1 and c1["ok"] and c1["demo"]["kept"] and c1["demo"]["tier"] == "R" and c2["demo"]["tier"] == "A"
    rows = demo.dr.read_sidecar(demo.rec.root); assert [x["quality"]["tier"] for x in rows] == ["R", "A"] and rows[0]["teacher"]["n_attempts"] == 2 and rows[1]["teacher"]["n_attempts"] == 1
    assert rows[0]["n_frames"] > 1.7 * rows[1]["n_frames"], "the R episode holds the failed grasp and the re-grasp"


def test_the_jaw_check_decides(demo_rig, capsys):
    for held, kept, why in ((False, False, "not_picked"), (None, False, "jaw_check_missing"), (True, True, None)):
        r, demo = demo_rig(); r.world.in_jaws = held; s, out = r.run(capsys, cycles=1, max_retries=0, demo=demo); c, = events(out, "cycle")
        assert c["demo"]["kept"] is kept and c["demo"]["discard_reason"] == why, (held, c["demo"]); demo.rec.close()


def test_a_trip_in_the_middle_of_the_try_saves_nothing_anywhere(demo_rig, capsys):
    r, demo = demo_rig(); r.row_hook = lambda row: {**row, "max_err_deg": 9.0} if row["wp"] == "grasp" else row
    s, out = r.run(capsys, cycles=3, demo=demo); t, = events(out, "trip"); c, = events(out, "cycle")
    assert t["reason"] == "motion" and "demo" not in c and s["demos"]["aborted"] == 1 and s["demos"]["kept"] == s["demos"]["rejected"] == 0 and s["demos"]["episodes_total"] == 0
    assert not os.path.exists(demo.rec.root) and not os.path.exists(demo.rec.rejects_root) and os.listdir(demo.rec.spool_root) == []


def test_the_stop_file_between_two_stages_of_the_try_discards_the_episode(demo_rig, capsys):
    r, demo = demo_rig(); orig = demo.on_plan
    def touch_after_first_plan(plan, rows, max_deg_s):
        orig(plan, rows, max_deg_s)
        if demo.recording: open(r.stop_file, "w").close()   # TC hits STOP: the next plan is refused
    demo.on_plan = touch_after_first_plan; s, out = r.run(capsys, cycles=3, demo=demo)
    t, = events(out, "stop"); assert t["reason"] == "stop_file" and s["demos"]["aborted"] == 1 and s["demos"]["episodes_total"] == 0 and not os.path.exists(demo.rec.root)


def test_a_dead_recorder_is_a_trip_and_the_arm_never_moves(demo_rig, capsys):
    r, demo = demo_rig(); demo.alive = lambda: False; s, out = r.run(capsys, cycles=3, demo=demo)
    t, = events(out, "trip"); assert t["reason"] == "recorder" and s["cycles"] == 0 and r.arm.calls == [] and s["end"] == "recorder"


def test_a_trip_after_the_try_rejects_instead_of_keeping(demo_rig, capsys):
    r, demo = demo_rig(script=script(3, bad=(1, 2, 3), kind="drop")); s, out = r.run(capsys, cycles=5, max_retries=0, demo=demo)   # three failures in a row end the session with a trip
    assert s["end"] == "consecutive_failures" and s["demos"]["rejected"] == 3 and s["demos"]["kept"] == 0
    rows = demo.dr.read_sidecar(demo.rec.rejects_root); assert rows[-1]["verdict"]["reasons"][:2] == ["outside_radius", "guardrail:consecutive_failures"] and rows[-1]["teacher"]["trip"] == "consecutive_failures"


def test_the_reset_is_not_recorded_and_the_idle_gaps_are_dropped(demo_rig, capsys):
    r, demo = demo_rig(); seen = []; orig = demo.on_plan
    def spy(plan, rows, max_deg_s): was = demo.recording; n0 = len(r.buf.ticks); orig(plan, rows, max_deg_s); seen.append((was, n0, len(r.buf.ticks)))
    demo.on_plan = spy; s, out = r.run(capsys, cycles=1, max_retries=0, demo=demo)
    resets = [(a, b) for was, a, b in seen if not was]; tries = [(a, b) for was, a, b in seen if was]; assert resets and tries, "plans ran both inside and outside the episode"
    log = json.load(open(os.path.join(demo.rec.root, "meta", "lab", "frames", "episode_000000.json"))); first_reset_tick = resets[0][0] + 1
    assert max(log["tick_seq"]) < first_reset_tick and min(log["tick_seq"]) >= tries[0][0] + 1
    row, = demo.dr.read_sidecar(demo.rec.root); assert row["quality"]["dropped_idle_frames"] > 0 and row["n_frames"] < (row["t_end_mono"] - row["t_open_mono"]) * 30 - row["quality"]["dropped_idle_frames"] + 2


def test_a_nut_that_cannot_be_moved_off_the_target_opens_no_episode(demo_rig, capsys):
    r, demo = demo_rig(nut_in=(9.0, 6.0)); r.world.on_plan = lambda plan: None   # the pre-reset runs but the nut stays on the target
    s, out = r.run(capsys, cycles=1, demo=demo); c, = events(out, "cycle")
    assert c["error"].startswith("nut on the target") and "demo" not in c and s["demos"]["episodes_total"] == 0 and s["demos"]["kept"] == s["demos"]["rejected"] == s["demos"]["aborted"] == 0


def test_unhealthy_recording_rejects_a_landed_pick(demo_rig, dmods, capsys):
    dr = dmods[0]; r, demo = demo_rig(cams=dr.FakeCameraSource(SMALL, band={"top": 20.0})); s, out = r.run(capsys, cycles=1, max_retries=0, demo=demo); c, = events(out, "cycle")
    assert c["ok"] and not c["demo"]["kept"] and c["demo"]["discard_reason"] == "recording_health:banding:top" and s["demos"]["rejected"] == 1


def test_mlflow_gets_the_dataset_link_on_every_cycle_run_and_names_match_the_sidecar(demo_rig, capsys):
    from pickhelpers import Recorder
    r, demo = demo_rig(script=script(3, bad=(2,))); ds_track = Recorder(); demo.track = ds_track; s, out = r.run(capsys, cycles=3, max_retries=0, demo=demo)
    runs = {x.name: x for x in r.track.runs if x.name.startswith("autoloop-t-c")}; assert set(runs) == {"autoloop-t-c1", "autoloop-t-c2", "autoloop-t-c3"}
    p1, p2 = runs["autoloop-t-c1"].params_, runs["autoloop-t-c2"].params_
    assert (p1["kept"], p1["tier"], p1["config_id"], p1["dataset_repo"], p1["dataset"]) == (1, "A", "stock-v1", "lab/nut", "main") and (p2["kept"], p2["discard_reason"], p2["dataset_repo"]) == (0, "outside_radius", "lab/nut-rejects")
    assert runs["autoloop-t-c1"].metrics_["episode_index"] == 0 and runs["autoloop-t-c3"].metrics_["episode_index"] == 1 and runs["autoloop-t-c1"].metrics_["episode_frames"] > 30 and "episode_meta.json" in runs["autoloop-t-c1"].files
    for row in demo.dr.read_sidecar(demo.rec.root): assert row["links"]["mlflow_cycle_run_name"] == f"autoloop-t-c{row['teacher']['cycle']}"
    links = [json.loads(l) for l in open(os.path.join(demo.rec.root, "meta", "lab", "links.jsonl"))]; assert [(x["episode_index"], x["mlflow_cycle_run_name"]) for x in links] == [(0, "autoloop-t-c1"), (1, "autoloop-t-c3")]
    v, = ds_track.runs; assert v.name.startswith("dataset-nut-") and v.tags["kind"] == "dataset-version"
    assert v.params_["episodes_total"] == 2 and v.params_["config_stock-v1"] == 2 and v.params_["fps"] == 30 and v.params_["dataset_hash"] == s["demos"]["dataset_hash"] and v.metrics_["kept"] == 2.0 and v.metrics_["rejected"] == 1.0
    assert {"info.json", "episodes.jsonl"} <= {f.split("/")[-1] for f in v.files}


# ---------------------------------------------------------------- the command line
def run_main(al, tmp_path, capsys, extra, session="cli", cycles=3):
    out_dir = tmp_path / "out"; rc = al.main(["--dry-run", "--cycles", str(cycles), "--seed", "2", "--session", session, "--out-dir", str(out_dir), "--dry-miss", "0", "--settle", "0", "--stop-file", str(tmp_path / "none.stop"), *extra])
    return rc, out_dir, [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]


def test_main_dry_run_record_demos_writes_a_dataset_lerobot_loads_back(al, tmp_path, capsys):
    rc, out_dir, out = run_main(al, tmp_path, capsys, ["--record-demos", "--lighting-tag", "lamp-b"]); s = json.load(open(out_dir / "autoloop_cli.json"))
    assert rc == 0 and s["cycles"] == 3 and s["demos"]["kept"] == 3 and s["demos"]["rejected"] == 0
    root = out_dir / "demos" / "cli" / "so101-nut-pick"; ds = load_back(str(root), "lastlight/so101-nut-pick")
    assert ds.num_episodes == 3 and ds.fps == 30 and len(ds) > 3 * 300 and ds.meta.robot_type == "so_follower"
    assert set(ds.features) >= {"action", "observation.state", "observation.images.top", "observation.images.front", "observation.images.wrist", "observation.environment_state"}
    rows = [json.loads(l) for l in open(root / "meta" / "lab" / "episodes.jsonl")]; assert [x["configuration"]["lighting_tag"] for x in rows] == ["lamp-b"] * 3 and [x["episode_index"] for x in rows] == [0, 1, 2]
    assert len(rows[0]["configuration"]["calibration"]["top_table.json"]["sha12"]) == 12, "the repo's calibration files are hashed into every row"
    assert not (out_dir / "demos" / "cli" / "so101-nut-pick-rejects").exists(), "no rejected try, so no rejects dataset"
    blob = open(root / "meta" / "lab" / "episodes.jsonl").read(); assert "/Users/" not in blob and "@" not in blob


def test_record_demos_needs_dry_run_until_the_camera_tee_exists(al, tmp_path):
    with pytest.raises(SystemExit, match="phase 2"): al.main(["--record-demos", "--cycles", "1", "--out-dir", str(tmp_path)])


def test_with_the_flag_off_the_dry_run_output_is_what_main_printed_before_the_flag_existed(al, tmp_path, capsys):
    out_dir = tmp_path / "gold"; rc = al.main(["--dry-run", "--cycles", "6", "--seed", "11", "--session", "gold", "--out-dir", str(out_dir), "--dry-miss", "0.3", "--settle", "0", "--stop-file", str(tmp_path / "none.stop")])
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    norm = [{k: v for k, v in d.items() if k not in ("cycle_s", "duration_s")} for d in lines]
    assert rc == 0 and (hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest(), len(lines)) == GOLDEN_OFF
    assert not any("demo" in d for d in lines if d.get("event") == "cycle") and "demos" not in json.load(open(out_dir / "autoloop_gold.json")) and not (out_dir / "demos").exists()


def test_a_rig_run_without_a_demo_has_no_demo_keys(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=2); assert "demos" not in s and not any("demo" in c for c in events(out, "cycle")) and r.runner.demo is None
