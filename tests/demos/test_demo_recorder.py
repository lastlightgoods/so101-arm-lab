"""jetson/demo_recorder.py against fakes: real LeRobotDataset v3.0 datasets in tmp_path, synthetic ticks (the pick_seq --tick-sink format) and a FakeCameraSource.
Features, fps, episode counts, keep/discard routing, resume, tick/frame alignment, idle gaps, the sidecar and its crash repair, cadence and camera-freeze rejection."""
import json, os
import numpy as np
import pytest

pytest.importorskip("lerobot")   # why: these tests write and read real LeRobot datasets; CI installs only requirements-dev.txt (no lerobot), so the file skips there
from demohelpers import MOTORS, SMALL, feed_ticks, record_episode


def load_back(root, repo="lab/demo"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo, root=root)


def test_features_are_what_lerobot_record_builds_for_a_so_follower_plus_the_environment_vector(dr, lc):
    from lerobot.utils.feature_utils import hw_to_dataset_features, combine_feature_dicts
    mft = {f"{m}.pos": float for m in MOTORS}   # SOFollower._motors_ft (lerobot/robots/so_follower/so_follower.py:66)
    expected = combine_feature_dicts(hw_to_dataset_features(mft, "action"), hw_to_dataset_features({**mft, **{c: s for c, s in SMALL.items()}}, "observation"))
    f = dr.build_features(SMALL, len(lc.LabConfig().env_vector()))
    for k, v in expected.items(): assert f[k] == v, k
    assert f["action"]["names"] == [f"{m}.pos" for m in MOTORS] and f["observation.state"]["names"] == f["action"]["names"] and f["action"]["dtype"] == "float32" and f["action"]["shape"] == (6,)
    assert {k for k in f if k.startswith("observation.images.")} == {"observation.images.top", "observation.images.front", "observation.images.wrist"} and all(f[k]["dtype"] == "video" for k in f if "images" in k)
    assert f["observation.environment_state"]["shape"] == (5,) and f["observation.environment_state"]["dtype"] == "float32" and set(f) == set(expected) | {"observation.environment_state"}


def test_a_kept_episode_loads_back_with_the_right_features_fps_and_counts(rig, dr, lc):
    rec, buf, cams = rig(); res = record_episode(rec, buf); assert res["dataset"] == "main" and res["episode_index"] == 0 and res["frames"] > 30
    res2 = record_episode(rec, buf, t0=2000.0, base=1000.0); assert res2["episode_index"] == 1
    rec.close(); ds = load_back(rec.root)
    assert ds.fps == 30 and ds.num_episodes == 2 and len(ds) == res["frames"] + res2["frames"] and ds.meta.info["codebase_version"] == "v3.0" and ds.meta.robot_type == "so_follower"
    assert set(ds.features) >= {"action", "observation.state", "observation.images.top", "observation.images.front", "observation.images.wrist", "observation.environment_state"}
    fr = ds[0]; assert tuple(fr["action"].shape) == (6,) and tuple(fr["observation.environment_state"].shape) == (5,) and fr["task"] == dr.TASK
    assert np.allclose(fr["observation.environment_state"].numpy(), lc.LabConfig().env_vector()), "gripper one-hot, arm-version one-hot, jaw_max/100"


def test_dataset_timestamps_are_index_over_fps_and_actions_come_from_the_ticks(rig):
    rec, buf, _ = rig(); res = record_episode(rec, buf); rec.close(); ds = load_back(rec.root); n = res["frames"]
    ts = [float(ds.hf_dataset[i]["timestamp"]) for i in range(n)]; assert ts == pytest.approx([i / 30 for i in range(n)], abs=1e-6)
    a = np.array([ds.hf_dataset[i]["action"] for i in range(n)]); seq = np.round(a[:, 0] - 0).astype(int)   # action[0] = seq
    assert np.all(np.diff(seq) >= 0) and (np.diff(seq) == 0).sum() <= 2 and seq[0] >= 1 and np.allclose(a[:, 1] - a[:, 0], 0.1, atol=1e-4), "every action row is one of the synthesised ticks, in order"


def test_discard_goes_to_the_separate_rejects_dataset_and_leaves_main_alone(rig):
    rec, buf, _ = rig(); k = record_episode(rec, buf); r = record_episode(rec, buf, t0=2000.0, keep=False, verdict={"tier": None, "keep": False, "discard_reason": "outside_radius"}, base=500.0)
    assert (k["dataset"], r["dataset"]) == ("main", "rejects") and rec.counts == {"kept": 1, "rejected": 1, "aborted": 0, "empty": 0}; rec.close()
    assert rec.rejects_root != rec.root and load_back(rec.root).num_episodes == 1 and load_back(rec.rejects_root, "lab/demo-rejects").num_episodes == 1
    side = [json.loads(l) for l in open(os.path.join(rec.rejects_root, "meta", "lab", "episodes.jsonl"))]
    assert side[0]["kept"] is False and side[0]["verdict"]["discard_reason"] == "outside_radius" and not os.path.exists(os.path.join(rec.root, "meta", "lab", "frames", "episode_000001.json"))


def test_abort_saves_nothing_anywhere_and_leaves_no_spool_files(rig, tmp_path):
    rec, buf, _ = rig(); rec.open({}, 1000.0); feed_ticks(buf, 1000.2, 60)
    for k in range(40): rec.pump(1000.2 + k / 40)
    assert rec.ep.frames and any(os.scandir(rec.ep.spool)); assert rec.abort("trip:workspace") is True
    assert not os.path.exists(rec.root) and not os.path.exists(rec.rejects_root) and rec.counts["aborted"] == 1 and os.listdir(tmp_path / "spool") == []
    assert rec.abort("again") is False


def test_a_committed_episode_leaves_no_images_or_spool_behind(rig, tmp_path):
    rec, buf, _ = rig(); record_episode(rec, buf); assert os.listdir(tmp_path / "spool") == []
    rec.close(); assert not any(f.endswith((".png", ".npy", ".jpg")) for _, _, fs in os.walk(rec.root) for f in fs), "lerobot's own temp image folder is empty after save_episode"


def test_resume_appends_to_the_same_dataset(rig, dr, lc, tmp_path):
    rec, buf, _ = rig(); record_episode(rec, buf); rec.close()
    rec2, buf2, _ = rig(); res = record_episode(rec2, buf2, t0=5000.0); assert res["episode_index"] == 1; rec2.close()
    assert load_back(rec2.root).num_episodes == 2 and [r["episode_index"] for r in dr.read_sidecar(rec2.root)] == [0, 1]


def test_resume_refuses_a_dataset_with_other_cameras_or_fps(rig, dr):
    rec, buf, _ = rig(); record_episode(rec, buf); rec.close()
    other = dr.FakeCameraSource({"top": (36, 64, 3)})
    rec2, buf2, _ = rig(cams=other)
    with pytest.raises(ValueError, match="differ"): rec2.dataset("main")
    rec3, _, _ = rig(fps=25)
    with pytest.raises(ValueError, match="fps"): rec3.dataset("main")


def test_ticks_at_40hz_are_resampled_to_30hz_with_the_nearest_previous_tick(rig, dr):
    rec, buf, _ = rig(); rec.open({}, 1000.0); last = feed_ticks(buf, 1000.0, 120, hz=40.0)
    t = 1000.0
    while t <= last: rec.pump(t); t += 1 / 100
    rec.pump(last); fr = rec.ep.frames; assert abs(len(fr) - round(3 * 30)) <= 2, "119/40 s of ticks gives about 89 frames"
    assert all(0 <= f["tick_age_s"] <= 1 / 40 + 1e-9 for f in fr), "the newest tick at or before the boundary is at most one tick (25 ms) old"
    seqs = [f["tick_seq"] for f in fr]; assert seqs == sorted(seqs) and len(set(seqs)) >= len(seqs) - 2 and seqs[-1] - seqs[0] > len(fr), "40 Hz into 30 Hz skips ticks and repeats none (bar the last one, held for 60 ms after the ticks stop)"
    assert all(abs((f["t_sample"] - f["tick_t"]) - f["tick_age_s"]) < 1e-9 for f in fr)
    assert np.allclose(np.diff([f["t_sample"] for f in fr]), 1 / 30), "samples are 1/30 s apart"


def test_idle_gaps_between_stages_are_dropped_not_recorded(rig):
    rec, buf, _ = rig(); rec.open({}, 1000.0)
    a = feed_ticks(buf, 1000.5, 40, seq0=1, wp="s1"); b = feed_ticks(buf, a + 1.5, 40, seq0=41, wp="s2")   # the Mac thinks for 1.5 s between two staged runs
    t = 1000.0
    while t <= b + 0.2: rec.pump(t); t += 1 / 60
    h = rec.end(b + 0.2); fr = rec.ep.frames
    assert h["dropped_idle_frames"] >= 40, "~0.5 s before the first tick, ~1.5 s between stages, ~0.2 s after"
    assert not any(a + 0.062 < f["t_sample"] < a + 1.5 for f in fr), "no frame is recorded in the gap"
    assert {f["wp"] for f in fr} == {"s1", "s2"} and all(f["tick_age_s"] <= rec.max_tick_age for f in fr)
    n = len(fr); res = rec.commit(True, {"tier": "A"}); assert res["frames"] == n, "and the dataset has no placeholder frames for the gap"


def test_frame_log_aligns_ticks_and_camera_frames_by_timestamp(rig):
    rec, buf, cams = rig(); res = record_episode(rec, buf); rec.close()
    log = json.load(open(os.path.join(rec.root, "meta", "lab", "frames", "episode_000000.json"))); n = res["frames"]
    assert len(log["t_sample"]) == n == len(log["tick_seq"]) and set(log["cam_age_s"]) == {"top", "front", "wrist"}
    for i in range(n):
        assert log["tick_t"][i] <= log["t_sample"][i] + 1e-9 and log["t_sample"][i] - log["tick_t"][i] == pytest.approx(log["tick_age_s"][i])
        for c in cams.names: assert 0 <= log["cam_age_s"][c][i] <= 1 / 30 + cams.latency + 1e-9, "a camera frame is never from the future and at most one camera period (+ latency) old"
    assert np.allclose(np.diff(log["t_sample"][:20]), 1 / 30)


def test_the_fake_camera_frame_matches_its_capture_time(dr):
    cams = dr.FakeCameraSource(SMALL, cam_fps=30.0, latency_s=0.0)
    img, tc = cams.frame_at("top", 10.0 + 0.5 / 30); assert tc == pytest.approx(10.0) and img.shape == (36, 64, 3) and img.dtype == np.uint8
    assert int(img[0, 0, 2]) == int(round(10.0 * 30)) % 256, "the blue plane carries the capture frame number"


def test_sidecar_row_has_every_configuration_label_and_the_mlflow_key(rig, dr, lc):
    cfg = lc.LabConfig(config_id="stock-v1", gripper_type="stock", arm_version="stock", robot_id="ll_follower", servo_set="STS3215 x6", camera_mount="wrist-mount-r1", lighting_tag="lamp-b")
    rec, buf, _ = rig(cfg=cfg); res = record_episode(rec, buf, meta={"scene": {"start_in": [6, 8], "target_in": [9, 6]}, "teacher": {"session": "s1", "cycle": 3, "jaw_check": [{"in_jaws": True}]},
                                                                      "links": {"mlflow_cycle_run_name": "autoloop-s1-c3", "mlflow_session_run_id": "abc"}}, verdict={"tier": "R", "keep": True, "reasons": []})
    rec.close(); row, = dr.read_sidecar(rec.root); c = row["configuration"]
    for k in ("config_id", "gripper_type", "gripper_variant", "arm_version", "robot_id", "servo_set", "camera_mount", "lighting_tag", "jaw_max_mm", "calibration"): assert k in c, k
    assert (c["gripper_type"], c["arm_version"], c["robot_id"], c["lighting_tag"]) == ("stock", "stock", "ll_follower", "lamp-b") and c["calibration"]["top_table.json"]["sha12"] == "abc123def456"
    assert row["links"] == {"mlflow_cycle_run_name": "autoloop-s1-c3", "mlflow_session_run_id": "abc"} and row["teacher"]["cycle"] == 3 and row["scene"]["target_in"] == [9, 6]
    assert row["episode_index"] == 0 and row["kept"] is True and row["quality"]["tier"] == "R" and row["quality"]["frames"] == res["frames"] == row["n_frames"] and row["fps"] == 30
    assert {"over_budget_pct", "dropped_idle_frames", "tick_age_p95_s", "cam_age_max_s", "banding"} <= set(row["quality"])
    assert not any("/Users/" in json.dumps(row) or "@" in json.dumps(row) for _ in [0]), "no home paths or emails in a sidecar that gets published"


def test_a_crash_between_save_episode_and_the_sidecar_write_is_repaired_on_restart(rig, dr, monkeypatch):
    rec, buf, _ = rig(); record_episode(rec, buf)
    real = dr.append_jsonl; calls = []
    def dying(path, row): calls.append(path); raise KeyboardInterrupt("power cut")
    monkeypatch.setattr(dr, "append_jsonl", dying)
    with pytest.raises(KeyboardInterrupt): record_episode(rec, buf, t0=3000.0, base=300.0)
    monkeypatch.setattr(dr, "append_jsonl", real); rec._ds["main"].finalize(); rec._ds.clear()
    assert [r["episode_index"] for r in dr.read_sidecar(rec.root)] == [0]
    rec2, buf2, _ = rig(); ds = rec2.dataset("main"); assert ds.meta.total_episodes == 2 and rec2.reconciled["main"] == {"added": [1], "dropped": []}
    rows = dr.read_sidecar(rec2.root); assert [r["episode_index"] for r in rows] == [0, 1] and "reconciled" in rows[1]
    assert record_episode(rec2, buf2, t0=9000.0)["episode_index"] == 2


def test_reconcile_drops_rows_beyond_the_dataset_count(dr, tmp_path):
    root = str(tmp_path / "ds"); [dr.append_jsonl(os.path.join(dr.sidecar_dir(root), "episodes.jsonl"), {"episode_index": i}) for i in range(3)]
    assert dr.reconcile_sidecar(root, 2) == {"added": [], "dropped": [2]} and [r["episode_index"] for r in dr.read_sidecar(root)] == [0, 1]


def test_late_sampling_makes_the_episode_unhealthy(rig):
    rec, buf, _ = rig(); rec.open({}, 1000.0); last = feed_ticks(buf, 1000.0, 120)
    t = 1000.0
    while t <= last: rec.pump(t + 0.1); t += 0.1   # the sampler wakes every 100 ms: every frame is ~3 periods late
    h = rec.end(last + 0.1); assert not h["ok"] and h["over_budget_pct"] > 5 and any(r.startswith("cadence") for r in h["reasons"])


def test_a_frozen_camera_makes_the_episode_unhealthy(rig, dr):
    cams = dr.FakeCameraSource(SMALL, freeze=[("front", 1001.0, 1002.0)]); rec, buf, _ = rig(cams=cams); rec.open({}, 1000.0); last = feed_ticks(buf, 1000.0, 120)
    t = 1000.0
    while t <= last: rec.pump(t); t += 1 / 30
    h = rec.end(last); assert not h["ok"] and h["cam_age_max_s"]["front"] > 0.25 and "camera_frozen:front" in h["reasons"] and h["cam_age_max_s"]["top"] < 0.1


def test_a_300ms_freeze_in_the_budget_boundary_is_caught_but_a_200ms_one_is_not(rig, dr):
    for dur, bad in ((0.30, True), (0.20, False)):
        cams = dr.FakeCameraSource(SMALL, freeze=[("top", 1001.0, 1001.0 + dur)]); rec, buf, _ = rig(cams=cams); rec.open({}, 1000.0); last = feed_ticks(buf, 1000.0, 120)
        t = 1000.0
        while t <= last: rec.pump(t); t += 1 / 30
        assert (not rec.end(last)["ok"]) is bad; rec.abort("test")


def test_banding_over_the_limit_on_top_or_front_is_unhealthy_but_not_on_the_wrist(rig, dr):
    for band, bad in (({"top": 3.3, "front": 2.2, "wrist": 33.0}, False), ({"top": 12.0}, True), ({"front": 9.0}, True)):
        cams = dr.FakeCameraSource(SMALL, band=band); rec, buf, _ = rig(cams=cams); rec.open({}, 1000.0); last = feed_ticks(buf, 1000.0, 80)
        t = 1000.0
        while t <= last: rec.pump(t); t += 1 / 30
        h = rec.end(last); assert h["ok"] is not bad, (band, h["reasons"]); rec.abort("test")


def test_too_short_an_episode_is_unhealthy_and_an_empty_one_is_written_nowhere(rig):
    rec, buf, _ = rig(); rec.open({}, 1000.0); h = rec.end(1000.5); assert not h["ok"] and h["frames"] == 0 and h["reasons"][0].startswith("too_few_frames")
    assert rec.commit(True, {})["dataset"] is None and not os.path.exists(rec.root) and rec.counts["empty"] == 1


def test_open_twice_is_refused_and_commit_needs_end(rig):
    rec, buf, _ = rig(); rec.open({}, 1000.0)
    with pytest.raises(RuntimeError): rec.open({}, 1001.0)
    with pytest.raises(RuntimeError): rec.commit(True, {})
    rec.abort("x"); rec.open({}, 1002.0)   # after an abort a new one may start


def test_dataset_hash_is_stable_and_changes_with_content(rig, dr):
    rec, buf, _ = rig(); record_episode(rec, buf); rec.close(); h1 = dr.dataset_hash(rec.root); assert h1 == dr.dataset_hash(rec.root)
    dr.append_jsonl(os.path.join(dr.sidecar_dir(rec.root), "links.jsonl"), {"x": 1}); assert dr.dataset_hash(rec.root) == h1, "the append-only links file is outside the hash"
    dr.append_jsonl(os.path.join(dr.sidecar_dir(rec.root), "episodes.jsonl"), {"episode_index": 9}); assert dr.dataset_hash(rec.root) != h1


def test_lab_config_labels_hashes_and_validation(lc, tmp_path):
    c = lc.LabConfig(); assert c.env_vector() == [1.0, 0.0, 1.0, 0.0, 0.0] and len(c.env_vector()) == len(lc.ENV_NAMES)
    p = lc.LabConfig(gripper_type="robonine_parallel", arm_version="special_edition_1", jaw_max_mm=45.0).env_vector(); assert p == [0.0, 1.0, 0.0, 1.0, 0.45]
    with pytest.raises(ValueError): lc.LabConfig.from_dict({"gripper_type": "claw"})
    with pytest.raises(ValueError, match="unknown"): lc.LabConfig.from_dict({"gripperr": "stock"})
    d = tmp_path / "calib"; d.mkdir(); (d / "top_table.json").write_text("{}"); live = lc.calib_hashes(str(d)); assert set(live) == {"top_table.json"} and len(live["top_table.json"]["sha12"]) == 12
    pinned = lc.LabConfig(pinned_hashes={"top_table.json": live["top_table.json"]["sha12"]}); assert pinned.check_hashes(live) == []
    (d / "top_table.json").write_text('{"changed": 1}'); bad = pinned.check_hashes(lc.calib_hashes(str(d))); assert len(bad) == 1 and bad[0].startswith("top_table.json: pinned")
    assert pinned.check_hashes({}) == ["top_table.json: pinned " + live["top_table.json"]["sha12"] + ", live None"] and lc.LabConfig().check_hashes({}) == []
    f = tmp_path / "cfg.json"; f.write_text(json.dumps(pinned.to_dict())); assert lc.LabConfig.from_file(str(f)) == pinned, "label round trip"
    assert "pinned_hashes" not in pinned.sidecar_labels(live) and pinned.sidecar_labels(live)["calibration"]["top_table.json"]["sha12"]


def test_the_recorder_encodes_with_its_configured_codec_and_defaults_to_svtav1(rig, dr):
    import inspect, json
    assert inspect.signature(dr.DemoRecorder).parameters["vcodec"].default == "libsvtav1" == dr.DEFAULT_VCODEC
    rec, buf, cams = rig(); assert rec.vcodec == "h264" and rec._encoder().vcodec == "h264"   # the rig fixture passes h264 (SVT-AV1 crashes Python at teardown on macOS)
    record_episode(rec, buf); rec.close(); info = load_back(rec.root).meta.info   # why read back: meta/info.json records the codec the video was really encoded with
    assert {f["info"]["video.codec"] for f in info.features.values() if f["dtype"] == "video"} == {"h264"}


def test_a_dataset_moved_away_while_open_is_not_rebuilt_by_close_and_the_next_commit_still_lands(rig, tmp_path):
    """2026-11-11 21:35: pilot 1's dataset was moved aside while the recorder still held it; configure (close -> finalize) wrote a stray
    meta/episodes tree at the old path and pilot 2's commit then died in LeRobotDataset.create with FileExistsError."""
    rec, buf, _ = rig(); record_episode(rec, buf); os.rename(rec.root, str(tmp_path / "junk"))
    rec.close(); import gc; gc.collect()
    assert not os.path.exists(rec.root), "finalize must not re-create a dataset folder that was moved away"
    rec2, buf2, _ = rig(); res = record_episode(rec2, buf2); assert res["dataset"] == "main" and res["episode_index"] == 0
    rec2.close(); assert load_back(rec2.root).num_episodes == 1


def test_a_stray_non_dataset_folder_is_moved_aside_not_deleted_and_the_sidecar_links_carry_over(rig, dr, tmp_path):
    rec, buf, _ = rig(); stray = os.path.join(rec.root, "meta", "episodes", "chunk-000"); os.makedirs(stray); open(os.path.join(stray, "file-000.parquet"), "wb").write(b"x")
    rec.append_link("main", {"episode_index": 0, "mlflow_cycle_run_id": "r1"})
    res = record_episode(rec, buf); assert res["dataset"] == "main" and res["episode_index"] == 0
    aside = [p for p in os.listdir(tmp_path / "data") if p.startswith("demo.stray-")]; assert len(aside) == 1
    assert os.path.exists(tmp_path / "data" / aside[0] / "meta" / "episodes" / "chunk-000" / "file-000.parquet"), "moved aside, not deleted"
    assert [json.loads(l)["mlflow_cycle_run_id"] for l in open(os.path.join(dr.sidecar_dir(rec.root), "links.jsonl"))] == ["r1"]
    rec.close(); assert load_back(rec.root).num_episodes == 1


def test_a_stray_folder_holding_only_the_sidecar_is_removed_after_the_sidecar_moves(dr, tmp_path):
    root = str(tmp_path / "ds"); dr.append_jsonl(os.path.join(dr.sidecar_dir(root), "links.jsonl"), {"a": 1})
    stray = dr.move_aside(root); assert stray and not os.path.exists(root); os.makedirs(os.path.join(root, "meta"))
    dr.restore_sidecar(stray, root); assert not os.path.exists(stray) and os.path.exists(os.path.join(dr.sidecar_dir(root), "links.jsonl"))
    assert dr.move_aside(str(tmp_path / "absent")) is None
