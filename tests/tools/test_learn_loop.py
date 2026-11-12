"""tools/learn_loop.py: record -> train -> register -> deploy -> evaluate -> report, every stage in dry-run on the fakes of tools/learn_loop_fakes.py (a tiny dataset, a fake lerobot-train
writing a tiny checkpoint, the registry on a private SQLite store, a local directory standing in for the Jetson, a fake rollout). Resume after a failed stage, force/invalidation,
and the MLflow chain eval trial -> registry version -> train run -> dataset hash. No arm, no ssh, no camera, no GPU, no lerobot-train."""
import json, os, re, shutil, sys
import pytest
from pickhelpers import ROOT

from learn_loop_helpers import LL, FF, EV, PRE, run, state, client, events, lrt  # noqa: F401


# ---------------------------------------------------------------- pure pieces
def test_plan_prints_etas_from_the_measured_rates(tmp_path, capsys):
    assert run(tmp_path, "plan", extra=("--dry-run", "--cycles", "30", "--steps", "20000", "--trials", "10")) == 0
    out = capsys.readouterr().out
    assert "record" in out and "train" in out and "evaluate" in out and "1.45 step/s at batch 8" in out
    est = LL.estimates(LL.build_parser().parse_args(["plan", "--name", "x", "--steps", "20000", "--cycles", "30", "--trials", "10"]))
    assert est["train"] == pytest.approx(20000 / 1.45, rel=0.01) and est["record"] == pytest.approx(30 * 86) and est["evaluate"] == pytest.approx(10 * 70)


def test_train_argv_has_the_features_without_environment_state_and_no_machine_paths():
    a = LL.build_parser().parse_args(["train", "--name", "x"]); info = {"fps": 30, "total_episodes": 12, "total_frames": 5000, "features": FF.FEATURES}; cfg = LL.train_cfg(info, a)
    assert cfg["images"] == ["front", "top", "wrist"] and cfg["dropped_features"] == ["observation.environment_state"]
    argv = LL.train_argv(cfg, "lab/nut", "/data/nut", "/out/job", "job", a, binary="lerobot-train"); flat = " ".join(argv)
    feats = json.loads(next(x for x in argv if x.startswith("--policy.input_features="))[len("--policy.input_features="):])
    assert "observation.environment_state" not in feats and feats["observation.images.wrist"]["shape"] == [3, 480, 640] and feats["observation.state"] == {"type": "STATE", "shape": [6]}
    assert "--policy.type=act" in argv and "--policy.device=mps" in argv and "--dataset.repo_id=lab/nut" in argv and "--dataset.root=/data/nut" in argv and "--wandb.enable=false" in argv and "--batch_size=8" in argv
    assert "/Users/" not in flat
    assert LL.resume_argv("/out/job", "lerobot-train") == ["lerobot-train", "--resume=true", "--config_path=/out/job/checkpoints/last/pretrained_model/train_config.json"]


def test_train_cfg_features_equal_what_lerobot_derives_from_the_dataset():
    ft = pytest.importorskip("lerobot.utils.feature_utils"); a = LL.build_parser().parse_args(["train", "--name", "x"]); cfg = LL.train_cfg({"fps": 30, "features": FF.FEATURES}, a)
    ref = ft.dataset_to_policy_features(FF.FEATURES)
    for k, v in cfg["input_features"].items(): assert (ref[k].type.value, tuple(ref[k].shape)) == (v["type"], tuple(v["shape"])), k
    assert set(ref) - set(cfg["input_features"]) - {"action"} >= {"observation.environment_state"}


def test_a_camera_set_the_jetson_cannot_serve_is_refused():
    a = LL.build_parser().parse_args(["train", "--name", "x"]); feats = {k: v for k, v in FF.FEATURES.items() if k != "observation.images.front"}; feats["observation.images.side"] = FF.FEATURES["observation.images.front"]
    with pytest.raises(LL.StageError, match="no camera set"): LL.train_cfg({"fps": 30, "features": feats}, a)
    ok = {k: v for k, v in FF.FEATURES.items() if k != "observation.images.front"}   # top + wrist is the old 2-camera set (SO101_CAMS_2)
    assert LL.train_cfg({"fps": 30, "features": ok}, a)["images"] == ["top", "wrist"]


def test_parse_log_reads_lerobots_lines_and_the_last_rate():
    pts, rate = LL.parse_log("Training:  1%| | 200/25000 [02:10<4:19:25,  1.59step/s]INFO x step:200 loss:6.988 lr:1.0e-05\rTraining: 100%| | 1.45step/s]INFO step:25K smpl:200K loss:0.117 grdn:1")
    assert pts == [(200, 6.988), (25000, 0.117)] and rate == 1.45


# ---------------------------------------------------------------- each stage in dry-run
def test_every_stage_in_dry_run_one_by_one(tmp_path, capsys):
    fs = FF.FakeSys(kept=8)
    for s in PRE:
        assert run(tmp_path, s, sys_=fs) == 0, s
        assert state(tmp_path)["stages"][s]["status"] == "done"
    st = state(tmp_path)["stages"]; d = tmp_path / "ll" / "t1"
    rec, tr, reg = st["record"]["result"], st["train"]["result"], st["register"]["result"]
    assert rec["episodes"] == 8 and len(rec["dataset_hash"]) == 64 and os.path.exists(rec["dataset_root"] + "/meta/info.json") and rec["baseline"]["success_rate"] == 0.6 and rec["dataset_run"]
    argv = next(c for c in fs.calls if "lerobot-train" in str(c[0])); assert f"--dataset.root={rec['dataset_root']}" in argv and "--steps=1000" in argv
    assert tr["steps"] == 1000 and tr["final_loss"] is not None and os.path.exists(tr["checkpoint"] + "/model.safetensors") and tr["images"] == ["front", "top", "wrist"]
    assert reg["name"] == "so101-nutpick-dry" and reg["version"] == "1" and os.path.exists(d / "jetson_home" / "so101" / "policies" / reg["name"] / "current" / "model.safetensors")
    dep = json.load(open(d / "jetson_home" / "so101" / "policies" / "DEPLOYED.json")); assert dep["name"] == reg["name"] and dep["version"] == 1 and dep["sha256"] == reg["sha256"]
    assert state(tmp_path)["stages"]["evaluate"]["status"] == "pending"


def test_record_needs_enough_kept_episodes_and_names_the_fix(tmp_path, capsys):
    assert run(tmp_path, "record", sys_=FF.FakeSys(kept=1), extra=("--dry-run", "--cycles", "3", "--min-kept", "5")) == 1
    s = state(tmp_path)["stages"]["record"]; assert s["status"] == "failed" and "only 1 kept episodes" in s["error"] and "--min-kept 5" in s["error"]


def test_record_detects_a_dataset_changed_after_the_recorder_hashed_it(tmp_path):
    class Liar(FF.FakeSys):
        def do_record(self, argv, log, part):
            rc = super().do_record(argv, log, part); out = FF.argval(argv, "--out-dir"); p = os.path.join(out, f"autoloop_{FF.argval(argv, '--session')}.json"); j = json.load(open(p)); j["demos"]["dataset_hash"] = "ab" * 32; json.dump(j, open(p, "w")); return rc
    assert run(tmp_path, "record", sys_=Liar()) == 1 and "hash changed" in state(tmp_path)["stages"]["record"]["error"]


def test_adopting_an_existing_dataset_skips_the_recorder(tmp_path):
    ds = FF.make_dataset(str(tmp_path / "mine"), episodes=5); fs = FF.FakeSys()
    assert run(tmp_path, "record", sys_=fs, extra=("--dry-run", "--adopt-dataset", ds)) == 0 and fs.calls == []
    assert state(tmp_path)["stages"]["record"]["result"]["episodes"] == 5 and state(tmp_path)["stages"]["record"]["result"]["dataset_root"] == ds


def test_a_stage_that_needs_an_earlier_one_says_which(tmp_path, capsys):
    assert run(tmp_path, "train") == 1 and "stage `record` has not finished" in state(tmp_path)["stages"]["train"]["error"]


# ---------------------------------------------------------------- resume
def test_a_failed_train_resumes_from_its_last_checkpoint_without_redoing_record(tmp_path, capsys):
    fs = FF.FakeSys(fail={"train": 1})
    assert run(tmp_path, "record", "train", sys_=fs) == 1
    st = state(tmp_path)["stages"]; assert st["record"]["status"] == "done" and st["train"]["status"] == "failed" and "reached step 500 of 1000" in st["train"]["error"] and "repeat the command" in st["train"]["error"]
    assert run(tmp_path, *PRE, sys_=fs) == 0
    kinds = [("record" if any(str(x).endswith("autoloop.py") for x in c) else "train") for c in fs.calls]
    assert kinds == ["record", "train", "train"], "record ran once; train twice"
    again = fs.calls[2]; assert "--resume=true" in again and any(x.startswith("--config_path=") and x.endswith("checkpoints/last/pretrained_model/train_config.json") for x in again)
    st = state(tmp_path)["stages"]; assert st["train"]["status"] == "done" and st["train"]["result"]["steps"] == 1000 and st["train"]["result"]["resumed"] is True
    assert st["register"]["status"] == "done" and st["deploy"]["status"] == "done"


def test_done_stages_are_skipped_and_force_invalidates_what_follows(tmp_path, capsys):
    fs = FF.FakeSys(); assert run(tmp_path, *PRE, sys_=fs) == 0; n = len(fs.calls)
    capsys.readouterr(); assert run(tmp_path, *PRE, sys_=fs) == 0 and len(fs.calls) == n and capsys.readouterr().out.count("skipping") == 4
    assert run(tmp_path, "train", "--force", sys_=fs) == 0 and len(fs.calls) == n + 1
    st = state(tmp_path)["stages"]; assert st["train"]["status"] == "done" and st["register"]["status"] == "pending" and st["register"]["invalidated_by"] == "train" and st["deploy"]["status"] == "pending"


def test_a_crash_after_register_does_not_make_a_second_version(tmp_path):
    fs = FF.FakeSys(); assert run(tmp_path, "record", "train", "register", sys_=fs) == 0
    assert run(tmp_path, "register", "--force", sys_=fs) == 0
    vs = client(tmp_path).search_model_versions("name='so101-nutpick-dry'"); assert len(vs) == 1


def test_the_environment_is_left_as_found(tmp_path, monkeypatch):
    for k in ("MLFLOW_TRACKING_URI", "SO101_MLFLOW_URI", "SO101_DEPLOY_LOCAL_ROOT", "SO101_MLFLOW_ARTIFACTS"): monkeypatch.delenv(k, raising=False)
    before = dict(os.environ); assert run(tmp_path, "record", "train") == 0; assert dict(os.environ) == before


# ---------------------------------------------------------------- MLflow linkage
def test_the_chain_dataset_hash_train_run_registry_eval_trials(tmp_path, capsys, lrt):
    fs = FF.FakeSys(kept=8); assert run(tmp_path, *PRE, sys_=fs) == 0
    st = state(tmp_path)["stages"]; h = st["record"]["result"]["dataset_hash"]; c = client(tmp_path)
    # dataset-version run (the recorder's) -> carries the hash the loop computed
    ds = c.search_runs([c.get_experiment_by_name("so101-datasets-dry").experiment_id], f"params.dataset_hash = '{h}'"); assert len(ds) == 1 and st["record"]["result"]["dataset_run"] == ds[0].info.run_id
    # train run -> hash, dataset run, parent
    tr = c.get_run(st["train"]["result"]["train_run"]); assert tr.data.params["dataset_hash"] == h and tr.data.tags["dataset_hash"] == h and tr.data.tags["dataset_run_id"] == ds[0].info.run_id and tr.data.tags["registry_version"] == "1"
    assert tr.info.status == "FINISHED" and "final_loss" in tr.data.metrics and float(tr.data.params["steps"]) == 1000
    # registry version -> hash and train run; its run carries the metrics
    mv = c.get_model_version("so101-nutpick-dry", "1"); assert mv.tags["dataset_hash"] == h and mv.tags["train_run_id"] == tr.info.run_id and mv.tags["sha256"] == st["register"]["result"]["sha256"]
    rr = c.get_run(mv.run_id); assert rr.data.metrics["final_loss"] == pytest.approx(tr.data.metrics["final_loss"]) and rr.data.metrics["episodes"] == 8 and rr.data.metrics["scripted_success_rate"] == 0.6
    # evaluate -> every trial run carries the whole chain
    rt = lrt(["ok", "ok", "miss"]); assert run(tmp_path, "evaluate", sys_=fs, runtime=rt) == 0
    ev = c.search_runs([c.get_experiment_by_name("so101-learn-loop-dry").experiment_id], "tags.kind = 'eval-trial'"); assert len(ev) == 3
    for r in ev:
        t = r.data.tags; assert (t["dataset_hash"], t["train_run_id"], t["policy_version"], t["registry_run_id"], t["policy_name"]) == (h, tr.info.run_id, "1", mv.run_id, "so101-nutpick-dry")
    evs = c.search_runs([c.get_experiment_by_name("so101-learn-loop-dry").experiment_id], "tags.kind = 'eval-session'"); assert all(r.data.tags["mlflow.parentRunId"] == evs[0].info.run_id for r in ev); assert len(evs) == 1 and evs[0].data.metrics["success_rate"] == pytest.approx(2 / 3, abs=0.01)
    # the report links the loop's parent run to the numbers
    assert run(tmp_path, "report", sys_=fs) == 0
    par = c.get_run(state(tmp_path)["ids"]["parent"]); assert par.data.params["dataset_hash"] == h and par.data.params["policy_version"] == "1" and par.data.metrics["eval_success_rate"] == pytest.approx(2 / 3, abs=0.01) and par.data.metrics["scripted_success_rate"] == 0.6
    assert (tmp_path / "ll" / "t1" / "report.md").exists() and "scripted, this loop's session ll-t1" in (tmp_path / "ll" / "t1" / "report.md").read_text()


# ---------------------------------------------------------------- report
def test_report_compares_with_the_scripted_baseline_and_refuses_an_unfinished_evaluation(tmp_path, capsys, lrt):
    fs = FF.FakeSys(kept=8, success_rate=0.5); assert run(tmp_path, *PRE, sys_=fs) == 0
    assert run(tmp_path, "report", sys_=fs) == 1 and "evaluate has not finished" in state(tmp_path)["stages"]["report"]["error"]
    assert run(tmp_path, "evaluate", sys_=fs, runtime=lrt(["ok"] * 3)) == 0 and run(tmp_path, "report", sys_=fs) == 0
    r = json.load(open(tmp_path / "ll" / "t1" / "report.json")); assert r["policy_eval"]["success_rate"] == 1.0 and r["baseline_session"]["success_rate"] == 0.5 and r["baseline_session"]["cycles"] == 6
    assert r["policy_eval"]["ci95"][0] == pytest.approx(0.439, abs=0.01) and "not distinguishable" in r["baseline_session"]["verdict"] and r["partial"] is False
    assert r["policy_eval"]["median_placed_err_mm"] is not None


def test_wilson_interval_is_the_textbook_one():
    ev = EV
    assert ev.wilson(0, 0) == (None, None) and ev.wilson(5, 10) == (0.237, 0.763) and ev.wilson(10, 10)[1] == 1.0 and ev.wilson(0, 10)[0] == 0.0


# ---------------------------------------------------------------- the whole thing, and hygiene
def test_all_stages_in_one_dry_run_command(tmp_path, capsys, cc):
    assert run(tmp_path, "all", extra=("--dry-run", "--cycles", "6", "--trials", "3", "--steps", "500", "--save-freq", "250", "--dry-script", "ok", "ok", "ok")) == 0
    st = state(tmp_path)["stages"]; assert all(st[s]["status"] == "done" for s in LL.STAGES) and st["evaluate"]["result"]["landed"] == 3
    assert "# learn-loop t1" in capsys.readouterr().out


def test_new_files_carry_no_home_paths_lan_addresses_or_emails():
    bad = re.compile(r"/Users/|/home/\w|\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+(?:\.\d+)?\b|[\w.+-]+@[\w-]+\.[a-z]{2,}")
    for rel in ("tools/learn_loop.py", "tools/learn_loop_eval.py", "tools/learn_loop_fakes.py", "docs/learn-loop.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p): continue
        for i, line in enumerate(open(p), 1): assert not bad.search(line), f"{rel}:{i}: {line.strip()[:100]}"
