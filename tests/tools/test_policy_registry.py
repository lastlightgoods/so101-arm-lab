"""so101-policy register / list against a temp SQLite MLflow store: run params (the mlflow_backfill fields), tags, the checkpoint as a run
artifact, a model version with alias `candidate`, and the --from-hf path with a fake snapshot (HF_HUB_OFFLINE=1 throughout)."""
import json, os, shutil
import pytest
from toolhelpers import make_ckpt

mlflow = pytest.importorskip("mlflow")


def test_register_logs_params_tags_artifact_and_candidate_alias(lab):
    out = lab.run("register", lab.ckpt, "--name", "so101-fake", "--hf-job", "6ad8776800", "--note", "first cut")
    assert "registered so101-fake v1" in out and "alias candidate" in out
    c = lab.client(); mv = c.get_model_version("so101-fake", "1"); run = c.get_run(mv.run_id)
    p = run.data.params   # why: the same keys mlflow_backfill.params_for writes, so registry and training runs compare in one table
    assert p["policy"] == "act" and p["chunk_size"] == "100" and p["lr"] == "1e-05" and p["batch_size"] == "8" and p["steps"] == "5000" and p["seed"] == "1000"
    assert p["dataset"] == "lastlight/fake-set" and p["warm_start"] == "scratch" and p["device"] == "mps" and p["use_vae"] == "True"
    t = run.data.tags
    assert t["hf_job"] == "6ad8776800" and t["dataset"] == "lastlight/fake-set" and t["note"] == "first cut" and t["checkpoint_step"] == "005000" and t["host"] and t["git_commit"]
    assert t["sha256"] == lab.sha(os.path.join(lab.ckpt, "model.safetensors")) and t["source_dir"] == os.path.abspath(lab.ckpt)
    art = os.path.join(run.info.artifact_uri.replace("file://", ""), "model")
    assert sorted(os.listdir(art)) == ["config.json", "model.safetensors", "policy_preprocessor.json", "train_config.json"]
    assert mv.tags["sha256"] == t["sha256"] and mv.tags["hf_job"] == "6ad8776800" and mv.run_id == run.info.run_id and mv.description == "first cut"
    assert lab.tool.aliases_of(c, "so101-fake") == {"candidate": "1"}
    assert run.info.experiment_id == mlflow.get_experiment_by_name("so101-registry").experiment_id
    assert mlflow.get_experiment_by_name("so101-registry").artifact_location.startswith(os.environ["SO101_MLFLOW_ARTIFACTS"])


def test_second_register_makes_v2_and_moves_candidate(lab):
    lab.run("register", lab.ckpt, "--name", "so101-fake")
    ck2 = make_ckpt(os.path.join(lab.tmp, "train", "act_fake2"), seed=2)
    out = lab.run("register", ck2, "--name", "so101-fake", "--hf-repo", "tclastlight/fake_repo", "--dataset", "lastlight/other")
    assert "registered so101-fake v2" in out
    c = lab.client(); mv = c.get_model_version("so101-fake", "2")
    assert lab.tool.aliases_of(c, "so101-fake") == {"candidate": "2"} and mv.tags["hf_repo"] == "tclastlight/fake_repo" and mv.tags["dataset"] == "lastlight/other"
    out = lab.run("list")
    assert "so101-fake  (2 versions; candidate -> v2)" in out and "v2   [candidate]" in out and "from tclastlight/fake_repo" in out and "v1  " in out
    assert lab.run("list", "so101-fake").count("\n") == 3


def test_register_refuses_an_incomplete_dir(lab, tmp_path):
    d = tmp_path / "half"; d.mkdir(); (d / "config.json").write_text("{}")
    out = lab.run("register", str(d), "--name", "so101-x", rc=2)
    assert "not a pretrained_model dir" in out and "train_config.json" in out and "model.safetensors" in out
    assert "so101-x" not in [m.name for m in lab.client().search_registered_models()]
    assert "needs a checkpoint dir or --from-hf" in lab.run("register", "--name", "so101-x", rc=2)


def test_list_without_models(lab):
    assert lab.run("list").strip() == "no registered models"


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """huggingface_hub.snapshot_download replaced by a copy of a fake job repo (two checkpoints), recording how it was called."""
    hub = pytest.importorskip("huggingface_hub")
    repo = tmp_path / "fake_repo"; make_ckpt(str(repo), seed=3, step="002500"); make_ckpt(str(repo), seed=4, step="005000")
    (repo / "README.md").write_text("fake")
    calls = []
    def fake_snapshot(repo_id, local_dir, allow_patterns=None, local_files_only=False, **kw):
        calls.append({"repo_id": repo_id, "local_dir": local_dir, "allow_patterns": allow_patterns, "local_files_only": local_files_only})
        assert local_files_only, "HF_HUB_OFFLINE=1 must turn into local_files_only=True (no network from a test)"
        shutil.copytree(str(repo), local_dir, dirs_exist_ok=True); return local_dir
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
    return calls


def test_register_from_hf_downloads_into_policies_hub_and_registers_the_newest_checkpoint(lab, fake_hub):
    out = lab.run("register", "--from-hf", "tclastlight/fake_repo", "--name", "so101-cloud", "--hf-job", "6ad877680a35")
    assert "HF_HUB_OFFLINE: cache only" in out and "registered so101-cloud v1" in out
    assert fake_hub == [{"repo_id": "tclastlight/fake_repo", "local_dir": os.path.join(lab.tmp, "hub", "tclastlight__fake_repo"), "allow_patterns": None, "local_files_only": True}]
    mv = lab.client().get_model_version("so101-cloud", "1")
    assert mv.tags["hf_repo"] == "tclastlight/fake_repo" and mv.tags["hf_job"] == "6ad877680a35" and mv.tags["checkpoint_step"] == "005000"
    assert mv.tags["source_dir"] == os.path.join(lab.tmp, "hub", "tclastlight__fake_repo", "checkpoints", "005000", "pretrained_model")
    assert "from tclastlight/fake_repo (job 6ad877680a35) step 005000" in lab.run("list", "so101-cloud")


def test_register_from_hf_step_narrows_the_download(lab, fake_hub):
    lab.run("register", "--from-hf", "tclastlight/fake_repo", "--name", "so101-cloud", "--step", "002500")
    assert fake_hub[0]["allow_patterns"] == ["checkpoints/002500/*", "*.json", "*.safetensors", "*.md"]
    mv = lab.client().get_model_version("so101-cloud", "1"); assert mv.tags["checkpoint_step"] == "002500"
    assert "no pretrained_model found for step 009999" in lab.run("register", "--from-hf", "tclastlight/fake_repo", "--name", "so101-cloud", "--step", "009999", rc=2)


def test_register_from_hf_offline_cache_miss_fails_cleanly(lab):
    pytest.importorskip("huggingface_hub")
    out = lab.run("register", "--from-hf", "tclastlight/no-such-repo-for-this-test", "--name", "so101-x", rc=2)   # HF_HUB_OFFLINE=1 + empty HF_HOME: no network, no cache
    assert "snapshot_download tclastlight/no-such-repo-for-this-test" in out
    assert not any(m.name == "so101-x" for m in lab.client().search_registered_models())


def test_register_rejects_both_a_dir_and_from_hf(lab):
    assert "either a checkpoint dir or --from-hf" in lab.run("register", lab.ckpt, "--from-hf", "tclastlight/x", "--name", "so101-x", rc=2)


def test_pick_checkpoint_prefers_flat_dir_then_newest_step_and_skips_last(lab, tmp_path):
    t = lab.tool
    assert t.pick_checkpoint(lab.ckpt) == lab.ckpt
    root = str(tmp_path / "job"); make_ckpt(root, step="001000"); make_ckpt(root, step="002000"); os.symlink("002000", os.path.join(root, "checkpoints", "last"))
    assert t.pick_checkpoint(root).endswith(os.path.join("checkpoints", "002000", "pretrained_model"))
    assert t.pick_checkpoint(root, "001000").endswith(os.path.join("checkpoints", "001000", "pretrained_model"))
    assert t.ckpt_step(os.path.join(root, "checkpoints", "last", "pretrained_model")) == "002000" and t.ckpt_step(str(tmp_path)) is None
    assert t.ckpt_params(json.load(open(os.path.join(lab.ckpt, "train_config.json")))) == {"policy": "act", "chunk_size": 100, "n_action_steps": 100, "lr": 1e-05, "dim_model": 512, "use_vae": True, "batch_size": 8, "steps": 5000, "seed": 1000, "dataset": "lastlight/fake-set", "warm_start": "scratch", "device": "mps"}
