"""so101-policy deploy / rollback / status end to end against a local directory standing in for the Jetson (SO101_DEPLOY_LOCAL_ROOT):
rsync into <name>/v<N>/, the `current` symlink, DEPLOYED.json with its history, the `jetson` alias and version tags, idempotence,
--dry-run, the three-version retention, and status agreeing or disagreeing with the registry."""
import json, os
import pytest
from toolhelpers import make_ckpt

pytest.importorskip("mlflow")
NAME = "so101-fake"


@pytest.fixture
def two(lab):
    """v1 and v2 registered (different weights), nothing deployed."""
    lab.run("register", lab.ckpt, "--name", NAME)
    lab.ck2 = make_ckpt(os.path.join(lab.tmp, "train", "act_fake2"), seed=2); lab.run("register", lab.ck2, "--name", NAME, "--note", "v2")
    return lab


def test_deploy_copies_links_writes_deployed_json_and_moves_the_alias(two):
    lab = two; out = lab.run("deploy", f"{NAME}@1")
    assert f"deploy {NAME} v1 -> local stand-in" in out and "current -> v1" in out and "alias jetson -> v1" in out and "v1 is live (deploy)" in out
    v1 = os.path.join(lab.policies, NAME, "v1")
    assert sorted(os.listdir(v1)) == ["config.json", "model.safetensors", "policy_preprocessor.json", "train_config.json"]
    assert lab.sha(os.path.join(v1, "model.safetensors")) == lab.sha(os.path.join(lab.ckpt, "model.safetensors")) and lab.current(NAME) == "v1"
    d = lab.deployed(); c = lab.client(); mv = c.get_model_version(NAME, "1")
    assert d["name"] == NAME and d["version"] == 1 and d["run_id"] == mv.run_id and d["sha256"] == lab.sha(os.path.join(v1, "model.safetensors"))
    assert d["when"] and d["by"] and d["action"] == "deploy" and d["path"] == f"~/so101/policies/{NAME}/current" and len(d["history"]) == 1 and d["history"][0]["version"] == 1
    assert lab.tool.aliases_of(c, NAME) == {"candidate": "2", "jetson": "1"}
    assert mv.tags["deployed_at"] == d["when"] and mv.tags["deployed_to"] == "so101-test-no-such-host.invalid" and mv.tags["deployed_by"] == d["by"]
    assert "v1   [jetson]" in lab.run("list", NAME) and "DEPLOYED " + d["when"] in lab.run("list", NAME)


def test_redeploying_the_same_version_is_a_noop(two):
    lab = two; lab.run("deploy", f"{NAME}@1"); before = lab.deployed(); mtime = os.path.getmtime(os.path.join(lab.policies, "DEPLOYED.json"))
    out = lab.run("deploy", f"{NAME}@1")
    assert f"{NAME} v1 is already deployed" in out and "nothing to do" in out and "rsync" not in out
    assert lab.deployed() == before and os.path.getmtime(os.path.join(lab.policies, "DEPLOYED.json")) == mtime
    # the symlink is part of the check: a broken `current` gets repaired by a redeploy
    cur = os.path.join(lab.policies, NAME, "current"); os.unlink(cur); os.symlink("v9", cur)
    assert "v1 is live" in lab.run("deploy", f"{NAME}@1") and lab.current(NAME) == "v1" and len(lab.deployed()["history"]) == 2


def test_dry_run_changes_nothing(two):
    lab = two; out = lab.run("deploy", f"{NAME}@candidate", "--dry-run")
    assert out.startswith("dry run:") and f"deploy {NAME} v2 -> local stand-in" in out and "current -> v2" in out and "alias jetson -> v2" in out
    assert lab.deployed() is None and not os.path.exists(os.path.join(lab.policies, NAME)) and lab.tool.aliases_of(lab.client(), NAME) == {"candidate": "2"}


def test_deploy_by_alias_latest_and_custom_dest(two):
    lab = two; out = lab.run("deploy", f"{NAME}@candidate", "--dest", "~/so101/policies/elsewhere")
    assert f"deploy {NAME} v2" in out and "so101/policies/elsewhere/v2/" in out
    assert os.readlink(os.path.join(lab.policies, "elsewhere", "current")) == "v2" and lab.deployed()["path"] == "~/so101/policies/elsewhere/current"
    assert lab.tool.aliases_of(lab.client(), NAME)["jetson"] == "2"
    assert "v2 is live" in lab.run("deploy", NAME) and lab.current(NAME) == "v2"   # bare name = latest version, default dest


def test_deploy_errors_name_version_alias(two):
    lab = two
    assert "no version 9 (have 1, 2)" in lab.run("deploy", f"{NAME}@9", rc=2)
    assert "no alias 'champion' (aliases: candidate->v2)" in lab.run("deploy", f"{NAME}@champion", rc=2)
    assert "no versions registered" in lab.run("deploy", "so101-nobody@1", rc=2)
    assert lab.deployed() is None


def test_deploy_verifies_the_remote_sha(two, monkeypatch):
    lab = two; monkeypatch.setattr(lab.tool, "sha256_of", lambda p: "0" * 64)   # why: the local digest no longer matches what lands on the target
    assert "sha256 mismatch after copy" in lab.run("deploy", f"{NAME}@1", rc=2)
    assert "jetson" not in lab.tool.aliases_of(lab.client(), NAME), "the alias only moves after the copy is verified"


def test_rollback_goes_back_to_the_previous_version_and_records_it(two):
    lab = two; lab.run("deploy", f"{NAME}@1"); lab.run("deploy", f"{NAME}@2")
    out = lab.run("rollback", NAME)
    assert f"rollback {NAME}: v2 -> v1" in out and "v1 is live (rollback)" in out and "alias jetson -> v1 (was v2)" in out
    d = lab.deployed(); assert d["version"] == 1 and d["action"] == "rollback" and [h["version"] for h in d["history"]] == [1, 2, 1] and lab.current(NAME) == "v1"
    assert lab.versions(NAME) == ["v1", "v2"] and lab.tool.aliases_of(lab.client(), NAME)["jetson"] == "1"
    assert lab.client().get_model_version(NAME, "1").tags["deployed_at"] == d["when"]
    assert "dry run:" in lab.run("rollback", NAME, "--dry-run") and lab.deployed()["version"] == 1   # a second rollback would go forward to v2; dry run shows it only


def test_rollback_with_nothing_to_go_back_to(two):
    lab = two
    assert f"{NAME}: never deployed" in lab.run("rollback", NAME, rc=2)
    lab.run("deploy", f"{NAME}@1")
    assert f"{NAME}: only v1 was ever deployed" in lab.run("rollback", NAME, rc=2) and lab.current(NAME) == "v1"


def test_only_the_last_three_versions_stay_on_the_target(two):
    lab = two
    for i in (3, 4): lab.run("register", make_ckpt(os.path.join(lab.tmp, "train", f"act_fake{i}"), seed=i), "--name", NAME)
    for v in (1, 2, 3): lab.run("deploy", f"{NAME}@{v}")
    assert lab.versions(NAME) == ["v1", "v2", "v3"]
    out = lab.run("deploy", f"{NAME}@4")
    assert "keep v4, v3, v2; prune v1" in out and "pruned v1" in out and lab.versions(NAME) == ["v2", "v3", "v4"]
    out = lab.run("rollback", NAME)   # back to v3: nothing pruned, v2 still there for the next step back
    assert lab.current(NAME) == "v3" and lab.versions(NAME) == ["v2", "v3", "v4"] and "pruned" not in out
    lab.run("rollback", NAME); assert lab.current(NAME) == "v4"   # the previous distinct version of v3 is v4 (history 1,2,3,4,3)
    d = lab.deployed(); assert [h["version"] for h in d["history"]] == [1, 2, 3, 4, 3, 4]


def test_a_pruned_version_can_still_be_redeployed(two):
    lab = two
    for i in (3, 4): lab.run("register", make_ckpt(os.path.join(lab.tmp, "train", f"act_fake{i}"), seed=i), "--name", NAME)
    for v in (1, 2, 3, 4): lab.run("deploy", f"{NAME}@{v}")
    assert lab.versions(NAME) == ["v2", "v3", "v4"]
    assert "v1 is live" in lab.run("deploy", f"{NAME}@1") and lab.versions(NAME) == ["v1", "v3", "v4"]   # keep = v1 + the two most recent before it


def test_status_before_any_deploy(two):
    out = two.run("status")
    assert "DEPLOYED.json: none" in out and f"{NAME}: not on the jetson (registry jetson alias -> none)" in out and "so101-run default -> fallback (~/so101/policies/act_lighter)" in out


def test_status_agrees_then_catches_a_moved_alias_and_a_vla_default(two):
    lab = two; lab.run("deploy", f"{NAME}@2"); out = lab.run("status")
    assert f"DEPLOYED.json: {NAME} v2" in out and f"{NAME}: jetson current -> v2 (act, weights ok; on disk: v2); registry jetson alias -> v2  OK" in out
    assert f"so101-run default -> ~/so101/policies/{NAME}/current ({NAME} v2)" in out
    lab.tool.set_alias(lab.client(), NAME, "jetson", "1")   # someone moved the alias by hand in the UI
    out = lab.run("status", rc=1); assert "registry jetson alias -> v1  MISMATCH" in out
    lab.run("register", make_ckpt(os.path.join(lab.tmp, "train", "vla"), seed=9, kind="smolvla"), "--name", "so101-vla"); lab.run("deploy", "so101-vla@1")
    out = lab.run("status", "so101-vla"); assert "so101-vla: jetson current -> v1 (smolvla, weights ok" in out and "so101-run-vla default -> ~/so101/policies/so101-vla/current (so101-vla v1)" in out
    os.unlink(os.path.join(lab.policies, "so101-vla", "v1", "model.safetensors"))
    out = lab.run("status", "so101-vla", rc=1); assert "weights MISSING" in out and "so101-run default -> fallback" in out


def test_read_target_tolerates_a_corrupt_deployed_json(two):
    lab = two; os.makedirs(lab.policies, exist_ok=True); open(os.path.join(lab.policies, "DEPLOYED.json"), "w").write("{not json")
    assert "DEPLOYED.json: unreadable" in lab.run("status")
    assert "v1 is live" in lab.run("deploy", f"{NAME}@1") and lab.deployed()["history"][0]["version"] == 1   # the corrupt file is replaced, history starts over


def test_version_dir_downloads_when_the_artifact_store_is_not_local(two, monkeypatch):
    lab = two; c = lab.client(); mv = c.get_model_version(NAME, "1"); real_get_run = c.get_run
    class R:   # why: pretend the run's artifacts live behind a server; the tool must then pull them into policies_hub/registry/<name>/v<N>
        def __init__(self, r): self.info = type("I", (), {"artifact_uri": "http://mlflow.invalid/artifacts/" + r.info.run_id, "run_id": r.info.run_id})(); self.data = r.data
    monkeypatch.setattr(c, "get_run", lambda rid: R(real_get_run(rid)))
    import mlflow.artifacts
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: os.path.join(dst_path, "model"))
    d = lab.tool.version_dir(lab.tool.client()[0], c, NAME, mv)
    assert d == os.path.join(lab.tmp, "hub", "registry", NAME, "v1", "model")
    assert json.load(open(os.path.join(lab.ckpt, "train_config.json")))["steps"] == 5000   # the source checkpoint is untouched
