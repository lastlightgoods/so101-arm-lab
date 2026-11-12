"""`learn_loop.py train --where hf|jetson` (tools/learn_loop_train.py), TC 11 Nov 2026: training is on Hugging Face or on the Jetson, never on the Mac. Fakes only: FakeSys answers the `hf` CLI
(upload, repos tag create, jobs run/inspect/logs, download), FakeJetson is the Jetson's shell over a tmp folder, fetch is a local copy. These runs are NOT --dry-run (so the --yes gate and the
refusals are the real ones), with every seam injected; the root conftest guard would block ssh and rsync-to-host anyway. No network, no ssh, no spend."""
import json, os, shlex
import pytest
from learn_loop_helpers import LL, FF, client  # noqa: F401
from learn_loop_helpers import mlflow  # noqa: F401

TL = LL.mod("tools/learn_loop_train.py", "learn_loop_train_under_test")


class Rig:
    def __init__(self, tmp, **sys_kw):
        self.tmp = tmp; self.jet = FF.FakeJetson(str(tmp / "jetson_home"), polls=sys_kw.pop("polls", 1), busy=sys_kw.pop("busy", ()), fail_at=sys_kw.pop("fail_at", None)); self.fs = FF.FakeSys(jetson_home=self.jet.home, **sys_kw)
        self.fetches = []; self.procs = []
    def fetch(self, src, dst): self.fetches.append((src, dst)); return self.jet.fetch(src, dst)
    def go(self, *argv, where="hf", extra=()):
        uri = "sqlite:///" + str(self.tmp / "mlflow.db")
        return LL.main([*argv, "--name", "t1", "--out-dir", str(self.tmp / "ll"), "--mlflow-uri", uri, "--where", where, "--min-kept", "2", "--cycles", "6", "--steps", "1000", "--save-freq", "500", *extra],
                       sys_=self.fs, jetson=self.jet, fetch=self.fetch, local_procs=lambda: list(self.procs), sleep=lambda s: None)
    def state(self): return json.load(open(self.tmp / "ll" / "t1" / "state.json"))
    def client(self): return mlflow.MlflowClient(tracking_uri="sqlite:///" + str(self.tmp / "mlflow.db"))
    def hf_calls(self): return [c for c in self.fs.calls if str(c[0]) == "hf"]


@pytest.fixture
def rig(tmp_path): return Rig(tmp_path)


def recorded(r):
    assert r.go("record") == 0; return r.state()["stages"]["record"]["result"]


# ---------------------------------------------------------------- the record stage no longer copies the dataset
def test_a_real_record_leaves_the_dataset_on_the_jetson_and_trusts_the_recorders_hash(rig):
    rec = recorded(rig)
    assert rec["dataset_root"] is None and rec["remote_root"] == "~/so101/outputs/demos/ll-t1/so101-nut-pick" and len(rec["dataset_hash"]) == 64 and rec["episodes"] == 8 and rec["info"]["features"]["observation.images.top"]
    argv = next(c for c in rig.fs.calls if str(c[1]).endswith("autoloop.py")) if False else next(c for c in rig.fs.calls if any(str(x).endswith("autoloop.py") for x in c))
    assert "--demo-root" not in argv and "--dry-run" not in argv, "the Jetson recorder's own default folder is used"
    assert rig.fetches == [], "nothing was copied"


# ---------------------------------------------------------------- hf
def test_hf_prints_the_plan_and_refuses_to_spend_without_yes(rig, capsys):
    recorded(rig); capsys.readouterr()
    assert rig.go("train") == 1
    err = rig.state()["stages"]["train"]["error"]; out = capsys.readouterr().out
    assert "not launching" in err and "--yes" in err and "private repo" in err and "HF Job a100-large" in out and "3.1 step/s measured" in out and "$" in out
    assert rig.hf_calls() == [] and rig.fetches == [], "nothing uploaded, nothing launched, no dataset pulled"
    c = rig.client(); assert c.get_experiment_by_name("so101-act") is None or not c.search_runs([c.get_experiment_by_name("so101-act").experiment_id]), "a refusal leaves no FAILED train run"


def test_hf_with_yes_uploads_privately_launches_one_job_and_brings_the_model_back(rig, capsys):
    rec = recorded(rig); assert rig.go("train", extra=("--yes",)) == 0
    hf = rig.hf_calls(); verbs = [" ".join(map(str, c[1:3])) if c[1] == "jobs" else str(c[1]) for c in hf]
    assert verbs == ["upload", "repos", "jobs run", "jobs inspect", "jobs inspect", "jobs logs", "download"] and hf[0][2] == "tclastlight/so101-nut-pick-" + rec["dataset_hash"][:8]
    up = hf[0]; assert "--private" in up and up[up.index("--repo-type") + 1] == "dataset" and rig.fetches and rig.fetches[0][0].startswith("jetson:so101/outputs/demos/"), "the dataset came from the Jetson folder, then went to the Hub"
    assert list(hf[1][2:]) == ["tag", "create", up[2], "v3.0", "--repo-type", "dataset"]
    run = hf[2]; assert run[run.index("--flavor") + 1] == "a100-large" and "--detach" in run and run[run.index("--secrets") + 1] == "HF_TOKEN" and run[run.index("--timeout") + 1] == "1h" and run[run.index("--name") + 1] == "act_t1"
    inner = shlex.split(run[-1]); assert inner[0] == "lerobot-train" and "--policy.push_to_hub=true" in inner and "--policy.repo_id=tclastlight/act_t1" in inner and f"--dataset.repo_id={up[2]}" in inner and "--batch_size=32" in inner and "--steps=1000" in inner
    feats = json.loads(next(x for x in inner if x.startswith("--policy.input_features="))[len("--policy.input_features="):]); assert "observation.environment_state" not in feats and set(feats) >= {"observation.images.top", "observation.state"}
    assert any(str(c[1]).endswith("prep_ckpt.py") for c in rig.fs.calls if str(c[0]) != "hf"), "cloud checkpoints are prepped for lerobot 0.6.1"
    tr = rig.state()["stages"]["train"]["result"]; assert tr["where"] == "hf" and tr["extra"]["hf_repo"] == "tclastlight/act_t1" and len(tr["extra"]["hf_job"]) == 24 and tr["steps"] == 10000 and tr["final_loss"] == pytest.approx(0.081) and os.path.exists(tr["checkpoint"] + "/model.safetensors")
    r = rig.client().get_run(tr["train_run"]); assert r.data.tags["where"] == "hf" and r.data.tags["hf_job"] == tr["extra"]["hf_job"] and r.data.tags["dataset_hash"] == rec["dataset_hash"] and r.data.params["dataset_hash"] == rec["dataset_hash"] and r.info.status == "FINISHED"
    assert float(r.data.params["est_cost_usd"]) > 0 and "final_loss" in r.data.metrics


def test_the_registry_entry_of_an_hf_model_carries_the_hub_repo_and_job(rig):
    rec = recorded(rig); assert rig.go("train", "register", extra=("--yes",)) == 0
    reg = rig.state()["stages"]["register"]["result"]; mv = rig.client().get_model_version(reg["name"], reg["version"]); tr = rig.state()["stages"]["train"]["result"]
    assert mv.tags["hf_repo"] == "tclastlight/act_t1" and mv.tags["hf_job"] == tr["extra"]["hf_job"] and mv.tags["dataset_hash"] == rec["dataset_hash"]


def test_a_repeated_command_polls_the_job_it_already_paid_for(rig):
    recorded(rig); rig.fs.hf_stage = "ERROR"
    assert rig.go("train", extra=("--yes",)) == 1 and "ended ERROR" in rig.state()["stages"]["train"]["error"] and "train --force --yes" in rig.state()["stages"]["train"]["error"]
    jid = rig.state()["ids"]["hf_job"]; rig.fs.hf_stage = "COMPLETED"; rig.fs.hf_polls = 0; n = len(rig.hf_calls())
    assert rig.go("train") == 0, "polling an existing job needs no --yes"
    new = rig.hf_calls()[n:]; assert not any(c[1] in ("upload", "repos") or c[1:3] == ["jobs", "run"] for c in new) and rig.state()["ids"]["hf_job"] == jid and sum(1 for c in rig.hf_calls() if c[1:3] == ["jobs", "run"]) == 1


def test_a_dataset_whose_copy_does_not_match_the_recorders_hash_is_not_uploaded(rig):
    rec = recorded(rig)
    st = rig.state(); st["stages"]["record"]["result"]["dataset_hash"] = "ab" * 32; json.dump(st, open(rig.tmp / "ll" / "t1" / "state.json", "w"))
    assert rig.go("train", extra=("--yes",)) == 1 and "not uploading" in rig.state()["stages"]["train"]["error"] and not any(c[1] == "upload" for c in rig.hf_calls()) and rec["dataset_hash"] != "ab" * 32


def test_the_numbers_printed_before_launch_come_from_the_measured_rates():
    p = LL.build_parser(); hf = TL.plan_numbers(p.parse_args(["train", "--name", "x"]))
    assert hf["flavor"] == "a100-large" and hf["steps"] == 10000 and hf["batch"] == 32 and hf["steps_per_s"] == 3.1 and hf["wall_s"] == pytest.approx(10000 / 3.1 + 300) and hf["cost_usd"] == pytest.approx(2.45, abs=0.01) and hf["timeout"] == "2h" and hf["max_cost_usd"] == 5.0
    jt = TL.plan_numbers(p.parse_args(["train", "--name", "x", "--where", "jetson"])); assert jt["steps"] == 20000 and jt["batch"] == 4 and jt["steps_per_s"] == 1.9 and jt["wall_s"] == pytest.approx(20000 / 1.9) and jt["cost_usd"] == 0
    assert TL.plan_numbers(p.parse_args(["train", "--name", "x", "--where", "jetson", "--batch", "2"]))["steps_per_s"] == 3.8
    assert "$2.45" in TL.plan_text(hf) and "no cost" in TL.plan_text(jt)


# ---------------------------------------------------------------- jetson
@pytest.mark.parametrize("busy, why", [([".arms_busy exists"], "arms_busy"), ([".cams_busy exists"], "cams_busy"), (["lerobot job: 99 lerobot-rollout --x"], "lerobot job"), (["pick_seq session: 5 python pick_seq.py --serve"], "pick_seq session"),
                                         (["recorder has an open episode"], "open episode")])
def test_the_jetson_backend_refuses_while_the_arm_or_cameras_are_in_use(tmp_path, busy, why):
    r = Rig(tmp_path, busy=busy); recorded(r)
    assert r.go("train", where="jetson") == 1 and why in r.state()["stages"]["train"]["error"] and "refusing to train on the Jetson" in r.state()["stages"]["train"]["error"] and r.jet.launched == []
    c = r.client(); assert c.get_experiment_by_name("so101-act") is None or not c.search_runs([c.get_experiment_by_name("so101-act").experiment_id])


def test_the_jetson_backend_refuses_while_autoloop_or_tidy_runs_on_the_mac(rig):
    recorded(rig); rig.procs = ["12 /x/python /x/tools/tidy.py run --record"]
    assert rig.go("train", where="jetson") == 1 and "on the Mac: 12" in rig.state()["stages"]["train"]["error"] and rig.jet.launched == []


def test_the_busy_script_looks_for_every_reason_TC_named():
    for needle in (".arms_busy", ".cams_busy", "lerobot-(teleoperate|record|rollout|calibrate|replay|train)", "pick_seq.py.*--serve", "demo_recorderd.py", '"episode": null'): assert needle in TL.BUSY_SCRIPT, needle


def test_jetson_training_reads_the_dataset_where_the_recorder_wrote_it_and_copies_only_the_checkpoint(rig):
    rec = recorded(rig); assert rig.go("train", where="jetson") == 0
    sc = rig.jet.launched[0]
    assert '"--dataset.root=$HOME/so101/outputs/demos/ll-t1/so101-nut-pick"' in sc and "--batch_size=4" in sc and "--num_workers=1" in sc and "--policy.device=cuda" in sc and "--steps=1000" in sc and 'touch "$HOME/so101/.cams_busy"' in sc and "rm -f" in sc and "nohup" in sc
    assert "--policy.push_to_hub=false" in sc and "observation.environment_state" not in sc and "/Users/" not in sc
    assert len(rig.fetches) == 1 and rig.fetches[0][0] == "jetson:so101/outputs/train/act_t1/checkpoints/last/pretrained_model", "one rsync: the checkpoint; the dataset was never copied"
    tr = rig.state()["stages"]["train"]["result"]; assert tr["where"] == "jetson" and tr["steps"] == 1000 and tr["final_loss"] is not None and os.path.exists(tr["checkpoint"] + "/model.safetensors")
    r = rig.client().get_run(tr["train_run"]); assert r.data.tags["where"] == "jetson" and r.data.tags["dataset_hash"] == rec["dataset_hash"] and r.data.tags["learn_loop_run"] == rig.state()["ids"]["parent"] and r.info.status == "FINISHED"


def test_a_crashed_jetson_training_resumes_from_checkpoints_last(tmp_path):
    r = Rig(tmp_path, polls=0, fail_at=500); recorded(r)
    assert r.go("train", where="jetson") == 1 and "stopped at step 500 of 1000" in r.state()["stages"]["train"]["error"] and "resume" in r.state()["stages"]["train"]["error"]
    assert r.go("train", where="jetson") == 0
    assert len(r.jet.launched) == 2 and "--resume=true" in r.jet.launched[1] and '"--config_path=$HOME/so101/outputs/train/act_t1/checkpoints/last/pretrained_model/train_config.json"' in r.jet.launched[1] and "--dataset.root" not in r.jet.launched[1]
    assert r.state()["stages"]["train"]["result"]["steps"] == 1000


def test_a_job_still_running_on_the_jetson_is_polled_not_started_again(rig):
    recorded(rig); rig.jet.left = 3   # a training from an earlier, interrupted command is still going
    rig.jet.run("# ll:train-launch\nmkdir -p x; LOG=\"$HOME/so101/outputs/train/act_t1.log\"\n--steps=1000\n")   # it has written its checkpoints
    rig.jet.launched.clear(); rig.jet.left = 3
    assert rig.go("train", where="jetson") == 0 and rig.jet.launched == []


def test_a_dataset_that_is_not_on_the_jetson_is_not_trained_there(tmp_path):
    ds = FF.make_dataset(str(tmp_path / "mine"), episodes=5); r = Rig(tmp_path)
    assert r.go("record", extra=("--adopt-dataset", ds)) == 0 and r.go("train", where="jetson") == 1
    assert "not on the Jetson" in r.state()["stages"]["train"]["error"] and "--where hf" in r.state()["stages"]["train"]["error"]


def test_the_mac_is_not_a_training_target_outside_dry_run(rig):
    recorded(rig); assert rig.go("train", where="local") == 1 and "Mac is not a training target" in rig.state()["stages"]["train"]["error"] and not any("lerobot-train" in str(c[0]) for c in rig.fs.calls)


# ---------------------------------------------------------------- seams
def test_the_ssh_shell_pipes_the_script_over_stdin():
    seen = []
    class R:
        returncode = 3; stdout = "out"; stderr = "err"
    sh = TL.SshShell("jetson-test", run=lambda argv, **k: (seen.append((argv, k)), R)[1]); assert sh.run("echo hi\n") == (3, "outerr")
    assert seen[0][0][:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"] and seen[0][0][-3:] == ["jetson-test", "bash", "-s"] and seen[0][1]["input"] == "echo hi\n"


def test_q_keeps_home_expandable_and_everything_else_literal():
    assert TL.q("--dataset.root=$HOME/so101/x") == '"--dataset.root=$HOME/so101/x"' and TL.q("--a=b c") == "'--a=b c'" and TL.q('--policy.input_features={"a":1}') == "'--policy.input_features={\"a\":1}'"
