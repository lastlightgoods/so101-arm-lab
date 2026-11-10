"""parse_probe(): the ###-sectioned slow probe output, and probe_loop() merging Mac + Jetson results into state."""
import pytest

PROCS = (" 12345    1-02:03:04 /home/tc/so101/.venv/bin/python /home/tc/so101/.venv/bin/lerobot-record --dataset.repo_id=lastlight/cube\n"
         " 12399       05:12 python /home/tc/so101/dance.py 100\n")
JETSON_PROBE = """###mode
NV Power Mode: 25W
###uptime
up 2 days, 3 hours, 4 minutes
###disk
/dev/nvme0n1p1  915G  312G  557G  36% /
###procs
@PROCS@
###trainlog /home/tc/so101/outputs/train/act_cube/train.log
INFO 2026-11-07 step:1.0K smpl:8K ep:12 epch:0.50 loss:0.123 grdn:1.2 lr:1.0e-05 updt_s:0.1 data_s:0.0
INFO 2026-11-07 step:2.0K smpl:16K ep:24 epch:1.00 loss:0.098 grdn:1.1 lr:5.0e-06 updt_s:0.1 data_s:0.0
some unrelated line
###trainlog /home/tc/so101/outputs/train/smolvla_x.log
step:500 loss:1.5
###ckpts /home/tc/so101/outputs/train/act_cube/checkpoints
001000
002000
last
###ckpts /home/tc/so101/outputs/train/orphan/checkpoints
000500
###dataset /home/tc/.cache/huggingface/lerobot/lastlight/cube/meta/info.json
1762550000
{"total_episodes": 12, "total_frames": 9000, "fps": 30, "features": {"observation.images.top": {"dtype": "video"}, "observation.images.wrist": {"dtype": "video"}, "observation.state": {"dtype": "float32"}}}
###dataset /home/tc/.cache/huggingface/lerobot/lastlight/debug_cube/meta/info.json
1762550001
{"total_episodes": 1, "total_frames": 10, "fps": 30, "features": {}}
###dataset /home/tc/.cache/huggingface/lerobot/lastlight/broken/meta/info.json
1762550002
{this is not json
###dataset /home/tc/.cache/huggingface/lerobot/lastlight/pretty/meta/info.json
1762550003
{
  "total_episodes": 3,
  "total_frames": 300,
  "fps": 30,
  "features": {"observation.images.top": {"dtype": "video"}}
}
###policy /home/tc/so101/policies/ft_lighter/train_config.json
1762540000
{"policy": {"type": "act", "device": "cuda"}, "steps": 20000, "dataset": {"repo_id": "lastlight/cube"}}
###policy /home/tc/so101/outputs/train/act_cube/checkpoints/last/pretrained_model/train_config.json
1762540001
{"policy": {"type": "act"}, "steps": 5, "dataset": null}
###policy /home/tc/so101/policies/nostat/train_config.json
{"policy": {"type": "act"}, "steps": 5}
###end
""".replace("@PROCS@\n", PROCS)


@pytest.fixture
def out(srv, logs):
    return srv.parse_probe(JETSON_PROBE, "jetson")


def test_mode_uptime_disk(out):
    assert out["mode"] == "25W" and out["uptime"] == "2 days, 3 hours, 4 minutes"
    assert out["disk"] == {"used": "312G", "total": "915G", "pct": 36}


def test_procs_kind_and_elapsed_seconds(out):
    rec, dance = out["procs"]
    assert rec["kind"] == "record" and rec["secs"] == 1 * 86400 + 2 * 3600 + 3 * 60 + 4 and rec["host"] == "jetson"
    assert dance["kind"] == "dance" and dance["secs"] == 5 * 60 + 12
    assert rec["args"].startswith("/home/tc/so101/.venv/bin/python")


def test_trainlog_points_lr_and_job_names(out):
    t = out["training"]["act_cube"]                     # <job>/train.log -> job dir name
    assert t["points"] == [[1000.0, 0.123], [2000.0, 0.098]] and t["lr"] == "5.0e-06" and t["host"] == "jetson"
    assert out["training"]["smolvla_x"]["points"] == [[500.0, 1.5]]   # <job>.log -> stem


def test_ckpts_merge_into_training_or_create_entry(out):
    assert out["training"]["act_cube"]["ckpts"] == [1000, 2000] and out["training"]["act_cube"]["last_ckpt"] == 2000
    assert out["training"]["act_cube"]["points"], "ckpts must not wipe the trainlog points"
    assert out["training"]["orphan"] == {"host": "jetson", "points": [], "ckpts": [500], "last_ckpt": 500}


def test_dataset_mtime_from_stat_line_and_fields(out):
    d = out["datasets"]["cube"]
    assert d["mtime"] == 1762550000 and d["episodes"] == 12 and d["frames"] == 9000 and d["fps"] == 30
    assert d["hours"] == pytest.approx(9000 / 30 / 3600) and d["cams"] == ["top", "wrist"] and d["host"] == "jetson"


def test_dataset_pretty_printed_json_is_fine(out):
    assert out["datasets"]["pretty"]["mtime"] == 1762550003 and out["datasets"]["pretty"]["cams"] == ["top"]


def test_debug_dataset_skipped(out):
    assert "debug_cube" not in out["datasets"]


def test_bad_dataset_json_is_logged_not_fatal(out, logs):
    assert "broken" not in out["datasets"]
    assert any(l.startswith("dataset parse") and "broken/meta/info.json" in l for l in logs)
    assert "pretty" in out["datasets"] and out["policies"], "sections after the broken one still parse"


def test_policy_fields_and_mtime(out, logs):
    p = out["policies"]["ft_lighter"]
    assert p == {"host": "jetson", "type": "act", "steps": 20000, "dataset": "lastlight/cube", "device": "cuda", "mtime": 1762540000}
    q = out["policies"]["act_cube"]                     # outputs/train/<job>/checkpoints/last/... -> job name
    assert q["dataset"] == "?" and q["device"] == "?" and q["mtime"] == 1762540001
    assert "nostat" not in out["policies"] and any(l.startswith("policy parse") and "nostat" in l for l in logs)


def test_empty_and_partial_probe(srv, logs):
    assert srv.parse_probe("", "mac") == {"training": {}, "datasets": {}, "policies": {}, "procs": [], "mode": "", "uptime": "", "disk": ""}
    o = srv.parse_probe("###disk\n\n###procs\n\n###end\n", "mac")
    assert o["disk"] == "" and o["procs"] == []


def test_num_suffixes(srv):
    assert srv.num("1.5", "K") == 1500 and srv.num("2", "M") == 2e6 and srv.num("7", "") == 7


MAC_PROBE = JETSON_PROBE.replace("NV Power Mode: 25W", "mac").replace(PROCS, "")   # the Mac runs no lerobot processes here


def test_probe_loop_merges_hosts_and_announces_changes(srv, fake, run_loop, events, logs):
    # pass 2: the Jetson dataset grew by one episode and the lerobot processes finished
    grown = JETSON_PROBE.replace('"total_episodes": 12', '"total_episodes": 13').replace(PROCS, "")
    jetson = iter([JETSON_PROBE, grown])
    fake.on(lambda a: a[0] == "bash", MAC_PROBE)
    fake.on(lambda a: a[0] == "ssh" and a[-2:] == ["bash", "-s"], lambda argv, kw: next(jetson))
    run_loop(srv.probe_loop, 2)
    # the Mac probe is the same script with the Jetson-only bits swapped out; the Jetson one goes through ssh stdin
    mac_call, ssh_call = fake.calls[0], fake.calls[1]
    assert mac_call[0] == ["bash", "-s"] and "echo mac" in mac_call[1]["input"] and "stat -f %m" in mac_call[1]["input"]
    assert ssh_call[0] == srv.SSH + ["bash", "-s"] and ssh_call[1]["input"] == srv.PROBE and len(fake.calls) == 4
    # same dataset name on both hosts: Mac keeps the bare name, Jetson gets a @jetson suffix
    assert set(srv.state["datasets"]) == {"cube", "pretty", "cube@jetson", "pretty@jetson"}
    assert srv.state["datasets"]["cube"]["host"] == "mac" and srv.state["datasets"]["cube@jetson"]["host"] == "jetson"
    assert srv.state["datasets"]["cube@jetson"]["episodes"] == 13 and srv.state["procs"] == []
    assert srv.state["jetson"]["mode"] == "25W" and srv.state["jetson"]["disk"]["pct"] == 36
    assert set(srv.state["mac"]) == {"cpu", "ram_used", "ram_total", "disk_pct", "disk_free_gb"}
    assert set(srv.state["training"]) == {"act_cube", "smolvla_x", "orphan", "act_cube@jetson", "smolvla_x@jetson", "orphan@jetson"}
    assert events() == ["all lerobot processes finished", "cube@jetson: 13 episodes (+1)", "running: lerobot-dance, lerobot-record"]


def test_probe_loop_survives_ssh_failure(srv, fake, run_loop, logs):
    fake.on(lambda a: a[0] == "bash", MAC_PROBE)
    fake.on(lambda a: a[0] == "ssh", "", returncode=255, stderr="ssh: connect to host: timed out")
    run_loop(srv.probe_loop, 1)
    assert set(srv.state["datasets"]) == {"cube", "pretty"} and "mode" not in srv.state["jetson"]
