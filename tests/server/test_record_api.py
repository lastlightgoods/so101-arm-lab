"""record_start() / record_ctl() / record_apply() / record_reattach(): the commands sent to the edge host."""
import shlex, time
import pytest


def launch(fake):
    """The remote command of the nohup launch (the ssh call that runs record_ctl.py)."""
    return next(c for c in fake.ssh if "record_ctl.py" in c)


def inner(fake):
    """What `bash -lc` on the edge host receives as its one argument, after the remote shell unquotes it."""
    return shlex.split(launch(fake).split("bash -lc ", 1)[1].rsplit(" > /dev/null", 1)[0])[0]


@pytest.fixture
def remote_ok(fake):
    fake.on("record_ctl.py", "", 0)                     # the nohup launch
    fake.on("/tmp/so101_record_ctl", "", 0)             # control keys
    return fake


# ---- record_start ----
def test_start_resume_when_dataset_exists(srv, remote_ok, fake, events):
    fake.on("test -f", "yes\n")
    t0 = time.time()
    assert srv.record_start({"name": "cube", "task": "pick the cube", "episodes": 12, "episode_s": 30, "reset_s": 8}) == {"ok": True}
    assert fake.ssh[0] == "test -f ~/.cache/huggingface/lerobot/lastlight/cube/meta/info.json && echo yes || echo no"
    cmd = launch(fake)
    assert 'so101-record-more cube 12 "pick the cube"' in cmd and "so101-record cube" not in cmd
    assert "export SO101_EP_S=30 SO101_RESET_S=8;" in cmd and "source ~/so101/env.sh;" in cmd
    assert cmd.startswith("mkdir -p ~/so101/outputs/run; : > ~/so101/outputs/run/ui_record.log; rm -f /tmp/so101_record_ctl; nohup python3 ~/so101/record_ctl.py")
    assert cmd.endswith("> /dev/null 2>&1 &")
    R = srv.REC
    assert R["running"] and R["name"] == "cube" and R["task"] == "pick the cube" and R["episodes"] == 12 and R["episode"] == 0
    assert R["phase"] == "starting" and R["started"] == pytest.approx(t0, abs=1) and R["ep_started"] == 0 and R["paused_at"] == 0
    assert R["episode_s"] == 30 and R["reset_s"] == 8 and R["done"] is False and R["err"] == "" and R["_cal_noted"] is False
    assert R["done_total"] == 0 and R["episodes_target"] == 12 and R["chain_base"] is None and R["saved_total"] == 0 and R["_chain"] is False
    assert events() == ["RECORD cube · 12 episodes · 30s (dashboard)"]


def test_start_falls_back_to_new_dataset_when_absent(srv, remote_ok, fake, events):
    fake.on("test -f", "no\n")
    assert srv.record_start({"name": "cube", "task": "pick", "episodes": 5}) == {"ok": True}
    assert 'so101-record cube "pick" 5' in launch(fake) and "so101-record-more" not in launch(fake)
    assert events()[-1] == "RECORD cube: no existing dataset, creating a new one"


def test_start_resume_false_skips_the_check(srv, remote_ok, fake):
    assert srv.record_start({"name": "cube", "task": "pick", "resume": False})["ok"]
    assert all("test -f" not in c for c in fake.ssh) and 'so101-record cube "pick" 10' in launch(fake)


def test_start_sanitises_name_and_task(srv, remote_ok, fake):
    fake.on("test -f", "no\n")
    assert srv.record_start({"name": "my set/../x y!", "task": '  say "hi"; rm -rf ~  '})["ok"]
    assert srv.REC["name"] == "mysetxy" and srv.REC["task"] == "say 'hi'; rm -rf ~"
    # the whole inner command is shlex-quoted into one bash -lc argument: the task's quotes survive, nothing re-splits
    assert inner(fake) == 'source ~/so101/env.sh; export SO101_EP_S=25 SO101_RESET_S=10; so101-record mysetxy "say \'hi\'; rm -rf ~" 10'


def test_start_truncates_long_name_and_task(srv, remote_ok, fake):
    fake.on("test -f", "no\n")
    srv.record_start({"name": "a" * 100, "task": "t" * 300})
    assert len(srv.REC["name"]) == 60 and len(srv.REC["task"]) == 200


@pytest.mark.parametrize("p, want", [
    ({"episodes": 0, "episode_s": 1, "reset_s": 0}, (1, 5, 3)),
    ({"episodes": 999, "episode_s": 999, "reset_s": 999}, (200, 120, 60)),
    ({"episodes": "7", "episode_s": "40", "reset_s": "15"}, (7, 40, 15)),
    ({}, (10, 25, 10)),
])
def test_start_bounds(srv, remote_ok, fake, p, want):
    fake.on("test -f", "no\n")
    assert srv.record_start({"name": "x", "task": "y", **p})["ok"]
    assert (srv.REC["episodes"], srv.REC["episode_s"], srv.REC["reset_s"]) == want
    assert f"SO101_EP_S={want[1]} SO101_RESET_S={want[2]};" in launch(fake)


@pytest.mark.parametrize("p", [{"episodes": ""}, {"episode_s": "abc"}, {"reset_s": None}, {"episodes": "12.5"}])
def test_start_rejects_non_numeric_fields_instead_of_crashing(srv, fake, p):
    """An empty form field must come back as {"ok": False}, not as an unhandled ValueError inside the request handler."""
    r = srv.record_start({"name": "x", "task": "y", **p})
    assert r["ok"] is False and r["err"] and fake.calls == [] and srv.REC["running"] is False


@pytest.mark.parametrize("p", [{}, {"name": "x"}, {"task": "y"}, {"name": "!!!", "task": "y"}, {"name": "x", "task": "   "}])
def test_start_requires_name_and_task(srv, fake, p):
    assert srv.record_start(p) == {"ok": False, "err": "name and task are required"} and fake.calls == []


def test_start_refuses_while_running(srv, fake):
    srv.REC["running"] = True
    assert srv.record_start({"name": "x", "task": "y"}) == {"ok": False, "err": "already recording"} and fake.calls == []


def test_start_ssh_failure_clears_running(srv, fake, events):
    fake.on("test -f", "no\n"); fake.on("record_ctl.py", "", 255, "ssh: connect to host jetson: timed out\n")
    r = srv.record_start({"name": "x", "task": "y"})
    assert r == {"ok": False, "err": "ssh: connect to host jetson: timed out"}
    assert srv.REC["running"] is False and srv.REC["err"].endswith("timed out")
    assert events()[0].startswith("RECORD x · 10 episodes")


# ---- record_ctl ----
@pytest.mark.parametrize("cmd, key, spoken", [("next", "n", "Next."), ("rerecord", "r", "Redo. Reset first, then it records again."), ("stop", "q", "Stopping after this episode.")])
def test_ctl_keys_and_speech(srv, remote_ok, fake, said, events, cmd, key, spoken):
    srv.REC["running"] = True
    assert srv.record_ctl(cmd) == {"ok": True, "err": ""}
    assert fake.ssh == [f"printf {key} > /tmp/so101_record_ctl"] and said() == [spoken]
    assert events() == [f"RECORD {cmd.upper()} (dashboard)"]


def test_ctl_pause_maps_to_p_and_is_silent(srv, remote_ok, fake, said, events):
    srv.REC["running"] = True
    assert srv.record_ctl("pause") == {"ok": True, "err": ""}
    assert fake.ssh == ["printf p > /tmp/so101_record_ctl"] and said() == []   # busheal speaks pause/resume itself
    assert events() == ["RECORD PAUSE (dashboard)"]


def test_ctl_unknown_and_not_running(srv, fake, said):
    assert srv.record_ctl("dance") == {"ok": False, "err": "unknown command"}
    assert srv.record_ctl("next") == {"ok": False, "err": "not recording"}
    srv.REC["running"] = True
    assert srv.record_ctl("") == {"ok": False, "err": "unknown command"}
    assert fake.calls == [] and said() == []


def test_ctl_ssh_failure_is_reported_and_not_spoken(srv, fake, said):
    srv.REC["running"] = True; fake.on("/tmp/so101_record_ctl", "", 1, "fifo missing")
    assert srv.record_ctl("stop") == {"ok": False, "err": "fifo missing"} and said() == []


# ---- record_apply ----
def test_apply_stops_then_resumes_with_remaining_episodes(srv, remote_ok, fake, monkeypatch, said):
    srv.REC.update(running=True, name="cube", task="pick", episodes=10, episode=4, episodes_target=10, done_total=0, episode_s=25, reset_s=10, chain_base=7, saved_total=3)
    fake.on("test -f", "yes\n")
    monkeypatch.setattr(time, "sleep", lambda s: srv.REC.update(running=False))   # the recorder exits at the episode boundary
    assert srv.record_apply({"episodes": 12, "episode_s": 30, "task": ""}) == {"ok": True}
    assert fake.ssh[0] == "printf q > /tmp/so101_record_ctl"
    assert 'so101-record-more cube 9 "pick"' in launch(fake), "12 wanted - 3 already done (episode 4 was in progress)"
    assert "SO101_EP_S=30 SO101_RESET_S=10;" in launch(fake)
    R = srv.REC
    assert R["running"] and R["episodes"] == 9 and R["done_total"] == 3 and R["episodes_target"] == 12 and R["task"] == "pick"
    assert R["_chain"] is False and R["chain_base"] == 7 and R["saved_total"] == 3, "a chained start keeps the session totals"
    assert said() == ["Stopping after this episode."]


def test_apply_defaults_target_to_session_target(srv, remote_ok, fake, monkeypatch):
    srv.REC.update(running=True, name="cube", task="pick", episodes=10, episode=1, episodes_target=10)
    fake.on("test -f", "yes\n"); monkeypatch.setattr(time, "sleep", lambda s: srv.REC.update(running=False))
    assert srv.record_apply({"episodes": ""})["ok"]
    assert 'so101-record-more cube 10 "pick"' in launch(fake) and srv.REC["episodes_target"] == 10


def test_apply_when_recorder_does_not_stop(srv, remote_ok, fake, monkeypatch):
    srv.REC.update(running=True, name="cube", task="pick", episodes=10, episodes_target=10, episode=2)
    clock = {"t": time.time()}
    monkeypatch.setattr(time, "time", lambda: clock["t"])
    monkeypatch.setattr(time, "sleep", lambda s: clock.update(t=clock["t"] + 61))
    assert srv.record_apply({}) == {"ok": False, "err": "recorder did not stop"}
    assert len(fake.ssh) == 1 and srv.REC["running"] is True


def test_apply_requires_running(srv, fake):
    assert srv.record_apply({"episodes": 5}) == {"ok": False, "err": "not recording"} and fake.calls == []


# ---- record_reattach ----
PGREP = ('4242 /home/tc/so101/.venv/bin/python /home/tc/so101/.venv/bin/lerobot-record --robot.type=so101_follower '
         '--dataset.repo_id=lastlight/cube --dataset.num_episodes=12 --dataset.episode_time_s=30.0 --dataset.reset_time_s=8 '
         '--dataset.single_task="pick the cube" --dataset.push_to_hub=false\n')


def test_reattach_parses_running_command(srv, fake, logs):
    fake.on("pgrep -af", PGREP)
    t0 = time.time()
    assert srv.record_reattach() is True
    R = srv.REC
    assert R["running"] and R["name"] == "cube" and R["task"] == "pick the cube" and R["episodes"] == 12
    assert R["episode_s"] == 30 and R["reset_s"] == 8 and R["phase"] == "recording" and R["episode"] == 0
    assert R["episodes_target"] == 12 and R["done_total"] == 0 and R["chain_base"] is None and R["started"] == pytest.approx(t0, abs=1)
    assert any("reattached to running session cube (12 eps, 30s/8s)" in l for l in logs)


def test_reattach_nothing_running(srv, fake):
    fake.on("pgrep -af", "\n")
    assert srv.record_reattach() is False and srv.REC["running"] is False


def test_reattach_defaults_and_single_quotes(srv, fake):
    fake.on("pgrep -af", "99 lerobot-record --dataset.repo_id=lastlight/x --dataset.single_task='wave'\n")
    assert srv.record_reattach() is True
    assert (srv.REC["episodes"], srv.REC["episode_s"], srv.REC["reset_s"], srv.REC["task"]) == (10, 25, 10, "wave")


def test_reattach_survives_ssh_error(srv, fake, logs):
    fake.on("pgrep -af", lambda a, k: (_ for _ in ()).throw(TimeoutError("no route")))
    assert srv.record_reattach() is False and any("reattach failed" in l for l in logs)
