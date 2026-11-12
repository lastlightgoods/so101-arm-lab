"""The evaluate stage of tools/learn_loop.py (tools/learn_loop_eval.py): trials judged by the autoloop's own Runner.judge, and the guardrails that stop the arm path: workspace box on the
follower's read-back joints, the stop file, consecutive failures, a silent or hung or dead rollout. Runs on the ShortArm / FakeWorld rig with a FakeRollout for the Jetson and a virtual
clock; the ssh transport and the remote script are tested with a fake Popen. Nothing moves, nothing connects."""
import os, subprocess
import pytest
from learn_loop_helpers import LL, FF, EV, PRE, run, state, client, events, lrt  # noqa: F401


@pytest.fixture
def ready(tmp_path):
    """The record..deploy stages done once (fake train, private registry, local Jetson stand-in) so a test can go straight to `evaluate`."""
    assert run(tmp_path, *PRE) == 0; return tmp_path


def ev_args(tmp, rt, *extra, name="t1"): return run(tmp, "evaluate", runtime=rt, extra=("--dry-run", "--trials", "6", "--trial-s", "10", "--cycles", "6", "--steps", "1000", "--save-freq", "500", *extra))


def trials(tmp): return __import__("json").load(open(tmp / "ll" / "t1" / "eval.json"))


# ---------------------------------------------------------------- a normal session
def test_trials_are_judged_by_the_autoloop_judge_and_the_arm_parks_before_each_judgement(ready, lrt, capsys):
    rt = lrt(["ok", "miss", "drop", "ok"]); assert ev_args(ready, rt, "--trials", "4") == 0
    ts = trials(ready)["trials"]; assert [t["ok"] for t in ts] == [True, False, False, True]
    assert ts[0]["landed_err_mm"] <= 15 and ts[0]["picked"] and ts[1]["picked"] is False and ts[1]["moved_mm"] < 20 and ts[2]["picked"] and not ts[2]["landed"] and ts[2]["landed_err_mm"] > 15
    assert all(t["scene_live"] and t["rollout"]["rc"] == 0 and t["policy_s"] == 10.0 and t["rollout"]["fixes"] > 10 for t in ts)
    s = trials(ready)["summary"]; assert s["trials"] == 4 and s["landed"] == 2 and s["success_rate"] == 0.5 and s["end"] == "trials" and s["median_placed_err_mm"] <= 15
    rests = [i for i, m in enumerate(rt.moves) if m == ["rest"]]; assert len(rests) == 4, "one park per trial, before the judge"
    assert len(rt.transport.scripts) == 4 and all("--robot.max_relative_target=2" in sc and "--duration=10" in sc for sc in rt.transport.scripts)
    assert state(ready)["stages"]["evaluate"]["status"] == "done"


def test_a_nut_already_on_the_target_is_reset_away_first(ready, lrt, capsys):
    rt = lrt(["ok", "ok"]); rt.rig.world.nut = [9.0 * 25.4, 6.0 * 25.4]; assert ev_args(ready, rt, "--trials", "2") == 0
    t = trials(ready)["trials"][0]; assert t["pre_reset"]["ok"] is not None and t["nut_in"] != [9.0, 6.0] and t["ok"]


def test_a_resumed_evaluation_continues_after_the_last_finished_trial(ready, lrt, capsys):
    rt = lrt(["ok", "ok", "ok", "miss", "miss", "miss"]); assert ev_args(ready, rt, "--trials", "6", "--max-consecutive-failures", "2") == 2
    st = state(ready)["stages"]["evaluate"]; assert st["status"] == "failed" and "consecutive_failures" in st["error"] and len(trials(ready)["trials"]) == 5
    rt2 = lrt(["ok"]); rt2.rig.world.nut = [6.0 * 25.4, 8.0 * 25.4]; assert ev_args(ready, rt2, "--trials", "6", "--max-consecutive-failures", "9") == 0
    ts = trials(ready)["trials"]; assert [t["trial"] for t in ts] == [1, 2, 3, 4, 5, 6] and len(rt2.transport.scripts) == 1, "trials 1-5 were not repeated"


# ---------------------------------------------------------------- the guardrails
def test_leaving_the_workspace_box_stops_the_rollout_and_ends_the_session_without_another_arm_command(ready, lrt, capsys):
    rt = lrt(["ok", "escape", "ok"], escape_at_s=3.0); assert ev_args(ready, rt, "--trials", "6") == 2
    ts = trials(ready)["trials"]; assert [t["ok"] for t in ts] == [True, False] and ts[1]["trip"] == "workspace" and ts[1]["policy_trip"] is True and ts[1]["policy_s"] < 10
    assert rt.transport.stops >= 1 and len(rt.transport.scripts) == 2, "so101-stop was sent and trial 3 never started"
    assert rt.moves[-1] != ["rest"] and rt.moves.count(["rest"]) == 1, "the last plan the arm ran was trial 1's reset pick: the escape trial parked nothing and reset nothing"
    trip = [e for e in events(capsys) if e.get("event") == "trip"]; assert trip and trip[0]["reason"] == "workspace" and trip[0]["rest"] is False and trip[0]["detail"]["phase"] == "policy" and "outside the mat" in trip[0]["detail"]["why"]
    assert state(ready)["stages"]["evaluate"]["status"] == "failed" and "workspace" in state(ready)["stages"]["evaluate"]["error"]


def test_the_stop_file_during_a_rollout_stops_the_arm_cleanly(ready, lrt, capsys):
    stop = ready / "ll" / "t1" / "stop"
    rt = lrt(["ok", "ok"], on_tick=lambda n, el: stop.write_text("") if n == 2 and el >= 2.0 else None); assert ev_args(ready, rt, "--trials", "4") == 2
    ts = trials(ready)["trials"]; assert len(ts) == 2 and ts[1]["trip"] == "stop_file" and rt.transport.stops >= 1 and len(rt.transport.scripts) == 2
    e = [x for x in events(capsys) if x.get("event") == "stop"]; assert e and e[0]["reason"] == "stop_file"
    assert not any(m == ["rest"] for m in rt.moves[-1:]), "no park after a stop"


def test_the_stop_file_before_a_trial_means_no_rollout_at_all(ready, lrt, capsys):
    (ready / "ll" / "t1").mkdir(exist_ok=True); (ready / "ll" / "t1" / "stop").write_text("")
    rt = lrt(["ok"]); assert ev_args(ready, rt) == 2 and rt.transport.scripts == [] and rt.moves == []


def test_three_failed_trials_in_a_row_end_the_session(ready, lrt, capsys):
    rt = lrt(["miss", "miss", "miss", "ok", "ok"]); assert ev_args(ready, rt, "--trials", "6") == 2
    assert len(trials(ready)["trials"]) == 3 and "consecutive_failures" in state(ready)["stages"]["evaluate"]["error"] and len(rt.transport.scripts) == 3


def test_a_rollout_that_never_reports_joints_is_not_called_clean(ready, lrt, capsys):
    rt = lrt(["mute"]); assert ev_args(ready, rt, "--trials", "2") == 2
    t = trials(ready)["trials"][0]; assert t["trip"] == "unwatched" and t["rollout"]["fixes"] == 0


def test_a_rollout_the_owner_kills_with_so101_stop_is_a_clean_stop_not_a_failed_trial(ready, lrt, capsys):
    rt = lrt(["die"]); assert ev_args(ready, rt, "--trials", "2") == 2
    t = trials(ready)["trials"][0]; assert t["trip"] == "stop_arm" and t["ok"] is False
    assert [e for e in events(capsys) if e.get("event") == "stop"], "clean=True prints a stop line, not a trip line"


def test_a_crashed_rollout_is_a_fault_and_a_hung_one_is_stopped(ready, lrt, capsys):
    rt = lrt(["fail"]); assert ev_args(ready, rt, "--trials", "2") == 2 and trials(ready)["trials"][0]["trip"] == "rollout_failed"
    rt = lrt(["hang"]); assert ev_args(ready, rt, "--trials", "2", name="t1") == 2
    assert [t["trip"] for t in trials(ready)["trials"]] == ["rollout_failed", "rollout_hung"]


# ---------------------------------------------------------------- the watch, the transport, the remote script
def test_rolloutwatch_grace_staleness_and_box(autorig, al, gr):
    r = autorig(); lim = gr.Limits(stop_file=str(r.tmp / "s")); guard = gr.Guard(r.cc, r.table, lim, r.ctx); w = EV.RolloutWatch(gr, guard, r.cc, r.table, r.ctx)
    r.world.follower = {"pos": {}, "t": r.clock.now()}; assert w.check(5.0) is None, "before the first fix the arm is not under the policy yet"
    assert w.check(31.0).reason == "state_lost"
    w = EV.RolloutWatch(gr, guard, r.cc, r.table, r.ctx); r.world.follower = {"pos": dict(FF.HOVER), "mode": "deg", "t": r.clock.now()}; assert w.check(1.0) is None and w.worst[2] > 0
    r.clock.sleep(10); assert w.check(11.0).reason == "state_lost", "a fix 10 s old is stale"
    r.world.follower = {"pos": dict(FF.ESCAPE), "mode": "deg", "t": r.clock.now()}; t = w.check(12.0); assert t.reason == "workspace" and t.detail["phase"] == "policy"
    r.world.follower = {"pos": dict(FF.HOVER), "mode": "raw", "t": r.clock.now()}; w2 = EV.RolloutWatch(gr, guard, r.cc, r.table, r.ctx); assert w2.check(40.0).reason == "state_lost", "joints that are not in degrees cannot be put through the kinematics"


class FakePopen:
    def __init__(self, lines=None, ends_after=3): self.polls = 0; self.ends_after = ends_after; self.killed = False; self.stdin = type("S", (), {"written": "", "write": lambda s, x: setattr(s, "written", s.written + x), "close": lambda s: None})()
    def poll(self):
        self.polls += 1; return 0 if self.polls > self.ends_after else None
    def kill(self): self.killed = True


def make_transport(popen, clock):
    ran = []
    return EV.SshTransport("jetson-test", popen=lambda *a, **k: popen, run=lambda *a, **k: ran.append((a, k)), clock=clock.now, sleep=clock.sleep), ran


def test_ssh_transport_runs_the_script_over_stdin_and_stops_on_a_trip(lrt):
    clock = FF.Clock(); p = FakePopen(ends_after=99); tr, ran = make_transport(p, clock); script = "echo hi\n"
    out = tr.rollout(script, 10, lambda el: "TRIP" if el >= 1.0 else None); assert out["trip"] == "TRIP" and p.stdin.written == script and out["overrun"] is False
    assert ran and "so101-stop" in ran[0][1]["input"] and "jetson-test" in ran[0][0][0], "so101-stop went to the Jetson over ssh"
    p2 = FakePopen(ends_after=3); tr2, ran2 = make_transport(p2, FF.Clock()); out = tr2.rollout(script, 10, lambda el: None); assert out["rc"] == 0 and out["trip"] is None and out["overrun"] is False and out["elapsed_s"] == pytest.approx(0.75, abs=0.06) and ran2 == []
    p3 = FakePopen(ends_after=9999); tr3, ran3 = make_transport(p3, FF.Clock()); out = tr3.rollout(script, 5, lambda el: None); assert out["overrun"] is True and ran3 and p3.killed


def test_the_remote_script_is_the_dashboards_policy_button_with_the_speed_cap(monkeypatch):
    sc = EV.rollout_script("so101-nutpick", "act", 100, 25, ["front", "top", "wrist"])
    for must in ('source "$HOME/so101/env.sh"', "so101-nudge", "so101-cams-take", "lerobot-rollout --strategy.type=base", "--inference.type=rtc", "--inference.queue_threshold=90", '--policy.path="$HOME/so101/policies/so101-nutpick/current"', "--policy.device=cuda",
                 "--robot.max_relative_target=2", "--robot.disable_torque_on_disconnect=false", '--robot.cameras="$SO101_CAMS"', "--duration=25", "--display_data=false", "so101-cams-release", "exit $rc"): assert must in sc, must
    assert sc.index("so101-cams-take") < sc.index("lerobot-rollout") < sc.index("so101-cams-release") and "/Users/" not in sc and "Pick up the nut and place it on the target" in sc
    assert 'SO101_CAMS_2' in EV.rollout_script("p", "act", 50, 10, ["top", "wrist"]) and "--inference.queue_threshold=45" in EV.rollout_script("p", "act", 50, 10, ["top", "wrist"])
    vla = EV.rollout_script("p", "smolvla", 50, 10, ["top", "wrist"]); assert "--rename_map" in vla and "execution_horizon=10" in vla and "expandable_segments" in vla
    with pytest.raises(ValueError): EV.rollout_script("p", "act", 100, 10, ["top"])


def test_checkpoint_images_come_from_the_checkpoints_config(tmp_path):
    d = FF.make_ckpt(str(tmp_path / "c"), images=("top", "wrist")); assert EV.ckpt_images(d) == ["top", "wrist"] and EV.ckpt_images(str(tmp_path / "none")) == []
