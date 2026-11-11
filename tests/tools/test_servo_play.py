"""tools/servo_play.py and jetson/servo_probe.py against the fake bus: the analysis recovers the play the fake was built with, P is always restored
(exception, Ctrl-C, abort), the voltage / temperature limits abort, and nothing but P_Coefficient of the shoulder lift is ever written, never with Lock=0."""
import json, os, re, sys, types
import numpy as np
import pytest
from servohelpers import Spy, ROOT, FIXTURE_TABLE

PT = {"joint": "shoulder_lift", "scales": [1.0, 1.5, 2.0], "pose": "far", "move_to": "near"}


def summarise(sv, sv_cc, sv_table, ev, plan): return sv.analyse(ev, sv_cc, sv_table, plan)


# ---------------------------------------------------------------- the numbers come back
def test_recovers_backlash_and_compliance_per_joint(play, sv, sv_cc, sv_table):
    rig, ev, code, plan, _ = play(rig_kw={"backlash_deg": {"shoulder_lift": 1.5, "elbow_flex": 0.3, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 0},
                                          "compliance": {"shoulder_lift": 0.009, "elbow_flex": 0.002}})
    assert code == 0 and ev[-1]["event"] == "done"
    s = summarise(sv, sv_cc, sv_table, ev, plan); j = s["joints"]
    assert j["shoulder_lift"]["backlash_deg_mean"] == pytest.approx(1.5, abs=0.05) and j["elbow_flex"]["backlash_deg_mean"] == pytest.approx(0.3, abs=0.05)
    assert j["shoulder_lift"]["compliance_deg_per_100load"] == pytest.approx(-0.9, abs=0.06)   # deg per 100 load units, negative = sags toward lower angle
    assert j["elbow_flex"]["compliance_deg_per_100load_origin"] == pytest.approx(-0.2, abs=0.01) and j["shoulder_lift"]["compliance_deg_per_100load_origin"] == pytest.approx(-0.9, abs=0.02)
    assert "slope_note" in j["elbow_flex"] and "slope_note" not in j["shoulder_lift"]
    assert j["shoulder_lift"]["fit_r2"] > 0.99 and abs(j["shoulder_lift"]["deflection_deg_at_zero_load"]) < 0.05
    for r in s["points"]: assert r["scatter_deg"] < 0.05 and r["n_below"] == r["n_above"] == 2


def test_mm_at_the_tip_is_the_fk_lever(play, sv, sv_cc, sv_table):
    rig, ev, code, plan, _ = play(rig_kw={"backlash_deg": 1.0})
    s = summarise(sv, sv_cc, sv_table, ev, plan)
    for r in s["points"]:
        pose = next(p for p in plan["poses"] if p["name"] == r["pose"])
        a, b = dict(pose["joints"]), dict(pose["joints"]); a[r["joint"]] -= 0.5; b[r["joint"]] += 0.5
        lever = float(np.linalg.norm(sv.tip_xyz(sv_cc, sv_table, b) - sv.tip_xyz(sv_cc, sv_table, a)))   # mm per degree, independently
        assert r["tip_mm_per_deg"] == pytest.approx(lever, rel=0.02) and r["backlash_mm"] == pytest.approx(abs(r["backlash_deg"]) * lever, abs=0.02)
        assert r["backlash_mm"] > 0 and r["deflection_mm"] >= r["deflection_vertical_mm"] - 1e-6


def test_events_carry_every_register_asked_for(play):
    rig, ev, code, plan, _ = play()
    pts = [e for e in ev if e["event"] == "point"]
    assert len(pts) == 3 * 2 * 2 * 2   # poses x joints x reps x directions
    for e in pts: assert {"pos", "load", "current", "temp_c", "volts", "dir", "pose", "joint", "target"} <= set(e)
    assert {e["dir"] for e in pts} == {"below", "above"}
    assert next(e for e in ev if e["event"] == "settings")["regs"]["shoulder_lift"]["P_Coefficient"] == 16


def test_p_values_cut_sag_in_the_fake_and_baseline_is_unwritten(play, sv, sv_cc, sv_table):
    rig, ev, code, plan, _ = play(p_test=dict(PT), rig_kw={"p0": 16})
    pv = [e for e in ev if e["event"] == "p_value"]
    assert [e["p"] for e in pv] == [16, 24, 32]
    assert abs(pv[1]["sag_deg"]) < abs(pv[0]["sag_deg"]) and abs(pv[2]["sag_deg"]) < abs(pv[1]["sag_deg"])
    assert pv[2]["current_peak_move"] > pv[0]["current_peak_move"]
    s = summarise(sv, sv_cc, sv_table, ev, plan)
    assert s["p_values"][0]["sag_change_pct"] == pytest.approx(0.0) and s["p_values"][2]["sag_change_pct"] < -20
    assert [w[2] for w in rig.writes] == [24, 32, 16]   # two trials and the restore; the baseline wrote nothing


# ---------------------------------------------------------------- restore
def p_of(rig): return rig.regs["shoulder_lift"]["P_Coefficient"]


def test_restore_after_normal_run(play):
    rig, ev, code, plan, _ = play(p_test=dict(PT))
    assert code == 0 and p_of(rig) == 16
    r = [e for e in ev if e["event"] == "restore"][-1]; assert r["ok"] and r["original"] == 16 and r["readback"] == 16 and r["wrote"]


def test_restore_after_exception_mid_p_test(play):
    def wrap(rig, clk): return Spy(rig, clk)   # first a clean run to count the sends, then die 93 % of the way in (inside the last P trial)
    rig, ev, code, plan, spy = play(p_test=dict(PT), rig_wrap=wrap)
    n = len(spy.steps); assert code == 0
    def wrap2(rig, clk): return Spy(rig, clk, raise_on_send=int(n * 0.93))
    rig, ev, code, plan, spy = play(p_test=dict(PT), rig_wrap=wrap2)
    assert isinstance(code, RuntimeError)                      # the exception is NOT swallowed ...
    assert p_of(rig) == 16                                     # ... and P is back
    r = [e for e in ev if e["event"] == "restore"][-1]; assert r["ok"] and r["readback"] == 16 and r["wrote"]
    assert any(e["event"] == "p_value" for e in ev)            # it died after at least one trial


def test_restore_after_ctrl_c(play):
    def wrap(rig, clk): return Spy(rig, clk)
    _, _, _, _, spy = play(p_test=dict(PT), rig_wrap=wrap); n = len(spy.steps)
    rig, ev, code, plan, spy = play(p_test=dict(PT), rig_wrap=lambda r, c: Spy(r, c, interrupt_on_send=int(n * 0.9)))
    assert code == 2 and p_of(rig) == 16
    assert ev[-2]["event"] == "abort" and ev[-2]["reason"] == "interrupted" and ev[-1]["event"] == "restore" and ev[-1]["ok"]


def test_restore_unverified_exits_3(play, sp):
    class Sticky(sp.FakeRig):
        dropped = 0
        def _write(self, name, joint, value):
            if value == 16 and self.writes: Sticky.dropped += 1; return   # a restore write the servo never takes
            super()._write(name, joint, value)
    plan, _ = play(p_test=dict(PT))[3], None
    clk = sp.VClock(); first = plan["poses"][0]["joints"]; sticky = Sticky(clock=clk, start={**first, "gripper": 0.0}); out = []
    code = sp.run_play(plan, sticky, out.append, clk.now, clk.sleep, 50.0)
    assert code == 3 and out[-1]["event"] == "restore" and out[-1]["ok"] is False and out[-1]["readback"] != 16 and Sticky.dropped == 3   # 3 tries, then it says so


# ---------------------------------------------------------------- aborts
def test_undervoltage_aborts_before_the_next_value_and_restores(play):
    # volts fall with the summed servo current: fine at P=16, below 9.0 V once P is raised and the arm moves
    rig, ev, code, plan, _ = play(p_test=dict(PT), rig_kw={"volt_per_current": 0.12})
    ab = [e for e in ev if e["event"] == "abort"]
    assert code == 2 and ab and ab[0]["reason"] == "undervoltage" and ab[0]["detail"]["volts"] < 9.0 and ab[0]["detail"]["limit"] == 9.0
    assert p_of(rig) == 16 and [e for e in ev if e["event"] == "restore"][-1]["ok"]
    assert not any(e["event"] == "done" for e in ev) and len([e for e in ev if e["event"] == "p_value"]) < 3 and ev[-1]["event"] == "restore"


def test_limits_come_from_the_plan(play):
    rig, ev, code, plan, _ = play(limits={"min_volts": 11.9}, rig_kw={"volt0": 118})   # 11.8 V idle < 11.9 limit: refuses at the first reading
    assert code == 2 and ev[-1]["event"] == "abort" and ev[-1]["reason"] == "undervoltage"
    assert not any(e["event"] == "point" for e in ev)


def test_overtemperature_aborts_the_sweep(play):
    rig, ev, code, plan, _ = play(rig_kw={"temp0": 55, "temp_slope": 0.25})   # 60 C reached partway
    ab = ev[-1]; assert code == 2 and ab["event"] == "abort" and ab["reason"] == "overtemperature" and ab["detail"]["temp_c"] > 60 and ab["detail"]["limit"] == 60.0
    assert any(e["event"] == "point" for e in ev) and rig.writes == []


def test_overload_aborts(play):
    rig, ev, code, plan, _ = play(limits={"max_load": 100})
    assert code == 2 and ev[-1]["reason"] == "overload"


def test_oscillation_stops_the_climb_and_restores(play):
    rig, ev, code, plan, _ = play(p_test=dict(PT), rig_kw={"p_osc": 16})   # the fake chatters above P = 16
    stop = [e for e in ev if e["event"] == "p_stop"]
    assert stop and stop[0]["p"] == 24 and code == 0 and p_of(rig) == 16
    assert [e["p"] for e in ev if e["event"] == "p_value"] == [16, 24]


# ---------------------------------------------------------------- never EEPROM
def test_only_p_of_the_shoulder_lift_is_written_and_always_locked(play):
    rig, ev, code, plan, _ = play(p_test=dict(PT))
    assert rig.writes and {(w[0], w[1], w[3]) for w in rig.writes} == {("P_Coefficient", "shoulder_lift", 1)}
    assert rig.eeprom_writes == []
    assert all(1 <= w[2] <= 64 for w in rig.writes) and rig.regs["shoulder_lift"]["Lock"] == 1 and rig.regs["shoulder_lift"]["Torque_Enable"] == 1


def test_sweep_without_p_test_writes_nothing(play):
    rig, ev, code, plan, _ = play()
    assert rig.writes == [] and rig.eeprom_writes == [] and code == 0


def test_unlocked_servo_is_never_written(play):
    rig, ev, code, plan, _ = play(p_test=dict(PT), rig_kw={"lock": 0})
    assert code == 2 and rig.writes == [] and rig.eeprom_writes == []
    assert [e for e in ev if e["event"] == "abort"][0]["reason"] == "lock_not_set"


def test_write_guard_refuses_other_joints_and_big_values(sp):
    rig = sp.FakeRig()
    with pytest.raises(ValueError): rig.write_p("elbow_flex", 20, "shoulder_lift")
    with pytest.raises(ValueError): rig.write_p("shoulder_lift", 65, "shoulder_lift")
    with pytest.raises(ValueError): rig.write_p("shoulder_lift", 0, "shoulder_lift")
    assert rig.writes == []
    src = open(os.path.join(ROOT, "jetson", "servo_probe.py")).read()
    assert re.findall(r'_write\(\s*"([A-Za-z_]+)"', src) == ["P_Coefficient"]   # the only literal register name any write names
    assert src.count("bus.write(") == 1   # one raw bus write, in RealRig._write; no bus.disable_torque / enable_torque (those write Lock) anywhere
    assert "disable_torque(" not in src.replace("disable_torque_on_disconnect", "") and "enable_torque(" not in src


def test_ignored_write_is_reported_not_assumed(play):
    rig, ev, code, plan, _ = play(p_test=dict(PT), rig_kw={"lock_mode": "ignore"})
    ig = [e for e in ev if e["event"] == "p_write_ignored"]
    assert ig and ig[0]["wanted"] == 24 and ig[0]["readback"] == 16 and code == 0 and p_of(rig) == 16
    assert len([e for e in ev if e["event"] == "p_value"]) == 1


# ---------------------------------------------------------------- motion: speed, box, plan
def test_speed_never_exceeds_50_deg_s(play):
    rig, ev, code, plan, spy = play(p_test=dict(PT), rig_wrap=lambda r, c: Spy(r, c))
    worst = 0.0
    for (t0, a), (t1, b) in zip(spy.steps, spy.steps[1:]):
        if t1 - t0 > 1e-6: worst = max(worst, max(abs(b[j] - a[j]) for j in a) / (t1 - t0))
    assert 5.0 < worst <= 50.0 + 1e-6


def test_speed_cap_is_hard(sp):
    rig = sp.FakeRig(); steps = []; orig = rig.send
    rig.send = lambda c: (steps.append((rig.clock.now(), dict(c))), orig(c))[1]
    cmd = {j: 0.0 for j in sp.J}; tgt = dict(cmd); tgt["shoulder_lift"] = 40.0
    sp.ease_move(rig, cmd, tgt, 200.0, rig.clock.sleep)   # asked for 200 deg/s: capped to 50
    sp_ = max(abs(b["shoulder_lift"] - a["shoulder_lift"]) / (t1 - t0) for (t0, a), (t1, b) in zip(steps, steps[1:]))
    assert sp_ <= 50.0 + 1e-6


def test_every_commanded_point_is_inside_the_box_and_above_the_mat(play, sv, sv_cc, sv_table):
    gr = sv.guardrails(); lim = gr.Limits(margin_mm=20.0, floor_mm=0.0)
    rig, ev, code, plan, spy = play(rig_wrap=lambda r, c: Spy(r, c))
    bad = [gr.box_check(*gr.fk_mat_mm(sv_cc, sv_table, c), lim) for _, c in spy.steps]   # EVERY step of the run, the joint-space interpolation between poses included
    assert len(spy.steps) > 500 and [b for b in bad if b] == []
    assert min(gr.fk_mat_mm(sv_cc, sv_table, c)[2] for _, c in spy.steps) > 10.0   # never near the mat
    bad = []
    for p in plan["poses"]:
        for jn, d in p["approach_deg"].items():
            for s in (-1, 0, 1):
                x, y, h = gr.fk_mat_mm(sv_cc, sv_table, {**p["joints"], jn: p["joints"][jn] + s * d})
                why = gr.box_check(x, y, h, lim); bad += [why] if why else []
                assert h >= 0.0
    assert bad == []
    # the targets actually commanded by the run are the plan's, plus or minus the approach
    pts = {(e["pose"], e["joint"]): e["target"] for e in ev if e["event"] == "point"}
    for p in plan["poses"]:
        for jn in p["approach_deg"]: assert pts[(p["name"], jn)] == pytest.approx(p["joints"][jn], abs=1e-3)


def test_plan_shrinks_or_drops_what_does_not_fit(sv, sv_cc, sv_table):
    plan, notes = sv.build_plan(sv_cc, sv_table, {"near": 150.0, "far": 290.0}, approach_deg=60.0)
    assert all(d < 60.0 for p in plan["poses"] for d in p["approach_deg"].values()) or notes
    with pytest.raises(SystemExit): sv.build_plan(sv_cc, sv_table, {"x": 330.0}, height_mm=40.0)   # out of reach: no pose at all


# ---------------------------------------------------------------- the CLI
def run_cli(sv, tmp_path, *extra):
    out = tmp_path / "out.json"
    code = sv.main(["--table", FIXTURE_TABLE, "--no-mlflow", "--out", str(out), "--reps", "1", *extra]); return code, json.load(open(out))


def test_cli_dry_run_prints_json_lines(sv, tmp_path, capsys):
    code, doc = run_cli(sv, tmp_path, "--dry-run", "--p-test", "--session", "t")
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and "plan" in lines[0] and "summary" in lines[-1] and doc["summary"]["dry_run"] is True
    assert doc["extra"]["eeprom_writes"] == [] and doc["summary"]["restore"]["ok"]


def test_cli_cli_flags_recover_the_fake(sv, tmp_path):
    code, doc = run_cli(sv, tmp_path, "--dry-run", "--fake-backlash", "1.1", "--fake-compliance-lift", "0.005")
    j = doc["summary"]["joints"]; assert j["shoulder_lift"]["backlash_deg_mean"] == pytest.approx(1.1, abs=0.05) and j["shoulder_lift"]["compliance_deg_per_100load"] == pytest.approx(-0.5, abs=0.05)


def test_cli_refuses_live_without_tc_at_stop_and_speed_above_cap(sv, tmp_path):
    with pytest.raises(SystemExit, match="tc-at-stop"): sv.main(["--table", FIXTURE_TABLE])
    with pytest.raises(SystemExit, match="max-deg-s"): sv.main(["--table", FIXTURE_TABLE, "--dry-run", "--max-deg-s", "80"])


def test_plan_only_moves_nothing(sv, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sv.subprocess, "run", lambda *a, **k: pytest.fail("ssh/scp in --plan-only")); monkeypatch.setattr(sv.subprocess, "Popen", lambda *a, **k: pytest.fail("ssh in --plan-only"))
    assert sv.main(["--table", FIXTURE_TABLE, "--plan-only"]) == 0 and "plan" in json.loads(capsys.readouterr().out.splitlines()[0])


def test_live_run_is_scp_plus_ssh_like_pick_loop(sv, sp, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sv.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    class P:
        def __init__(self, cmd, **k):
            calls.append(cmd); plan = json.load(open("/tmp/servo_plan_local.json"))
            clk = sp.VClock(); first = plan["poses"][0]["joints"]; ev = []
            sp.run_play(plan, sp.FakeRig(clock=clk, start={**first, "gripper": 0.0}), ev.append, clk.now, clk.sleep, 50.0)
            self.stdout = [json.dumps(e) + "\n" for e in ev]; self.stderr = types.SimpleNamespace(read=lambda: "")
        def wait(self, timeout=None): return 0
        def terminate(self): pass
    monkeypatch.setattr(sv.subprocess, "Popen", P)
    code, doc = run_cli(sv, tmp_path, "--tc-at-stop", "--session", "live")
    ssh = [c for c in calls if c[0] == "ssh"][0]; scps = [c for c in calls if c[0] == "scp"]
    assert code == 0 and "servo_probe.py --plan /tmp/servo_plan.json --max-deg-s 50" in ssh[-1] and ssh[3] == "jetson" and len(scps) == 2
    assert any(c[-2].endswith("servo_probe.py") for c in scps)


def test_mlflow_run_in_the_servo_play_experiment(sv, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    uri = "sqlite:///" + str(tmp_path / "mlflow.db"); monkeypatch.setenv("SO101_MLFLOW_URI", uri)
    sv.main(["--table", FIXTURE_TABLE, "--dry-run", "--reps", "1", "--p-test", "--out", str(tmp_path / "o.json"), "--session", "ml"])
    from mlflow import MlflowClient
    c = MlflowClient(tracking_uri=uri); e = c.get_experiment_by_name("so101-servo-play-dry"); assert e is not None and c.get_experiment_by_name("so101-servo-play") is None
    r = c.search_runs([e.experiment_id])[0]
    assert r.data.metrics["backlash_deg_mean_shoulder_lift"] == pytest.approx(0.8, abs=0.05) and "p_sag_deg" in r.data.metrics and r.data.tags["restore_ok"] == "True"


# ---------------------------------------------------------------- the Jetson entry point
def test_probe_main_holds_arms_busy_refuses_lerobot_and_disconnects(sp, tmp_path, monkeypatch, sv, sv_cc, sv_table):
    plan, _ = sv.build_plan(sv_cc, sv_table, reps=1); pf = tmp_path / "plan.json"; pf.write_text(json.dumps(plan))
    busy = tmp_path / "arms_busy"; monkeypatch.setattr(sp, "BUSY", str(busy)); seen = {}
    clk = sp.VClock(); first = plan["poses"][0]["joints"]; fake = sp.FakeRig(clock=clk, start={**first, "gripper": 0.0})
    robot = types.SimpleNamespace(disconnect=lambda: seen.setdefault("disc", busy.exists()))
    monkeypatch.setitem(sys.modules, "pick_seq", types.SimpleNamespace(connect=lambda: (seen.setdefault("busy_at_connect", busy.exists()), robot)[1]))
    monkeypatch.setattr(sp.time, "sleep", lambda s: None); monkeypatch.setattr(sp, "RealRig", lambda r: fake)
    monkeypatch.setattr(sp.time, "time", clk.now); monkeypatch.setattr(sys, "argv", ["servo_probe.py", "--plan", str(pf)])
    monkeypatch.setattr(sp, "lerobot_running", lambda: True)
    with pytest.raises(SystemExit, match="lerobot job"): sp.main()
    assert not busy.exists() and "busy_at_connect" not in seen
    monkeypatch.setattr(sp, "lerobot_running", lambda: False)
    monkeypatch.setattr(fake.clock, "sleep", fake.clock.sleep)
    with pytest.raises(SystemExit) as e: sp.main()
    assert e.value.code == 0 and seen["busy_at_connect"] is True and seen["disc"] is True and not busy.exists()
