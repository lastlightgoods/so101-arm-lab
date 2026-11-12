"""The wrist tilt and the pregrasp_up approach (11 Nov 2026), checked against the real kinematics of the 9 Nov arm fit through ShortArm's /api/ik. No dashboard, no ssh, no arm.
  * GRASP_TILT_DEG on pregrasp, grasp, close1, close2 (and the --fast-close close): the IK target is shifted back by the tilt's own tip shift, so the tilted tip lands where the
    untilted one would; every other waypoint, and tilt 0, are the plan before the tilt;
  * the grasp and close are commanded at 1.0 mm by the IK; the flag and the autoloop runner carry the tilt to build_plan;
  * pregrasp_up: pregrasp's final command with shoulder_lift and elbow_flex APPROACH_DEG lower, never below pregrasp's height on the way down, never learned."""
import json, math
import numpy as np, pytest
from pickhelpers import ARM, PROBE_NUT_IN, ShortArm, drive

TILTED = ("pregrasp", "grasp", "close1", "close2"); UNTILTED = ("hover_pre", "hover_nut", "lift", "hover_tgt", "place", "open", "lift2", "rest")
SPOTS = [PROBE_NUT_IN, (8.0, 5.0), (14.0, 8.0), (10.0, 4.5), (12.5, 9.0)]
def stages(out): return {o["stage"]: o for o in out if "stage" in o}
def records(out): return [o for o in out if "picked" in o]


@pytest.fixture
def plans(pl, cc, table, monkeypatch):
    arm = ShortArm(cc, table); monkeypatch.setattr(pl, "api", arm.api)
    def make(tilt, nut=PROBE_NUT_IN, tgt=(9.0, 6.0), **kw): return {w["name"]: w for w in pl.build_plan(list(nut), list(tgt), kw.pop("biases", {}), tilt_deg=tilt, cc=cc, table=table, **kw)["waypoints"]}
    make.arm = arm; return make


@pytest.mark.parametrize("nut", SPOTS)
def test_the_tilted_tip_lands_where_the_untilted_one_would(pl, cc, table, plans, nut):
    p0, p1 = plans(0.0, nut), plans(pl.GRASP_TILT_DEG, nut)
    for n in TILTED:
        x0, y0, _ = pl.fk_mat(cc, table, p0[n]["desired"]); x1, y1, _ = pl.fk_mat(cc, table, p1[n]["desired"])
        assert math.hypot(x1 - x0, y1 - y0) * 25.4 < 0.3, f"{n}: the tilted tip is {math.hypot(x1 - x0, y1 - y0) * 25.4:.2f} mm off the untilted one"
        h0, h1 = pl.fk_mat(cc, table, p0[n]["desired"])[2], pl.fk_mat(cc, table, p1[n]["desired"])[2]; assert abs(h1 - h0) < 0.3, f"{n}: the tilted tip is {h1 - h0:+.2f} mm off the untilted height"


def test_the_tilted_wrist_flex_is_the_ik_at_the_shifted_point_plus_the_tilt_exactly(pl, cc, table, plans):
    for n in TILTED:
        p = plans(-3.0, names=(n,)); last = plans.arm.ik_calls[-1]; j = plans.arm.api(f"/api/ik?x_in={last[0]:.4f}&y_in={last[1]:.4f}&h_in={last[2]:.4f}")["joints_deg"]
        assert (last[0], last[1]) != pytest.approx(PROBE_NUT_IN, abs=0.01), "the IK target is shifted back, not the nut"
        assert p[n]["desired"]["wrist_flex"] - j["wrist_flex"] == pytest.approx(-3.0, abs=1e-9) and all(p[n]["desired"][k] == j[k] for k in ARM if k != "wrist_flex")
        assert p[n]["joints"] == p[n]["desired"], "no bias: the command is the desired (tilt included)"


def test_untilted_waypoints_are_unchanged_by_the_tilt(pl, plans):
    p0, p1 = plans(0.0), plans(-3.0)
    for n in UNTILTED: assert p1[n]["joints"] == p0[n]["joints"] and p1[n]["desired"] == p0[n]["desired"], n
    assert set(p0) == set(p1)


def test_tilt_zero_is_exactly_the_plan_before_the_tilt(pl, plans):
    p = plans(0.0); arm = plans.arm; hs = {w[0]: w[2] for w in pl.waypoints()}
    for n, w in p.items():
        if n in pl.BACKOFF: continue   # why: the back-off waypoints are another waypoint's command minus degrees (test_pan_approach below), not a mat point
        if n == "rest": assert w["joints"] == pl.REST; continue
        x, y = (PROBE_NUT_IN if next(v[1] for v in pl.WP if v[0] == n) == "nut" else (9.0, 6.0))
        assert w["joints"] == w["desired"] == arm.api(f"/api/ik?x_in={x:.4f}&y_in={y:.4f}&h_in={hs[n]:.4f}")["joints_deg"], n
    assert plans(None)["grasp"]["joints"] == p["grasp"]["joints"], "tilt None counts as 0"


def test_the_bias_is_added_to_the_tilted_command_and_desired_carries_the_tilt_only(pl, plans):
    b = {"grasp": {"shoulder_lift": 1.5, "wrist_flex": 0.7}}; p = plans(-3.0, biases=b)["grasp"]; q = plans(-3.0)["grasp"]
    assert p["desired"] == q["desired"] and p["joints"]["shoulder_lift"] == pytest.approx(q["desired"]["shoulder_lift"] + 1.5) and p["joints"]["wrist_flex"] == pytest.approx(q["desired"]["wrist_flex"] + 0.7)


def test_the_wrist_flex_clamp_applies_after_the_tilt(pl, plans):
    p = plans(-200.0, names=("grasp",))["grasp"]; assert p["joints"]["wrist_flex"] == -93.0 and p["desired"]["wrist_flex"] < -93.0


def test_fast_close_close_is_tilted_too(pl, cc, table, plans):
    p0, p1 = plans(0.0, fast_close=True), plans(-3.0, fast_close=True); assert "close" in p1 and "close1" not in p1
    x0, y0, h0 = pl.fk_mat(cc, table, p0["close"]["desired"]); x1, y1, h1 = pl.fk_mat(cc, table, p1["close"]["desired"]); assert math.hypot(x1 - x0, y1 - y0) * 25.4 < 0.3 and abs(h1 - h0) < 0.3


def test_the_grasp_and_close_are_commanded_at_1mm_by_the_ik(pl, plans):
    assert pl.GRASP_H_IN * 25.4 == pytest.approx(1.0) and pl.GRASP_TILT_DEG == -3.0
    for tilt in (0.0, -3.0):
        p = plans(tilt, names=("grasp", "close1", "close2"))   # why: with a tilt the IK target moves (side AND height) so the tilted tip lands on the point; what must be 1.0 mm is that tip
        assert all(pl.fk_mat(*pl.calib(), p[n]["desired"])[2] == pytest.approx(1.0, abs=0.3) for n in ("grasp", "close1", "close2"))
    assert plans(0.0)["grasp"]["desired"] and pl.fk_mat is not None


def test_the_tilt_and_the_calibration_default_from_the_cached_loader(pl, cc, table, plans):
    assert pl.calib()[0] is cc and pl.calib() is pl.calib() and pl.calib()[1]["arm_fit"] == table["arm_fit"]
    a = pl.build_plan(list(PROBE_NUT_IN), [9.0, 6.0], {}, tilt_deg=-3.0)["waypoints"]; b = pl.build_plan(list(PROBE_NUT_IN), [9.0, 6.0], {}, tilt_deg=-3.0, cc=cc, table=table)["waypoints"]
    assert a == b


# ---------------------------------------------------------------- the CLI flag and the runner reach build_plan
def spy(pl, monkeypatch):
    seen = []; orig = pl.build_plan
    def build_plan(*a, **kw): seen.append(kw.get("tilt_deg")); return orig(*a, **kw)
    monkeypatch.setattr(pl, "build_plan", build_plan); return seen


@pytest.mark.parametrize("argv,want", [([], -3.0), (["--grasp-tilt-deg", "-1.5"], -1.5), (["--grasp-tilt-deg", "0"], 0.0)])
def test_the_flag_reaches_build_plan_in_the_closed_loop_and_the_mlflow_param(pl, cc, table, monkeypatch, capsys, tmp_path, argv, want):
    seen = spy(pl, monkeypatch); arm = ShortArm(cc, table); drive(pl, monkeypatch, capsys, arm, ["--runs", "1", *argv, "--state", str(tmp_path / "s.json")])
    assert seen and set(seen) == {want} and pl.rec.runs[-1].params_["grasp_tilt_deg"] == want
    assert pl.parser().get_default("grasp_tilt_deg") == pl.GRASP_TILT_DEG


def test_the_flag_reaches_build_plan_in_the_open_loop_and_reset_too(pl, cc, table, monkeypatch, capsys, tmp_path):
    seen = spy(pl, monkeypatch); drive(pl, monkeypatch, capsys, ShortArm(cc, table), ["--runs", "1", "--open-loop", "--grasp-tilt-deg", "-2", "--state", str(tmp_path / "s.json")]); assert seen == [-2.0]
    seen.clear(); drive(pl, monkeypatch, capsys, ShortArm(cc, table), ["--runs", "1", "--open-loop", "--state", str(tmp_path / "s2.json")]); assert seen == [pl.GRASP_TILT_DEG]
    seen.clear(); drive(pl, monkeypatch, capsys, ShortArm(cc, table), ["--reset", "--grasp-tilt-deg", "-2", "--state", str(tmp_path / "s3.json")]); assert seen and set(seen) == {-2.0}


def test_the_autoloop_runner_passes_the_default_tilt_to_build_plan(autorig, monkeypatch, capsys):
    r = autorig(); seen = spy(r.pl, monkeypatch); s, out = r.run(capsys, cycles=1, max_retries=0)
    assert r.runner.pick_args.grasp_tilt_deg == r.pl.GRASP_TILT_DEG and seen and r.pl.GRASP_TILT_DEG in seen and set(seen) <= {r.pl.GRASP_TILT_DEG, None, 0.0}


# ---------------------------------------------------------------- pregrasp_up
def test_pregrasp_up_comes_right_before_pregrasp_with_the_approach_joints_8_deg_lower(pl):
    assert pl.APPROACH_DEG == 8.0 and pl.STAGES[2][:2] == ("pregrasp_up", "pregrasp") and pl.LIFT_STAGE[:2] == ("pregrasp_up", "pregrasp")
    assert [w[0] for w in pl.WP][:4] == ["hover_pre", "hover_nut", "pregrasp_up", "pregrasp"]   # why: hover_pre (the pan back-off, 12 Nov 2026) now leads the plan


@pytest.mark.parametrize("nut", SPOTS)
def test_pregrasp_up_is_defined_in_joint_space_from_pregrasps_command(pl, plans, nut):
    b = {"pregrasp": {"shoulder_lift": 1.0, "elbow_flex": -0.5, "shoulder_pan": 0.3}}; p = plans(-3.0, nut, biases=b, names=pl.LIFT_STAGE); names = list(p)
    assert names[:3] == ["pregrasp_up", "pregrasp", "grasp"]
    for fld in ("joints", "desired"):
        up, pre = p["pregrasp_up"][fld], p["pregrasp"][fld]
        assert up["shoulder_lift"] == pytest.approx(pre["shoulder_lift"] - 8.0, abs=1e-9) and up["elbow_flex"] == pytest.approx(pre["elbow_flex"] - 8.0, abs=1e-9)
        assert up["shoulder_pan"] == pytest.approx(pre["shoulder_pan"] - pl.PAN_APPROACH_DEG, abs=1e-9)   # why: pregrasp_up also takes the pan back-off now (12 Nov 2026)
        assert all(up[k] == pre[k] for k in ARM if k not in ("shoulder_lift", "elbow_flex", "shoulder_pan"))
    assert p["pregrasp_up"]["gripper"] == p["pregrasp"]["gripper"] == 35 and p["pregrasp_up"]["gripper_only"] is False


@pytest.mark.parametrize("nut", SPOTS)
def test_pregrasp_up_and_the_path_down_to_pregrasp_stay_at_or_above_pregrasp(pl, cc, table, plans, nut):
    p = plans(-3.0, nut, names=pl.LIFT_STAGE); up, pre = p["pregrasp_up"]["joints"], p["pregrasp"]["joints"]
    hpre = pl.fk_mat(cc, table, pre)[2]   # (the tilt moves the tip height by a few mm either way, so pregrasp is not exactly 10.2)
    hs = [pl.fk_mat(cc, table, {k: up[k] + t * (pre[k] - up[k]) for k in ARM})[2] for t in np.linspace(0.0, 1.0, 41)]
    assert min(hs) >= hpre - 0.5 and hs[0] > hpre + 30.0, f"pregrasp_up sits {hs[0]:.0f} mm up, the path never dips below {min(hs):.1f} (pregrasp {hpre:.1f})"
    xu, yu, _ = pl.fk_mat(cc, table, up); xp, yp, _ = pl.fk_mat(cc, table, pre); assert 20.0 < math.hypot(xu - xp, yu - yp) * 25.4 < 60.0, "and it sits back from the nut"


def test_pregrasp_up_does_not_change_the_biases_and_is_not_gripper_only(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table); st = tmp_path / "s.json"; drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(st)])
    saved = json.load(open(st)); assert "pregrasp_up" not in saved["biases"] and "pregrasp" in saved["biases"] and "pregrasp_up" in [w["name"] for w in arm.plans[2]["waypoints"]]
    assert "pregrasp_up" in pl.NO_LEARN and "pregrasp_up" not in pl.GRIPPER_ONLY
    # a bias under that name (an old state file cannot have one, but still) is not applied or written back
    arm2 = ShortArm(cc, table); s2 = tmp_path / "s2.json"; json.dump({"biases": {"pregrasp_up": {"shoulder_lift": 4.0}}, "place_offsets_in": {}, "history": []}, open(s2, "w"))
    drive(pl, monkeypatch, capsys, arm2, ["--runs", "1", "--state", str(s2)]); assert json.load(open(s2))["biases"]["pregrasp_up"] == {"shoulder_lift": 4.0}


# ---------------------------------------------------------------- the pan back-off (12 Nov 2026)
STAGE_NAMES = lambda pl: (("stage 1", pl.STAGES[0]), ("stage 2", pl.STAGES[1]), ("stage 3", pl.STAGES[2]), ("lift stage", pl.LIFT_STAGE), ("abort stage", pl.ABORT_STAGE))
BACKED = {"hover_pre": "hover_nut", "pregrasp_up": "pregrasp"}


def test_every_nut_side_arrival_is_preceded_by_its_back_off_in_every_stage(pl):
    assert pl.PAN_APPROACH_DEG == 3.0 and set(pl.BACKOFF) == set(BACKED) and set(pl.NO_LEARN) == set(BACKED)
    for label, st in STAGE_NAMES(pl):
        for back, tgt in BACKED.items():
            if tgt in st and (tgt != "hover_nut" or label != "stage 3"): assert st[st.index(tgt) - 1] == back, f"{label}: {back} comes right before {tgt}"
    assert pl.STAGES[0] == ("hover_pre", "hover_nut") and pl.STAGES[1] == ("hover_pre", "hover_nut", "grasp") and pl.ABORT_STAGE == ("hover_pre", "hover_nut", "rest")


@pytest.mark.parametrize("nut", SPOTS)
def test_the_back_off_is_exactly_minus_pan_approach_deg_on_pan_and_the_targets_other_joints(pl, plans, nut):
    b = {"hover_nut": {"shoulder_pan": 0.4, "elbow_flex": -0.3}, "pregrasp": {"shoulder_pan": 0.3}}; p = plans(-3.0, nut, biases=b)
    for back, tgt in BACKED.items():
        for fld in ("joints", "desired"):
            a, t = p[back][fld], p[tgt][fld]; d = pl.PAN_APPROACH_DEG; assert a["shoulder_pan"] == pytest.approx(t["shoulder_pan"] - d, abs=1e-9)
            drop = {"shoulder_lift": 8.0, "elbow_flex": 8.0} if back == "pregrasp_up" else {}
            assert all(a[k] == pytest.approx(t[k] - drop.get(k, 0.0), abs=1e-9) for k in ARM if k != "shoulder_pan"), f"{back}.{fld} keeps {tgt}'s other joints"
        assert p[back]["gripper"] == p[tgt]["gripper"] and p[back]["gripper_only"] is False


def test_hover_pre_is_at_the_hover_height_with_open_jaws_and_the_final_moves_increase_pan(pl, cc, table, plans):
    p = plans(-3.0); hs = {w[0]: w[2] for w in pl.WP}; assert hs["hover_pre"] == hs["hover_nut"] == 1.5 and p["hover_pre"]["gripper"] == p["hover_nut"]["gripper"] == 35
    assert p["hover_nut"]["joints"]["shoulder_pan"] - p["hover_pre"]["joints"]["shoulder_pan"] == pytest.approx(3.0)
    assert p["pregrasp"]["joints"]["shoulder_pan"] - p["pregrasp_up"]["joints"]["shoulder_pan"] == pytest.approx(3.0)
    assert p["grasp"]["joints"]["shoulder_pan"] >= p["pregrasp"]["joints"]["shoulder_pan"] - 0.05, "pregrasp -> grasp is not a pan reversal (same nut point)"
    for back in BACKED: assert [w for w in pl.WP if w[0] == back][0][4:] in ((1.6, 0.4), (0.8, 0.3)), "seconds/settle: the long rest -> hover move rides on hover_pre, pregrasp_up is as before"


def test_back_offs_never_change_biases_and_survive_the_jaw_retry_and_abort_paths(pl, cc, table, monkeypatch, capsys, tmp_path):
    from pickhelpers import jaw_frame, frame_feed
    arm = ShortArm(cc, table); st = tmp_path / "s.json"; json.dump({"biases": {"hover_pre": {"shoulder_pan": 4.0}}, "place_offsets_in": {}, "history": []}, open(st, "w"))
    monkeypatch.setattr(pl, "fetch_wrist", frame_feed(jaw_frame(ring=False))); drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(st)])
    saved = json.load(open(st))["biases"]; assert saved["hover_pre"] == {"shoulder_pan": 4.0} and "pregrasp_up" not in saved and "hover_nut" in saved
    names = [n for n, _ in arm.calls]; assert names[0] == ["hover_pre", "hover_nut"] and names.count(["hover_pre", "hover_nut", "grasp"]) == 2, "the retry re-approaches from the same side"
    assert names[-1] == ["hover_pre", "hover_nut", "rest"] and [w["name"] for w in arm.plans[-1]["waypoints"]] == ["hover_pre", "hover_nut", "rest"]
