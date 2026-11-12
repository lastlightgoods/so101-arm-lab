"""--grasp-roll (12 Nov 2026): turn the wrist so the jaws close ACROSS an elongated object (a spacer lying on its side was pushed, not picked, on 11 Nov).
Checked against the real kinematics of the 9 Nov arm fit through ShortArm's /api/ik (which takes &roll= like the dashboard): the roll that makes jaw_dir
perpendicular to a known axis at several mat spots, the sign of wrist_roll against the jaw's turn, round objects, off = today's plan, the tilted rolled tip,
the aim point on the rolled jaw_dir, the CLI and the autoloop pass-through. The angle convention itself is tests/scene/test_long_axis.py. No dashboard, no ssh, no arm."""
import math
import numpy as np, pytest
from pickhelpers import ARM, PROBE_NUT_IN, ShortArm, drive

SPOTS = [(3.9, 6.7), (11.85, 7.73), (14.4, 7.1), (8.0, 5.0), (10.0, 9.0), PROBE_NUT_IN]
ROLLED = ("hover_pre", "hover_nut", "pregrasp_up", "pregrasp", "grasp", "close1", "close2", "lift"); CARRY = ("hover_tgt", "place", "open", "lift2", "rest")
def stages(out): return {o["stage"]: o for o in out if "stage" in o}
def ang(u): return math.degrees(math.atan2(u[1], u[0]))
def unit(deg): return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


@pytest.fixture
def arm(pl, cc, table, monkeypatch):
    a = ShortArm(cc, table); monkeypatch.setattr(pl, "api", a.api); return a


def ik_joints(arm, spot, h_in=1.0 / 25.4): return arm.api(f"/api/ik?x_in={spot[0]:.4f}&y_in={spot[1]:.4f}&h_in={h_in:.4f}")["joints_deg"]


@pytest.mark.parametrize("spot", SPOTS)
@pytest.mark.parametrize("axis_deg", [0, 30, 45, 90, 120, 135, 170])
def test_the_roll_makes_jaw_dir_perpendicular_to_the_axis(pl, cc, table, arm, spot, axis_deg):
    j = ik_joints(arm, spot); r = pl.roll_to_perpendicular(cc, table, j, unit(axis_deg))
    assert r is not None and abs(r) <= 90.0
    u = pl.jaw_dir_mat(cc, table, dict(j, wrist_roll=j["wrist_roll"] + r)); assert abs(np.dot(u, unit(axis_deg))) < 1e-6, f"{spot} axis {axis_deg}: roll {r:.2f} left jaw_dir {u}"
    # the smaller of the two solutions: the other one is 180 deg away, so this one is within 90
    for other in (r - 180.0, r + 180.0):
        if abs(other) < abs(r): pytest.fail(f"{other:.1f} is the smaller solution than {r:.1f}")


@pytest.mark.parametrize("spot", SPOTS)
def test_positive_wrist_roll_turns_jaw_dir_counter_clockwise_on_the_mat_about_one_to_one(pl, cc, table, arm, spot):
    """Measured with the arm fit (11 Nov 2026): +30 deg of wrist_roll moves jaw_dir about +30 deg (CCW seen from above, mat +X toward +Y); not assumed."""
    j = ik_joints(arm, spot); a0 = ang(pl.jaw_dir_mat(cc, table, j))
    for d in (-30.0, 30.0):
        a1 = ang(pl.jaw_dir_mat(cc, table, dict(j, wrist_roll=j["wrist_roll"] + d))); turn = (a1 - a0 + 180.0) % 360.0 - 180.0
        assert turn * d > 0 and abs(turn / d - 1.0) < 0.1, f"{spot}: roll {d:+.0f} turned jaw_dir {turn:+.1f}"


def test_the_roll_follows_the_axis_through_the_whole_circle(pl, cc, table, arm):
    j = ik_joints(arm, (11.85, 7.73)); rolls = [pl.roll_to_perpendicular(cc, table, j, unit(a)) for a in range(0, 180, 10)]
    assert all(r is not None and abs(r) <= 90.0 for r in rolls) and max(rolls) - min(rolls) > 150.0   # sweeps (nearly) the whole -90..90 range once, wrapping at one point
    assert all(abs(((rolls[i + 1] - rolls[i]) + 90.0) % 180.0 - 90.0 - 10.0) < 3.0 for i in range(len(rolls) - 1)), "10 deg of axis is ~10 deg of roll (modulo the wrap)"


# ---------------------------------------------------------------- resolve: auto / numbers / round / off
def obj(x, y, w, h, a, i=1): return {"id": i, "x_in": x, "y_in": y, "w_mm": w, "h_mm": h, "angle_deg": a}


def test_auto_uses_the_scene_object_nearest_the_nut_and_the_roll_makes_the_jaws_cross_its_axis(pl, cc, table, arm):
    for spot in SPOTS:
        for a in (0.0, 35.0, -60.0, 85.0):
            r, info = pl.resolve_grasp_roll("auto", cc, table, spot, -3.0, objects=[obj(spot[0] + 5, spot[1], 40, 10, 0.0, 2), obj(spot[0] + 0.1, spot[1] - 0.1, 14.0, 6.0, a)])
            assert r is not None and abs(r) <= 90.0 and info["obj"]["id"] == 1 and info["axis"] == [round(unit(a)[0], 3), round(unit(a)[1], 3)]
            j = ik_joints(arm, spot); j["wrist_flex"] -= 3.0   # the grasp pose carries the tilt
            u = pl.jaw_dir_mat(cc, table, dict(j, wrist_roll=j["wrist_roll"] + r)); assert abs(np.dot(u, unit(a))) < 0.01


def test_round_or_missing_objects_give_no_roll(pl, cc, table, arm):
    for objs, why in (([obj(*SPOTS[1], 12.0, 11.0, 30.0)], "round"), ([obj(SPOTS[1][0] + 2.0, SPOTS[1][1], 20.0, 5.0, 0.0)], "no scene object"), ([], "no scene object")):
        r, info = pl.resolve_grasp_roll("auto", cc, table, SPOTS[1], -3.0, objects=objs); assert r is None and why in info["why"]


def test_off_and_numbers(pl, cc, table):
    assert pl.resolve_grasp_roll("off", cc, table, SPOTS[0]) == (None, {}) and pl.resolve_grasp_roll(None, cc, table, SPOTS[0]) == (None, {})
    assert pl.resolve_grasp_roll(25.0, cc, table, SPOTS[0])[0] == 25.0 and pl.resolve_grasp_roll("-40", cc, table, SPOTS[0])[0] == -40.0
    assert [pl.parse_grasp_roll(v) for v in ("auto", "AUTO", "off", "", None, "12.5")] == ["auto", "auto", "off", "off", "off", 12.5]
    with pytest.raises(SystemExit): pl.parse_grasp_roll("sideways")


# ---------------------------------------------------------------- build_plan
def plans(pl, cc, table, nut=PROBE_NUT_IN, **kw): return {w["name"]: w for w in pl.build_plan(list(nut), [9.0, 6.0], kw.pop("biases", {}), cc=cc, table=table, **kw)["waypoints"]}


@pytest.mark.parametrize("kw", [{}, {"tilt_deg": -3.0}, {"tilt_deg": -3.0, "fast_close": True}])
def test_roll_off_is_exactly_the_plan_of_before(pl, cc, table, arm, kw):
    base = plans(pl, cc, table, **kw); n = len(arm.ik_calls)
    for off in (None, 0.0):
        assert plans(pl, cc, table, roll_deg=off, **kw) == base
    assert arm.rolls == [float(table["arm_fit"]["wrist_roll_deg"])] * len(arm.rolls), "no roll was sent to the IK: every solve used the fit's roll"
    assert len(arm.ik_calls) > n


def test_off_sends_no_roll_parameter_to_the_ik(pl, cc, table, monkeypatch, arm):
    paths = []; monkeypatch.setattr(pl, "api", lambda p: (paths.append(p), arm.api(p))[1]); plans(pl, cc, table, tilt_deg=-3.0); assert paths and not any("roll" in p for p in paths)
    paths.clear(); plans(pl, cc, table, tilt_deg=-3.0, roll_deg=20.0); assert any("roll=" in p for p in paths)


@pytest.mark.parametrize("roll", [-60.0, -25.0, 40.0, 85.0])
def test_roll_goes_on_the_nut_side_waypoints_only(pl, cc, table, arm, roll):
    base = float(table["arm_fit"]["wrist_roll_deg"]); p0, p1 = plans(pl, cc, table, tilt_deg=-3.0), plans(pl, cc, table, tilt_deg=-3.0, roll_deg=roll)
    for n in ROLLED: assert p1[n]["joints"]["wrist_roll"] == pytest.approx(base + roll, abs=1e-6) and p1[n]["desired"]["wrist_roll"] == pytest.approx(base + roll, abs=1e-6), n
    for n in CARRY: assert p1[n] == p0[n], n
    assert set(p0) == set(p1)


@pytest.mark.parametrize("nut", SPOTS)
@pytest.mark.parametrize("roll", [-70.0, 30.0, 80.0])
def test_the_tilted_rolled_tip_still_lands_on_the_commanded_point(pl, cc, table, arm, nut, roll):
    p = plans(pl, cc, table, nut, tilt_deg=-3.0, roll_deg=roll, names=("hover_nut", "pregrasp", "grasp", "close1", "close2", "lift"))
    for n, h_mm in (("hover_nut", 1.5 * 25.4), ("pregrasp", pl.PREGRASP_H_IN * 25.4), ("grasp", 1.0), ("close1", 1.0), ("close2", 1.0), ("lift", 1.5 * 25.4)):
        x, y, h = pl.fk_mat(cc, table, p[n]["desired"])
        assert math.hypot(x - nut[0], y - nut[1]) * 25.4 < 0.3 and abs(h - h_mm) < 0.3, f"{n} roll {roll}: {math.hypot(x - nut[0], y - nut[1]) * 25.4:.2f} mm xy, {h - h_mm:+.2f} mm high"


def test_the_back_offs_carry_the_roll_and_keep_their_degrees(pl, cc, table, arm):
    p = plans(pl, cc, table, tilt_deg=-3.0, roll_deg=50.0, biases={"hover_nut": {"shoulder_pan": 0.4}, "pregrasp": {"elbow_flex": 0.7}})
    base = float(table["arm_fit"]["wrist_roll_deg"])
    for fld in ("joints", "desired"):
        assert p["hover_pre"][fld]["wrist_roll"] == pytest.approx(base + 50.0) and p["pregrasp_up"][fld]["wrist_roll"] == pytest.approx(base + 50.0)
        assert p["hover_pre"][fld]["shoulder_pan"] == pytest.approx(p["hover_nut"][fld]["shoulder_pan"] - pl.PAN_APPROACH_DEG)
        assert p["pregrasp_up"][fld]["shoulder_pan"] == pytest.approx(p["pregrasp"][fld]["shoulder_pan"] - pl.PAN_APPROACH_DEG)
        assert p["pregrasp_up"][fld]["shoulder_lift"] == pytest.approx(p["pregrasp"][fld]["shoulder_lift"] - pl.APPROACH_DEG)
        assert p["pregrasp_up"][fld]["elbow_flex"] == pytest.approx(p["pregrasp"][fld]["elbow_flex"] - pl.APPROACH_DEG)
    assert p["pregrasp"]["joints"]["elbow_flex"] == pytest.approx(p["pregrasp"]["desired"]["elbow_flex"] + 0.7), "biases still add on top"


# ---------------------------------------------------------------- the closed loop: aim point on the rolled jaw_dir, flag and MLflow
def drive_roll(pl, cc, table, monkeypatch, capsys, tmp_path, argv, scene=None, name="s.json"):
    a = ShortArm(cc, table, scene=scene); out = drive(pl, monkeypatch, capsys, a, ["--runs", "1", *argv, "--state", str(tmp_path / name)]); return a, out


def spy_roll(pl, monkeypatch):
    seen = []; orig = pl.build_plan
    def build_plan(*a, **kw): seen.append(kw.get("roll_deg")); return orig(*a, **kw)
    monkeypatch.setattr(pl, "build_plan", build_plan); return seen


def test_auto_rolls_the_closed_loop_and_the_aim_point_uses_the_rolled_jaw_dir(pl, cc, table, monkeypatch, capsys, tmp_path):
    seen = spy_roll(pl, monkeypatch); a, out = drive_roll(pl, cc, table, monkeypatch, capsys, tmp_path, ["--grasp-roll", "auto"], scene=[obj(PROBE_NUT_IN[0], PROBE_NUT_IN[1], 14.0, 6.0, 25.0)])
    st = stages(out); gr = st["grasp_roll"]; assert gr["roll_deg"] is not None and abs(gr["roll_deg"]) <= 90 and gr["axis"] == [round(unit(25.0)[0], 3), round(unit(25.0)[1], 3)]
    assert seen and set(seen) == {gr["roll_deg"]}, "every stage, the retry included, carries the same roll"
    u = st["hover"]["jaw_dir"]; assert abs(np.dot(u, unit(25.0))) < 0.06, f"hover jaw_dir {u} is not across the 25 deg axis"   # the hover readback is un-tilted, the roll was solved with the tilt: a few hundredths
    aim = st["hover"]["aim_in"]; assert math.hypot(aim[0] - PROBE_NUT_IN[0], aim[1] - PROBE_NUT_IN[1]) * 25.4 == pytest.approx(pl.GRASP_AIM_MM, abs=0.2)
    assert (PROBE_NUT_IN[0] - aim[0]) * u[0] + (PROBE_NUT_IN[1] - aim[1]) * u[1] > 0, "the aim sits behind the nut along the ROLLED jaw direction"
    rec = pl.rec.runs[-1]; assert rec.params_["grasp_roll"] == "auto" and rec.metrics_["grasp_roll_deg"] == pytest.approx(gr["roll_deg"])
    want = float(table["arm_fit"]["wrist_roll_deg"]) + gr["roll_deg"]
    assert all(w["joints"]["wrist_roll"] == pytest.approx(want, abs=1e-6) for p in a.plans for w in p["waypoints"] if w["name"] in ROLLED)
    assert all(w["joints"]["wrist_roll"] == pytest.approx(float(table["arm_fit"]["wrist_roll_deg"]), abs=1e-6) for p in a.plans for w in p["waypoints"] if w["name"] in CARRY and w["name"] != "rest")


def test_default_is_off_and_the_default_run_never_asks_for_a_roll(pl, cc, table, monkeypatch, capsys, tmp_path):
    assert pl.parser().get_default("grasp_roll") == "off"
    seen = spy_roll(pl, monkeypatch); a, out = drive_roll(pl, cc, table, monkeypatch, capsys, tmp_path, [], scene=[obj(PROBE_NUT_IN[0], PROBE_NUT_IN[1], 14.0, 6.0, 25.0)])
    assert seen and set(seen) == {None} and "grasp_roll" not in stages(out) and pl.rec.runs[-1].params_["grasp_roll"] == "off" and "grasp_roll_deg" not in pl.rec.runs[-1].metrics_
    assert a.rolls == [float(table["arm_fit"]["wrist_roll_deg"])] * len(a.rolls)


def test_auto_on_a_round_object_does_nothing(pl, cc, table, monkeypatch, capsys, tmp_path):
    seen = spy_roll(pl, monkeypatch); a, out = drive_roll(pl, cc, table, monkeypatch, capsys, tmp_path, ["--grasp-roll", "auto"], scene=[obj(PROBE_NUT_IN[0], PROBE_NUT_IN[1], 11.0, 10.0, 25.0)])
    assert set(seen) == {None} and stages(out)["grasp_roll"]["roll_deg"] is None and "round" in stages(out)["grasp_roll"]["why"] and a.rolls == [float(table["arm_fit"]["wrist_roll_deg"])] * len(a.rolls)


def test_a_number_and_the_open_loop_and_reset_reach_build_plan(pl, cc, table, monkeypatch, capsys, tmp_path):
    seen = spy_roll(pl, monkeypatch); drive_roll(pl, cc, table, monkeypatch, capsys, tmp_path, ["--grasp-roll", "-35"]); assert set(seen) == {-35.0} and pl.rec.runs[-1].params_["grasp_roll"] == "-35.0"
    seen.clear(); drive_roll(pl, cc, table, monkeypatch, capsys, tmp_path, ["--open-loop", "--grasp-roll", "20"], name="s2.json"); assert seen == [20.0] and pl.rec.runs[-1].metrics_["grasp_roll_deg"] == 20.0
    seen.clear(); drive_roll(pl, cc, table, monkeypatch, capsys, tmp_path, ["--open-loop"], name="s3.json"); assert seen == [None]
    with pytest.raises(SystemExit): pl.parser().parse_args(["--grasp-roll", "left"])


# ---------------------------------------------------------------- the autoloop
def test_the_autoloop_runner_passes_the_roll_and_defaults_to_off(autorig, al, monkeypatch, capsys):
    r = autorig(); seen = spy_roll(r.pl, monkeypatch); r.run(capsys, cycles=1, max_retries=0)
    assert r.runner.pick_args.grasp_roll == "off" and seen and set(seen) == {None}
    r = autorig(); seen = spy_roll(r.pl, monkeypatch); r.run(capsys, cycles=1, max_retries=0, grasp_roll=30.0)
    assert r.runner.pick_args.grasp_roll == 30.0 and seen and set(seen) == {30.0}


def test_the_autoloop_cli_flag_reaches_the_config(al, tmp_path):
    p = al.build_parser(); assert p.parse_args([]).grasp_roll == "off" and al.make_cfg(p.parse_args([]), str(tmp_path)).grasp_roll == "off"
    assert al.make_cfg(p.parse_args(["--grasp-roll", "auto"]), str(tmp_path)).grasp_roll == "auto" and al.make_cfg(p.parse_args(["--grasp-roll", "-20"]), str(tmp_path)).grasp_roll == "-20"
