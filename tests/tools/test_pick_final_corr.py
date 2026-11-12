"""tools/pick_loop.py's final-pose correction (the final stage split at the grasp when the pick_seq session is on: read the tip back, ONE corrective move at grasp
height when the height or xy is off) and the object-agnostic grip check (gripper readback after the lift OR the wrist-camera ring). Fake arm, no ssh, no arm motion."""
import json, math
import numpy as np, pytest
from pickhelpers import ARM, PROBE_NUT_IN, ShortArm, drive, frame_feed, jaw_frame, solve_joints, tip_of

HOVER, POSE = ["hover_pre", "hover_nut"], ["hover_pre", "hover_nut", "grasp"]
PRE, CORR, CLOSE, CARRY, ABORT = ["pregrasp_up", "pregrasp", "grasp"], ["grasp"], ["close1", "close2", "lift"], ["hover_tgt", "place", "open", "lift2", "rest"], ["hover_pre", "hover_nut", "rest"]


def stages(out, name): return [o for o in out if o.get("stage") == name]
def records(out): return [o for o in out if "picked" in o]


class FinalArm(ShortArm):
    """ShortArm whose FINAL approach (a plan with pregrasp_up) reads the grasp tip `dh` mm higher and `dxy` mm (mat-frame arm x, y) off, as the live 11 Nov arm did
    (stage 2 and the final stage differed by up to 4 mm); the corrective move (a plan of the grasp alone) keeps it: the pose error is the final approach's, a few mm of travel do not change it.
    `grip` maps the plan number to the gripper reading of its close2 / lift rows (a callable of (plan name list, wp) -> value or None to keep the commanded one)."""
    def __init__(self, *a, dh=0.0, dxy=(0.0, 0.0), grip=None, **k):
        super().__init__(*a, **k); self.dh, self.dxy, self.grip, self.ik_at = dh, np.array(dxy, float), grip, []

    def run_plan(self, plan, max_deg_s):
        self.ik_at.append(len(self.ik_calls)); names = [w["name"] for w in plan["waypoints"]]; rows = super().run_plan(plan, max_deg_s)
        for r in rows:
            if r["wp"] == "grasp" and ("pregrasp_up" in names or names in (["grasp"], ["pregrasp", "grasp"])) and (self.dh or self.dxy.any()):   # the lifted correction comes straight down from pregrasp like the final approach: same pose error
                want = tip_of(self.cc, self.table, r["reached"]) + np.array([*self.dxy, self.dh]); r["reached"].update({k: round(v, 3) for k, v in solve_joints(self.cc, self.table, r["reached"], want).items()})
            if self.grip and r["wp"] in ("close2", "lift"):
                g = self.grip(names, r["wp"]); r["reached"]["gripper"] = r["reached"]["gripper"] if g is None else g
        return rows

    def plan_iks(self, i): return self.ik_calls[(self.ik_at[i - 1] if i else 0):self.ik_at[i]]   # the /api/ik heights asked for while building plan i


def run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, extra=(), session=True, wrist=None):
    if wrist is not None: monkeypatch.setattr(pl, "fetch_wrist", wrist)
    return drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--grasp-tilt-deg", "0", "--grasp-aim-mm", "0", "--state", str(tmp_path / f"s{id(arm)}.json"), *extra], session=session)


# ---------------------------------------------------------------- the decision
def test_final_fix_fires_outside_the_band_or_over_the_xy_tolerance_and_caps(pl):
    f = pl.final_fix; mm = 1 / 25.4; T, (LO, HI) = pl.FINAL_H_TARGET_MM, pl.FINAL_H_BAND_MM
    assert (T, LO, HI) == (-1.0, -4.0, 1.0), "12 Nov 2026: the grasp aims 1 mm into the mat (the 11 Nov successes read -3.6 to +1.3 mm, median -1.4)"
    assert f(T, (0.0, 0.0), 0.0) is None and f(LO, (0.0, 0.0), 0.0) is None and f(HI, (0.0, 0.0), 0.0) is None, "inside the band, on the aim: nothing"
    assert f(T, (1.4 * mm, 0.0), 0.0) is None, "1.4 mm xy is inside the 1.5 tolerance"
    (cx, cy), lw = f(T, (1.6 * mm, 0.0), 0.0); assert cx == pytest.approx(1.6 * mm) and lw == 0.0, "xy alone: the height stays"
    assert f(HI + 0.1, (0, 0), 0.0)[1] == pytest.approx(HI + 0.1 - T) and f(LO - 0.1, (0, 0), 0.0)[1] == pytest.approx(LO - 0.1 - T), "outside the band: to the target"
    assert f(3.4, (0, 0), 0.0)[1] == pytest.approx(4.4) and f(3.4, (0, 0), 2.0)[1] == pytest.approx(6.0), "lowering is capped at 6 mm in all"
    assert f(-9.0, (0, 0), 0.0)[1] == -5.0 and f(30.0, (0, 0), 0.0)[1] == 6.0, "raise cap 5, lower cap 6"
    assert f(30.0, (0, 0), 6.0) is None and f(-9.0, (0, 0), -5.0) is None, "already at the cap and xy fine: no move"
    (cx, cy), _ = f(T, (10 * mm, 0.0), 0.0); assert math.hypot(cx, cy) / mm == pytest.approx(4.0), "xy capped at 4 mm"
    (cx, cy), _ = f(T, (3 * mm, -4 * mm), 0.0); assert (cx / mm, cy / mm) == (pytest.approx(2.4), pytest.approx(-3.2)), "a 5 mm residual scaled to 4 mm, direction kept"


# ---------------------------------------------------------------- the split stage on the fake arm
def test_no_correction_when_the_final_pose_is_on_target(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = FinalArm(cc, table, shortfall_mm=7.0); out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, wrist=frame_feed(jaw_frame(ring=True))); rec, = records(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE, CLOSE, CARRY], "the final stage is split at the grasp; no corrective stage"
    assert not stages(out, "final_corr") and rec["final_corr_mm"] == 0.0 and rec["final_h_before"] == rec["final_h_after"] and rec["picked"] is True
    r, = pl.rec.runs; assert r.metrics_["final_corr_mm"] == 0.0 and "final_h_before" in r.metrics_ and r.metrics_["final_h_after"] == r.metrics_["final_h_before"]


@pytest.mark.parametrize("dh,dxy", [(3.5, (0.0, 0.0)), (-5.5, (0.0, 0.0)), (0.0, (3.0, 0.0)), (2.5, (0.0, -2.5))])   # the fake reads the grasp at +1 mm (its command) + dh: 4.5 / -4.5 (outside -4..1) / xy only (height in band, kept) / 3.5
def test_one_corrective_move_at_grasp_height_lands_the_tip_on_target(pl, cc, table, monkeypatch, capsys, tmp_path, dh, dxy):
    arm = FinalArm(cc, table, shortfall_mm=7.0, dh=dh, dxy=dxy); out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, wrist=frame_feed(jaw_frame(ring=True))); rec, = records(out); c, = stages(out, "final_corr")
    lifted = math.hypot(*c["corr_mm"]) > pl.FINAL_SLIDE_MAX_MM; assert c["lifted"] is lifted
    corr = [["pregrasp"], ["pregrasp", "grasp"]] if lifted else [CORR]
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE, *corr, CLOSE, CARRY], "one corrective move before the close: the grasp waypoint alone, or up-over-down when it is bigger than FINAL_SLIDE_MAX_MM"
    in_band = pl.FINAL_H_BAND_MM[0] <= c["h_before"] <= pl.FINAL_H_BAND_MM[1]
    assert c["h_after"] == pytest.approx(c["h_before"] if in_band else pl.FINAL_H_TARGET_MM, abs=0.35) and math.hypot(*c["residual_mm"]) < 1.0, "the readback after the move is on the aim, and on the height target when it started outside the band"
    assert not in_band or math.hypot(*stages(out, "final")[0]["residual_mm"]) > 1.5, "it fired for a reason"
    assert rec["final_h_before"] == c["h_before"] and rec["final_h_after"] == c["h_after"] and rec["final_corr_mm"] == pytest.approx(math.hypot(*c["corr_mm"]), abs=0.1) and rec["final_residual_mm"] == pytest.approx(math.hypot(*c["residual_mm"]), abs=0.1)
    r, = pl.rec.runs; assert r.metrics_["final_corr_mm"] == rec["final_corr_mm"] and r.metrics_["final_h_before"] == c["h_before"] and r.metrics_["final_h_after"] == c["h_after"]
    gi, ci = (4, 5) if lifted else (3, 4)   # the plan that ends at the corrected grasp; the close plan
    assert arm.plans[gi]["waypoints"][-1]["name"] == "grasp" and arm.plans[ci]["waypoints"][0]["joints"] == arm.plans[ci]["waypoints"][1]["joints"], "the close steps hold the corrected grasp pose"
    assert -0.26 <= arm.plan_iks(gi)[-1][2] <= 0.26, "the grasp itself is asked at grasp height (within the 5 mm raise / 6 mm lower caps), never up in the air or deep under the mat"
    assert "grasp" in json.load(open(tmp_path / f"s{id(arm)}.json"))["biases"]


def test_a_big_xy_correction_lifts_the_open_jaws_and_a_small_one_slides(pl, cc, table, monkeypatch, capsys, tmp_path):
    """12 Nov 2026: lying spacers rolled away on every 2.5-4.2 mm slide of the open jaws at grasp height and were held after the ~1 mm ones."""
    arm = FinalArm(cc, table, shortfall_mm=7.0, dxy=(3.0, 0.0)); out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, wrist=frame_feed(jaw_frame(ring=True))); c, = stages(out, "final_corr")
    assert c["lifted"] is True and [n for n, _ in arm.calls][3:5] == [["pregrasp"], ["pregrasp", "grasp"]]
    up, over_down = arm.plan_iks(3), arm.plan_iks(4)
    assert up[0][2] == pytest.approx(pl.PREGRASP_H_IN, abs=0.02) and over_down[0][2] == pytest.approx(pl.PREGRASP_H_IN, abs=0.02), "up off the mat, and across at pregrasp height"
    assert all(w["gripper"] == pl.JAW_OPEN for p in (arm.plans[3], arm.plans[4]) for w in p["waypoints"]), "the jaws stay open until the close"
    assert math.hypot(up[0][0] - over_down[0][0], up[0][1] - over_down[0][1]) * 25.4 == pytest.approx(math.hypot(*c["corr_mm"]), abs=0.6), "the lift is at the old spot, the descent at the corrected one"
    arm2 = FinalArm(cc, table, shortfall_mm=7.0, dh=3.0, dxy=(1.0, 0.0)); out2 = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm2, wrist=frame_feed(jaw_frame(ring=True))); c2, = stages(out2, "final_corr")
    assert math.hypot(*c2["corr_mm"]) <= pl.FINAL_SLIDE_MAX_MM and c2["lifted"] is False and [n for n, _ in arm2.calls][3] == CORR, "a small correction still slides at grasp height"


def test_corrections_are_capped_and_never_repeated(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = FinalArm(cc, table, shortfall_mm=7.0, dh=14.0, dxy=(9.0, 0.0)); out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, wrist=frame_feed(jaw_frame(ring=True))); c, = stages(out, "final_corr")
    assert math.hypot(*c["corr_mm"]) == pytest.approx(4.0, abs=0.06), "xy move capped at 4 mm"
    assert c["lowered_mm"] == pytest.approx(6.0), "height lowered by the cap, not the 13 mm the reading asks"
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE, ["pregrasp"], ["pregrasp", "grasp"], CLOSE, CARRY] and c["lifted"], "the corrected reading is still off (the fake keeps it off) and there is still only ONE corrective move (lifted: 4 mm > FINAL_SLIDE_MAX_MM)"
    assert c["h_after"] > 2.0, "left as it landed: no second go"
    arm2 = FinalArm(cc, table, shortfall_mm=7.0, dh=-9.0); out2 = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm2, wrist=frame_feed(jaw_frame(ring=True))); c2, = stages(out2, "final_corr")
    assert c2["lowered_mm"] >= -5.0 - 1e-6 and all(-0.26 <= h <= 0.26 for _, _, h in arm2.plan_iks(3))


def test_session_off_keeps_the_single_lift_stage(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = FinalArm(cc, table, shortfall_mm=7.0, dh=4.0); out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, session=False, wrist=frame_feed(jaw_frame(ring=True))); rec, = records(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE + CLOSE, CARRY] and not stages(out, "final_corr") and rec["final_corr_mm"] is None and rec["final_h_before"] is None, "no split, no correction: it would cost a connect per stage"


def test_no_jaw_check_with_the_session_splits_at_the_grasp_and_carries_on(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = FinalArm(cc, table, shortfall_mm=7.0, dh=4.0); out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, extra=["--no-jaw-check"]); rec, = records(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE, CORR, [*CLOSE, *CARRY]] and not stages(out, "jaw_check") and stages(out, "final_corr") and rec["picked"] is True and rec["gripper"] is None
    arm2 = FinalArm(cc, table, shortfall_mm=7.0); out2 = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm2, extra=["--no-jaw-check"])
    assert [n for n, _ in arm2.calls] == [HOVER, POSE, PRE, [*CLOSE, *CARRY]] and records(out2)[0]["picked"] is True


# ---------------------------------------------------------------- the grip check
def rows_with(lift=None, close2=None):
    r = []
    if close2 is not None: r.append({"wp": "close2", "reached": {"gripper": close2}})
    if lift is not None: r.append({"wp": "lift", "reached": {"gripper": lift}})
    return r


@pytest.mark.parametrize("grip,ring,want", [
    (4.5, False, True), (4.5, True, True), (4.0, None, True),       # gripper high -> held, whatever the camera says
    (0.8, True, True), (1.4, True, True),                           # low gripper + camera held -> held (the wing nut by its wing: 1.3-1.5)
    (0.8, False, False), (1.5, False, False), (2.0, False, False),  # low gripper + camera empty -> not held (2.0 is not over the limit)
])
def test_grip_check_truth_table(pl, grip, ring, want):
    jc = pl.jaw_check(jaw_frame(ring=bool(ring)) if ring is not None else b"", rows=rows_with(lift=grip))
    assert jc["in_jaws"] is want and jc["gripper"] == grip and jc["camera"] is (ring if ring is not None else None)


def test_grip_check_unknown_camera_and_low_gripper_carries_unknown_and_close2_does_not_count(pl):
    jc = pl.jaw_check(b"", rows=rows_with(lift=0.6)); assert jc["in_jaws"] is None and jc["camera"] is None and jc["gripper"] == 0.6, "nothing says empty: unknown, carried as before"
    jc = pl.jaw_check(jaw_frame(ring=False), rows=rows_with(close2=4.49, lift=0.54)); assert jc["in_jaws"] is False, "close2 read 4.49 but the nut slipped out on the lift: only the lift reading counts"
    jc = pl.jaw_check(jaw_frame(ring=True), rows=rows_with(close2=4.49, lift=0.54)); assert jc["in_jaws"] is True, "... unless the camera says held"
    jc = pl.jaw_check(jaw_frame(ring=False)); assert jc["in_jaws"] is False and jc["gripper"] is None, "no rows: the camera alone, as before"
    jc = pl.jaw_check(jaw_frame(ring=False), rows=[{"wp": "rest"}, {"wp": "lift"}]); assert jc["in_jaws"] is False and jc["gripper"] is None, "a malformed row is no reading and never raises"
    assert pl.gripper_after_lift(None) is None and pl.GRIP_EMPTY_MAX == 2.0
    assert pl.jaw_check(jaw_frame(ring=False), rows=rows_with(lift=2.5), grip_max=3.0)["in_jaws"] is False, "the threshold is a parameter"


def test_stage_line_and_record_carry_both_readings_a_grip_pick_survives_a_blind_camera(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = FinalArm(cc, table, grip=lambda names, wp: 4.4 if wp in ("close2", "lift") else None)   # a wing nut / spacer the ring finder cannot see, held by the gripper reading
    out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, wrist=frame_feed(jaw_frame(ring=False))); jc, = stages(out, "jaw_check"); rec, = records(out)
    assert jc["in_jaws"] is True and jc["gripper"] == 4.4 and jc["camera"] is False and rec["picked"] is True and rec["jaw_retry"] is False and rec["gripper"] == 4.4 and rec["camera"] is False
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE, CLOSE, CARRY]
    r, = pl.rec.runs; assert r.metrics_["gripper_after_lift"] == 4.4 and r.metrics_["in_jaws"] == 1.0


def test_retry_with_the_new_check_corrects_each_attempt_once_and_carries(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = FinalArm(cc, table, dh=3.5, grip=lambda names, wp: 0.6 if wp == "lift" else None)   # empty readings, first frame empty, second shows the ring
    out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, wrist=frame_feed(jaw_frame(ring=False), jaw_frame(ring=True))); rec, = records(out); j1, j2 = stages(out, "jaw_check")
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE, CORR, CLOSE, POSE, PRE, CORR, CLOSE, CARRY], "the retry re-reads the hover, approaches again and gets its own single correction"
    assert (j1["in_jaws"], j1["gripper"], j1["camera"]) == (False, 0.6, False) and (j2["in_jaws"], j2["camera"]) == (True, True) and len(stages(out, "final_corr")) == 2
    assert rec["picked"] is True and rec["jaw_retry"] is True and rec["reason"] is None


def test_abort_with_the_new_check_when_both_say_empty_twice(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = FinalArm(cc, table, grip=lambda names, wp: 0.7 if wp == "lift" else (4.49 if wp == "close2" else None))   # close2 high, lift low: it slipped out each time
    out = run(pl, cc, table, monkeypatch, capsys, tmp_path, arm, extra=["--runs", "3"], wrist=frame_feed(jaw_frame(ring=False))); rec, = records(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, PRE, CLOSE, POSE, PRE, CLOSE, ABORT], "no carry after two empty checks: open over the spot, rest"
    assert [j["in_jaws"] for j in stages(out, "jaw_check")] == [False, False] and rec["picked"] is False and rec["reason"] == "not_in_jaws" and rec["gripper"] == 0.7 and rec["camera"] is False
    assert any(o.get("reason") == "not_in_jaws" for o in out if "stop" in o)
