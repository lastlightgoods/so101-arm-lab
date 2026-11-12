"""tools/pick_loop.py's closed loop against a fake arm that reaches 7 mm short along the base->tip direction (what the real arm did on
10 Nov 2026), and tools/wrist_check.py: the bright-ring finder on a synthetic wrist frame and on the real 10 Nov frame, the
project / back-project round trip, and the promise that the check never raises. No dashboard, no ssh, no arm."""
import json, math, os, subprocess, sys
import cv2, numpy as np, pytest
from pickhelpers import ARM, FIXTURE_TABLE, PROBE_FRAME, PROBE_JOINTS, PROBE_NUT_IN, SEED, TOOLS, WRIST_POSE, ShortArm, drive, mat_to_base, solve_joints, tip_of

LIFT = ["pregrasp_up", "pregrasp", "grasp", "close1", "close2", "lift"]; CARRY = ["hover_tgt", "place", "open", "lift2", "rest"]; FINAL = LIFT + CARRY
ALL = ["hover_nut", *FINAL]; GRIPPER_ONLY = {"close1", "close2", "open"}


def stages(out): return {o["stage"]: o for o in out if "stage" in o}
def records(out): return [o for o in out if "picked" in o]


def fk_mat(pl, cc, table, joints): return pl.fk_mat(cc, table, joints)


# ---------------------------------------------------------------- the staged corrections
def test_two_readback_corrections_converge_on_an_arm_that_reaches_7mm_short(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table, shortfall_mm=7.0); state = tmp_path / "s.json"
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--target", "9", "6", "--grasp-aim-mm", "0", "--state", str(state)])   # the corrections alone: the aim is test_pick_aim.py's
    for o in out: print(json.dumps(o))   # the transcript, visible with pytest -s
    st = stages(out); rec, = records(out)
    assert [n for n, _ in arm.calls] == [["hover_nut"], ["hover_nut", "grasp"], LIFT, CARRY], "four staged pick_seq runs: the jaw check (no frame here: unknown, carry) splits the final stage after the lift"
    assert all(v == 50.0 for _, v in arm.calls), "closed loop defaults to 50 deg/s"
    c1, c2 = st["hover"]["corr1_mm"], st["grasp_pose"]["corr2_mm"]
    assert 6.0 < math.hypot(*c1) < 8.0, f"the first correction is the 7 mm shortfall: {c1}"
    assert math.hypot(*c2) < 1.0, f"the second pass only trims the residual: {c2}"
    assert math.hypot(*st["final"]["residual_mm"]) < 1.0 and rec["final_residual_mm"] < 1.0
    x, y, h = fk_mat(pl, cc, table, {k: arm.rows[-2][2]["reached"][k] for k in ARM})   # the FINAL grasp (third waypoint of the lift stage, after pregrasp_up and pregrasp), read back
    assert math.hypot(x - PROBE_NUT_IN[0], y - PROBE_NUT_IN[1]) * 25.4 < 1.0 and abs(h - fk_mat(pl, cc, table, next(w for w in arm.plans[2]["waypoints"] if w["name"] == "grasp")["desired"])[2]) < 0.5   # the tilt moves the tip height by a mm or two (see test_pick_tilt.py), the readback matches the plan
    # the correction points along the reach direction (base -> nut), the way the real shortfall did
    base_in = np.array(cc.arm_to_table(table, 38.8353, 0.0)) / 25.4; d = np.array(PROBE_NUT_IN) - base_in; d /= np.linalg.norm(d)
    assert np.dot(np.array(c1) / np.linalg.norm(c1), d) > 0.98
    # the history record carries the stage numbers and the sag of the final stage only
    assert rec["mode"] == "closed" and rec["jaw_open"] == 35 and rec["corr1_mm"] == c1 and rec["corr2_mm"] == c2 and rec["grasp_h_mm"] == st["grasp_pose"]["h_mm"] and rec["grasp_lowered_mm"] == st["grasp_pose"]["lowered_mm"]   # (the tilt puts the stage-2 tip a mm under the IK height, so the raise rule may fire: the record carries whatever the rule decided)
    assert set(rec["sag_deg"]) == set(FINAL) and rec["picked"] is True and rec["in_jaws"] is None and rec["reason"] is None and rec["nut_source"] == "near"
    saved = json.load(open(state)); assert set(saved["biases"]) == set(FINAL) - GRIPPER_ONLY - {"pregrasp_up"}, "sag is learned on the final stages' waypoints, never on the jaw moves"
    assert saved["history"][-1]["corr1_mm"] == c1
    # and MLflow gets them as metrics through the (faked) labtrack
    r, = pl.rec.runs; m = r.metrics_
    assert m["corr1_mm"] == pytest.approx(math.hypot(*c1)) and m["corr1_x_mm"] == c1[0] and m["corr1_y_mm"] == c1[1] and m["corr2_mm"] == pytest.approx(math.hypot(*c2))
    assert m["grasp_h_mm"] == rec["grasp_h_mm"] and m["grasp_lowered_mm"] == rec["grasp_lowered_mm"] and m["final_residual_mm"] == rec["final_residual_mm"] and m["jaw_open"] == 35 and m["picked"] == 1.0
    assert {k for k in m if k.startswith("sag_")} == {f"sag_{n}" for n in FINAL} and "wrist_offset_mm" not in m
    assert r.params_["mode"] == "closed" and r.params_["jaw_open"] == 35 and r.params_["max_deg_s"] == 50.0 and r.params_["wrist_check"] == 0 and r.params_["jaw_check"] == 1
    assert "plan.json" in r.files and "waypoints.json" in r.files and "frames/landing.jpg" in r.files
    capsys.readouterr(); arm2 = ShortArm(cc, table, shortfall_mm=7.0); out2 = drive(pl, monkeypatch, capsys, arm2, ["--runs", "1", "--no-jaw-check", "--state", str(tmp_path / "s2.json")])   # (the transcript echo above is drained first)
    assert [n for n, _ in arm2.calls] == [["hover_nut"], ["hover_nut", "grasp"], FINAL] and records(out2)[0]["picked"] and "jaw_check" not in stages(out2), "--no-jaw-check: the three staged runs of before"


@pytest.mark.parametrize("high_mm", [4.0, 7.0, 14.0])   # why: GRASP_H_IN is 1.0 mm since 11 Nov 2026 (was 3.05), so these read 5.0 / 8.0 / 15.0 mm as the 2 / 5 / 12 did
def test_the_grasp_is_lowered_only_when_the_tip_sits_more_than_6mm_high(pl, cc, table, monkeypatch, capsys, tmp_path, high_mm):
    arm = ShortArm(cc, table, shortfall_mm=7.0, high_mm=high_mm)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--grasp-tilt-deg", "0", "--state", str(tmp_path / "s.json")]); st = stages(out); rec, = records(out)
    h = st["grasp_pose"]["h_mm"]; want = min(h - 3.0, 6.0) if h > 6.0 else 0.0
    assert h == pytest.approx(pl.GRASP_H_IN * 25.4 + high_mm, abs=0.2) and st["grasp_pose"]["lowered_mm"] == pytest.approx(want, abs=0.06) and rec["grasp_lowered_mm"] == pytest.approx(want, abs=0.06)
    assert (want == 0.0) == (high_mm == 4.0) and (want == 6.0) == (high_mm == 14.0), "5 mm high -> no change; 8 -> lowered 5; 15 -> capped at 6"
    low = [h_in for (_, _, h_in) in arm.ik_calls if h_in < 0.2]   # stage 2's grasp, then the lift stage's grasp, close1, close2 (--grasp-tilt-deg 0 here: the lower/raise rule is tested on its own; the tilt changes the tip height by a mm or two)
    assert len(low) == 4 and low[0] == pytest.approx(pl.GRASP_H_IN, abs=1e-4) and all(h == pytest.approx(pl.GRASP_H_IN - want / 25.4, abs=0.005) for h in low[1:]), "the lowering goes through /api/ik for the grasp AND the close steps (h is printed to 0.1 mm, so 0.1 mm of slack)"
    assert pl.rec.runs[0].metrics_["grasp_lowered_mm"] == pytest.approx(want, abs=0.06)


def test_open_loop_is_one_run_of_the_whole_plan_at_80(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--open-loop", "--state", str(tmp_path / "s.json")])
    assert arm.calls == [(ALL, 80.0)] and not stages(out)
    rec, = records(out); assert rec["mode"] == "open" and rec["corr1_mm"] is None and rec["corr2_mm"] is None and rec["grasp_h_mm"] is None and rec["final_residual_mm"] is None
    assert set(rec["sag_deg"]) == set(ALL); r, = pl.rec.runs
    assert r.params_["mode"] == "open" and r.params_["max_deg_s"] == 80.0 and "corr1_mm" not in r.metrics_ and "grasp_h_mm" not in r.metrics_ and r.metrics_["sag_hover_nut"] >= 0
    out2 = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--open-loop", "--max-deg-s", "30", "--state", str(tmp_path / "s2.json")])
    assert arm.calls[-1][1] == 30.0 and records(out2)[0]["mode"] == "open"


def test_jaws_open_to_35_by_default_and_follow_the_flag(pl, cc, table, monkeypatch, capsys, tmp_path):
    assert pl.JAW_OPEN == 35 and [g for _, _, _, g, _, _ in pl.WP if g] == [35, 35, 35, 35, 20, 35, 35]
    arm = ShortArm(cc, table); out = drive(pl, monkeypatch, capsys, arm, ["--runs", "2", "--state", str(tmp_path / "s.json")])
    grips = {w["name"]: w["gripper"] for p in arm.plans for w in p["waypoints"]}
    assert grips == {"hover_nut": 35, "pregrasp_up": 35, "pregrasp": 35, "grasp": 35, "close1": 20, "close2": 0, "lift": 0, "hover_tgt": 0, "place": 0, "open": 35, "lift2": 35, "rest": 0}
    assert [r["run"] for r in records(out)] == [1, 2] and all(r["picked"] for r in records(out)) and len(arm.calls) == 8, "two closed-loop picks, four stages each; the second starts from the learned biases"
    arm2 = ShortArm(cc, table); drive(pl, monkeypatch, capsys, arm2, ["--runs", "1", "--jaw-open", "30", "--state", str(tmp_path / "s2.json")])
    assert {w["gripper"] for p in arm2.plans for w in p["waypoints"] if w["name"] in ("hover_nut", "grasp", "open", "lift2")} == {30} and pl.rec.runs[-1].params_["jaw_open"] == 30


def test_closed_loop_refuses_a_table_without_an_arm_fit(pl, cc, table, monkeypatch, capsys, tmp_path):
    json.dump({"units": "in"}, open(os.path.join(pl.ROOT, "jetson", "calib", "top_table.json"), "w")); arm = ShortArm(cc, table)
    with pytest.raises(SystemExit, match="closed loop needs the arm fit"): drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")])


# ---------------------------------------------------------------- the wrist check
def synthetic_frame(ring_uv=(352, 262), r=18, shadow_uv=(335, 335), ring=True):
    """A grey mat with a bright annulus round a dark hole (the nut) and a dark blob 70 px away (the gripper's shadow)."""
    img = np.full((480, 640, 3), 110, np.uint8); cv2.ellipse(img, shadow_uv, (28, 18), 0, 0, 360, (20, 20, 20), -1)
    if ring: cv2.circle(img, ring_uv, r, (235, 235, 235), -1); cv2.circle(img, ring_uv, int(r * 0.4), (25, 25, 25), -1)
    return img


def test_ring_finder_picks_the_bright_ring_not_the_dark_shadow(wc):
    f = wc.find_ring(synthetic_frame(), (335, 256))
    assert f and abs(f["uv"][0] - 352) <= 2 and abs(f["uv"][1] - 262) <= 2 and 14 <= f["r_px"] <= 22 and f["score"] > 0.5, f
    assert wc.find_ring(synthetic_frame(ring=False), (335, 256)) is None, "a dark blob is not a ring"
    assert wc.find_ring(np.full((480, 640), 110, np.uint8), (335, 256)) is None, "nor is a bare mat"
    assert wc.find_ring(synthetic_frame(), (335, 256), radius_px=10) is None, "a ring outside the search radius is not the nut"
    edge = wc.find_ring(synthetic_frame(ring_uv=(20, 20)), (20, 20)); assert edge is None or (abs(edge["uv"][0] - 20) <= 2 and abs(edge["uv"][1] - 20) <= 2), "a ring at the frame edge: found or skipped, never a crash"
    assert wc.find_ring(synthetic_frame(ring_uv=(20, 20))[:30, :30], (15, 15)) is None, "an ROI smaller than the template is skipped"


def test_the_real_10nov_frame_the_ring_is_the_nut_and_the_prediction_is_close(wc, cc, table, wpose):
    """The frame the dark-blob prototype got wrong on: it chose the shadow at (333, 335); the nut is the ring at (350, 262)."""
    res = wc.wrist_check(cc, table, wpose, PROBE_NUT_IN, PROBE_JOINTS, open(PROBE_FRAME, "rb").read())
    assert res and abs(res["uv_found"][0] - 350) <= 3 and abs(res["uv_found"][1] - 262) <= 3 and res["r_px"] == 18
    assert math.hypot(res["uv_expected"][0] - 334.5, res["uv_expected"][1] - 255.8) < 12 and 130 < res["depth_mm"] < 165   # 10 Nov: (334.5, 255.8), 147 mm; a re-solved mount may move it a few px
    assert res["offset_norm_mm"] < 8 and abs(res["nut_seen_in"][0] - 11.5) < 0.3 and abs(res["nut_seen_in"][1] - 6.4) < 0.3   # the nut sat at about (11.1-11.6, 6.3-6.5)


@pytest.mark.parametrize("mat_in,roll", [(PROBE_NUT_IN, 0.57), ((9.0, 6.0), 0.46), ((6.0, 5.0), 12.0)])
def test_project_then_backproject_returns_the_mat_point(wc, cc, table, wpose, mat_in, roll):
    joints = PROBE_JOINTS if mat_in == PROBE_NUT_IN else {**solve_joints(cc, table, SEED, mat_to_base(cc, table, *mat_in, 38.1)), "wrist_roll": roll}
    P = wc.mat_to_base_mm(cc, table, *mat_in); uv, depth = wc.project_base_point(cc, table, wpose, P, joints)
    assert 100 < depth < 220 and 0 <= uv[0] < 640 and 0 <= uv[1] < 480, (uv, depth)
    back = wc.backproject_pixel(cc, table, wpose, uv, joints); assert back is not None and np.linalg.norm(back - P) < 0.5
    mx, my = wc.base_to_mat_in(cc, table, back[:2]); assert math.hypot(mx - mat_in[0], my - mat_in[1]) * 25.4 < 0.5
    for off in ((60, 30), (-80, -40)):   # and from the pixel side, with a raised plane
        uv2 = np.array(uv) + off; P2 = wc.backproject_pixel(cc, table, wpose, uv2, joints, height_mm=4.0); assert P2 is not None and abs(P2[2] - table["arm_fit"]["table_z_mm"] - 4.0) < 1e-6
        uv3, _ = wc.project_base_point(cc, table, wpose, P2, joints); assert np.linalg.norm(uv3 - uv2) < 0.05


def test_wrist_check_returns_none_instead_of_raising(wc, cc, table, wpose, tmp_path, capsys):
    out = str(tmp_path / "outputs" / "run" / "check.jpg")
    assert wc.wrist_check(cc, table, wpose, PROBE_NUT_IN, PROBE_JOINTS, b"not a jpeg") is None
    assert wc.wrist_check(cc, table, None, PROBE_NUT_IN, PROBE_JOINTS, cv2.imencode(".jpg", synthetic_frame())[1].tobytes()) is None
    assert wc.wrist_check(cc, {"units": "in"}, wpose, PROBE_NUT_IN, PROBE_JOINTS, synthetic_frame()) is None
    assert wc.wrist_check(cc, table, wpose, PROBE_NUT_IN, {"shoulder_pan": "x"}, synthetic_frame()) is None and "wrist_check" in capsys.readouterr().err
    assert wc.wrist_check(cc, table, wpose, PROBE_NUT_IN, PROBE_JOINTS, synthetic_frame(ring=False), out) is None and os.path.isfile(out), "no ring: None, but the annotated frame is still written"


def test_wrist_check_cli_annotates_and_reports(wc, cc, table, wpose, tmp_path):
    uv, _ = wc.project_base_point(cc, table, wpose, wc.mat_to_base_mm(cc, table, *PROBE_NUT_IN), PROBE_JOINTS); ring = (int(round(uv[0])) + 12, int(round(uv[1])) + 5)
    frame = tmp_path / "wrist.jpg"; cv2.imwrite(str(frame), synthetic_frame(ring_uv=ring, shadow_uv=(ring[0] - 20, ring[1] + 70))); out = tmp_path / "out" / "check.jpg"
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "wrist_check.py"), "--image", str(frame), "--joints", json.dumps(PROBE_JOINTS), "--nut", *map(str, PROBE_NUT_IN), "--table", FIXTURE_TABLE, "--wrist-pose", WRIST_POSE, "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(r.stdout); assert d["ok"] and abs(d["uv_found"][0] - ring[0]) <= 2 and abs(d["uv_found"][1] - ring[1]) <= 2 and d["annotated"] == str(out) and out.is_file()
    assert 2.0 < d["offset_norm_mm"] < 6.0, "12 px at 147 mm with fx 529 is about 3.3 mm"
    r2 = subprocess.run([sys.executable, os.path.join(TOOLS, "wrist_check.py"), "--image", str(frame), "--joints", json.dumps(PROBE_JOINTS), "--nut", "6", "5", "--table", FIXTURE_TABLE, "--wrist-pose", WRIST_POSE, "--out", str(tmp_path / "no.jpg")],
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 2 and json.loads(r2.stdout)["ok"] is False


def test_the_loop_runs_the_wrist_check_at_hover_and_survives_a_dead_feed(pl, wc, cc, table, wpose, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table, shortfall_mm=7.0)

    def live_frame():   # the nut drawn where the camera at the REACHED hover must see the commanded point, plus the shadow
        j = {k: arm.rows[-1][-1]["reached"][k] for k in ARM}; uv, _ = wc.project_base_point(cc, table, wpose, wc.mat_to_base_mm(cc, table, *PROBE_NUT_IN), j)
        return cv2.imencode(".jpg", synthetic_frame(ring_uv=(int(round(uv[0])), int(round(uv[1]))), shadow_uv=(int(uv[0]) - 15, int(uv[1]) + 75)))[1].tobytes()
    monkeypatch.setattr(pl, "fetch_wrist", live_frame)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--wrist-check", "--no-jaw-check", "--state", str(tmp_path / "s.json")]); st = stages(out); rec, = records(out)
    assert list(st) == ["hover", "wrist_check", "grasp_pose", "final"] and st["wrist_check"]["offset_norm_mm"] < 1.5 and st["wrist_check"]["px_off"] < 3
    assert rec["wrist_offset_mm"] == st["wrist_check"]["offset_mm"] and pl.rec.runs[0].metrics_["wrist_offset_mm"] < 1.5 and pl.rec.runs[0].params_["wrist_check"] == 1
    jpgs = os.listdir(os.path.join(pl.ROOT, "outputs", "run")); assert len(jpgs) == 1 and jpgs[0].startswith("wrist_check_") and f"frames/{jpgs[0]}" in pl.rec.runs[0].files
    assert math.hypot(*st["final"]["residual_mm"]) < 1.0, "the check does not steer the grasp"
    monkeypatch.setattr(pl, "fetch_wrist", lambda: None); arm2 = ShortArm(cc, table)
    out2 = drive(pl, monkeypatch, capsys, arm2, ["--runs", "1", "--wrist-check", "--no-jaw-check", "--state", str(tmp_path / "s2.json")]); st2 = stages(out2); rec2, = records(out2)
    assert st2["wrist_check"]["ring"] is None and rec2["wrist_offset_mm"] is None and rec2["picked"] and "wrist_offset_mm" not in pl.rec.runs[-1].metrics_, "no frame: the pick goes on without the check"


def test_fake_arm_is_honest(cc, table):
    """The helper itself: the fake's reached tip is exactly 7 mm short along base->tip, its IK lands on the point it was asked for."""
    arm = ShortArm(cc, table, shortfall_mm=7.0, high_mm=2.0); j = arm.api("/api/ik?x_in=11.44&y_in=6.26&h_in=1.5")["joints_deg"]
    tip = tip_of(cc, table, j); assert np.linalg.norm(tip - mat_to_base(cc, table, 11.44, 6.26, 1.5 * 25.4)) < 0.01 and j["wrist_roll"] == table["arm_fit"]["wrist_roll_deg"]
    got = tip_of(cc, table, arm.reached(j)); d = got - tip
    assert d[2] == pytest.approx(2.0, abs=0.01) and np.linalg.norm(d[:2]) == pytest.approx(7.0, abs=0.01) and np.dot(d[:2], tip[:2] - [38.8353, 0.0]) < 0


def test_the_last_stage_moves_to_the_corrected_point_in_the_air_then_comes_straight_down(pl, cc, table, monkeypatch, capsys, tmp_path):
    """TC, 10 Nov (first10c): the fixed jaw "lands a 1/4 in too far back, then slides up to the nut and hits it". The final stage used to start
    with `grasp` at the corrected point, a slide along the mat by c2; it now rises to PREGRASP_H_IN over that point first."""
    arm = ShortArm(cc, table, shortfall_mm=7.0); drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--grasp-tilt-deg", "0", "--state", str(tmp_path / "s.json")])
    final = arm.plans[2]["waypoints"]; assert [w["name"] for w in final][:3] == ["pregrasp_up", "pregrasp", "grasp"]
    pre_h = [h for (_, _, h) in arm.ik_calls if abs(h - pl.PREGRASP_H_IN) < 1e-6]; assert pre_h, "pregrasp goes through /api/ik at its own height"
    xa, ya, ha = pl.fk_mat(cc, table, final[1]["desired"]); xb, yb, hb = pl.fk_mat(cc, table, final[2]["desired"])
    assert math.hypot(xa - xb, ya - yb) * 25.4 < 0.5 and ha - hb == pytest.approx((pl.PREGRASP_H_IN - pl.GRASP_H_IN) * 25.4, abs=0.5), "straight down onto the corrected point"
    assert pl.PREGRASP_H_IN * 25.4 >= 10.0 - 0.2


@pytest.mark.parametrize("low_mm,raise_mm", [(-3.0, 3.0), (-7.0, 5.0), (1.0, 0.0)])   # why: h reads 1.0 + low_mm now that the grasp is at 1.0 mm (was 3.05): -2 -> raise 3, -6 -> capped at 5, +2 -> none
def test_a_grasp_that_presses_into_the_mat_is_raised(pl, cc, table, monkeypatch, capsys, tmp_path, low_mm, raise_mm):
    """10 Nov: at the longer reaches the open-jaw grasp read -3.4 to -0.8 mm (pressing on the mat), stalled short and then dragged. A reading
    under 1 mm raises the grasp so the readback lands at 1 mm, at most 5 mm."""
    arm = ShortArm(cc, table, shortfall_mm=7.0, high_mm=low_mm)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--grasp-tilt-deg", "0", "--state", str(tmp_path / "s.json")]); st = stages(out); rec, = records(out)
    h = st["grasp_pose"]["h_mm"]; assert h == pytest.approx(pl.GRASP_H_IN * 25.4 + low_mm, abs=0.2)
    assert st["grasp_pose"]["lowered_mm"] == pytest.approx(-raise_mm, abs=0.06) and rec["grasp_lowered_mm"] == pytest.approx(-raise_mm, abs=0.06)
    hs = [pl.fk_mat(cc, table, w["desired"])[2] for w in arm.plans[2]["waypoints"] if w["name"] in ("grasp", "close1", "close2")]   # the final stage, as planned
    assert len(hs) == 3 and all(x == pytest.approx(pl.GRASP_H_IN * 25.4 + raise_mm, abs=0.5) for x in hs), hs
