"""The grasp aim (10 Nov 2026): the arm fit's tip is where the CLOSED jaws meet, so a grasp aimed at the nut brought the fixed jaw down on it
and pushed it forward (seen by TC). tools/pick_loop.py now steers the fixed jaw's tip GRASP_AIM_MM behind the nut along the jaw-opening
direction, so the nut sits inside the open gap. Checked here: that direction against the URDF's moving-jaw joint at jaw opening 35 and against
the real wrist frame, the aim arithmetic, and the closed loop landing the fixed jaw beside the nut. No dashboard, no ssh, no arm."""
import math, os, xml.etree.ElementTree as ET
import numpy as np, pytest
from pickhelpers import ARM, PROBE_JOINTS, PROBE_NUT_IN, ROOT, WRIST_POSE, ShortArm, drive

URDF = os.path.join(ROOT, "dashboard", "urdf", "so101.urdf")
GRIPPER_RANGE_TICKS = (1395, 2865)   # follower gripper calibration since 29 Oct 2026 (NEXT_STEPS.md; ll_follower.json on the Jetson)


def stages(out): return {o["stage"]: o for o in out if "stage" in o}
def records(out): return [o for o in out if "picked" in o]


def _joint(name):
    j = next(j for j in ET.parse(URDF).getroot().iter("joint") if j.get("name") == name); o = j.find("origin")
    return np.array([float(v) for v in o.get("xyz").split()]), np.array([float(v) for v in o.get("rpy").split()]), j


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _rot(axis, a):
    k = np.asarray(axis, float) / np.linalg.norm(axis); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def moving_tip_local(jaw_open):
    """The moving jaw's tip in gripper_link (metres) at `jaw_open` on LeRobot's 0..100 scale, from the URDF joint: the tip coincides with the
    fixed tip when closed and swings about the `gripper` joint axis by the calibrated fraction of the gripper's tick range."""
    tip, _, _ = _joint("gripper_frame_joint"); piv, rpy, j = _joint("gripper")
    axis = _rpy(*rpy) @ np.array([float(v) for v in j.find("axis").get("xyz").split()])
    ang = math.radians((GRIPPER_RANGE_TICKS[1] - GRIPPER_RANGE_TICKS[0]) * 360.0 / 4096.0 * jaw_open / 100.0)
    return piv + _rot(axis, ang) @ (tip - piv)


def _mat_mm(cc, table, P_m): return np.array(cc.arm_to_table(table, P_m[0] * 1000.0, P_m[1] * 1000.0))


def test_the_jaw_direction_is_the_urdf_moving_jaw_opening_at_35(pl, cc, table):
    G = cc.fk_frames({k: PROBE_JOINTS[k] for k in ARM}, table["arm_fit"]["joint_offsets_deg"])["gripper_link"]
    fixed = G[:3, :3] @ _joint("gripper_frame_joint")[0] + G[:3, 3]; moving = G[:3, :3] @ moving_tip_local(35) + G[:3, 3]
    d = _mat_mm(cc, table, moving) - _mat_mm(cc, table, fixed); u = pl.jaw_dir_mat(cc, table, PROBE_JOINTS)
    assert 50.0 < np.linalg.norm(d) < 70.0, f"jaws at 35 open the tips {np.linalg.norm(d):.1f} mm apart on the mat (58-61 mm on 10 Nov)"
    ang = math.degrees(math.acos(np.clip(np.dot(d / np.linalg.norm(d), u), -1, 1)))
    assert ang < 2.0, f"jaw_dir_mat {u} is {ang:.2f} deg off the URDF moving jaw's opening {d / np.linalg.norm(d)}"
    assert abs(math.hypot(*u) - 1.0) < 1e-9


def test_the_jaw_direction_points_at_the_moving_jaw_in_the_real_wrist_frame(pl, cc, table, wc):
    """tests/tools/fixtures/wrist_probe_20261110.jpg (the hover at PROBE_JOINTS, jaws at 35): the fixed jaw's tip is near pixel (300, 330), the open
    moving jaw fills the right edge (x > 560). The hand-eye pose must put them there, and a step along jaw_dir_mat must move to the right."""
    import json
    wpose = json.load(open(WRIST_POSE)); G = cc.fk_frames({k: PROBE_JOINTS[k] for k in ARM}, table["arm_fit"]["joint_offsets_deg"])["gripper_link"]
    fixed_mm = (G[:3, :3] @ _joint("gripper_frame_joint")[0] + G[:3, 3]) * 1000.0; moving_mm = (G[:3, :3] @ moving_tip_local(35) + G[:3, 3]) * 1000.0
    (uf, vf), _ = wc.project_base_point(cc, table, wpose, fixed_mm, PROBE_JOINTS); (um, vm), _ = wc.project_base_point(cc, table, wpose, moving_mm, PROBE_JOINTS)
    assert 280 < uf < 345 and 290 < vf < 345, f"fixed tip at ({uf:.0f}, {vf:.0f})"
    assert um > 560 and 300 < vm < 400, f"moving tip at ({um:.0f}, {vm:.0f})"
    # 10 mm along jaw_dir_mat, in the arm's horizontal plane at the tip's height (the arm and mat frames differ by the fitted yaw only)
    u = np.array(pl.jaw_dir_mat(cc, table, PROBE_JOINTS)); yaw = math.radians(table["arm_yaw_deg"]); R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    step = np.r_[fixed_mm[:2] + 10.0 * (R @ u), fixed_mm[2]]; (us, vs), _ = wc.project_base_point(cc, table, wpose, step, PROBE_JOINTS)
    assert 35 < us - uf < 65 and abs(vs - vf) < 15, f"10 mm toward the moving jaw moves {us - uf:.0f} px right (48 px on 10 Nov), {vs - vf:.0f} px down"


def test_the_aim_point_moves_the_fixed_jaw_back_from_the_nut_along_the_jaw_direction(pl):
    assert pl.GRASP_AIM_MM == 4.0, "10 mm landed the fixed jaw 1/4 in too far back (TC, first10c)"
    a = pl.aim_point([10.0, 6.0], (0.0, 1.0), 25.4); assert a == pytest.approx([10.0, 5.0])
    a = pl.aim_point([10.0, 6.0], (0.6, 0.8), 10.0); assert math.hypot(10.0 - a[0], 6.0 - a[1]) * 25.4 == pytest.approx(10.0)
    assert (10.0 - a[0]) / (6.0 - a[1]) == pytest.approx(0.75), "straight back along the jaw direction"
    assert pl.aim_point([10.0, 6.0], None, 10.0) == [10.0, 6.0] and pl.aim_point([10.0, 6.0], (0.0, 1.0), 0.0) == [10.0, 6.0], "no direction or no aim: the nut itself"


def test_the_closed_loop_lands_the_fixed_jaw_grasp_aim_mm_behind_the_nut_by_default(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table, shortfall_mm=7.0)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")]); st = stages(out); rec, = records(out)
    final = {k: arm.rows[-2][1]["reached"][k] for k in ARM}   # the FINAL grasp (after pregrasp), read back
    x, y, _ = pl.fk_mat(cc, table, final); u = pl.jaw_dir_mat(cc, table, final)
    nut = np.array(PROBE_NUT_IN); tip = np.array([x, y]); d_mm = (nut - tip) * 25.4
    aim = pl.GRASP_AIM_MM; assert np.linalg.norm(d_mm) == pytest.approx(aim, abs=1.0), f"the fixed jaw's tip ends {np.linalg.norm(d_mm):.1f} mm from the nut"
    assert np.dot(d_mm / np.linalg.norm(d_mm), u) > 0.99, "and the nut lies toward the open moving jaw, inside the gap"
    assert st["hover"]["aim_mm"] == aim and st["hover"]["jaw_dir"] == rec["jaw_dir"] and rec["aim_mm"] == aim
    assert rec["aim_in"] == pytest.approx(pl.aim_point(PROBE_NUT_IN, rec["jaw_dir"], aim), abs=1e-3)
    assert rec["final_residual_mm"] < 1.0, "the residual is measured against the aim point, not the nut"
    r, = pl.rec.runs; assert r.params_["grasp_aim_mm"] == aim and r.metrics_["aim_mm"] == aim


def test_grasp_aim_0_aims_the_fixed_jaw_at_the_nut_as_before(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table, shortfall_mm=7.0)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--grasp-aim-mm", "0", "--state", str(tmp_path / "s.json")]); rec, = records(out)
    x, y, _ = pl.fk_mat(cc, table, {k: arm.rows[-2][1]["reached"][k] for k in ARM})
    assert math.hypot(x - PROBE_NUT_IN[0], y - PROBE_NUT_IN[1]) * 25.4 < 1.0 and rec["aim_mm"] == 0.0 and rec["jaw_dir"] is None
    assert pl.rec.runs[0].params_["grasp_aim_mm"] == 0.0


def test_the_place_is_unchanged_by_the_aim(pl, cc, table, monkeypatch, capsys, tmp_path):
    """The nut ends up pinched against the fixed jaw where it was held before, so the carry and place waypoints keep their targets."""
    plans = {}
    for aim in ("0", "10"):
        arm = ShortArm(cc, table, shortfall_mm=7.0); drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--grasp-aim-mm", aim, "--state", str(tmp_path / f"s{aim}.json")])
        plans[aim] = {w["name"]: w["desired"] for w in arm.plans[-1]["waypoints"]}
    for wp in ("hover_tgt", "place", "open", "lift2"):
        assert plans["0"][wp] == pytest.approx(plans["10"][wp], abs=1e-3), wp


def test_the_session_runner_grasps_with_the_default_aim(autorig, capsys, monkeypatch):
    rig = autorig(); seen = []; orig = rig.pl.closed_loop_pick
    def spy(*args, **kw):
        plan, rows, info = orig(*args, **kw); seen.append(info); return plan, rows, info
    monkeypatch.setattr(rig.pl, "closed_loop_pick", spy)
    summary, out = rig.run(capsys, cycles=1)
    assert rig.runner.pick_args.grasp_aim_mm == rig.pl.GRASP_AIM_MM == 4.0
    assert len(seen) == 2 and all(i["aim_mm"] == 4.0 and i["jaw_dir"] for i in seen), "the pick AND the reset aim the fixed jaw beside the nut"
    assert [o for o in out if o.get("event") == "cycle"][0]["ok"], "and the dry-run world (grasp within 15 mm) still counts it a pick"
