"""Inch layouts, the table->arm fit from jogged tip positions, and SO-101 forward kinematics."""
import json, math, subprocess, sys
import cv2, numpy as np, pytest
import synth
from synth import SCRIPT

PY = sys.executable
IN = 25.4


def run(*args):
    return subprocess.run([PY, SCRIPT, *map(str, args)], capture_output=True, text=True, timeout=120)


def inch_layout():
    """The voice chat's plan: Fiskars 18x12 in mat, origin at the arm-side left corner, four 2 in markers 2 in from each corner."""
    return {"units": "in", "dictionary": "DICT_4X4_50", "marker_size": 2.0,
            "markers": {"0": [2, 2], "1": [16, 2], "2": [2, 10], "3": [16, 10]}, "arm_base": [9, 0], "arm_yaw_deg": 90}


def write(tmp_path, obj, name):
    p = tmp_path / name; p.write_text(json.dumps(obj)); return str(p)


# ---------------------------------------------------------------- units
def test_inch_layout_is_converted_to_mm(cc, tmp_path):
    lay = cc.load_layout(write(tmp_path, inch_layout(), "l.json"))
    assert lay["units"] == "in" and lay["marker_size_mm"] == pytest.approx(50.8)
    assert lay["markers"][1] == pytest.approx([16 * IN, 2 * IN]) and lay["arm_base_mm"] == pytest.approx([9 * IN, 0.0]) and lay["arm_yaw_deg"] == 90


def test_mm_layout_still_accepts_old_keys(cc, tmp_path):
    lay = cc.load_layout(write(tmp_path, {**synth.default_layout(), "markers": {str(k): v for k, v in synth.default_layout()["markers"].items()}}, "l.json"))
    assert lay["units"] == "mm" and lay["marker_size_mm"] == 40.0 and lay["arm_base_mm"] == [200.0, 330.0]


@pytest.mark.parametrize("mutate,msg", [
    (lambda l: l.update(units="furlong"), "units must be one of"),
    (lambda l: l.pop("marker_size"), r"marker_size \(in in\) is required"),
    (lambda l: l.update(arm_base=[9]), r"arm_base must be \[x, y\]"),
])
def test_inch_layout_errors(cc, tmp_path, mutate, msg):
    lay = inch_layout(); mutate(lay)
    with pytest.raises(cc.CalibError, match=msg): cc.load_layout(write(tmp_path, lay, "l.json"))


def test_solve_table_reports_inches_too(cc, true_intrinsics):
    lay_in = {"units": "in", "dictionary": "DICT_4X4_50", "marker_size_mm": 50.8, "markers": {0: [2 * IN, 2 * IN], 1: [16 * IN, 2 * IN], 2: [2 * IN, 10 * IN], 3: [16 * IN, 10 * IN]}, "arm_base_mm": [9 * IN, 0.0], "arm_yaw_deg": 90.0}
    rvec, tvec = synth.table_camera_pose(height_mm=700.0, look_at=(9 * IN, 6 * IN))
    img = synth.render_table(lay_in, rvec, tvec)
    res = cc.solve_table(img, lay_in, true_intrinsics)
    assert res["units"] == "in" and res["markers_used"] == [0, 1, 2, 3]
    assert res["residual_in_max"] == pytest.approx(res["residual_mm_max"] / IN) and res["residual_in_max"] < 0.06
    x, y = cc.pixel_to_table(res, *cc.table_to_pixel(res, 9 * IN, 6 * IN)); assert cc.in_units(res, (x, y)) == pytest.approx([9, 6], abs=0.05)
    assert cc.table_to_arm(res, 9 * IN, 8 * IN) == pytest.approx((8 * IN, 0.0), abs=1e-6)          # 8 in straight ahead of a base at (9,0) facing +Y
    assert cc.table_to_arm(res, 12 * IN, 0.0) == pytest.approx((0.0, -3 * IN), abs=1e-6)          # +X on the mat is the arm's right


# ---------------------------------------------------------------- rigid fit
def test_fit_rigid_2d_recovers_pose_and_round_trips(cc):
    calib = {"arm_base_mm": [9 * IN, 0.0], "arm_yaw_deg": 90.0}
    table = np.array([[2 * IN, 2 * IN], [16 * IN, 2 * IN], [2 * IN, 10 * IN], [16 * IN, 10 * IN]])
    arm = np.array([cc.table_to_arm(calib, *p) for p in table]) + np.array([[0.3, -0.2], [-0.4, 0.1], [0.2, 0.3], [-0.1, -0.2]])   # jog noise, mm
    R, t, res = cc.fit_rigid_2d(table, arm); base, yaw = cc.arm_pose_from_rigid(R, t)
    assert base == pytest.approx([9 * IN, 0.0], abs=0.5) and yaw == pytest.approx(90.0, abs=0.2) and res.max() < 0.6
    fitted = {"arm_base_mm": base, "arm_yaw_deg": yaw}
    for p in table: assert cc.table_to_arm(fitted, *p) == pytest.approx(cc.table_to_arm(calib, *p), abs=0.6)


@pytest.mark.parametrize("yaw", [-90.0, 0.0, 37.5, 180.0])
def test_fit_rigid_2d_any_heading(cc, yaw):
    calib = {"arm_base_mm": [123.0, -45.0], "arm_yaw_deg": yaw}
    table = np.array([[0, 0], [300, 0], [0, 200], [300, 200], [150, 100]], float)
    arm = np.array([cc.table_to_arm(calib, *p) for p in table])
    R, t, _ = cc.fit_rigid_2d(table, arm); base, got = cc.arm_pose_from_rigid(R, t)
    assert base == pytest.approx([123.0, -45.0], abs=1e-6) and math.isclose(math.cos(math.radians(got - yaw)), 1.0, abs_tol=1e-9)


def test_fit_rigid_2d_refuses_mirror(cc):
    table = np.array([[0, 0], [300, 0], [0, 200], [300, 200]], float); arm = table * np.array([1, -1])
    with pytest.raises(cc.CalibError, match="mirror image"): cc.fit_rigid_2d(table, arm)
    with pytest.raises(cc.CalibError, match="at least 2"): cc.fit_rigid_2d(table[:1], arm[:1])


def test_load_pairs_by_marker_and_by_table_point(cc, tmp_path):
    calib = {"units": "in", "layout_markers": {"0": [2 * IN, 2 * IN], "1": [16 * IN, 2 * IN]}}
    p = write(tmp_path, {"arm_units": "in", "pairs": [{"marker": 0, "arm": [1, 2]}, {"marker": 1, "arm": [3, 4]}, {"table": [9, 6], "arm": [5, 6]}]}, "pairs.json")
    tab, arm = cc.load_pairs(p, calib)
    assert np.allclose(tab, [[2 * IN, 2 * IN], [16 * IN, 2 * IN], [9 * IN, 6 * IN]]) and np.allclose(arm, [[IN, 2 * IN], [3 * IN, 4 * IN], [5 * IN, 6 * IN]])
    for bad, msg in (({"pairs": [{"marker": 7, "arm": [1, 2]}, {"marker": 0, "arm": [1, 2]}]}, "not in the table calibration"),
                     ({"pairs": [{"marker": 0}, {"marker": 1, "arm": [1, 2]}]}, "needs 'arm'"),
                     ({"pairs": [{"marker": 0, "arm": [1, 2]}]}, "at least 2"),
                     ({"arm_units": "cubit", "pairs": []}, "arm_units must be")):
        with pytest.raises(cc.CalibError, match=msg): cc.load_pairs(write(tmp_path, bad, "bad.json"), calib)


# ---------------------------------------------------------------- forward kinematics
# Reference tip positions read from the dashboard's three.js URDF model (urdf-loader) at the same joint angles, 8 Nov 2026.
URDF_REF = [
    ((0, 0, 0, 0, 0), (391.4, 0.0, 226.5)),
    ((30, -45, 60, 20, 0), (236.4, -114.1, 88.3)),
    ((-20, 30, -40, -50, 90), (344.8, 102.8, 366.4)),
]


@pytest.mark.parametrize("deg,ref", URDF_REF)
def test_fk_matches_the_urdf_model(cc, deg, ref):
    tip, point = cc.fk_tip(dict(zip(cc.JOINT_ORDER, deg)))
    assert tip == pytest.approx(ref, abs=0.15) and np.linalg.norm(point) == pytest.approx(1.0)


def test_fk_rest_pose_points_forward_and_pan_rotates_about_z(cc):
    tip0, point0 = cc.fk_tip({})
    assert point0 == pytest.approx([1, 0, 0], abs=1e-3)
    tip90, _ = cc.fk_tip({"shoulder_pan": 90})
    assert math.hypot(tip90[0] - 38.8353, tip90[1]) == pytest.approx(math.hypot(tip0[0] - 38.8353, tip0[1]), abs=0.01) and tip90[2] == pytest.approx(tip0[2], abs=0.01)


def test_reach_is_plausible(cc):
    any_r, down_r = cc.reach_table(0.0, step_deg=6.0)
    assert 300 < any_r < 450 and 200 < down_r <= any_r


# ---------------------------------------------------------------- CLI
def test_cli_fk_and_armfit_round_trip(cc, tmp_path, true_intrinsics):
    lay = inch_layout(); write(tmp_path, lay, "top_layout.json")
    lay_mm = {"units": "in", "dictionary": "DICT_4X4_50", "marker_size_mm": 50.8, "markers": {int(k): [v[0] * IN, v[1] * IN] for k, v in lay["markers"].items()}}
    rvec, tvec = synth.table_camera_pose(height_mm=700.0, look_at=(9 * IN, 6 * IN)); scene = tmp_path / "scene"; scene.mkdir()
    cv2.imwrite(str(scene / "t.png"), synth.render_table(lay_mm, rvec, tvec))
    json.dump(true_intrinsics, open(tmp_path / "top_intrinsics.json", "w"))
    r = run("--calib-dir", tmp_path, "table", "--source", scene); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); assert out["residual_mm_max"] < 1.5
    # TC jogs the tip onto each marker; the "arm" numbers would come from `fk`. Here the true transform is base (9,0) facing +Y.
    truth = {"arm_base_mm": [9 * IN, 0.0], "arm_yaw_deg": 90.0}
    pairs = {"arm_units": "in", "pairs": [{"marker": i, "arm": [v / IN for v in cc.table_to_arm(truth, lay["markers"][str(i)][0] * IN, lay["markers"][str(i)][1] * IN)]} for i in range(4)]}
    pp = write(tmp_path, pairs, "pairs.json")
    r = run("--calib-dir", tmp_path, "armfit", "--pairs", pp); assert r.returncode == 0, r.stderr
    fit = json.loads(r.stdout)
    assert fit["arm_base_in"] == pytest.approx([9.0, 0.0], abs=0.01) and fit["arm_yaw_deg"] == pytest.approx(90.0, abs=0.05) and fit["residual_in_max"] < 0.01
    tab = json.load(open(tmp_path / "top_table.json")); assert tab["armfit"]["pairs"] == 4 and tab["arm_base_mm"] == pytest.approx([9 * IN, 0.0], abs=0.3)
    u, v = cc.table_to_pixel(tab, 9 * IN, 8 * IN)
    r = run("--calib-dir", tmp_path, "locate", f"{u:.2f}", f"{v:.2f}"); loc = json.loads(r.stdout)
    assert loc["table_in"] == pytest.approx([9, 8], abs=0.05) and loc["arm_in"] == pytest.approx([8, 0], abs=0.05)
    r = run("fk", "--deg", "0", "0", "0", "0", "0"); assert r.returncode == 0, r.stderr
    fk = json.loads(r.stdout); assert fk["tip_mm"] == pytest.approx([391.4, 0.0, 226.5], abs=0.1) and fk["tip_in"][0] == pytest.approx(15.41, abs=0.01)
    poses = write(tmp_path, {"m0": {"shoulder_pan": 30, "shoulder_lift": -45, "elbow_flex": 60, "wrist_flex": 20, "wrist_roll": 0, "gripper": 10}}, "poses.json")
    r = run("fk", "--pose", "m0", "--poses", poses); assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["tip_mm"] == pytest.approx([236.4, -114.1, 88.3], abs=0.1)
    r = run("fk", "--pose", "nope", "--poses", poses); assert r.returncode == 2 and "not in" in r.stderr
    r = run("fk"); assert r.returncode == 2 and "give --pose" in r.stderr


# ---------------------------------------------------------------- base pose + joint zero offsets from jogged points
def test_fit_arm_and_offsets_recovers_truth(cc):
    """Random reachable poses, true offsets applied, true base pose, 0.5 mm of jog noise: the fit must recover all seven numbers."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(5); true_off = {"shoulder_lift": -3.0, "elbow_flex": -7.2, "wrist_flex": 1.1}
    truth = {"arm_base_mm": [223.0, -77.0], "arm_yaw_deg": 89.3}; table_z = 5.0
    joints, pts = [], []

    def tip_of(j): return cc.fk_tip({k: v + true_off.get(k, 0.0) for k, v in j.items()})[0]

    while len(joints) < 7:
        j = {"shoulder_pan": rng.uniform(-70, 70), "shoulder_lift": rng.uniform(5, 55), "elbow_flex": rng.uniform(-40, 30), "wrist_flex": 60.0, "wrist_roll": rng.uniform(-30, 30), "gripper": 0.0}
        lo, hi = 30.0, 95.0                                                        # bring the tip down onto the table with the wrist, like a real jog
        if (tip_of({**j, "wrist_flex": lo})[2] - table_z) * (tip_of({**j, "wrist_flex": hi})[2] - table_z) > 0: continue
        for _ in range(40):
            mid = (lo + hi) / 2; zl = tip_of({**j, "wrist_flex": lo})[2] - table_z; zm = tip_of({**j, "wrist_flex": mid})[2] - table_z
            lo, hi = (lo, mid) if zl * zm <= 0 else (mid, hi)
        j["wrist_flex"] = (lo + hi) / 2; tip = tip_of(j)
        if abs(tip[2] - table_z) > 0.05 or math.hypot(tip[0], tip[1]) < 120: continue
        # the mat point the tip really touched: invert table_to_arm (arm -> table) for the true base pose
        yaw = math.radians(truth["arm_yaw_deg"]); ax, ay = tip[0], tip[1]
        tx = truth["arm_base_mm"][0] + ax * math.cos(yaw) - ay * math.sin(yaw); ty = truth["arm_base_mm"][1] + ax * math.sin(yaw) + ay * math.cos(yaw)
        # the recorded joints are what LeRobot reports (no offsets), the mat point is measured to +-0.5 mm
        joints.append(j); pts.append([tx + rng.normal(0, 0.5), ty + rng.normal(0, 0.5)])
    fit = cc.fit_arm_and_offsets(joints, pts)
    assert fit["n_points"] == 7 and fit["rms_mm"] < 1.0
    assert fit["arm_base_mm"] == pytest.approx(truth["arm_base_mm"], abs=2.0) and fit["arm_yaw_deg"] == pytest.approx(89.3, abs=0.3)
    assert fit["table_z_mm"] == pytest.approx(table_z, abs=1.0)
    for k, v in true_off.items(): assert fit["joint_offsets_deg"][k] == pytest.approx(v, abs=0.4), k
    assert max(fit["rigid_only_residual_mm"]) > 5, "the rigid-only fit should have been visibly worse"
    with pytest.raises(cc.CalibError, match="at least 4"): cc.fit_arm_and_offsets(joints[:3], pts[:3])


def test_fk_cli_applies_fitted_offsets(tmp_path):
    pytest.importorskip("scipy")
    json.dump({"units": "in", "arm_fit": {"joint_offsets_deg": {"shoulder_lift": -3.0, "elbow_flex": -7.2, "wrist_flex": 1.1}}}, open(tmp_path / "top_table.json", "w"))
    r = run("--calib-dir", tmp_path, "fk", "--deg", "0", "0", "0", "0", "0"); assert r.returncode == 0, r.stderr
    with_off = json.loads(r.stdout); assert with_off["joint_offsets_deg"]["elbow_flex"] == -7.2 and with_off["tip_mm"] != pytest.approx([391.4, 0.0, 226.5], abs=0.1)
    r = run("--calib-dir", tmp_path, "fk", "--deg", "0", "0", "0", "0", "0", "--raw"); raw = json.loads(r.stdout)
    assert "joint_offsets_deg" not in raw and raw["tip_mm"] == pytest.approx([391.4, 0.0, 226.5], abs=0.1)


# ---------------------------------------------------------------- inverse kinematics
def test_ik_round_trips_through_fk_with_offsets(cc):
    pytest.importorskip("scipy")
    off = {"shoulder_lift": -3.9, "elbow_flex": -5.2, "wrist_flex": -0.7}
    for target in ([200.0, 100.0, 10.0], [260.0, -140.0, 5.0], [150.0, 0.0, 40.0], [280.0, 60.0, 30.0]):   # radii 120-290 mm, inside the jogged envelope
        res = cc.ik_tip(target, off)
        assert res["pos_err_mm"] < 1.0 and res["tilt_deg"] < 2.0, (target, res)
        tip, point = cc.fk_tip({k: v + off.get(k, 0.0) for k, v in res["joints_deg"].items()})
        assert np.allclose(tip, target, atol=1.0) and point[2] < -0.999
        for k, (lo, hi) in cc.JOINT_LIMITS_DEG.items(): assert lo <= res["joints_deg"][k] <= hi


def test_ik_refuses_unreachable(cc):
    pytest.importorskip("scipy")
    with pytest.raises(cc.CalibError, match="not reachable"): cc.ik_tip([600.0, 0.0, 0.0], {})


def test_ik_cli_from_mat_point(tmp_path):
    pytest.importorskip("scipy")
    table = {"units": "in", "arm_base_mm": [223.8, -77.5], "arm_yaw_deg": 89.28, "arm_fit": {"table_z_mm": 4.2, "wrist_roll_deg": 1.1, "joint_offsets_deg": {"shoulder_lift": -3.91, "elbow_flex": -5.18, "wrist_flex": -0.69}}}
    json.dump(table, open(tmp_path / "top_table.json", "w"))
    r = run("--calib-dir", tmp_path, "ik", "--mat", "9", "6", "--units", "in", "--height", "1"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["pos_err_mm"] < 1.0 and out["tilt_deg"] < 2.0 and out["target_mm"][2] == pytest.approx(4.2 + 25.4, abs=0.01)
    assert out["wrist_roll_deg"] == 1.1 and out["joints_deg"]["wrist_roll"] == 1.1            # holds the roll the arm fit was jogged at
    r = run("--calib-dir", tmp_path, "ik", "--mat", "9", "6", "--units", "in", "--roll", "0"); assert json.loads(r.stdout)["joints_deg"]["wrist_roll"] == 0.0
    assert abs(out["joints_deg"]["shoulder_pan"]) < 5                      # the centre line is straight ahead of a base at x = 8.81 in
    r = run("--calib-dir", tmp_path, "ik", "--mat", "9", "40", "--units", "in"); assert r.returncode == 2 and "not reachable" in r.stderr


def test_fit_with_fixed_offsets_needs_only_three_points(cc):
    pytest.importorskip("scipy")
    off = {"shoulder_lift": -3.9, "elbow_flex": -5.2, "wrist_flex": -0.7}; truth = {"arm_base_mm": [230.0, -70.0], "arm_yaw_deg": 91.0}; z = 4.0
    rng = np.random.default_rng(11); joints, pts = [], []
    # lift/elbow pairs taken from the real jogged poses of 9 Nov (near markers ~ 11/28, far markers ~ 50/-35), pan varied
    for pan, lift, elbow in ((-64, 11, 28), (63, 11, 27), (-31, 49, -32), (32, 52, -37), (0, 30, 0), (-15, 40, -20), (20, 20, 15)):
        j = {"shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": 60.0, "wrist_roll": 0.0, "gripper": 0.0}
        f = lambda w: cc.fk_tip({**{k: v + off.get(k, 0.0) for k, v in j.items()}, "wrist_flex": w + off["wrist_flex"]})[0][2] - z
        ws = [w for w in range(0, 95) if f(w) > 0 and f(w + 1) <= 0]                # first wrist angle where the tip comes down through the table
        if not ws: continue
        lo, hi = float(ws[0]), float(ws[0] + 1)
        for _ in range(40):
            mid = (lo + hi) / 2; lo, hi = (lo, mid) if f(lo) * f(mid) <= 0 else (mid, hi)
        j["wrist_flex"] = (lo + hi) / 2; tip = cc.fk_tip({k: v + off.get(k, 0.0) for k, v in j.items()})[0]
        assert abs(tip[2] - z) < 0.05
        if len(joints) == 4: break
        yaw = math.radians(truth["arm_yaw_deg"]); tx = truth["arm_base_mm"][0] + tip[0] * math.cos(yaw) - tip[1] * math.sin(yaw); ty = truth["arm_base_mm"][1] + tip[0] * math.sin(yaw) + tip[1] * math.cos(yaw)
        joints.append(j); pts.append([tx + rng.normal(0, 0.3), ty + rng.normal(0, 0.3)])
    fit = cc.fit_arm_and_offsets(joints[:3], pts[:3], fixed_offsets=off)
    assert fit["offsets_fitted"] is False and fit["joint_offsets_deg"] == off
    assert fit["arm_base_mm"] == pytest.approx(truth["arm_base_mm"], abs=1.5) and fit["arm_yaw_deg"] == pytest.approx(91.0, abs=0.3) and fit["table_z_mm"] == pytest.approx(z, abs=0.5)
    with pytest.raises(cc.CalibError, match="at least 3"): cc.fit_arm_and_offsets(joints[:2], pts[:2], fixed_offsets=off)


# ---------------------------------------------------------------- re-solving the table keeps what is still valid
def _table_with_arm_fit(cc, tmp_path, table_scene, true_intrinsics, markers=None):
    lay = dict(table_scene["layout"])
    if markers: lay["markers"] = markers
    res = cc.solve_table(table_scene["img"], lay, true_intrinsics)
    res["arm_base_mm"] = [111.0, -22.0]; res["arm_yaw_deg"] = 88.0
    res["arm_fit"] = {"joint_offsets_deg": {"shoulder_lift": -3.9, "elbow_flex": -5.2, "wrist_flex": -0.7}, "table_z_mm": 3.3, "wrist_roll_deg": 0.5, "fitted": "2026-11-09 21:33", "points": [["table", [9.0, 3.0]]]}
    json.dump(res, open(tmp_path / "top_table.json", "w"))


def test_table_resolve_keeps_arm_pose_when_only_the_camera_moved(cc, tmp_path, table_scene, true_intrinsics):
    _table_with_arm_fit(cc, tmp_path, table_scene, true_intrinsics)
    json.dump(true_intrinsics, open(tmp_path / "top_intrinsics.json", "w"))
    lay = {**table_scene["layout"], "markers": {str(k): v for k, v in table_scene["layout"]["markers"].items()}}; (tmp_path / "top_layout.json").write_text(json.dumps(lay))
    scene = tmp_path / "scene"; scene.mkdir(); cv2.imwrite(str(scene / "t.png"), table_scene["img"])
    r = run("--calib-dir", tmp_path, "table", "--source", scene); assert r.returncode == 0, r.stderr
    assert "camera-only change" in r.stderr
    t = json.load(open(tmp_path / "top_table.json"))
    assert t["arm_base_mm"] == [111.0, -22.0] and t["arm_yaw_deg"] == 88.0 and t["arm_fit"]["wrist_roll_deg"] == 0.5 and t["arm_fit"]["joint_offsets_deg"]["elbow_flex"] == -5.2


def test_table_resolve_keeps_only_offsets_when_the_mat_moved(cc, tmp_path, table_scene, true_intrinsics):
    _table_with_arm_fit(cc, tmp_path, table_scene, true_intrinsics)
    json.dump(true_intrinsics, open(tmp_path / "top_intrinsics.json", "w"))
    moved = {str(k): [v[0] + 10.0, v[1]] for k, v in table_scene["layout"]["markers"].items()}     # layout shifted 10 mm: the mat moved
    (tmp_path / "top_layout.json").write_text(json.dumps({**table_scene["layout"], "markers": moved}))
    scene = tmp_path / "scene"; scene.mkdir(); cv2.imwrite(str(scene / "t.png"), table_scene["img"])
    r = run("--calib-dir", tmp_path, "table", "--source", scene); assert r.returncode == 0, r.stderr
    assert "marker layout changed" in r.stderr
    t = json.load(open(tmp_path / "top_table.json"))
    assert t["arm_fit"]["joint_offsets_deg"]["elbow_flex"] == -5.2 and "base_pose" in t["arm_fit"] and "wrist_roll_deg" not in t["arm_fit"]
    assert t["arm_base_mm"] == table_scene["layout"]["arm_base_mm"]                                   # back to the layout's guess


# ---------------------------------------------------------------- camera pose over the table
def test_camera_pose_from_markers_recovers_the_synthetic_camera(cc, table_scene):
    img = synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], dist=None)
    pose = cc.camera_pose_from_markers(img, table_scene["layout"], synth.K_TRUE)
    assert np.allclose(pose["position_mm"], table_scene["camera_position"], atol=6), (pose["position_mm"], table_scene["camera_position"])   # four 40 mm squares at 620 mm: a few mm is the floor
    assert pose["reproj_rms_px"] < 0.8 and 0 < pose["tilt_deg"] < 15 and pose["markers_used"] == [0, 1, 2, 3, 4, 5]
    assert math.hypot(*(np.array(pose["axis_hits_mm"]) - np.array([200.0, 150.0]))) < 60      # the synthetic camera looks roughly at the mat centre


def test_lens_height_pins_the_field_of_view(cc, table_scene):
    img = synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], dist=None)
    true_fx = synth.K_TRUE[0, 0]; true_hfov = math.degrees(2 * math.atan((synth.W / 2) / true_fx))
    hfov, pose = cc.fit_hfov_for_height(img, table_scene["layout"], synth.W, synth.H, table_scene["camera_position"][2])
    assert abs(hfov - true_hfov) < 1.0 and abs(pose["position_mm"][2] - table_scene["camera_position"][2]) < 1.0
    with pytest.raises(cc.CalibError, match="outside what FOV"): cc.fit_hfov_for_height(img, table_scene["layout"], synth.W, synth.H, 5000.0)


def test_campose_cli_writes_into_the_table_file(tmp_path, table_scene, true_intrinsics):
    lay = {**table_scene["layout"], "markers": {str(k): v for k, v in table_scene["layout"]["markers"].items()}}
    (tmp_path / "top_layout.json").write_text(json.dumps(lay)); json.dump({"units": "mm", "layout_markers": {}}, open(tmp_path / "top_table.json", "w"))
    scene = tmp_path / "scene"; scene.mkdir(); cv2.imwrite(str(scene / "t.png"), synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], dist=None))
    true_hfov = math.degrees(2 * math.atan((synth.W / 2) / synth.K_TRUE[0, 0]))
    r = run("--calib-dir", tmp_path, "campose", "--source", scene, "--hfov", f"{true_hfov:.3f}"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); assert abs(out["height_mm"] - table_scene["camera_position"][2]) < 5
    t = json.load(open(tmp_path / "top_table.json")); assert "camera_pose_est" in t and t["camera_pose_est"]["hfov_deg"] == pytest.approx(true_hfov, abs=1e-3)
    r = run("--calib-dir", tmp_path, "campose", "--source", scene, "--lens-height", f"{table_scene['camera_position'][2] / 25.4:.3f}", "--units", "in"); assert r.returncode == 0, r.stderr
    assert abs(json.loads(r.stdout)["hfov_deg"] - true_hfov) < 1.0
