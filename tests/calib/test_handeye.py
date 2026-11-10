"""Wrist camera hand-eye calibration (cam_calib.py handeye): a synthetic wrist camera bolted to gripper_link at a known pose
renders the real mat layout through the real arm kinematics for several follower poses; the solver must get the pose, the focal
length and the lens back. Then the real captures (jetson/calib/wristcam) must solve, write the file, and pass the check-image CLI."""
import json, math, os, subprocess, sys
import cv2, numpy as np, pytest
import synth
from synth import SCRIPT, ROOT

PY = sys.executable
IN = 25.4
REAL_VIEWS = os.path.join(ROOT, "jetson", "calib", "wristcam")
REAL_TABLE = os.path.join(ROOT, "jetson", "calib", "top_table.json")
REAL_LAYOUT = os.path.join(ROOT, "jetson", "calib", "top_layout.json")

# the real 9 Nov 2026 arm fit (top_table.json) and layout: the synthetic camera lives in the same world as the real one
CALIB = {"units": "in", "arm_base_mm": [229.55, -80.76], "arm_yaw_deg": 90.15, "layout_markers": {"0": [16 * IN, 10 * IN], "1": [2 * IN, 10 * IN], "2": [15 * IN, 4 * IN], "3": [3 * IN, 4 * IN]},
         "arm_fit": {"table_z_mm": 3.28, "joint_offsets_deg": {"shoulder_lift": -3.91, "elbow_flex": -5.18, "wrist_flex": -0.69}}}
LAYOUT_FILE = {"units": "in", "dictionary": "DICT_4X4_50", "marker_size": 2.0, "markers": {"0": [16, 10], "1": [2, 10], "2": [15, 4], "3": [3, 4]}}
# follower poses from the real session (w1, w2, w3, w5, w6) plus two variations: each looks at one marker from 120-220 mm
JOINTS = [
    {"shoulder_pan": -31.5, "shoulder_lift": 29.2, "elbow_flex": -25.7, "wrist_flex": 72.1, "wrist_roll": 13.5},
    {"shoulder_pan": -41.8, "shoulder_lift": -25.4, "elbow_flex": 48.0, "wrist_flex": 63.3, "wrist_roll": 12.9},
    {"shoulder_pan": 55.1, "shoulder_lift": -19.7, "elbow_flex": 53.2, "wrist_flex": 45.6, "wrist_roll": 17.3},
    {"shoulder_pan": 50.2, "shoulder_lift": -30.0, "elbow_flex": 46.8, "wrist_flex": 58.2, "wrist_roll": -20.8},
    {"shoulder_pan": -27.0, "shoulder_lift": 46.6, "elbow_flex": -40.1, "wrist_flex": 71.7, "wrist_roll": 38.4},
    {"shoulder_pan": -36.0, "shoulder_lift": 20.0, "elbow_flex": -10.0, "wrist_flex": 78.0, "wrist_roll": -15.0},
    {"shoulder_pan": 47.0, "shoulder_lift": -12.0, "elbow_flex": 40.0, "wrist_flex": 60.0, "wrist_roll": 35.0},
]
TRUE_POS = np.array([4.0, 66.0, -9.0])                 # mm in gripper_link
TRUE_FX, TRUE_DIST = 500.0, np.array([-0.30, 0.08, 0.0, 0.0, 0.0])
TRUE_ANGLES = {0: 180.0, 1: -88.5, 2: 179.0, 3: 180.7}  # glued at 90 deg steps, a degree or so off


def true_X(cc):
    rv, _ = cc._handeye_init(87.0, 66.0, -9.0, 29.0, 4.0)
    R = cc._rotvec_to_R(rv) @ cc._rotvec_to_R(np.radians([1.5, -2.0, 0.8]))   # a little extra so no start is exact
    return R, TRUE_POS


def render_view(cc, R_X, t_X, joints, layout_mm):
    """What the synthetic wrist camera sees from `joints`: table -> base (arm fit) -> gripper (FK) -> camera (X) -> pixels."""
    T_bg = cc.fk_frames(joints, CALIB["arm_fit"]["joint_offsets_deg"])["gripper_link"].copy(); T_bg[:3, 3] *= 1000.0
    T_bc = T_bg @ np.block([[R_X, t_X.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])
    T_ct = np.linalg.inv(T_bc) @ cc.table_to_base_matrix(CALIB)
    # rendered at 2x and averaged down: anti-aliased edges like a real sensor's (the aliased render puts a stray 1-2 px on an acute corner now and then)
    K2 = np.array([[2 * TRUE_FX, 0, 640.0], [0, 2 * TRUE_FX, 480.0], [0, 0, 1]])
    big = synth.render_table(layout_mm, cv2.Rodrigues(T_ct[:3, :3])[0].ravel(), T_ct[:3, 3], K=K2, dist=TRUE_DIST, size=(1280, 960), marker_px=480, angles_deg=TRUE_ANGLES)
    return cv2.resize(big, (640, 480), interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="module")
def synthetic_views(cc, tmp_path_factory):
    pytest.importorskip("scipy")
    d = tmp_path_factory.mktemp("handeye"); views = d / "wristcam"; views.mkdir()
    (d / "top_layout.json").write_text(json.dumps(LAYOUT_FILE)); (d / "top_table.json").write_text(json.dumps(CALIB))
    layout = cc.load_layout(str(d / "top_layout.json")); R_X, t_X = true_X(cc)
    for i, j in enumerate(JOINTS):
        img = render_view(cc, R_X, t_X, j, layout)
        cv2.imwrite(str(views / f"s{i + 1}.png"), img); (views / f"s{i + 1}.json").write_text(json.dumps({"name": f"s{i + 1}", "joints": {**j, "gripper": 0.0}}))
    return {"dir": d, "views": views, "layout": layout, "R_X": R_X, "t_X": t_X}


@pytest.fixture(scope="module")
def synthetic_solution(cc, synthetic_views):
    s = synthetic_views; views = cc.load_wrist_views(str(s["views"]), s["layout"])
    assert len(views) >= 5, [v["name"] for v in views]
    assert all(len(v["markers"]) >= 1 for v in views) and len({m for v in views for m in v["markers"]}) >= 3
    return views, cc.solve_handeye(views, CALIB, s["layout"])


# ---------------------------------------------------------------- building blocks
def test_fk_frames_match_fk_tip_and_name_every_link(cc):
    deg = dict(zip(cc.JOINT_ORDER, (30, -45, 60, 20, 15)))
    F = cc.fk_frames(deg)
    assert list(F) == ["base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link"]
    tip, point = cc.fk_tip(deg)
    assert np.allclose(F["gripper_frame_link"][:3, 3] * 1000, tip, atol=1e-6) and np.allclose(F["gripper_frame_link"][:3, :3] @ [0, 0, 1], point)
    # the tip frame sits 98 mm down gripper_link's -z: the fingers point along -z of gripper_link
    d = np.linalg.inv(F["gripper_link"]) @ F["gripper_frame_link"]
    assert d[2, 3] == pytest.approx(-0.0981274, abs=1e-6) and d[2, 2] == pytest.approx(-1.0, abs=1e-5)
    off = {"shoulder_lift": -3.9, "elbow_flex": -5.2, "wrist_flex": -0.7}
    assert np.allclose(cc.fk_frames(deg, off)["gripper_frame_link"][:3, 3] * 1000, cc.fk_tip({k: v + off.get(k, 0.0) for k, v in deg.items()})[0], atol=1e-6)


def test_project_points_matches_opencv(cc):
    rng = np.random.default_rng(1); P = np.column_stack([rng.uniform(-80, 80, 30), rng.uniform(-60, 60, 30), rng.uniform(100, 300, 30)])
    K = np.array([[510.0, 0, 322.0], [0, 505.0, 238.0], [0, 0, 1]]); dist = np.array([-0.3, 0.1, 0.001, -0.002, 0.02])
    ref, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    assert np.allclose(cc.project_points(P, K, dist), ref.reshape(-1, 2), atol=1e-6)
    assert np.allclose(cc.project_points(P, K, [-0.3, 0.1]), cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.array([-0.3, 0.1, 0, 0, 0]))[0].reshape(-1, 2), atol=1e-6)


def test_marker_corners_order_and_rotation(cc):
    lay = {"markers": {1: [100.0, 200.0]}, "marker_size_mm": 50.0}
    c = cc.marker_corners_table(lay, 1, 0.0)
    assert np.allclose(c, [[75, 225], [125, 225], [125, 175], [75, 175]])                       # TL, TR, BR, BL with the top toward +Y
    assert np.allclose(cc.marker_corners_table(lay, 1, 90.0), np.roll(c, 1, axis=0))            # a quarter turn CCW = the corner list shifts by one
    assert np.allclose(cc.marker_corners_table(lay, 1, 360.0), c)
    base = cc.marker_corners_base(CALIB, lay, 1, 0.0)
    for (x, y), b in zip(c, base):
        assert b[:2] == pytest.approx(cc.table_to_arm(CALIB, x, y), abs=1e-9) and b[2] == pytest.approx(3.28)
    with pytest.raises(cc.CalibError, match="no arm pose"): cc.table_to_base_matrix({"units": "in"})


def test_quaternion_and_rotvec_helpers_round_trip(cc):
    rng = np.random.default_rng(2)
    for _ in range(20):
        rv = rng.normal(size=3) * rng.uniform(0.1, 3.0); R = cc._rotvec_to_R(rv)
        assert np.allclose(R, cv2.Rodrigues(rv)[0]) and np.allclose(cc._R_to_rotvec(R), cv2.Rodrigues(R)[0].ravel())
        x, y, z, w = cc._R_to_quat_xyzw(R)
        Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(Rq, R, atol=1e-9)


def test_marker_angles_from_overhead(cc, table_scene, true_intrinsics):
    angles = {0: 180.0, 1: -90.0, 2: 12.5, 3: 0.0, 4: -7.0, 5: 90.0}
    img = synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], angles_deg=angles)
    cal = cc.solve_table(img, table_scene["layout"], true_intrinsics)
    got = cc.marker_angles_from_overhead(img, cal, table_scene["layout"])
    for m, a in angles.items(): assert math.isclose(math.cos(math.radians(got[m] - a)), 1.0, abs_tol=2e-4), (m, got[m], a)
    with pytest.raises(cc.CalibError, match="no layout markers"): cc.marker_angles_from_overhead(np.full((360, 640), 255, np.uint8), cal, table_scene["layout"])


def test_load_wrist_views_skips_views_without_a_marker(cc, synthetic_views, tmp_path):
    s = synthetic_views; (tmp_path / "x1.png").write_bytes((s["views"] / "s1.png").read_bytes()); (tmp_path / "x1.json").write_text((s["views"] / "s1.json").read_text())
    cv2.imwrite(str(tmp_path / "x2.png"), np.full((480, 640), 255, np.uint8)); (tmp_path / "x2.json").write_text(json.dumps({"joints": dict.fromkeys(cc.JOINT_ORDER, 0.0)}))
    (tmp_path / "x3.json").write_text(json.dumps({"nojoints": True}))
    views = cc.load_wrist_views(str(tmp_path), s["layout"])
    assert [v["name"] for v in views] == ["x1"] and views[0]["markers"] and all(c.shape == (4, 2) for c in views[0]["markers"].values())
    assert [v["name"] for v in cc.load_wrist_views(str(tmp_path), s["layout"], ["x1", "x2"])] == ["x1"]
    with pytest.raises(cc.CalibError, match="no capture json"): cc.load_wrist_views(str(tmp_path / "empty"), s["layout"])
    with pytest.raises(cc.CalibError, match="at least 3"): cc.solve_handeye(views, CALIB, s["layout"])


# ---------------------------------------------------------------- the synthetic solve
def test_synthetic_handeye_recovers_the_mount(cc, synthetic_views, synthetic_solution):
    s = synthetic_views; views, res = synthetic_solution
    R_true, t_true = s["R_X"], s["t_X"]; R = cc._rotvec_to_R(res["rotation"]["rvec"])
    pos_err = np.linalg.norm(np.array(res["position_mm"]) - t_true); rot_err = math.degrees(np.linalg.norm(cc._R_to_rotvec(R.T @ R_true)))
    assert res["frame"] == "gripper_link" and res["image_size"] == [640, 480] and res["views_used"] == [v["name"] for v in views]
    assert pos_err < 2.0, (res["position_mm"], t_true.tolist(), res["reproj_rms_px"])
    assert rot_err < 0.5, (res["rotation"]["optical_axis_gripper"], R_true[:, 2].tolist())
    assert abs(res["camera_matrix"][0][0] - TRUE_FX) / TRUE_FX < 0.02, res["camera_matrix"]
    assert res["hfov_deg"] == pytest.approx(math.degrees(2 * math.atan(320 / TRUE_FX)), abs=1.0)
    assert res["reproj_rms_px_corrected"] < 1.0 and res["reproj_rms_px"] < 1.5 and max(res["per_view_px"].values()) < 2.0
    assert max(res["leave_one_out_px"].values()) < 1.5, res["leave_one_out_px"]
    assert max(res["view_pose_corrections_deg"].values()) < 0.3                                            # perfect joints: nothing to correct
    for m, a in res["marker_angles_deg"].items(): assert math.isclose(math.cos(math.radians(a - TRUE_ANGLES[int(m)])), 1.0, abs_tol=1e-3), (m, a)
    q = res["rotation"]["quaternion_xyzw"]; assert math.isclose(sum(v * v for v in q), 1.0, abs_tol=1e-9)
    assert np.allclose(res["rotation"]["optical_axis_gripper"], R[:, 2]) and np.allclose(res["rotation"]["up_gripper"], -R[:, 1])
    assert abs(res["off_axis_mm"] - math.hypot(*t_true[:2])) < 2.0 and abs(res["tilt_from_gripper_axis_deg"] - math.degrees(math.acos(-R_true[2, 2]))) < 0.5


def test_synthetic_solve_with_given_marker_angles_and_rigid(cc, synthetic_views, synthetic_solution):
    """Angles measured from the overhead camera (fixed) and a rigid solve (pose_sigma 0) get the same answer on clean data."""
    s = synthetic_views; views, full = synthetic_solution
    res = cc.solve_handeye(views, CALIB, s["layout"], marker_angles=TRUE_ANGLES, pose_sigma_deg=0.0, leave_one_out=False, quick=True, init=(full["rotation"]["rvec"], full["position_mm"]))
    assert res["marker_angles_source"] == "given" and "leave_one_out_px" not in res and res["view_pose_corrections_deg"] == {v["name"]: 0.0 for v in views}
    assert np.linalg.norm(np.array(res["position_mm"]) - s["t_X"]) < 2.0 and res["reproj_rms_px"] < 1.5
    assert res["reproj_rms_px"] == pytest.approx(res["reproj_rms_px_corrected"])


def test_synthetic_solve_with_fixed_intrinsics(cc, synthetic_views, synthetic_solution):
    s = synthetic_views; views, full = synthetic_solution
    intr = {"image_size": [640, 480], "camera_matrix": [[TRUE_FX, 0, 320], [0, TRUE_FX, 240], [0, 0, 1]], "dist_coeffs": TRUE_DIST.tolist()}
    res = cc.solve_handeye(views, CALIB, s["layout"], intrinsics=intr, leave_one_out=False, quick=True)
    assert res["intrinsics_fixed"] is True and res["camera_matrix"][0][0] == TRUE_FX and res["dist_coeffs"] == TRUE_DIST.tolist()
    assert np.linalg.norm(np.array(res["position_mm"]) - s["t_X"]) < 1.0 and res["reproj_rms_px"] < 1.0
    with pytest.raises(cc.CalibError, match="intrinsics are for"): cc.solve_handeye(views, CALIB, s["layout"], intrinsics={**intr, "image_size": [1280, 960]})


def test_wrist_project_reproduces_the_detected_corners(cc, synthetic_views, synthetic_solution):
    s = synthetic_views; views, res = synthetic_solution
    dropped = {(o["view"], o["marker"], o["corner"]) for o in res["dropped_corners"]}
    for v in views:
        for mid, c in v["markers"].items():
            p = cc.wrist_project(res, CALIB, s["layout"], v["joints"], mid); keep = [k for k in range(4) if (v["name"], mid, k) not in dropped]
            assert p.shape == (4, 2) and np.all(np.isfinite(p)) and np.sqrt(np.mean(np.sum((p[keep] - c[keep]) ** 2, 1))) < 2.0, (v["name"], mid)
    for o in res["dropped_corners"]:   # a dropped corner really was a bad detection: far from where the solved pose puts it
        v = next(v for v in views if v["name"] == o["view"]); p = cc.wrist_project(res, CALIB, s["layout"], v["joints"], o["marker"])
        assert np.linalg.norm(p[o["corner"]] - v["markers"][o["marker"]][o["corner"]]) > 2.5
    img, errs = cc.annotate_wrist(views[0], res, CALIB, s["layout"])
    assert img.shape == views[0]["image"].shape and errs and max(errs.values()) < 2.0


# ---------------------------------------------------------------- CLI
def run(*args):
    return subprocess.run([PY, SCRIPT, *map(str, args)], capture_output=True, text=True, timeout=600)


def test_handeye_cli_writes_the_pose_and_the_check_image(synthetic_views):
    d = synthetic_views["dir"]
    r = run("--calib-dir", d, "handeye", "--no-loo", "--only", "s1,s2,s3,s4,s5"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); assert out["saved"] == str(d / "wrist_campose.json") and out["views_used"] == ["s1", "s2", "s3", "s4", "s5"]
    pose = json.load(open(d / "wrist_campose.json"))
    for k in ("frame", "position_mm", "rotation", "camera_matrix", "dist_coeffs", "image_size", "hfov_deg", "reproj_rms_px", "per_view_px", "views_used", "markers_used", "created", "source"): assert k in pose, k
    assert pose["source"]["views_dir"] == str(d / "wristcam") and np.linalg.norm(np.array(pose["position_mm"]) - TRUE_POS) < 8.0   # plumbing, not accuracy: 5 views, no LOO (the 7-view solve above is the accuracy test)
    r = run("--calib-dir", d, "handeye", "--check", "s3"); assert r.returncode == 0, r.stderr
    chk = json.loads(r.stdout); assert chk["saved"] == str(d / "wrist_check_s3.png") and os.path.exists(chk["saved"]) and max(chk["rigid_px"].values()) < 4.0
    im = cv2.imread(chk["saved"]); assert im is not None and im.shape[:2] == (480, 640)
    r = run("--calib-dir", d, "handeye", "--check", "nope"); assert r.returncode == 2 and "not found" in r.stderr
    r = run("--calib-dir", d, "handeye", "--views", str(d / "missing")); assert r.returncode == 2 and "no capture json" in r.stderr


# ---------------------------------------------------------------- the real captures of 9 Nov 2026
@pytest.mark.skipif(not os.path.isdir(REAL_VIEWS), reason="no real wrist captures in the repo")
def test_real_wrist_captures_solve(cc, tmp_path):
    """Five single-marker views (w4 shows no complete marker). The joints of a view are good to about a degree (the follower was
    hand-posed, the frame and the joint read are not simultaneous), which is 5-15 px at 150 mm, so the RIGID reprojection error
    is ~8 px and a held-out view predicts to ~15 px; with the per-view pose corrections the corner fit is sub-pixel. The bounds
    below are what this data supports; the mount itself (68 mm off the gripper axis, 30 deg tilt) is stable to a few mm / 1.5 deg
    under leave-one-out. More views with the follower torqued and still should bring the rigid rms toward 3 px."""
    pytest.importorskip("scipy")
    r = run("--calib-dir", tmp_path, "handeye", "--views", REAL_VIEWS, "--table", REAL_TABLE, "--layout", REAL_LAYOUT); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); res = json.load(open(tmp_path / "wrist_campose.json"))
    assert "skip w4" in r.stderr and out["views_used"] == ["w1", "w2", "w3", "w5", "w6"] and out["markers_used"] == [1, 2, 3]
    assert res["reproj_rms_px_corrected"] < 1.5 and res["reproj_rms_px"] < 12.0 and max(res["leave_one_out_px"].values()) < 20.0
    assert 55 < res["off_axis_mm"] < 80 and 20 < res["tilt_from_gripper_axis_deg"] < 40 and 50 < res["hfov_deg"] < 75
    assert res["rotation"]["optical_axis_gripper"][2] < -0.8, "the camera looks toward the fingertips (-z of gripper_link)"
    assert res["rotation"]["up_gripper"][2] < -0.3, "image-up points toward the fingertips (the jaws enter the frame from the bottom)"
    assert max(res["view_pose_corrections_deg"].values()) < 2.5 and max(s["deg"] for s in res["leave_one_out_shift"].values()) < 2.5
    # the glue orientations agree with the overhead camera: marker 1 at -90, 2 and 3 at 180 (within a few degrees)
    ang = res["marker_angles_deg"]
    assert math.isclose(math.cos(math.radians(ang["1"] + 90)), 1.0, abs_tol=0.01) and all(math.isclose(math.cos(math.radians(ang[m] - 180)), 1.0, abs_tol=0.01) for m in ("2", "3"))
    r = run("--calib-dir", tmp_path, "handeye", "--views", REAL_VIEWS, "--table", REAL_TABLE, "--layout", REAL_LAYOUT, "--check", "w3"); assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["rigid_px"]["2"] < 12.0


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "jetson", "calib", "wrist_campose.json")), reason="no committed wrist pose")
def test_committed_wrist_pose_has_the_dashboard_keys():
    p = json.load(open(os.path.join(ROOT, "jetson", "calib", "wrist_campose.json")))
    assert p["frame"] == "gripper_link" and len(p["position_mm"]) == 3 and len(p["rotation"]["quaternion_xyzw"]) == 4 and len(p["rotation"]["optical_axis_gripper"]) == 3
    assert p["image_size"] == [640, 480] and 10 < p["hfov_deg"] < 170 and p["views_used"] and p["markers_used"]
