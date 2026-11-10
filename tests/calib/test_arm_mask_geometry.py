"""cam_calib: the arm's geometry as the overhead camera sees it (fk_chain, arm_to_table, arm_points_to_camera_px,
arm_mask_px_from_joints) against the real 9 Nov calibration (fixtures/top_table_real.json: homography, arm fit with joint
offsets, camera pose pinned by the tape-measured lens height) and the follower joints jogged to mat (9, 7) in."""
import json, math, os
import numpy as np, pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "fixtures", "top_table_real.json")
JOGGED_97 = {"shoulder_pan": 0.13, "shoulder_lift": 20.88, "elbow_flex": 13.58, "wrist_flex": 60.5, "wrist_roll": 1.1}
IN = 25.4


@pytest.fixture(scope="session")
def real():
    return json.load(open(REAL))


def with_offsets(cc, real, deg):
    off = real["arm_fit"]["joint_offsets_deg"]
    return {k: float(deg.get(k, 0.0)) + float(off.get(k, 0.0)) for k in cc.JOINT_ORDER}


# ---------------------------------------------------------------- fk_chain
@pytest.mark.parametrize("deg", [{}, JOGGED_97, {"shoulder_pan": 30, "shoulder_lift": -45, "elbow_flex": 60, "wrist_flex": 20, "wrist_roll": 90}])
def test_fk_chain_ends_at_fk_tip(cc, deg):
    pts = cc.fk_chain(deg)
    assert len(pts) == len(cc.SO101_CHAIN) == 6 and all(p.shape == (3,) for p in pts)
    tip, _ = cc.fk_tip(deg)
    assert pts[-1] == pytest.approx(tip, abs=1e-9)
    assert pts[0] == pytest.approx([38.8353, 0.0, 62.4], abs=1e-3), "the first origin is the shoulder-pan axis on the base plate"
    off = (3.0, -4.0, 12.0)
    assert cc.fk_chain(deg, tip_offset_mm=off)[-1] == pytest.approx(cc.fk_tip(deg, tip_offset_mm=off)[0], abs=1e-9)
    assert np.allclose(np.array(cc.fk_chain(deg, tip_offset_mm=off)[:-1]), np.array(pts[:-1])), "a tip offset moves only the tip"


def test_fk_chain_link_lengths_are_rigid(cc):
    """Joint angles never change the distance between consecutive origins (apart from the first two, whose offset is fixed too)."""
    a = cc.fk_chain({}); b = cc.fk_chain(JOGGED_97)
    for (p, q), (r, s) in zip(zip(a, a[1:]), zip(b, b[1:])):
        assert np.linalg.norm(q - p) == pytest.approx(np.linalg.norm(s - r), abs=1e-6)


# ---------------------------------------------------------------- arm <-> table
@pytest.mark.parametrize("xy", [(0, 0), (228.6, 152.4), (457.2, 304.8), (12.3, -80.0), (400, 20)])
def test_arm_to_table_round_trips_table_to_arm(cc, real, xy):
    ax, ay = cc.table_to_arm(real, *xy)
    assert cc.arm_to_table(real, ax, ay) == pytest.approx(xy, abs=1e-9)
    assert cc.arm_to_table(real, 0.0, 0.0) == pytest.approx(real["arm_base_mm"], abs=1e-9), "the arm origin is arm_base_mm"
    assert cc.arm_to_table({"arm_yaw_deg": 90.0}, 1.0, 2.0) is None and cc.table_to_arm({}, 1.0, 2.0) is None


def test_arm_points_to_table_uses_the_fitted_table_height(cc, real):
    z0 = real["arm_fit"]["table_z_mm"]
    t = cc.arm_points_to_table([[100.0, 0.0, z0], [100.0, 0.0, z0 + 50.0]], real)
    assert t.shape == (2, 3) and t[0, 2] == pytest.approx(0.0) and t[1, 2] == pytest.approx(50.0)
    assert tuple(t[0, :2]) == pytest.approx(cc.arm_to_table(real, 100.0, 0.0))
    no_fit = {k: v for k, v in real.items() if k != "arm_fit"}
    assert cc.arm_points_to_table([[0, 0, 7.0]], no_fit)[0, 2] == pytest.approx(7.0), "no arm fit: base_link z is the table height"
    with pytest.raises(cc.CalibError, match="arm_base_mm"): cc.arm_points_to_table([[0, 0, 0]], {"image_size": [640, 360]})


# ---------------------------------------------------------------- projection through the camera pose
def test_jogged_tip_projects_where_the_homography_puts_the_mat_point(cc, real):
    """The follower jogged onto mat (9, 7) in: the tip through fk + arm fit + camera pose must land on the same pixel the
    table homography gives for (9, 7) (the two paths share nothing but the calibration)."""
    tip, _ = cc.fk_tip(with_offsets(cc, real, JOGGED_97))
    tab = cc.arm_points_to_table([tip], real)[0]
    assert abs(tab[2]) < 3.0, "the jogged tip sits on the mat"
    assert np.hypot(tab[0] - 9 * IN, tab[1] - 7 * IN) < 3.0
    px = cc.arm_points_to_camera_px([tip], real)
    assert px.shape == (1, 2)
    ref = cc.table_to_pixel(real, 9 * IN, 7 * IN)
    assert np.hypot(px[0, 0] - ref[0], px[0, 1] - ref[1]) < 6.0, (px, ref)
    chain_px = cc.arm_points_to_camera_px(cc.fk_chain(with_offsets(cc, real, JOGGED_97)), real)
    assert chain_px.shape == (6, 2) and np.all(np.isfinite(chain_px)) and chain_px[-1] == pytest.approx(px[0])


def test_parallax_moves_a_raised_point_away_from_the_camera_axis(cc, real):
    """Same table x, y, 150 mm up: the pixel moves away from where the optical axis hits the mat, by about
    h / (H - h) of the off-axis distance (H = lens height 419 mm), which the homography cannot know."""
    pose = real["camera_pose_est"]; axis = np.array(pose["axis_hits_mm"]); H = pose["position_mm"][2]
    xy = np.array([350.0, 60.0]); z0 = real["arm_fit"]["table_z_mm"]
    base_xy = cc.table_to_arm(real, *xy)
    flat = cc.arm_points_to_camera_px([[base_xy[0], base_xy[1], z0]], real)[0]
    high = cc.arm_points_to_camera_px([[base_xy[0], base_xy[1], z0 + 150.0]], real)[0]
    axis_px = np.array(cc.table_to_pixel(real, *axis))
    d_flat, d_high = np.linalg.norm(flat - axis_px), np.linalg.norm(high - axis_px)
    assert d_high > d_flat * 1.4
    assert d_high / d_flat == pytest.approx(H / (H - 150.0), rel=0.08)


def test_projection_needs_a_camera_pose(cc, real):
    no_pose = {k: v for k, v in real.items() if k != "camera_pose_est"}
    with pytest.raises(cc.CalibError, match="camera_pose_est"): cc.arm_points_to_camera_px([[100, 0, 0]], no_pose)


# ---------------------------------------------------------------- arm_mask_px_from_joints
def test_arm_mask_from_joints_with_the_camera_pose(cc, real):
    m = cc.arm_mask_px_from_joints({**JOGGED_97, "gripper": 12.0}, real)
    assert m["source"] == "pose" and m["names"] == list(cc.JOINT_ORDER) + ["tip", "fingers"]
    assert len(m["points"]) == len(m["radii_px"]) == len(m["table_mm"]) == 7 and all(r > 0 for r in m["radii_px"])
    assert all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in m["points"])
    # the tip is where the pose projection puts it, and the fingers 40 mm beyond it along the (near vertical) gripper axis
    tip, direction = cc.fk_tip(with_offsets(cc, real, JOGGED_97))
    assert m["points"][5] == pytest.approx(cc.arm_points_to_camera_px([tip], real)[0].tolist(), abs=1e-6)
    t_tip, t_fingers = np.array(m["table_mm"][5]), np.array(m["table_mm"][6])
    assert np.linalg.norm(t_fingers - t_tip) == pytest.approx(cc.ARM_MASK_FINGERS_MM, abs=1e-6) and t_fingers[2] < t_tip[2]
    # radii: mm x shadow x fx / depth. The tip is on the mat (depth = lens height); the elbow is ~210 mm up, so it looks bigger
    pose = real["camera_pose_est"]; fx = pose["camera_matrix"][0][0]; Hc = pose["position_mm"][2]
    assert m["radii_px"][5] == pytest.approx(cc.ARM_MASK_RADII_MM["tip"] * cc.ARM_MASK_SHADOW * fx / Hc, rel=0.02)
    z_elbow = m["table_mm"][2][2]; assert 150 < z_elbow < 260
    assert m["radii_px"][2] == pytest.approx(cc.ARM_MASK_RADII_MM["elbow_flex"] * cc.ARM_MASK_SHADOW * fx / (Hc - z_elbow), rel=0.03)
    assert m["radii_px"][2] / m["radii_px"][5] > 1.6
    # the joint offsets were applied: without them the tip lands elsewhere
    raw = cc.arm_mask_px_from_joints(JOGGED_97, {**real, "arm_fit": {**real["arm_fit"], "joint_offsets_deg": {}}})
    assert np.hypot(*(np.array(raw["points"][5]) - m["points"][5])) > 3
    # per-point radii can be overridden
    m2 = cc.arm_mask_px_from_joints(JOGGED_97, real, radii_mm={"tip": 60.0})
    assert m2["radii_px"][5] == pytest.approx(2 * m["radii_px"][5]) and m2["radii_px"][4] == pytest.approx(m["radii_px"][4])


def test_arm_mask_from_joints_falls_back_to_the_homography(cc, real):
    """No camera_pose_est: points at mat level through the homography, radii x 1.6 to make up for the ignored parallax."""
    no_pose = {k: v for k, v in real.items() if k != "camera_pose_est"}
    m = cc.arm_mask_px_from_joints(JOGGED_97, no_pose)
    assert m["source"] == "homography" and m["names"][-2:] == ["tip", "fingers"] and len(m["points"]) == 7
    tip, _ = cc.fk_tip(with_offsets(cc, real, JOGGED_97)); tx, ty, _ = cc.arm_points_to_table([tip], real)[0]
    assert m["points"][5] == pytest.approx(cc.table_to_pixel(no_pose, tx, ty), abs=1e-6)
    posed = cc.arm_mask_px_from_joints(JOGGED_97, real)
    assert np.hypot(*(np.array(m["points"][5]) - posed["points"][5])) < 2.0, "on the mat the two paths agree"
    assert m["radii_px"][5] == pytest.approx(posed["radii_px"][5] * cc.ARM_MASK_FLAT_INFLATE, rel=0.03)
    assert m["radii_px"][2] == pytest.approx(cc.ARM_MASK_RADII_MM["elbow_flex"] * cc.ARM_MASK_SHADOW * cc.ARM_MASK_FLAT_INFLATE * 1.11, rel=0.06), "no height scaling at all"
    with pytest.raises(cc.CalibError): cc.arm_mask_px_from_joints(JOGGED_97, {"homography_px_to_table": real["homography_px_to_table"]})
    with pytest.raises(cc.CalibError): cc.arm_mask_px_from_joints(JOGGED_97, {"arm_base_mm": [0, 0], "arm_yaw_deg": 0})
