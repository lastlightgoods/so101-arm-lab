"""The read-only calibration endpoints: /api/calib, /api/locate, /api/ik.

Files come either from a temp calib dir (``calib_dir`` points server.CALIB_DIR at tmp_path) or from the repo's real
jetson/calib/top_table.json (``real_calib``), so a recalibration cannot break these tests: every expectation about the
real file is derived from the file itself. cam_calib is imported for real (cv2 + numpy, scipy for ik) where the maths is
under test and faked where the degraded path is the point. Nothing here touches the Jetson, a camera or the arm: the
autouse ``fake`` rejects any subprocess, and ``test_ik_moves_nothing`` checks it explicitly.

Frames: mat = Fiskars 18x12 in, origin at the arm-side left corner, X along the 18 in edge, Y away from the arm; arm =
URDF base_link, X forward, Y to the arm's left. With the base at mat (9, -3) in facing +Y, mat (9, 3) is 6 in straight
ahead of the arm and mat (10, 3) is 1 in to the arm's RIGHT (arm Y = -1)."""
import json, os, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CALIB = os.path.join(ROOT, "jetson", "calib")
IN = 25.4
NO_TABLE = {"ok": False, "err": "no table calibration yet (run so101-calib table)"}
JOINTS = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}

# Synthetic table: the homography maps pixel (u, v) straight to mat inches (u in, v in), base at mat (9, -3) in, yaw 90.
SYNTH_TABLE = {
    "units": "in", "image_size": [640, 360], "marker_size_mm": 2 * IN, "markers_used": [0, 1],
    "layout_markers": {"0": [16 * IN, 10 * IN], "1": [2 * IN, 10 * IN]},
    "homography_px_to_table": [[IN, 0.0, 0.0], [0.0, IN, 0.0], [0.0, 0.0, 1.0]],
    "arm_base_mm": [9 * IN, -3 * IN], "arm_yaw_deg": 90.0, "residual_mm_max": 0.1234, "created": "2026-11-09 21:12:01",
    "arm_fit": {"arm_base_mm": [9 * IN, -3 * IN], "arm_yaw_deg": 90.0, "table_z_mm": 3.282,
                "joint_offsets_deg": {"shoulder_lift": -3.9064, "elbow_flex": -5.1761, "wrist_flex": -0.6875}, "wrist_roll_deg": 0.46,
                "fitted": "2026-11-09 21:37:42", "residual_xy_mm": [0.94, 0.71, 1.0925, 0.86],
                "points": [["table", [9.0, 3.0]], ["table", [6.0, 5.0]], ["table", [9.0, 7.0]], ["table", [12.0, 5.0]]]},
}
SYNTH_LAYOUT = {"units": "in", "marker_size": 2.0, "markers": {"0": [16, 10], "1": [2, 10]}, "arm_base": [9, -3], "arm_yaw_deg": 90}


@pytest.fixture
def calib_dir(srv, tmp_path, monkeypatch):
    """Point the server at a temp calib dir; returns write(table=..., layout=...) (None = that file is absent)."""
    d = tmp_path / "calib"; d.mkdir()
    monkeypatch.setattr(srv, "CALIB_DIR", str(d)); srv._calib_cache.clear()
    def write(table=SYNTH_TABLE, layout=SYNTH_LAYOUT, front=None):
        for name, obj in (("top_table.json", table), ("top_layout.json", layout), ("front_campose.json", front)):
            if obj is not None: (d / name).write_text(json.dumps(obj))
            elif (d / name).exists(): (d / name).unlink()
        return d
    return write


@pytest.fixture
def real_calib(srv, monkeypatch):
    """The repo copy of the Jetson's real table calibration, as the dict the server will read."""
    path = os.path.join(REAL_CALIB, "top_table.json")
    if not os.path.exists(path): pytest.skip("no real table calibration in the repo")
    monkeypatch.setattr(srv, "CALIB_DIR", REAL_CALIB); srv._calib_cache.clear()
    return json.load(open(path))


@pytest.fixture
def no_cam_calib(srv, monkeypatch):
    """cam_calib's import fails the way it does on a Mac without cv2."""
    def boom(): raise ImportError("No module named 'cv2'")
    monkeypatch.setattr(srv, "cam_calib", boom)


# ---------------------------------------------------------------- /api/calib
def test_calib_happy_path_from_temp_files(srv, http, calib_dir):
    d = calib_dir()
    c, _ = http.get_json("/api/calib")
    assert http.get("/api/calib")[1]["Cache-Control"] == "no-store"
    assert c["ok"] is True and c["units"] == "in" and c["mat"] == {"width_in": 18, "depth_in": 12} and c["image_size"] == [640, 360]
    assert c["marker_size_in"] == 2.0 and c["layout_markers_in"] == {"0": [16.0, 10.0], "1": [2.0, 10.0]}
    assert c["homography_px_to_table_mm"] == SYNTH_TABLE["homography_px_to_table"] and c["has_intrinsics"] is False
    assert c["camera"] == {"residual_mm_max": 0.12, "created": "2026-11-09 21:12:01", "markers_used": [0, 1], "pose": None, "front": None, "wrist": None,
                           "pose_live": False, "pose_at": None, "pose_moved_mm": None}   # pose tracking idle (tests/server/test_pose_api.py)
    assert c["arm"] == {"base_mm": [228.6, -76.2], "base_in": [9.0, -3.0], "yaw_deg": 90.0, "table_z_mm": 3.3,
                        "joint_offsets_deg": {"shoulder_lift": -3.91, "elbow_flex": -5.18, "wrist_flex": -0.69},
                        "wrist_roll_deg": 0.46, "fitted": "2026-11-09 21:37:42", "residual_xy_mm_max": 1.09, "points": 4}
    assert c["files"] == {"table": str(d / "top_table.json"), "layout": str(d / "top_layout.json"), "table_mtime": os.stat(d / "top_table.json").st_mtime}
    # self-healing fields (docs/mat-scene.md): idle values while the scene thread has not re-solved anything
    assert c["stored_homography"] == SYNTH_TABLE["homography_px_to_table"] and c["live"] is False and c["live_at"] is None and c["shift_px"] is None
    assert c["holding"] is False and c["calib_ok"] is None and c["calib_note"] is None and c["markers_seen"] is None and c["live_solve"] is None
    assert set(c) == {"ok", "units", "mat", "image_size", "marker_size_in", "layout_markers_in", "homography_px_to_table_mm", "has_intrinsics", "camera", "arm", "files",
                      "stored_homography", "live", "live_at", "shift_px", "holding", "calib_ok", "calib_note", "markers_seen", "live_solve"}


CAM_POSE = {"position_mm": [218.0, 162.0, 408.0], "optical_axis": [0.0, 0.0, -1.0], "up_table": [0.0, 1.0, 0.0], "tilt_deg": 0.0, "axis_hits_mm": [218.0, 162.0],
            "hfov_deg": 70.4, "image_size": [640, 360], "reproj_rms_px": 0.41, "source": "nominal C920 hfov, height from the homography scale", "created": "2026-11-10 09:00:00"}


def test_calib_camera_pose_passes_through_when_present(http, calib_dir):
    """camera_pose_est (cam_calib.py campose) -> camera.pose with the same keys plus position_in; absent or junk -> null."""
    calib_dir(table={**SYNTH_TABLE, "camera_pose_est": {**CAM_POSE, "extra": "kept"}})
    p = http.get_json("/api/calib")[0]["camera"]["pose"]
    assert p == {**CAM_POSE, "extra": "kept", "position_in": [8.583, 6.378, 16.063]}
    for junk in (None, "nominal", {"optical_axis": [0, 0, -1]}, {"position_mm": [1, 2]}, {"position_mm": ["a", "b", "c"]}):
        calib_dir(table={**SYNTH_TABLE, "camera_pose_est": junk} if junk is not None else SYNTH_TABLE)
        c = http.get_json("/api/calib")[0]; assert c["ok"] is True and c["camera"]["pose"] is None, junk


# ---------------------------------------------------------------- camera.front (front_campose.json -> pose + mat-surface homography)
# A synthetic front pose with a known K / R / t: camera centre C in the table frame, looking at the mat, image up roughly along table +Y
# (the real front camera is rolled that way). rvec is derived from R with cv2 where the maths is under test.
FRONT_C = [-8.0, 354.0, 467.0]; FRONT_AIM = [370.0, 180.0, 0.0]; FRONT_F = 466.649017432


def _front_rt():
    np = pytest.importorskip("numpy"); cv2 = pytest.importorskip("cv2")
    C, T = np.array(FRONT_C), np.array(FRONT_AIM); z = T - C; z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0]); up = up - z * (up @ z); up /= np.linalg.norm(up); yc = -up; xc = np.cross(yc, z)
    R = np.vstack([xc, yc, z]); t = -R @ C; rvec, _ = cv2.Rodrigues(R)
    K = np.array([[FRONT_F, 0.0, 320.0], [0.0, FRONT_F, 180.0], [0.0, 0.0, 1.0]])
    return R, t, rvec.ravel(), K


def front_pose(**over):
    """A front_campose.json as cam_calib.py campose --name front writes it (needs cv2 for the Rodrigues vector)."""
    np = pytest.importorskip("numpy"); R, t, rvec, K = _front_rt()
    axis = R.T @ np.array([0.0, 0.0, 1.0]); upt = R.T @ np.array([0.0, -1.0, 0.0]); s = -FRONT_C[2] / axis[2]; hit = np.array(FRONT_C) + s * axis
    p = {"position_mm": FRONT_C, "rvec": rvec.tolist(), "tvec": t.tolist(), "optical_axis": axis.tolist(), "tilt_deg": 41.703, "axis_hits_mm": hit[:2].tolist(),
         "up_table": upt.tolist(), "reproj_rms_px": 1.91, "markers_used": [0, 2, 3], "camera_matrix": K.tolist(), "hfov_deg": 68.88, "image_size": [640, 360],
         "source": "lens height 18.375 in (tape measure) -> hfov 68.88 deg", "units_note": "position_mm in the table frame (z up from the mat surface)", "created": "2026-11-09 22:58:43"}
    p.update(over); return p


def _apply(H, x, y):
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    return (H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w


def test_calib_front_pose_passes_through_with_a_derived_homography(http, calib_dir):
    """camera.front: the pose like camera.pose (plus position_in), reproj/created, and K [r1 r2 t] as table -> px with its inverse as px -> table;
    the derived table_to_px matches cv2.projectPoints on the mat surface and the pair round-trips."""
    np = pytest.importorskip("numpy"); cv2 = pytest.importorskip("cv2")
    pose = front_pose(extra="kept"); calib_dir(front=pose)
    c = http.get_json("/api/calib")[0]; fr = c["camera"]["front"]
    assert c["ok"] is True and set(fr) == {"pose", "homography_px_to_table_mm", "homography_table_to_px", "reproj_rms_px", "created",
                                           "pose_live", "pose_at", "live", "live_at", "moved_mm", "holding", "markers_seen", "live_solve"}
    assert fr["pose"] == {**pose, "position_in": [round(v / IN, 3) for v in FRONT_C]} and fr["reproj_rms_px"] == 1.91 and fr["created"] == "2026-11-09 22:58:43"
    assert fr["pose_live"] is False and fr["live"] is False and fr["pose_at"] is None and fr["live_at"] is None and fr["moved_mm"] is None and fr["live_solve"] is None
    assert fr["holding"] is False and fr["markers_seen"] is None, "pose tracking idle: the front loop has not run (tests/server/test_pose_api.py)"
    assert c["camera"]["pose"] is None and c["homography_px_to_table_mm"] == SYNTH_TABLE["homography_px_to_table"], "the top camera's fields are untouched"
    H, Hi = fr["homography_table_to_px"], fr["homography_px_to_table_mm"]
    assert len(H) == 3 and all(len(r) == 3 for r in H) and Hi[2][2] == 1.0
    for x, y in ((0, 0), (457.2, 0), (457.2, 304.8), (0, 304.8), (16 * IN, 10 * IN), (3 * IN, 4 * IN), (228.6, 152.4), (228.6, -76.2)):
        proj, _ = cv2.projectPoints(np.array([[x, y, 0.0]]), np.array(pose["rvec"]), np.array(pose["tvec"]), np.array(pose["camera_matrix"]), None)
        u, v = _apply(H, x, y)
        assert abs(u - proj[0, 0, 0]) < 1e-3 and abs(v - proj[0, 0, 1]) < 1e-3, (x, y, (u, v), proj.ravel())
        assert _apply(Hi, u, v) == pytest.approx([x, y], abs=1e-6)
    u0, v0 = _apply(H, 16 * IN, 10 * IN)
    assert 0 < u0 < 640 and 0 < v0 < 360, "marker 0 is in the front camera's frame with this pose"


def test_calib_front_is_null_when_absent_or_unusable(http, calib_dir, logs):
    calib_dir()
    assert http.get_json("/api/calib")[0]["camera"]["front"] is None
    pytest.importorskip("cv2")
    for junk in ({"position_mm": [1, 2, 3]}, {k: v for k, v in front_pose().items() if k != "rvec"}, front_pose(camera_matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                 front_pose(tvec=[0, 0]), front_pose(position_mm=["a", "b", "c"]), front_pose(rvec="nope")):
        calib_dir(front=junk)
        c = http.get_json("/api/calib")[0]; assert c["ok"] is True and c["camera"]["front"] is None, junk
    (calib_dir() / "front_campose.json").write_text("{not json")
    c = http.get_json("/api/calib")[0]
    assert c["ok"] is True and c["camera"]["front"] is None and any("api/calib front" in l for l in logs), "a corrupt front file never breaks /api/calib"


def test_calib_front_rereads_when_the_file_changes(http, calib_dir):
    pytest.importorskip("cv2"); d = calib_dir(front=front_pose())
    assert http.get_json("/api/calib")[0]["camera"]["front"]["reproj_rms_px"] == 1.91
    p = d / "front_campose.json"; p.write_text(json.dumps(front_pose(reproj_rms_px=0.5))); os.utime(p, (time.time() + 2, time.time() + 2))
    assert http.get_json("/api/calib")[0]["camera"]["front"]["reproj_rms_px"] == 0.5


def test_calib_real_front_file_if_present(http, real_calib):
    """The checked-in front pose (jetson/calib/front_campose.json): served with a homography that puts the real markers inside the frame."""
    path = os.path.join(REAL_CALIB, "front_campose.json")
    if not os.path.exists(path): pytest.skip("no front camera pose in the repo")
    est = json.load(open(path)); fr = http.get_json("/api/calib")[0]["camera"]["front"]
    assert fr["pose"]["position_mm"] == est["position_mm"] and fr["pose"]["rvec"] == est["rvec"] and fr["created"] == est["created"]
    for mid, (x, y) in real_calib["layout_markers"].items():
        u, v = _apply(fr["homography_table_to_px"], x, y)
        assert -20 <= u <= 660 and -20 <= v <= 380, f"marker {mid} projects far outside the front frame: {(u, v)}"
        assert _apply(fr["homography_px_to_table_mm"], u, v) == pytest.approx([x, y], abs=1e-6)


def test_calib_missing_table_file(http, calib_dir):
    calib_dir(table=None)
    st, hdr, body = http.get("/api/calib")
    assert st == 200 and hdr["Content-Type"] == "application/json" and json.loads(body) == NO_TABLE


def test_calib_arm_is_null_without_arm_fit(http, calib_dir):
    calib_dir(table={k: v for k, v in SYNTH_TABLE.items() if k != "arm_fit"})   # the layout's arm_base guess is still there, but it is not a fit
    c, _ = http.get_json("/api/calib")
    assert c["ok"] is True and c["arm"] is None and c["layout_markers_in"]["0"] == [16.0, 10.0]


def test_calib_markers_used_falls_back_to_the_layout_ids(http, calib_dir):
    calib_dir(table={k: v for k, v in SYNTH_TABLE.items() if k != "markers_used"})
    assert http.get_json("/api/calib")[0]["camera"]["markers_used"] == [0, 1]


def test_calib_is_pure_json_and_survives_a_missing_layout(srv, http, calib_dir, no_cam_calib):
    calib_dir(layout=None)
    c, _ = http.get_json("/api/calib")
    assert c["ok"] is True and c["marker_size_in"] == 2.0 and c["files"]["layout"].endswith("top_layout.json")


def test_calib_intrinsics_path_is_flagged(http, calib_dir):
    calib_dir(table={**SYNTH_TABLE, "rvec": [0.1, 0.2, 0.3], "tvec": [1, 2, 3], "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist_coeffs": [0, 0, 0, 0, 0]})
    assert http.get_json("/api/calib")[0]["has_intrinsics"] is True


def test_calib_rereads_the_file_when_it_changes(srv, http, calib_dir):
    d = calib_dir(); p = d / "top_table.json"
    assert http.get_json("/api/calib")[0]["camera"]["residual_mm_max"] == 0.12
    assert srv.load_calib_json(str(p)) is srv.load_calib_json(str(p)), "unchanged file is served from the mtime-keyed cache"
    p.write_text(json.dumps({**SYNTH_TABLE, "residual_mm_max": 0.5})); os.utime(p, (time.time() + 2, time.time() + 2))
    assert http.get_json("/api/calib")[0]["camera"]["residual_mm_max"] == 0.5
    assert set(srv._calib_cache) == {str(p), str(d / "top_layout.json")}


def test_calib_corrupt_table_reports_the_error(http, calib_dir, logs):
    (calib_dir() / "top_table.json").write_text("{not json")
    c, _ = http.get_json("/api/calib")
    assert c["ok"] is False and "JSONDecodeError" in c["err"] and logs and "api/calib" in logs[0]


def test_calib_real_repo_file(http, real_calib):
    """The checked-in Jetson calibration, with every expectation taken from the file itself."""
    c, _ = http.get_json("/api/calib")
    assert c["ok"] is True and c["units"] == real_calib["units"] and c["image_size"] == real_calib["image_size"]
    assert c["layout_markers_in"] == {k: [round(x / IN, 3), round(y / IN, 3)] for k, (x, y) in real_calib["layout_markers"].items()}
    assert c["marker_size_in"] == round(real_calib["marker_size_mm"] / IN, 3) and c["camera"]["markers_used"] == real_calib["markers_used"]
    assert c["has_intrinsics"] is (real_calib.get("rvec") is not None) and c["homography_px_to_table_mm"] == real_calib.get("homography_px_to_table")
    assert c["files"]["table"] == os.path.join(REAL_CALIB, "top_table.json")
    fit = real_calib.get("arm_fit")
    if fit:
        assert c["arm"]["base_in"] == [round(v / IN, 2) for v in real_calib["arm_base_mm"]] and c["arm"]["yaw_deg"] == round(real_calib["arm_yaw_deg"], 2)
        assert 0 < c["arm"]["base_in"][0] < 18 and c["arm"]["base_in"][1] < 0 and 80 < c["arm"]["yaw_deg"] < 100, "base just off the arm-side edge of the mat, facing +Y"
        assert c["arm"]["wrist_roll_deg"] == fit["wrist_roll_deg"] and c["arm"]["points"] == len(fit["points"]) and c["arm"]["residual_xy_mm_max"] == round(max(fit["residual_xy_mm"]), 2)
        assert set(c["arm"]["joint_offsets_deg"]) == {"shoulder_lift", "elbow_flex", "wrist_flex"}
    else:
        assert c["arm"] is None


# ---------------------------------------------------------------- /api/locate
def test_locate_round_trip_synthetic(http, calib_dir):
    pytest.importorskip("cv2"); calib_dir()
    r, _ = http.get_json("/api/locate?u=9&v=3")
    assert r["ok"] is True and r["pixel"] == [9.0, 3.0] and r["height_mm"] == 0.0
    assert r["table_mm"] == pytest.approx([228.6, 76.2]) and r["table_in"] == pytest.approx([9.0, 3.0])
    assert r["arm_mm"] == pytest.approx([152.4, 0.0], abs=1e-6) and r["arm_in"] == pytest.approx([6.0, 0.0], abs=1e-6), "6 in straight ahead of the base"
    r, _ = http.get_json("/api/locate?u=10&v=3&h_mm=0")
    assert r["arm_in"] == pytest.approx([6.0, -1.0], abs=1e-6), "mat +X is the arm's right (negative arm Y)"
    assert set(r) == {"ok", "pixel", "height_mm", "table_mm", "table_in", "arm_mm", "arm_in"}


def test_locate_arm_is_null_without_an_arm_pose(http, calib_dir):
    pytest.importorskip("cv2"); calib_dir(table={k: v for k, v in SYNTH_TABLE.items() if k not in ("arm_base_mm", "arm_yaw_deg", "arm_fit")})
    r, _ = http.get_json("/api/locate?u=9&v=3")
    assert r["ok"] is True and r["table_in"] == pytest.approx([9.0, 3.0]) and r["arm_mm"] is None and r["arm_in"] is None


def test_locate_real_marker_centres_land_on_the_layout(http, real_calib):
    """Each marker centre, projected into the camera with the real homography, must come back at its layout position."""
    pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    if real_calib.get("rvec") is not None or not real_calib.get("homography_px_to_table"): pytest.skip("real calibration is not homography-only")
    Hinv = np.linalg.inv(np.array(real_calib["homography_px_to_table"], float)); w, h = real_calib["image_size"]
    bx = real_calib["arm_base_mm"][0] / IN if real_calib.get("arm_base_mm") else None
    for mid, (x, y) in real_calib["layout_markers"].items():
        p = Hinv @ np.array([x, y, 1.0]); u, v = p[0] / p[2], p[1] / p[2]
        assert 0 <= u <= w and 0 <= v <= h, f"marker {mid} projects outside the frame: {(u, v)}"
        r, _ = http.get_json(f"/api/locate?u={u}&v={v}")
        assert r["ok"] is True and r["table_mm"] == pytest.approx([x, y], abs=0.01) and r["table_in"] == pytest.approx([x / IN, y / IN], abs=1e-3)
        if bx is not None:
            assert r["arm_in"][0] > 0, "every marker is in front of the arm"
            assert (r["arm_in"][1] < 0) == (x / IN > bx), f"marker {mid}: mat +X of the base must be the arm's right"


def test_locate_front_round_trip(http, calib_dir, no_cam_calib):
    """cam=front: the mat point under a front pixel through the derived homography, with the arm frame from the table file; pure Python
    (no cam_calib / cv2 needed at request time), so it works where the top path answers 'cam_calib unavailable'."""
    np = pytest.importorskip("numpy"); cv2 = pytest.importorskip("cv2"); pose = front_pose(); calib_dir(front=pose)
    for x_in, y_in in ((16, 10), (9, 3), (10, 3), (3, 4)):
        proj, _ = cv2.projectPoints(np.array([[x_in * IN, y_in * IN, 0.0]]), np.array(pose["rvec"]), np.array(pose["tvec"]), np.array(pose["camera_matrix"]), None)
        u, v = float(proj[0, 0, 0]), float(proj[0, 0, 1])
        r, _ = http.get_json(f"/api/locate?cam=front&u={u}&v={v}")
        assert r["ok"] is True and r["cam"] == "front" and r["pixel"] == [u, v] and r["height_mm"] == 0.0
        assert r["table_mm"] == pytest.approx([x_in * IN, y_in * IN], abs=0.01) and r["table_in"] == pytest.approx([x_in, y_in], abs=1e-3)
        assert set(r) == {"ok", "cam", "pixel", "height_mm", "table_mm", "table_in", "arm_mm", "arm_in"}
    assert r["arm_in"] == pytest.approx([7.0, 6.0], abs=1e-3), "mat (3, 4) with the base at (9, -3) facing +Y: 7 in ahead, 6 in to the arm's left"
    r, _ = http.get_json(f"/api/locate?cam=front&u={u}&v={v}&h_mm=0")
    assert r["ok"] is True
    # the top path is untouched by cam=top / no cam (and still needs cam_calib)
    assert http.get_json("/api/locate?u=9&v=3")[0] == {"ok": False, "err": "cam_calib unavailable: No module named 'cv2'"}
    assert http.get_json("/api/locate?cam=top&u=9&v=3")[0] == {"ok": False, "err": "cam_calib unavailable: No module named 'cv2'"}
    assert http.get_json("/api/locate?cam=side&u=9&v=3")[0] == {"ok": False, "err": "unknown cam 'side' (top or front)"}


def test_locate_front_refuses_a_height_and_needs_the_pose(http, calib_dir):
    pytest.importorskip("cv2"); calib_dir(front=front_pose())
    r, _ = http.get_json("/api/locate?cam=front&u=320&v=180&h_mm=5")
    assert r == {"ok": False, "err": "cam=front maps the mat surface only: h_mm must be 0 (the front camera has no height model)"}
    calib_dir()
    assert http.get_json("/api/locate?cam=front&u=320&v=180")[0] == {"ok": False, "err": "no front camera pose yet (run so101-calib campose --name front)"}
    r, _ = http.get_json("/api/locate?cam=front&u=&v=180")
    assert r["ok"] is False and r["err"].startswith("need u=<px>&v=<px>")
    # without an arm pose in the table file the mat answer still comes, arm null; without any table file, likewise
    calib_dir(table={k: v for k, v in SYNTH_TABLE.items() if k not in ("arm_base_mm", "arm_yaw_deg", "arm_fit")}, front=front_pose())
    r, _ = http.get_json("/api/locate?cam=front&u=320&v=180")
    assert r["ok"] is True and r["arm_mm"] is None and r["arm_in"] is None and 0 < r["table_in"][0] < 18
    calib_dir(table=None, front=front_pose())
    r, _ = http.get_json("/api/locate?cam=front&u=320&v=180")
    assert r["ok"] is True and r["arm_mm"] is None
    # a pixel above the horizon never meets the mat (this camera is rolled: image-right climbs in the world, so the horizon is off to the right)
    r, _ = http.get_json("/api/locate?cam=front&u=5000&v=180")
    assert r["ok"] is False and "does not meet the mat surface" in r["err"]


def test_locate_bad_params(http, calib_dir):
    calib_dir()
    for q in ("", "?u=1", "?v=1", "?u=abc&v=1", "?u=1&v=nan", "?u=inf&v=1", "?u=&v=1", "?u=1&v=1&h_mm=zz"):
        r, _ = http.get_json("/api/locate" + q)
        assert r["ok"] is False and r["err"].startswith("need u=<px>&v=<px>"), (q, r)


def test_locate_height_without_intrinsics_is_refused(http, calib_dir):
    pytest.importorskip("cv2"); calib_dir()
    r, _ = http.get_json("/api/locate?u=9&v=3&h_mm=5")
    assert r == {"ok": False, "err": "height needs intrinsics; this table calibration is homography-only"}


def test_locate_without_table_or_without_cam_calib(http, calib_dir, no_cam_calib):
    calib_dir(table=None)
    assert http.get_json("/api/locate?u=1&v=2")[0] == NO_TABLE
    calib_dir()
    st, hdr, body = http.get("/api/locate?u=1&v=2")
    assert st == 200 and hdr["Content-Type"] == "application/json" and json.loads(body) == {"ok": False, "err": "cam_calib unavailable: No module named 'cv2'"}


# ---------------------------------------------------------------- /api/ik
def test_ik_reachable_target_returns_joints_and_holds_the_roll(srv, http, real_calib):
    pytest.importorskip("scipy"); pytest.importorskip("cv2")
    fit = real_calib.get("arm_fit") or pytest.skip("real calibration has no arm fit")
    cc = srv.cam_calib()
    r, _ = http.get_json("/api/ik?x_in=9&y_in=5")
    assert r["ok"] is True and set(r["joints_deg"]) == JOINTS and r["pos_err_mm"] < 2 and r["tilt_deg"] < 5
    assert r["wrist_roll_deg"] == fit["wrist_roll_deg"] == r["joints_deg"]["wrist_roll"], "roll defaults to the roll the fit was jogged at"
    ax, ay = cc.table_to_arm(real_calib, 9 * IN, 5 * IN)
    assert r["target_mm"] == pytest.approx([ax, ay, fit["table_z_mm"]], abs=0.01) and r["tip_mm"] == pytest.approx(r["target_mm"], abs=2)
    off = fit["joint_offsets_deg"]; tip, _ = cc.fk_tip({k: v + off.get(k, 0.0) for k, v in r["joints_deg"].items()})
    assert list(tip) == pytest.approx(r["tip_mm"], abs=0.02), "the returned angles are LeRobot degrees: fk with the fitted offsets lands on tip_mm"
    for k, (lo, hi) in cc.JOINT_LIMITS_DEG.items(): assert lo <= r["joints_deg"][k] <= hi
    assert set(r) == {"ok", "target_mm", "joints_deg", "tip_mm", "pos_err_mm", "tilt_deg", "wrist_roll_deg"}
    r, _ = http.get_json("/api/ik?x_in=9&y_in=5&h_in=1&roll=7.5")
    assert r["ok"] is True and r["wrist_roll_deg"] == 7.5 == r["joints_deg"]["wrist_roll"] and r["target_mm"][2] == pytest.approx(fit["table_z_mm"] + IN, abs=0.01)


def test_ik_unreachable_target(http, real_calib):
    pytest.importorskip("scipy"); pytest.importorskip("cv2")
    if not real_calib.get("arm_fit"): pytest.skip("real calibration has no arm fit")
    r, _ = http.get_json("/api/ik?x_in=9&y_in=30")
    assert r["ok"] is False and "not reachable" in r["err"]


def test_ik_bad_params_no_fit_no_table(http, calib_dir, no_cam_calib):
    calib_dir()
    for q in ("", "?x_in=9", "?y_in=5", "?x_in=a&y_in=5", "?x_in=9&y_in=5&h_in=nan", "?x_in=9&y_in=5&roll=x"):
        r, _ = http.get_json("/api/ik" + q)
        assert r["ok"] is False and r["err"].startswith("need x_in=<mat x>&y_in=<mat y>"), (q, r)
    assert http.get_json("/api/ik?x_in=9&y_in=5")[0] == {"ok": False, "err": "cam_calib unavailable: No module named 'cv2'"}
    calib_dir(table={k: v for k, v in SYNTH_TABLE.items() if k != "arm_fit"})
    assert http.get_json("/api/ik?x_in=9&y_in=5")[0] == {"ok": False, "err": "no arm fit yet (run tools/jog_capture.py --fit)"}
    calib_dir(table=None)
    assert http.get_json("/api/ik?x_in=9&y_in=5")[0] == NO_TABLE


def test_ik_without_scipy_degrades(srv, http, calib_dir, monkeypatch):
    pytest.importorskip("cv2"); calib_dir()
    cc = srv.cam_calib()
    def no_scipy(*a, **k): raise cc.CalibError("ik needs scipy (pip install scipy)")
    monkeypatch.setattr(cc, "ik_tip", no_scipy)
    assert http.get_json("/api/ik?x_in=9&y_in=5")[0] == {"ok": False, "err": "ik needs scipy (pip install scipy)"}


def test_ik_moves_nothing(http, calib_dir, fake):
    """A dry-run calculator: no ssh, no subprocess at all, whatever the outcome."""
    pytest.importorskip("scipy"); pytest.importorskip("cv2"); calib_dir()
    http.get_json("/api/ik?x_in=9&y_in=5"); http.get_json("/api/ik?x_in=9&y_in=30"); http.get_json("/api/locate?u=9&v=3"); http.get_json("/api/calib")
    assert fake.calls == [] and fake.popens == []


def test_camera_wrist_passthrough(http, calib_dir):
    """wrist_campose.json (cam_calib.py handeye) -> camera.wrist passed through with position_in added; absent or junk -> null."""
    d = calib_dir()
    assert http.get_json("/api/calib")[0]["camera"]["wrist"] is None
    (d / "wrist_campose.json").write_text(json.dumps({"frame": "gripper_link", "position_mm": [1.2, 67.9, -11.3], "hfov_deg": 62.3,
                                                      "rotation": {"optical_axis_gripper": [-0.094, -0.493, -0.865]}}))
    w = http.get_json("/api/calib")[0]["camera"]["wrist"]
    assert w["frame"] == "gripper_link" and w["hfov_deg"] == 62.3 and w["rotation"]["optical_axis_gripper"][2] == -0.865
    assert w["position_in"] == pytest.approx([1.2 / 25.4, 67.9 / 25.4, -11.3 / 25.4], abs=1e-3)
    (d / "wrist_campose.json").write_text(json.dumps({"position_mm": "nope"}))
    assert http.get_json("/api/calib")[0]["camera"]["wrist"] is None
