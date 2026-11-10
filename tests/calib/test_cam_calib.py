"""Tests for jetson/cam_calib.py against a synthetic camera with known intrinsics, distortion and pose."""
import http.server, json, math, os, subprocess, sys, threading
import cv2, numpy as np, pytest
import synth
from synth import SCRIPT

PY = sys.executable


def run(*args, cwd=None):
    return subprocess.run([PY, SCRIPT, *map(str, args)], capture_output=True, text=True, cwd=cwd, timeout=120)


# ---------------------------------------------------------------- checkerboard / intrinsics
def test_synthetic_board_is_detectable(cc, board_images):
    for name, im in board_images:
        c = cc.find_board(im, 9, 6)
        assert c is not None and c.shape == (54, 1, 2), name


def test_board_object_points_order_matches_opencv(cc):
    p = cc.board_object_points(9, 6, 25.0)
    assert p.shape == (54, 3) and p.dtype == np.float32
    assert tuple(p[0]) == (0, 0, 0) and tuple(p[1]) == (25, 0, 0) and tuple(p[9]) == (0, 25, 0)   # column fastest, as findChessboardCorners orders them


def test_calibrate_recovers_true_camera(cc, board_images):
    r = cc.calibrate(board_images, 9, 6, 25.0)
    K = np.array(r["camera_matrix"]); d = np.array(r["dist_coeffs"])
    assert r["n_images"] == 14 and r["image_size"] == [synth.W, synth.H]
    assert r["rms_px"] < 0.5, r["rms_px"]
    assert abs(K[0, 0] - synth.K_TRUE[0, 0]) / synth.K_TRUE[0, 0] < 0.02
    assert abs(K[1, 1] - synth.K_TRUE[1, 1]) / synth.K_TRUE[1, 1] < 0.02
    assert abs(K[0, 2] - synth.K_TRUE[0, 2]) < 6 and abs(K[1, 2] - synth.K_TRUE[1, 2]) < 6
    assert d[0] < -0.03, "barrel distortion sign lost"
    # k1/k2 trade off against each other, so judge the lens model by what it does to pixels, not by coefficient values
    xs, ys = np.meshgrid(np.linspace(80, synth.W - 80, 13), np.linspace(60, synth.H - 60, 9)); grid = np.stack([xs, ys], -1).reshape(-1, 1, 2)   # the region the boards covered
    want = cv2.undistortPoints(grid, synth.K_TRUE, synth.DIST_TRUE, P=synth.K_TRUE); got = cv2.undistortPoints(grid, K, d, P=K)
    assert np.max(np.linalg.norm(want - got, axis=2)) < 1.5
    assert set(r["per_image_rms_px"]) == {n for n, _ in board_images} and max(r["per_image_rms_px"].values()) < 1.0
    assert r["board"] == {"cols": 9, "rows": 6, "square_mm": 25.0}


def test_calibrate_rejects_too_few_boards(cc, board_images):
    with pytest.raises(cc.CalibError, match=r"only 3 usable image\(s\) out of 3; need at least 8"):
        cc.calibrate(board_images[:3])


def test_calibrate_skips_images_without_a_board_and_counts_them(cc, board_images):
    blank = np.full((synth.H, synth.W), 255, np.uint8)
    imgs = board_images[:10] + [("blank_a.png", blank), ("blank_b.png", blank)]
    r = cc.calibrate(imgs)
    assert r["n_images"] == 10 and "blank_a.png" not in r["per_image_rms_px"]
    with pytest.raises(cc.CalibError, match="only 5 usable image\\(s\\) out of 7"):
        cc.calibrate(board_images[:5] + [("blank_a.png", blank), ("blank_b.png", blank)])


def test_calibrate_rejects_mixed_resolutions(cc, board_images):
    small = cv2.resize(board_images[0][1], (320, 180))
    with pytest.raises(cc.CalibError, match="320x180 but the first image is 640x360"):
        cc.calibrate(board_images[:8] + [("small.png", small)])


def test_calibrate_accepts_colour_images(cc, board_images):
    colour = [(n, cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)) for n, im in board_images[:9]]
    assert cc.calibrate(colour)["n_images"] == 9


# ---------------------------------------------------------------- layout validation
def write_layout(tmp_path, lay, name="layout.json"):
    p = tmp_path / name; p.write_text(json.dumps({**lay, "markers": {str(k): v for k, v in lay["markers"].items()}})); return str(p)


def test_layout_loads_and_normalises_ids(cc, tmp_path):
    lay = cc.load_layout(write_layout(tmp_path, synth.default_layout()))
    assert set(lay["markers"]) == {0, 1, 2, 3, 4, 5} and lay["markers"][2] == [360.0, 260.0]
    assert lay["arm_base_mm"] == [200.0, 330.0] and lay["arm_yaw_deg"] == -90.0 and lay["dictionary"] == "DICT_4X4_50"


@pytest.mark.parametrize("mutate,msg", [
    (lambda l: [l["markers"].pop(k) for k in (5, 4, 3, 2)], "at least 3 markers"),
    (lambda l: l.update(markers={0: [0, 0], 1: [100, 0], 2: [200, 0], 3: [300, 0]}), "collinear"),
    (lambda l: l.update(markers={**l["markers"], "x": [1, 2]}), "not an integer"),
    (lambda l: l.update(markers={**l["markers"], 7: [1, 2, 3]}), r"marker 7 must be \[x, y\]"),
    (lambda l: l.update(dictionary="DICT_NOPE"), "unknown ArUco dictionary"),
    (lambda l: l.update(marker_size_mm=0), "marker_size must be a positive number"),
    (lambda l: l.update(arm_base_mm=[1]), r"arm_base_mm must be \[x, y\]"),
])
def test_layout_rejects_bad_files(cc, tmp_path, mutate, msg):
    lay = synth.default_layout(); mutate(lay)
    with pytest.raises(cc.CalibError, match=msg): cc.load_layout(write_layout(tmp_path, lay))


def test_layout_unreadable_file(cc, tmp_path):
    with pytest.raises(cc.CalibError, match="cannot read layout"): cc.load_layout(str(tmp_path / "missing.json"))
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(cc.CalibError, match="cannot read layout"): cc.load_layout(str(tmp_path / "bad.json"))


# ---------------------------------------------------------------- ArUco table
def test_detect_markers_finds_every_layout_marker(cc, table_scene):
    seen = cc.detect_markers(table_scene["img"])
    assert set(seen) == {0, 1, 2, 3, 4, 5} and all(c.shape == (4, 2) for c in seen.values())


def test_solve_table_with_intrinsics_recovers_pose(cc, table_scene, true_intrinsics):
    res = cc.solve_table(table_scene["img"], table_scene["layout"], true_intrinsics)
    assert res["markers_used"] == [0, 1, 2, 3, 4, 5]
    assert np.allclose(res["camera_position_mm"], table_scene["camera_position"], atol=6), (res["camera_position_mm"], table_scene["camera_position"])
    assert res["residual_mm_max"] < 1.5 and res["reproj_rms_px"] < 1.0   # 24 individual corners carry more pixel noise than 6 averaged centres
    assert res["arm_base_mm"] == [200.0, 330.0] and res["arm_yaw_deg"] == -90.0


def test_pixel_to_table_round_trip_at_height(cc, table_scene, true_intrinsics):
    res = cc.solve_table(table_scene["img"], table_scene["layout"], true_intrinsics)
    rng = np.random.default_rng(1)
    for _ in range(30):
        x, y, h = rng.uniform(20, 380), rng.uniform(20, 280), rng.choice([0.0, 11.0, 30.0, 55.0])
        proj, _ = cv2.projectPoints(np.array([[x, y, h]]), table_scene["rvec"], table_scene["tvec"], synth.K_TRUE, synth.DIST_TRUE)   # where the true camera sees it
        u, v = proj.ravel()
        X, Y = cc.pixel_to_table(res, float(u), float(v), h)
        assert math.hypot(X - x, Y - y) < 1.5, (x, y, h, X, Y)
        u2, v2 = cc.table_to_pixel(res, X, Y, h)
        assert math.hypot(u2 - u, v2 - v) < 0.3


def test_height_matters(cc, table_scene, true_intrinsics):
    """A tall object's top lands on a different table XY than the same pixel at table level; ignoring height is the classic mistake."""
    res = cc.solve_table(table_scene["img"], table_scene["layout"], true_intrinsics)
    u, v = 100.0, 60.0
    flat = cc.pixel_to_table(res, u, v, 0.0); tall = cc.pixel_to_table(res, u, v, 60.0)
    assert math.hypot(flat[0] - tall[0], flat[1] - tall[1]) > 5


def test_solve_table_homography_only(cc, table_scene):
    img = synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], dist=None)
    res = cc.solve_table(img, table_scene["layout"], None)
    assert "homography_px_to_table" in res and "rvec" not in res and res["residual_mm_max"] < 1.0 and res["points_used"] == 24
    u, v = cc.table_to_pixel(res, 123.0, 210.0); X, Y = cc.pixel_to_table(res, u, v)
    assert math.hypot(X - 123, Y - 210) < 0.5
    with pytest.raises(cc.CalibError, match="height needs intrinsics"): cc.pixel_to_table(res, u, v, 10.0)


def test_solve_table_needs_three_layout_markers_in_view(cc, table_scene, true_intrinsics):
    lay = dict(table_scene["layout"]); lay["markers"] = {0: [40.0, 40.0], 1: [360.0, 40.0], 40: [0.0, 500.0], 41: [500.0, 500.0]}
    with pytest.raises(cc.CalibError, match=r"need 3\+ layout markers in view, saw 2: \[0, 1\]"):
        cc.solve_table(table_scene["img"], lay, true_intrinsics)


def test_three_markers_are_enough_because_corners_are_used(cc, table_scene, true_intrinsics):
    """One corner marker clipped off the frame (the real 8 Nov situation): the other three still give a full solve."""
    lay = dict(table_scene["layout"]); lay["markers"] = {0: [40.0, 40.0], 1: [360.0, 40.0], 2: [360.0, 260.0]}
    for intr in (true_intrinsics, None):
        img = table_scene["img"] if intr else synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], dist=None)
        res = cc.solve_table(img, lay, intr)
        assert res["markers_used"] == [0, 1, 2] and res["points_used"] == 12 and res["residual_mm_max"] < 1.5
        assert ("warning" in res) == (intr is None)                                  # homography from 3 small squares has a weak perspective term
        for x, y, tol in ((100.0, 120.0, 1.5), (300.0, 200.0, 1.5), (200.0, 150.0, 1.5), (40.0, 260.0, 1.5 if intr else 6.0)):   # last = where the missing marker sat, outside the triangle
            u, v = cv2.projectPoints(np.array([[x, y, 0.0]]), table_scene["rvec"], table_scene["tvec"], synth.K_TRUE, synth.DIST_TRUE if intr else None)[0].ravel()
            X, Y = cc.pixel_to_table(res, float(u), float(v)); assert math.hypot(X - x, Y - y) < tol, (x, y, X, Y, intr is not None)


def test_sloppily_rotated_markers_do_not_bias_the_solve(cc, table_scene, true_intrinsics):
    angles = {0: 7.0, 1: -12.0, 2: 31.0, 3: -4.0, 4: 90.0, 5: 180.0}
    img = synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], angles_deg=angles)
    res = cc.solve_table(img, table_scene["layout"], true_intrinsics)
    assert res["points_used"] == 24 and res["residual_mm_max"] < 1.5
    assert np.allclose(res["camera_position_mm"], table_scene["camera_position"], atol=6)


def test_solve_table_rejects_intrinsics_for_other_resolution(cc, table_scene, true_intrinsics):
    with pytest.raises(cc.CalibError, match=r"intrinsics are for \[1280, 720\]"):
        cc.solve_table(table_scene["img"], table_scene["layout"], {**true_intrinsics, "image_size": [1280, 720]})


def test_table_to_arm_frame(cc):
    calib = {"arm_base_mm": [200.0, 330.0], "arm_yaw_deg": -90.0}
    fwd = cc.table_to_arm(calib, 200.0, 230.0)          # 100 mm toward the operator = straight ahead of the arm
    assert abs(fwd[0] - 100) < 1e-9 and abs(fwd[1]) < 1e-9
    left = cc.table_to_arm(calib, 250.0, 330.0)          # +X on the table is the arm's left when it faces -Y
    assert abs(left[0]) < 1e-9 and abs(left[1] - 50) < 1e-9
    assert cc.table_to_arm({"arm_base_mm": None}, 1, 2) is None


def test_point_behind_camera_is_refused(cc, table_scene, true_intrinsics):
    res = cc.solve_table(table_scene["img"], table_scene["layout"], true_intrinsics)
    with pytest.raises(cc.CalibError, match="behind the camera"): cc.pixel_to_table(res, 320.0, 180.0, 5000.0)


def test_annotate_returns_colour_frame(cc, table_scene, true_intrinsics):
    res = cc.solve_table(table_scene["img"], table_scene["layout"], true_intrinsics)
    out = cc.annotate(cv2.cvtColor(table_scene["img"], cv2.COLOR_GRAY2BGR), res)
    assert out.shape == (synth.H, synth.W, 3) and not np.array_equal(out, cv2.cvtColor(table_scene["img"], cv2.COLOR_GRAY2BGR))


# ---------------------------------------------------------------- printables
def test_board_image_geometry(cc):
    img = cc.board_image(9, 6, 25.0, margin_mm=15.0)
    s, m = cc.mm_to_px(25.0), cc.mm_to_px(15.0)
    assert img.shape[1] == 10 * s + 2 * m and img.dtype == np.uint8
    assert img[m + s // 2, m + s // 2] == 0 and img[m + s // 2, m + s + s // 2] == 255      # alternating squares
    assert cc.find_board(cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), 9, 6) is not None


def test_markers_sheet_reads_back(cc):
    img = cc.markers_image([0, 3, 7, 12, 21], size_mm=40.0)
    seen = cc.detect_markers(img)
    assert set(seen) == {0, 3, 7, 12, 21}
    side = np.linalg.norm(seen[0][0] - seen[0][1])
    assert abs(side - cc.mm_to_px(40.0)) < 3           # printed at 300 dpi the marker is 40 mm


def test_parse_ids(cc):
    assert cc.parse_ids("0-3,7, 9") == [0, 1, 2, 3, 7, 9]
    with pytest.raises(cc.CalibError): cc.parse_ids(" , ")


def test_mm_to_px(cc):
    assert cc.mm_to_px(25.4) == 300 and cc.mm_to_px(0) == 0


# ---------------------------------------------------------------- frame sources and the busy flag
def test_folder_source_streams_then_ends(cc, boards_dir):
    s = cc.Source(str(boards_dir)); n = 0
    while s.read() is not None: n += 1
    assert n == 14 and s.kind == "folder"


def test_empty_folder_source_is_an_error(cc, tmp_path):
    with pytest.raises(cc.CalibError, match="no images in"): cc.Source(str(tmp_path))


def test_url_source_fetches_a_fresh_jpeg_each_time(cc, board_images):
    ok, jpg = cv2.imencode(".jpg", board_images[0][1]); hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(jpg.tobytes())
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = cc.Source(f"http://127.0.0.1:{srv.server_address[1]}/cam/top.jpg")
        a, b = s.read(), s.read()
        assert s.kind == "url" and a.shape == (synth.H, synth.W, 3) and b is not None
        assert len(hits) == 2 and hits[0] != hits[1] and all("?t=" in h for h in hits)     # cache-buster per request, like the dashboard does
        s.spec = f"http://127.0.0.1:{srv.server_address[1]}/cam/top.jpg?x=1"; assert "&t=" in (s.read() is not None and hits[-1])
    finally: srv.shutdown()


def test_url_source_returns_none_when_unreachable(cc):
    assert cc.Source("http://127.0.0.1:9/nothing.jpg").read() is None


def test_device_source_failure_is_a_calib_error(cc):
    with pytest.raises(cc.CalibError, match="cannot open camera"): cc.Source("/dev/does-not-exist-video99", open_timeout=0.6)


def test_device_source_retries_while_the_camera_is_held(cc, monkeypatch):
    """The dashboard's ffmpeg re-grabs the camera moments after the busy flag clears; the second open must wait it out."""
    attempts = []

    class FakeCap:
        def __init__(self, spec): attempts.append(spec); self.ok = len(attempts) >= 3
        def isOpened(self): return self.ok
        def release(self): pass
        def set(self, *a): pass
    monkeypatch.setattr(cc.cv2, "VideoCapture", FakeCap); monkeypatch.setattr(cc.time, "sleep", lambda s: None)
    src = cc.Source("/dev/video0", open_timeout=5.0)
    assert src.kind == "device" and len(attempts) == 3 and attempts == ["/dev/video0"] * 3


def test_busy_flag_lifecycle(cc, tmp_path):
    flag = tmp_path / ".cams_busy"
    with cc.BusyFlag(str(flag), enabled=True, settle=0):
        assert flag.exists()
    assert not flag.exists()
    flag.touch()
    with cc.BusyFlag(str(flag), enabled=True, settle=0): pass
    assert flag.exists(), "a flag someone else set must be left alone"
    flag.unlink()
    with cc.BusyFlag(str(flag), enabled=False, settle=0): assert not flag.exists()


def test_read_one_from_folder(cc, boards_dir, tmp_path):
    im = cc.read_one(str(boards_dir), 640, 360, str(tmp_path / "flag"))
    assert im.shape == (synth.H, synth.W, 3) and not (tmp_path / "flag").exists()


# ---------------------------------------------------------------- CLI end to end
def test_cli_board_and_markers(tmp_path):
    r = run("--calib-dir", tmp_path, "board"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); assert os.path.exists(out["saved"]) and out["square_mm"] == 25.0
    r = run("--calib-dir", tmp_path, "markers", "--ids", "0-5", "--size-mm", "35"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); assert out["ids"] == [0, 1, 2, 3, 4, 5] and cv2.imread(out["saved"]) is not None


def test_cli_capture_from_folder_keeps_only_new_views(tmp_path, boards_dir, board_images):
    r = run("--calib-dir", tmp_path, "capture", "--source", boards_dir, "--count", "5", "--interval", "0", "--busy-flag", tmp_path / "flag")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); assert out["saved"] == 5 and len(os.listdir(out["dir"])) == 5 and not (tmp_path / "flag").exists()
    same = tmp_path / "same"; same.mkdir()
    for i in range(4): cv2.imwrite(str(same / f"s{i}.png"), board_images[0][1])
    r = run("--calib-dir", tmp_path, "capture", "--source", same, "--count", "5", "--interval", "0", "--out", tmp_path / "dup")
    assert r.returncode == 0 and json.loads(r.stdout)["saved"] == 1, r.stderr          # the same view four times is one shot
    r = run("--calib-dir", tmp_path, "capture", "--source", same, "--count", "2", "--raw", "--out", tmp_path / "raw")
    assert json.loads(r.stdout)["saved"] == 2
    blank = tmp_path / "blank"; blank.mkdir(); cv2.imwrite(str(blank / "b.png"), np.full((360, 640), 255, np.uint8))
    r = run("--calib-dir", tmp_path, "capture", "--source", blank, "--count", "2", "--out", tmp_path / "none")
    assert r.returncode == 1 and json.loads(r.stdout)["saved"] == 0


def test_cli_full_pipeline(tmp_path, boards_dir, table_scene):
    r = run("--calib-dir", tmp_path, "intrinsics", "--images", boards_dir); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout); intr = json.load(open(out["saved"]))
    assert out["saved"] == str(tmp_path / "top_intrinsics.json") and out["n_images"] == 14 and out["rms_px"] < 0.5 and abs(out["fx"] - 520) < 11
    scene = tmp_path / "scene"; scene.mkdir(); cv2.imwrite(str(scene / "table.png"), table_scene["img"])
    write_layout(tmp_path, table_scene["layout"], "top_layout.json")
    r = run("--calib-dir", tmp_path, "table", "--source", scene, "--busy-flag", tmp_path / "flag"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["markers_used"] == [0, 1, 2, 3, 4, 5] and out["residual_mm_max"] < 1.5 and os.path.exists(out["check_image"])
    assert np.allclose(out["camera_position_mm"], table_scene["camera_position"], atol=12)   # recovered (not true) intrinsics: a 1 % focal error is ~6 mm of height at 600 mm
    tab = json.load(open(out["saved"])); assert tab["camera_matrix"] == intr["camera_matrix"]
    x, y, h = 150.0, 200.0, 11.0
    proj, _ = cv2.projectPoints(np.array([[x, y, h]]), table_scene["rvec"], table_scene["tvec"], synth.K_TRUE, synth.DIST_TRUE); u, v = proj.ravel()
    r = run("--calib-dir", tmp_path, "locate", f"{u:.2f}", f"{v:.2f}", "--height", h); assert r.returncode == 0, r.stderr
    loc = json.loads(r.stdout)
    assert math.hypot(loc["table_mm"][0] - x, loc["table_mm"][1] - y) < 1.5
    assert abs(loc["arm_mm"][0] - 130) < 1.5 and abs(loc["arm_mm"][1] + 50) < 1.5      # base (200,330) facing -Y: 130 mm ahead, 50 mm to its right
    r = run("--calib-dir", tmp_path, "check", "--source", scene); assert r.returncode == 0, r.stderr
    chk = json.loads(r.stdout); assert chk["markers_seen"] == [0, 1, 2, 3, 4, 5] and max(chk["error_mm"].values()) < 1.5 and os.path.exists(chk["saved"])


def test_cli_table_without_intrinsics_warns_and_still_works(tmp_path, table_scene):
    scene = tmp_path / "scene"; scene.mkdir()
    cv2.imwrite(str(scene / "table.png"), synth.render_table(table_scene["layout"], table_scene["rvec"], table_scene["tvec"], dist=None))
    write_layout(tmp_path, table_scene["layout"], "top_layout.json")
    r = run("--calib-dir", tmp_path, "table", "--source", scene); assert r.returncode == 0, r.stderr
    assert "no intrinsics" in r.stderr and json.loads(r.stdout)["camera_position_mm"] is None
    r = run("--calib-dir", tmp_path, "locate", "100", "100", "--height", "5"); assert r.returncode == 2 and "height needs intrinsics" in r.stderr


def test_cli_errors_are_exit_2_with_a_message(tmp_path):
    r = run("--calib-dir", tmp_path, "intrinsics", "--images", tmp_path); assert r.returncode == 2 and "no images in" in r.stderr
    r = run("--calib-dir", tmp_path, "locate", "1", "2"); assert r.returncode == 2 and "cannot read table calibration" in r.stderr
    r = run("--calib-dir", tmp_path, "table", "--source", tmp_path); assert r.returncode == 2 and "cannot read layout" in r.stderr
    r = run("--calib-dir", tmp_path, "markers", "--ids", "0-3", "--dictionary", "DICT_NOPE"); assert r.returncode == 2 and "unknown ArUco dictionary" in r.stderr


def test_cli_intrinsics_too_few_images(tmp_path, board_images):
    few = tmp_path / "few"; few.mkdir()
    for n, im in board_images[:4]: cv2.imwrite(str(few / n), im)
    r = run("--calib-dir", tmp_path, "intrinsics", "--images", few)
    assert r.returncode == 2 and "only 4 usable image(s) out of 4; need at least 8" in r.stderr
