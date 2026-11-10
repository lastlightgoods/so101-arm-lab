"""Live camera poses: the server re-solves each camera's position and orientation from the ArUco markers with that lens's FIXED
field of view (hfov_deg from `campose --lens-height`) so the 3D stage follows the cameras when their mounts move.

Overhead camera: the scene loop (tests/server/test_scene_api.py drives it the same way) with obj.jpg / calib_a.json plus a stored
camera_pose_est solved from the unmoved frame itself; a bumped camera is the frame shifted with cv2.warpAffine. Front camera: there is
no real front frame in the fixtures, so one is synthesised by rendering the four layout markers of the real top_table.json through the
real front pose (jetson/calib/front_campose.json) with cv2.projectPoints + warpPerspective of generated marker images, like
tests/calib/synth.py does for the top; the solved pose must come back within 5 mm / 1 deg of the pose it was rendered from. A moved
front camera is rendered from a translated camera centre. No camera, no network, no Jetson: the autouse ``fake`` rejects any subprocess."""
import importlib.util, json, math, os, time
import pytest

cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(ROOT, "tests", "scene", "fixtures")
REAL_CALIB = os.path.join(ROOT, "jetson", "calib")
CAM_CALIB = os.path.join(ROOT, "jetson", "cam_calib.py")
TOP_HFOV = 69.16     # the overhead C920, pinned by its tape-measured lens height (16.5 in) on 9 Nov 2026
ARM = {"arm_base_mm": [228.6, -76.2], "arm_yaw_deg": 90.0,
       "arm_fit": {"arm_base_mm": [228.6, -76.2], "arm_yaw_deg": 90.0, "table_z_mm": 3.282, "joint_offsets_deg": {"shoulder_lift": -3.9064, "elbow_flex": -5.1761, "wrist_flex": -0.6875},
                   "wrist_roll_deg": 0.46, "fitted": "2026-11-09 21:37:42", "residual_xy_mm": [0.94, 0.71, 1.0925, 0.86],
                   "points": [["table", [9.0, 3.0]], ["table", [6.0, 5.0]], ["table", [9.0, 7.0]], ["table", [12.0, 5.0]]]}}


@pytest.fixture(scope="session")
def cc():
    spec = importlib.util.spec_from_file_location("pose_cam_calib", CAM_CALIB); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def img(name):
    im = cv2.imread(os.path.join(FIX, name), cv2.IMREAD_COLOR); assert im is not None, name; return im


def jpg(im):
    ok, b = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85]); assert ok; return b.tobytes()


def via_jpg(im):
    """The frame as the server sees it: decoded from the JPEG the camera loop keeps in memory."""
    return cv2.imdecode(np.frombuffer(jpg(im), np.uint8), cv2.IMREAD_COLOR)


def shifted(im, dx, dy):
    """The same view after the camera stand was bumped: every pixel moves by (dx, dy)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]]); return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def layout_of(table):
    return {"dictionary": table["dictionary"], "marker_size_mm": table["marker_size_mm"], "markers": {int(k): list(v) for k, v in table["layout_markers"].items()}}


def delta(a, b):
    """(mm between two poses' camera centres, degrees between their optical axes)."""
    mm = float(np.linalg.norm(np.array(a["position_mm"]) - np.array(b["position_mm"])))
    deg = math.degrees(math.acos(np.clip(np.dot(a["optical_axis"], b["optical_axis"]) / (np.linalg.norm(a["optical_axis"]) * np.linalg.norm(b["optical_axis"])), -1, 1)))
    return mm, deg


def solve(cc, frame, table, hfov):
    """What the server should come up with: cam_calib's pose from this frame with the fixed FOV."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cc.camera_pose_from_markers(gray, layout_of(table), cc.nominal_K(gray.shape[1], gray.shape[0], hfov))


# ---------------------------------------------------------------- the overhead camera (scene loop)
class TopLab:
    def __init__(self, srv, cc, d, hfov=TOP_HFOV):
        self.srv, self.dir = srv, d
        self.calib_a = json.load(open(os.path.join(FIX, "calib_a.json")))
        self.frames = {n: img(n + ".jpg") for n in ("bg", "obj")}
        est = solve(cc, self.frames["obj"], self.calib_a, hfov) if hfov else {}
        self.stored = {**est, "hfov_deg": hfov, "image_size": [640, 360], "source": "lens height 16.5 in (tape measure) -> hfov 69.16 deg",
                       "units_note": "position_mm in the table frame (z up from the mat surface)", "created": "2026-11-09 23:06:12"}
        if hfov is None: self.stored = {k: v for k, v in self.stored.items() if k != "hfov_deg"}
        self.table = {**self.calib_a, **ARM, "camera_pose_est": self.stored}
        self.path = os.path.join(str(d), "top_table.json"); self.write_table(self.table)

    def write_table(self, table):
        with open(self.path, "w") as f: json.dump(table, f)
        os.utime(self.path, (time.time() + 2, time.time() + 2)); self.srv._calib_cache.clear()

    def read_table(self): return json.load(open(self.path))

    def frame(self, im, t=None): self.srv.cam_frames["top"] = (jpg(im), time.time() if t is None else t)

    def learn_bg(self, http):
        self.frame(self.frames["bg"]); r = http.post_json("/api/scene/background"); assert r["ok"] is True, r


@pytest.fixture
def top(srv, cc, tmp_path, monkeypatch):
    """Temp CALIB_DIR with calib A + arm fit + a stored camera pose solved from the unmoved obj.jpg (hfov 69.16), fresh scene state."""
    d = tmp_path / "calib"; d.mkdir()
    monkeypatch.setattr(srv, "CALIB_DIR", str(d)); monkeypatch.setattr(srv, "SCENE_BG_SPACING_S", 0)
    srv._calib_cache.clear(); srv.scene_reset(); srv.front_reset()
    yield TopLab(srv, cc, d)
    srv.scene_reset(); srv.front_reset()


def test_top_pose_follows_a_bumped_camera_and_travels_with_the_write_back(srv, http, top, cc, events):
    top.learn_bg(http); T = time.time()
    c0 = http.get_json("/api/calib")[0]["camera"]
    assert c0["pose_live"] is False and c0["pose_at"] is None and c0["pose_moved_mm"] is None and c0["pose"]["position_mm"] == top.stored["position_mm"]
    # the stand is bumped: 10 px in the image is ~8.5 mm of camera travel at 419 mm (past the 5 mm hysteresis)
    bumped = shifted(top.frames["obj"], 10, 0); want = solve(cc, via_jpg(bumped), top.calib_a, TOP_HFOV)
    top.frame(bumped, t=T); res = srv.scene_step(now=T)
    assert res["refitted"] and res["calib_ok"] and res["n_markers_seen"] == 4
    c = http.get_json("/api/calib")[0]; cam = c["camera"]; p = cam["pose"]
    assert c["live"] is True and cam["pose_live"] is True and cam["pose_at"] == T and 7 < cam["pose_moved_mm"] < 11, cam["pose_moved_mm"]
    assert p["position_mm"] == pytest.approx(want["position_mm"], abs=1e-6) and p["rvec"] == pytest.approx(want["rvec"], abs=1e-9), "cam_calib's solve, with the stored FOV"
    mm, deg = delta(p, top.stored); assert 7 < mm < 11 and deg < 0.5, (mm, deg)
    assert p["hfov_deg"] == TOP_HFOV and p["image_size"] == [640, 360] and p["created"] == top.stored["created"], "the FOV is the lens's: carried over, never re-estimated"
    assert p["source"] == "live marker re-solve, hfov from lens height 16.5 in (tape measure) -> hfov 69.16 deg"
    assert p["camera_matrix"][0][0] == pytest.approx(cc.nominal_K(640, 360, TOP_HFOV)[0, 0]) and p["position_in"] == [round(v / 25.4, 3) for v in p["position_mm"]]
    assert {"optical_axis", "up_table", "tilt_deg", "axis_hits_mm", "reproj_rms_px", "markers_used", "tvec"} <= set(p) and p["markers_used"] == [0, 1, 2, 3]
    # one event line each for the homography and the pose
    assert len([m for m in events() if m.startswith("overhead camera pose re-solved: moved")]) == 1 and any(m.startswith("overhead camera moved 10 px") or m.startswith("overhead camera moved 11 px") for m in events())
    # the write-back (same step: the minute's write was due) carries the pose as camera_pose_est and keeps the arm fit
    t = top.read_table()
    assert srv.SCENE["writes"] == 1 and t["camera_pose_est"]["rvec"] == p["rvec"] and t["camera_pose_est"]["tvec"] == p["tvec"] and t["camera_pose_est"]["source"] == p["source"]
    assert t["arm_fit"] == ARM["arm_fit"] and t["arm_base_mm"] == ARM["arm_base_mm"] and t["created"] == top.calib_a["created"] and t["homography_px_to_table"] == srv.SCENE["live"]["H"]
    assert t["live_solve"]["camera_pose"]["moved_mm"] == cam["pose_moved_mm"] and t["live_solve"]["camera_pose"]["markers_seen"] == 4
    # the next detect agrees with the file: nothing is live any more, the file's (re-solved) pose is what is served
    top.frame(bumped, t=T + 1); srv.scene_step(now=T + 1)
    c = http.get_json("/api/calib")[0]
    assert c["live"] is False and c["camera"]["pose_live"] is False and c["camera"]["pose_at"] is None and c["camera"]["pose"]["position_mm"] == p["position_mm"]
    # a second bump within the minute: served live (moved relative to the pose now in the file), not written yet, no second event line within 30 s
    bumped2 = shifted(top.frames["obj"], 20, 0); top.frame(bumped2, t=T + 5); srv.scene_step(now=T + 5)
    c = http.get_json("/api/calib")[0]; cam = c["camera"]
    assert c["live"] is True and cam["pose_live"] is True and cam["pose_at"] == T + 5 and 7 < cam["pose_moved_mm"] < 11 and srv.SCENE["writes"] == 1
    assert delta(cam["pose"], p)[0] == pytest.approx(cam["pose_moved_mm"], abs=0.06)
    assert len([m for m in events() if m.startswith("overhead camera pose re-solved")]) == 1
    top.frame(bumped2, t=T + 31); srv.scene_step(now=T + 31)
    assert len([m for m in events() if m.startswith("overhead camera pose re-solved")]) == 1, "the pose did not change again: no new line (the solve is cached per re-fit)"
    assert http.get_json("/api/calib")[0]["camera"]["pose_at"] == T + 5
    top.frame(bumped2, t=T + 61); srv.scene_step(now=T + 61)
    assert srv.SCENE["writes"] == 2 and top.read_table()["camera_pose_est"]["position_mm"] == cam["pose"]["position_mm"] and top.read_table()["arm_fit"] == ARM["arm_fit"]


JOINTS_65 = {"shoulder_pan": -24.4, "shoulder_lift": 10.77, "elbow_flex": 27.74, "wrist_flex": 60.84, "wrist_roll": 1.1, "gripper": 1.5}   # follower with the tip on mat (6, 5) in


def test_top_live_pose_drives_the_kinematic_arm_mask(srv, http, top, cc, monkeypatch):
    """The arm silhouette is projected through the pose the server serves: the live re-solve after a bump, not the stale file."""
    top.learn_bg(http); T = time.time(); srv.SCENE["persisted_at"] = T     # the minute's write "already happened": the live pose stays live
    t = T; srv.state["arms"]["follower"] = {"t": t, "role": "follower", "mode": "deg", "src": "idle-poll", "pos": dict(JOINTS_65), "rx": t}
    top.frame(top.frames["obj"], t=T); srv.scene_step(now=T)
    got = []; real = cc.arm_mask_px_from_joints
    monkeypatch.setattr(srv.cam_calib(), "arm_mask_px_from_joints", lambda pos, calib: got.append(calib) or real(pos, calib))
    srv.scene_step(now=T + 1)
    assert srv.SCENE["arm_px"]["source"] == "pose" and got[-1]["camera_pose_est"]["position_mm"] == top.stored["position_mm"], "the file's pose while nothing is live"
    top.frame(shifted(top.frames["obj"], 10, 0), t=T + 2); srv.scene_step(now=T + 2)
    assert srv.SCENE["pose"] is not None and srv.SCENE["writes"] == 0
    srv.state["arms"]["follower"]["rx"] = T + 3; srv.scene_step(now=T + 3)
    assert got[-1]["camera_pose_est"] is srv.SCENE["pose"]["est"] and srv.SCENE["arm_px"]["source"] == "pose"
    assert np.hypot(*(np.array(got[-1]["camera_pose_est"]["position_mm"][:2]) - np.array(top.stored["position_mm"][:2]))) > 7


def test_top_pose_ignores_jitter_and_sub_hysteresis_moves(srv, http, top, events):
    top.learn_bg(http); T = time.time()
    # 0.5 px: MatScene does not even re-fit (under its 3 mm), nothing is live, the pose is the file's
    top.frame(shifted(top.frames["obj"], 0.5, 0), t=T); res = srv.scene_step(now=T)
    assert not res["refitted"] and srv.SCENE["live"] is None and srv.SCENE["pose"] is None
    c = http.get_json("/api/calib")[0]; assert c["live"] is False and c["camera"]["pose_live"] is False and c["camera"]["pose"]["position_mm"] == top.stored["position_mm"]
    # 4 px: the homography goes live (> 3 px) but the camera only moved ~3.4 mm: the served pose stays put and nothing is logged for it
    top.frame(shifted(top.frames["obj"], 4, 0), t=T + 1); res = srv.scene_step(now=T + 1)
    c = http.get_json("/api/calib")[0]
    assert c["live"] is True and 3 < c["shift_px"] < 6 and c["camera"]["pose_live"] is False and c["camera"]["pose_moved_mm"] is None
    assert c["camera"]["pose"]["position_mm"] == top.stored["position_mm"] and srv.SCENE["pose_H"] == srv.SCENE["live"]["H"], "solved once for this re-fit, held"
    assert not [m for m in events() if "pose re-solved" in m] and srv.SCENE["writes"] == 1 and top.read_table()["camera_pose_est"] == top.stored, "the write-back kept the stored pose"


def test_top_pose_is_not_tracked_without_an_hfov(srv, http, top, cc, logs, events):
    """The FOV is pinned by campose --lens-height; a pose without one is served as is and never re-solved (one LOG line)."""
    top.write_table({**top.table, "camera_pose_est": {k: v for k, v in top.stored.items() if k != "hfov_deg"}})
    top.learn_bg(http); T = time.time()
    for i in range(3): top.frame(shifted(top.frames["obj"], 10, 0), t=T + i); srv.scene_step(now=T + i)
    c = http.get_json("/api/calib")[0]
    assert srv.SCENE["writes"] == 1 and c["camera"]["pose_live"] is False and c["camera"]["pose"]["position_mm"] == top.stored["position_mm"] and "hfov_deg" not in c["camera"]["pose"]
    told = [l for l in logs if "overhead camera pose has no hfov; run so101-calib campose --lens-height" in l]
    assert len(told) == 1 and srv.SCENE["pose_err"] and not [m for m in events() if "pose re-solved" in m]
    assert "hfov_deg" not in top.read_table()["camera_pose_est"], "the write-back carried the homography, the pose stayed what it was"
    # no pose at all (a table that never had campose): the same, with its own line
    srv.scene_reset(); top.write_table({k: v for k, v in top.table.items() if k != "camera_pose_est"}); top.learn_bg(http)
    top.frame(shifted(top.frames["obj"], 10, 0), t=T + 10); srv.scene_step(now=T + 10)
    assert len([l for l in logs if "overhead camera has no pose; run so101-calib campose --lens-height" in l]) == 1
    assert http.get_json("/api/calib")[0]["camera"]["pose"] is None and "camera_pose_est" not in top.read_table()


# ---------------------------------------------------------------- the front camera (its own loop on cam_frames["front"])
def render_front(layout, rvec, tvec, K, size=(640, 360), marker_px=240, quiet=0.25):
    """The mat's markers seen by a camera at (rvec, tvec, K): each generated marker image (plus a white quiet zone) is warped so its outer
    corners land where cv2.projectPoints puts them, in the pixel-centre convention (pixel i covers [i, i + 1), so a continuous corner
    coordinate u is pixel index u - 0.5; without that half pixel every marker renders a pixel too big and the solve sits ~5 mm low)."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout["dictionary"])); half = layout["marker_size_mm"] / 2.0; q = int(marker_px * quiet)
    canvas = np.full((size[1], size[0]), 255, np.uint8); qmm = 2 * half * q / marker_px
    for mid, (x, y) in layout["markers"].items():
        patch = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, int(mid), marker_px), q, q, q, q, cv2.BORDER_CONSTANT, value=255)
        obj = np.array([[x - half - qmm, y + half + qmm, 0], [x + half + qmm, y + half + qmm, 0], [x + half + qmm, y - half - qmm, 0], [x - half - qmm, y - half - qmm, 0]], np.float64)
        dst, _ = cv2.projectPoints(obj, np.asarray(rvec, np.float64), np.asarray(tvec, np.float64), K, None)
        src = np.array([[0, 0], [patch.shape[1], 0], [patch.shape[1], patch.shape[0]], [0, patch.shape[0]]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, (dst.reshape(4, 2) - 0.5).astype(np.float32))
        np.minimum(canvas, cv2.warpPerspective(patch, Hm, size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=255), out=canvas)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


class FrontLab:
    def __init__(self, srv, cc, d):
        self.srv, self.dir, self.cc = srv, d, cc
        self.table = json.load(open(os.path.join(REAL_CALIB, "top_table.json")))       # the real mat layout (the front camera has no table of its own)
        self.stored = json.load(open(os.path.join(REAL_CALIB, "front_campose.json")))  # the real front pose: hfov 68.88 from the 18.375 in lens height
        self.layout = layout_of(self.table); self.R = cv2.Rodrigues(np.array(self.stored["rvec"]))[0]
        self.K = cc.nominal_K(640, 360, self.stored["hfov_deg"])
        assert self.K[0, 0] == pytest.approx(self.stored["camera_matrix"][0][0], abs=1e-6), "the file's K is nominal_K at its hfov"
        with open(os.path.join(str(d), "top_table.json"), "w") as f: json.dump(self.table, f)
        self.path = os.path.join(str(d), "front_campose.json"); self.write_pose(self.stored)

    def write_pose(self, pose):
        with open(self.path, "w") as f: json.dump(pose, f)
        os.utime(self.path, (time.time() + 2, time.time() + 2)); self.srv._calib_cache.clear()

    def read_pose(self): return json.load(open(self.path))

    def view(self, centre=None, rvec=None):
        """The front frame from camera centre `centre` (table mm; default the stored pose's) with the stored orientation."""
        C = np.array(self.stored["position_mm"] if centre is None else centre, np.float64); rvec = self.stored["rvec"] if rvec is None else rvec
        R = cv2.Rodrigues(np.array(rvec, np.float64))[0]
        return render_front(self.layout, rvec, -R @ C, self.K)

    def frame(self, im, t=None): self.srv.cam_frames["front"] = (jpg(im), time.time() if t is None else t)

    def covered(self, im, ids, centre=None):
        """A frame (rendered from camera centre `centre`) with the given markers hidden (the arm, the overhead stand)."""
        out = im.copy(); C = np.array(self.stored["position_mm"] if centre is None else centre, np.float64)
        for mid in ids:
            x, y = self.layout["markers"][mid]
            u, v = cv2.projectPoints(np.array([[x, y, 0.0]]), np.array(self.stored["rvec"]), -self.R @ C, self.K, None)[0].ravel()
            cv2.rectangle(out, (int(u) - 45, int(v) - 45), (int(u) + 45, int(v) + 45), (120, 120, 120), -1)
        return out


@pytest.fixture
def front(srv, cc, tmp_path, monkeypatch):
    """Temp CALIB_DIR with the real table (layout) and the real front pose; fresh front state."""
    if not os.path.exists(os.path.join(REAL_CALIB, "front_campose.json")): pytest.skip("no front camera pose in the repo")
    d = tmp_path / "calib"; d.mkdir()
    monkeypatch.setattr(srv, "CALIB_DIR", str(d)); srv._calib_cache.clear(); srv.scene_reset(); srv.front_reset()
    yield FrontLab(srv, cc, d)
    srv.scene_reset(); srv.front_reset()


def test_front_synthetic_frame_solves_back_to_the_pose_it_was_rendered_from(front, cc):
    """The fixture's own check: three of the four markers are in the real front view (marker 1 is past the frame edge, as in the real
    file's markers_used), and the solve lands within 5 mm / 1 deg of the pose that rendered the frame, through the JPEG the server will decode."""
    im = cv2.imdecode(np.frombuffer(jpg(front.view()), np.uint8), cv2.IMREAD_COLOR)
    seen = cc.detect_markers(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "DICT_4X4_50"); assert sorted(seen) == [0, 2, 3]
    got = solve(cc, im, front.table, front.stored["hfov_deg"]); mm, deg = delta(got, front.stored)
    assert mm < 5 and deg < 1, (mm, deg, got["position_mm"], front.stored["position_mm"])
    moved = np.array(front.stored["position_mm"]) + [-30, 0, 30]
    got2 = solve(cc, front.view(moved), front.table, front.stored["hfov_deg"])
    assert np.linalg.norm(np.array(got2["position_mm"]) - moved) < 5 and delta(got2, front.stored)[1] < 1


def test_front_pose_idle_until_the_feed_is_live(srv, http, front, logs):
    assert srv.front_pose_step() is None and srv.FRONT["err"] == "front camera not live"
    fr = http.get_json("/api/calib")[0]["camera"]["front"]
    assert fr["pose_live"] is False and fr["live"] is False and fr["pose_at"] is None and fr["live_at"] is None and fr["moved_mm"] is None and fr["holding"] is False and fr["markers_seen"] is None
    assert fr["pose"]["position_mm"] == front.stored["position_mm"] and fr["live_solve"] is None
    front.frame(front.view(), t=time.time() - 10)
    assert srv.front_pose_step() is None and srv.FRONT["err"] == "front camera not live", "a stale frame is no feed"
    os.unlink(front.path); srv._calib_cache.clear(); front.frame(front.view())
    assert srv.front_pose_step() is None and srv.FRONT["err"].startswith("no front camera pose yet") and http.get_json("/api/calib")[0]["camera"]["front"] is None
    assert not [l for l in logs if "front" in l]


def test_front_pose_unmoved_camera_keeps_the_file_pose(srv, http, front, events):
    T = time.time(); front.frame(front.view(), t=T); r = srv.front_pose_step(now=T)
    assert r["markers_seen"] == 3 and r["holding"] is False and r["moved"] is False and r["moved_mm"] < 5 and r["turned_deg"] < 1
    fr = http.get_json("/api/calib")[0]["camera"]["front"]
    assert fr["pose_live"] is False and fr["pose"]["position_mm"] == front.stored["position_mm"] and fr["markers_seen"] == 3 and fr["holding"] is False
    assert srv.FRONT["writes"] == 0 and front.read_pose() == front.stored and not [m for m in events() if "front camera pose" in m]


def test_front_pose_follows_a_moved_camera_with_hysteresis_log_and_write_back(srv, http, front, cc, events):
    T = time.time(); moved = np.array(front.stored["position_mm"]) + [-30, 0, 30]   # the clamp slipped: 30 mm back and 30 mm up, same aim
    view = front.view(moved); front.frame(view, t=T); r = srv.front_pose_step(now=T)
    assert r["markers_seen"] == 4 and r["moved"] is True and 35 < r["moved_mm"] < 50
    c = http.get_json("/api/calib")[0]; fr = c["camera"]["front"]; p = fr["pose"]
    assert fr["pose_live"] is True and fr["live"] is True and fr["pose_at"] == fr["live_at"] == T and fr["moved_mm"] == r["moved_mm"] and fr["holding"] is False and fr["markers_seen"] == 4
    assert np.linalg.norm(np.array(p["position_mm"]) - moved) < 5 and delta(p, front.stored)[1] < 1, "the pose it was rendered from"
    want = solve(cc, cv2.imdecode(np.frombuffer(jpg(view), np.uint8), cv2.IMREAD_COLOR), front.table, front.stored["hfov_deg"])
    assert p["position_mm"] == pytest.approx(want["position_mm"], abs=1e-6) and p["rvec"] == pytest.approx(want["rvec"], abs=1e-9)
    assert p["hfov_deg"] == front.stored["hfov_deg"] and p["camera_matrix"] == front.stored["camera_matrix"] and p["image_size"] == [640, 360], "the FOV never changes"
    assert p["source"] == "live marker re-solve, hfov from " + front.stored["source"] and p["created"] == front.stored["created"] and p["position_in"] == [round(v / 25.4, 3) for v in p["position_mm"]]
    assert c["camera"]["pose_live"] is False and c["live"] is False, "the top camera's fields are untouched"
    # the homographies are re-derived from the live pose: marker 0 projects where cv2 puts it through the new pose, not the old
    H = fr["homography_table_to_px"]; x, y = front.layout["markers"][0]
    u, v = cv2.projectPoints(np.array([[x, y, 0.0]]), np.array(p["rvec"]), np.array(p["tvec"]), np.array(p["camera_matrix"]), None)[0].ravel()
    w = H[2][0] * x + H[2][1] * y + H[2][2]; assert abs((H[0][0] * x + H[0][1] * y + H[0][2]) / w - u) < 1e-3 and abs((H[1][0] * x + H[1][1] * y + H[1][2]) / w - v) < 1e-3
    u0, v0 = cv2.projectPoints(np.array([[x, y, 0.0]]), np.array(front.stored["rvec"]), np.array(front.stored["tvec"]), front.K, None)[0].ravel()
    assert abs(u - u0) + abs(v - v0) > 5
    loc = http.get_json(f"/api/locate?cam=front&u={u}&v={v}")[0]; assert loc["ok"] is True and loc["table_mm"] == pytest.approx([x, y], abs=0.5)
    # one event line, one write (the minute's write was due): the file is the live pose plus a live_solve block
    assert [m for m in events() if m.startswith("front camera pose re-solved: moved")] == [f"front camera pose re-solved: moved {r['moved_mm']:.0f} mm"]
    f = front.read_pose()
    assert srv.FRONT["writes"] == 1 and f["rvec"] == p["rvec"] and f["tvec"] == p["tvec"] and f["hfov_deg"] == front.stored["hfov_deg"] and f["source"] == p["source"]
    assert f["live_solve"]["moved_mm"] == r["moved_mm"] and f["live_solve"]["markers_seen"] == 4 and "campose --lens-height" in f["live_solve"]["note"]
    assert http.get_json("/api/calib")[0]["camera"]["front"]["live_solve"] == f["live_solve"]
    # next pass: the file holds it, nothing is live, the same pose is served from the file
    front.frame(view, t=T + 1); r2 = srv.front_pose_step(now=T + 1)
    assert r2["moved"] is False and r2["moved_mm"] < 5
    fr = http.get_json("/api/calib")[0]["camera"]["front"]
    assert fr["pose_live"] is False and fr["live_at"] is None and fr["moved_mm"] is None and fr["pose"]["position_mm"] == p["position_mm"]
    # a second move within the minute: served live, not written, no second event line within 30 s; written after the minute
    moved2 = moved + [0, -20, 0]; view2 = front.view(moved2); front.frame(view2, t=T + 5); r3 = srv.front_pose_step(now=T + 5)
    assert r3["moved"] is True and 12 < r3["moved_mm"] < 30 and srv.FRONT["writes"] == 1
    fr = http.get_json("/api/calib")[0]["camera"]["front"]; assert fr["pose_live"] is True and fr["pose_at"] == T + 5 and np.linalg.norm(np.array(fr["pose"]["position_mm"]) - moved2) < 10   # three markers from here: a looser solve
    assert len([m for m in events() if m.startswith("front camera pose re-solved")]) == 1
    front.frame(view2, t=T + 31); r4 = srv.front_pose_step(now=T + 31)
    assert r4["moved"] is False and srv.FRONT["writes"] == 1 and http.get_json("/api/calib")[0]["camera"]["front"]["pose_at"] == T + 5, "held: the solve agrees with the live pose"
    assert len([m for m in events() if m.startswith("front camera pose re-solved")]) == 1
    front.frame(view2, t=T + 61); srv.front_pose_step(now=T + 61)
    assert srv.FRONT["writes"] == 2 and front.read_pose()["position_mm"] == fr["pose"]["position_mm"]


def test_front_with_two_markers_holds_the_last_pose(srv, http, front, events):
    T = time.time(); moved = np.array(front.stored["position_mm"]) + [-30, 0, 30]
    srv.FRONT["persisted_at"] = T                                                  # pretend the minute's write already happened: the live pose stays live
    front.frame(front.view(moved), t=T); assert srv.front_pose_step(now=T)["moved"] is True
    live = http.get_json("/api/calib")[0]["camera"]["front"]["pose"]
    # the arm swings through and hides two markers: nothing to solve from, the live pose is held
    front.frame(front.covered(front.view(moved), (2, 3), moved), t=T + 1); r = srv.front_pose_step(now=T + 1)
    assert r == {"markers_seen": 2, "holding": True, "moved": False}, r
    fr = http.get_json("/api/calib")[0]["camera"]["front"]
    assert fr["holding"] is True and fr["markers_seen"] == 2 and fr["pose_live"] is True and fr["pose"] == live and srv.FRONT["writes"] == 0
    front.frame(front.covered(front.view(moved), (1, 2, 3), moved), t=T + 2); r = srv.front_pose_step(now=T + 2)
    assert r["markers_seen"] == 1 and r["holding"] is True and http.get_json("/api/calib")[0]["camera"]["front"]["pose"] == live
    # with nothing live yet, the file's pose is what is held
    srv.front_reset(); srv.FRONT["persisted_at"] = T
    front.frame(front.covered(front.view(), (2,)), t=T + 3); assert srv.front_pose_step(now=T + 3)["holding"] is True
    fr = http.get_json("/api/calib")[0]["camera"]["front"]
    assert fr["holding"] is True and fr["markers_seen"] == 2 and fr["pose_live"] is False and fr["pose"]["position_mm"] == front.stored["position_mm"]
    assert front.read_pose() == front.stored and len([m for m in events() if "front camera pose" in m]) == 1


def test_front_pose_is_not_tracked_without_an_hfov(srv, http, front, logs, events):
    front.write_pose({k: v for k, v in front.stored.items() if k != "hfov_deg"}); T = time.time()
    moved = np.array(front.stored["position_mm"]) + [-30, 0, 30]
    for i in range(3): front.frame(front.view(moved), t=T + i); assert srv.front_pose_step(now=T + i) is None
    told = [l for l in logs if "front camera pose has no hfov; run so101-calib campose --name front --lens-height" in l]
    assert len(told) == 1 and srv.FRONT["err"] == srv.FRONT["pose_err"] and srv.FRONT["writes"] == 0
    fr = http.get_json("/api/calib")[0]["camera"]["front"]
    assert fr["pose_live"] is False and fr["pose"]["position_mm"] == front.stored["position_mm"] and "hfov_deg" not in fr["pose"] and fr["markers_seen"] is None
    assert not [m for m in events() if "front camera pose" in m] and "hfov_deg" not in front.read_pose()
    front.write_pose(front.stored)                                                  # campose --lens-height re-run: tracked again
    front.frame(front.view(moved), t=T + 10); assert srv.front_pose_step(now=T + 10)["moved"] is True and srv.FRONT["pose_err"] is None


def test_front_pose_needs_the_table_layout_and_skips_the_top_scene(srv, http, front):
    os.unlink(os.path.join(str(front.dir), "top_table.json")); srv._calib_cache.clear(); front.frame(front.view())
    assert srv.front_pose_step() is None and srv.FRONT["err"] == "no table calibration yet (run so101-calib table)"
    assert srv.SCENE["res"] is None and srv.SCENE["pose"] is None, "the front loop never touches the overhead scene"


def test_front_pose_loop_runs_and_sleeps(srv, front, run_loop):
    front.frame(front.view()); run_loop(srv.front_pose_loop, 2)
    assert srv.FRONT["n"] == 2 and srv.FRONT["markers_seen"] == 3


def test_top_pose_tracking_moves_nothing(srv, http, top, fake):
    top.learn_bg(http); top.frame(shifted(top.frames["obj"], 10, 0)); srv.scene_step(); http.get_json("/api/calib")
    assert srv.SCENE["pose"] is not None and fake.calls == [] and fake.popens == []


def test_front_pose_tracking_moves_nothing(srv, http, front, fake):
    front.frame(front.view(np.array(front.stored["position_mm"]) + [-30, 0, 30])); srv.front_pose_step()
    http.get_json("/api/calib"); http.get_json("/api/locate?cam=front&u=320&v=180")
    assert srv.FRONT["pose"] is not None and fake.calls == [] and fake.popens == []
