"""The mat-scene endpoints (/api/scene, /api/scene.png, POST /api/scene/background) and the self-healing overhead calibration.

The "camera" is cam_frames["top"] filled by hand with JPEG-encoded fixture frames from tests/scene/fixtures (real frames of
9 Nov 2026: bg.jpg = empty mat, obj.jpg = one hex nut at (10.94, 6.94) in, new.jpg = the mat turned round and the markers
re-glued), the calibration is calib_a.json plus a synthetic arm fit in a temp CALIB_DIR, and scene_step(now=...) is driven by
hand with a chosen clock, so the rate limits (one LOG line per 30 s, one write-back per 60 s) are checked without waiting.
A moved camera is obj.jpg translated by a few pixels with cv2.warpAffine. No camera, no network, no Jetson: the autouse
``fake`` rejects any subprocess."""
import json, os, time
import pytest

cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(ROOT, "tests", "scene", "fixtures")
NO_TABLE = "no table calibration yet (run so101-calib table)"
NUT_IN = (10.94, 6.94)
IN = 25.4
ARM = {"arm_base_mm": [228.6, -76.2], "arm_yaw_deg": 90.0,
       "arm_fit": {"arm_base_mm": [228.6, -76.2], "arm_yaw_deg": 90.0, "table_z_mm": 3.282,
                   "joint_offsets_deg": {"shoulder_lift": -3.9064, "elbow_flex": -5.1761, "wrist_flex": -0.6875}, "wrist_roll_deg": 0.46,
                   "fitted": "2026-11-09 21:37:42", "residual_xy_mm": [0.94, 0.71, 1.0925, 0.86],
                   "points": [["table", [9.0, 3.0]], ["table", [6.0, 5.0]], ["table", [9.0, 7.0]], ["table", [12.0, 5.0]]]}}


def img(name):
    im = cv2.imread(os.path.join(FIX, name), cv2.IMREAD_COLOR); assert im is not None, name; return im


def jpg(im):
    ok, b = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85]); assert ok; return b.tobytes()


def shifted(im, dx, dy):
    """The same view after the camera stand was bumped: every pixel moves by (dx, dy)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]]); return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def h_apply(H, u, v):
    p = np.array(H, float) @ np.array([u, v, 1.0]); return [float(p[0] / p[2]), float(p[1] / p[2])]


def marker_px(table, mid):
    Hinv = np.linalg.inv(np.array(table["homography_px_to_table"], float)); return h_apply(Hinv, *table["layout_markers"][str(mid)])


def masked(im, table, ids):
    """obj.jpg with the given markers painted over (a hand, a tool box... two of them gone leaves too few to re-solve)."""
    out = im.copy()
    for mid in ids:
        u, v = marker_px(table, mid); cv2.rectangle(out, (int(u) - 45, int(v) - 45), (int(u) + 45, int(v) + 45), (120, 120, 120), -1)
    return out


class Lab:
    def __init__(self, srv, d):
        self.srv, self.dir = srv, d
        self.calib_a = json.load(open(os.path.join(FIX, "calib_a.json")))
        self.table = {**self.calib_a, **ARM}
        self.path = os.path.join(str(d), "top_table.json")
        self.write_table(self.table)
        self.frames = {n: img(n + ".jpg") for n in ("bg", "obj", "new")}

    def write_table(self, table):
        with open(self.path, "w") as f: json.dump(table, f)
        os.utime(self.path, (time.time() + 2, time.time() + 2)); self.srv._calib_cache.clear()

    def read_table(self): return json.load(open(self.path))

    def frame(self, im, t=None):
        """The newest top-camera JPEG, as jetson_cam_loop would leave it."""
        self.srv.cam_frames["top"] = (jpg(im), time.time() if t is None else t)

    def learn_bg(self, http):
        self.frame(self.frames["bg"]); r = http.post_json("/api/scene/background"); assert r["ok"] is True, r; return r


@pytest.fixture
def lab(srv, tmp_path, monkeypatch):
    """Temp CALIB_DIR with calib A (+ arm fit), fresh scene state, no pauses between median-background frames."""
    d = tmp_path / "calib"; d.mkdir()
    monkeypatch.setattr(srv, "CALIB_DIR", str(d)); monkeypatch.setattr(srv, "SCENE_BG_SPACING_S", 0)
    srv._calib_cache.clear(); srv.scene_reset()
    yield Lab(srv, d)
    srv.scene_reset()


@pytest.fixture
def no_cv2(srv, monkeypatch):
    def boom(): raise ImportError("No module named 'cv2'")
    monkeypatch.setattr(srv, "scene_mod", boom)


# ---------------------------------------------------------------- degraded answers
def test_scene_is_ok_false_without_table_frame_or_module(srv, http, lab, no_cv2, events):
    os.unlink(lab.path); srv._calib_cache.clear()
    r, _ = http.get_json("/api/scene"); assert r == {"ok": False, "err": NO_TABLE, "age_s": None}
    assert http.get("/api/scene")[1]["Cache-Control"] == "no-store"
    assert http.post_json("/api/scene/background") == {"ok": False, "err": NO_TABLE}
    lab.write_table(lab.table)
    assert http.get_json("/api/scene")[0] == {"ok": False, "err": "top camera not live", "age_s": None}
    assert http.post_json("/api/scene/background") == {"ok": False, "err": "top camera not live"}
    lab.frame(lab.frames["bg"])
    assert srv.scene_step() is None and srv.SCENE["err"] == "scene_detect unavailable: No module named 'cv2'"
    assert http.get_json("/api/scene")[0] == {"ok": False, "err": "scene_detect unavailable: No module named 'cv2'", "age_s": None}
    assert http.post_json("/api/scene/background") == {"ok": False, "err": "scene_detect unavailable: No module named 'cv2'"}
    st, hdr, body = http.get("/api/scene.png")
    assert st == 200 and hdr["Content-Type"] == "application/json" and json.loads(body)["ok"] is False
    c, _ = http.get_json("/api/calib")
    assert c["ok"] is True and c["live"] is False and c["holding"] is False and c["homography_px_to_table_mm"] == c["stored_homography"] == lab.calib_a["homography_px_to_table"]
    assert all("camera moved" not in m for m in events())


def test_stale_frame_counts_as_no_camera(srv, http, lab):
    lab.frame(lab.frames["bg"], t=time.time() - 10)
    assert srv.scene_step() is None and srv.SCENE["err"] == "top camera not live"
    assert http.get_json("/api/scene")[0]["err"] == "top camera not live"


# ---------------------------------------------------------------- background and objects
def test_background_missing_then_learned_and_saved(srv, http, lab, events):
    lab.frame(lab.frames["bg"])
    res = srv.scene_step()
    assert res["background"] == "missing" and res["objects"] == [] and res["n_markers_seen"] == 4
    s, _ = http.get_json("/api/scene")
    assert s["ok"] is True and s["background"] == "missing" and s["objects"] == [] and s["calib_ok"] is True and s["holding"] is False and s["calib_live"] is False
    assert 0 <= s["age_s"] < 5
    r = http.post_json("/api/scene/background")
    png = os.path.join(str(lab.dir), "top_mat_bg.png")
    assert r == {"ok": True, "saved": png, "frames": 1} and os.path.exists(png)
    bg = cv2.imread(png, cv2.IMREAD_COLOR); assert bg is not None and bg.shape == (152, 229, 3), "top-down mat image at 0.5 px/mm"
    assert http.get_json("/api/scene")[0]["background"] == "ok", "the POST refreshed the map itself"
    assert srv.scene_step()["background"] == "ok"
    assert any(m.startswith("mat background learned") for m in events())
    # the median option: capped at 15 frames; junk is refused
    assert http.post_json("/api/scene/background", {"frames": 50})["frames"] == 15
    assert http.post_json("/api/scene/background", {"frames": 3})["frames"] == 3
    assert http.post_json("/api/scene/background", {"frames": "lots"}) == {"ok": False, "err": "frames must be a whole number"}
    # a restart finds the saved background
    srv.scene_reset(); lab.frame(lab.frames["bg"])
    assert srv.scene_step()["background"] == "ok" and srv.SCENE["scene"].bg_source == png


def test_scene_finds_the_nut_in_mat_inches(srv, http, lab):
    """The server's scene needs SCENE_MIN_HITS consecutive sightings before an object is listed (one-frame shadows and
    flicker never make it to the map); until then it is counted in `pending`."""
    lab.learn_bg(http); lab.frame(lab.frames["obj"])
    for i in range(srv.SCENE_MIN_HITS - 1):
        r = srv.scene_step(); assert r is not None and r["objects"] == [] and r["pending"] == 1, (i, r.get("objects"))
    assert srv.scene_step() is not None
    s, body = http.get_json("/api/scene")
    assert s["ok"] is True and s["background"] == "ok" and len(s["objects"]) == 1 and s["pending"] == 0, s.get("objects")
    o = s["objects"][0]
    assert abs(o["x_in"] - NUT_IN[0]) * IN < 6 and abs(o["y_in"] - NUT_IN[1]) * IN < 6
    assert o["id"] == 1 and 5 <= o["w_mm"] <= 25 and 5 <= o["h_mm"] <= 25 and len(o["color"]) == 3 and len(o["px"]) == 2 and len(o["bbox_mm"]) == 4
    assert {"id", "x_mm", "y_mm", "x_in", "y_in", "w_mm", "h_mm", "angle_deg", "area_mm2", "color", "px", "bbox_mm"} <= set(o)
    assert s["n_markers_seen"] == 4 and s["calib_ok"] is True and s["calib_note"] is None and s["calib_live"] is False and s["holding"] is False
    assert {"t", "age_s", "arm_in_view", "arm_mask_frac", "refitted", "mat_shift_mm", "mat_rot_deg", "px_per_mm", "mat_mm", "calib_live_at", "shift_px"} <= set(s)
    # the camera drops out: the last map is not served as if it were current
    srv.cam_frames.clear()
    s2, _ = http.get_json("/api/scene"); assert s2["ok"] is False and s2["err"] == "top camera not live" and s2["age_s"] is not None


def test_scene_png_is_the_annotated_top_down_view(srv, http, lab):
    lab.learn_bg(http); lab.frame(lab.frames["obj"]); srv.scene_step()
    st, hdr, body = http.get("/api/scene.png")
    assert st == 200 and hdr["Content-Type"] == "image/png" and hdr["Cache-Control"] == "no-store" and body[:8] == b"\x89PNG\r\n\x1a\n"
    im = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert im.shape == (152 * 4, 229 * 4, 3)


def test_background_adapts_every_tenth_detect(srv, http, lab, monkeypatch):
    lab.learn_bg(http); lab.frame(lab.frames["obj"]); srv.scene_step()
    sc = srv.SCENE["scene"]; calls = []
    monkeypatch.setattr(sc, "update_background", lambda frame, **kw: calls.append(kw))
    for _ in range(19): srv.scene_step()
    assert srv.SCENE["n"] == 20 and calls == [{}, {}], "detects 10 and 20, module defaults"


def test_scene_is_rebuilt_when_the_table_file_changes(srv, http, lab):
    lab.learn_bg(http); lab.frame(lab.frames["obj"]); srv.scene_step()
    first = srv.SCENE["scene"]; assert srv.scene_step() is not None and srv.SCENE["scene"] is first
    calib_b = json.load(open(os.path.join(FIX, "calib_b.json"))); lab.write_table({**calib_b, **ARM})
    srv.scene_step()
    assert srv.SCENE["scene"] is not first and srv.SCENE["scene"].calib["created"] == calib_b["created"]
    assert srv.SCENE["scene"].bg is not None, "the saved background PNG is reloaded for the new scene"


# ---------------------------------------------------------------- self-healing calibration
def test_camera_moved_live_solve_log_line_and_write_back(srv, http, lab, events):
    lab.learn_bg(http)
    T = time.time(); H0 = lab.calib_a["homography_px_to_table"]
    for i in range(srv.SCENE_MIN_HITS - 1, 0, -1): lab.frame(lab.frames["obj"], t=T - i); srv.scene_step(now=T - i)   # the nut is already a persisted track
    lab.frame(shifted(lab.frames["obj"], 10, 0), t=T)
    res = srv.scene_step(now=T)
    assert res["refitted"] and res["calib_ok"] and res["n_markers_seen"] == 4 and len(res["objects"]) == 1
    o = res["objects"][0]; assert abs(o["x_in"] - NUT_IN[0]) * IN < 6 and abs(o["y_in"] - NUT_IN[1]) * IN < 6, "the nut is still where it is on the mat"
    # (a) /api/calib hands out the live solve, the stored one beside it
    c, _ = http.get_json("/api/calib")
    assert c["ok"] is True and c["live"] is True and c["live_at"] == T and 8 < c["shift_px"] < 13 and c["holding"] is False and c["calib_ok"] is True and c["markers_seen"] == 4
    assert c["stored_homography"] == srv.SCENE["live"]["H"] == c["homography_px_to_table_mm"] != H0, "the file already holds the re-solve (written in the same step)"
    assert c["arm"]["base_mm"] == [228.6, -76.2] and c["camera"]["created"] == lab.calib_a["created"]
    assert c["live_solve"]["markers_seen"] == 4 and 8 < c["live_solve"]["shift_px"] < 13 and "camera moved" in c["live_solve"]["note"]
    # /api/locate uses it: the nut's raw pixel in the shifted frame lands on the nut's mat position, which the old solve misses by ~9 mm
    u, v = o["px"]; r, _ = http.get_json(f"/api/locate?u={u}&v={v}")
    assert r["ok"] is True and r["table_mm"] == pytest.approx(h_apply(c["homography_px_to_table_mm"], u, v), abs=0.01)
    assert r["table_in"] == pytest.approx(NUT_IN, abs=6 / IN) and np.hypot(*(np.array(h_apply(H0, u, v)) - np.array(r["table_mm"]))) > 6
    assert r["arm_in"] is not None, "the arm pose is still there"
    # (b) one LOG line
    moved = [m for m in events() if m.startswith("overhead camera moved")]
    assert len(moved) == 1 and "10 px" in moved[0] or "11 px" in moved[0], moved
    # (c) written back once: arm pose and 'created' preserved, per-marker residuals of the old solve gone, a live_solve block
    t = lab.read_table()
    assert srv.SCENE["writes"] == 1 and t["homography_px_to_table"] == srv.SCENE["live"]["H"]
    assert t["arm_fit"] == ARM["arm_fit"] and t["arm_base_mm"] == ARM["arm_base_mm"] and t["arm_yaw_deg"] == ARM["arm_yaw_deg"] and t["created"] == lab.calib_a["created"]
    assert t["layout_markers"] == lab.calib_a["layout_markers"] and t["markers_used"] == [0, 1, 2, 3] and "residual_mm" not in t and t["residual_mm_max"] < 1.5
    assert t["live_solve"]["at"] == time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(T))
    assert srv.SCENE["scene"].calib is not t, "our own write does not rebuild the scene"
    sc = srv.SCENE["scene"]
    # a second bump 5 s later (another 10 px): live again, but no new LOG line and no new write yet
    lab.frame(shifted(lab.frames["obj"], 20, 0), t=T + 5); srv.scene_step(now=T + 5)
    c, _ = http.get_json("/api/calib")
    assert c["live"] is True and 8 < c["shift_px"] < 13 and c["live_at"] == T + 5 and c["stored_homography"] == t["homography_px_to_table"] != c["homography_px_to_table_mm"]
    assert len([m for m in events() if m.startswith("overhead camera moved")]) == 1 and srv.SCENE["writes"] == 1 and srv.SCENE["scene"] is sc
    assert srv.SCENE["res"]["objects"][0]["id"] == 1, "tracks survive the re-solves"
    lab.frame(shifted(lab.frames["obj"], 20, 0), t=T + 31); srv.scene_step(now=T + 31)
    assert len([m for m in events() if m.startswith("overhead camera moved")]) == 2 and srv.SCENE["writes"] == 1
    lab.frame(shifted(lab.frames["obj"], 20, 0), t=T + 61); srv.scene_step(now=T + 61)
    assert srv.SCENE["writes"] == 2 and lab.read_table()["homography_px_to_table"] == srv.SCENE["live"]["H"] and lab.read_table()["arm_fit"] == ARM["arm_fit"]
    # once the file holds the solve, the next detect agrees with it: nothing is "live" any more
    lab.frame(shifted(lab.frames["obj"], 20, 0), t=T + 62); srv.scene_step(now=T + 62)
    c, _ = http.get_json("/api/calib")
    assert c["live"] is False and c["shift_px"] is None and c["homography_px_to_table_mm"] == c["stored_homography"] == lab.read_table()["homography_px_to_table"]
    assert http.get_json("/api/scene")[0]["calib_live"] is False


def test_fewer_than_three_markers_holds_the_last_good_solve(srv, http, lab, events):
    lab.learn_bg(http); T = time.time(); H0 = lab.calib_a["homography_px_to_table"]
    lab.frame(masked(lab.frames["obj"], lab.table, (0, 1)), t=T); res = srv.scene_step(now=T)
    assert res["n_markers_seen"] == 2 and res["calib_ok"] is True and not res["refitted"]
    c, _ = http.get_json("/api/calib")
    assert c["holding"] is True and c["live"] is False and c["markers_seen"] == 2 and c["homography_px_to_table_mm"] == H0
    s, _ = http.get_json("/api/scene"); assert s["holding"] is True and s["n_markers_seen"] == 2
    assert srv.SCENE["writes"] == 0 and lab.read_table()["homography_px_to_table"] == H0
    # the camera moves while two markers are covered: nothing to re-solve from, so the last solve is held and the note says the markers are off
    lab.frame(shifted(masked(lab.frames["obj"], lab.table, (0, 1)), 10, 0), t=T + 1); res = srv.scene_step(now=T + 1)
    assert res["n_markers_seen"] == 2 and res["calib_ok"] is False and "off the layout" in res["calib_note"]
    c, _ = http.get_json("/api/calib")
    assert c["holding"] is True and c["live"] is False and c["calib_ok"] is False and "off the layout" in c["calib_note"] and c["homography_px_to_table_mm"] == H0
    assert srv.SCENE["writes"] == 0 and not [m for m in events() if m.startswith("overhead camera moved")]
    # a live solve already in hand is what gets held
    srv.SCENE["persisted_at"] = T                                   # pretend the minute's write already happened
    lab.frame(shifted(lab.frames["obj"], 10, 0), t=T + 2); srv.scene_step(now=T + 2)
    c, _ = http.get_json("/api/calib"); assert c["live"] is True and c["holding"] is False and srv.SCENE["writes"] == 0
    H_live = c["homography_px_to_table_mm"]
    lab.frame(shifted(masked(lab.frames["obj"], lab.table, (0, 1)), 10, 0), t=T + 3); srv.scene_step(now=T + 3)
    c, _ = http.get_json("/api/calib")
    assert c["live"] is True and c["holding"] is True and c["homography_px_to_table_mm"] == H_live and c["stored_homography"] == H0
    assert srv.SCENE["writes"] == 0 and lab.read_table()["homography_px_to_table"] == H0


def test_a_moved_mat_is_flagged_and_never_written_back(srv, http, lab, events):
    lab.learn_bg(http); T = time.time(); before = lab.read_table()
    lab.frame(lab.frames["new"], t=T); res = srv.scene_step(now=T)      # the mat turned round and the markers re-glued (calib B's world, calib A's file)
    assert res["n_markers_seen"] == 4 and res["refitted"] and res["calib_ok"] is False and "re-run cam_calib.py table" in res["calib_note"]
    assert res["mat_shift_mm"] > 40 or abs(res["mat_rot_deg"]) > 150
    s, _ = http.get_json("/api/scene"); assert s["calib_ok"] is False and s["calib_note"] == res["calib_note"] and s["calib_live"] is False
    c, _ = http.get_json("/api/calib")
    assert c["live"] is False and c["calib_ok"] is False and c["calib_note"] == res["calib_note"] and c["holding"] is False and c["markers_seen"] == 4
    assert c["homography_px_to_table_mm"] == c["stored_homography"] == before["homography_px_to_table"]
    assert lab.read_table() == before and srv.SCENE["writes"] == 0
    assert not [m for m in events() if m.startswith("overhead camera moved")]
    r, _ = http.get_json("/api/locate?u=320&v=180"); assert r["ok"] is True and r["table_mm"] == pytest.approx(h_apply(before["homography_px_to_table"], 320, 180), abs=0.01)


# ---------------------------------------------------------------- the kinematic arm mask (follower joints -> arm_px -> detect)
JOINTS_65 = {"shoulder_pan": -24.4, "shoulder_lift": 10.77, "elbow_flex": 27.74, "wrist_flex": 60.84, "wrist_roll": 1.1, "gripper": 1.5}   # tip on mat (6, 5) in, 135 mm from the nut


def follower(srv, pos=JOINTS_65, mode="deg", age=0.0):
    """What arm_loop leaves in state['arms'] from the Jetson's arm_state.py stream, `age` seconds ago."""
    t = time.time() - age
    srv.state["arms"]["follower"] = {"t": t, "role": "follower", "mode": mode, "src": "idle-poll", "pos": dict(pos), "rx": t}


def spy_detect(srv, monkeypatch):
    sc = srv.SCENE["scene"]; real = sc.detect; got = []
    def detect(frame, t=None, arm_px=None):
        got.append(arm_px); return real(frame, t=t, arm_px=arm_px)
    monkeypatch.setattr(sc, "detect", detect); return got


def test_fresh_follower_joints_mask_the_arm(srv, http, lab, monkeypatch):
    lab.learn_bg(http); lab.frame(lab.frames["obj"])
    assert srv.scene_step()["arm_masked_by"] == "edge", "no follower yet: the edge heuristic"
    assert http.get_json("/api/scene")[0]["arm_masked_by"] == "edge"
    follower(srv); got = spy_detect(srv, monkeypatch)
    for _ in range(srv.SCENE_MIN_HITS): res = srv.scene_step()            # the nut has to persist before it is listed
    assert res["arm_masked_by"] == "kinematics" and len(got) == srv.SCENE_MIN_HITS
    arm = got[-1]
    assert arm["source"] == "homography", "the lab's table has no camera_pose_est: mat-level projection with inflated radii"
    assert arm["names"] == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "tip", "fingers"] and len(arm["points"]) == len(arm["radii_px"]) == 7
    assert srv.SCENE["arm_px"] is arm and srv.SCENE["arm_err"] is None
    u, v = arm["points"][5]; x, y = np.array(h_apply(lab.table["homography_px_to_table"], u, v)) / IN
    assert abs(x - 6) < 0.3 and abs(y - 5) < 0.3, "the tip lands on mat (6, 5) in through the arm fit"
    assert len(res["objects"]) == 1 and abs(res["objects"][0]["x_in"] - NUT_IN[0]) * IN < 6, "the nut, 135 mm from the gripper, is still there"
    s, _ = http.get_json("/api/scene"); assert s["ok"] is True and s["arm_masked_by"] == "kinematics" and s["arm_in_view"] is True


def test_stale_or_raw_joints_skip_the_mask(srv, http, lab, monkeypatch):
    lab.learn_bg(http); lab.frame(lab.frames["obj"]); srv.scene_step(); got = spy_detect(srv, monkeypatch)
    follower(srv, age=5.0)
    assert srv.scene_step()["arm_masked_by"] == "edge" and got[-1] is None, "joints older than 2 s"
    follower(srv, mode="raw")
    assert srv.scene_step()["arm_masked_by"] == "edge" and got[-1] is None, "raw motor ticks are not degrees"
    follower(srv, pos={})
    assert srv.scene_step()["arm_masked_by"] == "edge" and got[-1] is None, "busy / unplugged: empty pos"
    srv.state["arms"]["leader"] = {"t": time.time(), "role": "leader", "mode": "deg", "pos": dict(JOINTS_65), "rx": time.time()}; srv.state["arms"].pop("follower")
    assert srv.scene_step()["arm_masked_by"] == "edge" and got[-1] is None, "only the follower is over the mat"
    follower(srv, age=1.0)
    assert srv.scene_step()["arm_masked_by"] == "kinematics" and got[-1]["source"] == "homography"
    assert http.get_json("/api/scene")[0]["arm_masked_by"] == "kinematics"
    now = time.time()
    assert srv.scene_joints(now) == JOINTS_65 and srv.scene_joints(now + 3) is None


def test_table_without_an_arm_pose_logs_once_and_falls_back(srv, http, lab, logs, monkeypatch):
    lab.write_table({k: v for k, v in lab.calib_a.items() if k not in ("arm_base_mm", "arm_yaw_deg")})   # homography only: no arm pose
    lab.learn_bg(http); lab.frame(lab.frames["obj"]); follower(srv); got = spy_detect(srv, monkeypatch)
    for _ in range(3): assert srv.scene_step()["arm_masked_by"] == "edge" and got[-1] is None
    skipped = [l for l in logs if "arm mask skipped" in l]
    assert len(skipped) == 1 and "arm_base_mm" in skipped[0] and "arm_base_mm" in srv.SCENE["arm_err"]
    lab.write_table(lab.table)                                                # the arm fit is back: masked again, error cleared
    assert srv.scene_step()["arm_masked_by"] == "kinematics" and srv.SCENE["arm_err"] is None


def test_scene_moves_nothing(srv, http, lab, fake):
    lab.learn_bg(http); lab.frame(shifted(lab.frames["obj"], 10, 0)); srv.scene_step()
    http.get_json("/api/scene"); http.get_json("/api/calib"); http.get("/api/scene.png")
    assert fake.calls == [] and fake.popens == []
