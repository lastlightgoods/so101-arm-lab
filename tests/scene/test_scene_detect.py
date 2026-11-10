"""dashboard/scene_detect.py: objects on the mat from overhead frames. Real fixtures (hex nut, arm, two mat layouts)
plus synthetic squares un-warped into the camera frame through the calibration."""
import json, os, time
import cv2, numpy as np, pytest
from shelpers import FIX, paste_square

NUT_IN = (10.97, 6.94)            # the hex nut in obj.jpg, mat inches, measured by hand
NUT_PX = (273.0, 209.5)           # its raw-frame centroid


def dist_mm(o, x, y): return float(np.hypot(o["x_mm"] - x, o["y_mm"] - y))


# ---------------------------------------------------------------- construction and conventions
def test_no_homography_raises(sd):
    with pytest.raises(ValueError, match="homography"): sd.MatScene({"image_size": [640, 360]})
    with pytest.raises(ValueError): sd.MatScene({"homography_px_to_table": [[1, 0, 0], [0, 1, 0], [0, 0, 0]]})
    with pytest.raises(ValueError): sd.MatScene("not a dict")


def test_warp_conventions(sd, calib_a, frames):
    s = sd.MatScene(calib_a, px_per_mm=0.5)
    w = s.warp(frames["bg"])
    assert w.shape == (152, 229, 3)                                              # 304.8 x 457.2 mm at 0.5 px/mm
    assert s.mm_to_warp(0, 0) == pytest.approx((0, 152.4)) and s.mm_to_warp(457.2, 304.8) == pytest.approx((228.6, 0))   # origin bottom-left
    x, y = s.warp_to_mm(*s.mm_to_warp(123.4, 56.7)); assert (x, y) == pytest.approx((123.4, 56.7))
    # marker 0 sits at (50.8, 50.8): bottom-left of the warp, and its square is high-contrast against the plain mat
    c, r = s.mm_to_warp(50.8, 50.8); c, r = int(c), int(r)
    assert r > s.Hh * 0.7 and c < s.W * 0.3
    g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
    assert g[r - 10:r + 10, c - 10:c + 10].std() > 30 and g[r - 10:r + 10, c - 10:c + 10].min() < 60 and g[r - 10:r + 10, c - 10:c + 10].max() > 150
    # raw pixel <-> mm round trips through the calibration
    u, v = s.mm_to_px(100, 100); assert s.px_to_mm(u, v) == pytest.approx((100, 100), abs=1e-6)


def test_default_px_per_mm_is_sane(sd, calib_a):
    s = sd.MatScene(calib_a); assert 0.3 <= s.ppm <= 1.0 and s.W * s.Hh < 60000


# ---------------------------------------------------------------- real frames
def test_hex_nut_found_and_nothing_else(scene_a, frames):
    res = scene_a.detect(frames["obj"], t=100.0)
    assert res["background"] == "ok" and res["n_markers_seen"] == 4 and res["calib_ok"] and res["calib_note"] is None
    assert len(res["objects"]) == 1, res["objects"]
    o = res["objects"][0]
    assert abs(o["x_in"] - NUT_IN[0]) * 25.4 < 6 and abs(o["y_in"] - NUT_IN[1]) * 25.4 < 6
    assert o["x_mm"] == pytest.approx(o["x_in"] * 25.4, abs=0.1)
    assert abs(o["px"][0] - NUT_PX[0]) < 5 and abs(o["px"][1] - NUT_PX[1]) < 5
    assert 40 <= o["area_mm2"] <= 400 and 5 <= o["w_mm"] <= 25 and 5 <= o["h_mm"] <= 25
    assert res["t"] == 100.0 and o["id"] == 1
    assert len(o["color"]) == 3 and all(0 <= c <= 255 for c in o["color"])


def test_clean_frames_have_no_objects(scene_a, scene_b, frames):
    r = scene_a.detect(frames["bg"]); assert r["objects"] == [] and not r["arm_in_view"]
    r = scene_a.detect(frames["top_now"]); assert r["objects"] == [] and r["arm_in_view"]      # arm raised at the top: masked, not an object
    r = scene_b.detect(frames["new"]); assert r["objects"] == [] and not r["arm_in_view"]


def test_arm_is_masked_not_reported(scene_a, frames):
    r = scene_a.detect(frames["obj"])
    assert r["arm_in_view"] and r["arm_mask_frac"] > 0.02
    assert all(o["y_mm"] > 30 for o in r["objects"])                               # nothing reported along the arm edge
    arm = scene_a.last["arm"]
    assert arm[-3:].any() and not arm[:arm.shape[0] // 2].any()                    # the arm blob touches the bottom edge only
    r2 = scene_a.detect(frames["top_now"]); assert r2["arm_in_view"] and r2["arm_mask_frac"] > r["arm_mask_frac"] * 0.8


def test_exposure_drift_is_tolerated(scene_a, frames):
    for k in (0.85, 1.15):
        f = np.clip(frames["bg"].astype(np.float32) * k, 0, 255).astype(np.uint8)
        r = scene_a.detect(f); assert r["objects"] == [], (k, r["objects"])
        assert abs(r["gain"] * k - 1) < 0.08
    f = np.clip(frames["obj"].astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
    r = scene_a.detect(f); assert len(r["objects"]) == 1 and abs(r["objects"][0]["x_in"] - NUT_IN[0]) * 25.4 < 6


def test_other_resolution_frames_are_scaled(scene_a, frames):
    big = cv2.resize(frames["obj"], (1280, 720), interpolation=cv2.INTER_CUBIC)
    r = scene_a.detect(big)
    assert len(r["objects"]) == 1 and dist_mm(r["objects"][0], NUT_IN[0] * 25.4, NUT_IN[1] * 25.4) < 6
    assert abs(r["objects"][0]["px"][0] - 2 * NUT_PX[0]) < 10                      # raw pixel reported in the frame's own resolution


def test_background_from_several_frames_is_a_median(sd, calib_a, frames):
    """Three frames, the nut and a synthetic square each in only one of them: the median drops both. (The arm must be
    somewhere different in most frames, or it gets baked in: documented in docs/mat-scene.md.)"""
    third = paste_square(sd.MatScene(calib_a), frames["bg"], 150, 250, 30)
    s = sd.MatScene(calib_a); s.set_background([frames["bg"], frames["obj"], third])
    assert s.detect(frames["bg"])["objects"] == []
    assert len(s.detect(third)["objects"]) == 1
    r = s.detect(frames["obj"]); assert len(r["objects"]) == 1 and dist_mm(r["objects"][0], NUT_IN[0] * 25.4, NUT_IN[1] * 25.4) < 6


# ---------------------------------------------------------------- masks
def test_blob_on_a_marker_is_ignored(scene_a, frames, calib_a):
    mx, my = calib_a["layout_markers"]["0"]
    on_marker = paste_square(scene_a, frames["bg"], mx, my, 30)
    assert scene_a.detect(on_marker)["objects"] == []
    control = paste_square(scene_a, frames["bg"], mx + 100, my, 30)              # same square, same row, off the marker
    assert len(scene_a.detect(control)["objects"]) == 1


def test_specks_are_rejected(scene_a, frames):
    assert scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 4))["objects"] == []       # 16 mm2
    assert len(scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 10))["objects"]) == 1  # 100 mm2


def test_big_blob_from_a_side_edge_is_an_intrusion_not_an_object(scene_a, frames):
    f = paste_square(scene_a, frames["bg"], 30, 150, 80)                           # spans x -10..70: a hand from the left
    r = scene_a.detect(f); assert r["objects"] == []
    f = paste_square(scene_a, frames["bg"], 228, 15, 40)                           # touching the arm edge: arm
    r = scene_a.detect(f); assert r["objects"] == [] and r["arm_in_view"]


def test_object_touching_the_arm_is_absorbed(scene_a, frames):
    f = paste_square(scene_a, frames["obj"], 230, 55, 25)                          # right against the arm blob in obj.jpg
    r = scene_a.detect(f)
    assert len(r["objects"]) == 1 and dist_mm(r["objects"][0], NUT_IN[0] * 25.4, NUT_IN[1] * 25.4) < 6   # only the nut survives


# ---------------------------------------------------------------- synthetic objects: geometry
@pytest.mark.parametrize("x,y", [(100, 100), (228.6, 152.4), (23, 150), (434, 150), (228.6, 23), (150, 282), (23, 282), (434, 282), (300, 40)])
def test_synthetic_square_position(scene_a, frames, x, y):
    """30 mm squares across the mat, down to 8 mm from the edges (anything within ~5 mm of an edge merges with it: the
    2 mm border margin plus one pixel of interpolation spill) and clear of the marker squares (+10 mm margin)."""
    r = scene_a.detect(paste_square(scene_a, frames["bg"], x, y, 30, color=(40, 60, 230)))
    assert len(r["objects"]) == 1, r["objects"]
    o = r["objects"][0]
    assert dist_mm(o, x, y) < 3, o
    assert abs(o["w_mm"] - 30) < 8 and abs(o["h_mm"] - 30) < 8 and 500 < o["area_mm2"] < 1400
    assert o["color"][0] > o["color"][1] + 60 and o["color"][0] > o["color"][2] + 60            # red square
    u, v = scene_a.mm_to_px(x, y); assert np.hypot(o["px"][0] - u, o["px"][1] - v) < 3
    assert abs(o["angle_deg"]) % 90 < 12 or abs(o["angle_deg"]) % 90 > 78
    assert o["bbox_mm"][0] < x < o["bbox_mm"][0] + o["bbox_mm"][2] and o["bbox_mm"][1] < y < o["bbox_mm"][1] + o["bbox_mm"][3]


def test_synthetic_square_angle(scene_a, frames):
    o = scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 40, angle_deg=30))["objects"][0]
    assert min(abs(o["angle_deg"] - a) for a in (30, -60, 120)) < 10


def test_two_objects(scene_a, frames):
    f = paste_square(scene_a, frames["bg"], 120, 120, 30, color=(40, 60, 230))
    f = paste_square(scene_a, f, 300, 240, 30, color=(230, 60, 40))
    r = scene_a.detect(f); assert len(r["objects"]) == 2
    by_x = sorted(r["objects"], key=lambda o: o["x_mm"])
    assert dist_mm(by_x[0], 120, 120) < 3 and dist_mm(by_x[1], 300, 240) < 3
    assert by_x[0]["color"][0] > 150 and by_x[1]["color"][2] > 150


# ---------------------------------------------------------------- tracking
def test_ids_stable_for_small_moves_new_for_jumps(scene_a, frames):
    a = scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 30), t=1)["objects"][0]
    b = scene_a.detect(paste_square(scene_a, frames["bg"], 205, 150, 30), t=2)["objects"][0]
    assert b["id"] == a["id"] and abs(b["x_mm"] - 205) < 3
    c = scene_a.detect(paste_square(scene_a, frames["bg"], 305, 150, 30), t=3)["objects"][0]
    assert c["id"] != a["id"]


def test_id_survives_a_missed_frame_then_expires(scene_a, frames):
    a = scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 30), t=1)["objects"][0]
    assert scene_a.detect(frames["bg"], t=2)["objects"] == []
    b = scene_a.detect(paste_square(scene_a, frames["bg"], 202, 150, 30), t=3)["objects"][0]
    assert b["id"] == a["id"]
    for t in range(4, 10): scene_a.detect(frames["bg"], t=t)
    c = scene_a.detect(paste_square(scene_a, frames["bg"], 200, 150, 30), t=10)["objects"][0]
    assert c["id"] != a["id"]


def test_ids_unique_within_a_frame(scene_a, frames):
    f = paste_square(scene_a, frames["bg"], 120, 120, 30); f = paste_square(scene_a, f, 340, 220, 30)
    ids = [o["id"] for o in scene_a.detect(f)["objects"]]; assert len(ids) == len(set(ids)) == 2


# ---------------------------------------------------------------- background handling
def test_background_missing(sd, calib_a, frames):
    s = sd.MatScene(calib_a)
    r = s.detect(frames["obj"])
    assert r["objects"] == [] and r["background"] == "missing" and r["arm_in_view"] is False
    assert r["n_markers_seen"] == 4                                                # markers are still checked
    s.update_background(frames["obj"])                                             # no-op, no crash
    with pytest.raises(ValueError): s.save_background("/nonexistent/x.png")


def test_background_save_load(sd, calib_a, scene_a, frames, tmp_path):
    p = tmp_path / "bg.png"; scene_a.save_background(p); assert p.exists()
    s = sd.MatScene(calib_a); s.load_background(p)
    r = s.detect(frames["obj"]); assert len(r["objects"]) == 1 and dist_mm(r["objects"][0], NUT_IN[0] * 25.4, NUT_IN[1] * 25.4) < 6
    with pytest.raises(OSError): s.load_background(tmp_path / "missing.png")


def test_set_background_accepts_a_topdown_image(sd, calib_a, scene_a, frames):
    s = sd.MatScene(calib_a); s.set_background(scene_a.warp(frames["bg"]))
    assert len(s.detect(frames["obj"])["objects"]) == 1


def test_update_background_keeps_objects_and_tracks_light(scene_a, frames):
    f = paste_square(scene_a, frames["bg"], 200, 150, 30)
    for _ in range(10): scene_a.update_background(f, alpha=0.5)
    assert len(scene_a.detect(f)["objects"]) == 1                                  # object pixels are never absorbed
    bright = np.clip(frames["bg"].astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    before = float(scene_a.bg.mean())
    for _ in range(5): scene_a.update_background(bright, alpha=0.5)
    assert scene_a.bg.mean() > before * 1.1                                         # the free mat followed the light
    assert len(scene_a.detect(f)["objects"]) == 1


def test_update_background_alpha_busy_absorbs_eventually(scene_a, frames):
    f = paste_square(scene_a, frames["bg"], 200, 150, 30)
    assert len(scene_a.detect(f)["objects"]) == 1
    for _ in range(12): scene_a.update_background(f, alpha=0.5, alpha_busy=0.5)
    assert scene_a.detect(f)["objects"] == []


# ---------------------------------------------------------------- calibration drift: markers re-fit the homography
def shifted(calib, dx_mm, dy_mm):
    c = json.loads(json.dumps(calib)); T = np.array([[1, 0, dx_mm], [0, 1, dy_mm], [0, 0, 1.0]])
    c["homography_px_to_table"] = (T @ np.array(calib["homography_px_to_table"])).tolist(); return c


def test_refit_recovers_from_a_moved_mat(sd, calib_b, frames):
    s = sd.MatScene(shifted(calib_b, 15, -8)); s.set_background(frames["new"])
    r = s.detect(frames["new"])
    assert r["refitted"] and r["calib_err_mm"] < 1.5 and r["calib_ok"] and r["objects"] == []
    assert 12 < r["mat_shift_mm"] < 22 and abs(r["mat_rot_deg"]) < 2                 # the 15/-8 mm shift we injected, seen as a mat move
    H_true = np.array(calib_b["homography_px_to_table"])
    for x, y in [(50, 50), (400, 50), (228.6, 152.4), (50, 280), (400, 280)]:         # the re-fit agrees with the real calibration across the mat
        u, v = sd._h_apply(np.linalg.inv(H_true), [(x, y)])[0]
        assert np.hypot(*(np.array(s.px_to_mm(u, v)) - (x, y))) < 2.0
    s2 = sd.MatScene(shifted(calib_b, 15, -8), refit=False); s2.set_background(frames["new"])
    r2 = s2.detect(frames["new"]); assert not r2["refitted"] and r2["calib_err_mm"] > 10 and not r2["calib_ok"]


def test_refit_keeps_the_nut_in_place(sd, calib_a, frames):
    s = sd.MatScene(shifted(calib_a, -12, 9)); s.set_background(frames["bg"])
    r = s.detect(frames["obj"]); assert len(r["objects"]) == 1 and dist_mm(r["objects"][0], NUT_IN[0] * 25.4, NUT_IN[1] * 25.4) < 6


def test_reglued_markers_are_flagged_as_a_big_move(sd, calib_a, frames):
    """new.jpg: the mat turned 180 deg and the markers re-glued in a new layout. The four centres still fit calib A's
    layout as a rigid move (the layouts are congruent), so the re-fit cannot tell; what it can do is report that the
    mat is nowhere near where it was calibrated, and clear calib_ok."""
    s = sd.MatScene(calib_a); s.set_background(frames["bg"])
    r = s.detect(frames["new"])
    assert r["n_markers_seen"] == 4 and r["refitted"] and not r["calib_ok"] and "re-run" in r["calib_note"]
    assert r["mat_shift_mm"] > 40 or abs(r["mat_rot_deg"]) > 150
    s2 = sd.MatScene(calib_a, refit=False); s2.set_background(frames["bg"])
    r2 = s2.detect(frames["new"]); assert r2["calib_err_mm"] > 200 and not r2["calib_ok"] and r2["mat_shift_mm"] == 0


def test_homography_from_markers_matches_calibration(sd, calib_a, frames):
    gray = cv2.cvtColor(frames["bg"], cv2.COLOR_BGR2GRAY)
    seen = sd.detect_markers(gray, calib_a["dictionary"])
    lay = {int(k): v for k, v in calib_a["layout_markers"].items()}
    H = sd.homography_from_markers(seen, lay, calib_a["marker_size_mm"])
    for i, c in seen.items():
        p = sd._h_apply(H, c.mean(0))[0]; assert np.hypot(*(p - lay[i])) < 1.0
    assert sd.homography_from_markers({k: v for k, v in list(seen.items())[:2]}, lay, 50.8) is None


# ---------------------------------------------------------------- speed and CLI
def test_detect_is_fast(scene_a, frames):
    scene_a.detect(frames["obj"])
    ts = []
    for _ in range(10):
        t0 = time.perf_counter(); scene_a.detect(frames["obj"]); ts.append(time.perf_counter() - t0)
    assert sorted(ts)[len(ts) // 2] * 1e3 < 150, ts


def test_render(scene_a, frames):
    img = scene_a.render(scale=2); assert img.shape == (scene_a.Hh * 2, scene_a.W * 2, 3)
    scene_a.detect(frames["obj"]); img = scene_a.render(scale=3); assert img.shape[0] == scene_a.Hh * 3


def test_cli(sd, tmp_path, capsys):
    out = tmp_path / "warp.png"
    rc = sd.main(["--calib", os.path.join(FIX, "calib_a.json"), "--bg", os.path.join(FIX, "bg.jpg"), os.path.join(FIX, "obj.jpg"), os.path.join(FIX, "bg.jpg"),
                  "--save-warp", str(out), "--save-bg", str(tmp_path / "bg.png")])
    assert rc == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2 and lines[0]["frame"] == "obj.jpg" and len(lines[0]["objects"]) == 1 and lines[1]["objects"] == []
    assert lines[0]["arm_in_view"] and "ms" in lines[0]
    assert (tmp_path / "warp_0.png").exists() and (tmp_path / "warp_1.png").exists() and (tmp_path / "bg.png").exists()
    rc = sd.main(["--calib", os.path.join(FIX, "calib_a.json"), "--bg", str(tmp_path / "bg.png"), os.path.join(FIX, "obj.jpg")])   # saved PNG as background
    assert rc == 0 and len(json.loads(capsys.readouterr().out.strip())["objects"]) == 1
    assert sd.main(["--calib", os.path.join(FIX, "calib_a.json"), str(tmp_path / "nope.jpg")]) == 2


def test_min_hits_hides_one_frame_objects(sd, calib_a, frames):
    """min_hits=3: an object is reported only once it has persisted for three detects (a one-frame blob never shows), the
    waiting count is exposed as `pending`, and it drops out after the usual miss budget once it is gone. min_hits=1 (default) is unchanged."""
    s = sd.MatScene(calib_a, min_hits=3); s.set_background(frames["bg"])
    img = paste_square(s, frames["bg"], 200, 150, 30)
    r1, r2, r3 = s.detect(img), s.detect(img), s.detect(img)
    assert r1["objects"] == [] and r1["pending"] == 1 and r2["objects"] == [] and r2["pending"] == 1
    assert len(r3["objects"]) == 1 and r3["pending"] == 0 and r3["objects"][0]["id"] == 1        # same track id all along
    for _ in range(s.track_miss + 1): r = s.detect(frames["bg"])
    assert r["objects"] == [] and s._tracks == {}
    one = sd.MatScene(calib_a); one.set_background(frames["bg"])
    assert len(one.detect(img)["objects"]) == 1 and one.detect(img)["pending"] == 0
