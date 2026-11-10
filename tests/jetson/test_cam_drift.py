"""jetson/cam_drift.py without cameras: shifted / rotated / blank copies of a real overhead frame
(tests/scene/fixtures/new.jpg, whose calibration is calib_b.json), the marker-displacement path against that
calibration, the moved-camera / moved-mat / knocked-marker verdicts, and the set/check CLI through a folder
source and through a tiny local http server standing in for the dashboard's /cam/<name>.jpg."""
import http.server, importlib.util, json, math, os, shutil, threading
import cv2, numpy as np
import pytest
from jhelpers import JETSON

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(ROOT, "tests", "scene", "fixtures")


@pytest.fixture(scope="session")
def cd():
    spec = importlib.util.spec_from_file_location("cam_drift", os.path.join(JETSON, "cam_drift.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def ref():
    im = cv2.imread(os.path.join(FIX, "new.jpg"), cv2.IMREAD_COLOR); assert im is not None and im.shape == (360, 640, 3)
    return im


@pytest.fixture(scope="session")
def calib():
    return json.load(open(os.path.join(FIX, "calib_b.json")))


def shifted(im, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def rotated(im, deg, about=None, dx=0.0, dy=0.0):
    c = about or ((im.shape[1] - 1) / 2.0, (im.shape[0] - 1) / 2.0)
    M = cv2.getRotationMatrix2D(c, deg, 1.0); M[:, 2] += (dx, dy)
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- frame shift estimation
@pytest.mark.parametrize("dx,dy", [(3.4, -2.2), (-7.0, 5.5), (0.0, 0.0), (12.3, 0.4)])
def test_shift_recovered_within_half_pixel(cd, ref, dx, dy):
    s = cd.estimate_shift(ref, shifted(ref, dx, dy))
    assert abs(s["dx"] - dx) < 0.5 and abs(s["dy"] - dy) < 0.5, s
    assert s["reliable"] and s["response"] > 0.8 and abs(s["rot_deg"]) < 0.1


def test_shift_sign_convention(cd, ref):
    """Positive dx/dy = the scene content moved right / down in the new frame."""
    s = cd.estimate_shift(ref, shifted(ref, 9.0, 4.0))
    assert s["dx"] > 8 and s["dy"] > 3


def test_translation_only_path(cd, ref):
    s = cd.estimate_shift(ref, shifted(ref, 3.4, -2.2), rotation=False)
    assert s["method"] == "phase" and s["rot_deg"] is None and abs(s["dx"] - 3.4) < 0.5 and abs(s["dy"] + 2.2) < 0.5


@pytest.mark.parametrize("deg", [1.5, -2.0])
def test_rotation_recovered(cd, ref, deg):
    s = cd.estimate_shift(ref, rotated(ref, deg))
    assert s["rot_deg"] is not None and abs(s["rot_deg"] - deg) < 0.2, s
    assert s["method"] == "ecc" and s["reliable"]
    assert math.hypot(s["dx"], s["dy"]) < 0.7          # rotation about the centre: the centre does not move


def test_rotation_plus_shift(cd, ref):
    s = cd.estimate_shift(ref, rotated(ref, 1.0, dx=4.0, dy=-3.0))
    assert abs(s["rot_deg"] - 1.0) < 0.2 and abs(s["dx"] - 4.0) < 0.5 and abs(s["dy"] + 3.0) < 0.5


def test_no_rotation_reports_zero(cd, ref):
    s = cd.estimate_shift(ref, ref)
    assert abs(s["rot_deg"]) < 0.02 and math.hypot(s["dx"], s["dy"]) < 0.05 and s["response"] > 0.99


def test_blank_and_unrelated_frames_are_low_confidence(cd, ref):
    blank = np.zeros_like(ref)
    s = cd.estimate_shift(ref, blank)
    assert not s["reliable"] and s["response"] < 0.05 and s["rot_deg"] is None
    noise = np.random.default_rng(1).integers(0, 255, ref.shape, np.uint8)
    assert not cd.estimate_shift(ref, noise)["reliable"]
    other = cv2.imread(os.path.join(FIX, "top_now.jpg"))           # same camera, the mat turned round: nothing lines up
    assert not cd.estimate_shift(ref, other)["reliable"]


def test_different_size_frame_is_resized(cd, ref):
    big = cv2.resize(shifted(ref, 4.0, 2.0), (1280, 720))
    s = cd.estimate_shift(ref, big)
    assert s["reliable"] and abs(s["dx"] - 4.0) < 0.6 and abs(s["dy"] - 2.0) < 0.6


# ---------------------------------------------------------------- markers against the calibration
def test_markers_on_the_calibrated_frame_sit_where_projected(cd, ref, calib):
    m = cd.marker_check(ref, calib)
    assert m["expected"] == 4 and m["detected"] == 4 and m["missing"] == []
    assert m["max_norm"] < 0.6 and not m["moved"] and abs(m["mean_dx"]) < 0.3 and abs(m["mean_dy"]) < 0.3
    for mid, p in m["markers"].items():
        assert p["detected"] is not None and abs(p["norm"] - math.hypot(*p["disp"])) < 0.02


def test_marker_displacement_follows_a_shift(cd, ref, calib):
    m = cd.marker_check(shifted(ref, 8.0, -5.0), calib)
    assert m["detected"] == 4 and m["moved"]
    assert abs(m["mean_dx"] - 8.0) < 0.5 and abs(m["mean_dy"] + 5.0) < 0.5 and m["spread_px"] < 1.0


def test_marker_check_with_missing_markers(cd, ref, calib):
    half = ref.copy(); half[:, :320] = 128                              # paint over the left half: two markers gone
    m = cd.marker_check(half, calib)
    assert m["detected"] == 2 and len(m["missing"]) == 2 and m["moved"] is False
    m0 = cd.marker_check(np.zeros_like(ref), calib)
    assert m0["detected"] == 0 and m0["moved"] is None and m0["max_norm"] is None


def test_expected_pixels_use_the_inverse_homography(cd, calib):
    exp = cd.expected_marker_pixels(calib)
    H = np.array(calib["homography_px_to_table"])
    for mid, (u, v) in exp.items():
        p = H @ np.array([u, v, 1.0]); x, y = p[0] / p[2], p[1] / p[2]
        assert np.allclose([x, y], calib["layout_markers"][str(mid)], atol=1e-6)


# ---------------------------------------------------------------- verdicts
def judge(cd, ref, calib, new, **kw):
    s = cd.estimate_shift(ref, new); m = cd.marker_check(new, calib, kw.get("threshold", cd.THRESHOLD_PX))
    return cd.combine(s, m, size=(ref.shape[1], ref.shape[0]), **kw), s, m


def test_verdict_in_place(cd, ref, calib):
    v, _, _ = judge(cd, ref, calib, ref)
    assert v["ok"] and v["status"] == "OK" and v["authority"] == "markers"


def test_verdict_small_shift_is_still_ok(cd, ref, calib):
    v, _, _ = judge(cd, ref, calib, shifted(ref, 3.0, 2.0))
    assert v["ok"] and v["status"] == "OK" and not v["frame_moved"]


def test_verdict_camera_moved(cd, ref, calib):
    v, s, m = judge(cd, ref, calib, shifted(ref, 9.0, -4.0))
    assert not v["ok"] and v["status"] == "CAMERA MOVED" and v["frame_moved"] and m["moved"]
    assert v["marker_vs_frame_px"] < 1.5 and "so101-calib table" in v["verdict"]


def test_verdict_camera_rolled(cd, ref, calib):
    v, s, m = judge(cd, ref, calib, rotated(ref, 1.5))
    assert v["status"] == "CAMERA MOVED" and "roll" in v["verdict"] and abs(s["rot_deg"] - 1.5) < 0.2


def test_verdict_mat_moved(cd, ref, calib):
    """The markers moved (the calibration is stale) but the frame did not follow: fake it by keeping the frame and
    handing combine a calibration whose layout the detections no longer match."""
    H = np.array(calib["homography_px_to_table"]); T = np.array([[1, 0, -9.0], [0, 1, 6.0], [0, 0, 1.0]])   # expected pixels shifted by (+9, -6)
    stale = dict(calib); stale["homography_px_to_table"] = (H @ T).tolist()
    s = cd.estimate_shift(ref, ref); m = cd.marker_check(ref, stale)
    assert m["moved"] and abs(m["mean_dx"] + 9.0) < 0.6 and abs(m["mean_dy"] - 6.0) < 0.6
    v = cd.combine(s, m, size=(640, 360))
    assert not v["ok"] and v["status"] == "MAT MOVED" and v["marker_rigid_residual_px"] < 1.0 and "refit the arm" in v["verdict"]


def test_verdict_one_marker_knocked(cd, ref, calib):
    """Three markers in place, one displaced on its own: paste a shifted copy of its patch."""
    exp = cd.expected_marker_pixels(calib); u, v_ = exp[0]; u, v_ = int(round(u)), int(round(v_))
    new = ref.copy(); r = 34
    patch = ref[v_ - r:v_ + r, u - r:u + r].copy()
    new[v_ - r:v_ + r, u - r:u + r] = cv2.medianBlur(patch, 21)       # hide the original
    new[v_ - r + 9:v_ + r + 9, u - r + 12:u + r + 12] = patch           # and re-stick it 12 px right, 9 px down
    v, s, m = judge(cd, ref, calib, new)
    assert m["detected"] == 4 and m["markers"][0]["norm"] > 10 and max(m["markers"][i]["norm"] for i in (1, 2, 3)) < 1.0
    assert v["status"] == "MARKER MOVED" and "worst id 0" in v["verdict"]


def test_verdict_scene_changed_but_markers_in_place(cd, ref, calib):
    """A big frame shift that the markers do not share means the camera did not move: the markers win."""
    s = {"dx": 20.0, "dy": -11.0, "response": 0.6, "rot_deg": 0.0, "method": "ecc", "ecc_cc": 0.9, "reliable": True}
    m = cd.marker_check(ref, calib)
    v = cd.combine(s, m, size=(640, 360))
    assert v["ok"] and v["status"] == "OK" and v["frame_moved"] and "scene changes" in v["verdict"]


def test_verdict_frame_only_cameras(cd, ref):
    ok = cd.combine(cd.estimate_shift(ref, shifted(ref, 2.0, 1.0)), None)
    assert ok["ok"] and ok["status"] == "OK" and ok["authority"] == "frame"
    moved = cd.combine(cd.estimate_shift(ref, shifted(ref, 5.0, 5.0)), None)        # 7.07 px > 6
    assert not moved["ok"] and moved["status"] == "MOVED" and moved["shift_px"] > 6
    assert cd.combine(cd.estimate_shift(ref, shifted(ref, 5.0, 5.0)), None, threshold=8.0)["ok"]
    low = cd.combine(cd.estimate_shift(ref, np.zeros_like(ref)), None)
    assert not low["ok"] and low["status"] == "LOW CONFIDENCE"
    assert cd.combine(None, None)["status"] == "LOW CONFIDENCE"


def test_too_few_markers_falls_back_to_the_frame(cd, ref, calib):
    new = shifted(ref, 9.0, 0.0); new[:, :330] = 128                   # the two left markers gone, the right ones shifted
    v, s, m = judge(cd, ref, calib, new)
    assert m["detected"] <= 2
    if m["detected"] < cd.MIN_MARKERS: assert v["authority"] == "frame" and "marker_note" in v


# ---------------------------------------------------------------- set / check CLI through a folder source
@pytest.fixture
def rig(tmp_path, ref, calib):
    """A calib dir with the fixture calibration and a frames dir holding the reference scene for all three cameras."""
    cdir = tmp_path / "calib"; cdir.mkdir(); json.dump(calib, open(cdir / "top_table.json", "w"))
    frames = tmp_path / "frames"; frames.mkdir()
    for n in ("top", "front"): cv2.imwrite(str(frames / f"{n}.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(frames / "wrist.png"), cv2.resize(ref, (640, 480)))
    return {"calib": str(cdir), "frames": str(frames), "tmp": tmp_path}


def write_frames(folder, top=None, front=None, wrist=None):
    os.makedirs(folder, exist_ok=True)
    for n, im in (("top", top), ("front", front), ("wrist", wrist)):
        if im is not None: cv2.imwrite(os.path.join(folder, f"{n}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return folder


def test_set_writes_references_and_meta(cd, rig, capsys):
    code = cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]])
    out = capsys.readouterr().out
    assert code == 0
    for n in ("top", "front", "wrist"): assert os.path.exists(os.path.join(rig["calib"], f"ref_{n}.jpg"))
    meta = json.load(open(os.path.join(rig["calib"], "ref_meta.json")))
    assert meta["top"]["markers"]["detected"] == 4 and meta["top"]["markers"]["max_norm"] < 0.6 and meta["wrist"]["size"] == [640, 480]
    assert "markers 4/4 seen" in out


def test_set_json_and_missing_frame(cd, rig, capsys):
    os.remove(os.path.join(rig["frames"], "wrist.png"))
    code = cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"], "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 2 and rep["cameras"]["wrist"] == {"ok": False, "error": "no frame"} and rep["cameras"]["top"]["ok"]


def test_check_without_reference_is_exit_2(cd, rig, capsys):
    code = cd.main(["check", "--cams", "top", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]])
    assert code == 2 and "NO REFERENCE" in capsys.readouterr().out


def test_check_all_in_place(cd, rig, capsys):
    assert cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    capsys.readouterr()
    code = cd.main(["check", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]])
    out = capsys.readouterr().out
    assert code == 0 and "RESULT: all cameras in place" in out
    assert out.count(" OK ") == 3 and "markers 4/4" in out


def test_check_detects_a_moved_front_camera_and_reports_json(cd, rig, ref, capsys):
    assert cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    moved = write_frames(str(rig["tmp"] / "moved"), top=ref, front=shifted(ref, -8.0, 3.0), wrist=cv2.resize(ref, (640, 480)))
    capsys.readouterr()
    code = cd.main(["check", "--from-dir", moved, "--calib-dir", rig["calib"], "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 1 and not rep["ok"]
    by = {c["camera"]: c for c in rep["cameras"]}
    assert by["top"]["ok"] and by["wrist"]["ok"] and not by["front"]["ok"] and by["front"]["status"] == "MOVED"
    assert abs(by["front"]["shift"]["dx"] + 8.0) < 0.5 and abs(by["front"]["shift"]["dy"] - 3.0) < 0.5
    assert "markers" in by["top"] and "markers" not in by["front"]
    assert rep["calib"].endswith("top_table.json") and rep["threshold_px"] == 6.0


def test_check_camera_moved_verdict_end_to_end(cd, rig, ref, capsys):
    assert cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    moved = write_frames(str(rig["tmp"] / "moved"), top=shifted(ref, 9.0, -4.0))
    capsys.readouterr()
    code = cd.main(["check", "--cams", "top", "--from-dir", moved, "--calib-dir", rig["calib"]])
    out = capsys.readouterr().out
    assert code == 1 and "CAMERA MOVED" in out and "so101-calib table" in out


def test_check_black_or_white_frame_is_bad_exposure(cd, rig, ref, capsys):
    assert cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    blank = write_frames(str(rig["tmp"] / "blank"), top=np.zeros_like(ref))
    capsys.readouterr()
    code = cd.main(["check", "--cams", "top", "--from-dir", blank, "--calib-dir", rig["calib"]])
    assert code == 1 and "BAD EXPOSURE" in capsys.readouterr().out
    white = write_frames(str(rig["tmp"] / "white"), top=np.full_like(ref, 250))
    code = cd.main(["check", "--cams", "top", "--from-dir", white, "--calib-dir", rig["calib"], "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 1 and rep["cameras"][0]["status"] == "BAD EXPOSURE" and "blown out" in rep["cameras"][0]["verdict"] and "shift" not in rep["cameras"][0]


def test_check_unrelated_scene(cd, rig, ref, capsys):
    """top_now.jpg is the same camera with the mat turned round: a marker-less camera cannot match it (LOW CONFIDENCE);
    the top camera still finds the markers, hundreds of px from the calibration, and says so."""
    assert cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    other = write_frames(str(rig["tmp"] / "other"), top=cv2.imread(os.path.join(FIX, "top_now.jpg")), front=cv2.imread(os.path.join(FIX, "top_now.jpg")))
    capsys.readouterr()
    code = cd.main(["check", "--cams", "top,front", "--from-dir", other, "--calib-dir", rig["calib"], "--json"])
    rep = json.loads(capsys.readouterr().out); by = {c["camera"]: c for c in rep["cameras"]}
    assert code == 1 and by["front"]["status"] == "LOW CONFIDENCE"
    assert by["top"]["status"] in ("MAT MOVED", "MARKER MOVED") and by["top"]["markers"]["max_norm"] > 100


def test_frame_quality(cd, ref):
    assert cd.frame_quality(ref)["bad"] is None
    assert "blown out" in cd.frame_quality(np.full_like(ref, 245))["bad"]
    assert "black" in cd.frame_quality(np.full_like(ref, 5))["bad"]
    assert "flat" in cd.frame_quality(np.full_like(ref, 128))["bad"]


def test_set_refuses_a_blown_out_reference(cd, rig, ref, capsys):
    """A white reference would later match a white frame and pass: the set command must not save it."""
    assert cd.main(["set", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    before = open(os.path.join(rig["calib"], "ref_wrist.jpg"), "rb").read()
    white = write_frames(str(rig["tmp"] / "white"), top=ref, front=ref, wrist=np.full((480, 640, 3), 252, np.uint8))
    capsys.readouterr()
    code = cd.main(["set", "--from-dir", white, "--calib-dir", rig["calib"], "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 1 and not rep["cameras"]["wrist"]["ok"] and "blown out" in rep["cameras"]["wrist"]["error"] and rep["cameras"]["top"]["ok"]
    assert open(os.path.join(rig["calib"], "ref_wrist.jpg"), "rb").read() == before          # the old reference survives
    meta = json.load(open(os.path.join(rig["calib"], "ref_meta.json")))
    assert meta["wrist"]["exposure"]["bad"] is None and meta["top"]["exposure"]["mean"] > 50


def test_check_flags_a_bad_reference(cd, rig, ref, capsys):
    os.makedirs(rig["calib"], exist_ok=True)
    cv2.imwrite(os.path.join(rig["calib"], "ref_front.jpg"), np.full_like(ref, 250))
    code = cd.main(["check", "--cams", "front", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]])
    assert code == 1 and "BAD REFERENCE" in capsys.readouterr().out


def test_check_without_calibration_uses_the_frame_only(cd, rig, ref, capsys):
    os.remove(os.path.join(rig["calib"], "top_table.json"))
    assert cd.main(["set", "--cams", "top", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    moved = write_frames(str(rig["tmp"] / "moved"), top=shifted(ref, 9.0, -4.0))
    capsys.readouterr()
    code = cd.main(["check", "--cams", "top", "--from-dir", moved, "--calib-dir", rig["calib"], "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 1 and rep["calib"] is None and rep["cameras"][0]["status"] == "MOVED" and rep["cameras"][0]["authority"] == "frame"


def test_threshold_and_rotation_flags(cd, rig, ref, capsys):
    assert cd.main(["set", "--cams", "front", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 0
    moved = write_frames(str(rig["tmp"] / "moved"), front=shifted(ref, 5.0, 5.0))
    capsys.readouterr()
    assert cd.main(["check", "--cams", "front", "--from-dir", moved, "--calib-dir", rig["calib"]]) == 1
    assert cd.main(["check", "--cams", "front", "--from-dir", moved, "--calib-dir", rig["calib"], "--threshold", "8"]) == 0
    capsys.readouterr()
    assert cd.main(["check", "--cams", "front", "--from-dir", moved, "--calib-dir", rig["calib"], "--no-rotation", "--json"]) == 1
    rep = json.loads(capsys.readouterr().out); assert rep["cameras"][0]["shift"]["rot_deg"] is None


def test_unknown_camera_is_rejected(cd, rig):
    assert cd.main(["check", "--cams", "side", "--from-dir", rig["frames"], "--calib-dir", rig["calib"]]) == 2


# ---------------------------------------------------------------- the dashboard URL source
@pytest.fixture
def cam_server(tmp_path):
    """Serves /cam/<name>.jpg from a folder the test fills, like the Mac dashboard; 404 for anything else."""
    folder = tmp_path / "served"; folder.mkdir()
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k): super().__init__(*a, directory=str(folder), **k)
        def log_message(self, *a): pass
        def do_GET(self):
            path = self.path.split("?")[0]
            if not path.startswith("/cam/") or not path.endswith(".jpg"): self.send_response(404); self.end_headers(); return
            f = folder / path[5:]
            if not f.exists(): self.send_response(404); self.end_headers(); return
            data = f.read_bytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}", "folder": str(folder)}
    srv.shutdown(); srv.server_close()


def test_url_source_set_and_check(cd, rig, ref, cam_server, capsys):
    write_frames(cam_server["folder"], top=ref, front=ref, wrist=cv2.resize(ref, (640, 480)))
    assert cd.main(["set", "--url", cam_server["url"], "--calib-dir", rig["calib"]]) == 0
    meta = json.load(open(os.path.join(rig["calib"], "ref_meta.json")))
    assert meta["front"]["source"] == cam_server["url"] and meta["top"]["markers"]["detected"] == 4
    capsys.readouterr()
    assert cd.main(["check", "--url", cam_server["url"], "--calib-dir", rig["calib"]]) == 0
    assert "all cameras in place" in capsys.readouterr().out
    write_frames(cam_server["folder"], wrist=cv2.resize(shifted(ref, 6.0, 6.0), (640, 480)))   # the wrist moved
    code = cd.main(["check", "--url", cam_server["url"], "--calib-dir", rig["calib"], "--json"])
    rep = json.loads(capsys.readouterr().out); by = {c["camera"]: c for c in rep["cameras"]}
    assert code == 1 and by["wrist"]["status"] == "MOVED" and by["top"]["ok"] and by["front"]["ok"]


def test_url_source_missing_camera_is_exit_2(cd, rig, ref, cam_server, capsys):
    write_frames(cam_server["folder"], top=ref)
    assert cd.main(["set", "--cams", "top,front", "--url", cam_server["url"], "--calib-dir", rig["calib"], "--json"]) == 2
    rep = json.loads(capsys.readouterr().out)
    assert rep["cameras"]["top"]["ok"] and not rep["cameras"]["front"]["ok"]
    assert cd.main(["check", "--cams", "top,front", "--url", cam_server["url"], "--calib-dir", rig["calib"]]) == 2
    assert "NO REFERENCE" in capsys.readouterr().out


def test_url_fetch_handles_a_dead_server(cd):
    assert cd.fetch_url("http://127.0.0.1:9", "top", timeout=0.5) is None
