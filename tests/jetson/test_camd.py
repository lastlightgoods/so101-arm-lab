"""jetson/camd.py (the camera owner) and jetson/demo_cams.py (the recorder's reader), against a fake ffmpeg: a python script that prints JPEGs. No V4L2 device, no ssh, no arm.
The point of the design: ONE process owns each device and fans the frames out, so the recorder is a reader and can never take a camera from the dashboard's feed."""
import importlib.util, os, shutil, socket, subprocess, sys, tempfile, threading, time
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JETSON = os.path.join(ROOT, "jetson")
PY = sys.executable
FAKE_FFMPEG = r'''
import sys, time, cv2, numpy as np
name, fps, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
h, w = (48, 64) if name == "wrist" else (36, 64)
for k in range(n):
    im = np.zeros((h, w, 3), np.uint8); im[..., 0] = (k * 7) % 256; im[:, : w // 2, 1] = 200
    ok, jpg = cv2.imencode(".jpg", im); sys.stdout.buffer.write(jpg.tobytes()); sys.stdout.buffer.flush(); time.sleep(1 / fps)
'''


def load_mod(name, path):
    sys.path.insert(0, JETSON)
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


@pytest.fixture(scope="module")
def camd():
    pytest.importorskip("cv2"); return load_mod("camd", os.path.join(JETSON, "camd.py"))


@pytest.fixture(scope="module")
def dcams(camd): return load_mod("demo_cams", os.path.join(JETSON, "demo_cams.py"))


@pytest.fixture
def sockdir():
    d = tempfile.mkdtemp(prefix="cd"); yield d   # why: a unix socket path is limited to ~100 characters, pytest's tmp_path is longer on macOS
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def ffscript(sockdir):
    p = os.path.join(sockdir, "ff.py"); open(p, "w").write(FAKE_FFMPEG); return p


@pytest.fixture
def server(camd, sockdir, ffscript):
    made = []
    def make(names=("top", "front", "wrist"), fps=30, frames=None, busy=None, missing=(), stale_s=5.0, retry_s=0.05):
        cmd = lambda n: None if n in missing else [PY, ffscript, n, str(fps)] + ([str(frames)] if frames else [])
        cams = {n: camd.Camera(n, cmd, busy=busy or os.path.join(sockdir, "busy"), retry_s=retry_s) for n in names}
        srv = camd.Server(cams, os.path.join(sockdir, "c.sock"), stale_s=stale_s).start(); made.append(srv); return srv
    yield make
    for s in made: s.stop()


def wait_until(fn, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v: return v
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_split_jpegs_finds_whole_frames_and_keeps_the_partial_one(camd):
    a, b = b"\xff\xd8AAA\xff\xd9", b"\xff\xd8BB\xff\xd9"
    assert camd.split_jpegs(a + b + b"\xff\xd8CC") == ([a, b], b"\xff\xd8CC") and camd.split_jpegs(b"junk") == ([], b"nk")


def test_two_readers_share_one_camera_owner(camd, dcams, server):
    srv = server(); src = dcams.CamdCameraSource(srv.path, shapes={"top": (36, 64, 3), "front": (36, 64, 3), "wrist": (48, 64, 3)}).start()
    try:
        wait_until(lambda: not src.missing()); s = camd.subscribe("top", 15, srv.path); it = camd.read_frames(s); got = [next(it) for _ in range(5)]   # the dashboard's tap and the recorder at the same time
        assert all(j.startswith(b"\xff\xd8") and j.endswith(b"\xff\xd9") for _, j in got) and [g[0] for g in got] == sorted(g[0] for g in got)
        gaps = np.diff([g[0] for g in got]); assert 0.04 < float(np.median(gaps)) < 0.12   # a 15 fps reader of the 30 fps camera
        img, tc = src.frame_at("top", time.monotonic()); assert isinstance(img, bytes) and 0 <= time.monotonic() - tc < 0.2
        st = srv.status(); assert st["cams"]["top"]["seq"] > 5 and st["cams"]["top"]["restarts"] == 0
    finally: src.close()


def test_frame_at_returns_the_newest_frame_captured_at_or_before_t(camd, dcams):
    src = dcams.CamdCameraSource("/nonexistent", shapes={"top": (36, 64, 3)}); src.ring["top"].extend([(10.0, b"a"), (10.1, b"b"), (10.2, b"c")])
    assert src.frame_at("top", 10.15) == (b"b", 10.1) and src.frame_at("top", 10.0) == (b"a", 10.0)
    img, tc = src.frame_at("top", 9.0); assert img.shape == (36, 64, 3) and not img.any() and 9.0 - tc >= 999   # nothing yet: a black frame stamped 1000 s old => camera_frozen => the episode is rejected


def test_a_missing_camera_is_reported_not_raised(dcams, server):
    srv = server(missing=("wrist",)); src = dcams.CamdCameraSource(srv.path, shapes={"top": (36, 64, 3), "front": (36, 64, 3), "wrist": (48, 64, 3)}).start()
    try:
        wait_until(lambda: src.ages()["top"] is not None and src.ages()["front"] is not None); time.sleep(0.4)
        assert src.missing() == ["wrist"] and src.ages()["wrist"] is None
    finally: src.close()


def test_no_camd_means_the_reader_waits_and_tap_exits_3(camd, dcams, sockdir):
    p = os.path.join(sockdir, "none.sock"); src = dcams.CamdCameraSource(p, shapes={"top": (36, 64, 3)}, retry_s=0.05).start()
    try:
        time.sleep(0.2); assert src.missing() == ["top"] and camd.tap("top", 15, p, out=open(os.devnull, "wb")) == 3
    finally: src.close()


def test_a_slow_reader_does_not_slow_the_camera_or_another_reader(camd, server):
    srv = server(names=("top",)); s_slow = camd.subscribe("top", 30, srv.path)   # connected and never read: its socket buffer fills
    s_fast = camd.subscribe("top", 30, srv.path); it = camd.read_frames(s_fast); t0 = time.time(); n = 0
    while time.time() - t0 < 1.0: next(it); n += 1
    assert n >= 20 and srv.cams["top"].restarts == 0   # ~30 fps kept flowing to the fast reader
    s_slow.close()


def test_cams_busy_makes_camd_yield_the_device_and_take_it_back(camd, server, sockdir):
    busy = os.path.join(sockdir, "busy"); srv = server(names=("top",), busy=busy); cam = srv.cams["top"]
    wait_until(lambda: cam.latest[0] > 3); open(busy, "w").close()
    wait_until(lambda: cam.proc is None or cam.proc.poll() is not None); seq = cam.latest[0]; time.sleep(0.5); assert cam.latest[0] <= seq + 1   # no new frames while a lerobot job / cam_lock owns the camera
    os.unlink(busy); wait_until(lambda: cam.latest[0] > seq + 5)


def test_a_reader_is_cut_off_when_its_camera_goes_silent(camd, server):
    srv = server(names=("top",), frames=3, stale_s=0.6, retry_s=60.0)   # the fake ffmpeg prints 3 frames and ends; camd keeps retrying it
    s = camd.subscribe("top", 30, srv.path); it = camd.read_frames(s); t0 = time.time(); got = list(it)
    assert len(got) >= 3 and time.time() - t0 < 6.0   # EOF => the dashboard's loop reconnects instead of showing a frozen picture


def run_dash_snippet(camd, sockdir, tap_rc):
    """The shell text the dashboard's remote command gains, run with a stub `python3` (exit code tap_rc) and a stub fallback."""
    bin_ = os.path.join(sockdir, "bin"); os.makedirs(bin_, exist_ok=True)
    open(os.path.join(bin_, "python3"), "w").write(f"#!/bin/sh\necho tap-ran\nexit {tap_rc}\n"); os.chmod(os.path.join(bin_, "python3"), 0o755)
    cmd = camd.DASH_PREFIX.format(name="top") + "echo ffmpeg-fallback"
    return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, env={"PATH": bin_ + ":/usr/bin:/bin", "HOME": sockdir}).stdout.split()


def test_dashboard_command_falls_back_to_its_own_ffmpeg_only_when_camd_is_absent(camd, sockdir):
    assert run_dash_snippet(camd, sockdir, 3) == ["tap-ran", "ffmpeg-fallback"]   # no camd: today's behaviour
    assert run_dash_snippet(camd, sockdir, 1) == ["tap-ran"]                       # camd was there and the stream ended: reconnect, never start a second owner of the device


def test_none_of_the_camera_side_code_can_move_the_arm():
    for f in ("camd.py", "demo_cams.py"):
        src = open(os.path.join(JETSON, f)).read()
        assert not any(w in src for w in ("send_action", "import lerobot", "from lerobot", "import serial", "motors", "so101_follower", "FOLLOWER", "ttyACM")), f
