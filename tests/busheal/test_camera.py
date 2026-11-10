"""OpenCVCamera: stale frames are reused instead of ending the session; a dropped device is reopened."""
import threading
import types

import numpy as np
import pytest

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture
def cam(tmp_path, monkeypatch):
    # read_latest is guarded by @check_if_not_connected, which wants a real opened cv2.VideoCapture: fake the property
    monkeypatch.setattr(OpenCVCamera, "is_connected", property(lambda self: True))
    c = OpenCVCamera(OpenCVCameraConfig(index_or_path=str(tmp_path / "video-by-id"), fps=30, width=4, height=4))
    c.thread = types.SimpleNamespace(is_alive=lambda: True)
    c._postprocess_image = lambda f: f
    c._configure_capture_settings = lambda: None
    return c


def _frame_aged(cam, clock, age_s):
    cam.latest_frame = FRAME
    cam.latest_timestamp = clock.perf_counter() - age_s


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_read_latest_wrapped_and_stale_default(plugin):
    assert OpenCVCamera.read_latest._busheal and OpenCVCamera._read_loop._busheal
    # defined twice in the plugin (3 s, then 6 s alongside the reopen logic); the module global is what is used
    assert plugin.CAM_STALE_S == 6.0 and plugin.CAM_REOPEN_S == 20.0


def test_fresh_frame_passes_through(plugin, cam, clock):
    _frame_aged(cam, clock, 0.1)
    assert cam.read_latest() is FRAME
    assert not hasattr(cam, "_stale_count")


def test_stale_frame_is_reused_with_rate_limited_warning(plugin, cam, clock, busheal_log):
    _frame_aged(cam, clock, 0.7)
    assert cam.read_latest() is FRAME
    assert cam._stale_count == 1
    assert [m for m in _warnings(busheal_log) if "frame 700 ms old, reusing it (#1; raise after 6s)" in m]
    clock.advance(0.5)
    assert cam.read_latest() is FRAME
    assert cam._stale_count == 1, "warned at most once per 2 s"
    clock.advance(2.0)
    assert cam.read_latest() is FRAME
    assert cam._stale_count == 2


def test_frame_older_than_cam_stale_s_raises(plugin, cam, clock):
    _frame_aged(cam, clock, 6.5)
    with pytest.raises(TimeoutError, match="too old"):
        cam.read_latest()


def test_custom_max_age_only_decides_when_the_fallback_kicks_in(plugin, cam, clock):
    _frame_aged(cam, clock, 0.3)
    assert cam.read_latest(max_age_ms=400) is FRAME
    assert not hasattr(cam, "_stale_count")
    # 300 ms > 200 ms is "too old" for lerobot, but far below CAM_STALE_S: the frame is reused, with a warning
    assert cam.read_latest(max_age_ms=200) is FRAME
    assert cam._stale_count == 1


def test_no_frame_yet_errors_are_not_swallowed(plugin, cam, clock):
    cam.latest_frame = None
    with pytest.raises(RuntimeError, match="has not captured any frames yet"):
        cam.read_latest()
    cam.thread = types.SimpleNamespace(is_alive=lambda: False)
    with pytest.raises(RuntimeError, match="read thread is not running"):
        cam.read_latest()


# --- read loop with device reopen ---------------------------------------------------------------------------------

def _run_loop(cam, reads):
    """Feed `reads` (frames or exceptions) to the loop; stop once they are consumed."""
    it = iter(reads)
    cam.stop_event = threading.Event()

    def read():
        try:
            item = next(it)
        except StopIteration:
            cam.stop_event.set()
            raise DeviceNotConnectedError("done") from None
        if isinstance(item, Exception):
            raise item
        return item

    cam._read_from_hardware = read
    OpenCVCamera._read_loop(cam)


def test_read_loop_reopens_after_three_failures_and_continues(plugin, cam, clock, monkeypatch, busheal_log):
    calls = []
    monkeypatch.setattr(plugin, "_reopen", lambda self, ev: calls.append(self) or True)
    err = RuntimeError("read failed")
    _run_loop(cam, [FRAME, err, err, err, FRAME])
    assert calls == [cam]
    assert cam.latest_frame is FRAME and cam.latest_timestamp == clock.perf_counter()
    assert sum("consecutive read failures, reopening" in m for m in _warnings(busheal_log)) == 1


def test_read_loop_raises_when_device_does_not_come_back(plugin, cam, clock, monkeypatch):
    monkeypatch.setattr(plugin, "_reopen", lambda self, ev: False)
    err = RuntimeError("read failed")
    with pytest.raises(RuntimeError, match="did not come back within 20s"):
        _run_loop(cam, [err, err, err])


def test_read_loop_exits_quietly_when_stopped_during_reopen(plugin, cam, clock, monkeypatch):
    def stop_instead(self, ev):
        ev.set()
        return False

    monkeypatch.setattr(plugin, "_reopen", stop_instead)
    err = RuntimeError("read failed")
    _run_loop(cam, [err, err, err, FRAME])
    assert cam.latest_frame is None


class FakeVideoCapture:
    instances = []

    def __init__(self, path, backend, opened=True, frame_ok=True):
        self.path, self.backend, self.opened, self.frame_ok = path, backend, opened, frame_ok
        self.released = False
        FakeVideoCapture.instances.append(self)

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frame_ok, FRAME if self.frame_ok else None

    def release(self):
        self.released = True


def test_reopen_waits_for_the_device_node_then_reopens(plugin, cam, clock, monkeypatch, say_lines, tmp_path):
    FakeVideoCapture.instances = []
    monkeypatch.setattr(plugin.cv2, "VideoCapture", FakeVideoCapture)
    dead = FakeVideoCapture("old", 0)
    cam.videocapture = dead
    path = tmp_path / "video-by-id"
    # the by-id node re-enumerates after ~1.5 s of polling
    real_exists = plugin.os.path.exists

    def exists(p):
        if p == str(path) and clock.now - 1000.0 >= 1.5:
            path.write_bytes(b"")
        return real_exists(p)

    monkeypatch.setattr(plugin.os.path, "exists", exists)
    ok = plugin._reopen(cam, threading.Event())
    assert ok and dead.released
    assert cam.videocapture is FakeVideoCapture.instances[-1] and cam.videocapture.path == str(path)
    assert cam._reopen_count == 1
    assert [l["say"] for l in say_lines()] == ["camera reconnected"]
    assert 1.5 <= clock.now - 1000.0 <= 2.5


def test_reopen_gives_up_after_cam_reopen_s(plugin, cam, clock, monkeypatch):
    monkeypatch.setattr(plugin, "CAM_REOPEN_S", 2.0)
    monkeypatch.setattr(plugin.cv2, "VideoCapture", lambda *a: FakeVideoCapture(*a, opened=False))
    cam.index_or_path = 0      # an index: no device-node wait, every attempt tries and fails to open
    assert plugin._reopen(cam, threading.Event()) is False
    assert cam.videocapture is None and not hasattr(cam, "_reopen_count")
    assert clock.now - 1000.0 >= 2.0


def test_reopen_stops_when_asked(plugin, cam, clock, monkeypatch):
    ev = threading.Event()
    ev.set()
    assert plugin._reopen(cam, ev) is False and clock.sleeps == []
