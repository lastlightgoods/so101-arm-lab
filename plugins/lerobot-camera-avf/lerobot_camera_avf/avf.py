"""AVFoundation camera for LeRobot that selects devices by name and native format, never by index."""
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np
import objc
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_avf import FORBIDDEN_NAME_PARTS, AVFCameraConfig

logger = logging.getLogger(__name__)


def _avf():
    import AVFoundation as AVF  # noqa: N812
    import CoreMedia
    import Quartz
    import dispatch
    from Foundation import NSObject

    return AVF, CoreMedia, Quartz, dispatch, NSObject


def _native_size(device) -> tuple[int, int]:
    """Largest video format the device offers, without opening it."""
    _, CoreMedia, _, _, _ = _avf()
    best = (0, 0)
    for f in device.formats():
        dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
        if dims.width * dims.height > best[0] * best[1]:
            best = (int(dims.width), int(dims.height))
    return best


def list_devices() -> list[dict[str, Any]]:
    AVF, *_ = _avf()
    out = []
    for d in AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo):
        name = str(d.localizedName())
        w, h = _native_size(d)
        out.append({
            "name": name,
            "unique_id": str(d.uniqueID()),
            "native": f"{w}x{h}",
            "allowed": not any(p in name.lower() for p in FORBIDDEN_NAME_PARTS),
        })
    return out


class _Sink:
    """Built lazily so importing this module never touches AVFoundation."""

    _cls = None

    @classmethod
    def make(cls, owner):
        if cls._cls is None:
            _, CoreMedia, Quartz, _, NSObject = _avf()

            class Sink(NSObject):
                def initWithOwner_(self, owner):
                    self = objc.super(Sink, self).init()
                    self.owner = owner
                    return self

                def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sbuf, conn):
                    try:
                        pb = CoreMedia.CMSampleBufferGetImageBuffer(sbuf)
                        Quartz.CVPixelBufferLockBaseAddress(pb, 1)
                        try:
                            w = Quartz.CVPixelBufferGetWidth(pb)
                            h = Quartz.CVPixelBufferGetHeight(pb)
                            stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
                            buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(stride * h)
                            bgr = np.frombuffer(buf, np.uint8).reshape(h, stride // 4, 4)[:, :w, :3].copy()
                        finally:
                            Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
                        self.owner._on_frame(bgr)
                    except Exception as e:  # never let an exception cross into ObjC
                        self.owner._on_error(e)

            cls._cls = Sink
        return cls._cls.alloc().initWithOwner_(owner)


class AVFCamera(Camera):
    def __init__(self, config: AVFCameraConfig):
        super().__init__(config)
        self.config = config
        self.color_mode = config.color_mode
        self.session = None
        self.device = None
        self.sink = None
        self.queue = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self._errors = 0

    def __str__(self) -> str:
        who = self.config.unique_id or f"{self.config.match}/{self.config.native or '*'}"
        return f"AVFCamera({who})"

    @property
    def is_connected(self) -> bool:
        if getattr(self, "_reconnecting", False):
            return True   # a reconnect is in flight; readers get the last frame meanwhile
        return self.session is not None and bool(self.session.isRunning())

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return list_devices()

    # -- selection -------------------------------------------------------------------------------
    def _select_device(self):
        AVF, *_ = _avf()
        cfg = self.config
        cands = []
        for d in AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo):
            name = str(d.localizedName()).lower()
            if any(p in name for p in FORBIDDEN_NAME_PARTS):
                continue
            if cfg.match.lower() not in name:
                continue
            if cfg.unique_id and str(d.uniqueID()) != cfg.unique_id:
                continue
            if cfg.native:
                w, h = _native_size(d)
                if f"{w}x{h}" != cfg.native:
                    continue
            cands.append(d)
        if not cands:
            raise DeviceNotConnectedError(f"{self}: no allowed camera matches. Devices: {list_devices()}")
        if len(cands) > 1:
            raise ValueError(
                f"{self}: {len(cands)} cameras match; add `native` or `unique_id` to pick one. "
                f"Candidates: {[(str(d.localizedName()), str(d.uniqueID()), _native_size(d)) for d in cands]}"
            )
        d = cands[0]
        assert not any(p in str(d.localizedName()).lower() for p in FORBIDDEN_NAME_PARTS)
        return d

    # -- lifecycle -------------------------------------------------------------------------------
    def connect(self, warmup: bool = True) -> None:
        if self.session is not None and bool(self.session.isRunning()):
            raise DeviceAlreadyConnectedError(f"{self} already connected.")
        AVF, _, Quartz, dispatch, _ = _avf()
        self.device = self._select_device()
        logger.info(f"{self}: using '{self.device.localizedName()}' uid={self.device.uniqueID()} native={_native_size(self.device)}")
        session = AVF.AVCaptureSession.alloc().init()
        for preset in self._presets(AVF):
            if self.device.supportsAVCaptureSessionPreset_(preset):
                session.setSessionPreset_(preset)
                break
        inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(self.device, None)
        if inp is None:
            raise DeviceNotConnectedError(f"{self}: cannot open input: {err}")
        session.addInput_(inp)
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        out.setAlwaysDiscardsLateVideoFrames_(True)
        self.sink = _Sink.make(self)
        self.queue = dispatch.dispatch_queue_create(b"lerobot.avf", None)
        out.setSampleBufferDelegate_queue_(self.sink, self.queue)
        session.addOutput_(out)
        self._apply_fps()
        session.startRunning()
        self.session = session
        deadline = time.time() + 5.0
        t_start = time.perf_counter()
        while (self.latest_timestamp is None or self.latest_timestamp < t_start) and time.time() < deadline:
            time.sleep(0.02)
        if self.latest_timestamp is None or self.latest_timestamp < t_start:
            self.disconnect()
            raise DeviceNotConnectedError(f"{self}: no frames within 5 s")
        if warmup:
            time.sleep(self.config.warmup_s)
        logger.info(f"{self} connected.")

    def _presets(self, AVF):  # noqa: N803
        wanted = (self.width or 0, self.height or 0)
        table = [
            ((1920, 1080), AVF.AVCaptureSessionPreset1920x1080),
            ((1280, 720), AVF.AVCaptureSessionPreset1280x720),
            ((640, 480), AVF.AVCaptureSessionPreset640x480),
        ]
        exact = [p for size, p in table if size == wanted]
        # smallest standard preset that still covers the requested size, so we never decode 1080p to make 640x360
        ar = wanted[0] / wanted[1] if wanted[1] else 0
        covering = [p for size, p in sorted(table, key=lambda sp: sp[0][0] * sp[0][1])
                    if size[0] >= wanted[0] and size[1] >= wanted[1] and abs(size[0] / size[1] - ar) < 0.05]  # same aspect: keep the training field of view
        return exact + covering + [AVF.AVCaptureSessionPresetHigh]

    def _apply_fps(self) -> None:
        if not self.fps:
            return
        try:
            ranges = list(self.device.activeFormat().videoSupportedFrameRateRanges())
            if not ranges:
                return
            best = min(ranges, key=lambda r: abs(float(r.maxFrameRate()) - float(self.fps)))
            if abs(float(best.maxFrameRate()) - float(self.fps)) > 0.5:
                logger.warning(f"{self}: fps={self.fps} not offered; using {float(best.maxFrameRate()):.0f}")
            ok, _ = self.device.lockForConfiguration_(None)
            if ok:
                self.device.setActiveVideoMinFrameDuration_(best.minFrameDuration())
                self.device.setActiveVideoMaxFrameDuration_(best.maxFrameDuration())
                self.device.unlockForConfiguration()
        except Exception as e:
            logger.warning(f"{self}: could not set fps={self.fps}: {str(e)[:120]}")

    def disconnect(self) -> None:
        if self.session is None:
            raise DeviceNotConnectedError(f"{self} not connected.")
        try:
            self.session.stopRunning()
        finally:
            self.session = None
            self.device = None
            with self.frame_lock:
                self.latest_frame = None
                self.latest_timestamp = None
            self.new_frame_event.clear()
        logger.info(f"{self} disconnected.")

    # -- frames ----------------------------------------------------------------------------------
    def _on_frame(self, bgr: NDArray[Any]) -> None:
        if (self.width, self.height) != (bgr.shape[1], bgr.shape[0]):
            bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if self.color_mode == ColorMode.RGB else bgr
        with self.frame_lock:
            self.latest_frame = img
            self.latest_timestamp = time.perf_counter()
        self.new_frame_event.set()
        self._errors = 0

    def _on_error(self, e: Exception) -> None:
        self._errors += 1
        if self._errors <= 3:
            logger.warning(f"{self}: frame error: {e!r}")

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        frame = self.async_read(timeout_ms=1000)
        if color_mode is not None and color_mode != self.color_mode:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR if self.color_mode == ColorMode.RGB else cv2.COLOR_BGR2RGB)
        return frame

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self}: no new frame within {timeout_ms} ms")
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"{self}: frame event set but no frame")
        return frame

    STALE_RECONNECT_MS = 1500     # frame older than this: kick off a reconnect in the background
    STALE_FATAL_MS = 15000        # frame older than this: give up and raise

    def _reconnect_async(self) -> None:
        """Rebuild the capture session on a worker thread; readers keep getting the last frame meanwhile."""
        if getattr(self, "_reconnecting", False):
            return
        self._reconnecting = True

        def work():
            try:
                logger.warning(f"{self}: no frames for >{self.STALE_RECONNECT_MS} ms, reconnecting camera")
                try:
                    if self.session is not None:
                        self.session.stopRunning()
                except Exception:  # nosec B110
                    pass
                self.session = None
                for attempt in range(1, 30):
                    try:
                        keep = self.latest_frame, self.latest_timestamp
                        self.connect(warmup=False)
                        logger.warning(f"{self}: camera reconnected (attempt {attempt})")
                        return
                    except Exception as e:
                        with self.frame_lock:
                            self.latest_frame, self.latest_timestamp = keep
                        if attempt in (1, 5, 10, 20):
                            logger.warning(f"{self}: reconnect attempt {attempt} failed: {str(e)[:80]}")
                        time.sleep(0.5)
                logger.error(f"{self}: could not reconnect camera")
            finally:
                self._reconnecting = False

        threading.Thread(target=work, name=f"{self}-reconnect", daemon=True).start()

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.session is None and not getattr(self, "_reconnecting", False):
            raise DeviceNotConnectedError(f"{self} not connected.")
        with self.frame_lock:
            frame, ts = self.latest_frame, self.latest_timestamp
        if frame is None or ts is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age = (time.perf_counter() - ts) * 1e3
        if age > self.STALE_FATAL_MS:
            raise TimeoutError(f"{self} latest frame is too old: {age:.0f} ms and reconnect failed")
        if age > self.STALE_RECONNECT_MS:
            self._reconnect_async()
        if age > max_age_ms:
            n = getattr(self, "_stale_count", 0) + 1; self._stale_count = n
            if n in (1, 30, 300) or n % 1000 == 0:
                logger.warning(f"{self}: serving a stale frame ({age:.0f} ms old, #{n}); recording continues")
        return frame
