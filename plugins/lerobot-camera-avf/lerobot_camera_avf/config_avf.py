"""AVFCameraConfig: the LeRobot camera config for the `avf` camera type (AVFoundation by device name on macOS).
Imported by lerobot_camera_avf/__init__.py; selected from the CLI with --robot.cameras='{top: {type: avf, match: icspring, ...}}'."""
from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode

# Device names that must never be opened, whatever the user asks for.
FORBIDDEN_NAME_PARTS = ("facetime", "iphone", "ipad", "continuity", "desk view")


@CameraConfig.register_subclass("avf")
@dataclass
class AVFCameraConfig(CameraConfig):
    """macOS AVFoundation camera selected by device NAME, never by numeric index.

    Camera indices on macOS are not stable (they reshuffle between enumerations), so this backend
    picks the AVCaptureDevice object whose localizedName contains `match`. If several devices match
    (e.g. two identical USB cams), disambiguate with `native` ("1920x1080") = the device's largest
    supported format, or `unique_id`. Built-in and iPhone cameras are refused unconditionally.

    Example (lerobot-record):
        --robot.cameras="{top: {type: avf, match: icspring, native: 1920x1080, width: 640, height: 360, fps: 30},
                          wrist: {type: avf, match: icspring, native: 640x480, width: 640, height: 480, fps: 30}}"
    """

    match: str = "icspring"
    native: str | None = None      # "WxH" of the device's largest format, to tell identical cams apart
    unique_id: str | None = None   # AVCaptureDevice.uniqueID, exact match (changes if re-plugged)
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: float = 1.0

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        low = self.match.lower()
        if not low or any(p in low for p in FORBIDDEN_NAME_PARTS):
            raise ValueError(f"`match`={self.match!r} is empty or names a forbidden camera")
        if self.native is not None:
            w, h = self.native.lower().split("x")
            self.native = f"{int(w)}x{int(h)}"
        if self.width is None or self.height is None:
            raise ValueError("avf camera needs explicit `width` and `height` (frames are resized to them)")
