"""LeRobot plugin: `type: avf` cameras selected by name on macOS. Importing registers the config."""
from .avf import AVFCamera, list_devices
from .config_avf import AVFCameraConfig

__all__ = ["AVFCamera", "AVFCameraConfig", "list_devices"]
