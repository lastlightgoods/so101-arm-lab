"""python -m lerobot_camera_avf  -> list video devices (does not open any camera)."""
from .avf import list_devices

for d in list_devices():
    flag = "" if d["allowed"] else "   (FORBIDDEN, never opened)"
    print(f"{d['name']:28s} native {d['native']:10s} uid {d['unique_id']}{flag}")
