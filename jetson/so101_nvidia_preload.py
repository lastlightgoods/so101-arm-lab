"""Jetson site-packages preload: dlopen the two NVIDIA libs the JetPack torch wheel needs (libcudss, libcusparseLt) at interpreter start.
Not run by hand: jetson_setup.sh copies it into site-packages next to a so101_nvidia_preload.pth that imports it.
# why: default (RTLD_LOCAL) mode on purpose, RTLD_GLOBAL made `import cv2` abort with free(): invalid pointer."""
import ctypes, glob, os
_sp = os.path.dirname(__file__)
for _pat in ("nvidia/cu12/lib/libcudss.so.0", "nvidia/cusparselt/lib/libcusparseLt.so.0"):
    for _p in glob.glob(os.path.join(_sp, _pat)):
        try: ctypes.CDLL(_p)
        except OSError: pass
