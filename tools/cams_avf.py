"""Shop camera wall on monitor 2, selecting cameras by DEVICE OBJECT via AVFoundation.
Only devices whose name contains an ALLOWED string are ever started. The built-in FaceTime
camera and iPhone continuity camera are never touched. No numeric camera indices anywhere.
q in the window quits."""
import sys, time, threading, numpy as np, cv2, objc
import AVFoundation as AVF, Quartz, CoreMedia
from Foundation import NSObject
ALLOWED = ("icspring",)
import ctypes, ctypes.util
def second_display():
    cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
    class R(ctypes.Structure): _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("w", ctypes.c_double), ("h", ctypes.c_double)]
    cg.CGDisplayBounds.restype = R; cg.CGMainDisplayID.restype = ctypes.c_uint32
    ids = (ctypes.c_uint32 * 8)(); n = ctypes.c_uint32(); cg.CGGetActiveDisplayList(8, ids, ctypes.byref(n)); main = cg.CGMainDisplayID()
    ext = [cg.CGDisplayBounds(ids[i]) for i in range(n.value) if ids[i] != main]
    b = ext[0] if ext else cg.CGDisplayBounds(main)
    return (int(b.x), int(b.y)), (int(b.w), int(b.h))
ORIGIN, SIZE = second_display()
PORTRAIT = SIZE[1] > SIZE[0]
COLS = 1 if PORTRAIT else 2
TILE_W = SIZE[0] // COLS; TILE_H = int(TILE_W * 9 / 16)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
log("monitor 2:", ORIGIN, SIZE, "portrait" if PORTRAIT else "landscape", "tile", (TILE_W, TILE_H))
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
frames = {}; lock = threading.Lock()
class Sink(NSObject):
    def initWithKey_(self, key):
        self = objc.super(Sink, self).init(); self.key = key; return self
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sbuf, conn):
        try: grab_frame(self.key, sbuf)
        except Exception as e:
            if self.key not in warned: warned.add(self.key); log("frame error", self.key, repr(e))
warned = set()
def grab_frame(key, sbuf):
        pb = CoreMedia.CMSampleBufferGetImageBuffer(sbuf)
        Quartz.CVPixelBufferLockBaseAddress(pb, 1)
        try:
            w = Quartz.CVPixelBufferGetWidth(pb); h = Quartz.CVPixelBufferGetHeight(pb)
            stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
            base = Quartz.CVPixelBufferGetBaseAddress(pb)
            buf = base.as_buffer(stride * h)
            img = np.frombuffer(buf, np.uint8).reshape(h, stride // 4, 4)[:, :w, :3].copy()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
        with lock: frames[key] = img
devs = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
log("video devices:", [(d.localizedName(), d.uniqueID()) for d in devs])
chosen = [d for d in devs if any(a in d.localizedName().lower() for a in ALLOWED)]
if not chosen: log("no allowed cameras"); sys.exit(1)
for d in chosen: assert "facetime" not in d.localizedName().lower() and "iphone" not in d.localizedName().lower()
sessions = []; sinks = []
for d in chosen:
    key = d.uniqueID(); log("starting", d.localizedName(), key)
    s = AVF.AVCaptureSession.alloc().init()
    for p in (AVF.AVCaptureSessionPreset1280x720, AVF.AVCaptureSessionPresetHigh):
        if d.supportsAVCaptureSessionPreset_(p): s.setSessionPreset_(p); break
    inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(d, None)
    if inp is None: log("input failed", err); continue
    s.addInput_(inp)
    out = AVF.AVCaptureVideoDataOutput.alloc().init()
    out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
    out.setAlwaysDiscardsLateVideoFrames_(True)
    sink = Sink.alloc().initWithKey_(key); sinks.append(sink)
    import dispatch  # pyobjc libdispatch
    out.setSampleBufferDelegate_queue_(sink, dispatch.dispatch_queue_create(b"cam", None))
    s.addOutput_(out); s.startRunning(); sessions.append((s, key, d.localizedName()))
keys = [k for _, k, _ in sessions]
win = "Shop Cameras"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
cv2.imshow(win, np.vstack([np.hstack([blank] * COLS)] * 2)); cv2.waitKey(50)
cv2.moveWindow(win, ORIGIN[0] + 40, ORIGIN[1] + 60); cv2.resizeWindow(win, SIZE[0] - 80, SIZE[1] - 140)
last_seen = {k: 0 for k in keys}; stamp = {k: 0.0 for k in keys}
shape_logged = set(); placed = False; last_dump = 0.0
try:
    while True:
        tiles = []
        with lock: snap = {k: frames.get(k) for k in keys}
        for n, k in enumerate(keys):
            f = snap[k]
            if f is not None and f is not last_seen[k]: last_seen[k] = f; stamp[k] = time.time()
            if f is not None and k not in shape_logged: shape_logged.add(k); log("frame shape", k, f.shape)
            if f is not None and time.time() - stamp[k] < 3:
                t = cv2.resize(f, (TILE_W, TILE_H)); cv2.putText(t, f"cam {n+1}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            else:
                t = blank.copy(); cv2.putText(t, f"cam {n+1}: NO SIGNAL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            tiles.append(t)
        while len(tiles) % COLS: tiles.append(blank.copy())
        rows = [np.hstack(tiles[k:k+COLS]) for k in range(0, len(tiles), COLS)]
        composed = np.vstack(rows)
        cv2.imshow(win, composed)
        if not placed and any(snap[k] is not None for k in keys):
            placed = True; log("composed shape", composed.shape)
            cv2.moveWindow(win, ORIGIN[0] + 40, ORIGIN[1] + 60); cv2.waitKey(100)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)   # fullscreen on monitor 2
        if time.time() - last_dump > 5:
            last_dump = time.time(); cv2.imwrite("/tmp/so101_cams_last.jpg", cv2.resize(composed, (960, int(960 * composed.shape[0] / composed.shape[1]))))
        if cv2.waitKey(30) & 0xFF == ord("q"): break
finally:
    for s, _, _ in sessions: s.stopRunning()
    cv2.destroyAllWindows()
