#!/usr/bin/env python3
"""demo_cams.py - the recorder's real CameraSource (docs/scripted-demos.md, phase 2): a reader of jetson/camd.py's socket.

camd owns the V4L2 devices (see its docstring for why); this class only SUBSCRIBES, so it can never take a camera away from the dashboard's live feed, and a
crash here changes nothing for anyone else. One thread per camera keeps the last `ring_s` seconds of (capture time, JPEG). frame_at(cam, t) hands the recorder the
newest frame captured at or before t as the JPEG bytes (the recorder spools them as they are and decodes only at commit), with its capture time on the
same time.monotonic() clock the pick_seq tick sink uses (camd and the sink run on the same Jetson). A camera that never delivered gives a black frame
stamped 1000 s old, which the recorder's health check reports as camera_frozen, so the episode is rejected instead of the arm waiting.
"""
import collections, os, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camd  # noqa: E402
from demo_recorder import CameraSource, CAM_FREEZE_S  # noqa: E402

SHAPES = {"top": (360, 640, 3), "front": (360, 640, 3), "wrist": (480, 640, 3)}   # camd's CAMS sizes; the dataset features are built from these


class CamdCameraSource(CameraSource):
    def __init__(self, path=camd.SOCK, shapes=None, fps=30, ring_s=1.5, clock=time.monotonic, retry_s=1.0):
        self.shapes = dict(shapes or SHAPES); self.names = tuple(self.shapes); self.path, self.fps, self.ring_s, self.clock, self.retry = path, fps, ring_s, clock, retry_s
        self.ring = {c: collections.deque() for c in self.names}; self.lock = threading.Lock(); self.stop = False; self.connected = {c: False for c in self.names}; self.threads = []

    def start(self):
        for c in self.names: t = threading.Thread(target=self.reader, args=(c,), daemon=True); t.start(); self.threads.append(t)
        return self

    def close(self): self.stop = True

    def reader(self, cam):
        while not self.stop:
            try:
                s = camd.subscribe(cam, self.fps, self.path); self.connected[cam] = True
                for t, jpg in camd.read_frames(s):
                    if self.stop: break
                    with self.lock:
                        r = self.ring[cam]; r.append((t, jpg))
                        while r and r[0][0] < t - self.ring_s: r.popleft()
            except OSError: pass
            self.connected[cam] = False
            if not self.stop: time.sleep(self.retry)

    def frame_at(self, cam, t):
        best = None
        with self.lock:
            for tc, jpg in reversed(self.ring[cam]):
                if tc <= t: best = (tc, jpg); break
        if best is None: return np.zeros(self.shapes[cam], np.uint8), t - 1000.0
        return best[1], best[0]

    def ages(self):
        """{cam: seconds since the newest frame, or None when none arrived}; the daemon's readiness check."""
        now = self.clock()
        with self.lock: return {c: (now - r[-1][0] if r else None) for c, r in self.ring.items()}

    def missing(self, max_age=CAM_FREEZE_S):
        return sorted(c for c, a in self.ages().items() if a is None or a > max_age)
