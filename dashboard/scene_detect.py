#!/usr/bin/env python3
"""scene_detect.py - what is lying on the mat, from one overhead camera frame.

Turns a raw 640x360 frame of the overhead C920 into a list of objects on the Fiskars mat, in mat millimetres and
inches, so the wall dashboard can draw a live map of what sits under the camera. Pure cv2 + numpy; no network,
no camera: the caller hands in frames and gets dicts back.

Pipeline (see docs/mat-scene.md):
  1. warp       the frame is rectified top-down into MAT SPACE with the calibration homography
                (jetson/calib/top_table.json), so one image pixel is a fixed number of millimetres everywhere;
  2. background an empty-mat reference image in the same space (set_background / update_background);
  3. diff       per-channel absolute difference after an exposure-gain match, shadow suppression (darker version of
                the background with the same colour), median filter, threshold, closing;
  4. mask       the four ArUco marker squares (+ margin) and a thin mat border are ignored;
  5. split      the ARM is masked from its known geometry when the caller passes `arm_px` (circles + thick polyline
                in raw camera pixels, from jetson/cam_calib.arm_mask_px_from_joints: the follower's joint angles
                through the forward kinematics and the camera pose); blobs touching the arm-side edge are the arm
                too (plus a halo around them; the only arm rule when arm_px is None), large blobs touching the
                other edges are intrusions (a hand), the rest are objects;
  6. report     centroid / oriented box / area / mean colour of each object in mm, inches and raw pixels, with
                ids kept stable by nearest-neighbour tracking between calls.

Mat frame: X along the long edge (0..457.2 mm), Y away from the arm (0..304.8 mm), origin at the corner on the
arm's side, left. Warped image: column = x * px_per_mm, row = (304.8 - y) * px_per_mm, i.e. origin BOTTOM-LEFT,
the arm enters from the bottom edge of the image. Calibration drift: the markers are detected in every frame and
the homography is re-fitted from them when the mat has moved under the camera (the background lives in mat
space, so it stays valid).

CLI:  python dashboard/scene_detect.py --calib jetson/calib/top_table.json --bg empty.jpg frame.jpg [...] [--save-warp out.png]
"""
import argparse, json, math, os, sys, time
import numpy as np
import cv2

MAT_W_MM, MAT_H_MM = 457.2, 304.8          # Fiskars 18 x 12 in
MM_PER_IN = 25.4
DEFAULT_DICT = "DICT_4X4_50"



def long_axis_mat(obj, min_ratio=1.3):
    """Unit (x, y) of an object's long axis in the MAT frame (the frame of x_in / y_in and pick_loop.jaw_dir_mat: X along the long edge,
    Y away from the arm), or None when it is roughly round (long / short < min_ratio) or has no usable size. The sign is arbitrary (an axis).
    why: 11 Nov 2026 the detector's angle_deg is counter-clockwise from mat +X (detect() negates cv2's image-frame angle because image rows run
    against mat +Y) of the LONG side (w_mm >= h_mm); tests/scene/test_long_axis.py proves it on synthetic bars at known angles."""
    try: w, h, a = float(obj["w_mm"]), float(obj["h_mm"]), math.radians(float(obj["angle_deg"]))
    except (KeyError, TypeError, ValueError): return None
    if min(w, h) <= 0.0 or max(w, h) / min(w, h) < min_ratio: return None
    if h > w: a += math.pi / 2.0   # a caller that hands the sides the other way round
    return (math.cos(a), math.sin(a))

# ---------------------------------------------------------------- small helpers
def _h_apply(H, pts):
    """Apply a 3x3 homography to (N,2) points -> (N,2)."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def _as_homography(calib):
    H = calib.get("homography_px_to_table") if isinstance(calib, dict) else None
    if H is None: raise ValueError("calibration has no 'homography_px_to_table' (run cam_calib.py table first)")
    H = np.array(H, np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("calibration 'homography_px_to_table' must be an invertible 3x3 matrix")
    return H


def detect_markers(gray, dict_name=DEFAULT_DICT):
    """{id: (4,2) float32 corners} for every ArUco marker seen in a grayscale image."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None: return {}
    return {int(i): c.reshape(4, 2).astype(np.float32) for i, c in zip(ids.ravel(), corners)}


def homography_from_markers(seen, layout, size_mm, min_markers=3):
    """Pixel -> mat homography from detected marker corners and the layout's marker CENTRES (same recipe as
    cam_calib.marker_corners_on_table: an affine fit through the centres gives each marker's in-plane rotation, a
    Procrustes snap turns the detected corners into an exact square of the known size around the measured centre,
    then all 4N corners feed findHomography). Returns None when fewer than `min_markers` layout markers are seen."""
    common = sorted(i for i in seen if i in layout)
    if len(common) < max(3, min_markers): return None
    c_img = np.array([seen[i].mean(0) for i in common], np.float64); c_tab = np.array([layout[i] for i in common], np.float64)
    X = np.hstack([c_img, np.ones((len(common), 1))])
    A, *_ = np.linalg.lstsq(X, c_tab, rcond=None); A = A.T
    if not np.all(np.isfinite(A)) or abs(np.linalg.det(A[:, :2])) < 1e-9: return None
    half = size_mm / 2.0; canon = np.array([[-half, half], [half, half], [half, -half], [-half, -half]])
    img_pts, tab_pts = [], []
    for i in common:
        c = seen[i].astype(np.float64); back = c @ A[:, :2].T + A[:, 2]
        centre = np.array(layout[i], np.float64); P = back - centre
        U, _, Vt = np.linalg.svd(canon.T @ P); R = Vt.T @ U.T
        img_pts.extend(c.tolist()); tab_pts.extend((centre + canon @ R.T).tolist())
    H, _ = cv2.findHomography(np.array(img_pts), np.array(tab_pts), 0)
    if H is None or not np.all(np.isfinite(H)): return None
    return H


# ---------------------------------------------------------------- the scene
class MatScene:
    """Overhead-camera scene model for one mat. Construct with the calibration dict, teach it the empty mat with
    set_background(frame), then call detect(frame) at a few Hz. All geometry is cached; detect() costs a few ms."""

    def __init__(self, calib, px_per_mm=0.5, mat_mm=(MAT_W_MM, MAT_H_MM), marker_margin_mm=10.0, edge_margin_mm=2.0,
                 min_area_mm2=40.0, max_area_frac=0.35, diff_thresh=28, shadow_tol=16, arm_halo_mm=10.0, big_mm=60.0,
                 track_mm=15.0, track_miss=3, min_hits=1, refit=True, refit_mm=3.0, refit_min_markers=3, markers_every=1,
                 max_shift_mm=150.0, max_rot_deg=30.0):
        if not isinstance(calib, dict): raise ValueError("calib must be the dict loaded from top_table.json")
        self.calib = calib
        self.H = _as_homography(calib)                                        # raw px -> mat mm (as calibrated)
        self.H0 = self.H.copy()
        self.image_size = tuple(int(v) for v in calib.get("image_size", (640, 360)))
        self.ppm = float(px_per_mm)
        if not (0.05 <= self.ppm <= 5): raise ValueError("px_per_mm out of range (0.05..5)")
        self.mat_w, self.mat_h = float(mat_mm[0]), float(mat_mm[1])
        self.W, self.Hh = int(round(self.mat_w * self.ppm)), int(round(self.mat_h * self.ppm))
        self.dict_name = calib.get("dictionary", DEFAULT_DICT)
        self.marker_size = float(calib.get("marker_size_mm", 50.8))
        self.layout = {int(k): (float(v[0]), float(v[1])) for k, v in (calib.get("layout_markers") or {}).items()}
        self.marker_margin, self.edge_margin = float(marker_margin_mm), float(edge_margin_mm)
        self.min_area_px = float(min_area_mm2) * self.ppm ** 2
        self.max_area_px = float(max_area_frac) * self.W * self.Hh
        self.diff_thresh, self.shadow_tol = int(diff_thresh), int(shadow_tol)
        self.arm_halo_px = max(1, int(round(arm_halo_mm * self.ppm)))
        self.big_px = big_mm * self.ppm
        self.track_mm, self.track_miss = float(track_mm), int(track_miss)
        self.min_hits = max(1, int(min_hits))   # why: the live map showed one-frame shadows/flicker as objects; a real object persists across detects
        self.refit, self.refit_mm, self.refit_min_markers, self.markers_every = bool(refit), float(refit_mm), int(refit_min_markers), max(1, int(markers_every))
        self.max_shift_mm, self.max_rot_deg = float(max_shift_mm), float(max_rot_deg)
        self._set_H(self.H)
        self.valid = self._valid_mask()
        self.bg = None; self.bg_gray = None; self.bg_source = None
        self._tracks = {}; self._next_id = 1; self._calls = 0; self.pending = 0
        self.last = None                                                      # debug state from the last detect()
        self._last_kin = None                                                 # kinematic arm mask of the last detect() (warp space)
        self.calib_err_mm = None; self.n_markers_seen = 0; self.refitted = False
        self.mat_shift_mm = 0.0; self.mat_rot_deg = 0.0                        # how far the mat sits from where it was calibrated

    # ------------------------------------------------ geometry
    def _set_H(self, H):
        self.H = np.array(H, np.float64); self.Hinv = np.linalg.inv(self.H)
        A = np.array([[self.ppm, 0, 0], [0, -self.ppm, self.mat_h * self.ppm], [0, 0, 1]], np.float64)   # mm -> warp px (bottom-left origin)
        self.A = A; self.Ainv = np.linalg.inv(A)
        self.M = A @ self.H; self.Minv = np.linalg.inv(self.M)
        if getattr(self, "H0", None) is not None:                              # where the calibrated mat centre / +X axis land now
            c0 = np.array([self.mat_w / 2, self.mat_h / 2]); pts = _h_apply(self.H, _h_apply(np.linalg.inv(self.H0), [c0, c0 + [50, 0]]))
            self.mat_shift_mm = float(np.hypot(*(pts[0] - c0))); d = pts[1] - pts[0]
            self.mat_rot_deg = float(math.degrees(math.atan2(d[1], d[0])))

    def _frame_H(self, frame):
        """Homography for a frame of any resolution (scaled from the calibrated image size)."""
        h, w = frame.shape[:2]
        if (w, h) == self.image_size: return self.H
        s = np.diag([self.image_size[0] / w, self.image_size[1] / h, 1.0]); return self.H @ s

    def mm_to_warp(self, x_mm, y_mm): return (x_mm * self.ppm, (self.mat_h - y_mm) * self.ppm)
    def warp_to_mm(self, col, row): return (col / self.ppm, self.mat_h - row / self.ppm)
    def mm_to_px(self, x_mm, y_mm): return tuple(float(v) for v in _h_apply(self.Hinv, [(x_mm, y_mm)])[0])
    def px_to_mm(self, u, v): return tuple(float(v_) for v_ in _h_apply(self.H, [(u, v)])[0])

    def warp(self, frame_bgr):
        """Top-down BGR image of the mat: shape (Hh, W, 3), column = x*px_per_mm, row = (304.8 - y)*px_per_mm
        (origin bottom-left, arm side at the bottom). The frame is softened first so the ~2x downscale does not alias
        the mat's printed grid."""
        frame_bgr = np.asarray(frame_bgr)
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3: raise ValueError("frame must be an HxWx3 BGR image")
        M = self.A @ self._frame_H(frame_bgr)
        sigma = max(0.3, 0.5 * (frame_bgr.shape[1] / self.image_size[0]) / max(self.ppm, 1e-6) / 2.2)
        soft = cv2.GaussianBlur(frame_bgr, (0, 0), sigma)
        return cv2.warpPerspective(soft, M, (self.W, self.Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _valid_mask(self):
        """Pixels of the warp that count: inside the mat border, outside the marker squares (+margin)."""
        m = np.zeros((self.Hh, self.W), np.uint8)
        e = int(round(self.edge_margin * self.ppm)); m[e:self.Hh - e, e:self.W - e] = 255
        half = self.marker_size / 2.0 + self.marker_margin
        for x, y in self.layout.values():
            c0, r0 = self.mm_to_warp(x - half, y + half); c1, r1 = self.mm_to_warp(x + half, y - half)
            cv2.rectangle(m, (int(math.floor(c0)), int(math.floor(r0))), (int(math.ceil(c1)), int(math.ceil(r1))), 0, -1)
        return m

    def kinematic_mask(self, arm_px):
        """The arm's silhouette in WARP space (uint8 0/255, shape Hh x W) from `arm_px`: {"points": [[u, v], ...],
        "radii_px": [r, ...]} in raw camera pixels of the calibrated image size (jetson/cam_calib.arm_mask_px_from_joints).
        Each point is a filled circle and consecutive points are joined by a stroke as thick as the larger of the two
        radii, rasterised in raw pixel space and warped with the current homography (exactly what the frame goes
        through), so parallax-correct raw pixels stay correct. None when arm_px has no usable point."""
        if not isinstance(arm_px, dict): raise ValueError("arm_px must be a dict with 'points' and 'radii_px'")
        pts = arm_px.get("points") or []; radii = arm_px.get("radii_px") or []
        if len(pts) != len(radii): raise ValueError("arm_px: points and radii_px must have the same length")
        P = []
        for (u, v), r in zip(pts, radii):
            u, v, r = float(u), float(v), float(r)
            if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(r) and r > 0): continue
            P.append((int(round(max(-1e5, min(1e5, u)))), int(round(max(-1e5, min(1e5, v)))), max(1, int(math.ceil(r)))))
        if not P: return None
        raw = np.zeros((self.image_size[1], self.image_size[0]), np.uint8)
        for (u, v, r) in P: cv2.circle(raw, (u, v), r, 255, -1, lineType=cv2.LINE_8)
        for (u0, v0, r0), (u1, v1, r1) in zip(P, P[1:]): cv2.line(raw, (u0, v0), (u1, v1), 255, thickness=2 * max(r0, r1), lineType=cv2.LINE_8)
        warped = cv2.warpPerspective(raw, self.M, (self.W, self.Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.where(warped > 0, 255, 0).astype(np.uint8)

    # ------------------------------------------------ markers / calibration drift
    def check_markers(self, frame_bgr):
        """Detect the ArUco markers, measure how far they land from the layout under the current homography and,
        when allowed, re-fit the homography from them. Returns (n_seen, err_mm or None, refitted_now)."""
        gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        seen = detect_markers(gray, self.dict_name)
        H = self._frame_H(np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8))
        common = [i for i in seen if i in self.layout]
        self.n_markers_seen = len(common)
        if not common: self.calib_err_mm = None; return 0, None, False
        err = max(float(np.hypot(*(_h_apply(H, seen[i].mean(0))[0] - self.layout[i]))) for i in common)
        refit_now = False
        if self.refit and err > self.refit_mm:
            if gray.shape[1] != self.image_size[0] or gray.shape[0] != self.image_size[1]:
                s = np.diag([gray.shape[1] / self.image_size[0], gray.shape[0] / self.image_size[1], 1.0])
                seen_cal = {i: _h_apply(s, c).astype(np.float32) for i, c in seen.items()}          # corners in calibrated-size pixels
            else: seen_cal = seen
            Hn = homography_from_markers(seen_cal, self.layout, self.marker_size, self.refit_min_markers)
            if Hn is not None:
                Hs = Hn if seen_cal is seen else Hn @ np.diag([self.image_size[0] / gray.shape[1], self.image_size[1] / gray.shape[0], 1.0])
                err_n = max(float(np.hypot(*(_h_apply(Hs, seen[i].mean(0))[0] - self.layout[i]))) for i in common)
                if err_n < err: self._set_H(Hn); err = err_n; refit_now = True; self.refitted = True
        self.calib_err_mm = err
        return len(common), err, refit_now

    # ------------------------------------------------ background
    def _to_warp(self, img):
        img = np.asarray(img)
        if img.shape[:2] == (self.Hh, self.W): return img
        return self.warp(img)

    def set_background(self, frame_bgr, check_markers=True):
        """Learn the empty-mat reference. Accepts a raw frame, a top-down image (shape Hh x W), or a list of raw frames
        (their per-pixel median: lets the arm be somewhere different in each). The markers are checked first so the
        background is learned in a correctly aligned mat frame."""
        frames = frame_bgr if isinstance(frame_bgr, (list, tuple)) else [frame_bgr]
        if not frames: raise ValueError("set_background needs at least one frame")
        if check_markers and np.asarray(frames[0]).shape[:2] != (self.Hh, self.W): self.check_markers(np.asarray(frames[0]))
        warps = [self._to_warp(f).astype(np.float32) for f in frames]
        bg = warps[0] if len(warps) == 1 else np.median(np.stack(warps), axis=0).astype(np.float32)
        self.bg = bg; self.bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY); self.bg_source = "frame"

    def save_background(self, path):
        if self.bg is None: raise ValueError("no background to save")
        ok = cv2.imwrite(str(path), np.clip(self.bg, 0, 255).astype(np.uint8))
        if not ok: raise OSError(f"could not write {path}")

    def load_background(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None: raise OSError(f"could not read background image {path}")
        if img.shape[:2] != (self.Hh, self.W): img = cv2.resize(img, (self.W, self.Hh), interpolation=cv2.INTER_AREA)
        self.set_background(img, check_markers=False); self.bg_source = str(path)

    def update_background(self, frame_bgr, alpha=0.02, alpha_busy=0.0, arm_px=None, keep_mm=()):
        """Slow adaptation: bg <- (1-alpha)*bg + alpha*current on pixels NOT currently flagged as object/arm (with a halo);
        flagged pixels adapt at `alpha_busy` (default 0 = never). Trade-off: with alpha_busy = 0 an object left on the
        mat stays reported indefinitely, but a persistent false detection (a shadow the suppression missed, a spot
        where the mat got scuffed) never heals either; a small alpha_busy (e.g. 0.002) absorbs objects after ~1/alpha_busy
        updates (8 minutes at 2 Hz and 0.002), making them vanish from the map. Lighting drift is tracked through the
        exposure gain in detect() as well, so alpha can stay small. No-op without a background.
        arm_px: the arm silhouette for this frame (see detect); None reuses the last detect()'s, which is right when
        the caller updates with the frame it has just detected on (the server does).
        keep_mm: discs [(x_mm, y_mm, r_mm)] that never adapt, like object pixels. The server passes the tape ring's footprint: parts of its wall are not
        flagged as object, and 12 Nov 2026 a ring that stood still was absorbed in minutes, then left a ring-shaped ghost that swallowed the pile when it was lifted."""
        if self.bg is None: return
        cur = self._to_warp(frame_bgr)
        kin = self.kinematic_mask(arm_px) if arm_px is not None else self._last_kin
        fg, arm, _ = self._foreground(cur, kin)
        busy = cv2.dilate(cv2.bitwise_or(fg, arm), np.ones((3, 3), np.uint8), iterations=self.arm_halo_px)
        for x, y, r in keep_mm:
            c, row = self.mm_to_warp(float(x), float(y)); cv2.circle(busy, (int(round(c)), int(round(row))), int(math.ceil(float(r) * self.ppm)), 255, -1)
        a = np.where(busy > 0, float(alpha_busy), float(alpha)).astype(np.float32)[:, :, None]
        self.bg = (1.0 - a) * self.bg + a * cur.astype(np.float32)
        self.bg_gray = cv2.cvtColor(self.bg, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------ foreground
    def _foreground(self, cur, kin=None):
        """(fg, arm, gain): fg = object pixels, arm = the arm mask (both uint8 0/255): the kinematic silhouette `kin`
        (warp space, from kinematic_mask) when given, plus blobs touching the arm-side edge and big edge intrusions;
        gain = the exposure factor applied to the frame before differencing."""
        cur_f = cur.astype(np.float32)
        cur_gray = cv2.cvtColor(cur_f, cv2.COLOR_BGR2GRAY)
        v = self.valid > 0
        m_bg = float(np.median(self.bg_gray[v])); m_cur = float(np.median(cur_gray[v]))
        gain = float(np.clip(m_bg / max(m_cur, 1.0), 0.6, 1.6)) if m_cur > 0 else 1.0
        cur_f = cur_f * gain; cur_gray = cur_gray * gain
        diff = np.max(np.abs(cur_f - self.bg), axis=2)
        diff8 = cv2.medianBlur(np.clip(diff, 0, 255).astype(np.uint8), 3)
        fg = np.where(diff8 > self.diff_thresh, 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        fg = cv2.bitwise_and(fg, self.valid)
        # arm: components touching the bottom (arm-side) edge; big components touching other edges are intrusions.
        # The arm's attached shadow is part of its blob on purpose: it belongs to the arm mask, not to the objects.
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        arm = np.zeros_like(fg); e = int(round(self.edge_margin * self.ppm))     # a blob touches an edge when it reaches the last valid row/col
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            touches_arm_edge = y + h >= self.Hh - e
            touches_other = x <= e or y <= e or x + w >= self.W - e
            if touches_arm_edge or (touches_other and (w >= self.big_px or h >= self.big_px)):
                arm[lab == k] = 255
        halo = cv2.dilate(arm, np.ones((3, 3), np.uint8), iterations=self.arm_halo_px) if np.any(arm) else None
        if kin is not None and np.any(kin):
            # the kinematic silhouette already carries its own margins (shadow, parallax): anything touching it goes, no 10 mm halo
            kh = cv2.dilate(kin, np.ones((3, 3), np.uint8), iterations=1)
            halo = kh if halo is None else cv2.bitwise_or(halo, kh); arm = cv2.bitwise_or(arm, kin)
        if halo is not None: fg[arm > 0] = 0
        # per-component triage: fragments hanging off the arm (cables, shadow bits) and detached shadows go.
        # A shadow is the background, only darker: ratio uniformly in (0.4, 0.97), no bright pixels, same colour.
        # Per component rather than per pixel so a dark grey object with internal contrast (a nut) is kept.
        ratio = cur_gray / np.maximum(self.bg_gray, 1.0)
        n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for k in range(1, n2):
            x, y, w, h, area = stats2[k]
            sub = lab2[y:y + h, x:x + w] == k
            if halo is not None and np.any(halo[y:y + h, x:x + w][sub]): fg[y:y + h, x:x + w][sub] = 0; continue
            r = ratio[y:y + h, x:x + w][sub]
            p10, p90 = np.percentile(r, [10, 90])
            if p10 > 0.40 and p90 < 0.97 and r.std() < 0.12:
                mism = np.abs(cur_f[y:y + h, x:x + w][sub] - self.bg[y:y + h, x:x + w][sub] * r[:, None]).max(axis=1)
                if np.median(mism) < self.shadow_tol: fg[y:y + h, x:x + w][sub] = 0
        return fg, arm, gain

    # ------------------------------------------------ main call
    def detect(self, frame_bgr, t=None, arm_px=None):
        """Objects on the mat in `frame_bgr` (raw camera frame). See the module docstring for the dict layout.
        arm_px: the arm's silhouette as {"points": [[u, v], ...], "radii_px": [...]} in raw pixels of the calibrated
        image size (kinematic_mask); masked pixels count as arm (arm_in_view, arm_mask_frac) and nothing inside or
        touching them is an object. None: only the edge heuristic masks the arm. "arm_masked_by" in the result says
        which ("kinematics" / "edge", "none" without a background)."""
        t = time.time() if t is None else float(t)
        frame_bgr = np.asarray(frame_bgr)
        kin = self.kinematic_mask(arm_px) if arm_px is not None else None
        self._last_kin = kin
        self._calls += 1
        if self._calls % self.markers_every == 1 or self.markers_every == 1: n_seen, err, _ = self.check_markers(frame_bgr)
        else: n_seen, err = self.n_markers_seen, self.calib_err_mm
        moved = self.mat_shift_mm > self.max_shift_mm or abs(self.mat_rot_deg) > self.max_rot_deg
        note = None
        if err is None: note = "no layout markers seen"
        elif err > max(self.refit_mm, 3.0) * 2: note = f"markers {err:.0f} mm off the layout: re-run cam_calib.py table"
        elif moved: note = f"mat moved {self.mat_shift_mm:.0f} mm / {self.mat_rot_deg:.0f} deg since calibration (markers re-glued?): re-run cam_calib.py table"
        base = {"t": t, "objects": [], "arm_in_view": False, "arm_mask_frac": 0.0, "background": "ok" if self.bg is not None else "missing",
                "n_markers_seen": int(n_seen), "calib_err_mm": None if err is None else round(err, 2), "calib_ok": note is None, "calib_note": note,
                "refitted": bool(self.refitted), "mat_shift_mm": round(self.mat_shift_mm, 1), "mat_rot_deg": round(self.mat_rot_deg, 1),
                "px_per_mm": self.ppm, "mat_mm": [self.mat_w, self.mat_h], "arm_masked_by": "none"}
        if self.bg is None:
            self.last = None; return base
        cur = self.warp(frame_bgr)
        fg, arm, gain = self._foreground(cur, kin)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
        Hf = self._frame_H(frame_bgr); Hf_inv = np.linalg.inv(Hf)
        dets = []
        for k in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[k])
            if area < self.min_area_px or area > self.max_area_px: continue
            cx, cy = float(cents[k][0]), float(cents[k][1])
            sub = (lab[y:y + h, x:x + w] == k)
            pts = np.argwhere(sub)[:, ::-1].astype(np.float32) + np.array([x, y], np.float32)
            (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(pts)
            rw, rh = rw + 1.0, rh + 1.0                                      # pixel extents, not pixel-centre extents
            if rw < rh: rw, rh, ang = rh, rw, ang + 90.0
            ang = -ang                                                       # image rows run against mat +Y
            ang = (ang + 90.0) % 180.0 - 90.0
            xm, ym = self.warp_to_mm(cx, cy)
            u, v = _h_apply(Hf_inv, [(xm, ym)])[0]
            col = cur[y:y + h, x:x + w][sub].reshape(-1, 3).mean(0) * gain
            dets.append({"x_mm": xm, "y_mm": ym, "w_mm": rw / self.ppm, "h_mm": rh / self.ppm, "area_mm2": area / self.ppm ** 2,
                         "angle_deg": float(ang), "color": [int(np.clip(col[2], 0, 255)), int(np.clip(col[1], 0, 255)), int(np.clip(col[0], 0, 255))],
                         "px": [float(u), float(v)], "bbox_mm": [x / self.ppm, self.mat_h - (y + h) / self.ppm, w / self.ppm, h / self.ppm]})
        objs = self._track(dets, t)
        self.last = {"warp": cur, "fg": fg, "arm": arm, "objects": objs, "gain": gain, "t": t}
        arm_frac = float(np.count_nonzero(arm)) / float(self.W * self.Hh)
        base.update({"objects": objs, "pending": self.pending, "arm_in_view": bool(arm_frac > 0), "arm_mask_frac": round(arm_frac, 4), "gain": round(gain, 3),
                     "arm_masked_by": "kinematics" if kin is not None else "edge"})
        return base

    def _track(self, dets, t):
        """Greedy nearest-neighbour association against the tracks from previous calls; ids survive `track_miss` misses."""
        order = sorted(((math.hypot(d["x_mm"] - tr["x"], d["y_mm"] - tr["y"]), i, tid) for i, d in enumerate(dets) for tid, tr in self._tracks.items()))
        used_d, used_t = set(), set()
        assign = {}
        for dist, i, tid in order:
            if dist > self.track_mm: break
            if i in used_d or tid in used_t: continue
            assign[i] = tid; used_d.add(i); used_t.add(tid)
        out = []; self.pending = 0
        for i, d in enumerate(dets):
            tid = assign.get(i)
            if tid is None: tid = self._next_id; self._next_id += 1
            hits = self._tracks.get(tid, {}).get("hits", 0) + 1
            self._tracks[tid] = {"x": d["x_mm"], "y": d["y_mm"], "t": t, "miss": 0, "hits": hits}
            if hits < self.min_hits: self.pending += 1; continue                # seen and remembered, not reported until it persists
            o = {"id": tid, "x_mm": round(d["x_mm"], 1), "y_mm": round(d["y_mm"], 1), "x_in": round(d["x_mm"] / MM_PER_IN, 3), "y_in": round(d["y_mm"] / MM_PER_IN, 3),
                 "w_mm": round(d["w_mm"], 1), "h_mm": round(d["h_mm"], 1), "area_mm2": round(d["area_mm2"], 0), "angle_deg": round(d["angle_deg"], 1),
                 "color": d["color"], "px": [round(d["px"][0], 1), round(d["px"][1], 1)], "bbox_mm": [round(v, 1) for v in d["bbox_mm"]]}
            out.append(o)
        for tid in list(self._tracks):
            if tid not in used_t and tid not in {o["id"] for o in out} and self._tracks[tid]["t"] != t:
                self._tracks[tid]["miss"] += 1
                if self._tracks[tid]["miss"] > self.track_miss: del self._tracks[tid]
        out.sort(key=lambda o: o["id"])
        return out

    # ------------------------------------------------ pictures for humans
    def render(self, scale=4, grid_in=True):
        """Annotated top-down image (BGR) of the last detect(): arm mask tinted red, foreground green, marker squares
        yellow, objects boxed with id and inches. Without a detect() yet, the background (or a grey canvas) is drawn."""
        s = int(scale)
        if self.last is not None: base = self.last["warp"].copy()
        elif self.bg is not None: base = np.clip(self.bg, 0, 255).astype(np.uint8)
        else: base = np.full((self.Hh, self.W, 3), 90, np.uint8)
        img = cv2.resize(base, (self.W * s, self.Hh * s), interpolation=cv2.INTER_NEAREST)
        if self.last is not None:
            arm = cv2.resize(self.last["arm"], (self.W * s, self.Hh * s), interpolation=cv2.INTER_NEAREST) > 0
            fg = cv2.resize(self.last["fg"], (self.W * s, self.Hh * s), interpolation=cv2.INTER_NEAREST) > 0
            img[arm] = (0.5 * img[arm] + 0.5 * np.array([40, 40, 220])).astype(np.uint8)
            img[fg] = (0.5 * img[fg] + 0.5 * np.array([40, 220, 40])).astype(np.uint8)
        if grid_in:
            for i in range(1, int(self.mat_w / MM_PER_IN) + 1):
                c = int(round(i * MM_PER_IN * self.ppm * s)); cv2.line(img, (c, 0), (c, img.shape[0]), (70, 70, 70), 1)
            for j in range(1, int(self.mat_h / MM_PER_IN) + 1):
                r = int(round((self.mat_h - j * MM_PER_IN) * self.ppm * s)); cv2.line(img, (0, r), (img.shape[1], r), (70, 70, 70), 1)
        half = self.marker_size / 2.0 + self.marker_margin
        for mid, (x, y) in self.layout.items():
            c0, r0 = self.mm_to_warp(x - half, y + half); c1, r1 = self.mm_to_warp(x + half, y - half)
            cv2.rectangle(img, (int(c0 * s), int(r0 * s)), (int(c1 * s), int(r1 * s)), (0, 220, 255), 1)
            cv2.putText(img, f"m{mid}", (int(c0 * s) + 2, int(r1 * s) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1)
        if self.last is not None:
            for o in self.last["objects"]:
                bx, by, bw, bh = o["bbox_mm"]
                c0, r0 = self.mm_to_warp(bx, by + bh); c1, r1 = self.mm_to_warp(bx + bw, by)
                cv2.rectangle(img, (int(c0 * s), int(r0 * s)), (int(c1 * s), int(r1 * s)), (0, 255, 0), 1)
                cv2.putText(img, f"#{o['id']} {o['x_in']:.2f},{o['y_in']:.2f}in", (int(c0 * s), max(10, int(r0 * s) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(img, "arm side", (4, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        return img


# ---------------------------------------------------------------- CLI
def _load_json(path):
    with open(path) as f: return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="objects on the mat from overhead frames (JSON per frame)")
    ap.add_argument("frames", nargs="+", help="raw camera frames (jpg/png)")
    ap.add_argument("--calib", required=True, help="top_table.json from cam_calib.py")
    ap.add_argument("--bg", action="append", default=[], help="empty-mat frame (repeat for a median of several), or a saved background PNG")
    ap.add_argument("--save-warp", help="write the annotated top-down image here (one per frame: name_N.png when several)")
    ap.add_argument("--save-bg", help="write the learned background (top-down PNG) here")
    ap.add_argument("--px-per-mm", type=float, default=0.5)
    ap.add_argument("--no-refit", action="store_true", help="do not re-fit the homography from the markers")
    ap.add_argument("--indent", type=int, default=None)
    a = ap.parse_args(argv)
    scene = MatScene(_load_json(a.calib), px_per_mm=a.px_per_mm, refit=not a.no_refit)
    if a.bg:
        imgs = []
        for p in a.bg:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None: print(f"cannot read {p}", file=sys.stderr); return 2
            imgs.append(im)
        if len(imgs) == 1 and imgs[0].shape[:2] == (scene.Hh, scene.W): scene.load_background(a.bg[0])
        else: scene.set_background(imgs)
    if a.save_bg: scene.save_background(a.save_bg)
    for i, p in enumerate(a.frames):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: print(f"cannot read {p}", file=sys.stderr); return 2
        t0 = time.perf_counter(); res = scene.detect(im); ms = (time.perf_counter() - t0) * 1e3
        res["frame"] = os.path.basename(p); res["ms"] = round(ms, 1)
        print(json.dumps(res, indent=a.indent))
        if a.save_warp:
            out = a.save_warp if len(a.frames) == 1 else os.path.splitext(a.save_warp)[0] + f"_{i}" + (os.path.splitext(a.save_warp)[1] or ".png")
            cv2.imwrite(out, scene.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
