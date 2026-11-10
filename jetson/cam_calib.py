#!/usr/bin/env python3
"""cam_calib.py - calibrate the overhead C920 and map its pixels to table / arm coordinates.

Two stages, each saved to a JSON file under ~/so101/calib (override with --calib-dir):

  1. intrinsics   checkerboard calibration (camera matrix + lens distortion) from 15-20 photos
                  of a printed board held at varied angles and distances.
  2. table        ArUco markers taped to the table at measured positions give the camera's pose
                  over the table plane, so any pixel can be turned into a table (X, Y) in mm for an
                  object resting at a known height, and from there into the arm's frame.

Workflow on the Jetson (source $CAM_TOP, or ~/so101/cam_calib.py ... from anywhere):
  python cam_calib.py board      --out ~/so101/calib/print_board.png          # print at 100 %, measure a square
  python cam_calib.py markers    --ids 0-7 --size-mm 40                       # print, cut out, tape to the mat
  python cam_calib.py capture    --count 18                                   # hold the board up, it auto-shoots
  python cam_calib.py intrinsics                                              # -> calib/top_intrinsics.json
  python cam_calib.py table      --layout calib/top_layout.json               # -> calib/top_table.json
  python cam_calib.py check                                                   # annotated frame to eyeball
  python cam_calib.py locate 412 188 --height 11                              # pixel -> table mm (+ arm mm)
  python cam_calib.py fk --pose m0                                             # gripper tip xyz in the arm base frame, from a so101-mark pose
  python cam_calib.py armfit --pairs calib/top_armfit.json                    # table->arm transform from tip positions jogged onto the markers
  python cam_calib.py handeye                                                  # wrist camera pose on gripper_link from calib/wristcam/*.{jpg,json} -> calib/wrist_campose.json
  python cam_calib.py handeye --check w3                                       # reproject the layout markers into one wrist view (calib/wrist_check_w3.png)

Frame sources (--source): a /dev/v4l path or index (holds ~/so101/.cams_busy while open so the dashboard
lets go of the camera), an http URL that returns one JPEG per request (the Mac dashboard's
http://localhost:8765/cam/top.jpg), or a folder of images (offline re-runs, tests).

Table frame convention: X to the right and Y away from you as you stand at the operator's side, origin
wherever the layout file says (a mat corner is handy), Z up from the table surface. The layout's `units`
field is "mm" (default) or "in"; everything in the layout (marker centres, marker size, arm base) is read in
that unit, the maths runs in mm, and results are printed in both. Marker positions are the marker CENTRES,
so marker orientation does not matter; use at least four markers, six to eight spread around the workspace
is better. The optional arm_base_mm / arm_yaw_deg fields (or arm_base in layout units) put the arm's base in
that frame (yaw = direction the arm's forward X axis points, degrees counter-clockwise from table +X) and
`locate` then also prints arm-frame coordinates. `armfit` measures that transform instead of guessing it:
jog the gripper tip onto each marker centre, save the pose with so101-mark, convert with `fk`, and fit.
"""
import argparse, glob, json, math, os, sys, time, urllib.request
import numpy as np
import cv2

DEFAULT_SOURCE = os.environ.get("CAM_TOP", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0")
DEFAULT_COLS, DEFAULT_ROWS, DEFAULT_SQUARE_MM = 9, 6, 25.0        # inner corners of the standard OpenCV board
DEFAULT_DICT = "DICT_4X4_50"
CALIB_DIR = os.path.expanduser("~/so101/calib")
BUSY_FLAG = os.path.expanduser("~/so101/.cams_busy")
MIN_IMAGES, GOOD_IMAGES = 8, 15
UNIT_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4}
PRINT_DPI = 300


class CalibError(Exception):
    """A user-facing calibration problem: wrong inputs, not enough data, nothing detected."""


def log(*a): print(*a, file=sys.stderr, flush=True)


def mm_to_px(mm, dpi=PRINT_DPI): return int(round(mm / 25.4 * dpi))


# ---------------------------------------------------------------- frame sources
class Source:
    """Frames from a V4L device (path or index), an http(s) URL giving one JPEG per GET, or a folder of images."""
    def __init__(self, spec, width=640, height=360, fourcc="MJPG", open_timeout=6.0):
        self.spec = str(spec); self.cap = None; self.files = []
        if os.path.isdir(self.spec):
            self.kind = "folder"
            self.files = sorted(f for f in glob.glob(os.path.join(self.spec, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
            if not self.files: raise CalibError(f"no images in {self.spec}")
        elif self.spec.startswith(("http://", "https://")):
            self.kind = "url"
        else:
            self.kind = "device"
            # why: the dashboard's ffmpeg re-grabs the device within a second or two of the busy flag clearing, so a
            # back-to-back run can find it still held; retry for a few seconds before giving up
            deadline = time.time() + open_timeout
            while True:
                self.cap = cv2.VideoCapture(int(self.spec) if self.spec.isdigit() else self.spec)
                if self.cap.isOpened(): break
                self.cap.release(); self.cap = None
                if time.time() >= deadline: raise CalibError(f"cannot open camera {self.spec} (busy or unplugged; is the dashboard or a lerobot job holding it?)")
                time.sleep(0.5)
            if fourcc: self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """One BGR frame, or None (folder exhausted / fetch failed / camera hiccup)."""
        if self.kind == "folder":
            if not self.files: return None
            f = self.files.pop(0); img = cv2.imread(f)
            if img is None: log(f"skip {f}: not an image")
            return img
        if self.kind == "url":
            url = self.spec + ("&" if "?" in self.spec else "?") + f"t={time.time():.3f}"
            try: data = urllib.request.urlopen(url, timeout=3).read()
            except Exception as e: log("fetch failed:", e); return None
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        ok, img = self.cap.read()
        return img if ok else None

    def close(self):
        if self.cap is not None: self.cap.release(); self.cap = None


class BusyFlag:
    """Hold ~/so101/.cams_busy while a /dev camera is open, so the dashboard's ffmpeg releases it. Leaves a pre-existing flag alone."""
    def __init__(self, path, enabled, settle=1.5): self.path, self.enabled, self.settle, self.made = path, enabled, settle, False
    def __enter__(self):
        if self.enabled and not os.path.exists(self.path):
            open(self.path, "w").close(); self.made = True; time.sleep(self.settle)
        return self
    def __exit__(self, *exc):
        if self.made:
            try: os.remove(self.path)
            except FileNotFoundError: pass
        return False


def read_one(spec, width, height, busy_flag):
    """Grab a single frame from any source kind."""
    with BusyFlag(busy_flag, str(spec).startswith("/dev")):
        src = Source(spec, width, height)
        try:
            for _ in range(30):
                img = src.read()
                if img is not None: return img
                if src.kind == "folder": break
                time.sleep(0.1)
        finally: src.close()
    raise CalibError(f"no frame from {spec}")


# ---------------------------------------------------------------- checkerboard
def board_object_points(cols, rows, square_mm):
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return pts


def find_board(gray, cols, rows):
    """Inner-corner positions as an (N,1,2) float32 array, or None. The SB detector is sub-pixel and tolerant of
    blur; the classic detector plus cornerSubPix is the fallback for builds without it."""
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0) | getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0) | getattr(cv2, "CALIB_CB_ACCURACY", 0)
        ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), flags)
        if ok and c is not None and len(c) == cols * rows: return c.reshape(-1, 1, 2).astype(np.float32)   # why: OpenCV 5 returns (N,2), OpenCV 4 (N,1,2); calibrateCamera wants the latter
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok: return None
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)).reshape(-1, 1, 2).astype(np.float32)


def to_gray(im): return im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def calibrate(images, cols=DEFAULT_COLS, rows=DEFAULT_ROWS, square_mm=DEFAULT_SQUARE_MM):
    """images: iterable of (name, ndarray). Returns a JSON-ready dict. Raises CalibError when the data cannot
    support a calibration (too few boards found, mixed resolutions)."""
    objp = board_object_points(cols, rows, square_mm); obj, pts, used, size, total = [], [], [], None, 0
    for name, im in images:
        total += 1
        g = to_gray(im); s = (g.shape[1], g.shape[0])
        if size is None: size = s
        elif s != size: raise CalibError(f"{name} is {s[0]}x{s[1]} but the first image is {size[0]}x{size[1]}; calibrate one resolution at a time")
        c = find_board(g, cols, rows)
        if c is None: log(f"skip {name}: no {cols}x{rows} board found"); continue
        obj.append(objp); pts.append(c); used.append(name)
    if len(used) < MIN_IMAGES:
        raise CalibError(f"only {len(used)} usable image(s) out of {total}; need at least {MIN_IMAGES} ({GOOD_IMAGES}-20 recommended)")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, pts, size, None, None, flags=cv2.CALIB_FIX_K3)   # why: k3 overfits on a small webcam frame unless the board reaches every corner
    per = {}
    for i, name in enumerate(used):
        proj, _ = cv2.projectPoints(obj[i], rvecs[i], tvecs[i], K, dist)
        per[name] = float(np.sqrt(np.mean(np.sum((proj - pts[i]) ** 2, axis=2))))
    if len(used) < GOOD_IMAGES: log(f"note: {len(used)} boards found; {GOOD_IMAGES}-20 at varied angles gives a steadier result")
    return {"image_size": [int(size[0]), int(size[1])], "camera_matrix": K.tolist(), "dist_coeffs": dist.ravel().tolist(),
            "rms_px": float(rms), "per_image_rms_px": per, "n_images": len(used),
            "board": {"cols": cols, "rows": rows, "square_mm": square_mm}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------------------------------------------------------------- ArUco table calibration
def aruco_dict(name):
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name): raise CalibError(f"unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def load_layout(path):
    """Validate the marker layout file. Returns the dict with integer marker ids."""
    try: lay = json.load(open(path))
    except (OSError, ValueError) as e: raise CalibError(f"cannot read layout {path}: {e}") from e
    name = lay.get("dictionary", DEFAULT_DICT); aruco_dict(name); lay["dictionary"] = name
    units = str(lay.get("units", "mm")).lower()
    if units not in UNIT_MM: raise CalibError(f"layout: units must be one of {sorted(UNIT_MM)}, not {units!r}")
    k = UNIT_MM[units]; lay["units"] = units
    if "marker_size" in lay: size, scale = lay["marker_size"], k            # in layout units
    elif "marker_size_mm" in lay: size, scale = lay["marker_size_mm"], 1.0  # always mm
    else: raise CalibError(f"layout: marker_size (in {units}) is required")
    if not isinstance(size, (int, float)) or size <= 0: raise CalibError("layout: marker_size must be a positive number")
    lay["marker_size_mm"] = float(size) * scale
    raw = lay.get("markers")
    if not isinstance(raw, dict) or len(raw) < 3: raise CalibError("layout: need at least 3 markers as {\"id\": [x, y]} in layout units (4-8 spread around the workspace is better)")
    markers = {}
    for key, v in raw.items():
        try: mid = int(key)
        except ValueError: raise CalibError(f"layout: marker id {key!r} is not an integer") from None
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)): raise CalibError(f"layout: marker {key} must be [x, y]")
        markers[mid] = [float(v[0]) * k, float(v[1]) * k]
    pts = np.array(list(markers.values()))
    if np.linalg.matrix_rank(pts - pts.mean(0)) < 2: raise CalibError("layout: marker centres are collinear; spread them over the table")
    lay["markers"] = markers
    for key, scale in (("arm_base", k), ("arm_base_mm", 1.0)):   # arm_base is in layout units, arm_base_mm always mm
        base = lay.get(key)
        if base is None: continue
        if not (isinstance(base, (list, tuple)) and len(base) == 2 and all(isinstance(x, (int, float)) for x in base)): raise CalibError(f"layout: {key} must be [x, y]")
        lay["arm_base_mm"] = [float(base[0]) * scale, float(base[1]) * scale]; break
    lay.setdefault("arm_base_mm", None)
    lay["arm_yaw_deg"] = float(lay.get("arm_yaw_deg", 0.0))
    return lay


def in_units(calib, mm):
    """mm -> the calibration's layout units (a float or a sequence)."""
    k = UNIT_MM.get(calib.get("units", "mm"), 1.0)
    return [float(v) / k for v in mm] if isinstance(mm, (list, tuple, np.ndarray)) else float(mm) / k


def detect_markers(gray, dict_name=DEFAULT_DICT):
    """{id: (4,2) float32 corners} for every marker seen."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
    det = cv2.aruco.ArucoDetector(aruco_dict(dict_name), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None: return {}
    return {int(i): c.reshape(4, 2).astype(np.float32) for i, c in zip(ids.ravel(), corners)}


MIN_MARKERS = 3


def marker_corners_on_table(seen, layout, common):
    """Table-frame (mm) positions of every detected marker corner, so the solve gets four points per marker instead
    of one. Only the CENTRES are measured; each marker's in-plane rotation (and the image/table handedness) is
    recovered from the detection itself: an affine fit through the centres maps the detected corners back onto the
    table, and a Procrustes fit snaps them to an exact square of the known size around the measured centre.
    Returns (img_pts, tab_pts) as (4N, 2) arrays in matching order, plus the fitted centre-only affine residual."""
    centres_img = np.array([seen[i].mean(0) for i in common], np.float64)
    centres_tab = np.array([layout["markers"][i] for i in common], np.float64)
    X = np.hstack([centres_img, np.ones((len(common), 1))])                         # plain least squares: exact with 3 centres, averaged with more
    A, *_ = np.linalg.lstsq(X, centres_tab, rcond=None); A = A.T                     # (2,3): table = A[:, :2] @ img + A[:, 2]
    if not np.all(np.isfinite(A)) or abs(np.linalg.det(A[:, :2])) < 1e-9: raise CalibError("could not fit the marker centres; are the layout coordinates right?")
    half = layout["marker_size_mm"] / 2.0; canon = np.array([[-half, half], [half, half], [half, -half], [-half, -half]])
    img_pts, tab_pts = [], []
    for i in common:
        c = seen[i].astype(np.float64); back = c @ A[:, :2].T + A[:, 2]           # detected corners, roughly on the table
        centre = np.array(layout["markers"][i]); P = back - centre
        U, _, Vt = np.linalg.svd(canon.T @ P); R = Vt.T @ U.T                        # orthogonal (reflection allowed): table may be mirrored vs image
        ideal = centre + canon @ R.T
        img_pts.extend(c.tolist()); tab_pts.extend(ideal.tolist())
    return np.array(img_pts), np.array(tab_pts)


def solve_table(gray, layout, intrinsics=None):
    """Camera pose over the table from the markers in `gray`. With intrinsics the result is a full pose
    (works at any object height); without, a flat homography valid only for points on the table surface.
    Uses every detected corner (see marker_corners_on_table), so three markers are enough."""
    seen = detect_markers(gray, layout["dictionary"])
    common = sorted(i for i in seen if i in layout["markers"])
    if len(common) < MIN_MARKERS: raise CalibError(f"need {MIN_MARKERS}+ layout markers in view, saw {len(common)}: {common} (detected ids {sorted(seen)})")
    img_pts, tab_pts = marker_corners_on_table(seen, layout, common)
    out = {"dictionary": layout["dictionary"], "marker_size_mm": layout["marker_size_mm"], "markers_used": common, "units": layout.get("units", "mm"), "points_used": int(len(img_pts)),
           "layout_markers": {str(k): v for k, v in layout["markers"].items()}, "image_size": [int(gray.shape[1]), int(gray.shape[0])],
           "arm_base_mm": layout.get("arm_base_mm"), "arm_yaw_deg": layout.get("arm_yaw_deg", 0.0), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if intrinsics:
        K = np.array(intrinsics["camera_matrix"], np.float64); d = np.array(intrinsics["dist_coeffs"], np.float64)
        if intrinsics.get("image_size") and list(intrinsics["image_size"]) != out["image_size"]:
            raise CalibError(f"intrinsics are for {intrinsics['image_size']} but this frame is {out['image_size']}")
        obj = np.hstack([tab_pts, np.zeros((len(tab_pts), 1))])
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, d, flags=cv2.SOLVEPNP_IPPE)
        if not ok: raise CalibError("solvePnP failed on the marker centres")
        if hasattr(cv2, "solvePnPRefineLM"): rvec, tvec = cv2.solvePnPRefineLM(obj, img_pts, K, d, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, d)
        R, _ = cv2.Rodrigues(rvec); C = (-R.T @ tvec).ravel()
        out.update({"camera_matrix": K.tolist(), "dist_coeffs": d.ravel().tolist(), "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
                    "camera_position_mm": C.tolist(), "reproj_rms_px": float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, axis=1))))})
    else:
        H, _ = cv2.findHomography(img_pts, tab_pts, 0)
        if H is None: raise CalibError("homography failed on the marker corners")
        out.update({"homography_px_to_table": H.tolist(), "note": "no intrinsics: valid only for points on the table surface (height 0)"})
        if len(common) < 4:
            out["warning"] = "only 3 markers and no intrinsics: perspective is weakly constrained, expect a few mm of error outside the triangle the markers span; add a 4th marker or run intrinsics"
            log("warning:", out["warning"])
    res = {}
    for i in common:   # residual = how far each MEASURED centre lands from where the layout says it is
        p = seen[i].mean(0); x, y = pixel_to_table(out, float(p[0]), float(p[1]), 0.0)
        res[str(i)] = float(math.hypot(x - layout["markers"][i][0], y - layout["markers"][i][1]))
    out["residual_mm"] = res; out["residual_mm_max"] = max(res.values())
    if out["units"] != "mm": out[f"residual_{out['units']}"] = {k: in_units(out, v) for k, v in res.items()}; out[f"residual_{out['units']}_max"] = in_units(out, out["residual_mm_max"])
    return out


def pixel_to_table(calib, u, v, height_mm=0.0):
    """Table (X, Y) in mm of the point under pixel (u, v) that sits `height_mm` above the table surface."""
    if calib.get("rvec") is not None:
        K = np.array(calib["camera_matrix"], np.float64); d = np.array(calib["dist_coeffs"], np.float64)
        n = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, d).reshape(2)
        R, _ = cv2.Rodrigues(np.array(calib["rvec"], np.float64)); t = np.array(calib["tvec"], np.float64).reshape(3)
        C = -R.T @ t; ray = R.T @ np.array([n[0], n[1], 1.0])
        if abs(ray[2]) < 1e-12: raise CalibError("ray is parallel to the table")
        s = (height_mm - C[2]) / ray[2]
        if s <= 0: raise CalibError("that point is behind the camera")
        P = C + s * ray; return float(P[0]), float(P[1])
    if height_mm: raise CalibError("height needs intrinsics; this table calibration is homography-only")
    H = np.array(calib["homography_px_to_table"], np.float64); p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def table_to_pixel(calib, x, y, height_mm=0.0):
    if calib.get("rvec") is not None:
        proj, _ = cv2.projectPoints(np.array([[x, y, height_mm]], np.float64), np.array(calib["rvec"]), np.array(calib["tvec"]),
                                    np.array(calib["camera_matrix"]), np.array(calib["dist_coeffs"]))
        return float(proj[0, 0, 0]), float(proj[0, 0, 1])
    H = np.linalg.inv(np.array(calib["homography_px_to_table"], np.float64)); p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def table_to_arm(calib, x, y):
    """Arm-frame (X forward, Y left) mm from table mm, or None when the layout has no arm_base_mm."""
    base = calib.get("arm_base_mm")
    if base is None: return None
    yaw = math.radians(calib.get("arm_yaw_deg", 0.0)); dx, dy = x - base[0], y - base[1]
    return (dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))


def arm_to_table(calib, ax, ay):
    """Table mm from arm-frame (X forward, Y left) mm: the inverse of table_to_arm. None without arm_base_mm."""
    base = calib.get("arm_base_mm")
    if base is None: return None
    yaw = math.radians(calib.get("arm_yaw_deg", 0.0))
    return (base[0] + ax * math.cos(yaw) - ay * math.sin(yaw), base[1] + ax * math.sin(yaw) + ay * math.cos(yaw))


# ---------------------------------------------------------------- table -> arm transform from jogged tip positions
def fit_rigid_2d(src, dst):
    """Least-squares rotation+translation with dst ~= R @ src + t (no scale). Returns (R, t, per-point residuals).
    Refuses a mirrored fit, which means an axis convention is flipped somewhere."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 2: raise CalibError("armfit: need at least 2 (table, arm) point pairs")
    ms, md = src.mean(0), dst.mean(0); H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: raise CalibError("armfit: the best fit is a mirror image; one of the two frames has its Y axis flipped (table Y should point away from the operator, arm Y to the arm's left)")
    t = md - R @ ms; res = np.linalg.norm((src @ R.T + t) - dst, axis=1)
    return R, t, res


def arm_pose_from_rigid(R, t):
    """arm = R @ table + t  ->  (arm_base_mm in table frame, arm_yaw_deg) as table_to_arm() expects them."""
    base = -R.T @ t; yaw = math.degrees(math.atan2(R[0, 1], R[0, 0]))   # arm +X axis expressed in the table frame is R^T[:,0] = (R00, R01)
    return [float(base[0]), float(base[1])], float(yaw)


def load_pairs(path, calib):
    """armfit input: {"arm_units": "mm"|"in", "pairs": [{"marker": 0, "arm": [x, y]} | {"table": [x, y], "arm": [x, y]}, ...]}.
    `table` points are in the calibration's layout units; `arm` points in arm_units (default mm). Returns (table_mm, arm_mm)."""
    d = load_json(path, "armfit pairs"); k_arm = UNIT_MM.get(str(d.get("arm_units", "mm")).lower()); k_tab = UNIT_MM[calib.get("units", "mm")]
    if k_arm is None: raise CalibError(f"armfit: arm_units must be one of {sorted(UNIT_MM)}")
    pairs = d.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 2: raise CalibError("armfit: need a 'pairs' list with at least 2 entries")
    tab, arm = [], []
    for i, pr in enumerate(pairs):
        a = pr.get("arm")
        if not (isinstance(a, (list, tuple)) and len(a) >= 2): raise CalibError(f"armfit: pair {i} needs 'arm': [x, y]")
        if "marker" in pr:
            m = calib.get("layout_markers", {}).get(str(pr["marker"]))
            if m is None: raise CalibError(f"armfit: pair {i} names marker {pr['marker']}, which is not in the table calibration")
            tab.append(m)
        elif "table" in pr and isinstance(pr["table"], (list, tuple)) and len(pr["table"]) == 2: tab.append([pr["table"][0] * k_tab, pr["table"][1] * k_tab])
        else: raise CalibError(f"armfit: pair {i} needs 'marker': id or 'table': [x, y]")
        arm.append([a[0] * k_arm, a[1] * k_arm])
    return np.array(tab), np.array(arm)


# SO-101 kinematic chain, base_link -> gripper_frame_link (the point between the jaw tips), copied from dashboard/urdf/so101.urdf.
# Each entry: (joint name, xyz of the joint origin in the parent frame, rpy of that origin). Every moving joint turns about its own Z.
SO101_CHAIN = (
    ("shoulder_pan",  (0.0388353, -8.97657e-09, 0.0624),       (3.14159, 4.18253e-17, -3.14159)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542),       (-1.5708, -1.5708, 0.0)),
    ("elbow_flex",    (-0.11257, -0.028, 1.73763e-16),         (-3.63608e-16, 8.74301e-16, 1.5708)),
    ("wrist_flex",    (-0.1349, 0.0052, 3.62355e-17),          (4.02456e-15, 8.67362e-16, -1.5708)),
    ("wrist_roll",    (5.55112e-17, -0.0611, 0.0181),          (1.5708, 0.0486795, 3.14159)),
    (None,            (-0.0079, -0.000218121, -0.0981274),     (0.0, 3.14159, 0.0)),      # fixed: gripper_frame_joint
)
JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _fk_transforms(deg, chain=SO101_CHAIN, tip_offset_mm=None):
    """The 4x4 base_link -> frame transform of every entry of `chain` (after that entry's joint rotation), in metres,
    plus the tip_offset applied to the last one when given. One entry per chain link, in chain order."""
    T = np.eye(4); out = []
    for name, xyz, rpy in chain:
        A = np.eye(4); A[:3, :3] = _rpy(*rpy); A[:3, 3] = xyz; T = T @ A
        if name is not None:
            q = math.radians(float(deg.get(name, 0.0))); c, s_ = math.cos(q), math.sin(q)
            Rz = np.eye(4); Rz[:2, :2] = [[c, -s_], [s_, c]]; T = T @ Rz
        out.append(T)
    if tip_offset_mm is not None:
        out[-1] = out[-1] @ np.array([[1, 0, 0, tip_offset_mm[0] / 1000.0], [0, 1, 0, tip_offset_mm[1] / 1000.0], [0, 0, 1, tip_offset_mm[2] / 1000.0], [0, 0, 0, 1]])
    return out


def fk_chain(deg, chain=SO101_CHAIN, tip_offset_mm=None):
    """Origins of every frame of the chain in the base_link frame, mm, in chain order: shoulder_pan, shoulder_lift,
    elbow_flex, wrist_flex, wrist_roll, tip (the last one is exactly fk_tip's position). Same joint-angle
    convention as fk_tip. Used to draw the arm's links (dashboard mat scene: the kinematic arm mask)."""
    return [T[:3, 3] * 1000.0 for T in _fk_transforms(deg, chain, tip_offset_mm)]


def fk_tip(deg, chain=SO101_CHAIN, tip_offset_mm=None):
    """Gripper-tip position (mm, base_link frame: Z up, X forward) and the tip's pointing direction (unit vector)
    for joint angles in LeRobot's calibrated degrees (0 = URDF zero; the dashboard uses the same convention).
    tip_offset_mm: optional (x, y, z) in the gripper_frame_link frame, for a contact point that is not the URDF tip
    (the fixed jaw's tip sits off the wrist-roll axis; fitted by fit_arm_model)."""
    T = _fk_transforms(deg, chain, tip_offset_mm)[-1]
    tip = T[:3, 3] * 1000.0; point = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return tip, point


# ---------------------------------------------------------------- arm geometry seen by the overhead camera
# Used by the dashboard's mat scene (docs/mat-scene.md, "Kinematic arm mask"): the arm's links are projected into the
# camera image from the follower's joint angles and masked out of the object detector, so the gripper reaching over
# the mat is never reported as an object.
# Radii (mm) around each chain point: the SO-101 links are 40-60 mm across (radius 20-30), the base plate ~100 mm, the gripper
# with its wrist camera mount ~70 mm; each gets ~10 mm of slack for cables and print, then ARM_MASK_SHADOW for the cast shadow.
# Mind the parallax: a radius is converted to pixels at the point's own height, so the elbow 210 mm up under the 420 mm camera
# covers twice the mat area its millimetres suggest (that is what the camera sees, and what the detector must ignore).
ARM_MASK_RADII_MM = {"shoulder_pan": 50.0, "shoulder_lift": 40.0, "elbow_flex": 30.0, "wrist_flex": 30.0, "wrist_roll": 35.0, "tip": 30.0, "fingers": 20.0}
ARM_MASK_SHADOW = 1.3            # the cast shadow hugs the links: radii + 30 %
ARM_MASK_FLAT_INFLATE = 1.6      # homography-only fallback (no camera_pose_est): no parallax, so radii + 60 % on top
ARM_MASK_FINGERS_MM = 40.0       # the jaws reach this far beyond gripper_frame_link along the gripper axis


def arm_points_to_table(points_mm_base, calib):
    """(N,3) base_link-frame points (mm) -> (N,3) TABLE-frame points: x, y through arm_base_mm / arm_yaw_deg
    (arm_to_table), z above the mat surface = z_base - arm_fit.table_z_mm (0 without an arm fit)."""
    if calib.get("arm_base_mm") is None: raise CalibError("calibration has no arm pose (arm_base_mm): run the arm fit first")
    P = np.asarray(points_mm_base, np.float64).reshape(-1, 3)
    z0 = float((calib.get("arm_fit") or {}).get("table_z_mm", 0.0) or 0.0)
    out = np.empty_like(P)
    for i, (ax, ay, az) in enumerate(P):
        tx, ty = arm_to_table(calib, ax, ay); out[i] = (tx, ty, az - z0)
    return out


def _pose_K_rt(calib):
    """(K, rvec, tvec, scale) of calib['camera_pose_est'] (table -> camera, no distortion), or None. `scale` turns the
    pose's image pixels into the table calibration's (they are the same 640x360 today)."""
    pose = calib.get("camera_pose_est")
    if not isinstance(pose, dict) or pose.get("rvec") is None or pose.get("tvec") is None or pose.get("camera_matrix") is None: return None
    K = np.array(pose["camera_matrix"], np.float64); r = np.array(pose["rvec"], np.float64).reshape(3); t = np.array(pose["tvec"], np.float64).reshape(3)
    ps = pose.get("image_size") or calib.get("image_size"); cs = calib.get("image_size") or ps
    s = (float(cs[0]) / float(ps[0]), float(cs[1]) / float(ps[1])) if ps and cs else (1.0, 1.0)
    return K, r, t, s


def arm_points_to_camera_px(points_mm_base, calib):
    """Project base_link-frame 3D points (mm) to raw camera pixels through the table frame and the camera pose
    estimate in calib['camera_pose_est'] (rvec, tvec, camera_matrix from `campose`; no distortion). Honest about
    height, unlike the homography: a raised gripper lands where the camera really sees it. Returns (N,2) pixels.
    CalibError without a camera pose or an arm pose."""
    got = _pose_K_rt(calib)
    if got is None: raise CalibError("calibration has no camera_pose_est (run cam_calib.py campose) to project 3D points")
    K, r, t, s = got
    tab = arm_points_to_table(points_mm_base, calib)
    proj, _ = cv2.projectPoints(tab.reshape(-1, 1, 3), r, t, K, None)
    px = proj.reshape(-1, 2); px[:, 0] *= s[0]; px[:, 1] *= s[1]
    return px


def arm_mask_px_from_joints(joints_deg, calib, radii_mm=None, shadow=ARM_MASK_SHADOW):
    """The arm as circles in RAW camera pixels for scene_detect.MatScene.detect(arm_px=...): {"points": [[u, v], ...],
    "radii_px": [...], "names": [...], "source": "pose" | "homography", "table_mm": [[x, y, z], ...]}.
    Points: fk_chain of the joints with the fitted joint offsets (calib['arm_fit']['joint_offsets_deg']) applied, no
    tip offset, plus a "fingers" point ARM_MASK_FINGERS_MM beyond the tip along the gripper axis. Radii: per-point mm
    (ARM_MASK_RADII_MM, the wrist camera and cables included) x `shadow`, converted to pixels AT EACH POINT'S HEIGHT:
    with a camera pose, px/mm = fx / depth (the pinhole; a link 200 mm up under a 420 mm camera looks 1.9x bigger);
    without one (homography-only calibration) the points are projected at mat level through the homography, px/mm is
    the homography's local scale there, and the radii get ARM_MASK_FLAT_INFLATE on top to cover the ignored parallax
    (a raised part reads displaced away from the camera axis by height x off-axis distance / camera height).
    Points behind the camera (depth < 1 mm) are dropped. Extra joint keys (gripper) are ignored."""
    fit = calib.get("arm_fit") or {}; off = fit.get("joint_offsets_deg") or {}
    deg = {k: float(joints_deg.get(k, 0.0)) + float(off.get(k, 0.0) or 0.0) for k in JOINT_ORDER}
    Ts = _fk_transforms(deg)
    pts = [T[:3, 3] * 1000.0 for T in Ts]; direction = Ts[-1][:3, :3] @ np.array([0.0, 0.0, 1.0])
    pts.append(pts[-1] + ARM_MASK_FINGERS_MM * direction)
    names = list(JOINT_ORDER) + ["tip", "fingers"]
    radii = {**ARM_MASK_RADII_MM, **(radii_mm or {})}
    P = np.vstack(pts); tab = arm_points_to_table(P, calib)
    got = _pose_K_rt(calib)
    out_pts, out_r, out_n, out_t = [], [], [], []
    if got is not None:
        K, r, t, s = got
        R, _ = cv2.Rodrigues(r); cam = tab @ R.T + t; depth = cam[:, 2]
        proj, _ = cv2.projectPoints(tab.reshape(-1, 1, 3), r, t, K, None); px = proj.reshape(-1, 2)
        fx = 0.5 * (K[0, 0] * s[0] + K[1, 1] * s[1])
        for i, n in enumerate(names):
            if not (depth[i] >= 1.0 and np.all(np.isfinite(px[i]))): continue
            out_pts.append([float(px[i, 0] * s[0]), float(px[i, 1] * s[1])]); out_r.append(float(radii[n] * shadow * fx / depth[i]))
            out_n.append(n); out_t.append([float(v) for v in tab[i]])
        source = "pose"
    else:
        if calib.get("homography_px_to_table") is None: raise CalibError("calibration has neither camera_pose_est nor homography_px_to_table")
        Hinv = np.linalg.inv(np.array(calib["homography_px_to_table"], np.float64))
        def hp(x, y):
            p = Hinv @ np.array([x, y, 1.0]); return p[:2] / p[2]
        for i, n in enumerate(names):
            x, y, z = tab[i]; c = hp(x, y)
            if not np.all(np.isfinite(c)): continue
            ppm = math.sqrt(abs(np.linalg.det(np.column_stack([hp(x + 1.0, y) - c, hp(x, y + 1.0) - c]))))
            out_pts.append([float(c[0]), float(c[1])]); out_r.append(float(radii[n] * shadow * ARM_MASK_FLAT_INFLATE * ppm))
            out_n.append(n); out_t.append([float(v) for v in tab[i]])
        source = "homography"
    return {"points": out_pts, "radii_px": out_r, "names": out_n, "source": source, "table_mm": out_t}


def reach_table(z_table_mm=0.0, tol_mm=8.0, step_deg=3.0):
    """Max horizontal distance from the shoulder-pan axis at which the tip can touch the table plane (z_table_mm in the
    base frame), with any wrist angle and with the gripper pointing down (within 25 deg of -Z). Brute force over the
    planar joints; pan does not change the radius."""
    best_any = best_down = 0.0
    lims = {"shoulder_lift": (-100, 100), "elbow_flex": (-96, 96), "wrist_flex": (-95, 95)}   # URDF limits in degrees
    for a in np.arange(*lims["shoulder_lift"], step_deg):
        for b in np.arange(*lims["elbow_flex"], step_deg):
            for c in np.arange(*lims["wrist_flex"], step_deg):
                tip, point = fk_tip({"shoulder_lift": a, "elbow_flex": b, "wrist_flex": c})
                if abs(tip[2] - z_table_mm) > tol_mm: continue
                r = math.hypot(tip[0] - 38.8353, tip[1])
                best_any = max(best_any, r)
                if point[2] < -math.cos(math.radians(25)): best_down = max(best_down, r)
    return best_any, best_down


OFFSET_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")


def fit_arm_and_offsets(joint_sets, table_pts_mm, table_z_mm=None, fixed_offsets=None):
    """Nonlinear fit of the arm's base pose AND the zero offsets of the planar joints from jogged points.
    LeRobot's degrees are physical but zeroed at the middle of each joint's calibrated range, which is not exactly
    the URDF zero; a few degrees of offset in lift/elbow/wrist shows up as reach that is short near the base and
    right far out. Unknowns: base x, y (table mm), yaw (deg), table z in the arm frame (mm, unless given), and
    the offsets (deg) added to the three planar joints before fk_tip. Residuals are tip xy minus the mapped mat
    point, plus tip z minus table z. Returns a dict with the fit and per-point residuals (mm)."""
    try: from scipy.optimize import least_squares
    except ImportError: raise CalibError("fit_arm_and_offsets needs scipy (pip install scipy)") from None
    J = list(joint_sets); T = np.asarray(table_pts_mm, np.float64); n = len(J)
    fit_z = table_z_mm is None; fit_off = fixed_offsets is None
    if n < (4 if fit_off else 3): raise CalibError(f"need at least {4 if fit_off else 3} jogged points for this fit")

    def unpack(x):
        bx, by, yaw = x[0], x[1], x[2]; i = 3
        z = x[i] if fit_z else table_z_mm; i += 1 if fit_z else 0
        off = dict(zip(OFFSET_JOINTS, x[i:i + 3])) if fit_off else {k: float(fixed_offsets.get(k, 0.0)) for k in OFFSET_JOINTS}
        return bx, by, yaw, z, off

    def resid(x):
        bx, by, yaw, z, off = unpack(x); calib = {"arm_base_mm": [bx, by], "arm_yaw_deg": yaw}; r = []
        for jd, (tx, ty) in zip(J, T):
            tip, _ = fk_tip({k: float(v) + off.get(k, 0.0) for k, v in jd.items()})
            ax, ay = table_to_arm(calib, tx, ty); r += [tip[0] - ax, tip[1] - ay, tip[2] - z]
        return np.array(r)

    # start from the rigid fit (with the fixed offsets applied if given, else none)
    pre = {k: float(fixed_offsets.get(k, 0.0)) for k in OFFSET_JOINTS} if not fit_off else {}
    tips0 = [fk_tip({k: float(v) + pre.get(k, 0.0) for k, v in jd.items()})[0] for jd in J]
    arm0 = np.array([tp[:2] for tp in tips0]); R, t, _ = fit_rigid_2d(T, arm0); base0, yaw0 = arm_pose_from_rigid(R, t)
    z0 = float(np.mean([tp[2] for tp in tips0]))
    head = [base0[0], base0[1], yaw0] + ([z0] if fit_z else [])
    starts = [head + [o1, o2, o3] for o1 in (-6.0, 0.0, 6.0) for o2 in (-6.0, 0.0, 6.0) for o3 in (-6.0, 0.0, 6.0)] if fit_off else [head]
    sol = None   # why: a few degrees of offset in three coupled joints gives the cost several basins; start from a small grid and keep the best
    for x0 in starts:
        cand = least_squares(resid, x0, method="lm" if 3 * n >= len(x0) else "trf")
        if sol is None or cand.cost < sol.cost: sol = cand
    bx, by, yaw, z, off = unpack(sol.x); r = sol.fun.reshape(n, 3)
    return {"arm_base_mm": [float(bx), float(by)], "arm_yaw_deg": float(yaw), "table_z_mm": float(z), "joint_offsets_deg": {k: float(v) for k, v in off.items()},
            "offsets_fitted": fit_off, "residual_xy_mm": [float(np.hypot(a, b)) for a, b, _ in r], "residual_z_mm": [float(c) for _, _, c in r], "rms_mm": float(np.sqrt(np.mean(sol.fun ** 2))),
            "rigid_only_residual_mm": [float(v) for v in np.linalg.norm((T @ R.T + t) - arm0, axis=1)], "n_points": n}


JOINT_LIMITS_DEG = {"shoulder_pan": (-105.0, 105.0), "shoulder_lift": (-98.0, 98.0), "elbow_flex": (-95.0, 95.0), "wrist_flex": (-93.0, 93.0)}
IK_SEEDS = ((0.0, 30.0, 0.0, 70.0), (0.0, 10.0, 30.0, 55.0), (0.0, 50.0, -35.0, 75.0), (0.0, 70.0, -60.0, 85.0), (0.0, -10.0, 60.0, 40.0))


def ik_tip(target_mm, offsets=None, roll_deg=0.0, down=True, seeds=IK_SEEDS, pos_tol_mm=2.0, tilt_tol_deg=5.0):
    """Joint angles (LeRobot degrees, ready to command) that put the gripper tip at target_mm in the base frame with
    the gripper pointing straight down. `offsets` are the fitted zero offsets (added inside, as fk does), so the
    returned angles are what the real arm needs. Numerical: scipy least_squares from several seeds within the joint
    limits; the best solution is checked against pos_tol_mm and tilt_tol_deg and refused otherwise."""
    try: from scipy.optimize import least_squares
    except ImportError: raise CalibError("ik needs scipy (pip install scipy)") from None
    off = offsets or {}; target = np.asarray(target_mm, np.float64); names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
    lo = [JOINT_LIMITS_DEG[n][0] for n in names]; hi = [JOINT_LIMITS_DEG[n][1] for n in names]

    def joints(x): return {**dict(zip(names, (float(v) for v in x))), "wrist_roll": float(roll_deg)}

    def resid(x):
        tip, point = fk_tip({k: v + off.get(k, 0.0) for k, v in joints(x).items()}); r = list(tip - target)
        if down: r += [100.0 * point[0], 100.0 * point[1], 100.0 * (point[2] + 1.0)]   # why: prefer a truly vertical gripper over the last half millimetre of position
        return r

    pan0 = math.degrees(math.atan2(target[1], target[0] - 38.8353))   # pan axis sits 38.8 mm ahead of base_link
    best = None
    for sd in seeds:
        x0 = np.clip([pan0, sd[1], sd[2], sd[3]], lo, hi)
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf")
        if best is None or sol.cost < best.cost: best = sol
    q = joints(best.x); tip, point = fk_tip({k: v + off.get(k, 0.0) for k, v in q.items()})
    err = float(np.linalg.norm(tip - target)); tilt = math.degrees(math.acos(max(-1.0, min(1.0, -point[2]))))
    if err > pos_tol_mm or (down and tilt > tilt_tol_deg):
        raise CalibError(f"target {np.round(target, 1).tolist()} mm not reachable with a vertical gripper (best: {err:.1f} mm off, {tilt:.1f} deg tilt)")
    return {"joints_deg": {k: round(v, 3) for k, v in q.items()}, "tip_mm": [round(float(v), 2) for v in tip], "pos_err_mm": round(err, 3), "tilt_deg": round(tilt, 2)}


def cmd_ik(a):
    tpath = a.table or os.path.join(a.calib_dir, f"{a.name}_table.json"); calib = load_json(tpath, "table calibration") if os.path.exists(tpath) else {}
    off = (calib.get("arm_fit") or {}).get("joint_offsets_deg") or {}
    k = UNIT_MM[a.units]
    if a.mat is not None:
        if not calib.get("arm_base_mm"): raise CalibError("--mat needs a table calibration with the arm pose (run armfit / jog_capture --fit)")
        ax, ay = table_to_arm(calib, a.mat[0] * k, a.mat[1] * k); z = (calib.get("arm_fit") or {}).get("table_z_mm", 0.0) + a.height * k
        target = [ax, ay, z]
    else: target = [v * k for v in a.arm]
    roll = a.roll if a.roll is not None else float((calib.get("arm_fit") or {}).get("wrist_roll_deg", 0.0))   # why: the jaw tip is off the roll axis, so use the roll the arm fit was jogged at
    res = ik_tip(target, off, roll_deg=roll); res["target_mm"] = [round(v, 2) for v in target]; res["offsets_applied"] = off; res["wrist_roll_deg"] = roll
    print(json.dumps(res))


def nominal_K(width, height, hfov_deg=70.42):
    """Pinhole intrinsics for a camera with a known horizontal field of view (C920 at 16:9: 70.42 deg nominal)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


def camera_pose_from_markers(gray, layout, K):
    """Camera position/orientation in the table frame from the marker corners with intrinsics K (no distortion).
    Returns dict: position_mm (camera centre, table frame, z up), rvec/tvec (table -> camera), optical_axis (unit, table
    frame), tilt_deg (from straight down), axis_hits_mm (where the optical axis meets the table), up_table (image -y in
    table frame), reproj_rms_px, markers_used."""
    seen = detect_markers(gray, layout["dictionary"]); common = sorted(i for i in seen if i in layout["markers"])
    if len(common) < MIN_MARKERS: raise CalibError(f"need {MIN_MARKERS}+ layout markers in view for a camera pose, saw {len(common)}: {common}")
    img_pts, tab_pts = marker_corners_on_table(seen, layout, common); obj = np.hstack([tab_pts, np.zeros((len(tab_pts), 1))])
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok: raise CalibError("solvePnP failed on the marker corners")
    if hasattr(cv2, "solvePnPRefineLM"): rvec, tvec = cv2.solvePnPRefineLM(obj, img_pts, K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec); C = (-R.T @ tvec).ravel(); axis = R.T @ np.array([0.0, 0.0, 1.0]); up = R.T @ np.array([0.0, -1.0, 0.0])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None); rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, axis=1))))
    s = -C[2] / axis[2] if abs(axis[2]) > 1e-9 else float("nan"); hit = (C + s * axis) if np.isfinite(s) else np.array([float("nan")] * 3)
    return {"position_mm": C.tolist(), "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(), "optical_axis": axis.tolist(),
            "tilt_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, -axis[2]))))), "axis_hits_mm": hit[:2].tolist(), "up_table": up.tolist(),
            "reproj_rms_px": rms, "markers_used": common, "camera_matrix": K.tolist()}


def fit_hfov_for_height(gray, layout, width, height, lens_height_mm, lo=40.0, hi=110.0):
    """The marker geometry alone cannot separate focal length from camera height; a tape-measured lens height pins both.
    Bisect the horizontal FOV until the solved camera height equals lens_height_mm. Returns (hfov_deg, pose)."""
    def h_of(fov): return camera_pose_from_markers(gray, layout, nominal_K(width, height, fov))["position_mm"][2]
    if (h_of(lo) - lens_height_mm) * (h_of(hi) - lens_height_mm) > 0: raise CalibError(f"lens height {lens_height_mm:.0f} mm is outside what FOV {lo}-{hi} deg can explain ({h_of(hi):.0f}-{h_of(lo):.0f} mm)")
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if (h_of(lo) - lens_height_mm) * (h_of(mid) - lens_height_mm) <= 0: hi = mid
        else: lo = mid
    fov = (lo + hi) / 2.0
    return fov, camera_pose_from_markers(gray, layout, nominal_K(width, height, fov))


def cmd_campose(a):
    layout = load_layout(a.layout or os.path.join(a.calib_dir, f"{a.name}_layout.json"))
    img = read_one(a.source, a.width, a.height, a.busy_flag); gray = to_gray(img); h, w = gray.shape[:2]
    if a.lens_height is not None:
        hfov, pose = fit_hfov_for_height(gray, layout, w, h, a.lens_height * UNIT_MM[a.units]); src = f"lens height {a.lens_height} {a.units} (tape measure) -> hfov {hfov:.2f} deg"
    else:
        hfov = a.hfov; pose = camera_pose_from_markers(gray, layout, nominal_K(w, h, hfov)); src = f"nominal hfov {hfov} deg (height uncertain by ~ +-12 %)"
    pose.update({"hfov_deg": hfov, "image_size": [w, h], "source": src, "units_note": "position_mm in the table frame (z up from the mat surface)", "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    tpath = a.table or os.path.join(a.calib_dir, f"{a.name}_table.json")
    if os.path.exists(tpath):
        t = load_json(tpath, "table calibration"); t["camera_pose_est"] = pose; save_json(tpath, t); saved = tpath
    else: saved = save_json(os.path.join(a.calib_dir, f"{a.name}_campose.json"), pose)
    k = UNIT_MM[a.units]; P = pose["position_mm"]
    print(json.dumps({"saved": saved, "position_" + a.units: [round(v / k, 2) for v in P], "height_" + a.units: round(P[2] / k, 2), "tilt_deg": round(pose["tilt_deg"], 1),
                      "axis_hits_" + a.units: [round(v / k, 2) for v in pose["axis_hits_mm"]], "hfov_deg": round(hfov, 2), "reproj_rms_px": round(pose["reproj_rms_px"], 2), "source": src}))
    track_calib("campose", a.name, {"layout": a.layout or os.path.join(a.calib_dir, f"{a.name}_layout.json"), "markers": ",".join(map(str, pose.get("markers_used", []))), "units": a.units, "hfov_source": src,
                                    "lens_height": a.lens_height, "source": str(a.source), "image_size": f"{w}x{h}", "out": saved},
                {"height_mm": P[2], "tilt_deg": pose["tilt_deg"], "hfov_deg": hfov, "reproj_rms_px": pose["reproj_rms_px"], "markers": len(pose.get("markers_used", []))}, [saved])


def cmd_fk(a):
    if a.pose:
        poses = load_json(a.poses, "poses file")
        if a.pose not in poses: raise CalibError(f"pose {a.pose!r} not in {a.poses}; have {sorted(poses)}")
        deg = poses[a.pose]
    elif a.deg is not None:
        if len(a.deg) < 4: raise CalibError("fk --deg needs pan lift elbow wrist [roll] in degrees")
        deg = dict(zip(JOINT_ORDER, a.deg))
    elif a.reach:
        deg = None
    else: raise CalibError("fk: give --pose NAME, --deg ..., or --reach")
    out = {}
    if deg is not None:
        off = {}
        if not a.raw:
            tpath = a.table or os.path.join(a.calib_dir, f"{a.name}_table.json")
            if os.path.exists(tpath): off = (load_json(tpath, "table calibration").get("arm_fit") or {}).get("joint_offsets_deg") or {}
        tip, point = fk_tip({k: float(v) + off.get(k, 0.0) for k, v in deg.items()})
        if off: out["joint_offsets_deg"] = off
        out.update({"joints_deg": {k: float(deg.get(k, 0.0)) for k in JOINT_ORDER}, "tip_mm": [round(float(v), 1) for v in tip], "tip_in": [round(float(v) / 25.4, 3) for v in tip],
                    "tip_points": [round(float(v), 3) for v in point], "gripper_down_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, -point[2])))), 1),
                    "note": "arm frame: base_link origin at the base plate, X forward (toward the arm's front), Y left, Z up. Feed tip_mm[:2] to armfit as 'arm'."})
    if a.reach:
        any_r, down_r = reach_table(a.table_z)
        out["reach_mm"] = {"table_z_mm": a.table_z, "any_wrist": round(any_r, 0), "gripper_down": round(down_r, 0)}
        out["reach_in"] = {"any_wrist": round(any_r / 25.4, 1), "gripper_down": round(down_r / 25.4, 1)}
    print(json.dumps(out))


def cmd_armfit(a):
    path = a.table or os.path.join(a.calib_dir, f"{a.name}_table.json"); calib = load_json(path, "table calibration")
    tab, arm = load_pairs(a.pairs, calib)
    R, t, res = fit_rigid_2d(tab, arm); base, yaw = arm_pose_from_rigid(R, t)
    calib["arm_base_mm"] = base; calib["arm_yaw_deg"] = yaw
    calib["armfit"] = {"pairs": len(tab), "residual_mm": [round(float(r), 2) for r in res], "residual_mm_max": round(float(res.max()), 2), "source": os.path.abspath(a.pairs), "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(path, calib)
    out = {"saved": path, "arm_base_mm": [round(v, 2) for v in base], "arm_yaw_deg": round(yaw, 2), "residual_mm_max": calib["armfit"]["residual_mm_max"], "pairs": len(tab)}
    if calib.get("units", "mm") != "mm": out[f"arm_base_{calib['units']}"] = [round(v, 3) for v in in_units(calib, base)]; out[f"residual_{calib['units']}_max"] = round(in_units(calib, float(res.max())), 3)
    print(json.dumps(out))
    if res.max() > 10: log("warning: a jogged point is more than 10 mm off the rigid fit; re-jog it or check that the tip was over the marker centre")
    track_calib("armfit", a.name, {"pairs_file": os.path.abspath(a.pairs), "units": calib.get("units", "mm"), "markers": ",".join(map(str, calib.get("markers_used", []))), "out": path},
                {**{f"residual_mm_{i}": float(r) for i, r in enumerate(res)}, "residual_mm_max": float(res.max()), "pairs": len(tab), "arm_yaw_deg": yaw, "arm_base_x_mm": base[0], "arm_base_y_mm": base[1]}, [path, a.pairs])


# ---------------------------------------------------------------- wrist camera: hand-eye calibration
# The wrist camera (icSpring UVC module, 640x480) is bolted to gripper_link. Several frames of a single ArUco marker on the mat,
# each with the follower's joints, pin down X = T_gripper_link<-camera (6 DOF), the focal length and the lens's radial distortion
# by minimising the reprojection error of the marker corners through the arm's forward kinematics (docs/camera-calibration.md,
# "Wrist camera (hand-eye)"). The joints of each view are not perfect (the follower settles, the frame and the joint read are not
# simultaneous, the kinematic model is a few tenths of a degree off), so every view also gets a small rotation correction with a
# prior of `pose_sigma_deg`; the reported errors are both WITH those corrections (how well the model explains the data) and WITHOUT
# them (what a rigid reprojection from the saved pose achieves), plus a leave-one-out prediction, which is the honest number.
LINK_NAMES = ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link")
WRIST_IMAGE_SIZE = (640, 480)
HANDEYE_FX_STARTS = (250.0, 300.0, 350.0, 400.0, 450.0, 550.0)
HANDEYE_MIN_VIEWS = 3


def fk_frames(deg, joint_offsets=None):
    """4x4 base_link -> link transforms (metres) of every URDF link frame for LeRobot degrees `deg` plus the fitted zero
    offsets (added to the joints they name, exactly as fk_tip's callers do): base_link (identity), shoulder_link, upper_arm_link,
    lower_arm_link, wrist_link, gripper_link (after the wrist_roll joint; the wrist camera is rigid in this one),
    gripper_frame_link (the fixed tip frame). The three.js stage's URDF link objects have the same local frames."""
    off = joint_offsets or {}
    d = {k: float(deg.get(k, 0.0)) + float(off.get(k, 0.0) or 0.0) for k in JOINT_ORDER}
    out = {"base_link": np.eye(4)}
    for name, T in zip(LINK_NAMES, _fk_transforms(d)): out[name] = T
    return out


def project_points(P_cam, K, dist):
    """OpenCV pinhole + radial/tangential model, vectorised: (N,3) camera-frame points (any unit, z forward) -> (N,2) pixels.
    dist = (k1, k2, p1, p2[, k3]); the same numbers cv2.projectPoints takes."""
    P = np.asarray(P_cam, np.float64).reshape(-1, 3); K = np.asarray(K, np.float64); d = list(np.asarray(dist, np.float64).ravel()) + [0.0] * 5
    k1, k2, p1, p2, k3 = d[:5]
    z = np.where(np.abs(P[:, 2]) < 1e-9, 1e-9, P[:, 2]); x = P[:, 0] / z; y = P[:, 1] / z; r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * f + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]])


def marker_corners_table(layout, mid, angle_deg=0.0):
    """(4,2) table-frame mm corners of layout marker `mid` in OpenCV's detection order (the marker's top-left, top-right,
    bottom-right, bottom-left): glued with its top toward table +Y that is (cx-h, cy+h), (cx+h, cy+h), (cx+h, cy-h), (cx-h, cy-h);
    angle_deg turns the marker counter-clockwise about its centre (90 deg steps for the way it was glued, plus the small error)."""
    cx, cy = layout["markers"][mid]; h = layout["marker_size_mm"] / 2.0; a = math.radians(angle_deg); c, s = math.cos(a), math.sin(a)
    canon = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])
    return canon @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def table_to_base_matrix(calib):
    """4x4 table-frame mm -> base_link mm (table_to_arm for x, y; z_base = z_table + arm_fit.table_z_mm). CalibError without an arm pose."""
    base = calib.get("arm_base_mm")
    if base is None: raise CalibError("calibration has no arm pose (arm_base_mm / arm_yaw_deg): run the arm fit (jog_capture --fit) first")
    yaw = math.radians(float(calib.get("arm_yaw_deg", 0.0))); c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]); T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = -R @ np.array([float(base[0]), float(base[1]), 0.0]) + np.array([0.0, 0.0, float((calib.get("arm_fit") or {}).get("table_z_mm", 0.0) or 0.0)])
    return T


def marker_corners_base(calib, layout, mid, angle_deg=0.0):
    """(4,3) base_link mm corners of a layout marker lying on the mat (see marker_corners_table)."""
    tab = marker_corners_table(layout, mid, angle_deg); T = table_to_base_matrix(calib)
    return np.hstack([tab, np.zeros((4, 1)), np.ones((4, 1))]) @ T.T[:, :3]


def load_wrist_views(folder, layout, names=None):
    """The hand-eye captures in `folder` (tools/jog_capture.py --wristcam NAME: NAME.jpg + NAME.json with "joints" in LeRobot
    degrees). Returns [{"name", "joints", "image", "markers": {id: (4,2) corners}}] for every view that shows at least one COMPLETE
    layout marker; ids outside the layout are ignored, views without a layout marker are skipped with a note."""
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if names: files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in set(names)]
    if not files: raise CalibError(f"no capture json files in {folder} (jog_capture --wristcam NAME saves NAME.jpg + NAME.json there)")
    views = []
    for jf in files:
        name = os.path.splitext(os.path.basename(jf))[0]; meta = load_json(jf, "wrist capture"); joints = meta.get("joints") if isinstance(meta, dict) else None
        if not isinstance(joints, dict) or not all(k in joints for k in JOINT_ORDER): log(f"skip {name}: no follower joints in {jf}"); continue
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            if os.path.exists(jf[:-5] + ext): img = cv2.imread(jf[:-5] + ext); break
        if img is None: log(f"skip {name}: no image next to {jf}"); continue
        seen = {m: c for m, c in detect_markers(to_gray(img), layout["dictionary"]).items() if m in layout["markers"]}
        if not seen: log(f"skip {name}: no complete layout marker in the frame"); continue
        views.append({"name": name, "joints": {k: float(joints[k]) for k in JOINT_ORDER}, "image": img, "markers": seen})
    return views


def marker_angles_from_overhead(gray, calib, layout):
    """{id: deg} in-plane orientation of every layout marker the overhead camera sees, measured through the table calibration
    (pixel_to_table): the rotation that takes marker_corners_table(layout, id, 0) onto the detected corners. Degrees, (-180, 180]."""
    seen = detect_markers(gray, layout["dictionary"]); out = {}
    for mid, c in seen.items():
        if mid not in layout["markers"]: continue
        tab = np.array([pixel_to_table(calib, float(u), float(v), 0.0) for u, v in c]); P = tab - tab.mean(0); Q = marker_corners_table(layout, mid, 0.0) - np.array(layout["markers"][mid])
        a = math.degrees(math.atan2(np.sum(Q[:, 0] * P[:, 1] - Q[:, 1] * P[:, 0]), np.sum(Q[:, 0] * P[:, 0] + Q[:, 1] * P[:, 1])))
        out[mid] = float(a)
    if not out: raise CalibError("no layout markers in the overhead frame")
    return out


def _rotvec_to_R(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-12: return np.eye(3)
    k = np.asarray(rv, np.float64) / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _R_to_rotvec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def _R_to_quat_xyzw(R):
    """Unit quaternion (x, y, z, w) of a rotation matrix (three.js order)."""
    m = np.asarray(R, np.float64); tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0: s = math.sqrt(tr + 1.0) * 2; w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]: s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]: s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else: s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w]); return (q / np.linalg.norm(q)).tolist()


def _handeye_init(az_deg, dist_mm, z_mm, tilt_deg, roll_deg):
    """A starting X: camera `dist_mm` from the gripper axis in direction az_deg (gripper_link xy), z_mm along the axis, looking
    toward the fingertips (-z of gripper_link) tilted tilt_deg back toward the axis, image-up toward the fingertips, then rolled."""
    az = math.radians(az_deg); d = np.array([math.cos(az), math.sin(az), 0.0]); pos = d * dist_mm + np.array([0.0, 0.0, z_mm])
    tl = math.radians(tilt_deg); axis = -np.array([0.0, 0.0, 1.0]) * math.cos(tl) - d * math.sin(tl)
    up = np.array([0.0, 0.0, -1.0]); up = up - axis * float(up @ axis); up /= np.linalg.norm(up); up = _rotvec_to_R(axis * math.radians(roll_deg)) @ up
    ycam = -up; xcam = np.cross(ycam, axis)
    return _R_to_rotvec(np.column_stack([xcam, ycam, axis])), pos


class _HandEye:
    """The least-squares problem. Parameter vector: rvec(3) + t(3, mm) of X = T_gripper<-camera, [fx], [k1, k2], [cx, cy],
    per-marker angle (deg, unless fixed), per-view rotation correction (deg x3, unless pose_sigma_deg == 0)."""
    def __init__(self, views, calib, layout, intrinsics=None, marker_angles=None, pose_sigma_deg=1.0, angle_prior_px_per_deg=0.2, free_centre=False, image_size=WRIST_IMAGE_SIZE, mids=None):
        self.views = views; self.layout = layout; self.T_tb = table_to_base_matrix(calib); self.offsets = (calib.get("arm_fit") or {}).get("joint_offsets_deg") or {}
        self.mids = list(mids) if mids is not None else sorted({m for v in views for m in v["markers"]})   # mids: share a parameter layout with another solver (leave-one-out)
        self.fixed_angles = marker_angles; self.sigma = float(pose_sigma_deg); self.prior = angle_prior_px_per_deg
        self.w, self.h = image_size; self.intr = None
        if intrinsics:
            K = np.array(intrinsics["camera_matrix"], np.float64); self.intr = (K, np.array(intrinsics["dist_coeffs"], np.float64).ravel())
        self.free_centre = bool(free_centre) and not intrinsics; self.loss = "soft_l1"
        self.n_ang = 0 if marker_angles is not None else len(self.mids); self.n_pose = 3 * len(views) if self.sigma > 0 else 0
        for v in views:
            T = fk_frames(v["joints"], self.offsets)["gripper_link"].copy(); T[:3, 3] *= 1000.0; v["T_bg"] = T
            v.setdefault("weights", {m: np.ones(4) for m in v["markers"]})   # 0 = corner dropped as a bad detection (solve_handeye's outlier pass)
        self.n = 6 + (0 if intrinsics else 3 + (2 if self.free_centre else 0)) + self.n_ang + self.n_pose

    def unpack(self, x):
        rv = x[0:3]; t = x[3:6]; i = 6
        if self.intr is not None: K, dist = self.intr
        else:
            fx = x[i]; k1, k2 = x[i + 1], x[i + 2]; i += 3
            if self.free_centre: cx, cy = x[i], x[i + 1]; i += 2
            else: cx, cy = self.w / 2.0, self.h / 2.0
            K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]]); dist = np.array([k1, k2, 0.0, 0.0, 0.0])
        ang = dict(zip(self.mids, x[i:i + self.n_ang])) if self.n_ang else {m: 0.0 for m in self.mids}; i += self.n_ang
        corr = x[i:i + self.n_pose].reshape(-1, 3) if self.n_pose else np.zeros((len(self.views), 3))
        return rv, t, K, dist, ang, corr

    def x0(self, rv, pos, fx):
        x = list(rv) + list(pos)
        if self.intr is None: x += [fx, -0.2, 0.0] + ([self.w / 2.0, self.h / 2.0] if self.free_centre else [])
        return np.array(x + [0.0] * (self.n_ang + self.n_pose))

    def marker_angle(self, mid, quads, ang):
        return float(self.fixed_angles.get(mid, 0.0)) if self.fixed_angles is not None else 90.0 * quads.get(mid, 0) + ang[mid]

    def corners_base(self, mid, angle_deg):
        tab = marker_corners_table(self.layout, mid, angle_deg)
        return np.hstack([tab, np.zeros((4, 1)), np.ones((4, 1))]) @ self.T_tb.T[:, :3]

    def residuals(self, x, quads, views=None, corrections=True, priors=True):
        rv, t, K, dist, ang, corr = self.unpack(x); R = _rotvec_to_R(rv); out = []
        for i, v in enumerate(self.views if views is None else views):
            Rbg = v["T_bg"][:3, :3]; tbg = v["T_bg"][:3, 3]
            if corrections and self.n_pose and views is None: Rbg = Rbg @ _rotvec_to_R(np.radians(corr[i]))
            for mid, c in v["markers"].items():
                Pg = (self.corners_base(mid, self.marker_angle(mid, quads, ang)) - tbg) @ Rbg; Pc = (Pg - t) @ R
                if np.any(Pc[:, 2] < 5.0): out.extend([1e3] * 8); continue
                out.extend(((project_points(Pc, K, dist) - c) * v["weights"][mid][:, None]).ravel())
        if priors:
            if self.n_ang: out.extend([self.prior * ang[m] for m in self.mids])
            if self.n_pose: out.extend((corr / self.sigma).ravel())
        return np.array(out)

    def best_quads(self, x):
        """Which 90 deg step each marker was glued at: the one whose corners reproject best, summed over the views that see it."""
        if self.fixed_angles is not None: return {}
        rv, t, K, dist, ang, corr = self.unpack(x); R = _rotvec_to_R(rv); q = {}
        for mid in self.mids:
            if not any(mid in v["markers"] for v in self.views): continue
            errs = []
            for s in range(4):
                e = 0.0
                for i, v in enumerate(self.views):
                    if mid not in v["markers"]: continue
                    Rbg = v["T_bg"][:3, :3] @ _rotvec_to_R(np.radians(corr[i])) if self.n_pose else v["T_bg"][:3, :3]
                    Pc = ((self.corners_base(mid, 90.0 * s + ang[mid]) - v["T_bg"][:3, 3]) @ Rbg - t) @ R
                    e += 1e9 if np.any(Pc[:, 2] < 5.0) else float(np.sum(((project_points(Pc, K, dist) - v["markers"][mid]) * v["weights"][mid][:, None]) ** 2))
                errs.append(e)
            q[mid] = int(np.argmin(errs))
        return q

    def bounds(self):
        """Only the camera model is bounded: fx 100-3000 px, k1 in [-1.5, 0.5], k2 in [-2, 5] (a webcam's barrel), the principal
        point inside the frame. Keeps a thin data set from explaining a bad corner with an impossible lens."""
        lo = np.full(self.n, -np.inf); hi = np.full(self.n, np.inf)
        if self.intr is None:
            lo[6:9] = (100.0, -1.5, -2.0); hi[6:9] = (3000.0, 0.5, 5.0)
            if self.free_centre: lo[9:11] = (0.0, 0.0); hi[9:11] = (self.w, self.h)
        return lo, hi

    def fit(self, x0, quads, max_nfev=3000, f_scale=2.0):
        from scipy.optimize import least_squares
        lo, hi = self.bounds(); x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        # why: soft_l1 for the coarse search (smooth, forgiving), cauchy for the refinement so one wild corner cannot steer the lens model
        return least_squares(self.residuals, x0, args=(quads,), loss=self.loss, f_scale=f_scale if self.loss == "soft_l1" else 1.5, bounds=(lo, hi), max_nfev=max_nfev)

    def fit_quads(self, x0, quads, max_nfev):
        """Alternate the continuous fit and the discrete glue-orientation choice until the choice is stable."""
        sol = None
        for _ in range(3):
            sol = self.fit(x0, quads, max_nfev); q2 = self.best_quads(sol.x); x0 = sol.x
            if q2 == quads: break
            quads = q2
        return sol, quads

    def corner_errors(self, x, quads, corrections, views=None):
        """[(view, marker, corner index, error px)] for every corner that is in use, with or without the per-view corrections."""
        r = self.residuals(x, quads, views=views, corrections=corrections, priors=False).reshape(-1, 2); out = []; i = 0
        for v in (self.views if views is None else views):
            for mid in v["markers"]:
                for k in range(4):
                    if v["weights"][mid][k] > 0: out.append((v["name"], mid, k, float(np.linalg.norm(r[i + k]))))
                i += 4
        return out

    def rms_per_view(self, x, quads, corrections, views=None):
        """(overall rms px, {view: rms px}) of the corner reprojection over the corners in use, with or without the per-view rotation corrections."""
        errs = self.corner_errors(x, quads, corrections, views); per = {}
        for v in (self.views if views is None else views):
            e = [c[3] for c in errs if c[0] == v["name"]]; per[v["name"]] = float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")
        return float(np.sqrt(np.mean(np.square([c[3] for c in errs])))) if errs else float("nan"), per


def solve_handeye(views, calib, layout, intrinsics=None, marker_angles=None, pose_sigma_deg=1.0, free_centre=False, image_size=WRIST_IMAGE_SIZE,
                  fx_starts=HANDEYE_FX_STARTS, init=None, leave_one_out=True, quick=False):
    """Hand-eye solve over the wrist views (load_wrist_views). Returns the wrist_campose dict (frame gripper_link; position_mm,
    rotation {rvec, quaternion_xyzw, optical_axis_gripper, up_gripper}, camera_matrix, dist_coeffs, hfov_deg, errors, ...).
    Multi-start over the camera's placement around the gripper (8 azimuths x 3 tilts x 4 rolls) and fx_starts, the best basins
    refined with the per-view corrections; `init` = (rvec, t_mm) adds a start (e.g. the previous solution). With `intrinsics`
    (a checkerboard calibration of the wrist camera at the same resolution) the camera model is fixed and only X is solved."""
    try: from scipy.optimize import least_squares  # noqa: F401
    except ImportError: raise CalibError("handeye needs scipy (pip install scipy)") from None
    if len(views) < HANDEYE_MIN_VIEWS: raise CalibError(f"need at least {HANDEYE_MIN_VIEWS} wrist views with a layout marker, have {len(views)} (6+ recommended)")
    if intrinsics and list(intrinsics.get("image_size") or image_size) != list(image_size): raise CalibError(f"intrinsics are for {intrinsics.get('image_size')} but the wrist frames are {list(image_size)}")
    for v in views:
        h, w = v["image"].shape[:2]
        if (w, h) != tuple(image_size): raise CalibError(f"{v['name']} is {w}x{h}, expected {image_size[0]}x{image_size[1]}")
    coarse = _HandEye(views, calib, layout, intrinsics, marker_angles, 0.0, free_centre=False, image_size=image_size)   # rigid, fixed centre: find the basin
    full = _HandEye(views, calib, layout, intrinsics, marker_angles, pose_sigma_deg, free_centre=free_centre, image_size=image_size); full.loss = "cauchy"
    fxs = list(fx_starts) if intrinsics is None else [0.0]
    starts = [(_handeye_init(az, 55.0, 0.0, tilt, roll), fx) for az in range(0, 360, 45) for tilt in ((20.0, 40.0) if quick else (15.0, 30.0, 45.0)) for roll in (0, 90, 180, 270) for fx in (fxs[len(fxs) // 2],)]
    if init is not None: starts += [((np.asarray(init[0], np.float64), np.asarray(init[1], np.float64)), fx) for fx in fxs]
    cands = []
    for (rv, pos), fx in starts:
        x0 = coarse.x0(rv, pos, fx); q = coarse.best_quads(x0)
        sol, q = coarse.fit_quads(x0, q, 40); cands.append((sol.cost, sol.x, q))
    cands.sort(key=lambda c: c[0]); best = None
    for cost, x, q in cands[:4 if quick else 6]:
        for fx in (fxs if intrinsics is None else [0.0]):
            x1 = x.copy()
            if intrinsics is None: x1[6] = fx if fx else x1[6]
            sol, q1 = coarse.fit_quads(x1, q, 1500)
            if best is None or sol.cost < best[0].cost: best = (sol, q1)
    sol, quads = best
    rv, t, K, dist, ang, _ = coarse.unpack(sol.x)
    x0 = full.x0(rv, t, K[0, 0])
    if full.intr is None: x0[7], x0[8] = dist[0], dist[1]
    if full.n_ang: x0[6 + (0 if full.intr is not None else 3 + (2 if full.free_centre else 0)):][:full.n_ang] = [ang[m] for m in full.mids]
    sol, quads = full.fit_quads(x0, quads, 4000)
    dropped = []
    for _ in range(4):   # bad corner detections (a jaw over the corner, the refinement on the wrong saddle): drop corners far outside the corrected fit and refit
        errs = full.corner_errors(sol.x, quads, True); thr = max(2.5, 8.0 * float(np.median([c[3] for c in errs])))   # why: the median, not the rms, so one wild corner cannot hide behind the scale it inflates
        bad = [c for c in errs if c[3] > thr]
        if not bad or len(dropped) + len(bad) > 0.2 * sum(4 * len(v["markers"]) for v in views): break
        for name, mid, k, e in bad:
            next(v for v in views if v["name"] == name)["weights"][mid][k] = 0.0; dropped.append({"view": name, "marker": int(mid), "corner": int(k), "px": round(e, 1)})
        sol, quads = full.fit_quads(sol.x, quads, 4000)
    rv, t, K, dist, ang, corr = full.unpack(sol.x); R = _rotvec_to_R(rv)
    rms_c, per_c = full.rms_per_view(sol.x, quads, True); rms_r, per_r = full.rms_per_view(sol.x, quads, False)
    angles = {str(m): float(full.marker_angle(m, quads, ang)) for m in full.mids}
    fx = float(K[0, 0]); hfov = math.degrees(2 * math.atan((image_size[0] / 2.0) / fx)); vfov = math.degrees(2 * math.atan((image_size[1] / 2.0) / fx))
    out = {"frame": "gripper_link", "position_mm": [float(v) for v in t],
           "rotation": {"rvec": [float(v) for v in rv], "quaternion_xyzw": _R_to_quat_xyzw(R), "optical_axis_gripper": [float(v) for v in R[:, 2]], "up_gripper": [float(v) for v in -R[:, 1]],
                        "note": "rotation of X = T_gripper_link<-camera: camera axes (x right, y down, z forward) expressed in gripper_link"},
           "camera_matrix": K.tolist(), "dist_coeffs": [float(v) for v in dist], "image_size": [int(image_size[0]), int(image_size[1])], "hfov_deg": float(hfov), "vfov_deg": float(vfov),
           "tilt_from_gripper_axis_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))), "off_axis_mm": float(math.hypot(t[0], t[1])),
           "reproj_rms_px": rms_r, "per_view_px": per_r, "reproj_rms_px_corrected": rms_c, "per_view_px_corrected": per_c,
           "view_pose_corrections_deg": {v["name"]: float(np.linalg.norm(corr[i])) for i, v in enumerate(views)}, "pose_sigma_deg": float(pose_sigma_deg),
           "marker_angles_deg": angles, "marker_angles_source": "fitted (90 deg steps chosen, small angle free)" if marker_angles is None else "given",
           "views_used": [v["name"] for v in views], "markers_used": [int(m) for m in full.mids], "points_used": int(sum(float(np.sum(w)) for v in views for w in v["weights"].values())), "dropped_corners": dropped,
           "intrinsics_fixed": bool(intrinsics), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if leave_one_out and len(views) > HANDEYE_MIN_VIEWS: out["leave_one_out_px"], out["leave_one_out_shift"] = handeye_leave_one_out(views, calib, layout, sol.x, quads, full)
    return out


def handeye_leave_one_out(views, calib, layout, x_full, quads, solver):
    """Refit without each view in turn (warm start from the full solution), reproject the held-out view's corners RIGIDLY (its
    own pose correction is unknown) and report the rms px, plus how far the mount estimate moved (mm, deg)."""
    px, shift = {}, {}
    rv0, t0, *_ = solver.unpack(x_full); R0 = _rotvec_to_R(rv0)
    for hold in range(len(views)):
        sub = [v for j, v in enumerate(views) if j != hold]
        s = _HandEye(sub, calib, layout, solver.intr and {"camera_matrix": solver.intr[0].tolist(), "dist_coeffs": solver.intr[1].tolist()}, solver.fixed_angles, solver.sigma, free_centre=solver.free_centre, image_size=(solver.w, solver.h), mids=solver.mids); s.loss = solver.loss
        x0 = np.concatenate([x_full[:len(x_full) - solver.n_pose], np.delete(x_full[len(x_full) - solver.n_pose:].reshape(-1, 3), hold, axis=0).ravel()]) if solver.n_pose else x_full.copy()
        sol = s.fit(x0, quads, 2000)
        px[views[hold]["name"]] = s.rms_per_view(sol.x, quads, False, views=[views[hold]])[0]
        rv, t, *_ = s.unpack(sol.x); dR = _rotvec_to_R(rv).T @ R0
        shift[views[hold]["name"]] = {"mm": float(np.linalg.norm(t - t0)), "deg": float(math.degrees(np.linalg.norm(_R_to_rotvec(dR))))}
    return px, shift


def wrist_project(pose, calib, layout, joints, mid, angle_deg=None):
    """(4,2) pixels of layout marker `mid`'s corners in the wrist camera for follower `joints` (LeRobot deg), rigidly from a saved
    wrist_campose dict (position_mm + rotation.rvec + camera model). Points behind the camera come back NaN."""
    off = (calib.get("arm_fit") or {}).get("joint_offsets_deg") or {}
    T = fk_frames(joints, off)["gripper_link"].copy(); T[:3, 3] *= 1000.0
    R = _rotvec_to_R(np.asarray(pose["rotation"]["rvec"], np.float64)); t = np.asarray(pose["position_mm"], np.float64)
    a = float(angle_deg if angle_deg is not None else (pose.get("marker_angles_deg") or {}).get(str(mid), 0.0))
    Pc = ((marker_corners_base(calib, layout, mid, a) - T[:3, 3]) @ T[:3, :3] - t) @ R
    px = project_points(Pc, np.array(pose["camera_matrix"]), np.array(pose["dist_coeffs"])); px[Pc[:, 2] < 5.0] = np.nan
    return px


def annotate_wrist(view, pose, calib, layout):
    """The wrist frame with the detected markers (green) and every layout marker reprojected through the arm + hand-eye pose (red
    outline, id). Returns (image, {id: rms px} for the markers both seen and predicted)."""
    out = view["image"].copy(); errs = {}
    if view["markers"]: cv2.aruco.drawDetectedMarkers(out, [c.reshape(1, 4, 2) for c in view["markers"].values()], np.array(list(view["markers"].keys())).reshape(-1, 1))
    h, w = out.shape[:2]
    weights = view.get("weights") or {m: np.ones(4) for m in view["markers"]}   # corners the solve dropped as bad detections stay out of the numbers (and get a yellow cross)
    for o in pose.get("dropped_corners") or []:
        if o.get("view") == view["name"] and o.get("marker") in weights: weights[o["marker"]] = weights[o["marker"]].copy(); weights[o["marker"]][int(o["corner"])] = 0.0
    for mid in sorted(layout["markers"]):
        p = wrist_project(pose, calib, layout, view["joints"], mid)
        if not np.all(np.isfinite(p)) or np.any(p < -w) or np.any(p > 2 * w): continue
        pts = np.round(p).astype(int); cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[0]), 4, (0, 0, 255), 1); cv2.putText(out, str(mid), (int(p[:, 0].mean()) - 6, int(p[:, 1].mean()) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if mid in view["markers"]:
            wt = weights.get(mid, np.ones(4)); e = np.sum((p - view["markers"][mid]) ** 2, 1)
            errs[str(mid)] = float(np.sqrt(np.sum(e * wt) / max(1.0, np.sum(wt))))
            for k in np.flatnonzero(wt == 0): cv2.drawMarker(out, tuple(np.round(view["markers"][mid][k]).astype(int)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
    cv2.putText(out, f"{view['name']}  rigid reprojection  " + "  ".join(f"m{k} {v:.1f}px" for k, v in errs.items()), (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out, errs


def cmd_handeye(a):
    layout = load_layout(a.layout or os.path.join(a.calib_dir, f"{a.name}_layout.json"))
    calib = load_json(a.table or os.path.join(a.calib_dir, f"{a.name}_table.json"), "table calibration")
    folder = a.views or os.path.join(a.calib_dir, "wristcam"); out_path = a.out or os.path.join(a.calib_dir, "wrist_campose.json")
    if a.check:
        pose = load_json(a.pose or out_path, "wrist camera pose")
        if not os.path.exists(os.path.join(folder, a.check + ".json")): raise CalibError(f"view {a.check!r} not found in {folder}")
        views = load_wrist_views(folder, layout, [a.check])
        if not views: raise CalibError(f"view {a.check!r} shows no complete layout marker")
        img, errs = annotate_wrist(views[0], pose, calib, layout); path = a.check_out or os.path.join(a.calib_dir, f"wrist_check_{a.check}.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); cv2.imwrite(path, img)
        print(json.dumps({"saved": path, "view": a.check, "markers_seen": sorted(views[0]["markers"]), "rigid_px": {k: round(v, 2) for k, v in errs.items()}})); return 0
    views = load_wrist_views(folder, layout, a.only.split(",") if a.only else None)
    intr = load_json(a.intrinsics, "wrist intrinsics") if a.intrinsics else None
    angles = None
    if a.overhead:
        angles = marker_angles_from_overhead(to_gray(read_one(a.overhead, calib.get("image_size", [640, 360])[0], calib.get("image_size", [640, 360])[1], a.busy_flag)), calib, layout)
        log("marker orientations from the overhead camera (deg):", {k: round(v, 1) for k, v in angles.items()})
    init = None
    if os.path.exists(out_path) and not a.fresh:
        prev = load_json(out_path, "previous wrist pose"); init = (prev["rotation"]["rvec"], prev["position_mm"])
    res = solve_handeye(views, calib, layout, intr, angles, a.pose_sigma_deg, a.free_centre, init=init, leave_one_out=not a.no_loo)
    res["source"] = {"views_dir": os.path.abspath(folder), "table": os.path.abspath(a.table or os.path.join(a.calib_dir, f"{a.name}_table.json")), "layout": os.path.abspath(a.layout or os.path.join(a.calib_dir, f"{a.name}_layout.json")),
                     "intrinsics": os.path.abspath(a.intrinsics) if a.intrinsics else None, "overhead": a.overhead, "arm_fit": (calib.get("arm_fit") or {}).get("fitted")}
    save_json(out_path, res)
    loo = res.get("leave_one_out_px") or {}
    print(json.dumps({"saved": out_path, "position_mm": [round(v, 1) for v in res["position_mm"]], "off_axis_mm": round(res["off_axis_mm"], 1), "optical_axis_gripper": [round(v, 3) for v in res["rotation"]["optical_axis_gripper"]],
                      "tilt_from_gripper_axis_deg": round(res["tilt_from_gripper_axis_deg"], 1), "hfov_deg": round(res["hfov_deg"], 1), "fx": round(res["camera_matrix"][0][0], 1), "dist": [round(v, 3) for v in res["dist_coeffs"][:2]],
                      "reproj_rms_px": round(res["reproj_rms_px"], 2), "reproj_rms_px_corrected": round(res["reproj_rms_px_corrected"], 2), "per_view_px": {k: round(v, 1) for k, v in res["per_view_px"].items()},
                      "view_pose_corrections_deg": {k: round(v, 2) for k, v in res["view_pose_corrections_deg"].items()}, "leave_one_out_px": {k: round(v, 1) for k, v in loo.items()},
                      "marker_angles_deg": {k: round(v, 1) for k, v in res["marker_angles_deg"].items()}, "views_used": res["views_used"], "markers_used": res["markers_used"]}))
    if loo and max(loo.values()) > 15: log("warning: a held-out view predicts worse than 15 px; re-capture it with the follower still (torque on, wait a second) or add views")
    if not a.intrinsics: log("note: fx and the distance along the optical axis trade off with single-marker views; a checkerboard intrinsics file (--intrinsics) pins both")
    track_calib("handeye", "wrist", {"views_dir": res["source"]["views_dir"], "layout": res["source"]["layout"], "table": res["source"]["table"], "intrinsics": res["source"]["intrinsics"] or "none", "units": calib.get("units", "mm"),
                                     "markers": ",".join(map(str, res["markers_used"])), "views": ",".join(res["views_used"]), "pose_sigma_deg": a.pose_sigma_deg, "free_centre": a.free_centre, "overhead": a.overhead or "none", "out": out_path},
                {"reproj_rms_px": res["reproj_rms_px"], "reproj_rms_px_corrected": res["reproj_rms_px_corrected"], "off_axis_mm": res["off_axis_mm"], "tilt_from_gripper_axis_deg": res["tilt_from_gripper_axis_deg"], "hfov_deg": res["hfov_deg"],
                 "views": len(res["views_used"]), "markers": len(res["markers_used"]), "points_used": res["points_used"], "dropped_corners": len(res.get("dropped_corners", [])),
                 **{f"per_view_px_{k}": v for k, v in res["per_view_px"].items()}, **{f"leave_one_out_px_{k}": v for k, v in loo.items()}}, [out_path])
    return 0


# ---------------------------------------------------------------- printables
def board_image(cols=DEFAULT_COLS, rows=DEFAULT_ROWS, square_mm=DEFAULT_SQUARE_MM, margin_mm=15.0, dpi=PRINT_DPI):
    """A (cols+1)x(rows+1)-square checkerboard with `cols` x `rows` inner corners, white margin, caption."""
    s = mm_to_px(square_mm, dpi); m = mm_to_px(margin_mm, dpi)
    w, h = (cols + 1) * s + 2 * m, (rows + 1) * s + 2 * m + mm_to_px(10, dpi)
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0: img[m + r * s:m + (r + 1) * s, m + c * s:m + (c + 1) * s] = 0
    cv2.putText(img, f"{cols}x{rows} inner corners  {square_mm:g} mm squares  print at 100% and measure one square", (m, h - mm_to_px(3, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    return img


def markers_image(ids, size_mm=40.0, dict_name=DEFAULT_DICT, per_row=4, dpi=PRINT_DPI):
    """A sheet of ArUco markers, each `size_mm` square with a white quiet zone and its id printed under it."""
    d = aruco_dict(dict_name); side = mm_to_px(size_mm, dpi); quiet = side // 6; cell = side + 2 * quiet; label = mm_to_px(8, dpi)
    rows = math.ceil(len(ids) / per_row); cols = min(per_row, len(ids))
    img = np.full((rows * (cell + label) + quiet, cols * cell + quiet), 255, np.uint8)
    for n, mid in enumerate(ids):
        r, c = divmod(n, per_row); y0 = quiet + r * (cell + label) + quiet; x0 = quiet + c * cell + quiet
        img[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(d, int(mid), side)
        cv2.putText(img, f"id {mid}  {size_mm:g} mm", (x0, y0 + side + label - mm_to_px(2, dpi)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    return img


def parse_ids(spec):
    ids = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part: continue
        if "-" in part: a, b = part.split("-", 1); ids.extend(range(int(a), int(b) + 1))
        else: ids.append(int(part))
    if not ids: raise CalibError("no marker ids given")
    return ids


# ---------------------------------------------------------------- annotated check frame
def annotate(img, calib, layout_markers=None):
    out = img.copy(); gray = to_gray(img); seen = detect_markers(gray, calib["dictionary"])
    if seen:
        cv2.aruco.drawDetectedMarkers(out, [c.reshape(1, 4, 2) for c in seen.values()], np.array(list(seen.keys())).reshape(-1, 1))
    for mid, (x, y) in (layout_markers or calib.get("layout_markers", {})).items():
        u, v = table_to_pixel(calib, x, y, 0.0); cv2.drawMarker(out, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
        cv2.putText(out, str(mid), (int(u) + 8, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    o = table_to_pixel(calib, 0, 0); ax = table_to_pixel(calib, 50, 0); ay = table_to_pixel(calib, 0, 50)
    for p, col, name in ((ax, (0, 0, 255), "X"), (ay, (0, 255, 0), "Y")):
        cv2.arrowedLine(out, (int(o[0]), int(o[1])), (int(p[0]), int(p[1])), col, 2, tipLength=0.2)
        cv2.putText(out, name, (int(p[0]) + 4, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    base = calib.get("arm_base_mm")
    if base: b = table_to_pixel(calib, base[0], base[1]); cv2.circle(out, (int(b[0]), int(b[1])), 6, (255, 200, 0), 2); cv2.putText(out, "ARM", (int(b[0]) + 8, int(b[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- commands
def load_json(path, what):
    try: return json.load(open(path))
    except (OSError, ValueError) as e: raise CalibError(f"cannot read {what} {path}: {e}") from e


def save_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=1)
    return path


def track_calib(kind, camera, params, metrics, files):
    """One MLflow run in experiment `so101-calibration` for a calibration file just written (jetson/labtrack.py; docs/mlops/tracking.md).
    Called after the result is saved and printed, only by commands that write a file; a no-op without a tracking server; never raises."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("labtrack", os.path.join(os.path.dirname(os.path.abspath(__file__)), "labtrack.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        with m.LabTrack("so101-calibration").run(f"{kind}-{camera}-{time.strftime('%Y%m%d-%H%M%S')}", tags={"kind": kind, "camera": camera}) as r:
            r.params({"camera": camera, **params}); r.metrics(metrics)
            for f in files: r.artifact(f, "calib")
    except Exception as e: log(f"tracking skipped: {e}")


def cmd_board(a):
    path = a.out or os.path.join(a.calib_dir, "print_board.png"); os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, board_image(a.cols, a.rows, a.square_mm)); print(json.dumps({"saved": path, "dpi": PRINT_DPI, "square_mm": a.square_mm}))


def cmd_markers(a):
    path = a.out or os.path.join(a.calib_dir, "print_markers.png"); os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ids = parse_ids(a.ids); cv2.imwrite(path, markers_image(ids, a.size_mm, a.dictionary)); print(json.dumps({"saved": path, "ids": ids, "size_mm": a.size_mm, "dpi": PRINT_DPI}))


def cmd_capture(a):
    out = a.out or os.path.join(a.calib_dir, "captures"); os.makedirs(out, exist_ok=True)
    start = len(glob.glob(os.path.join(out, f"{a.prefix}_*.png"))); accepted = []; last = 0.0
    with BusyFlag(a.busy_flag, a.source.startswith("/dev")):
        src = Source(a.source, a.width, a.height); deadline = time.time() + a.timeout
        if src.kind == "device": log(f"camera open; show the {a.cols}x{a.rows} board, move it between shots (tilt, distance, corners of the view)")
        try:
            while len(accepted) < a.count and time.time() < deadline:
                img = src.read()
                if img is None:
                    if src.kind == "folder": break
                    time.sleep(0.05); continue
                corners = None
                if not a.raw:
                    corners = find_board(to_gray(img), a.cols, a.rows)
                    if corners is None: continue
                    if time.time() - last < a.interval: continue
                    if any(float(np.mean(np.linalg.norm(corners - p, axis=2))) < a.min_move for p in accepted if p is not None): continue
                path = os.path.join(out, f"{a.prefix}_{start + len(accepted):03d}.png"); cv2.imwrite(path, img)
                accepted.append(corners); last = time.time(); log(f"[{len(accepted)}/{a.count}] {path}")
        finally: src.close()
    print(json.dumps({"saved": len(accepted), "dir": out}))
    return 0 if accepted else 1


def cmd_intrinsics(a):
    folder = a.images or os.path.join(a.calib_dir, "captures")
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if not files: raise CalibError(f"no images in {folder}")
    imgs = ((os.path.basename(f), im) for f in files for im in [cv2.imread(f)] if im is not None)
    res = calibrate(imgs, a.cols, a.rows, a.square_mm)
    path = save_json(a.out or os.path.join(a.calib_dir, f"{a.name}_intrinsics.json"), res)
    K = res["camera_matrix"]; worst = max(res["per_image_rms_px"].items(), key=lambda kv: kv[1])
    print(json.dumps({"saved": path, "n_images": res["n_images"], "rms_px": round(res["rms_px"], 3), "fx": round(K[0][0], 1), "fy": round(K[1][1], 1),
                      "cx": round(K[0][2], 1), "cy": round(K[1][2], 1), "worst_image": [worst[0], round(worst[1], 3)]}))
    if res["rms_px"] > 1.0: log("warning: RMS above 1 px; drop the worst images (per_image_rms_px in the file) or reshoot with a flatter board")


def cmd_table(a):
    layout = load_layout(a.layout or os.path.join(a.calib_dir, f"{a.name}_layout.json"))
    intr_path = a.intrinsics or os.path.join(a.calib_dir, f"{a.name}_intrinsics.json")
    intr = load_json(intr_path, "intrinsics") if (a.intrinsics or os.path.exists(intr_path)) and not a.no_intrinsics else None
    if intr is None: log("no intrinsics: homography-only result, valid for objects on the table surface")
    img = read_one(a.source, a.width, a.height, a.busy_flag)
    res = solve_table(to_gray(img), layout, intr)
    path = a.out or os.path.join(a.calib_dir, f"{a.name}_table.json")
    if os.path.exists(path):
        prev = load_json(path, "previous table calibration"); old_fit = prev.get("arm_fit") or {}
        same_mat = {str(k): [round(float(v), 3) for v in vv] for k, vv in (prev.get("layout_markers") or {}).items()} == {str(k): [round(float(v), 3) for v in vv] for k, vv in res["layout_markers"].items()}
        if old_fit.get("joint_offsets_deg") and same_mat and prev.get("arm_fit", {}).get("points"):
            # why: the camera moved but the mat and arm did not (same marker layout): the measured arm pose is still valid, keep all of it
            res["arm_fit"] = {**old_fit, "carried_over_from": old_fit.get("fitted")}; res["arm_base_mm"] = prev.get("arm_base_mm"); res["arm_yaw_deg"] = prev.get("arm_yaw_deg")
            log("same marker layout as the previous calibration: kept the measured arm pose and joint offsets (camera-only change)")
        elif old_fit.get("joint_offsets_deg"):
            # why: the fitted joint offsets belong to the motor calibration, not the mat; keep them when the mat/layout changed, but the base pose must be re-jogged
            res["arm_fit"] = {"joint_offsets_deg": old_fit["joint_offsets_deg"], "table_z_mm": old_fit.get("table_z_mm", 0.0), "carried_over_from": old_fit.get("fitted"),
                              "base_pose": "layout guess; rerun jog_capture --fit --keep-offsets to measure arm_base_mm / arm_yaw_deg / table_z_mm"}
            log("marker layout changed: kept the fitted joint offsets only; base pose is the layout guess until the next jog fit")
    path = save_json(path, res)
    shot = os.path.splitext(path)[0] + "_check.png"; cv2.imwrite(shot, annotate(img, res))
    print(json.dumps({"saved": path, "check_image": shot, "markers_used": res["markers_used"], "residual_mm_max": round(res["residual_mm_max"], 2),
                      "camera_position_mm": [round(v, 1) for v in res["camera_position_mm"]] if "camera_position_mm" in res else None}))
    if res["residual_mm_max"] > 5: log("warning: a marker lands more than 5 mm off; re-measure the layout positions or add markers")
    track_calib("table", a.name, {"layout": a.layout or os.path.join(a.calib_dir, f"{a.name}_layout.json"), "markers": ",".join(map(str, res["markers_used"])), "units": res["units"], "intrinsics": intr_path if intr else "none",
                                  "source": str(a.source), "image_size": "x".join(map(str, res["image_size"])), "arm_fit": (res.get("arm_fit") or {}).get("carried_over_from") or "none", "out": path},
                {**{f"residual_mm_{k}": v for k, v in res["residual_mm"].items()}, "residual_mm_max": res["residual_mm_max"], "reproj_rms_px": res.get("reproj_rms_px"), "points_used": res["points_used"], "markers": len(res["markers_used"]),
                 "camera_height_mm": res["camera_position_mm"][2] if "camera_position_mm" in res else None, "arm_fit_rms_mm": (res.get("arm_fit") or {}).get("rms_mm")}, [path, shot])


def cmd_check(a):
    calib = load_json(a.table or os.path.join(a.calib_dir, f"{a.name}_table.json"), "table calibration")
    img = read_one(a.source, a.width, a.height, a.busy_flag)
    seen = detect_markers(to_gray(img), calib["dictionary"]); errs = {}
    for mid, c in seen.items():
        if str(mid) in calib.get("layout_markers", {}):
            x, y = pixel_to_table(calib, *map(float, c.mean(0)), 0.0); lx, ly = calib["layout_markers"][str(mid)]; errs[str(mid)] = round(math.hypot(x - lx, y - ly), 2)
    path = a.out or os.path.join(a.calib_dir, f"{a.name}_check.png"); cv2.imwrite(path, annotate(img, calib))
    out = {"saved": path, "markers_seen": sorted(seen), "error_mm": errs}
    if calib.get("units", "mm") != "mm": out[f"error_{calib['units']}"] = {k: round(in_units(calib, v), 3) for k, v in errs.items()}
    print(json.dumps(out))


def cmd_locate(a):
    calib = load_json(a.table or os.path.join(a.calib_dir, f"{a.name}_table.json"), "table calibration")
    x, y = pixel_to_table(calib, a.u, a.v, a.height); out = {"pixel": [a.u, a.v], "height_mm": a.height, "table_mm": [round(x, 2), round(y, 2)]}
    arm = table_to_arm(calib, x, y)
    if arm: out["arm_mm"] = [round(arm[0], 2), round(arm[1], 2)]
    u = calib.get("units", "mm")
    if u != "mm":
        out[f"table_{u}"] = [round(v, 3) for v in in_units(calib, (x, y))]
        if arm: out[f"arm_{u}"] = [round(v, 3) for v in in_units(calib, arm)]
    print(json.dumps(out))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--calib-dir", default=CALIB_DIR); p.add_argument("--name", default="top", help="camera name used in output file names")
    sub = p.add_subparsers(dest="cmd", required=True)

    def board_args(sp):
        sp.add_argument("--cols", type=int, default=DEFAULT_COLS, help="inner corners across"); sp.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="inner corners down")
        sp.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)

    def source_args(sp):
        sp.add_argument("--source", default=DEFAULT_SOURCE, help="/dev/v4l path, camera index, JPEG URL, or image folder")
        sp.add_argument("--width", type=int, default=640); sp.add_argument("--height", type=int, default=360); sp.add_argument("--busy-flag", default=BUSY_FLAG)

    s = sub.add_parser("board", help="print-ready checkerboard PNG"); board_args(s); s.add_argument("--out"); s.set_defaults(fn=cmd_board)
    s = sub.add_parser("markers", help="print-ready ArUco marker sheet"); s.add_argument("--ids", default="0-7"); s.add_argument("--size-mm", type=float, default=40.0)
    s.add_argument("--dictionary", default=DEFAULT_DICT); s.add_argument("--out"); s.set_defaults(fn=cmd_markers)
    s = sub.add_parser("capture", help="auto-save frames that show the board from new angles"); board_args(s); source_args(s)
    s.add_argument("--count", type=int, default=18); s.add_argument("--timeout", type=float, default=300); s.add_argument("--interval", type=float, default=0.8, help="seconds between shots")
    s.add_argument("--min-move", type=float, default=20.0, help="mean corner shift (px) needed before another shot"); s.add_argument("--raw", action="store_true", help="save every frame, no board check")
    s.add_argument("--prefix", default="board"); s.add_argument("--out"); s.set_defaults(fn=cmd_capture)
    s = sub.add_parser("intrinsics", help="checkerboard calibration from captured images"); board_args(s); s.add_argument("--images", help="folder (default <calib-dir>/captures)"); s.add_argument("--out"); s.set_defaults(fn=cmd_intrinsics)
    s = sub.add_parser("table", help="camera pose over the table from ArUco markers"); source_args(s); s.add_argument("--layout"); s.add_argument("--intrinsics"); s.add_argument("--no-intrinsics", action="store_true"); s.add_argument("--out"); s.set_defaults(fn=cmd_table)
    s = sub.add_parser("check", help="annotated frame: detected markers vs layout, table axes"); source_args(s); s.add_argument("--table"); s.add_argument("--out"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("locate", help="pixel -> table mm (and arm mm)"); s.add_argument("u", type=float); s.add_argument("v", type=float); s.add_argument("--height", type=float, default=0.0, help="object height above the table, mm")
    s.add_argument("--table"); s.set_defaults(fn=cmd_locate)
    s = sub.add_parser("fk", help="gripper tip position in the arm base frame from joint degrees (so101-mark pose or --deg)"); s.add_argument("--pose", help="name in the poses file")
    s.add_argument("--poses", default=os.path.expanduser("~/so101/poses.json")); s.add_argument("--deg", type=float, nargs="+", help="pan lift elbow wrist [roll] in degrees")
    s.add_argument("--reach", action="store_true", help="also report the max horizontal reach to the table plane"); s.add_argument("--table-z", type=float, default=0.0, help="table surface height in the base frame, mm")
    s.add_argument("--table", help="table calibration whose fitted joint offsets to apply (default <calib-dir>/<name>_table.json)"); s.add_argument("--raw", action="store_true", help="ignore fitted joint offsets")
    s.set_defaults(fn=cmd_fk)
    s = sub.add_parser("armfit", help="fit the table->arm transform from tip positions jogged onto known table points"); s.add_argument("--pairs", required=True, help="JSON: {arm_units, pairs:[{marker|table, arm}]}")
    s.add_argument("--table"); s.set_defaults(fn=cmd_armfit)
    s = sub.add_parser("campose", help="estimate the camera's position over the table from the markers (nominal FOV, or exact with --lens-height)"); source_args(s)
    s.add_argument("--layout"); s.add_argument("--table"); s.add_argument("--hfov", type=float, default=70.42, help="horizontal field of view, deg (C920 16:9 nominal 70.42)")
    s.add_argument("--lens-height", type=float, help="tape-measured height of the lens above the mat, in --units; pins the focal length"); s.add_argument("--units", default="mm", choices=sorted(UNIT_MM))
    s.set_defaults(fn=cmd_campose)
    s = sub.add_parser("ik", help="joint angles (LeRobot deg, offsets applied) to put the tip at a point with the gripper vertical")
    g = s.add_mutually_exclusive_group(required=True); g.add_argument("--mat", type=float, nargs=2, metavar=("X", "Y"), help="mat point in --units"); g.add_argument("--arm", type=float, nargs=3, metavar=("X", "Y", "Z"), help="arm-frame point in --units")
    s.add_argument("--height", type=float, default=0.0, help="with --mat: tip height above the mat surface, in --units"); s.add_argument("--units", default="mm", choices=sorted(UNIT_MM))
    s.add_argument("--roll", type=float, default=None, help="wrist_roll to hold, deg (default: the roll the arm fit was jogged at, else 0)"); s.add_argument("--table"); s.set_defaults(fn=cmd_ik)
    s = sub.add_parser("handeye", help="wrist camera pose on gripper_link (hand-eye) from single-marker wrist views captured with jog_capture --wristcam")
    s.add_argument("--views", help="folder of NAME.jpg + NAME.json captures (default <calib-dir>/wristcam)"); s.add_argument("--layout"); s.add_argument("--table"); s.add_argument("--out", help="default <calib-dir>/wrist_campose.json")
    s.add_argument("--only", help="comma-separated view names to use"); s.add_argument("--intrinsics", help="checkerboard intrinsics of the wrist camera (640x480): fixes fx/cx/cy/distortion")
    s.add_argument("--overhead", help="overhead frame (device, URL or image) to measure the markers' glue angles through the table calibration instead of fitting them"); s.add_argument("--busy-flag", default=BUSY_FLAG)
    s.add_argument("--pose-sigma-deg", type=float, default=1.0, help="prior on the per-view rotation correction (0 = rigid: trust the joints completely)"); s.add_argument("--free-centre", action="store_true", help="also fit the principal point")
    s.add_argument("--fresh", action="store_true", help="ignore a previous wrist_campose.json as a starting point"); s.add_argument("--no-loo", action="store_true", help="skip the leave-one-out check")
    s.add_argument("--check", metavar="VIEW", help="reproject the layout markers into this captured view with the saved pose and write an annotated PNG"); s.add_argument("--pose", help="wrist pose file for --check (default the --out path)"); s.add_argument("--check-out", help="PNG path for --check")
    s.set_defaults(fn=cmd_handeye)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try: return a.fn(a) or 0
    except CalibError as e: log("error:", e); return 2


if __name__ == "__main__": sys.exit(main())
