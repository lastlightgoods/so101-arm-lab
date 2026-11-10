#!/usr/bin/env python3
"""cam_drift.py - has a camera (or the mat) moved since the reference was taken?

  cam_drift.py set            save one reference frame per camera to ~/so101/calib/ref_<cam>.jpg (+ ref_meta.json)
  cam_drift.py check          grab a fresh frame per camera and compare it with the reference

Per camera the check estimates the whole-frame shift between the reference and the fresh frame
(cv2.phaseCorrelate on Hanning-windowed grayscale, refined with an ECC Euclidean fit that also gives the
rotation) and prints dx, dy in px, rotation in degrees and the phase-correlation response (1.0 = identical
scene, ~0.0 = nothing in common). For the TOP camera it additionally projects every marker of the stored
table calibration (~/so101/calib/top_table.json: layout_markers through the inverse homography, or the full
pose when intrinsics are present) to its expected pixel and compares that with the ArUco centre actually
detected: the marker displacement is the precise number, and together with the frame shift it tells a
moved camera from a moved mat (see `combine` below and docs/camera-health.md).

Frames come from the devices (opened briefly with ~/so101/.cams_busy held so the Mac dashboard's ffmpeg lets
go), or with --url http://localhost:8765 from the dashboard's /cam/<name>.jpg stills so the check can run from
the Mac without touching the cameras, or with --from-dir DIR from DIR/<cam>.jpg (offline runs, tests).

Exit status: 0 everything in place; 1 when any camera's |shift| > --threshold (6 px), its markers moved,
or the comparison is unreliable (response < --min-response); 2 when a frame or a reference is missing.
Run with the arm at REST: the wrist camera rides on the arm, so its frame only compares at the same pose.
"""
import argparse, importlib.util, json, math, os, sys, time, urllib.request

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_cam_calib():
    spec = importlib.util.spec_from_file_location("cam_calib", os.path.join(HERE, "cam_calib.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


cc = _load_cam_calib()

CALIB_DIR = os.path.expanduser("~/so101/calib")
BUSY_FLAG = os.path.expanduser("~/so101/.cams_busy")
CAMS = {   # recording geometry (SO101_CAMS in env.sh); fourcc None = the device default (the icSpring only does YUYV)
    "top":   {"path": os.environ.get("CAM_TOP", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_9D46D6AF-video-index0"), "width": 640, "height": 360, "fourcc": "MJPG"},
    "front": {"path": os.environ.get("CAM_FRONT", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_B6A1F66F-video-index0"), "width": 640, "height": 360, "fourcc": "MJPG"},
    "wrist": {"path": os.environ.get("CAM_WRIST", "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"), "width": 640, "height": 480, "fourcc": None},
}
THRESHOLD_PX = 6.0        # |frame shift| or marker displacement above this = moved
MIN_RESPONSE = 0.3        # phase-correlation response below this = the two frames do not match well enough to judge
MIN_MARKERS = 2           # markers needed before they outrank the frame shift
PHASE_FLOOR = 0.05        # below this phase response the frames share nothing (blank, other scene): skip ECC
ECC_MIN_CC = 0.8          # ECC correlation that counts as a trustworthy Euclidean fit even when the phase peak was weak
WARM_FRAMES = 6           # minimum frames discarded after opening a device (first frames are stale)
WARM_SECONDS = 2.0        # and keep discarding until this long after the first frame: the icSpring opens blown-out white and takes ~1.5 s to settle
EXPOSURE_LIMITS = {"mean_min": 12.0, "mean_max": 240.0, "std_min": 8.0}   # outside this a frame is black, white or flat and cannot be compared


def log(*a): print(*a, file=sys.stderr, flush=True)


def to_gray(img):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return g


# ---------------------------------------------------------------- shift estimation
def estimate_shift(ref, new, rotation=True, min_response=MIN_RESPONSE):
    """Shift of `new` relative to `ref` (both BGR or gray, same size): dx, dy in px (positive = the scene content
    moved right / down in the new frame), phase-correlation response, and with rotation=True the in-plane rotation
    in degrees (counter-clockwise positive, cv2.getRotationMatrix2D's convention) from an ECC Euclidean fit seeded
    with the translation. dx, dy are the displacement of the image centre, so a pure rotation reports ~0 shift."""
    a = to_gray(ref).astype(np.float32); b = to_gray(new).astype(np.float32)
    if a.shape != b.shape: b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = a.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (px, py), resp = cv2.phaseCorrelate(a, b, win)
    resp = float(resp) if math.isfinite(resp) else 0.0
    finite = math.isfinite(px) and math.isfinite(py)
    phase_ok = resp >= min_response and finite
    out = {"dx": float(px), "dy": float(py), "response": resp, "rot_deg": None, "method": "phase", "ecc_cc": None, "reliable": phase_ok}
    # a small rotation (1-2 deg) alone drags the phase response down to ~0.2, so ECC gets a go on any plausible peak
    if not rotation or not finite or resp < PHASE_FLOOR: return out
    # ECC on a half-size copy (speed) seeded with the phase translation; its warp maps ref coords -> new coords
    s = 0.5; small = (int(round(w * s)), int(round(h * s)))
    a2 = cv2.resize(a, small, interpolation=cv2.INTER_AREA); b2 = cv2.resize(b, small, interpolation=cv2.INTER_AREA)
    a2 = cv2.GaussianBlur(a2, (3, 3), 0); b2 = cv2.GaussianBlur(b2, (3, 3), 0)
    W = np.array([[1, 0, px * s], [0, 1, py * s]], np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        ecc, W = cv2.findTransformECC(a2, b2, W, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return out
    if not math.isfinite(ecc): return out
    cx, cy = (w - 1) / 2.0 * s, (h - 1) / 2.0 * s
    moved = W @ np.array([cx, cy, 1.0]); edx, edy = (moved[0] - cx) / s, (moved[1] - cy) / s
    rot = -math.degrees(math.atan2(W[1, 0], W[0, 0]))
    agrees = math.hypot(edx - px, edy - py) <= 3.0
    if (phase_ok and agrees and ecc > 0) or ecc >= ECC_MIN_CC:   # a converged, well-correlated Euclidean fit is the better estimate
        out.update({"dx": float(edx), "dy": float(edy), "rot_deg": float(rot), "method": "ecc", "ecc_cc": float(ecc), "reliable": True})
    else:
        out.update({"rot_deg": float(rot), "ecc_cc": float(ecc), "method": "phase (ecc disagreed)"})
    return out


def euclidean_displacement(points, shift, size):
    """Where the frame transform (rotation about the image centre + shift) would move each pixel: (N,2) displacements."""
    pts = np.asarray(points, np.float64); c = np.array([(size[0] - 1) / 2.0, (size[1] - 1) / 2.0])
    th = math.radians(-(shift.get("rot_deg") or 0.0))        # rot_deg is counter-clockwise positive, image y points down
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return (pts - c) @ R.T + c + np.array([shift["dx"], shift["dy"]]) - pts


# ---------------------------------------------------------------- markers against the stored calibration
def expected_marker_pixels(calib):
    """{id: (u, v)} where the calibration says each layout marker centre should appear."""
    return {int(k): cc.table_to_pixel(calib, float(x), float(y)) for k, (x, y) in calib["layout_markers"].items()}


def marker_check(frame, calib, threshold=THRESHOLD_PX):
    """Detected ArUco centres vs the calibration's expected pixels. Per-marker displacement, mean vector, max norm."""
    gray = to_gray(frame)
    expected = expected_marker_pixels(calib)
    seen = cc.detect_markers(gray, calib.get("dictionary", cc.DEFAULT_DICT))
    per = {}
    for mid, (eu, ev) in sorted(expected.items()):
        det = seen.get(mid)
        if det is None: per[mid] = {"expected": [round(eu, 2), round(ev, 2)], "detected": None, "disp": None, "norm": None}; continue
        cu, cv_ = map(float, det.mean(0))
        per[mid] = {"expected": [round(eu, 2), round(ev, 2)], "detected": [round(cu, 2), round(cv_, 2)], "disp": [round(cu - eu, 2), round(cv_ - ev, 2)], "norm": round(math.hypot(cu - eu, cv_ - ev), 2)}
    found = [p for p in per.values() if p["disp"] is not None]
    out = {"expected": len(expected), "detected": len(found), "missing": [m for m, p in per.items() if p["disp"] is None], "markers": per,
           "mean_dx": None, "mean_dy": None, "max_norm": None, "spread_px": None, "moved": None, "threshold_px": threshold}
    if found:
        d = np.array([p["disp"] for p in found]); norms = np.array([p["norm"] for p in found])
        out.update({"mean_dx": round(float(d[:, 0].mean()), 2), "mean_dy": round(float(d[:, 1].mean()), 2), "max_norm": round(float(norms.max()), 2),
                    "spread_px": round(float(np.linalg.norm(d - d.mean(0), axis=1).max()), 2), "moved": bool(norms.max() > threshold)})
    return out


# ---------------------------------------------------------------- verdicts
def combine(shift, markers, threshold=THRESHOLD_PX, min_response=MIN_RESPONSE, size=(640, 360)):
    """One verdict per camera from the frame shift (every camera) and the marker displacement (top only).

    Reasoning: a moved CAMERA moves everything in the frame, markers included, by the same rigid transform
    (shift, maybe a little roll); a moved MAT moves the markers as one body but not the table / arm base behind
    them, so the frame peak stays put (or weakens); a single knocked or peeled marker breaks the markers'
    rigidity among themselves; and a frame shift with the markers still where the calibration projects them
    means the camera has NOT moved relative to the mat: the scene content changed (arm pose, objects,
    lighting), or camera and mat moved as one rigid unit (the table was bumped), which leaves the
    pixel->table calibration valid either way. Hence the markers outrank the frame shift whenever enough
    of them are seen, and the frame transform is used to explain WHY they moved."""
    mag = math.hypot(shift["dx"], shift["dy"]) if shift else None
    frame_moved = bool(shift and shift["reliable"] and mag > threshold)
    frame_low = not (shift and shift["reliable"])
    v = {"shift_px": round(mag, 2) if mag is not None else None, "frame_moved": frame_moved, "frame_unreliable": frame_low}
    if markers and markers["detected"] >= MIN_MARKERS:
        found = {m: p for m, p in markers["markers"].items() if p["disp"] is not None}
        exp = np.array([p["expected"] for p in found.values()], np.float64); det = np.array([p["detected"] for p in found.values()], np.float64)
        disp = det - exp; tol = max(2.0, threshold / 2)
        v["marker_shift_px"] = round(float(np.hypot(markers["mean_dx"], markers["mean_dy"])), 2); v["authority"] = "markers"
        if not markers["moved"]:
            if frame_moved or frame_low:
                v.update(ok=True, status="OK", verdict=f"camera and mat in place (markers within {markers['max_norm']} px of the calibration); the frame shift comes from scene changes, not geometry")
            else:
                v.update(ok=True, status="OK", verdict=f"camera and mat in place (markers within {markers['max_norm']} px, frame shift {v['shift_px']} px)")
            return v
        # the markers moved: did they follow the frame (camera), move as a body on their own (mat), or break rank (one marker)?
        follow = None
        if shift and shift["reliable"]:
            follow = float(np.linalg.norm(disp - euclidean_displacement(exp, shift, size), axis=1).max()); v["marker_vs_frame_px"] = round(follow, 2)
        try:
            R, t, res = cc.fit_rigid_2d(exp, det); rigid = float(res.max()); v["marker_rigid_residual_px"] = round(rigid, 2)
            v["marker_rot_deg"] = round(-math.degrees(math.atan2(R[1, 0], R[0, 0])), 3)
        except cc.CalibError:
            rigid, res = float("inf"), np.full(len(found), float("inf"))
        rot = shift.get("rot_deg") if shift else None
        rot_txt = f", roll {rot:+.2f} deg" if rot is not None and abs(rot) >= 0.2 else ""
        if follow is not None and follow <= tol:
            v.update(ok=False, status="CAMERA MOVED", verdict=f"frame and markers moved together by ({markers['mean_dx']:+.1f}, {markers['mean_dy']:+.1f}) px{rot_txt}: re-run `so101-calib table` (camera-only re-solve keeps the arm fit), then `cam_drift.py set`")
        elif rigid > tol and len(found) >= 3:
            worst = list(found)[int(np.argmax(res))]
            v.update(ok=False, status="MARKER MOVED", verdict=f"markers do not move as one body (rigid-fit residual {rigid:.1f} px, worst id {worst}): a marker was knocked or peeled; re-stick it or re-solve the table")
        else:
            v.update(ok=False, status="MAT MOVED", verdict=f"markers moved as a body by ({markers['mean_dx']:+.1f}, {markers['mean_dy']:+.1f}) px while the frame did not follow (frame {v['shift_px']} px): the mat moved relative to camera and arm; put it back or re-solve the table AND refit the arm")
        return v
    v["authority"] = "frame"
    if markers is not None: v["marker_note"] = f"only {markers['detected']}/{markers['expected']} markers seen (need {MIN_MARKERS}); judged from the frame shift"
    if frame_low:
        v.update(ok=False, status="LOW CONFIDENCE", verdict=f"response {shift['response'] if shift else 0:.2f} < {min_response}: blank/dark frame or the scene changed too much to compare; look at the picture")
    elif frame_moved:
        v.update(ok=False, status="MOVED", verdict=f"frame shifted {v['shift_px']} px ({shift['dx']:+.1f}, {shift['dy']:+.1f}) > {threshold} px since the reference")
    else:
        v.update(ok=True, status="OK", verdict=f"frame shift {v['shift_px']} px, within {threshold} px")
    return v


# ---------------------------------------------------------------- frame sources
def fetch_url(base, name, timeout=4.0):
    url = f"{base.rstrip('/')}/cam/{name}.jpg?t={time.time():.3f}"
    try: data = urllib.request.urlopen(url, timeout=timeout).read()
    except Exception as e: log(f"{name}: fetch {url} failed: {e}"); return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: log(f"{name}: {url} did not return a JPEG")
    return img


def read_dir(folder, name):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p): return cv2.imread(p, cv2.IMREAD_COLOR)
    log(f"{name}: no {name}.jpg/.png in {folder}"); return None


def read_device(cfg, warm=WARM_FRAMES, warm_s=WARM_SECONDS):
    """Open the device (the caller holds the busy flag), stream through the warm-up (at least `warm` frames AND
    `warm_s` seconds after the first one, so auto-exposure has settled), return the next frame."""
    try: src = cc.Source(cfg["path"], cfg["width"], cfg["height"], cfg.get("fourcc"))
    except cc.CalibError as e: log(str(e)); return None
    try:
        got, t0, last = 0, None, None
        for _ in range(warm + 400):
            img = src.read()
            if img is None: time.sleep(0.05); continue
            got += 1; last = img; t0 = t0 or time.monotonic()
            if got > warm and time.monotonic() - t0 >= warm_s: return img
        if last is not None: return last
    finally: src.close()
    log(f"{cfg['path']}: no frame"); return None


def frame_quality(img, limits=EXPOSURE_LIMITS):
    """Is the frame usable at all? Mean grey and contrast, plus a reason when it is blown out, black or flat
    (auto-exposure off with the wrong gain, lens covered, camera pointing at a lamp)."""
    g = to_gray(img); mean, std = float(g.mean()), float(g.std())
    if mean > limits["mean_max"]: bad = f"blown out (mean grey {mean:.0f})"
    elif mean < limits["mean_min"]: bad = f"black (mean grey {mean:.0f})"
    elif std < limits["std_min"]: bad = f"flat (contrast std {std:.1f})"
    else: bad = None
    return {"mean": round(mean, 1), "std": round(std, 1), "bad": bad}


def grab_frames(names, a):
    """{name: BGR frame or None} from whichever source the options select."""
    if a.url: return {n: fetch_url(a.url, n) for n in names}
    if a.from_dir: return {n: read_dir(a.from_dir, n) for n in names}
    with cc.BusyFlag(a.busy_flag, not a.no_busy):
        return {n: read_device(CAMS[n]) for n in names}


# ---------------------------------------------------------------- commands
def ref_path(calib_dir, name): return os.path.join(calib_dir, f"ref_{name}.jpg")
def meta_path(calib_dir): return os.path.join(calib_dir, "ref_meta.json")


def load_calib(a):
    path = a.calib or os.path.join(a.calib_dir, "top_table.json")
    if not os.path.exists(path): return None, path
    try: return json.load(open(path)), path
    except ValueError as e: log(f"cannot read {path}: {e}"); return None, path


def cmd_set(a, names):
    frames = grab_frames(names, a)
    calib, _ = load_calib(a)
    os.makedirs(a.calib_dir, exist_ok=True)
    meta = json.load(open(meta_path(a.calib_dir))) if os.path.exists(meta_path(a.calib_dir)) else {}
    out, code = {}, 0
    for n in names:
        img = frames.get(n)
        if img is None: out[n] = {"ok": False, "error": "no frame"}; code = 2; continue
        q = frame_quality(img)
        if q["bad"]:   # never save a useless reference: a white frame would later match a white frame and pass
            out[n] = {"ok": False, "error": f"frame is {q['bad']}; not saved (exposure/gain stuck, lens covered?)", "exposure": q}; code = max(code, 1); continue
        p = ref_path(a.calib_dir, n); cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": a.url or a.from_dir or CAMS[n]["path"], "size": [int(img.shape[1]), int(img.shape[0])], "file": p, "exposure": q}
        if n == "top" and calib is not None:
            mk = marker_check(img, calib, a.threshold)
            entry["markers"] = {"detected": mk["detected"], "expected": mk["expected"], "max_norm": mk["max_norm"], "mean": [mk["mean_dx"], mk["mean_dy"]], "centres": {m: v["detected"] for m, v in mk["markers"].items()}}
        meta[n] = entry; out[n] = {"ok": True, **entry}
    json.dump(meta, open(meta_path(a.calib_dir), "w"), indent=1)
    if a.json: print(json.dumps({"command": "set", "calib_dir": a.calib_dir, "cameras": out}, indent=1))
    else:
        for n in names:
            e = out[n]
            if not e["ok"]: print(f"{n:<6} FAIL  {e['error']}"); continue
            line = f"{n:<6} saved {e['file']} ({e['size'][0]}x{e['size'][1]})"
            if "markers" in e: line += f"  markers {e['markers']['detected']}/{e['markers']['expected']} seen, max {e['markers']['max_norm']} px from the calibration"
            print(line)
    return code


def check_one(name, ref, img, calib, a):
    """Everything the check knows about one camera, as a JSON-ready dict."""
    r = {"camera": name, "reference": ref_path(a.calib_dir, name)}
    if ref is None: r.update(ok=False, status="NO REFERENCE", verdict="no reference frame; run `cam_drift.py set` first"); return r
    if img is None: r.update(ok=False, status="NO FRAME", verdict="no frame from the source"); return r
    q_new, q_ref = frame_quality(img), frame_quality(ref); r["exposure"] = {"frame": q_new, "reference": q_ref}
    if q_ref["bad"]: r.update(ok=False, status="BAD REFERENCE", verdict=f"the reference frame is {q_ref['bad']}; run `cam_drift.py set` again with the camera exposing properly"); return r
    if q_new["bad"]: r.update(ok=False, status="BAD EXPOSURE", verdict=f"frame is {q_new['bad']}: exposure/gain stuck or lens covered (check `v4l2-ctl --list-ctrls`: auto_exposure, gain); drift cannot be judged"); return r
    if (img.shape[1], img.shape[0]) != (ref.shape[1], ref.shape[0]): r["size_note"] = f"frame {img.shape[1]}x{img.shape[0]} vs reference {ref.shape[1]}x{ref.shape[0]} (resized for the comparison)"
    shift = estimate_shift(ref, img, rotation=not a.no_rotation, min_response=a.min_response)
    r["shift"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in shift.items()}
    markers = marker_check(img, calib, a.marker_threshold) if (name == "top" and calib is not None) else None
    if markers is not None: r["markers"] = markers
    r.update(combine(shift, markers, a.threshold, a.min_response, (int(ref.shape[1]), int(ref.shape[0]))))
    return r


def format_check_line(r):
    s = r.get("shift")
    head = f"{r['camera']:<6} {r['status']:<15}"
    if s is None: return f"{head} {r['verdict']}"
    rot = "" if s.get("rot_deg") is None else f" rot {s['rot_deg']:+.2f} deg"
    parts = [f"shift ({s['dx']:+.1f}, {s['dy']:+.1f}) px{rot} resp {s['response']:.2f}"]
    m = r.get("markers")
    if m is not None and m["detected"]:
        parts.append(f"markers {m['detected']}/{m['expected']} mean ({m['mean_dx']:+.1f}, {m['mean_dy']:+.1f}) px max {m['max_norm']} px")
    elif m is not None: parts.append(f"markers 0/{m['expected']}")
    return f"{head} {' | '.join(parts)} -> {r['verdict']}"


def cmd_check(a, names):
    calib, calib_path = load_calib(a)
    if "top" in names and calib is None: log(f"note: no table calibration at {calib_path}; the top camera is judged from the frame shift only")
    frames = grab_frames(names, a)
    results = []
    for n in names:
        ref = cv2.imread(ref_path(a.calib_dir, n), cv2.IMREAD_COLOR) if os.path.exists(ref_path(a.calib_dir, n)) else None
        results.append(check_one(n, ref, frames.get(n), calib, a))
    code = 0
    for r in results:
        if r["status"] in ("NO REFERENCE", "NO FRAME"): code = 2
        elif not r["ok"] and code == 0: code = 1
    if a.json:
        print(json.dumps({"command": "check", "ok": code == 0, "threshold_px": a.threshold, "min_response": a.min_response, "calib": calib_path if calib else None, "cameras": results}, indent=1))
    else:
        for r in results: print(format_check_line(r))
        print("RESULT: " + ("all cameras in place" if code == 0 else "ATTENTION: " + ", ".join(f"{r['camera']} {r['status']}" for r in results if not r["ok"])))
    return code


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("command", choices=("set", "check"))
    p.add_argument("--cams", default="top,front,wrist", help="comma list (default all three)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--url", help="dashboard base URL, frames from <url>/cam/<name>.jpg (run from the Mac: http://localhost:8765)")
    src.add_argument("--from-dir", help="read <dir>/<cam>.jpg instead of a camera")
    p.add_argument("--calib-dir", default=CALIB_DIR, help="where ref_<cam>.jpg live (default ~/so101/calib)")
    p.add_argument("--calib", help="table calibration json for the marker check (default <calib-dir>/top_table.json)")
    p.add_argument("--threshold", type=float, default=THRESHOLD_PX, help="frame shift in px that counts as moved (default 6)")
    p.add_argument("--marker-threshold", type=float, default=None, help="marker displacement in px that counts as moved (default = --threshold)")
    p.add_argument("--min-response", type=float, default=MIN_RESPONSE, help="phase-correlation response below which the comparison is unreliable (default 0.3)")
    p.add_argument("--no-rotation", action="store_true", help="skip the ECC rotation refinement")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-busy", action="store_true", help="do not touch ~/so101/.cams_busy around device reads")
    p.add_argument("--busy-flag", default=BUSY_FLAG)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.marker_threshold is None: a.marker_threshold = a.threshold
    names = [c.strip() for c in a.cams.split(",") if c.strip()]
    bad = [n for n in names if n not in CAMS]
    if bad: log(f"unknown camera(s) {bad}; choose from {', '.join(CAMS)}"); return 2
    return cmd_set(a, names) if a.command == "set" else cmd_check(a, names)


if __name__ == "__main__":
    sys.exit(main())
