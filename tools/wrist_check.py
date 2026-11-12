#!/usr/bin/env python3
"""wrist_check.py - does the wrist camera agree with the commanded nut position? A check, never a controller.

At the pick loop's hover the follower's REACHED joints (pick_seq's readback) and the hand-eye pose (jetson/calib/wrist_campose.json)
say where the commanded nut point must appear in the wrist frame. The nut is then looked for near that pixel as a BRIGHT RING WITH A
DARK CENTRE (an M4 nut 147 mm below the lens is ~36 px across: a bright annulus round a dark hole; a dark-blob finder picked the
gripper's shadow instead, 10 Nov 2026), its centre is back-projected onto the mat plane and the offset to the command is reported,
with an annotated JPEG (expected = yellow, found = red) in outputs/run/. The grasp itself trusts the servo readback
(tools/pick_loop.py, closed loop); this only says whether camera, arm fit and command agree (10 Nov: within ~10 px at hover).
Nothing here raises into the pick loop: no ring, no frame, no pose -> None.

  tools/wrist_check.py --image frame.jpg --joints '{"shoulder_pan": 14.9, ...}' --nut 11.44 6.26 [--out outputs/run/wrist_check.jpg]
  (offline, on a saved frame; `pick_loop.py --wrist-check` runs the same on the live frame at hover)
"""
import argparse, importlib.util, json, math, os, sys, time
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
RING_R_PX = (12, 26)      # nut radius in pixels at 120-180 mm depth with fx 529 (M4: 7 mm across flats -> 18 px at 147 mm)
RING_SEARCH_PX = 80       # the open-loop command was 5-8 mm off (15-30 px at hover); 80 px keeps the receptacle's screw holes out
RING_MIN_SCORE = 0.35     # TM_CCOEFF_NORMED of the annulus template; the real nut scored 0.49, a plain mat square < 0.2


def cam_calib():
    spec = importlib.util.spec_from_file_location("cam_calib", os.path.join(ROOT, "jetson", "cam_calib.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def cam_pose_base(cc, table, wpose, joints):
    """(R, o): the wrist camera's axes and centre in base_link, metres, for the follower joints (with the arm fit's zero offsets)."""
    off = (table.get("arm_fit") or {}).get("joint_offsets_deg") or {}
    T = cc.fk_frames({k: float(joints[k]) for k in ARM}, off)["gripper_link"]
    R_gc = cv2.Rodrigues(np.asarray(wpose["rotation"]["rvec"], np.float64))[0]; t_gc = np.asarray(wpose["position_mm"], np.float64) / 1000.0
    return T[:3, :3] @ R_gc, T[:3, :3] @ t_gc + T[:3, 3]


def table_z_mm(table): return float((table.get("arm_fit") or {}).get("table_z_mm", 0.0) or 0.0)


def mat_to_base_mm(cc, table, x_in, y_in, height_mm=0.0):
    X, Y = cc.table_to_arm(table, x_in * 25.4, y_in * 25.4); return np.array([X, Y, table_z_mm(table) + height_mm])


def base_to_mat_in(cc, table, xy_mm):
    x, y = cc.arm_to_table(table, float(xy_mm[0]), float(xy_mm[1])); return x / 25.4, y / 25.4


def project_base_point(cc, table, wpose, P_mm, joints):
    """(pixel (2,), depth_mm) of a base_link point in the wrist frame; depth < 0 means behind the camera."""
    R, o = cam_pose_base(cc, table, wpose, joints); pc = R.T @ (np.asarray(P_mm, np.float64) / 1000.0 - o)
    uv = cc.project_points(pc.reshape(1, 3), np.asarray(wpose["camera_matrix"]), np.asarray(wpose["dist_coeffs"]))[0]
    return uv, float(pc[2] * 1000.0)


def backproject_pixel(cc, table, wpose, uv, joints, height_mm=0.0):
    """base_link mm of the point under pixel `uv` that lies `height_mm` above the mat plane, or None when the ray misses it."""
    n = cv2.undistortPoints(np.array([[uv]], np.float64), np.asarray(wpose["camera_matrix"], np.float64), np.asarray(wpose["dist_coeffs"], np.float64)).ravel()
    R, o = cam_pose_base(cc, table, wpose, joints); d = R @ np.array([n[0], n[1], 1.0]); z = (table_z_mm(table) + height_mm) / 1000.0
    if abs(d[2]) < 1e-9: return None
    s = (z - o[2]) / d[2]
    return None if s <= 0 else (o + s * d) * 1000.0


def find_ring(gray, near_uv, radius_px=RING_SEARCH_PX, r_range=RING_R_PX, min_score=RING_MIN_SCORE, rect=None, hole_max=None):
    """The bright ring with a dark centre nearest `near_uv`: {"uv", "r_px", "score", "dist_px"} or None. A mean-free annulus template
    (ring +1, hole -1, surroundings 0) is correlated (TM_CCOEFF_NORMED) over the ROI at every radius in r_range; a dark blob (the
    gripper's shadow) anti-correlates, a uniform mat square scores ~0, so only a ring can pass min_score. `rect` (x0, y0, x1, y1), when given, also masks every ring centre outside it (None: the disc alone); `hole_max`, when given, drops rings whose hole is brighter than that mean grey level (a nut's hole is dark, a bright jaw edge is not)."""
    g = (gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)).astype(np.float32); h, w = g.shape; u, v = float(near_uv[0]), float(near_uv[1])
    pad = r_range[1] + 6; x0, y0 = int(max(0, u - radius_px - pad)), int(max(0, v - radius_px - pad)); x1, y1 = int(min(w, u + radius_px + pad)), int(min(h, v + radius_px + pad))
    if x1 - x0 < 2 * pad or y1 - y0 < 2 * pad: return None
    roi = cv2.GaussianBlur(g[y0:y1, x0:x1], (3, 3), 0); best = None
    for r in range(r_range[0], r_range[1] + 1, 2):
        s = 2 * r + 7; c = s // 2; yy, xx = np.mgrid[:s, :s]; d = np.hypot(xx - c, yy - c)
        t = np.zeros((s, s), np.float32); t[(d <= r) & (d >= 0.45 * r)] = 1.0; t[d < 0.35 * r] = -1.0
        if roi.shape[0] < s or roi.shape[1] < s: continue
        m = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.mgrid[:m.shape[0], :m.shape[1]]; cu = xs + c + x0; cv_ = ys + c + y0; dist = np.hypot(cu - u, cv_ - v)
        sc = m - 0.002 * dist; sc[dist > radius_px] = -9.0   # why: among rings prefer the one where the model says the nut is, mildly (0.16 over the full radius)
        if hole_max is not None: kh = (d < 0.35 * r).astype(np.float32); sc[cv2.matchTemplate(roi, kh / kh.sum(), cv2.TM_CCORR) > hole_max] = -9.0   # why: on the closed jaws the bright jaw edge scored 0.40-0.42 as a ring (11 Nov 2026, hole mean grey 204-222); the held nut's hole is 19-56
        if rect is not None: sc[(cu < rect[0]) | (cu > rect[2]) | (cv_ < rect[1]) | (cv_ > rect[3])] = -9.0   # why: jaw_check wants the best ring inside the jaw window, not the best in a disc that a false ring on the jaw edge can win (11 Nov 2026)
        i = np.unravel_index(int(np.argmax(sc)), sc.shape)
        if m[i] >= min_score and (best is None or sc[i] > best[0]): best = (float(sc[i]), {"uv": [float(cu[i]), float(cv_[i])], "r_px": r, "score": round(float(m[i]), 3), "dist_px": round(float(dist[i]), 1)})
    return best[1] if best else None


def annotate(img, uv_exp, found, text):
    out = img.copy(); cv2.circle(out, (int(round(uv_exp[0])), int(round(uv_exp[1]))), 10, (0, 255, 255), 2)
    if found: cv2.circle(out, (int(round(found["uv"][0])), int(round(found["uv"][1]))), int(found["r_px"]), (0, 0, 255), 2)
    cv2.putText(out, text, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA); return out


def wrist_check(cc, table, wpose, nut_in, joints, jpeg, out_path=None, height_mm=0.0):
    """One check: expected pixel of the commanded nut, the ring found near it, its mat position and the offset (mm). Writes the annotated
    JPEG to out_path when given. Returns the dict, or None for anything that is not a clean result (no ring, bad frame, no pose)."""
    try:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if isinstance(jpeg, (bytes, bytearray)) else jpeg
        if img is None or wpose is None or not table.get("arm_base_mm"): return None
        uv_exp, depth = project_base_point(cc, table, wpose, mat_to_base_mm(cc, table, nut_in[0], nut_in[1], height_mm), joints)
        if depth <= 0 or not np.all(np.isfinite(uv_exp)): return None
        found = find_ring(img, uv_exp); res = None
        if found:
            P = backproject_pixel(cc, table, wpose, found["uv"], joints, height_mm)
            if P is not None:
                mx, my = base_to_mat_in(cc, table, P[:2]); off = ((mx - nut_in[0]) * 25.4, (my - nut_in[1]) * 25.4)
                res = {"uv_expected": [round(float(v), 1) for v in uv_exp], "uv_found": [round(v, 1) for v in found["uv"]], "r_px": found["r_px"], "score": found["score"], "px_off": found["dist_px"],
                       "depth_mm": round(depth, 1), "nut_seen_in": [round(mx, 3), round(my, 3)], "offset_mm": [round(off[0], 1), round(off[1], 1)], "offset_norm_mm": round(math.hypot(*off), 1)}
        if out_path:
            text = f"expected ({uv_exp[0]:.0f},{uv_exp[1]:.0f}) depth {depth:.0f} mm  " + (f"ring ({found['uv'][0]:.0f},{found['uv'][1]:.0f}) r{found['r_px']} score {found['score']:.2f}  nut off ({res['offset_mm'][0]:+.1f},{res['offset_mm'][1]:+.1f}) mm" if res else "no ring near the expected pixel")
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True); cv2.imwrite(out_path, annotate(img, uv_exp, found, text))
        return res
    except Exception as e:   # why: a check must never take the pick down with it; the loop prints the note and grasps on the readback
        print(json.dumps({"wrist_check": "failed", "err": f"{type(e).__name__}: {e}"}), file=sys.stderr, flush=True); return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="a saved wrist frame (JPEG/PNG)"); ap.add_argument("--joints", required=True, help="JSON: the follower joints the frame was taken at (LeRobot deg)")
    ap.add_argument("--nut", type=float, nargs=2, required=True, metavar=("X_IN", "Y_IN"), help="the commanded nut point, mat inches")
    ap.add_argument("--table", default=os.path.join(ROOT, "jetson", "calib", "top_table.json")); ap.add_argument("--wrist-pose", default=os.path.join(ROOT, "jetson", "calib", "wrist_campose.json"))
    ap.add_argument("--height-mm", type=float, default=0.0, help="height of the ring above the mat plane (0: the mat itself)")
    ap.add_argument("--out", default=None, help="annotated JPEG (default outputs/run/wrist_check_<time>.jpg)")
    a = ap.parse_args(argv); cc = cam_calib(); table = json.load(open(a.table)); wpose = json.load(open(a.wrist_pose))
    joints = json.loads(a.joints) if a.joints.strip().startswith("{") else json.load(open(a.joints))
    out = a.out or os.path.join(ROOT, "outputs", "run", time.strftime("wrist_check_%Y%m%d-%H%M%S.jpg"))
    res = wrist_check(cc, table, wpose, a.nut, joints, open(a.image, "rb").read(), out, a.height_mm)
    print(json.dumps({"ok": res is not None, "annotated": out, **(res or {})})); return 0 if res else 2


if __name__ == "__main__": sys.exit(main())
