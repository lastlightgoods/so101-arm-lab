#!/usr/bin/env python3
"""jog_capture.py - record (mat point, arm tip) pairs for `cam_calib armfit` while TC drives the follower's tip onto
known mat points (teleop or by hand). Each capture reads the follower joints from the wall dashboard (/api/state),
runs the SO-101 forward kinematics from jetson/cam_calib.py, and appends a pair to jetson/calib/top_armfit.json.

  jog_capture.py --marker 0              tip is on the centre of marker 0 (mat point comes from the table calibration)
  jog_capture.py --table 9 6             tip is on mat point (9, 6) in layout units (inches here)
  jog_capture.py --show                  just print the current tip position, record nothing
  jog_capture.py --list                  show recorded pairs
  jog_capture.py --drop 0                drop the pair for marker 0 (or --drop 9,6 for a table point)
  jog_capture.py --fit                   run armfit on the pairs and write arm_base/yaw into the table calibration
"""
import argparse, importlib.util, json, math, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(ROOT, "jetson", "calib")
PAIRS = os.path.join(CALIB, "top_armfit.json")
TABLE = os.path.join(CALIB, "top_table.json")
API = os.environ.get("SO101_DASH", "http://localhost:8765") + "/api/state"


def cam_calib():
    spec = importlib.util.spec_from_file_location("cam_calib", os.path.join(ROOT, "jetson", "cam_calib.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def follower_pose(max_age=2.0):
    s = json.load(urllib.request.urlopen(API, timeout=3)); f = (s.get("arms") or {}).get("follower") or {}
    if f.get("mode") != "deg" or not f.get("pos"): raise SystemExit(f"no calibrated follower pose on the dashboard (src={f.get('src')}, mode={f.get('mode')})")
    if time.time() - f.get("t", 0) > max_age: raise SystemExit(f"follower pose is {time.time() - f['t']:.1f}s old; is the arm stream up?")
    return {k: float(v) for k, v in f["pos"].items()}


def load_pairs():
    if os.path.exists(PAIRS): return json.load(open(PAIRS))
    return {"arm_units": "mm", "pairs": [], "note": "tip positions from jog_capture.py (dashboard joints -> cam_calib.fk_tip), arm frame = base_link, mm"}


def save_pairs(d): json.dump(d, open(PAIRS, "w"), indent=1); return PAIRS


def key(pr): return ("marker", pr["marker"]) if "marker" in pr else ("table", tuple(pr["table"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--marker", type=int); g.add_argument("--table", type=float, nargs=2); g.add_argument("--show", action="store_true")
    g.add_argument("--list", action="store_true"); g.add_argument("--drop"); g.add_argument("--fit", action="store_true")
    g.add_argument("--wristcam", metavar="NAME", help="save the wrist camera frame (dashboard /cam/wrist.jpg) plus the follower joints to jetson/calib/wristcam/NAME.{jpg,json} for the hand-eye solve")
    ap.add_argument("--hold-out", nargs="*", default=[], help="with --fit: marker ids or x,y table points to leave out of the fit and predict instead")
    ap.add_argument("--keep-offsets", action="store_true", help="with --fit: hold the joint offsets already in the table calibration, solve only base x/y/yaw and table z (3 points suffice)")
    ap.add_argument("--reset", action="store_true", help="archive the current pairs file and start a new one (after the mat or arm moved)")
    a = ap.parse_args(argv); cc = cam_calib()
    if a.reset and os.path.exists(PAIRS):
        arch = PAIRS.replace(".json", time.strftime("_%Y%m%d_%H%M%S.json")); os.rename(PAIRS, arch); print(f"archived old pairs to {arch}")
    d = load_pairs()
    a.hold_out = [("marker", int(h)) if h.isdigit() else ("table", tuple(float(v) for v in h.split(","))) for h in a.hold_out]
    if a.list:
        for pr in d["pairs"]: print(json.dumps({k: pr[k] for k in pr if k != "joints"}))
        print(f"{len(d['pairs'])} pair(s) in {PAIRS}"); return 0
    if a.drop:
        k = ("marker", int(a.drop)) if a.drop.isdigit() else ("table", tuple(float(v) for v in a.drop.split(",")))
        before = len(d["pairs"]); d["pairs"] = [pr for pr in d["pairs"] if key(pr) != k]; save_pairs(d); print(f"dropped {before - len(d['pairs'])}"); return 0
    if a.fit:
        table = json.load(open(TABLE)); tab, arm = cc.load_pairs(PAIRS, table)
        joints = [pr["joints"] for pr in d["pairs"]]
        if a.hold_out:
            keep = [i for i, pr in enumerate(d["pairs"]) if key(pr) not in a.hold_out]; test = [i for i in range(len(joints)) if i not in keep]
        else: keep, test = list(range(len(joints))), []
        fixed = ((table.get("arm_fit") or {}).get("joint_offsets_deg") if a.keep_offsets else None)
        if a.keep_offsets and not fixed: raise SystemExit("--keep-offsets: the table calibration has no fitted joint offsets yet")
        fit = cc.fit_arm_and_offsets([joints[i] for i in keep], tab[keep], fixed_offsets=fixed)
        base, yaw = fit["arm_base_mm"], fit["arm_yaw_deg"]
        table["arm_base_mm"], table["arm_yaw_deg"] = base, yaw
        rolls = [joints[i].get("wrist_roll", 0.0) for i in keep]
        table["arm_fit"] = {**fit, "source": PAIRS, "fitted": time.strftime("%Y-%m-%d %H:%M:%S"), "points": [key(d["pairs"][i]) for i in keep],
                            "wrist_roll_deg": round(sum(rolls) / len(rolls), 2), "wrist_roll_spread_deg": round(max(rolls) - min(rolls), 2),
                            "note": "contact point moves with wrist roll (fixed jaw is off the roll axis): jog and command with wrist_roll near wrist_roll_deg"}
        if max(rolls) - min(rolls) > 10: print(f"warning: wrist roll varied {max(rolls) - min(rolls):.0f} deg across the jogged points; the contact point moves with roll, keep it constant", file=sys.stderr)
        out = {"saved": TABLE, "arm_base_mm": [round(v, 1) for v in base], "arm_yaw_deg": round(yaw, 2), "table_z_mm": round(fit["table_z_mm"], 1), "wrist_roll_deg": table["arm_fit"]["wrist_roll_deg"],
               "joint_offsets_deg": {k: round(v, 2) for k, v in fit["joint_offsets_deg"].items()}, "residual_xy_mm": [round(v, 2) for v in fit["residual_xy_mm"]],
               "residual_z_mm": [round(v, 2) for v in fit["residual_z_mm"]], "rigid_only_residual_mm": [round(v, 1) for v in fit["rigid_only_residual_mm"]], "points": len(keep)}
        if test:   # predict the held-out points with the fitted offsets and base: the honest accuracy number
            calib = {"arm_base_mm": base, "arm_yaw_deg": yaw}; errs = {}
            for i in test:
                tip, _ = cc.fk_tip({k: v + fit["joint_offsets_deg"].get(k, 0.0) for k, v in joints[i].items()}); ax, ay = cc.table_to_arm(calib, *tab[i])
                errs[str(key(d["pairs"][i]))] = {"xy_mm": round(math.hypot(tip[0] - ax, tip[1] - ay), 2), "z_mm": round(float(tip[2] - fit["table_z_mm"]), 2)}
            out["held_out"] = errs; table["arm_fit"]["held_out"] = errs
        json.dump(table, open(TABLE, "w"), indent=1)
        if table.get("units") == "in": out["arm_base_in"] = [round(v / 25.4, 3) for v in base]
        print(json.dumps(out)); return 0
    if a.wristcam:
        import urllib.request as _u
        wdir = os.path.join(CALIB, "wristcam"); os.makedirs(wdir, exist_ok=True)
        pose = follower_pose(); jpg = _u.urlopen(os.environ.get("SO101_DASH", "http://localhost:8765") + f"/cam/wrist.jpg?t={time.time():f}", timeout=5).read()
        open(os.path.join(wdir, a.wristcam + ".jpg"), "wb").write(jpg)
        json.dump({"name": a.wristcam, "joints": pose, "t": time.time(), "when": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(wdir, a.wristcam + ".json"), "w"), indent=1)
        seen = {}
        try:
            import cv2, numpy as np
            im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR); seen = cc.detect_markers(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        except Exception as e: print("marker check skipped:", e, file=sys.stderr)
        print(json.dumps({"saved": os.path.join(wdir, a.wristcam), "markers_in_wrist_view": sorted(seen), "enough": len(seen) >= 3, "joints": {k: round(v, 1) for k, v in pose.items()}}))
        return 0
    if a.marker is None and a.table is None and not a.show:
        if a.reset: return 0
        ap.error("one of --marker, --table, --show, --list, --drop, --fit is required")
    pose = follower_pose(); tip, point = cc.fk_tip(pose)
    down = math.degrees(math.acos(max(-1.0, min(1.0, -point[2]))))
    rec = {"joints": {k: round(v, 2) for k, v in pose.items()}, "tip_mm": [round(float(v), 1) for v in tip], "tip_in": [round(float(v) / 25.4, 3) for v in tip], "gripper_down_deg": round(down, 1), "t": time.strftime("%H:%M:%S")}
    if a.show: print(json.dumps(rec)); return 0
    pr = {"arm": [float(tip[0]), float(tip[1])], **rec}
    if a.marker is not None: pr["marker"] = a.marker
    else: pr["table"] = list(a.table)
    d["pairs"] = [p for p in d["pairs"] if key(p) != key(pr)] + [pr]; save_pairs(d)
    print(json.dumps({"recorded": key(pr), "tip_in": rec["tip_in"], "tip_z_mm": rec["tip_mm"][2], "gripper_down_deg": down, "pairs": len(d["pairs"])}))
    if down > 40: print(f"note: gripper is {down:.0f} deg off vertical; a steeper approach gives a cleaner tip contact", file=sys.stderr)
    return 0


if __name__ == "__main__": sys.exit(main())
