#!/usr/bin/env python3
"""pick_loop.py - repeat "pick the nut, place it on the mat centre" and get better each time (iterative learning).

Per run: find the nut in the overhead frame (dark blob near its last known spot), build a plan from /api/ik, bias every waypoint by
the servo sag measured at the SAME waypoint last time (iterative learning control), shift the place target by the camera-measured
landing error of the previous run (outcome learning), run jetson/pick_seq.py over ssh in one bus session per stage, then measure
where the nut landed. Stops when the nut is not found, not picked up, or the error grows.

Closed loop (default, docs/pick-loop.md): the open-loop command reaches the hover 5-8 mm short along the reach direction (10 Nov 2026),
but the servo readback plus the arm fit is good to a millimetre, so the pick is three staged pick_seq runs: (a) hover, read back the
reached joints, FK them to the mat, correction c1 = commanded - reached; (b) corrected hover + grasp height with the jaws open, read
back again, c2, and the tip height h decides whether the grasp target is lowered (h > 6 mm: by min(h - 3, 6) mm through /api/ik);
(c) the pick itself at the twice-corrected point. Two corrections put the final grasp within 0.1 mm of the command by FK; the nut
landed 6.2 mm from the target. `--open-loop` is the single-run plan of before.

  tools/pick_loop.py --runs 4 [--target 9 6] [--open-loop] [--jaw-open 35] [--wrist-check] [--state jetson/calib/pick_state.json]
State (biases, place offset, history) persists in --state so later sessions start where this one ended.
Every pick is also one MLflow run in experiment `so101-picks` (jetson/labtrack.py: params, landing error, per-waypoint sag, the stage
corrections, the overhead frame at landing, the plan and the state file); with no tracking server the loop runs exactly as before.
"""
import argparse, json, math, os, subprocess, time, urllib.request
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH = os.environ.get("SO101_DASH", "http://localhost:8765")
REST = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 72.0, "wrist_roll": 0.5}
GRASP_H_IN = 0.12
# why: jaws at 35 grasp the nut; at 45 the SAME twice-corrected pose failed twice (10 Nov 2026): the moving jaw's closing arc flicked the nut 5-13 mm before the fixed jaw met it
JAW_OPEN = 35
WP = [("hover_nut", "nut", 1.5, JAW_OPEN, 1.6, 0.4), ("grasp", "nut", GRASP_H_IN, JAW_OPEN, 0.9, 0.5), ("close", "nut", GRASP_H_IN, 0, 0.6, 0.4),
      ("lift", "nut", 1.5, 0, 0.8, 0.2), ("hover_tgt", "tgt", 1.5, 0, 1.2, 0.3), ("place", "tgt", 0.3, 0, 0.9, 0.5),
      ("open", "tgt", 0.3, JAW_OPEN, 0.5, 0.4), ("lift2", "tgt", 1.5, JAW_OPEN, 0.7, 0.2), ("rest", None, 0, 0, 1.8, 0.3)]
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
STAGES = (("hover_nut",), ("hover_nut", "grasp"), ("grasp", "close", "lift", "hover_tgt", "place", "open", "lift2", "rest"))
SPEED = {"closed": 50.0, "open": 80.0}   # why: the staged runs start each stage from a standstill close to the nut; 50 deg/s settles cleaner for the readback


def api(path):
    return json.load(urllib.request.urlopen(DASH + path, timeout=10))


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "jetson" if name != "wrist_check" else "tools", f"{name}.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def cam_calib(): return _load("cam_calib")


def wrist_mod(): return _load("wrist_check")


def labtrack(): return _load("labtrack").LabTrack("so101-picks")


def fetch_top():
    """One fresh overhead JPEG from the dashboard (bytes)."""
    return urllib.request.urlopen(DASH + f"/cam/top.jpg?t={time.time():f}", timeout=5).read()


def fetch_wrist():
    """One fresh wrist-camera JPEG from the dashboard (bytes), or None when the feed is down (the check is optional)."""
    try: return urllib.request.urlopen(DASH + f"/cam/wrist.jpg?t={time.time():f}", timeout=5).read()
    except Exception: return None


def find_nut(cc, table, near_in, radius_px=70, jpeg=None):
    """Dark blob nearest to `near_in` (mat inches) in a fresh overhead frame (or the given JPEG bytes) -> (mat_in, pixel, area) or None."""
    img = cv2.imdecode(np.frombuffer(fetch_top() if jpeg is None else jpeg, np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); cu, cv_ = cc.table_to_pixel(table, near_in[0] * 25.4, near_in[1] * 25.4)
    x0, y0 = max(0, int(cu) - radius_px), max(0, int(cv_) - radius_px); roi = g[y0:y0 + 2 * radius_px, x0:x0 + 2 * radius_px]
    m = (roi < 60).astype(np.uint8); n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    cands = [(math.hypot(x0 + cen[i][0] - cu, y0 + cen[i][1] - cv_), int(st[i, cv2.CC_STAT_AREA]), x0 + cen[i][0], y0 + cen[i][1]) for i in range(1, n) if 8 <= st[i, cv2.CC_STAT_AREA] <= 300]
    if not cands: return None
    _, area, u, v = min(cands); X, Y = cc.pixel_to_table(table, u, v)
    return [X / 25.4, Y / 25.4], [u, v], area


def build_plan(nut_in, tgt_in, biases, names=None, jaw_open=JAW_OPEN, grasp_h_in=None):
    """The waypoints of WP (all, or only `names`, in WP order) through /api/ik with the learned biases; `jaw_open` replaces every
    open-jaw value, `grasp_h_in` the grasp height (the closed loop lowers it when the readback says the tip sits high)."""
    def ik(x, y, h):
        r = api(f"/api/ik?x_in={x:.4f}&y_in={y:.4f}&h_in={h:.4f}")
        if not r.get("ok"): raise SystemExit(f"ik failed for ({x:.2f},{y:.2f},{h}): {r.get('err')}")
        return r["joints_deg"]
    wps = []
    for name, which, h, grip, secs, settle in WP:
        if names is not None and name not in names: continue
        if name == "grasp" and grasp_h_in is not None: h = grasp_h_in
        j = dict(REST) if which is None else ik(*(nut_in if which == "nut" else tgt_in), h)
        b = biases.get(name, {}); cmd = {k: j[k] + b.get(k, 0.0) for k in j}; cmd["wrist_flex"] = max(-93.0, min(93.0, cmd["wrist_flex"]))
        wps.append({"name": name, "joints": cmd, "desired": j, "gripper": jaw_open if grip else 0, "seconds": secs, "settle": settle, "gripper_only": name in ("close", "open")})
    return {"waypoints": wps}


def run_plan(plan, max_deg_s):
    p = "/tmp/pick_plan.json"; open("/tmp/pick_plan_local.json", "w").write(json.dumps(plan))
    subprocess.run(["scp", "-q", "/tmp/pick_plan_local.json", f"jetson:{p}"], check=True)
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "jetson", f"cd ~/so101 && source env.sh >/dev/null 2>&1; ~/so101/.venv/bin/python pick_seq.py --plan {p} --correct 0 --max-deg-s {max_deg_s}"], capture_output=True, text=True, timeout=180)
    rows = []
    for line in out.stdout.splitlines():
        try: d = json.loads(line)
        except ValueError: continue
        if "wp" in d: rows.append(d)
    if not rows: raise SystemExit("pick_seq produced no waypoint rows:\n" + out.stdout[-800:] + out.stderr[-800:])
    return rows


def fk_mat(cc, table, joints):
    """Where the tip really is for read-back joints: (mat x_in, mat y_in, height above the mat in mm), through the arm fit's zero offsets
    and tip offset. This is the measurement the closed loop corrects on (10 Nov 2026: it predicted the nut's wrist pixel within ~10 px)."""
    fit = table["arm_fit"]; off = fit.get("joint_offsets_deg") or {}
    tip, _ = cc.fk_tip({k: float(joints[k]) + float(off.get(k, 0.0)) for k in ARM}, tip_offset_mm=fit.get("tip_offset_mm"))
    x, y = cc.arm_to_table(table, float(tip[0]), float(tip[1])); return x / 25.4, y / 25.4, float(tip[2]) - float(fit.get("table_z_mm", 0.0) or 0.0)


def reached_joints(rows, name):
    r = [r for r in rows if r["wp"] == name][-1]; return {k: float(r["reached"][k]) for k in ARM}


def _mm(c): return [round(c[0] * 25.4, 1), round(c[1] * 25.4, 1)]


def closed_loop_pick(cc, table, nut_in, place_in, biases, a, run, say, wpose=None, check_path=None):
    """The three staged pick_seq runs of the closed loop (module docstring). Returns (final plan, its rows, info) where info carries
    corr1_mm, corr2_mm, grasp_h_mm, grasp_lowered_mm, final_residual_mm and wrist_check for the history record and MLflow."""
    def stage(xy, names, grasp_h_in=None):
        p = build_plan(list(xy), list(place_in), biases, names=names, jaw_open=a.jaw_open, grasp_h_in=grasp_h_in); return p, run_plan(p, a.max_deg_s)
    # (a) hover over the commanded point; the readback says where the tip really went
    _, r1 = stage(nut_in, STAGES[0]); j1 = reached_joints(r1, "hover_nut"); x, y, h = fk_mat(cc, table, j1)
    c1 = (nut_in[0] - x, nut_in[1] - y); info = {"corr1_mm": _mm(c1), "wrist_check": None}
    say({"run": run, "stage": "hover", "cmd_in": [round(v, 3) for v in nut_in], "reached_in": [round(x, 3), round(y, 3)], "h_mm": round(h, 1), "corr1_mm": info["corr1_mm"]})
    if a.wrist_check:   # a look, not a correction: does the camera see the nut where the command says it is?
        info["wrist_check"] = wrist_mod().wrist_check(cc, table, wpose, nut_in, j1, fetch_wrist() or b"", check_path) if wpose else None
        say({"run": run, "stage": "wrist_check", **(info["wrist_check"] or {"ring": None, "annotated": check_path if wpose else None})})
    xy1 = [nut_in[0] + c1[0], nut_in[1] + c1[1]]
    # (b) corrected hover, then down to grasp height with the jaws open; read back again, and measure the tip height
    _, r2 = stage(xy1, STAGES[1]); j2 = reached_joints(r2, "grasp"); x, y, h = fk_mat(cc, table, j2)
    c2 = (nut_in[0] - x, nut_in[1] - y); xy2 = [xy1[0] + c2[0], xy1[1] + c2[1]]
    lower = min(h - 3.0, 6.0) if h > 6.0 else 0.0   # why: the jaws must reach the mat to get under the nut; the cap keeps the IK target within 3 mm below the mat plane the readback contradicts
    info.update({"corr2_mm": _mm(c2), "grasp_h_mm": round(h, 1), "grasp_lowered_mm": round(lower, 1)})
    say({"run": run, "stage": "grasp_pose", "cmd_in": [round(v, 3) for v in xy1], "reached_in": [round(x, 3), round(y, 3)], "h_mm": round(h, 1), "corr2_mm": info["corr2_mm"], "lowered_mm": round(lower, 1)})
    # (c) the pick at the twice-corrected point: grasp, close, lift, carry, place, open, lift, rest
    plan, rows = stage(xy2, STAGES[2], grasp_h_in=(GRASP_H_IN - lower / 25.4) if lower else None)
    j3 = reached_joints(rows, "grasp"); x, y, h = fk_mat(cc, table, j3); res = (nut_in[0] - x, nut_in[1] - y)
    info["final_residual_mm"] = round(math.hypot(*res) * 25.4, 2); info["grasp_cmd_in"] = [round(v, 3) for v in xy2]
    say({"run": run, "stage": "final", "cmd_in": info["grasp_cmd_in"], "reached_in": [round(x, 3), round(y, 3)], "h_mm": round(h, 1), "residual_mm": _mm(res), "sag_deg": {r["wp"]: r["max_err_deg"] for r in rows}})
    return plan, rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=4); ap.add_argument("--target", type=float, nargs=2, default=[9.0, 6.0])
    ap.add_argument("--max-deg-s", type=float, default=None, help="pick_seq speed cap (default 50 closed loop, 80 open loop)")
    ap.add_argument("--state", default=os.path.join(ROOT, "jetson", "calib", "pick_state.json")); ap.add_argument("--gain", type=float, default=0.7); ap.add_argument("--outcome-gain", type=float, default=0.5)
    ap.add_argument("--alt", type=float, nargs=2, default=[6.0, 5.0], help="second target; runs alternate alt, target, alt, target ... so every pick and place is unambiguous; the LAST run always goes to --target")
    ap.add_argument("--closed-loop", dest="closed_loop", action="store_true", default=True, help="hover, read back, correct; grasp pose, read back, correct; pick (default)")
    ap.add_argument("--open-loop", dest="closed_loop", action="store_false", help="one pick_seq run of the whole plan, no readback correction")
    ap.add_argument("--jaw-open", type=int, default=JAW_OPEN, help=f"gripper opening for hover/grasp/open/lift2 (default {JAW_OPEN}; 45 flicks the nut away, see docs/pick-loop.md)")
    ap.add_argument("--wrist-check", action="store_true", help="closed loop: at hover, look for the nut in the wrist frame near its predicted pixel and report the offset (check only; annotated JPEG in outputs/run/)")
    a = ap.parse_args(); cc = cam_calib(); table = json.load(open(os.path.join(ROOT, "jetson", "calib", "top_table.json"))); track = labtrack()
    mode = "closed" if a.closed_loop else "open"; a.max_deg_s = a.max_deg_s if a.max_deg_s is not None else SPEED[mode]
    if a.closed_loop and not (table.get("arm_fit") and table.get("arm_base_mm")): raise SystemExit("closed loop needs the arm fit in jetson/calib/top_table.json (tools/jog_capture.py --fit); or run --open-loop")
    wpath = os.path.join(ROOT, "jetson", "calib", "wrist_campose.json"); wpose = json.load(open(wpath)) if a.wrist_check and os.path.exists(wpath) else None
    if a.wrist_check and wpose is None: print(json.dumps({"note": "no wrist_campose.json; --wrist-check skipped"}), flush=True)
    state = json.load(open(a.state)) if os.path.exists(a.state) else {"biases": {}, "place_offsets_in": {}, "history": []}
    state.setdefault("place_offsets_in", {}); near = state["history"][-1]["landed_in"] if state["history"] and state["history"][-1].get("landed_in") else list(a.target)
    session = time.strftime("%Y%m%d-%H%M%S"); say = lambda d: print(json.dumps(d), flush=True)
    for run in range(1, a.runs + 1):
        tgt = list(a.target) if (run == a.runs or run % 2 == 0) else list(a.alt); key = f"{tgt[0]:.2f},{tgt[1]:.2f}"
        off = state["place_offsets_in"].setdefault(key, [0.0, 0.0])
        found = find_nut(cc, table, near) or find_nut(cc, table, list(a.target), radius_px=140) or find_nut(cc, table, list(a.alt), radius_px=140)
        if not found: print(json.dumps({"run": run, "stop": "nut not found near", "near": near})); break
        nut_in, nut_px, area = found
        place_in = [tgt[0] + off[0], tgt[1] + off[1]]
        with track.run(f"pick-{session}-r{run}", tags={"host": "mac", "run": run, "nut_in": f"{nut_in[0]:.2f},{nut_in[1]:.2f}", "target_in": key, "session": session}) as tr:   # why: one MLflow run per pick; a no-op when there is no server
            tr.params({"target_in": f"{a.target[0]:.2f},{a.target[1]:.2f}", "alt_in": f"{a.alt[0]:.2f},{a.alt[1]:.2f}", "place_in": f"{place_in[0]:.3f},{place_in[1]:.3f}", "gain": a.gain, "outcome_gain": a.outcome_gain,
                       "max_deg_s": a.max_deg_s, "runs": a.runs, "state": os.path.relpath(a.state, ROOT) if a.state.startswith(ROOT) else a.state, "dash": DASH, "mode": mode, "jaw_open": a.jaw_open, "wrist_check": int(a.wrist_check)})
            t0 = time.time(); check_path = os.path.join(ROOT, "outputs", "run", f"wrist_check_{session}_r{run}.jpg")
            if a.closed_loop: plan, rows, info = closed_loop_pick(cc, table, nut_in, place_in, state["biases"], a, run, say, wpose, check_path); tr.json(plan, "plan.json")
            else: plan = build_plan(nut_in, place_in, state["biases"], jaw_open=a.jaw_open); tr.json(plan, "plan.json"); rows = run_plan(plan, a.max_deg_s); info = {}
            dt = time.time() - t0
            # learn the servo sag per waypoint: bias += gain * (target - reached), capped; closed loop: on the FINAL stage only (its waypoints are the ones the corrected command drove)
            desired = {w["name"]: w["desired"] for w in plan["waypoints"]}
            for r in rows:
                if r["wp"] in ("close", "open"): continue
                b = state["biases"].setdefault(r["wp"], {})
                for k in ARM: b[k] = max(-6.0, min(6.0, b.get(k, 0.0) + a.gain * (desired[r["wp"]][k] - r["reached"][k])))   # drive DESIRED - reached to zero, not command - reached
                r["max_err_deg"] = round(max(abs(desired[r["wp"]][k] - r["reached"][k]) for k in ARM), 2)
            time.sleep(1.0)
            landing_jpg = fetch_top(); landed = find_nut(cc, table, place_in, radius_px=60, jpeg=landing_jpg)
            before_mm = math.hypot(nut_in[0] - tgt[0], nut_in[1] - tgt[1]) * 25.4
            err_mm = math.hypot(landed[0][0] - tgt[0], landed[0][1] - tgt[1]) * 25.4 if landed else None
            picked = landed is not None and (err_mm < before_mm - 5.0 or before_mm < 40.0)   # it got closer, or it was already there and stayed within the search radius
            if landed and picked:
                off[0] = max(-0.6, min(0.6, off[0] + a.outcome_gain * (tgt[0] - landed[0][0]))); off[1] = max(-0.6, min(0.6, off[1] + a.outcome_gain * (tgt[1] - landed[0][1])))
            wc = info.get("wrist_check")
            rec = {"run": run, "target_in": tgt, "nut_in": [round(v, 2) for v in nut_in], "picked": picked, "landed_in": [round(v, 2) for v in landed[0]] if landed else None, "err_mm": round(err_mm, 1) if err_mm is not None else None,
                   "seconds": round(dt, 1), "sag_deg": {r["wp"]: r["max_err_deg"] for r in rows}, "place_offset_in": [round(v, 3) for v in off], "mode": mode, "jaw_open": a.jaw_open,
                   "corr1_mm": info.get("corr1_mm"), "corr2_mm": info.get("corr2_mm"), "grasp_h_mm": info.get("grasp_h_mm"), "grasp_lowered_mm": info.get("grasp_lowered_mm"), "final_residual_mm": info.get("final_residual_mm"),
                   "wrist_offset_mm": wc["offset_mm"] if wc else None, "t": time.strftime("%H:%M:%S")}
            state["history"].append(rec); json.dump(state, open(a.state, "w"), indent=1); print(json.dumps(rec), flush=True)
            c1, c2 = info.get("corr1_mm"), info.get("corr2_mm")
            tr.metrics({"landing_err_mm": err_mm, "before_mm": before_mm, "seconds": dt, "picked": int(picked), "found_at_landing": int(landed is not None), **{f"sag_{r['wp']}": r["max_err_deg"] for r in rows if "max_err_deg" in r},
                        "corr1_mm": math.hypot(*c1) if c1 else None, "corr1_x_mm": c1[0] if c1 else None, "corr1_y_mm": c1[1] if c1 else None, "corr2_mm": math.hypot(*c2) if c2 else None, "corr2_x_mm": c2[0] if c2 else None, "corr2_y_mm": c2[1] if c2 else None,
                        "grasp_h_mm": info.get("grasp_h_mm"), "grasp_lowered_mm": info.get("grasp_lowered_mm"), "final_residual_mm": info.get("final_residual_mm"), "wrist_offset_mm": wc["offset_norm_mm"] if wc else None, "jaw_open": a.jaw_open})
            tr.file("landing.jpg", landing_jpg, "frames"); tr.json(rows, "waypoints.json"); tr.artifact(a.state, "state")
            if a.wrist_check: tr.artifact(check_path, "frames")
            if not landed: print(json.dumps({"run": run, "stop": "nut not found near the target after placing; it may have been dropped elsewhere"})); break
            if not picked: print(json.dumps({"run": run, "stop": "nut did not get closer to the target; grasp probably missed"})); break
            near = landed[0]


if __name__ == "__main__": main()
