#!/usr/bin/env python3
"""pick_loop.py - repeat "pick the nut, place it on the mat centre" and get better each time (iterative learning).

Per run: find the nut in the overhead frame (dark blob near its last known spot), build a 9-waypoint plan from /api/ik,
bias every waypoint by the servo sag measured at the SAME waypoint last time (iterative learning control), shift the place
target by the camera-measured landing error of the previous run (outcome learning), run jetson/pick_seq.py over ssh in
one bus session, then measure where the nut landed. Stops when the nut is not found, not picked up, or the error grows.

  tools/pick_loop.py --runs 4 [--target 9 6] [--state jetson/calib/pick_state.json]
State (biases, place offset, history) persists in --state so later sessions start where this one ended.
"""
import argparse, json, math, os, subprocess, time, urllib.request
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH = os.environ.get("SO101_DASH", "http://localhost:8765")
REST = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 72.0, "wrist_roll": 0.5}
WP = [("hover_nut", "nut", 1.5, 35, 1.6, 0.4), ("grasp", "nut", 0.12, 35, 0.9, 0.5), ("close", "nut", 0.12, 0, 0.6, 0.4),
      ("lift", "nut", 1.5, 0, 0.8, 0.2), ("hover_tgt", "tgt", 1.5, 0, 1.2, 0.3), ("place", "tgt", 0.3, 0, 0.9, 0.5),
      ("open", "tgt", 0.3, 35, 0.5, 0.4), ("lift2", "tgt", 1.5, 35, 0.7, 0.2), ("rest", None, 0, 0, 1.8, 0.3)]
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def api(path):
    return json.load(urllib.request.urlopen(DASH + path, timeout=10))


def cam_calib():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cam_calib", os.path.join(ROOT, "jetson", "cam_calib.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def find_nut(cc, table, near_in, radius_px=70):
    """Dark blob nearest to `near_in` (mat inches) in a fresh overhead frame -> (mat_in, pixel, area) or None."""
    img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(DASH + f"/cam/top.jpg?t={time.time():f}", timeout=5).read(), np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); cu, cv_ = cc.table_to_pixel(table, near_in[0] * 25.4, near_in[1] * 25.4)
    x0, y0 = max(0, int(cu) - radius_px), max(0, int(cv_) - radius_px); roi = g[y0:y0 + 2 * radius_px, x0:x0 + 2 * radius_px]
    m = (roi < 60).astype(np.uint8); n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    cands = [(math.hypot(x0 + cen[i][0] - cu, y0 + cen[i][1] - cv_), int(st[i, cv2.CC_STAT_AREA]), x0 + cen[i][0], y0 + cen[i][1]) for i in range(1, n) if 8 <= st[i, cv2.CC_STAT_AREA] <= 300]
    if not cands: return None
    _, area, u, v = min(cands); X, Y = cc.pixel_to_table(table, u, v)
    return [X / 25.4, Y / 25.4], [u, v], area


def build_plan(nut_in, tgt_in, biases):
    def ik(x, y, h):
        r = api(f"/api/ik?x_in={x:.4f}&y_in={y:.4f}&h_in={h}")
        if not r.get("ok"): raise SystemExit(f"ik failed for ({x:.2f},{y:.2f},{h}): {r.get('err')}")
        return r["joints_deg"]
    wps = []
    for name, which, h, grip, secs, settle in WP:
        j = dict(REST) if which is None else ik(*(nut_in if which == "nut" else tgt_in), h)
        b = biases.get(name, {}); cmd = {k: j[k] + b.get(k, 0.0) for k in j}; cmd["wrist_flex"] = max(-93.0, min(93.0, cmd["wrist_flex"]))
        wps.append({"name": name, "joints": cmd, "desired": j, "gripper": grip, "seconds": secs, "settle": settle, "gripper_only": name in ("close", "open")})
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=4); ap.add_argument("--target", type=float, nargs=2, default=[9.0, 6.0]); ap.add_argument("--max-deg-s", type=float, default=80.0)
    ap.add_argument("--state", default=os.path.join(ROOT, "jetson", "calib", "pick_state.json")); ap.add_argument("--gain", type=float, default=0.7); ap.add_argument("--outcome-gain", type=float, default=0.5)
    ap.add_argument("--alt", type=float, nargs=2, default=[6.0, 5.0], help="second target; runs alternate alt, target, alt, target ... so every pick and place is unambiguous; the LAST run always goes to --target")
    a = ap.parse_args(); cc = cam_calib(); table = json.load(open(os.path.join(ROOT, "jetson", "calib", "top_table.json")))
    state = json.load(open(a.state)) if os.path.exists(a.state) else {"biases": {}, "place_offsets_in": {}, "history": []}
    state.setdefault("place_offsets_in", {}); near = state["history"][-1]["landed_in"] if state["history"] and state["history"][-1].get("landed_in") else list(a.target)
    for run in range(1, a.runs + 1):
        tgt = list(a.target) if (run == a.runs or run % 2 == 0) else list(a.alt); key = f"{tgt[0]:.2f},{tgt[1]:.2f}"
        off = state["place_offsets_in"].setdefault(key, [0.0, 0.0])
        found = find_nut(cc, table, near) or find_nut(cc, table, list(a.target), radius_px=140) or find_nut(cc, table, list(a.alt), radius_px=140)
        if not found: print(json.dumps({"run": run, "stop": "nut not found near", "near": near})); break
        nut_in, nut_px, area = found
        place_in = [tgt[0] + off[0], tgt[1] + off[1]]
        plan = build_plan(nut_in, place_in, state["biases"]); t0 = time.time(); rows = run_plan(plan, a.max_deg_s); dt = time.time() - t0
        # learn the servo sag per waypoint: bias += gain * (target - reached), capped
        desired = {w["name"]: w["desired"] for w in plan["waypoints"]}
        for r in rows:
            if r["wp"] in ("close", "open"): continue
            b = state["biases"].setdefault(r["wp"], {})
            for k in ARM: b[k] = max(-6.0, min(6.0, b.get(k, 0.0) + a.gain * (desired[r["wp"]][k] - r["reached"][k])))   # drive DESIRED - reached to zero, not command - reached
            r["max_err_deg"] = round(max(abs(desired[r["wp"]][k] - r["reached"][k]) for k in ARM), 2)
        time.sleep(1.0)
        landed = find_nut(cc, table, place_in, radius_px=60)
        before_mm = math.hypot(nut_in[0] - tgt[0], nut_in[1] - tgt[1]) * 25.4
        err_mm = math.hypot(landed[0][0] - tgt[0], landed[0][1] - tgt[1]) * 25.4 if landed else None
        picked = landed is not None and (err_mm < before_mm - 5.0 or before_mm < 40.0)   # it got closer, or it was already there and stayed within the search radius
        if landed and picked:
            off[0] = max(-0.6, min(0.6, off[0] + a.outcome_gain * (tgt[0] - landed[0][0]))); off[1] = max(-0.6, min(0.6, off[1] + a.outcome_gain * (tgt[1] - landed[0][1])))
        rec = {"run": run, "target_in": tgt, "nut_in": [round(v, 2) for v in nut_in], "picked": picked, "landed_in": [round(v, 2) for v in landed[0]] if landed else None, "err_mm": round(err_mm, 1) if err_mm is not None else None,
               "seconds": round(dt, 1), "sag_deg": {r["wp"]: r["max_err_deg"] for r in rows}, "place_offset_in": [round(v, 3) for v in off], "t": time.strftime("%H:%M:%S")}
        state["history"].append(rec); json.dump(state, open(a.state, "w"), indent=1); print(json.dumps(rec), flush=True)
        if not landed: print(json.dumps({"run": run, "stop": "nut not found near the target after placing; it may have been dropped elsewhere"})); break
        if not picked: print(json.dumps({"run": run, "stop": "nut did not get closer to the target; grasp probably missed"})); break
        near = landed[0]


if __name__ == "__main__": main()
