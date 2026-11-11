#!/usr/bin/env python3
"""servo_play.py - measure the play in the shoulder lift and the elbow (gearbox backlash vs compliance under load), optionally try the shoulder lift's P gain.

  tools/servo_play.py --dry-run                         # the whole pipeline against a fake bus (jetson/servo_probe.py FakeRig); no arm, no ssh
  tools/servo_play.py --plan-only                       # print the plan (poses, joints, tip heights, box check), move nothing
  tools/servo_play.py --tc-at-stop [--p-test]           # LIVE: TC at STOP; ships jetson/servo_probe.py to the Jetson and runs it like pick_seq.py

Per pose (near / mid / far reach, straight ahead of the base, 40 mm over the mat, every commanded point inside the mat shrunk by 20 mm and not below the mat),
per joint (shoulder lift, elbow): command the target after approaching it from BELOW (target - d) and from ABOVE (target + d), `reps` times each, settle 1 s, read
Present_Position / Present_Load / Present_Current / Present_Temperature / Present_Voltage.
  backlash   = mean(reached after approach from above) - mean(reached after approach from below), deg, per pose (the gravity sag is the same both ways and cancels);
  deflection = mean of the two approaches minus the target, deg; compliance = slope of deflection against Present_Load over the poses (deg per load unit);
  both converted to mm at the tip with the arm fit's FK (finite difference of cam_calib.fk_tip about the pose, offsets and tip offset applied).
Caveat printed with every run: the STS3215's encoder is read at the output side as far as Feetech documents it, so play BEYOND the encoder (a loose horn) is invisible to
this test; play inside the control loop shows up as the above/below difference and as dead band.
--p-test (shoulder lift only): P_Coefficient default, x1.5, x2 written with Lock=1 (never unlocked, never EEPROM-path), sag / oscillation / current / voltage dip measured,
original restored in a finally block and read back; aborts below --min-volts or above --max-temp. Output: JSON lines on stdout, one MLflow run in `so101-servo-play`
(`so101-servo-play-dry` for --dry-run), summary in outputs/run/servo_play_<session>.json.
"""
import argparse, importlib.util, json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "jetson", "calib", "top_table.json")
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TEST_JOINTS = ("shoulder_lift", "elbow_flex")
POSES = {"near": 150.0, "mid": 220.0, "far": 290.0}   # tip reach in the arm frame, mm straight ahead of base_link (10 Nov grasps sat at 158-291 mm)
HEIGHT_MM = 40.0                                         # over the mat: the hover height of the pick loop is 37.7 mm; the floor is the mat itself (0)
MAX_DEG_S = 50.0
REMOTE_PLAN = "/tmp/servo_plan.json"
CAVEAT = ("encoder at the output side (Feetech): play beyond the encoder, e.g. a loose horn or bracket, is NOT visible in Present_Position; "
          "play inside the control loop appears as the above/below difference and dead band")


def _load(name, folder):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, folder, f"{name}.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def cam_calib(): return _load("cam_calib", "jetson")
def guardrails(): return _load("guardrails", "tools")
def probe(): return _load("servo_probe", "jetson")


# ---------------------------------------------------------------- kinematics: the plan and the mm conversions
def _off(table): return (table.get("arm_fit") or {}).get("joint_offsets_deg") or {}


def tip_xyz(cc, table, joints):
    """Tip position (arm frame, mm) of LeRobot-degree `joints` through the arm fit's zero offsets and tip offset (the chain of guardrails.fk_mat_mm)."""
    fit = table["arm_fit"]; off = _off(table)
    tip, _ = cc.fk_tip({k: float(joints[k]) + float(off.get(k, 0.0)) for k in ARM}, tip_offset_mm=fit.get("tip_offset_mm")); return np.asarray(tip, float)


def mm_per_deg(cc, table, joints, joint, h=0.5):
    """(tip mm moved per degree of `joint`, of which vertical) by a central difference about `joints`."""
    a, b = dict(joints), dict(joints); a[joint] = joints[joint] - h; b[joint] = joints[joint] + h
    d = (tip_xyz(cc, table, b) - tip_xyz(cc, table, a)) / (2 * h); return float(np.linalg.norm(d)), float(abs(d[2]))


def lever_fn(cc, table):
    """For probe.FakeRig: horizontal mm from the shoulder-lift / elbow joint origin to the tip (what gravity turns the joint with)."""
    off = _off(table)
    def fn(cmd):
        o = cc.fk_chain({k: float(cmd[k]) + float(off.get(k, 0.0)) for k in ARM}, tip_offset_mm=table["arm_fit"].get("tip_offset_mm")); tip = o[-1]
        return {"shoulder_lift": float(np.hypot(*(tip[:2] - o[1][:2]))), "elbow_flex": float(np.hypot(*(tip[:2] - o[2][:2])))}
    return fn


def box_problem(cc, table, g, joints, lim):
    x, y, h = g.fk_mat_mm(cc, table, joints); return g.box_check(x, y, h, lim)


def build_plan(cc, table, reaches=None, height_mm=HEIGHT_MM, approach_deg=4.0, reps=3, settle_s=1.0, test_joints=TEST_JOINTS, p_test=None, limits=None, min_approach_deg=1.5):
    """IK a pose per reach, then shrink each joint's approach amplitude until pre and target points are inside the box (mat - 20 mm, floor 0 = the mat, ceiling 4 in).
    Returns (plan, notes); a joint whose approach cannot fit inside the box even at min_approach_deg is dropped from that pose (a note says so)."""
    g = guardrails(); lim = g.Limits(margin_mm=20.0, floor_mm=0.0); notes = []; poses = []
    fit = table["arm_fit"]; roll = float(fit.get("wrist_roll_deg", 0.0))
    for name, reach in (reaches or POSES).items():
        try: res = cc.ik_tip([float(reach), 0.0, float(fit.get("table_z_mm", 0.0)) + float(height_mm)], _off(table), roll_deg=roll)
        except cc.CalibError as e: notes.append(f"pose {name} ({reach:g} mm) dropped: {e}"); continue
        j = {k: float(res["joints_deg"][k]) for k in ARM}
        why = box_problem(cc, table, g, j, lim)
        if why: notes.append(f"pose {name} ({reach:g} mm) dropped: {why}"); continue
        appr = {}
        for jn in test_joints:
            d = float(approach_deg)
            while d >= min_approach_deg:
                if all(box_problem(cc, table, g, {**j, jn: j[jn] + s * d}, lim) is None for s in (-1, 1)): break
                d = round(d - 0.5, 3)
            if d < min_approach_deg: notes.append(f"pose {name}: {jn} approach does not fit the box even at {min_approach_deg} deg; joint skipped there"); continue
            appr[jn] = d
        if not appr: continue
        x, y, h = g.fk_mat_mm(cc, table, j)
        poses.append({"name": name, "reach_mm": float(reach), "joints": {k: round(v, 3) for k, v in j.items()}, "approach_deg": appr, "tip_mat_mm": [round(x, 1), round(y, 1), round(h, 1)]})
    if not poses: raise SystemExit("no pose fits the box: " + "; ".join(notes))
    plan = {"poses": poses, "test_joints": [j for j in test_joints if any(j in p["approach_deg"] for p in poses)], "reps": int(reps), "settle_s": float(settle_s), "limits": limits or {}, "p_test": p_test}
    if p_test:
        names = [p["name"] for p in poses]
        if p_test["pose"] not in names: raise SystemExit(f"p-test pose {p_test['pose']} is not in the plan ({names})")
        if p_test.get("move_to") not in names: p_test["move_to"] = names[0] if names[0] != p_test["pose"] else names[-1]
    return plan, notes


# ---------------------------------------------------------------- analysis of the events
def _mean(v): return float(sum(v) / len(v))


def analyse(events, cc, table, plan):
    """Backlash / compliance per pose and joint, the compliance slope per joint, and the P table; all from the `point` / `p_value` events (so a test can recover a model)."""
    poses = {p["name"]: p for p in plan["poses"]}; per = {}
    for e in events:
        if e.get("event") == "point": per.setdefault((e["pose"], e["joint"]), {"below": [], "above": []})[e["dir"]].append(e)
    rows = []
    for (pn, jn), d in per.items():
        if not d["below"] or not d["above"]: continue
        eb, ea = [x["err"] for x in d["below"]], [x["err"] for x in d["above"]]; allp = d["below"] + d["above"]
        mpd, vert = mm_per_deg(cc, table, poses[pn]["joints"], jn)
        within = [x - _mean(eb) for x in eb] + [x - _mean(ea) for x in ea]; sd = float(np.sqrt(sum(w * w for w in within) / max(1, len(within) - 2)))
        bl, df = _mean(ea) - _mean(eb), (_mean(ea) + _mean(eb)) / 2
        rows.append({"pose": pn, "joint": jn, "reach_mm": poses[pn].get("reach_mm"), "n_below": len(eb), "n_above": len(ea), "backlash_deg": round(bl, 3), "backlash_mm": round(abs(bl) * mpd, 2),
                     "deflection_deg": round(df, 3), "deflection_mm": round(abs(df) * mpd, 2), "deflection_vertical_mm": round(abs(df) * vert, 2), "load": round(_mean([x["load"] for x in allp]), 1),
                     "current": round(_mean([x["current"] for x in allp]), 1), "temp_c": max(x["temp_c"] for x in allp), "volts_min": min(x["volts"] for x in allp), "scatter_deg": round(sd, 4), "tip_mm_per_deg": round(mpd, 2)})
    joints = {}
    for jn in sorted({r["joint"] for r in rows}):
        rr = [r for r in rows if r["joint"] == jn]; L = np.array([r["load"] for r in rr]); D = np.array([r["deflection_deg"] for r in rr]); out = {"poses": len(rr), "backlash_deg_mean": round(_mean([r["backlash_deg"] for r in rr]), 3)}
        low = min(rr, key=lambda r: abs(r["load"])); out["backlash_deg_lowest_load"] = low["backlash_deg"]; out["backlash_mm_lowest_load"] = low["backlash_mm"]; out["backlash_mm_mean"] = round(_mean([r["backlash_mm"] for r in rr]), 2)
        k0 = float((D * L).sum() / (L * L).sum()) if float((L * L).sum()) > 0 else 0.0   # through the origin: no sag at no load (physical), steadier when the poses' loads are close
        out["compliance_deg_per_100load_origin"] = round(k0 * 100, 3); out["compliance_mm_per_100load_origin"] = round(abs(k0) * 100 * _mean([r["tip_mm_per_deg"] for r in rr]), 2)
        out["load_range_pct_of_mean"] = round(100 * float(np.ptp(L)) / max(1e-9, float(np.mean(np.abs(L)))), 1)
        if out["load_range_pct_of_mean"] < 40: out["slope_note"] = "the poses' loads span < 40 % of their mean: trust the through-origin figure, not the fitted slope"
        if len(rr) >= 2 and np.ptp(L) > 1e-9:
            slope, icpt = np.polyfit(L, D, 1); pred = slope * L + icpt; ss = float(((D - D.mean()) ** 2).sum()); r2 = 1.0 - float(((D - pred) ** 2).sum()) / ss if ss > 1e-12 else 1.0
            mpd = _mean([r["tip_mm_per_deg"] for r in rr]); out.update({"compliance_deg_per_100load": round(float(slope) * 100, 3), "compliance_mm_per_100load": round(abs(float(slope)) * 100 * mpd, 2), "deflection_deg_at_zero_load": round(float(icpt), 3), "fit_r2": round(r2, 3)})
        else: out["note"] = "need >= 2 poses with different loads for a compliance slope"
        joints[jn] = out
    ps = [e for e in events if e.get("event") == "p_value"]; base = ps[0] if ps else None
    ptab = [{**{k: e[k] for k in ("p", "scale", "sag_deg", "osc_std_deg", "load", "current_hold", "current_peak_move", "volts_idle", "volts_min_move", "volts_dip", "temp_c")},
             "sag_change_pct": (round(100 * (abs(e["sag_deg"]) - abs(base["sag_deg"])) / abs(base["sag_deg"]), 1) if base and abs(base["sag_deg"]) > 1e-9 else None)} for e in ps]
    end = [e for e in events if e.get("event") in ("done", "abort")]; rest = [e for e in events if e.get("event") == "restore"]
    return {"points": rows, "joints": joints, "p_values": ptab, "end": end[-1] if end else None, "restore": rest[-1] if rest else None, "caveat": CAVEAT}


# ---------------------------------------------------------------- running it
def run_dry(cc, table, plan, args):
    sp = probe(); clk = sp.VClock(); first = plan["poses"][0]["joints"]; ev = []
    rig = sp.FakeRig(clock=clk, start={**{j: first[j] for j in ARM}, "gripper": 0.0}, backlash_deg=args.fake_backlash, compliance={"shoulder_lift": args.fake_compliance_lift, "elbow_flex": args.fake_compliance_elbow},
                     lever_fn=lever_fn(cc, table), lock_mode=args.fake_lock_mode, seed=args.seed)
    def emit(d): ev.append(d); print(json.dumps(d), flush=True)
    code = sp.run_play(plan, rig, emit, clk.now, clk.sleep, args.max_deg_s); return ev, code, {"virtual_s": round(clk.t, 1), "writes": rig.writes, "eeprom_writes": rig.eeprom_writes}


def run_live(plan, args):
    """Like pick_loop.run_plan: scp the plan (and the probe), run it over ssh, stream the JSON lines. The Jetson-side script holds ~/so101/.arms_busy and refuses while lerobot jobs run."""
    local = "/tmp/servo_plan_local.json"; open(local, "w").write(json.dumps(plan))
    subprocess.run(["scp", "-q", local, f"jetson:{REMOTE_PLAN}"], check=True)
    subprocess.run(["scp", "-q", os.path.join(ROOT, "jetson", "servo_probe.py"), "jetson:so101/servo_probe.py"], check=True)
    cmd = ["ssh", "-o", "ConnectTimeout=8", "jetson", f"cd ~/so101 && source env.sh >/dev/null 2>&1; ~/so101/.venv/bin/python servo_probe.py --plan {REMOTE_PLAN} --max-deg-s {args.max_deg_s:g}"]
    ev = []; p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in p.stdout:
            try: d = json.loads(line)
            except ValueError: continue
            ev.append(d); print(json.dumps(d), flush=True)
        code = p.wait(timeout=60)
    except BaseException:
        p.terminate(); raise
    if not ev: raise SystemExit("servo_probe produced no events:\n" + (p.stderr.read() or "")[-800:])
    return ev, code, {}


def track(args, plan, events, summary, extra):
    """One MLflow run in so101-servo-play (so101-servo-play-dry for --dry-run); a no-op when there is no server (jetson/labtrack.py)."""
    lt = _load("labtrack", "jetson").LabTrack("so101-servo-play-dry" if args.dry_run else "so101-servo-play")
    with lt.run(f"servo-play-{args.session}", tags={"host": "mac", "session": args.session, "dry_run": int(args.dry_run), "p_test": int(bool(plan.get("p_test")))}) as tr:
        tr.params({"reps": plan["reps"], "settle_s": plan["settle_s"], "poses": ",".join(p["name"] for p in plan["poses"]), "joints": ",".join(plan["test_joints"]), "min_volts": plan["limits"].get("min_volts", 9.0),
                   "max_temp_c": plan["limits"].get("max_temp_c", 60.0), "max_deg_s": args.max_deg_s, "p_scales": ",".join(map(str, (plan.get("p_test") or {}).get("scales", []))), "caveat": CAVEAT})
        m = {}
        for r in summary["points"]: m[f"backlash_deg_{r['pose']}_{r['joint']}"] = r["backlash_deg"]; m[f"backlash_mm_{r['pose']}_{r['joint']}"] = r["backlash_mm"]; m[f"deflection_deg_{r['pose']}_{r['joint']}"] = r["deflection_deg"]; m[f"load_{r['pose']}_{r['joint']}"] = r["load"]; m[f"temp_c_{r['pose']}_{r['joint']}"] = r["temp_c"]
        for jn, o in summary["joints"].items():
            for k, v in o.items():
                if isinstance(v, (int, float)): m[f"{k}_{jn}"] = v
        for i, e in enumerate(summary["p_values"]):
            tr.metrics({f"p_{k}": v for k, v in e.items() if isinstance(v, (int, float))}, step=i)
        tr.metrics(m); tr.json(events, "events.json"); tr.json(summary, "summary.json"); tr.json(plan, "plan.json"); tr.json(extra, "extra.json")
        tr.tags({"end": (summary["end"] or {}).get("event"), "restore_ok": (summary["restore"] or {}).get("ok")})
    return lt.enabled


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="fake bus, no ssh, virtual clock"); ap.add_argument("--plan-only", action="store_true"); ap.add_argument("--tc-at-stop", action="store_true", help="required for a live run: TC is at STOP")
    ap.add_argument("--reach-mm", type=float, nargs=3, default=list(POSES.values()), metavar=("NEAR", "MID", "FAR")); ap.add_argument("--height-mm", type=float, default=HEIGHT_MM)
    ap.add_argument("--approach-deg", type=float, default=4.0); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--p-test", action="store_true"); ap.add_argument("--p-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0]); ap.add_argument("--p-pose", default="far"); ap.add_argument("--p-move-to", default="near")
    ap.add_argument("--min-volts", type=float, default=9.0); ap.add_argument("--max-temp", type=float, default=60.0); ap.add_argument("--max-load", type=int, default=800); ap.add_argument("--max-osc-deg", type=float, default=0.5)
    ap.add_argument("--max-deg-s", type=float, default=MAX_DEG_S); ap.add_argument("--session", default=time.strftime("%Y%m%d_%H%M")); ap.add_argument("--no-mlflow", action="store_true"); ap.add_argument("--out")
    ap.add_argument("--table", default=TABLE); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fake-backlash", type=float, default=0.8); ap.add_argument("--fake-compliance-lift", type=float, default=0.006); ap.add_argument("--fake-compliance-elbow", type=float, default=0.004); ap.add_argument("--fake-lock-mode", default="ram", choices=("ram", "ignore"))
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.max_deg_s > MAX_DEG_S: raise SystemExit(f"--max-deg-s above {MAX_DEG_S:g} (the pick loop's closed-loop cap) is not allowed here")
    if not (a.dry_run or a.plan_only or a.tc_at_stop): raise SystemExit("a live run moves the arm: pass --tc-at-stop once TC is at STOP (or use --dry-run / --plan-only)")
    cc = cam_calib(); table = json.load(open(a.table))
    ptest = {"joint": "shoulder_lift", "scales": a.p_scales, "pose": a.p_pose, "move_to": a.p_move_to} if a.p_test else None
    limits = {"min_volts": a.min_volts, "max_temp_c": a.max_temp, "max_load": a.max_load, "max_osc_deg": a.max_osc_deg}
    plan, notes = build_plan(cc, table, dict(zip(POSES, a.reach_mm)), a.height_mm, a.approach_deg, a.reps, a.settle, TEST_JOINTS, ptest, limits)
    print(json.dumps({"plan": plan, "notes": notes}), flush=True)
    if a.plan_only: return 0
    t0 = time.time(); ev, code, extra = run_dry(cc, table, plan, a) if a.dry_run else run_live(plan, a)
    summary = analyse(ev, cc, table, plan); summary.update({"exit_code": code, "session": a.session, "dry_run": a.dry_run, "wall_s": round(time.time() - t0, 1), "notes": notes})
    print(json.dumps({"summary": summary}), flush=True)
    out = a.out or os.path.join(ROOT, "outputs", "run", f"servo_play_{a.session}{'_dry' if a.dry_run else ''}.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"plan": plan, "events": ev, "summary": summary, "extra": extra}, open(out, "w"), indent=1)
    if not a.no_mlflow: track(a, plan, ev, summary, extra)
    return code


if __name__ == "__main__": sys.exit(main())
