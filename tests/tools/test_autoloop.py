"""tools/autoloop.py against the ShortArm (real kinematics, reaches 7 mm short), a FakeWorld dashboard and the Recorder labtrack. A seeded
10-cycle session with injected failures, the caps, the retry cap, and the promise that every trip is one JSON line and the end of the
session. Pick numbering in the scripts: with max_retries=0 every cycle is two picks, try = 2(c-1) and reset = 2(c-1)+1. No arm, no ssh."""
import json, math, os, random, statistics
import pytest
from autohelpers import SAFE, events

IN = 25.4


def script(n_cycles, bad=(), kind="drop"):
    """Outcomes for n_cycles at two picks per cycle: `kind` for the tries of the 1-based cycles in `bad`, ok elsewhere."""
    s = ["ok"] * (2 * n_cycles)
    for c in bad: s[2 * (c - 1)] = kind
    return s


# ---------------------------------------------------------------- a 10-cycle session
def test_ten_cycle_dry_run_success_rate_comes_from_the_injected_failures(autorig, capsys):
    r = autorig(script=script(10, bad=(3, 7))); s, out = r.run(capsys, cycles=10, max_retries=0)
    cyc = events(out, "cycle")
    for o in out: print(json.dumps(o))   # the transcript, visible with pytest -s
    assert len(cyc) == 10 and s["cycles"] == 10 and s["landed"] == 8 and s["success_rate"] == 0.8 and s["end"] == "cycles" and s["stops"] == []
    assert [c["cycle"] for c in cyc if not c["ok"]] == [3, 7] and all(c["picked"] for c in cyc), "a dropped nut was picked but did not land"
    bad = [c for c in cyc if c["cycle"] in (3, 7)]; assert all(c["landed_err_mm"] > 15 and c["retries"] == 0 for c in bad)
    good = [c for c in cyc if c["ok"]]; assert all(c["landed_err_mm"] <= 15 and c["trip"] is None and c["found"] for c in good)
    errs = [c["landed_err_mm"] for c in cyc]; assert s["mean_err_mm"] == pytest.approx(statistics.mean(errs), abs=0.06) and s["median_err_mm"] == pytest.approx(statistics.median(errs), abs=0.06)
    assert s["duration_s"] == pytest.approx(10 * 2 * 1.0, abs=0.1) and out[-1]["event"] == "summary" and out[-1]["success_rate"] == 0.8, "two settles a cycle: the landing and the reset are both judged"
    assert s["resets"] == 10 and s["resets_ok"] == 10 and s["reset_median_err_mm"] <= 15 and s["pre_resets"] == 0
    # the summary file is the same dict
    saved = json.load(open(os.path.join(r.tmp, "out", "autoloop_t.json"))); assert saved == s
    # every reset spot obeys the box rules and moves the nut >= 60 mm
    for c in cyc[:-1]:
        x, y = c["reset_to_in"]; assert SAFE[0] <= x <= SAFE[2] and SAFE[1] <= y <= SAFE[3] and all(math.hypot(x * IN - m[0], y * IN - m[1]) >= 50 for m in r.table["layout_markers"].values())


def test_the_same_seed_gives_the_same_session(autorig, capsys):
    a = autorig(seed=5); sa, oa = a.run(capsys, cycles=4); b = autorig(seed=5); sb, ob = b.run(capsys, cycles=4)
    strip = lambda o: [{k: v for k, v in x.items() if k not in ("duration_s",)} for x in o]
    assert strip(oa) == strip(ob) and sa["median_err_mm"] == sb["median_err_mm"]


def test_mlflow_gets_one_run_per_cycle_and_a_parent_with_the_summary(autorig, capsys):
    r = autorig(script=script(4, bad=(2,))); s, out = r.run(capsys, cycles=4, max_retries=0, session="sx")
    parent, *kids = r.track.runs; assert parent.name == "autoloop-sx" and [k.name for k in kids] == [f"autoloop-sx-c{i}" for i in range(1, 5)]
    for i, k in enumerate(kids, 1):
        assert k.params_["session"] == "sx" and k.params_["cycle"] == i and k.params_["mode"] == "fixed" and k.params_["target"] == "9.00,6.00"
        assert {"landed_err_mm", "picked", "landed", "cycle_s", "retries", "trip"} <= set(k.metrics_) and k.metrics_["trip"] == 0.0 and k.files == ["cycle.json"]
    assert [k.metrics_["landed"] for k in kids] == [1.0, 0.0, 1.0, 1.0] and kids[1].metrics_["landed_err_mm"] > 15
    assert parent.metrics_["success_rate"] == 0.75 and parent.metrics_["cycles"] == 4 and parent.metrics_["trips"] == 0.0 and "summary.json" in parent.files and parent.params_["radius_mm"] == 15.0


# ---------------------------------------------------------------- retries, failures, caps
def test_a_missed_pick_is_retried_once_and_never_twice(autorig, capsys):
    r = autorig(script=["miss", "miss", "ok"]); s, out = r.run(capsys, cycles=1)   # try misses, the retry misses, the reset (3rd pick) goes
    c, = events(out, "cycle"); assert c["retries"] == 1 and not c["picked"] and not c["ok"] and r.world.picks == 3 and len(r.arm.calls) == 12   # 4 run_plans per pick: hover, grasp pose, grasp..lift (jaw check), carry..rest
    r0 = autorig(script=["miss", "ok"]); s0, o0 = r0.run(capsys, cycles=1, max_retries=0); c0, = events(o0, "cycle")
    assert c0["retries"] == 0 and r0.world.picks == 2, "--max-retries 0 turns the retry off"
    r2 = autorig(script=["miss", "ok", "ok"]); s2, o2 = r2.run(capsys, cycles=1); c2, = events(o2, "cycle")
    assert c2["retries"] == 1 and c2["picked"] and c2["ok"], "a retry that works rescues the cycle"
    r3 = autorig(script=["drop", "ok"]); s3, o3 = r3.run(capsys, cycles=1); c3, = events(o3, "cycle")
    assert c3["retries"] == 0 and c3["picked"] and not c3["landed"], "a nut that was picked and dropped is not retried; it is somewhere else now"
    assert s["retries"] == 1 and s2["retries"] == 1


def test_three_consecutive_failures_end_the_session_with_that_reason(autorig, capsys):
    r = autorig(script=["miss"] * 40); s, out = r.run(capsys, cycles=10, max_retries=0)
    t, = events(out, "trip"); assert t["reason"] == "consecutive_failures" and t["detail"]["failures"] == 3 and t["cycle"] == 3 and t["rest"] is False
    assert s["cycles"] == 3 and s["end"] == "consecutive_failures" and s["success_rate"] == 0.0 and len(s["stops"]) == 1 and events(out, "cycle")[-1]["trip"] == "consecutive_failures"
    assert out[-1]["event"] == "summary" and out[-2]["event"] == "trip", "the trip line is the last thing before the summary"


def test_two_failures_then_a_success_keep_the_session_going(autorig, capsys):
    r = autorig(script=script(5, bad=(1, 2, 4, 5), kind="miss")); s, out = r.run(capsys, cycles=5, max_retries=0)
    assert s["cycles"] == 5 and s["landed"] == 1 and s["end"] == "cycles" and not events(out, "trip")


def test_the_cycle_cap(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=4); assert s["cycles"] == 4 and s["end"] == "cycles" and len(events(out, "cycle")) == 4 and not events(out, "trip")


def test_the_minutes_cap_stops_starting_cycles(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=50, minutes=5.0, settle_s=60.0)   # a cycle takes 120 s of fake time (two settles: landing and reset): 3 cycles = 360 s >= 300 s
    assert s["cycles"] == 3 and s["end"] == "minutes" and s["stops"] == [] and s["duration_s"] == pytest.approx(360.0, abs=1.0) and not events(out, "trip")
    r1 = autorig(); s1, _ = r1.run(capsys, cycles=50, minutes=4.0, settle_s=60.0); assert s1["cycles"] == 2, "4 min = 240 s: the check is >= at the start of a cycle"


# ---------------------------------------------------------------- every trip: one line, session over
def poison_row(r):
    r.row_hook = lambda row: {**row, "max_err_deg": 9.0} if row["wp"] == "grasp" else row


@pytest.mark.parametrize("setup,reason,rest", [
    (lambda r: setattr(r.world, "jetson_up", False), "panel", False),
    (lambda r: setattr(r.world, "calib_ok", False), "calib", False),
    (lambda r: setattr(r.world, "markers", 2), "markers", False),
    (lambda r: setattr(r.world, "holding", True), "holding", False),
    (lambda r: setattr(r.world, "hidden", True), "nut_lost", False),
    (poison_row, "motion", True),
    (lambda r: setattr(r, "limits", {"ceiling_in": 1.0}), "workspace", False),
    (lambda r: open(r.stop_file, "w").close(), "stop_file", False)])
def test_every_trip_is_one_json_line_and_ends_the_session(autorig, capsys, setup, reason, rest):
    r = autorig(); setup(r); s, out = r.run(capsys, cycles=10)
    lines = [o for o in out if o.get("event") in ("trip", "stop")]; assert len(lines) == 1, [o.get("event") for o in out]
    t = lines[0]; assert t["reason"] == reason and isinstance(t["detail"], dict) and t["rest"] is rest and (t["event"] == "stop") == (reason == "stop_file")
    assert s["end"] == reason and len(s["stops"]) == 1 and s["stops"][0]["reason"] == reason and s["stops"][0]["detail"] == t["detail"] and out[-1]["event"] == "summary"
    assert s["cycles"] <= 3, "the session does not carry on after a trip"
    if reason in ("panel", "calib", "markers", "holding", "stop_file"): assert s["cycles"] == 0 and r.arm.calls == [], "before-cycle trips: the arm never moves"
    if reason == "motion": assert r.arm.calls[-1][0] == ["rest"] and events(out, "cycle")[-1]["trip"] == "motion", "a motion trip comes home when the readback is healthy"
    if reason == "workspace": assert not any("rest" in n for n, _ in r.arm.calls[:1])
    k = [x for x in r.track.runs if x.name.startswith("autoloop-t-c")]
    if k: assert k[-1].metrics_["trip"] == (0.0 if reason == "stop_file" or s["cycles"] == 0 else 1.0)


def test_a_dead_run_plan_is_a_trip_and_sends_no_more_motion(autorig, capsys):
    r = autorig(); orig = r.pl.run_plan
    def stopped(plan, max_deg_s):
        if len(r.arm.calls) == 1: raise SystemExit("pick_seq produced no waypoint rows:")   # so101-stop killed pick_seq mid-pick
        return orig(plan, max_deg_s)
    r.pl.run_plan = stopped; s, out = r.run(capsys, cycles=5)
    t, = events(out, "trip"); assert t["reason"] == "run_plan" and "pick_seq" in t["detail"]["err"] and t["rest"] is False and len(r.arm.calls) == 1 and s["end"] == "run_plan"


def test_a_trip_mid_motion_with_a_dead_panel_does_not_go_to_rest(autorig, capsys):
    r = autorig(); poison_row(r); orig = r.pl.run_plan
    def kill_panel(plan, max_deg_s):
        rows = orig(plan, max_deg_s); r.world.jetson_up = False; return rows
    r.pl.run_plan = kill_panel
    s, out = r.run(capsys, cycles=3); t, = events(out, "trip")
    assert t["reason"] == "motion" and t["rest"] is False and all(n != ["rest"] for n, _ in r.arm.calls), "the last readback is not healthy: no motion command"


def test_markers_trip_costs_ten_seconds_of_polling_not_more(autorig, capsys):
    r = autorig(); r.world.markers = 2; s, out = r.run(capsys, cycles=2)
    t, = events(out, "trip"); assert t["reason"] == "markers" and t["detail"]["seconds"] == 10.0 and s["duration_s"] == 10.0


def test_nut_lost_after_three_cycles_without_motion(autorig, capsys):
    r = autorig(); r.world.hidden = True; s, out = r.run(capsys, cycles=10)
    cyc = events(out, "cycle"); t, = events(out, "trip")
    assert [c["found"] for c in cyc] == [False] * 3 and t["reason"] == "nut_lost" and t["detail"]["cycles"] == 3 and r.arm.calls == []


# ---------------------------------------------------------------- judging
def test_in_jaws_false_means_not_picked_even_if_the_nut_moved(autorig, capsys):
    r = autorig(); r.world.in_jaws = False; s, out = r.run(capsys, cycles=1, max_retries=0); c, = events(out, "cycle")
    assert not c["picked"] and not c["ok"] and c["moved_mm"] > 20 and c["landed_err_mm"] < 15
    r2 = autorig(); r2.world.in_jaws = True; s2, o2 = r2.run(capsys, cycles=1, max_retries=0); assert events(o2, "cycle")[0]["ok"]


def test_a_nut_that_moved_less_than_20mm_is_not_picked(autorig, capsys, monkeypatch):
    r = autorig(nut_in=(10.5, 6.0)); run = r.world.on_plan   # 38 mm from the target: tried, and the pick leaves it put
    monkeypatch.setattr(r.world, "on_plan", lambda plan: None if "place" in {w["name"] for w in plan["waypoints"]} and not r.world.picks else run(plan))
    s, out = r.run(capsys, cycles=1, max_retries=0); c, = events(out, "cycle")
    assert c["moved_mm"] < 20 and not c["picked"] and not c["ok"] and "pre_reset" not in c


def test_a_nut_already_on_the_target_is_reset_away_before_the_try(autorig, capsys):
    """first10b cycles 5-6: the nut started on the target, so placing it there again moved it < 20 mm and good picks scored as misses."""
    r = autorig(nut_in=(9.5, 6.0)); s, out = r.run(capsys, cycles=1, max_retries=0); c, = events(out, "cycle")   # starts 12.7 mm from the target
    pr = c["pre_reset"]; assert pr["ok"] and pr["how"] == "at_spot" and pr["err_mm"] <= 15
    assert c["nut_in"] == pytest.approx(pr["at_in"], abs=1e-3) and c["picked"] and c["landed"] and c["ok"] and c["moved_mm"] > 20
    assert s["pre_resets"] == 1 and r.world.picks == 3, "pre-reset, try, reset"


def test_a_nut_on_the_target_that_the_reset_cannot_move_fails_the_cycle_without_a_try(autorig, capsys):
    r = autorig(nut_in=(9.5, 6.0), script=["miss"]); s, out = r.run(capsys, cycles=1, max_retries=0); c, = events(out, "cycle")
    assert c["pre_reset"]["how"] == "not_moved" and not c["pre_reset"]["ok"] and c["error"].startswith("nut on the target") and not c["ok"]
    assert r.world.picks == 1 and "reset_ok" not in c, "no try and no post-cycle reset after a failed pre-reset"


def test_every_reset_is_judged_and_the_next_cycle_starts_where_the_nut_really_is(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=2, max_retries=0); c1, c2 = events(out, "cycle")
    assert c1["reset_ok"] and c1["reset_how"] == "at_spot" and c1["reset_err_mm"] <= 15
    assert c2["locates"][0]["near_in"] == pytest.approx(c1["reset_found_in"], abs=1e-3), "cycle 2 looks near where the map saw the nut, not near the planned spot"
    assert c1["reset_found_in"] != c1["reset_to_in"] and c2["nut_in"] == pytest.approx(c1["reset_found_in"], abs=1e-3)


def test_a_missed_reset_is_recorded_and_the_next_cycle_looks_where_the_nut_stayed(autorig, capsys):
    """first10b cycle 4: the reset's jaw check saw nothing held, the nut stayed on the target, and the runner believed the planned spot."""
    r = autorig(script=["ok", "miss"]); s, out = r.run(capsys, cycles=2, max_retries=0); c1, c2 = events(out, "cycle")
    assert c1["ok"] and c1["reset_ok"] is False and c1["reset_how"] == "not_moved" and c1["reset_err_mm"] >= 60 - 15
    assert math.hypot(c1["reset_found_in"][0] - 9.0, c1["reset_found_in"][1] - 6.0) * 25.4 < 15, "the map found it still on the target"
    assert c2["pre_reset"]["ok"] and c2["ok"], "cycle 2 sees it on the target, resets it away first, then lands it"
    assert s["resets"] == 2 and s["resets_ok"] == 1


def test_judging_waits_while_the_map_has_objects_pending(autorig, capsys):
    r = autorig(); r.run(capsys, cycles=1); run = r.runner; seq = [{"ok": True, "pending": 1, "objects": []}] * 3 + [{"ok": True, "pending": 0, "objects": [{"x_in": 9.0, "y_in": 6.0}]}]
    it = iter(seq); run.scene = lambda: next(it); t0 = r.clock.now()
    sc, waited = run.settled_scene(); assert waited == 1.5 and sc["objects"] and r.clock.now() - t0 == pytest.approx(run.cfg.settle_s + 1.5)
    it2 = iter([{"ok": True, "pending": 2, "objects": []}] * 50); run.scene = lambda: next(it2)
    _, waited2 = run.settled_scene(); assert waited2 == 4.0, "capped at judge_wait_s"


def test_frames_are_saved_and_attached_when_on_and_every_locate_logs_its_candidates(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=1, max_retries=0, save_frames=True); c, = events(out, "cycle")
    d = os.path.join(r.tmp, "out", "autoloop_t_frames"); assert sorted(os.listdir(d)) == ["c1_reset.jpg", "c1_try1.jpg"]
    run, = [x for x in r.track.runs if x.name.endswith("-c1")]; assert {"frames/c1_try1.jpg", "frames/c1_reset.jpg"} <= set(run.files)
    loc, reset_loc = c["locates"]; assert loc["n_candidates"] == 1 and loc["chosen_in"] == pytest.approx(c["nut_in"], abs=1e-3) and loc["source"] == "dark"
    assert loc["candidates"][0]["w_mm"] == 12.0 and reset_loc["near_in"] is not None, "the try's locate, then the reset's"


def test_stage_lines_reach_the_session_log(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=1, max_retries=0); st = events(out, "stage")
    assert {x["stage"] for x in st} >= {"hover", "grasp_pose", "final", "jaw_check"} and all(x["cycle"] == 1 for x in st)
    assert sum(1 for x in st if x["stage"] == "hover") == 2, "the try and the reset"


def test_the_landing_radius_is_a_flag(autorig, capsys):
    r = autorig(script=script(1)); s, out = r.run(capsys, cycles=1, max_retries=0, radius_mm=0.5); c, = events(out, "cycle")
    assert c["picked"] and not c["landed"] and c["landed_err_mm"] > 0.5


# ---------------------------------------------------------------- targets, resets and the pick_loop adapters
def test_random_targets_stay_in_the_box_away_from_the_nut_and_the_markers(autorig, capsys):
    r = autorig(seed=11); s, out = r.run(capsys, cycles=6, random_target=True); cyc = events(out, "cycle"); tg = [c["target_in"] for c in cyc]
    assert len({tuple(t) for t in tg}) == 6 and all(SAFE[0] <= x <= SAFE[2] and SAFE[1] <= y <= SAFE[3] for x, y in tg)
    for c in cyc: assert math.hypot((c["target_in"][0] - c["nut_in"][0]) * IN, (c["target_in"][1] - c["nut_in"][1]) * IN) >= 59.0
    assert all(k.params_["mode"] == "random" for k in r.track.runs[1:])


def test_sample_spot_obeys_every_rule_and_gives_up_when_it_cannot(al):
    rng = random.Random(1); markers = [(100.0, 100.0)]; others = [[7.0, 6.0]]; cur = [10.0, 6.0]
    for _ in range(200):
        p = al.sample_spot(rng, SAFE, others, markers, cur)
        assert SAFE[0] <= p[0] <= SAFE[2] and SAFE[1] <= p[1] <= SAFE[3] and al.dist_mm(p, others[0]) >= 40 and al.dist_mm(p, cur) >= 60 and math.hypot(p[0] * IN - 100, p[1] * IN - 100) >= 50
    assert al.sample_spot(rng, (5, 5, 5.5, 5.5), [[5.2, 5.2]], [], None, tries=50) is None


def test_other_objects_are_avoided_by_the_reset(autorig, capsys):
    r = autorig(); r.world.extra = [[7.0 * IN, 6.0 * IN], [11.0 * IN, 5.0 * IN]]; s, out = r.run(capsys, cycles=5, max_retries=0)
    for c in events(out, "cycle"): assert all(math.hypot(c["reset_to_in"][0] * IN - e[0], c["reset_to_in"][1] * IN - e[1]) >= 40 for e in r.world.extra)


def test_the_reset_never_calls_pl_reset_nut_and_never_writes_the_pick_state(autorig, capsys, tmp_path):
    """pick_loop.reset_nut saves its history over the pick loop's state file and samples its own spot: the runner must not use it."""
    r = autorig(); called = []; r.pl.reset_nut = lambda *a, **k: called.append(a)
    state = os.path.join(r.pl.ROOT, "jetson", "calib", "pick_state.json"); before = open(state).read() if os.path.exists(state) else None
    s, out = r.run(capsys, cycles=2, max_retries=0)
    assert not called and r.world.picks == 4, "two tries and two resets, all through closed_loop_pick"
    assert (open(state).read() if os.path.exists(state) else None) == before, "the pick state file is untouched"


def test_learned_biases_are_read_from_the_pick_state_and_passed_to_every_pick(autorig, capsys):
    r = autorig(); seen = []; orig = r.pl.closed_loop_pick
    r.pl.closed_loop_pick = lambda cc, table, nut, place, biases, *a, **k: seen.append(biases) or orig(cc, table, nut, place, biases, *a, **k)
    s, out = r.run(capsys, cycles=1, max_retries=0)
    assert seen and all(b == seen[0] for b in seen)


def test_pick_loop_find_nut_anywhere_is_called_with_near_in_and_its_xy_source_pair_is_used(autorig, capsys):
    r = autorig(); seen = []; r.pl.find_nut_near = lambda *a, **k: None; r.pl.find_nut_anywhere = lambda cc, table, scene_objects=None, near_in=None, jpeg=None, markers_in=None: seen.append((len(scene_objects), near_in)) or ([6.0, 8.0], "scene")
    s, out = r.run(capsys, cycles=1, max_retries=0); assert seen and seen[0] == (1, None) and events(out, "cycle")[0]["nut_in"] == [6.0, 8.0]
    r2 = autorig(); r2.pl.find_nut = lambda *a, **k: None   # without find_nut_anywhere: the overhead search sees nothing; the dashboard's scene still has the nut
    if hasattr(r2.pl, "find_nut_anywhere"): delattr(r2.pl, "find_nut_anywhere") if "find_nut_anywhere" in vars(r2.pl) else None
    s2, o2 = r2.run(capsys, cycles=1, max_retries=0); assert events(o2, "cycle")[0]["found"]


def test_a_failed_ik_is_a_failed_try_not_a_dead_runner(autorig, capsys):
    r = autorig(); r.pl.api = lambda path: {"ok": False, "err": "unreachable"}
    s, out = r.run(capsys, cycles=2, max_retries=0); cyc = events(out, "cycle")
    assert len(cyc) == 2 and all("unreachable" in c["error"] for c in cyc) and not any(c["ok"] for c in cyc)


# ---------------------------------------------------------------- the command line
def test_main_dry_run_end_to_end(al, tmp_path, capsys, monkeypatch):
    out_dir = tmp_path / "o"; monkeypatch.chdir(tmp_path)
    rc = al.main(["--dry-run", "--cycles", "4", "--seed", "2", "--session", "cli", "--out-dir", str(out_dir), "--dry-miss", "0", "--settle", "0", "--stop-file", str(tmp_path / "none.stop")])
    lines = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]; summ = [x for x in lines if x.get("event") == "summary"][0]
    assert rc == 0 and summ["cycles"] == 4 and summ["success_rate"] == 1.0 and summ["dry_run"] is True and json.load(open(out_dir / "autoloop_cli.json"))["session"] == "cli"
    assert [x["note"].split(" not")[0] for x in lines if "note" in x and x.get("event") != "stage"] == ["overload", "fps"]
    rc2 = al.main(["--dry-run", "--cycles", "10", "--seed", "2", "--session", "cli2", "--out-dir", str(out_dir), "--dry-miss", "1", "--settle", "0", "--max-retries", "0", "--stop-file", str(tmp_path / "none.stop")])
    assert rc2 == 2 and any(x.get("event") == "trip" and x["reason"] == "consecutive_failures" for x in map(json.loads, capsys.readouterr().out.strip().splitlines()))


def test_flags_target_and_random_target_exclude_each_other(al):
    with pytest.raises(SystemExit): al.build_parser().parse_args(["--target", "9", "6", "--random-target"])
    a = al.build_parser().parse_args([]); assert a.cycles == 10 and a.radius_mm == 15.0 and a.max_retries == 1 and a.max_consecutive_failures == 3 and a.target == [9.0, 6.0]


def test_the_default_safe_box_is_the_approved_one_and_every_reset_stays_inside_it(al, pl, autorig, capsys):
    """The owner approved x 6..13, y 4..8 in (handoff docs 42/44); the runner and the pick loop must agree on it."""
    assert tuple(al.SAFE_BOX_IN) == (6.0, 4.0, 13.0, 8.0) == tuple(pl.SAFE_BOX)
    r = autorig(); s, out = r.run(capsys, cycles=10, max_retries=0)
    for c in events(out, "cycle"):
        if c.get("reset_to_in"): x, y = c["reset_to_in"]; assert 6.0 <= x <= 13.0 and 4.0 <= y <= 8.0, c["reset_to_in"]


def test_start_near_is_the_first_cycles_search_hint(autorig, capsys, al):
    r = autorig(); s, out = r.run(capsys, cycles=1, max_retries=0, start_near=(6.1, 7.9)); c, = events(out, "cycle")
    assert c["locates"][0]["near_in"] == [6.1, 7.9] and c["ok"]
    r2 = autorig(); s2, out2 = r2.run(capsys, cycles=1, max_retries=0); c2, = events(out2, "cycle")
    assert c2["locates"][0]["near_in"] is None, "no flag: no hint, as before"
    a = al.build_parser().parse_args(["--start-near", "11.3", "5.8"]); assert al.make_cfg(a, "o").start_near == (11.3, 5.8)
    assert al.make_cfg(al.build_parser().parse_args([]), "o").start_near is None


def test_the_judge_looks_at_the_image_when_the_map_misses_the_nut_at_the_target(autorig, capsys, monkeypatch):
    """pregrasp1 (10 Nov): the nut was placed 5.5 mm from the target on the mat's dark centre square and the map reported no objects, so a good
    place scored as a miss and was retried. The judge now asks the whole-mat image search near the target."""
    r = autorig(); world = r.world; place = world.on_plan
    def on_plan(plan):
        place(plan)
        if "place" in {w["name"] for w in plan["waypoints"]} and world.picks == 1: world.hidden = True   # the map loses the nut after the try's place
    monkeypatch.setattr(world, "on_plan", on_plan)
    monkeypatch.setattr(r.pl, "find_nut_mat", lambda cc, table, near_in=None, *a, **k: ([world.nut[0] / 25.4, world.nut[1] / 25.4], [320, 180], 30))
    s, out = r.run(capsys, cycles=1, max_retries=1); c, = events(out, "cycle")
    assert c["landed"] and c["ok"] and c["retries"] == 0 and c["judge_source"] == "mat" and c["landed_err_mm"] <= 15
    assert c["reset_how"] == "at_spot" and c["reset_ok"], "and the reset is judged through the same fallback while the map stays blind"


def test_a_far_mat_blob_is_not_taken_for_the_landed_nut(autorig, capsys, monkeypatch):
    r = autorig(); world = r.world; place = world.on_plan
    def on_plan(plan):
        place(plan)
        if "place" in {w["name"] for w in plan["waypoints"]} and world.picks == 1: world.hidden = True
    monkeypatch.setattr(world, "on_plan", on_plan)
    monkeypatch.setattr(r.pl, "find_nut_mat", lambda cc, table, near_in=None, *a, **k: ([near_in[0] + 2.0, near_in[1]], [320, 180], 30))   # 51 mm away
    s, out = r.run(capsys, cycles=1, max_retries=0); c, = events(out, "cycle")
    assert not c["landed"] and c["landed_err_mm"] is None, "a blob 51 mm away is outside 2 x radius: not seen"
