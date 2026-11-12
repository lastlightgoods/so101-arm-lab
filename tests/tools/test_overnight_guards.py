"""The overnight (unattended) guards: servo heat / current / supply / error bits / stall, the session caps (tries, minutes), the dashboard STOP's stop file reaching the loops,
and the park-at-rest on every exit path. Pure functions first, then tools/autoloop.py and tools/tidy.py on their fakes. No dashboard, no ssh, no arm.
Every threshold has a test at the boundary: just under passes, the limit trips."""
import json, os, sys
import pytest
from autohelpers import events
from pickhelpers import ROOT
sys.path.insert(0, os.path.join(ROOT, "jetson"))
import servo_telemetry as st   # noqa: E402


def lim(gr, **kw): return gr.Limits(stop_file="", **kw)


def summ(**per_joint):
    """A Summary.out()-shaped dict: summ(shoulder_lift={"temp_max_c": 66}, total_a_max=1.0)."""
    top = {k: per_joint.pop(k) for k in ("total_a_max", "volt_min", "flags") if k in per_joint}
    j = {n: {"temp_max_c": 40, "load_abs_max_pct": 10.0, "cur_max_a": 0.5, "stall_s_max": 0.0, **e} for n, e in per_joint.items()}
    return {"n": 5, "total_a_max": 1.0, "volt_min": 11.9, "flags": {}, "j": j, **top}


# ---------------------------------------------------------------- (a) heat, current, supply, error bits, stall: the thresholds
def test_heat_trips_at_the_limit_and_not_below_and_the_limit_is_a_flag(gr):
    assert gr.servo_health_trip(summ(shoulder_lift={"temp_max_c": 64}), lim(gr)) is None
    t = gr.servo_health_trip(summ(shoulder_lift={"temp_max_c": 65}), lim(gr)); assert t.reason == "servo_hot" and t.detail == {"joint": "shoulder_lift", "temp_c": 65, "limit_c": 65.0} and not t.clean
    assert gr.servo_health_trip(summ(gripper={"temp_max_c": 66}), lim(gr)).reason == "servo_hot", "the gripper is checked for heat"
    assert gr.servo_health_trip(summ(elbow_flex={"temp_max_c": 66}), lim(gr, temp_limit_c=70.0)) is None


def test_one_arm_joint_at_stall_current_trips_and_the_gripper_is_exempt(gr):
    assert gr.servo_health_trip(summ(elbow_flex={"cur_max_a": 2.69}), lim(gr)) is None
    t = gr.servo_health_trip(summ(elbow_flex={"cur_max_a": 2.7}), lim(gr)); assert t.reason == "servo_current" and t.detail["scope"] == "joint" and t.detail["joint"] == "elbow_flex"
    assert gr.servo_health_trip(summ(gripper={"cur_max_a": 3.5}), lim(gr)) is None, "the gripper stalls on the object by design (busheal: 541 counts at torque limit 500)"


def test_bus_current_over_the_supply_rating_trips(gr):
    assert gr.servo_health_trip(summ(total_a_max=5.0), lim(gr)) is None
    t = gr.servo_health_trip(summ(total_a_max=5.01), lim(gr)); assert t.reason == "servo_current" and t.detail["scope"] == "bus"


def test_supply_sag_trips_below_10_v_not_at_10(gr):
    assert gr.servo_health_trip(summ(volt_min=10.0), lim(gr)) is None
    t = gr.servo_health_trip(summ(volt_min=9.9), lim(gr)); assert t.reason == "servo_voltage" and t.detail["limit_v"] == 10.0


def test_a_stall_needs_three_seconds_and_the_gripper_may_stall(gr):
    assert gr.servo_health_trip(summ(shoulder_lift={"stall_s_max": 2.9}), lim(gr)) is None
    t = gr.servo_health_trip(summ(shoulder_lift={"stall_s_max": 3.0, "load_abs_max_pct": 80.0}), lim(gr)); assert t.reason == "servo_stall" and t.detail["limit_s"] == 3.0 and t.detail["load_pct"] == 80.0
    assert gr.servo_health_trip(summ(gripper={"stall_s_max": 30.0}), lim(gr)) is None
    assert gr.servo_health_trip(summ(shoulder_lift={"stall_s_max": 2.0}), lim(gr, stall_s=2.0)).reason == "servo_stall"


def test_the_servos_own_error_bits_trip_except_the_gripper_and_the_undocumented_bit(gr):
    assert gr.servo_health_trip(summ(flags={"gripper:current": 4, "wrist_flex:bit4": 2}), lim(gr)) is None
    t = gr.servo_health_trip(summ(flags={"shoulder_lift:overload": 1}), lim(gr)); assert t.reason == "servo_error" and t.detail["flags"] == ["shoulder_lift:overload"]
    assert gr.servo_health_trip(summ(flags=["elbow_flex:temp"]), lim(gr)).reason == "servo_error", "a live sample carries its flags as a list"
    assert gr.servo_health_trip(None, lim(gr)) is None and gr.servo_health_trip({}, lim(gr)) is None


def test_stall_run_is_measured_by_the_servo_summary_across_waypoints():
    """jetson/servo_telemetry.Summary: sag above STALL_ERR_DEG with load at STALL_LOAD_PCT, unbroken, from the sample times; a good sample resets the run; merge keeps the longest."""
    def smp(t, sag, load, on=1): return {"t": t, "total_a": 1.0, "volt_min": 11.9, "flags": [], "dt_ms": 2.0, "j": {"shoulder_lift": {"temp_c": 40, "load_pct": load, "cur_a": 1.0, "volt_v": 11.9, "sag": sag, "moving": 0, "on": on}}}
    s = st.Summary()
    for t, sag, load in ((0, 7, 60), (1, 7, 60), (2, 7, 60), (3, 7, 60)): s.add(smp(t, sag, load))
    assert s.out()["j"]["shoulder_lift"]["stall_s_max"] == 3.0
    nxt = st.Summary(); nxt.run0 = s.run0; nxt.add(smp(5, 7, 60)); assert nxt.out()["j"]["shoulder_lift"]["stall_s_max"] == 5.0, "the run carries over to the next waypoint's summary"
    for sag, load in ((6.0, 99), (7, 49.9)):   # at the limit is not a stall: the error must be ABOVE 6 deg and the load AT LEAST 50 %
        q = st.Summary(); q.add(smp(0, sag, load)); q.add(smp(4, sag, load)); assert q.out()["j"]["shoulder_lift"]["stall_s_max"] == 0.0
    r = st.Summary(); r.add(smp(0, 7, 60)); r.add(smp(2, 1, 60)); r.add(smp(3, 7, 60)); r.add(smp(4, 7, 60)); assert r.out()["j"]["shoulder_lift"]["stall_s_max"] == 1.0
    assert st.merge([s.out(), r.out()])["j"]["shoulder_lift"]["stall_s_max"] == 3.0


def test_every_waypoint_row_is_checked_for_motion_then_health(gr):
    g = gr.Guard(None, None, lim(gr))
    assert g.check_waypoint({"wp": "lift", "max_err_deg": 1.0, "tele": summ(shoulder_lift={"temp_max_c": 64})}) is None
    assert g.check_waypoint({"wp": "lift", "max_err_deg": 1.0, "tele": summ(shoulder_lift={"temp_max_c": 66})}).reason == "servo_hot"
    assert g.check_waypoint({"wp": "lift", "max_err_deg": 9.0, "tele": summ(shoulder_lift={"temp_max_c": 66})}).reason == "motion", "motion first"
    rows = [{"wp": "a", "max_err_deg": 1.0}, {"wp": "b", "max_err_deg": 1.0, "tele": summ(shoulder_lift={"temp_max_c": 66})}]
    assert g.check_rows(rows).reason == "servo_hot"


def test_without_telemetry_a_plan_passes_unless_health_is_required(gr):
    rows = [{"wp": "a", "max_err_deg": 1.0}, {"wp": "b", "max_err_deg": 1.0}]
    assert gr.Guard(None, None, lim(gr)).check_rows(rows) is None
    t = gr.Guard(None, None, lim(gr, require_health=True)).check_rows(rows); assert t.reason == "no_servo_telemetry"
    assert gr.Guard(None, None, lim(gr, require_health=True)).check_rows([{"wp": "a", "tele": summ()}]) is None and gr.Guard(None, None, lim(gr, require_health=True)).check_rows([]) is None


# ---------------------------------------------------------------- live /api/tele between picks
def tele_api(temp=40, cur=0.5, volt=11.9, status=0, total=1.0, age=0.4, joint="shoulder_lift"):
    cells = lambda tp, c, s: [tp, 5.0, c, volt, 0.0, 0.0, 0.1, 0.0, 1, 0, s]   # temp load cur volt pos goal sag vel on moving status
    row = [100.0, total, volt, [cells(temp if n == joint else 30, cur if n == joint else 0.1, status if n == joint else 0) for n in st.JOINTS]]
    return {"keys": ["temp", "load", "cur", "volt", "pos", "goal", "sag", "vel", "on", "moving", "status"], "joints": list(st.JOINTS), "rows": [row], "age": age}


def test_live_tele_summary_reads_heat_current_voltage_and_status_bits(gr):
    l = lim(gr)
    assert gr.servo_health_trip(gr.summary_of_tele_api(tele_api(temp=64), l), l) is None
    assert gr.servo_health_trip(gr.summary_of_tele_api(tele_api(temp=65), l), l).reason == "servo_hot"
    assert gr.servo_health_trip(gr.summary_of_tele_api(tele_api(cur=2.7), l), l).reason == "servo_current"
    assert gr.servo_health_trip(gr.summary_of_tele_api(tele_api(volt=9.9), l), l).reason == "servo_voltage"
    t = gr.servo_health_trip(gr.summary_of_tele_api(tele_api(status=1 << 5), l), l); assert t.reason == "servo_error" and t.detail["flags"] == ["shoulder_lift:overload"]
    assert gr.summary_of_tele_api(tele_api(age=5.01), l) is None and gr.summary_of_tele_api({"rows": []}, l) is None and gr.summary_of_tele_api(tele_api(age=None), l) is None, "a stale or empty stream is not looked at"


def test_between_picks_a_hot_servo_in_the_dashboard_stream_ends_the_session(autorig, gr, capsys):
    r = autorig(); r.ctx.fetch_tele = lambda: tele_api(temp=66); s, out = r.run(capsys, cycles=5)
    t, = events(out, "trip"); assert t["reason"] == "servo_hot" and s["cycles"] == 0 and r.arm.calls == [], "caught before the arm moves"


def test_a_stale_or_broken_tele_stream_is_noted_once_and_does_not_stop_the_run(autorig, gr, capsys):
    r = autorig(); r.ctx.fetch_tele = lambda: tele_api(temp=90, age=60.0); s, out = r.run(capsys, cycles=2)
    assert s["end"] == "cycles" and [n["note"] for n in r.notes if "live servo health" in n["note"]] == ["live servo health not checked: /api/tele has no fresh sample (plan rows are the source)"]
    r2 = autorig(); r2.ctx.fetch_tele = lambda: (_ for _ in ()).throw(OSError("down")); s2, _ = r2.run(capsys, cycles=2); assert s2["end"] == "cycles"


# ---------------------------------------------------------------- (a) in the autoloop: a hot / stalled row ends the session and the arm goes home
def inject(r, tele, when=2):
    n = [0]
    def hook(row):
        n[0] += 1
        return {**row, "tele": tele} if n[0] >= when else row
    r.row_hook = hook


@pytest.mark.parametrize("tele,reason", [(summ(shoulder_lift={"temp_max_c": 65}), "servo_hot"), (summ(elbow_flex={"cur_max_a": 2.7}), "servo_current"),
                                         (summ(shoulder_lift={"stall_s_max": 3.0}), "servo_stall"), (summ(volt_min=9.0), "servo_voltage"), (summ(flags={"elbow_flex:overload": 1}), "servo_error")])
def test_autoloop_stops_on_a_servo_row_and_goes_to_rest(autorig, capsys, tele, reason):
    r = autorig(); inject(r, tele); s, out = r.run(capsys, cycles=5)
    t, = events(out, "trip"); assert t["reason"] == reason and t["rest"] is True and r.arm.calls[-1][0] == ["rest"] and s["end"] == reason and not s["stops"][0]["clean"]


def test_autoloop_require_health_stops_a_run_whose_plans_come_back_without_telemetry(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=3, require_health=True)
    t, = events(out, "trip"); assert t["reason"] == "no_servo_telemetry"
    r2 = autorig(); s2, _ = r2.run(capsys, cycles=1); assert s2["end"] == "cycles", "off by default"


# ---------------------------------------------------------------- (b) the caps
def test_max_tries_ends_the_session_cleanly_after_n_cycles(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=10, max_tries=3)
    assert s["cycles"] == 3 and s["end"] == "max_tries" and s["stops"][0]["clean"] and s["stops"][0]["detail"] == {"tries": 3, "limit": 3}
    (st_,) = events(out, "stop"); assert st_["reason"] == "max_tries"


def test_max_minutes_ends_the_session_between_cycles_not_mid_pick(autorig, capsys):
    r = autorig(); s, out = r.run(capsys, cycles=50, max_minutes=5.0, settle_s=60.0)   # a cycle takes 120 s of fake time: 3 cycles = 360 s >= 300 s
    assert s["cycles"] == 3 and s["end"] == "max_minutes" and s["stops"][0]["clean"] and all(c["trip"] is None for c in events(out, "cycle")), "every started cycle finished"
    assert s["stops"][0]["detail"]["limit"] == 5.0


def test_budget_is_checked_in_order_and_counts_every_kind_of_try(gr):
    now = [1000.0]; ctx = gr.Ctx(lambda: {}, lambda: {}, lambda: now[0], lambda s: None, lambda d: None)
    g = gr.Guard(None, None, lim(gr, max_tries=3, max_minutes=10), ctx)
    assert g.check_budget(ctx) is None
    g.check_after_cycle({"ok": True}); g.count_try(); assert g.check_budget(ctx) is None
    g.check_after_cycle({"ok": False}); assert g.check_budget(ctx).reason == "max_tries" and g.check_budget(ctx).clean
    g2 = gr.Guard(None, None, lim(gr, max_minutes=10), ctx); g2.check_budget(ctx); now[0] += 599.9; assert g2.check_budget(ctx) is None
    now[0] += 0.1; t = g2.check_budget(ctx); assert t.reason == "max_minutes" and t.clean and t.detail["limit"] == 10
    assert gr.Guard(None, None, lim(gr), ctx).check_budget(ctx) is None, "no caps set: never"


def test_unattended_refuses_to_start_without_both_caps(gr, al):
    l = lambda **kw: gr.Limits(**kw)
    assert len(gr.unattended_problems(l())) == 2 and len(gr.unattended_problems(l(max_tries=5))) == 1 and gr.unattended_problems(l(max_tries=5, max_minutes=60)) == []
    with pytest.raises(SystemExit, match="--max-minutes.*--max-tries"): al.main(["--dry-run", "--unattended"])
    with pytest.raises(SystemExit, match="--max-tries"): al.main(["--dry-run", "--unattended", "--max-minutes", "60"])


def test_consecutive_failures_still_end_the_session_after_n(autorig, capsys):
    r = autorig(script=["miss"] * 10); s, out = r.run(capsys, cycles=10, max_failures=2, max_retries=0)
    assert s["end"] == "consecutive_failures" and s["cycles"] == 2


# ---------------------------------------------------------------- (c) the stop file halts motion at the next plan, with the arm sent home
def test_a_stop_file_created_mid_pick_stops_before_the_next_plan_and_parks(autorig, capsys):
    r = autorig(); orig = r.pl.run_plan; n = [0]
    def touch_after_first(plan, max_deg_s):
        rows = orig(plan, max_deg_s); n[0] += 1
        if n[0] == 1: open(r.stop_file, "w").write("dashboard STOP")   # what the dashboard's STOP now does
        return rows
    r.pl.run_plan = touch_after_first; s, out = r.run(capsys, cycles=5)
    (st_,) = events(out, "stop"); assert st_["reason"] == "stop_file" and st_["rest"] is True and r.arm.calls[-1][0] == ["rest"] and n[0] == 2, "one plan ran, then the check_plan refusal, then rest"


# ---------------------------------------------------------------- (e) park on every exit path
def test_ctrl_c_mid_pick_parks_the_arm_and_still_writes_the_summary(autorig, capsys):
    r = autorig(); orig = r.pl.run_plan; n = [0]
    def interrupt(plan, max_deg_s):
        n[0] += 1
        if n[0] == 2: raise KeyboardInterrupt
        return orig(plan, max_deg_s)
    r.pl.run_plan = interrupt; s, out = r.run(capsys, cycles=5)
    (i,) = events(out, "interrupted"); assert i["park"]["parked"] is True and r.arm.calls[-1][0] == ["rest"] and s["end"] == "interrupted" and out[-1]["event"] == "summary"


def test_a_crash_mid_pick_parks_the_arm_writes_the_summary_and_reraises(autorig, capsys):
    r = autorig(); n = [0]; real = r.gr.Guard.check_rows
    def bug(self, rows):   # a bug in the code that looks at the rows, after the second plan has moved the arm
        n[0] += 1
        if n[0] == 2: raise ValueError("bug after the plan ran")
        return real(self, rows)
    r.gr.Guard.check_rows = bug
    try:
        with pytest.raises(ValueError, match="bug after the plan ran"): r.run(capsys, cycles=3)
    finally: r.gr.Guard.check_rows = real
    assert r.arm.calls[-1][0] == ["rest"], "parked before the exception left"
    summary = json.load(open(r.tmp / "out" / "autoloop_t.json")); assert summary["end"] == "error"


def test_park_on_exit_decides_in_one_place(gr):
    class G:
        def __init__(self, ok): self.ok = ok
        def readback_healthy(self, ctx): return self.ok
    sent = []; send = lambda: sent.append(1)
    assert gr.park_on_exit(G(True), None, send, False, "x") == {"parked": False, "why": "already at rest"} and sent == []
    assert gr.park_on_exit(G(True), None, send, True, "run_plan")["parked"] is False and sent == [], "owner stopped the arm / dead link: no motion"
    assert gr.park_on_exit(G(False), None, send, True, "panel")["parked"] is False and sent == [], "no live readback: no motion"
    assert gr.park_on_exit(G(True), None, send, True, "servo_hot") == {"parked": True, "why": "servo_hot"} and sent == [1]
    r = gr.park_on_exit(G(True), None, lambda: (_ for _ in ()).throw(SystemExit("pick_seq produced no waypoint rows")), True, "x"); assert r["parked"] is False and "rest failed" in r["why"]
    with pytest.raises(KeyboardInterrupt): gr.park_on_exit(G(True), None, lambda: (_ for _ in ()).throw(KeyboardInterrupt()), True, "x")


# ---------------------------------------------------------------- the tidy loop
def test_tidy_unattended_refuses_without_caps_before_touching_anything():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tidy_cli_guard", os.path.join(ROOT, "tools", "tidy.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    with pytest.raises(SystemExit, match="--max-minutes.*--max-tries"): m.main(["run", "--unattended"])
    a = m.build_parser().parse_args(["loop", "--max-tries", "9", "--max-minutes", "120", "--unattended", "--temp-limit", "58"])
    assert (a.max_tries, a.max_minutes, a.unattended, a.temp_limit) == (9, 120.0, True, 58.0)
    assert m.CLEAN_ENDS == ("all_tidy", "max", "max_tries", "max_minutes")


# ---------------------------------------------------------------- fail closed: a missing telemetry feed stops an unattended run before it moves
def test_unattended_start_without_a_fresh_feed_stops_before_the_arm_moves(autorig, capsys):
    for api in (lambda: {"rows": [], "n": 0, "age": None}, lambda: tele_api(age=60.0), lambda: (_ for _ in ()).throw(OSError("down"))):
        r = autorig(); r.ctx.fetch_tele = api; s, out = r.run(capsys, cycles=3, require_health=True)
        t, = events(out, "trip"); assert t["reason"] == "no_servo_telemetry" and s["cycles"] == 0 and r.arm.calls == []
    r = autorig(); r.ctx.fetch_tele = lambda: {"rows": [], "n": 0, "age": None}; s, _ = r.run(capsys, cycles=1); assert s["end"] == "cycles", "without --require-health a missing feed is only noted"


def test_unattended_start_with_a_servo_already_at_60_c_stops(autorig, capsys):
    r = autorig(); r.ctx.fetch_tele = lambda: tele_api(temp=60); s, out = r.run(capsys, cycles=2, require_health=True)
    t, = events(out, "trip"); assert t["reason"] == "servo_hot_start" and t["detail"]["start_limit_c"] == 60.0 and r.arm.calls == []
    r = autorig(); r.ctx.fetch_tele = lambda: tele_api(temp=59); s, _ = r.run(capsys, cycles=1, require_health=True); assert s["end"] == "no_servo_telemetry" and r.arm.calls, "59 C starts; the fake plan rows then carry no telemetry"
