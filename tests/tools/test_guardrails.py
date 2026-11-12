"""tools/guardrails.py: every trip fires exactly at its threshold and not below. Pure functions first, then the Guard against a fake dashboard
on a fake clock, then the promise that matters most: a plan outside the workspace box never reaches run_plan. No dashboard, no ssh, no arm."""
import time, pytest
from autohelpers import events


def lim(gr, **kw): return gr.Limits(stop_file="", **kw)


# ---------------------------------------------------------------- the box
@pytest.mark.parametrize("x,y,h,ok", [
    (20.0, 150.0, 10.0, True), (19.99, 150.0, 10.0, False),                     # left margin, 20 mm
    (437.2, 150.0, 10.0, True), (437.21, 150.0, 10.0, False),                   # right: 457.2 - 20
    (200.0, 20.0, 10.0, True), (200.0, 19.99, 10.0, False),                     # near edge (the arm's side)
    (200.0, 284.8, 10.0, True), (200.0, 284.81, 10.0, False),                   # far edge: 304.8 - 20
    (200.0, 150.0, 101.6, True), (200.0, 150.0, 101.61, False),                 # ceiling 4 in
    (200.0, 150.0, -10.0, True), (200.0, 150.0, -10.01, False)])                # floor
def test_box_edges_are_inside_and_a_hair_beyond_is_not(gr, x, y, h, ok):
    assert (gr.box_check(x, y, h, lim(gr)) is None) is ok


def test_margin_and_ceiling_are_configurable(gr):
    assert gr.box_check(30.0, 150.0, 10.0, lim(gr, margin_mm=40.0)) and gr.box_check(30.0, 150.0, 10.0, lim(gr, margin_mm=10.0)) is None
    assert gr.box_check(200.0, 150.0, 60.0, lim(gr, ceiling_in=2.0)) and gr.box_check(200.0, 150.0, 60.0, lim(gr, ceiling_in=3.0)) is None


def test_a_real_plan_inside_the_mat_passes_and_one_at_the_edge_names_the_waypoint(autorig, gr, pl, capsys):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx)
    ok = pl.build_plan([9.0, 6.0], [9.0, 6.0], {}); assert g.check_plan(ok) is None
    bad = pl.build_plan([9.0, 6.0], [0.5, 6.0], {}); t = g.check_plan(bad)   # place target 0.5 in from the left edge: inside the 20 mm margin
    assert t.reason == "workspace" and t.detail["wp"] in ("hover_tgt", "place", "open", "lift2") and t.detail["x_in"] < 0.79
    rest = pl.build_plan(None, None, {}, names=("rest",)); assert g.check_plan(rest) is None, "rest is folded behind the mat on purpose"


def test_tip_heights_come_from_fk_with_the_arm_fit(autorig, gr, pl):
    r = autorig(); plan = pl.build_plan([9.0, 6.0], [9.0, 6.0], {}); hs = {n: h for n, _, _, h in gr.plan_tips(r.cc, r.table, plan)}
    assert hs["grasp"] == pytest.approx(pl.GRASP_H_IN * 25.4, abs=0.3) and hs["hover_nut"] == pytest.approx(1.5 * 25.4, abs=0.3) and "rest" not in hs
    g = gr.Guard(r.cc, r.table, lim(gr, ceiling_in=1.0), r.ctx); t = g.check_plan(plan)
    assert t.reason == "workspace" and "above the mat" in t.detail["why"], "a 1 in ceiling refuses the 1.5 in hovers"


# ---------------------------------------------------------------- motion
def test_motion_trips_above_8_deg_and_not_at_8(gr):
    g = gr.Guard(None, None, lim(gr))
    assert g.check_waypoint({"wp": "lift", "max_err_deg": 8.0}) is None and g.check_waypoint({"wp": "lift"}) is None
    t = g.check_waypoint({"wp": "lift", "max_err_deg": 8.01}); assert t.reason == "motion" and t.detail["wp"] == "lift" and t.detail["limit_deg"] == 8.0
    assert lim(gr, max_err_deg=3.0) and gr.Guard(None, None, lim(gr, max_err_deg=3.0)).check_waypoint({"wp": "x", "max_err_deg": 3.5})


# ---------------------------------------------------------------- perception
class Scene:
    """A /api/scene that depends on the fake clock: `low` markers until `until`, then 4."""
    def __init__(self, clock, until, key="n_markers_seen", bad=2, good=4): self.clock, self.t0, self.until, self.key, self.bad, self.good = clock, clock.t, until, key, bad, good
    def __call__(self):
        d = {"ok": True, "objects": [], "n_markers_seen": 4, "calib_ok": True, "holding": False}
        d[self.key] = self.bad if self.clock.t - self.t0 < self.until else self.good; return d


def poll(autorig, gr, until, key="n_markers_seen", bad=2, good=4, **kw):
    r = autorig(); r.ctx.fetch_scene = Scene(r.clock, until, key, bad, good); g = gr.Guard(r.cc, r.table, lim(gr, poll_s=0.5, **kw), r.ctx)
    return g.check_before_cycle(), r.clock.t - r.ctx.fetch_scene.t0


@pytest.mark.parametrize("until,trips", [(0.0, False), (9.5, False), (9.75, False), (10.0, False), (10.25, True), (60.0, True)])
def test_fewer_than_3_markers_trip_after_10_s_continuously_and_not_before(autorig, gr, until, trips):
    t, spent = poll(autorig, gr, until)   # polls every 0.5 s; low at 10.0 s means 10.0 s elapsed -> trip
    assert (t is not None) is trips
    if trips: assert t.reason == "markers" and t.detail["seconds"] == 10.0 and t.detail["limit_s"] == 10.0 and spent == 10.0
    else: assert spent <= 10.0


def test_three_markers_are_enough(autorig, gr):
    t, spent = poll(autorig, gr, 99.0, bad=3); assert t is None and spent == 0.0


@pytest.mark.parametrize("until,trips", [(0.0, False), (10.0, False), (10.25, False), (10.75, True), (60.0, True)])
def test_holding_trips_only_after_more_than_10_s(autorig, gr, until, trips):
    r = autorig(); sc = Scene(r.clock, until, key="holding", bad=True, good=False); r.ctx.fetch_scene = sc; g = gr.Guard(r.cc, r.table, lim(gr, poll_s=0.5, markers_s=1e9), r.ctx)
    t = g.check_before_cycle(); assert (t is not None) is trips
    if trips: assert t.reason == "holding" and t.detail["seconds"] == 10.5, "10.0 s is not more than 10 s; the next poll (10.5 s) is"


def test_calib_not_ok_trips_at_once_and_unknown_does_not(autorig, gr):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx); r.world.calib_ok = False
    t = g.check_before_cycle(); assert t.reason == "calib" and r.clock.t == 1000.0
    r.world.calib_ok = None; assert g.check_before_cycle() is None


def test_scene_not_available_trips_perception(autorig, gr):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx)
    r.ctx.fetch_scene = lambda: {"ok": False, "err": "top camera not live"}; assert g.check_before_cycle().reason == "perception"
    def boom(): raise OSError("refused")
    r.ctx.fetch_scene = boom; assert g.check_before_cycle().reason == "perception"


def test_nut_not_found_for_3_cycles_trips_and_a_find_resets(gr):
    g = gr.Guard(None, None, lim(gr, max_failures=99))   # failures counted apart
    assert [g.check_after_cycle({"ok": False, "found": False}) for _ in range(2)] == [None, None]
    assert g.check_after_cycle({"ok": False, "found": True}) is None and g.lost == 0
    assert [g.check_after_cycle({"ok": False, "found": False}) for _ in range(2)] == [None, None]
    t = g.check_after_cycle({"ok": False, "found": False}); assert t.reason == "nut_lost" and t.detail["cycles"] == 3


def test_three_failures_in_a_row_trip_and_a_success_resets(gr):
    g = gr.Guard(None, None, lim(gr)); bad, good = {"ok": False, "found": True}, {"ok": True, "found": True}
    assert [g.check_after_cycle(c) for c in (bad, bad, good, bad, bad)] == [None] * 5
    assert g.check_after_cycle(bad).reason == "consecutive_failures"
    assert gr.Guard(None, None, lim(gr, max_failures=2)).check_after_cycle(bad) is None


# ---------------------------------------------------------------- the panel, the stop file, the unexposed signals
def test_a_dead_panel_trips(autorig, gr):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx)
    def dead(): raise TimeoutError("timed out")
    r.ctx.fetch_state = dead; t = g.check_before_cycle(); assert t.reason == "panel" and "TimeoutError" in t.detail["err"] and g.healthy is False
    r.ctx.fetch_state = lambda: {"error": "boom"}; assert g.check_before_cycle().reason == "panel"
    r.ctx.fetch_state = r.state; r.world.jetson_up = False; t = g.check_before_cycle(); assert t.reason == "panel" and "jetson" in t.detail["err"]
    r.world.jetson_up = True; assert g.check_before_cycle() is None and g.healthy


@pytest.mark.parametrize("answer_s,trips", [(2.9, False), (3.0, False), (3.01, True)])
def test_the_panel_must_answer_within_3_s(autorig, gr, answer_s, trips):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx)
    def slow(): r.clock.sleep(answer_s); return r.state()
    r.ctx.fetch_state = slow; t = g.check_before_cycle(); assert (t is not None) is trips
    if trips: assert t.reason == "panel" and t.detail["limit_s"] == 3.0


def test_stop_file_is_a_clean_stop_before_a_cycle_and_before_a_plan(autorig, gr, pl, tmp_path):
    sf = tmp_path / "x.stop"; r = autorig(); g = gr.Guard(r.cc, r.table, gr.Limits(stop_file=str(sf)), r.ctx); plan = pl.build_plan([9.0, 6.0], [9.0, 6.0], {})
    assert g.check_before_cycle() is None and g.check_plan(plan) is None
    sf.write_text("")
    for t in (g.check_before_cycle(), g.check_plan(plan)): assert t.reason == "stop_file" and t.clean and t.detail["path"] == str(sf)
    sf.unlink(); assert g.check_before_cycle() is None
    assert gr.Limits().stop_file.endswith("outputs/run/autoloop.stop") and "~" not in gr.Limits().stop_file


def test_fps_and_overload_are_noted_once_when_the_dashboard_has_none(autorig, gr):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx)
    assert g.check_before_cycle() is None and g.check_before_cycle() is None
    assert [n["note"].split(" not")[0] for n in r.notes] == ["overload", "fps"], "each gap is logged once, not once per cycle"


def test_fps_and_overload_trip_when_the_dashboard_publishes_them(autorig, gr):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx)
    r.world.state_extra = {"fps": {"top": 25.0, "wrist": 30.0}, "overload": False}; assert g.check_before_cycle() is None and not r.notes
    r.world.state_extra = {"fps": {"top": 24.9, "wrist": 30.0}, "overload": False}; t = g.check_before_cycle(); assert t.reason == "fps" and t.detail["limit"] == 25.0
    r.world.state_extra = {"fps": 30, "overload": True}; assert g.check_before_cycle().reason == "overload"


def test_readback_healthy_needs_a_live_follower_stream(autorig, gr):
    r = autorig(); g = gr.Guard(r.cc, r.table, lim(gr), r.ctx); assert g.readback_healthy()
    r.world.follower = {"pos": {}, "src": "busy"}; assert not g.readback_healthy()
    r.world.follower = {"pos": {"shoulder_pan": 1.0}}; r.world.jetson_up = False; assert not g.readback_healthy()


# ---------------------------------------------------------------- the session, end to end
def test_a_plan_outside_the_box_never_calls_run_plan(autorig, capsys):
    r = autorig()
    s, out = r.run(capsys, cycles=3, target=(0.5, 6.0))
    names = [n for n, _ in r.arm.calls]   # the hover and the grasp pose go (the nut is inside the mat); the final stage carries the bad place target and is refused whole
    assert names[:2] == [["hover_nut"], ["hover_nut", "grasp"]] and not any("place" in n for n in names) and s["end"] == "workspace"
    t, = events(out, "trip"); assert t["reason"] == "workspace" and t["detail"]["x_in"] < 0.79 and t["rest"] is True and names[-1] == ["rest"], "refused mid-pick: the arm hovers, so it goes to rest"
    assert len(events(out, "cycle")) == 1 and events(out, "cycle")[0]["trip"] == "workspace"


def test_the_stop_file_between_cycles_ends_cleanly_without_a_trip(autorig, capsys):
    r = autorig(); open(r.stop_file, "w").close()
    s, out = r.run(capsys, cycles=5)
    assert s["cycles"] == 0 and s["end"] == "stop_file" and r.arm.calls == [] and not events(out, "trip") and events(out, "stop")[0]["reason"] == "stop_file" and s["stops"][0]["clean"]


def test_the_stop_file_appearing_mid_cycle_stops_before_the_next_run_plan(autorig, capsys):
    r = autorig(); orig = r.pl.run_plan
    def drop_file(plan, max_deg_s):
        rows = orig(plan, max_deg_s); open(r.stop_file, "w").close(); return rows
    r.pl.run_plan = drop_file   # the first stage runs, then the file appears; the second stage must not start
    s, out = r.run(capsys, cycles=3)
    st, = events(out, "stop"); assert [n for n, _ in r.arm.calls][0] == ["hover_nut"] and len(r.arm.calls) == 2 and s["end"] == "stop_file"
    assert st["reason"] == "stop_file" and st["rest"] is True and r.arm.calls[-1][0] == ["rest"], "stopped over the nut: it goes home, through the unguarded run_plan"


def test_a_dead_panel_ends_the_session_before_any_motion(autorig, capsys):
    r = autorig(); r.world.jetson_up = False
    s, out = r.run(capsys, cycles=3)
    t, = events(out, "trip"); assert t["reason"] == "panel" and t["rest"] is False and r.arm.calls == [] and s["cycles"] == 0 and s["end"] == "panel"



def test_another_arm_job_trips_arm_busy_before_the_cycle_and_a_failing_probe_only_notes(gr, cc, table):
    """11 Nov first10: lerobot-teleoperate started mid-session; the guard must catch it before a pick is planned."""
    st = lambda: {"jetson": {"online": True, "last": time.time()}, "arms": {"follower": {"pos": {}}}}
    sc = lambda: {"ok": True, "objects": [], "n_markers_seen": 4, "calib_ok": True, "holding": False}
    notes = []
    busy = gr.Ctx(st, sc, time.time, lambda s: None, notes.append, arm_jobs=lambda: ["123 /home/tc/so101/.venv/bin/lerobot-teleoperate --robot.type=so101_follower"])
    t = gr.Guard(cc, table, gr.Limits(stop_file=""), busy).check_before_cycle()
    assert t and t.reason == "arm_busy" and "lerobot-teleoperate" in t.detail["jobs"][0]
    free = gr.Ctx(st, sc, time.time, lambda s: None, notes.append, arm_jobs=lambda: [])
    assert gr.Guard(cc, table, gr.Limits(stop_file=""), free).check_before_cycle() is None
    def boom(): raise RuntimeError("ssh down")
    down = gr.Ctx(st, sc, time.time, lambda s: None, notes.append, arm_jobs=boom)
    assert gr.Guard(cc, table, gr.Limits(stop_file=""), down).check_before_cycle() is None and any("arm jobs not checked" in str(n) for n in notes)
