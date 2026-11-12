"""tools/pick_loop.py find_nut_near (11 Nov 2026): the scene map answers first, the dark-blob search find_nut is the fallback. Live that day the dark
search took small fragments of the printed squares for the nut, e.g. (6.78, 4.81) in, 10 px, and never saw the real nut at (10.17, 7.32) on a light square,
where it looks silver; the scene map found it every time."""
import math
import pytest
from autohelpers import events
from pickhelpers import ShortArm, drive, overhead_frame

HINT, NUT = (10.0, 7.0), (10.17, 7.32)   # the nut is ~9 mm from the hint, inside a 70 px box
FRAG = (10.0, 6.8)                        # a dark fragment nearer the hint than the nut
NUT_OBJ = {"id": 1, "x_in": NUT[0], "y_in": NUT[1], "w_mm": 12.0, "h_mm": 10.0}
BIG = {"id": 2, "x_in": HINT[0], "y_in": HINT[1], "w_mm": 60.0, "h_mm": 40.0}
dist_mm = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) * 25.4


class Dash:
    """A scripted fetch_scene_state: answers[i] on the i-th call (the last repeats); the fake clock `t` advances by every sleep."""
    def __init__(self, *answers): self.answers, self.calls, self.t = answers, 0, 0.0
    def __call__(self): self.calls += 1; return self.answers[min(self.calls, len(self.answers)) - 1]
    def sleep(self, s): self.t += s


@pytest.fixture
def frame(cc, table): return overhead_frame(cc, table, [FRAG], r_px=2)


def test_the_dark_search_alone_takes_the_fragment(pl, cc, table, frame):
    got = pl.find_nut(cc, table, HINT, jpeg=frame, joints=None); assert got and dist_mm(got[0], FRAG) < 3.0, "the old behaviour, still reachable: the fallback"


def test_the_scene_nut_wins_over_a_nearer_dark_fragment(pl, cc, table, frame, monkeypatch):
    monkeypatch.setattr(pl, "fetch_scene_state", Dash(([NUT_OBJ], 0)))
    got = pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None)
    assert got and got.source == "scene" and dist_mm(got[0], NUT) < 0.1 and got[2] is None and len(got[1]) == 2


def test_pending_objects_make_it_wait_then_answer(pl, cc, table, frame, monkeypatch):
    d = Dash(([], 2), ([], 1), ([NUT_OBJ], 0)); monkeypatch.setattr(pl, "fetch_scene_state", d)
    got = pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None, sleep=d.sleep)
    assert got.source == "scene" and d.calls == 3 and d.t == 1.0, "polled every 0.5 s until nothing was pending"


def test_the_wait_is_bounded_then_the_dark_search_answers(pl, cc, table, frame, monkeypatch):
    d = Dash(([], 1)); monkeypatch.setattr(pl, "fetch_scene_state", d)
    got = pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None, sleep=d.sleep, wait_s=3.0)
    assert d.t == 3.0 and got.source == "dark" and dist_mm(got[0], FRAG) < 3.0


def test_nothing_pending_means_no_wait(pl, cc, table, frame, monkeypatch):
    d = Dash(([], 0)); monkeypatch.setattr(pl, "fetch_scene_state", d)
    assert pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None, sleep=d.sleep).source == "dark" and d.t == 0.0 and d.calls == 1


def test_no_candidate_within_the_radius_falls_back_to_the_dark_search(pl, cc, table, frame, monkeypatch):
    far = {**NUT_OBJ, "x_in": 5.0, "y_in": 3.0}; monkeypatch.setattr(pl, "fetch_scene_state", Dash(([far, BIG], 0)))   # a nut-like one far away; a big one close, not a nut
    got = pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None); assert got.source == "dark" and dist_mm(got[0], FRAG) < 3.0
    assert pl.find_nut_near(cc, table, HINT, radius_mm=1000.0, jpeg=frame, joints=None).source == "scene", "the radius is the only thing that kept it out"


def test_a_dashboard_error_falls_back_and_never_raises(pl, cc, table, frame, monkeypatch):
    def down(path): raise OSError("dashboard down")
    monkeypatch.undo(); monkeypatch.setattr(pl, "api", down); monkeypatch.setattr(pl, "follower_joints", lambda: None)
    assert pl.fetch_scene_state() == ([], 0) and pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None).source == "dark"
    def boom(): raise RuntimeError("bad json")
    monkeypatch.setattr(pl, "fetch_scene_state", boom); assert pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None).source == "dark"
    monkeypatch.setattr(pl, "fetch_scene_state", lambda: ([{"x_in": "nan?", "y_in": None, "w_mm": 12, "h_mm": 10}], 0)); assert pl.find_nut_near(cc, table, HINT, jpeg=frame, joints=None).source == "dark"


def test_nothing_anywhere_is_none(pl, cc, table, monkeypatch):
    monkeypatch.setattr(pl, "fetch_scene_state", Dash(([], 0))); assert pl.find_nut_near(cc, table, HINT, jpeg=overhead_frame(cc, table, []), joints=None) is None


def test_the_landing_check_uses_it_and_the_record_says_which_search_saw_the_nut(pl, cc, table, monkeypatch, capsys, tmp_path):
    calls = []; real = pl.find_nut_near
    def spy(cc_, table_, near_in, **kw): calls.append((list(near_in), kw)); return real(cc_, table_, near_in, **kw)
    monkeypatch.setattr(pl, "find_nut_near", spy); arm = ShortArm(cc, table)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")]); rec, = [o for o in out if "picked" in o]
    land = [c for c in calls if c[1].get("radius_px") == 60 and c[1].get("jpeg") is not None]; assert len(land) == 1, "the landing check, on the frame taken after the place"
    assert rec["nut_source"] == "dark" and rec["landing_source"] == "dark", "the fake dashboard has no scene objects: the dark search answered"
    r, = pl.rec.runs; assert r.params_["nut_source"] == "dark" and r.params_["landing_source"] == "dark"
    arm2 = ShortArm(cc, table, scene=[{**NUT_OBJ, "x_in": arm.nut_in[0], "y_in": arm.nut_in[1]}])
    out2 = drive(pl, monkeypatch, capsys, arm2, ["--runs", "1", "--state", str(tmp_path / "s2.json")]); rec2, = [o for o in out2 if "picked" in o]
    assert rec2["nut_source"] == "scene", "the scene map was asked first and the pick record says so"


def test_autoloop_locate_uses_it(autorig, capsys):
    r = autorig(nut_in=(9.0, 6.2)); seen = []; real = r.pl.find_nut_near   # the scene's nut is near the target, the first hint
    r.pl.find_nut_near = lambda cc, table, near_in, **kw: seen.append((list(near_in), sorted(kw))) or real(cc, table, near_in, **kw)
    s, out = r.run(capsys, cycles=1, max_retries=0); loc = events(out, "cycle")[0]["locates"][0]
    assert seen and "sleep" in seen[0][1], "polls on the runner's clock, not the wall clock"
    assert loc["source"] == "scene" and loc["chosen_in"] == pytest.approx([9.0, 6.2], abs=1e-3)
    r2 = autorig(); r2.pl.find_nut_near = lambda *a, **k: None   # nothing near: the whole-mat finder answers as before
    s2, o2 = r2.run(capsys, cycles=1, max_retries=0); l2 = events(o2, "cycle")[0]["locates"][0]; assert l2["source"] == "scene" and l2["chosen_in"] == pytest.approx([6.0, 8.0], abs=1e-3)


def test_the_scene_at_every_hint_beats_a_dark_fragment_at_the_first(pl, cc, table, monkeypatch):
    """11 Nov 2026: asked hint by hint, the dark search's printed-square fragment at the first hint beat the nut the scene map saw from the second."""
    nut = {"x_in": 10.17, "y_in": 7.32, "w_mm": 10.0, "h_mm": 8.0, "area_mm2": 60.0}
    monkeypatch.setattr(pl, "fetch_scene_state", lambda: ([nut], 0))
    frag = ([6.78, 4.81], [370.6, 153.7], 10); monkeypatch.setattr(pl, "find_nut", lambda *a, **k: frag)
    got = pl.find_nut_hints(cc, table, (([6.8, 4.8], 70), ([10.0, 7.2], 140)), sleep=lambda s: None)
    assert got and got.source == "scene" and got[0] == [10.17, 7.32], "the map's nut from the second hint, not the fragment at the first"
    monkeypatch.setattr(pl, "fetch_scene_state", lambda: ([], 0))
    got = pl.find_nut_hints(cc, table, (([6.8, 4.8], 70), ([10.0, 7.2], 140)), sleep=lambda s: None)
    assert got and got.source == "dark" and got[0] == [6.78, 4.81], "no scene nut anywhere: the dark search still answers"
