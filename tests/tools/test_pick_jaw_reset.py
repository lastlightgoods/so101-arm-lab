"""tools/pick_loop.py beyond the readback corrections: the nut-in-jaws check after the lift (found -> carry; not found -> one corrected
retry; not found twice -> picked=false, reason not_in_jaws, rest), its promise never to raise, the two-step close and --fast-close,
--reset's target sampling and end-to-end run, the whole-mat finder (scene map first, then the masked image search) and the loop's
`lost` stop. Everything runs against the fake arm of pickhelpers; no dashboard, no ssh, no arm."""
import json, math, os, random
import pytest
from pickhelpers import JAW_RING_UV, PROBE_NUT_IN, ShortArm, drive, frame_feed, jaw_frame, overhead_frame

LIFT = ["pregrasp", "grasp", "close1", "close2", "lift"]; CARRY = ["hover_tgt", "place", "open", "lift2", "rest"]; ABORT = ["hover_nut", "rest"]
HOVER, POSE = ["hover_nut"], ["hover_nut", "grasp"]
NUT_OBJ = {"id": 1, "x_in": PROBE_NUT_IN[0], "y_in": PROBE_NUT_IN[1], "w_mm": 12.0, "h_mm": 10.0, "px": [320, 180], "color": [119, 124, 128]}
BIG_OBJ = {"id": 2, "x_in": 3.0, "y_in": 9.0, "w_mm": 60.0, "h_mm": 40.0, "px": [100, 60], "color": [200, 200, 200]}


def stages(out): return [o for o in out if "stage" in o]
def records(out): return [o for o in out if "picked" in o]
def jaw_lines(out): return [o for o in stages(out) if o["stage"] == "jaw_check"]
def stops(out): return [o for o in out if "stop" in o]


# ---------------------------------------------------------------- 2A(i) the jaw check
def test_nut_seen_in_the_jaws_carries(pl, cc, table, monkeypatch, capsys, tmp_path):
    feed = frame_feed(jaw_frame(ring=True)); monkeypatch.setattr(pl, "fetch_wrist", feed); arm = ShortArm(cc, table)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")]); rec, = records(out); jc, = jaw_lines(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, LIFT, CARRY], "hover, grasp pose, grasp + lift, look, carry: four pick_seq sessions"
    assert feed.calls == 1 and jc["attempt"] == 1 and jc["in_jaws"] is True and math.hypot(jc["ring"]["uv"][0] - JAW_RING_UV[0], jc["ring"]["uv"][1] - JAW_RING_UV[1]) <= 2 and jc["window"] == [300, 300, 380, 360]
    assert rec["picked"] is True and rec["in_jaws"] is True and rec["jaw_retry"] is False and rec["reason"] is None and rec["jaw_ring"] == jc["ring"]["uv"]
    r, = pl.rec.runs; assert r.metrics_["in_jaws"] == 1.0 and r.metrics_["jaw_retry"] == 0.0 and r.metrics_["not_in_jaws"] == 0.0 and r.metrics_["picked"] == 1.0
    frames = [f for f in r.files if f.startswith("frames/in_jaws_")]; assert len(frames) == 1 and frames[0].endswith("_r1.jpg") and os.path.isfile(jc["frame"]) and jc["frame"].endswith(os.path.basename(frames[0]))
    assert not stops(out)


def test_not_in_the_jaws_once_retries_the_corrected_grasp_then_carries(pl, cc, table, monkeypatch, capsys, tmp_path):
    feed = frame_feed(jaw_frame(ring=False), jaw_frame(ring=True)); monkeypatch.setattr(pl, "fetch_wrist", feed); arm = ShortArm(cc, table)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")]); rec, = records(out); j1, j2 = jaw_lines(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, LIFT, POSE, LIFT, CARRY], "the retry is stage 2 (the corrected hover opens the jaws) plus grasp + lift, once"
    assert feed.calls == 2 and (j1["attempt"], j1["in_jaws"]) == (1, False) and (j2["attempt"], j2["in_jaws"]) == (2, True)
    assert [s["attempt"] for s in stages(out) if s["stage"] == "grasp_pose"] == [1, 2] and [s["attempt"] for s in stages(out) if s["stage"] == "final"] == [1, 2]
    assert rec["picked"] is True and rec["in_jaws"] is True and rec["jaw_retry"] is True and rec["reason"] is None and not stops(out)
    r, = pl.rec.runs; frames = sorted(f for f in r.files if f.startswith("frames/in_jaws_")); assert len(frames) == 2 and frames[0].endswith("_r1.jpg") and frames[1].endswith("_r1_retry.jpg")
    assert r.metrics_["jaw_retry"] == 1.0 and r.metrics_["in_jaws"] == 1.0
    assert set(json.load(open(tmp_path / "s.json"))["biases"]) == {"pregrasp", "grasp", "lift", "hover_tgt", "place", "lift2", "rest"}, "sag is learned from the lift stage that was carried plus the carry; never from the jaw moves"


def test_not_in_the_jaws_twice_opens_over_the_spot_rests_and_stops(pl, cc, table, monkeypatch, capsys, tmp_path):
    feed = frame_feed(jaw_frame(ring=False)); monkeypatch.setattr(pl, "fetch_wrist", feed); arm = ShortArm(cc, table)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "3", "--state", str(tmp_path / "s.json")]); rec, = records(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, LIFT, POSE, LIFT, ABORT], "no carry: the jaws open at the hover over the nut's spot, then rest"
    assert feed.calls == 2 and [j["in_jaws"] for j in jaw_lines(out)] == [False, False]
    assert rec["picked"] is False and rec["reason"] == "not_in_jaws" and rec["in_jaws"] is False and rec["jaw_retry"] is True and rec["jaw_ring"] is None
    assert math.hypot(rec["landed_in"][0] - PROBE_NUT_IN[0], rec["landed_in"][1] - PROBE_NUT_IN[1]) < 0.2, "the landing look is near the nut's spot, not the target"
    s, = stops(out); assert s["reason"] == "not_in_jaws" and "two grasps" in s["stop"] and len(records(out)) == 1, "the loop stops after the failed pick"
    r, = pl.rec.runs; assert r.metrics_["picked"] == 0.0 and r.metrics_["in_jaws"] == 0.0 and r.metrics_["not_in_jaws"] == 1.0 and r.metrics_["jaw_retry"] == 1.0
    saved = json.load(open(tmp_path / "s.json")); assert saved["history"][-1]["reason"] == "not_in_jaws" and saved["place_offsets_in"]["6.00,5.00"] == [0.0, 0.0], "a failed pick teaches the place offset nothing"
    assert "hover_nut" in saved["biases"] and "close1" not in saved["biases"]


def test_the_jaw_check_never_raises(pl, cc, table, monkeypatch, capsys, tmp_path):
    def dead(): raise RuntimeError("wrist feed exploded")
    monkeypatch.setattr(pl, "fetch_wrist", dead); arm = ShortArm(cc, table)
    out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")]); rec, = records(out); jc, = jaw_lines(out)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, LIFT, CARRY] and rec["picked"] is True and rec["in_jaws"] is None and rec["reason"] is None, "unknown -> carry anyway"
    assert jc["in_jaws"] is None and "failed" in jc["note"] and "RuntimeError" in jc["note"] and jc["frame"] is None and "in_jaws" not in pl.rec.runs[0].metrics_
    # the function itself, on every bad input
    assert pl.jaw_check(b"not a jpeg")["in_jaws"] is None and pl.jaw_check(b"")["in_jaws"] is None and "no wrist frame" in pl.jaw_check(b"")["note"]
    assert pl.jaw_check(None)["in_jaws"] is None, "jpeg None fetches, the fetch raises, the check answers unknown"
    assert pl.jaw_check(jaw_frame(ring=True), window=(300, 300, 380, 360))["in_jaws"] is True and pl.jaw_check(jaw_frame(ring=False))["in_jaws"] is False
    away = pl.jaw_check(jaw_frame(ring=True, uv=(340, 372))); assert away["in_jaws"] is False and away["note"] == "ring outside the jaw window", "a ring on the mat just below the jaws is not a held nut"
    assert pl.jaw_check(jaw_frame(ring=True), window=(0, 0, 10, 10))["in_jaws"] is False, "a window too small for the template: not found, no crash"
    out_path = str(tmp_path / "x" / "jaw.jpg"); assert pl.jaw_check(jaw_frame(ring=True, jpeg=False), out_path=out_path)["frame"] == out_path and os.path.isfile(out_path), "a decoded frame is accepted too"
    (tmp_path / "afile").write_text("x"); bad = pl.jaw_check(jaw_frame(ring=True), out_path=str(tmp_path / "afile" / "y.jpg"))
    assert bad["in_jaws"] is True and bad["frame"] is None and "failed" in bad["note"], "an unwritable artifact path is caught, not raised; the verdict stands"


# ---------------------------------------------------------------- 2A(ii) the two-step close
def test_two_step_close_and_fast_close(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table); drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--state", str(tmp_path / "s.json")])
    lift = arm.plans[2]["waypoints"]; assert [w["name"] for w in lift] == LIFT
    assert [w["gripper"] for w in lift] == [35, 35, 20, 0, 0] and [w["settle"] for w in lift] == [0.3, 0.5, 0.4, 0.6, 0.2] and [w["gripper_only"] for w in lift] == [False, False, True, True, False]
    assert lift[2]["joints"] == lift[1]["joints"] == lift[3]["joints"], "the close steps hold the grasp pose; only the jaws move"
    assert pl.CLOSE_MID == 20 and pl.waypoints()[3][:4] == ("close1", "nut", pl.GRASP_H_IN, 20) and pl.waypoints()[4][:4] == ("close2", "nut", pl.GRASP_H_IN, 0) and pl.waypoints()[4][5] == 0.6
    biases = json.load(open(tmp_path / "s.json"))["biases"]; assert "close1" not in biases and "close2" not in biases and "grasp" in biases and "lift" in biases, "learned biases index by waypoint name and skip the jaw-only steps"
    # an old state file with a `close` bias is harmless, and the learned grasp bias is applied to the next plan by name
    st = {"biases": {"close": {"shoulder_pan": 3.0}, "grasp": {"shoulder_pan": 1.5}}, "place_offsets_in": {}, "history": []}; json.dump(st, open(tmp_path / "old.json", "w"))
    arm2 = ShortArm(cc, table); drive(pl, monkeypatch, capsys, arm2, ["--runs", "1", "--state", str(tmp_path / "old.json")])
    g = [w for w in arm2.plans[2]["waypoints"] if w["name"] == "grasp"][0]; assert g["joints"]["shoulder_pan"] == pytest.approx(g["desired"]["shoulder_pan"] + 1.5)
    # --fast-close: one `close` waypoint straight to 0, in the closed loop and in the open loop
    arm3 = ShortArm(cc, table); out3 = drive(pl, monkeypatch, capsys, arm3, ["--runs", "1", "--fast-close", "--state", str(tmp_path / "s3.json")])
    assert [n for n, _ in arm3.calls] == [HOVER, POSE, ["pregrasp", "grasp", "close", "lift"], CARRY]
    c = arm3.plans[2]["waypoints"][2]; assert (c["gripper"], c["settle"], c["seconds"], c["gripper_only"]) == (0, 0.4, 0.6, True) and records(out3)[0]["fast_close"] is True and pl.rec.runs[-1].params_["fast_close"] == 1
    assert set(json.load(open(tmp_path / "s3.json"))["biases"]) == {"pregrasp", "grasp", "lift", "hover_tgt", "place", "lift2", "rest"}
    arm4 = ShortArm(cc, table); drive(pl, monkeypatch, capsys, arm4, ["--runs", "1", "--open-loop", "--fast-close", "--state", str(tmp_path / "s4.json")])
    assert [n for n, _ in arm4.calls] == [["hover_nut", "pregrasp", "grasp", "close", "lift", *CARRY]]
    arm5 = ShortArm(cc, table); drive(pl, monkeypatch, capsys, arm5, ["--runs", "1", "--open-loop", "--state", str(tmp_path / "s5.json")])
    assert [n for n, _ in arm5.calls] == [["hover_nut", *LIFT, *CARRY]], "open loop defaults to the two-step close too"


# ---------------------------------------------------------------- 2B reset: the target sampler
def test_reset_target_sampling_respects_box_avoid_markers_and_the_move(pl, table):
    markers = pl.marker_centres_in(table); assert sorted(markers) == [[2.0, 10.0], [3.0, 4.0], [15.0, 4.0], [16.0, 10.0]]
    nut, avoid, box, rng = (9.0, 6.0), [(7.0, 5.0), (11.0, 7.0), (12.5, 4.5), (9.0, 6.5), (6.2, 7.8)], pl.SAFE_BOX, random.Random(7); pts = []
    for _ in range(500):
        p = pl.sample_reset_target(nut, box, avoid, markers, rng); pts.append(p)
        assert box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3], p
        assert math.hypot(p[0] - nut[0], p[1] - nut[1]) * 25.4 >= 60.0, p
        assert all(math.hypot(p[0] - q[0], p[1] - q[1]) * 25.4 >= 40.0 for q in avoid), p
        assert all(math.hypot(p[0] - q[0], p[1] - q[1]) * 25.4 >= 50.0 for q in markers), p
    xs, ys = {round(p[0]) for p in pts}, {round(p[1]) for p in pts}; assert xs == {10, 11, 12, 13} and len(ys) >= 4, "the samples spread over the free part of the box (the nut's 60 mm disc and the avoid discs leave x 10..13 only)"
    assert pts == [pl.sample_reset_target(nut, box, avoid, markers, random.Random(7)) for _ in range(500)][:1] + pts[1:], "seeded: the first draw repeats"
    assert pl.sample_reset_target(nut, (13.0, 8.0, 6.0, 4.0), avoid, markers, random.Random(1)) is not None, "a box given corner-last is the same box"
    with pytest.raises(SystemExit, match="no clear spot"): pl.sample_reset_target((9.2, 6.2), (9.0, 6.0, 9.5, 6.5), [], markers, random.Random(1))   # every point is within 60 mm of the nut
    grid = [(x, y) for x in range(6, 14) for y in range(4, 9)]   # objects every inch: nowhere is 40 mm clear
    with pytest.raises(SystemExit, match="40 objects to keep 40 mm from"): pl.sample_reset_target((20.0, 20.0), box, grid, markers, random.Random(1))
    with pytest.raises(SystemExit, match="markers at 50 mm"): pl.sample_reset_target((20.0, 20.0), (15.5, 9.5, 16.5, 10.5), [], markers, random.Random(1))   # a box on marker 0


def test_marker_centres_come_from_the_layout_file_or_the_table(pl, table):
    assert pl.load_layout() is None and sorted(pl.marker_centres_in(table, None)) == [[2.0, 10.0], [3.0, 4.0], [15.0, 4.0], [16.0, 10.0]], "the test ROOT has no layout file; the table's layout_markers stand in"
    inches = {"units": "in", "marker_size": 2.0, "markers": {"0": [16, 10], "1": [2, 10], "2": [15, 4], "3": [3, 4]}}
    assert sorted(pl.marker_centres_in(table, inches)) == sorted(pl.marker_centres_in(table)) and pl.marker_size_in(table, inches) == 2.0 and pl.marker_size_in(table) == pytest.approx(2.0)
    mm = {"units": "mm", "marker_size_mm": 50.8, "markers": {"0": [406.4, 254.0]}}; assert pl.marker_centres_in(table, mm) == [[16.0, 10.0]] and pl.marker_size_in(table, mm) == pytest.approx(2.0)


def test_reset_mode_picks_from_wherever_and_places_inside_the_safe_box(pl, cc, table, monkeypatch, capsys, tmp_path):
    others = [BIG_OBJ, {"id": 3, "x_in": 9.0, "y_in": 6.0, "w_mm": 25.0, "h_mm": 25.0}, {"id": 4, "x_in": 7.0, "y_in": 7.0, "w_mm": 15.0, "h_mm": 15.0}]
    monkeypatch.setattr(pl, "fetch_wrist", frame_feed(jaw_frame(ring=True))); arm = ShortArm(cc, table, scene=[NUT_OBJ, *others])
    out = drive(pl, monkeypatch, capsys, arm, ["--reset", "--seed", "3", "--runs", "9", "--state", str(tmp_path / "s.json")]); rec, = records(out); head, = [o for o in out if o.get("reset")]
    assert [n for n, _ in arm.calls] == [HOVER, POSE, LIFT, CARRY], "one closed-loop pick, jaw check included; --runs is not a loop in reset mode"
    assert rec["mode"] == "reset" and rec["control"] == "closed" and rec["safe_box"] == [6.0, 4.0, 13.0, 8.0] and rec["nut_source"] == "near" and rec["picked"] is True and rec["in_jaws"] is True
    t = rec["target_in"]; assert t == head["target_in"] and 6.0 <= t[0] <= 13.0 and 4.0 <= t[1] <= 8.0 and math.hypot(t[0] - PROBE_NUT_IN[0], t[1] - PROBE_NUT_IN[1]) * 25.4 >= 60.0
    assert all(math.hypot(t[0] - o["x_in"], t[1] - o["y_in"]) * 25.4 >= 40.0 for o in others) and head["avoid"] == 3, "the nut itself is not in the avoid list; the other three are"
    assert all(math.hypot(t[0] - m[0], t[1] - m[1]) * 25.4 >= 50.0 for m in pl.marker_centres_in(table)) and head["nut_in"] == [round(v, 3) for v in PROBE_NUT_IN]
    assert math.hypot(rec["landed_in"][0] - t[0], rec["landed_in"][1] - t[1]) * 25.4 < 5.0 and rec["place_offset_in"] == [pytest.approx(-0.05), pytest.approx(0.025)], "the reset's place offset is a throwaway"
    saved = json.load(open(tmp_path / "s.json")); assert saved["history"][-1]["mode"] == "reset" and saved["place_offsets_in"] == {} and "grasp" in saved["biases"], "sag learning carries over, place offsets are untouched"
    r, = pl.rec.runs; assert r.name.startswith("reset-") and r.tags["mode"] == "reset" and r.params_["mode"] == "reset" and r.metrics_["picked"] == 1.0
    # the same with a custom box and seed: reproducible, inside the box
    arm2 = ShortArm(cc, table, scene=[NUT_OBJ]); out2 = drive(pl, monkeypatch, capsys, arm2, ["--reset", "--seed", "11", "--safe-box", "7", "5", "8", "6", "--state", str(tmp_path / "s2.json")])
    t2 = records(out2)[0]["target_in"]; assert 7.0 <= t2[0] <= 8.0 and 5.0 <= t2[1] <= 6.0 and records(out2)[0]["safe_box"] == [7.0, 5.0, 8.0, 6.0]
    assert t2 == pl.sample_reset_target(PROBE_NUT_IN, (7, 5, 8, 6), [], pl.marker_centres_in(table), random.Random(11))
    # the nut hidden from the near search: the scene map finds it; nothing anywhere: a clear SystemExit
    arm3 = ShortArm(cc, table, found=False, scene=[BIG_OBJ, NUT_OBJ]); out3 = drive(pl, monkeypatch, capsys, arm3, ["--reset", "--seed", "1", "--state", str(tmp_path / "s3.json")])
    assert records(out3)[0]["nut_source"] == "scene" and records(out3)[0]["picked"] is True and len(arm3.calls) == 4
    arm4 = ShortArm(cc, table, found=False, scene=[BIG_OBJ])
    with pytest.raises(SystemExit, match="nut not found anywhere"): drive(pl, monkeypatch, capsys, arm4, ["--reset", "--state", str(tmp_path / "s4.json")])
    assert arm4.calls == []


def test_reset_nut_as_a_function_with_a_given_avoid_list(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table); monkeypatch.setattr(pl, "find_nut", arm.find_nut); monkeypatch.setattr(pl, "api", arm.api); monkeypatch.setattr(pl, "run_plan", arm.run_plan)
    a = pl.default_args(); a.state = str(tmp_path / "s.json"); a.jaw_check = False; state = {"biases": {}, "place_offsets_in": {}, "history": []}; said = []
    avoid = [list(PROBE_NUT_IN), [8.0, 6.0], [12.0, 7.5]]   # the caller passed every scene object, the nut included
    rec = pl.reset_nut(state, cc, table, (6, 4, 13, 8), avoid, random.Random(5), a=a, say=said.append, markers_in=pl.marker_centres_in(table), session="t", track=pl.rec)
    assert [n for n, _ in arm.calls] == [HOVER, POSE, [*LIFT, *CARRY]] and rec["mode"] == "reset" and rec is state["history"][-1] and said[0]["avoid"] == 2 and said[0]["reset"] is True
    t = rec["target_in"]; assert all(math.hypot(t[0] - q[0], t[1] - q[1]) * 25.4 >= 40.0 for q in avoid[1:]) and math.hypot(t[0] - PROBE_NUT_IN[0], t[1] - PROBE_NUT_IN[1]) * 25.4 >= 60.0
    assert a.max_deg_s == 50.0 and a.jaw_open == 35 and a.jaw_window == [300, 300, 380, 360] and a.lost_after == 3 and a.safe_box == [6.0, 4.0, 13.0, 8.0] and a.fast_close is False
    out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]; assert out[-1]["mode"] == "reset", "the record is printed like any pick's"


# ---------------------------------------------------------------- 2B the whole-mat finder
def test_find_nut_anywhere_prefers_the_scene_object_nearest_the_last_spot(pl, cc, table, monkeypatch):
    real = pl.fetch_scene_objects; monkeypatch.setattr(pl, "fetch_scene_objects", lambda: (_ for _ in ()).throw(AssertionError("given objects: the dashboard is not asked")))
    objs = [{"x_in": 5.0, "y_in": 9.0, "w_mm": 12.0, "h_mm": 10.0}, {"x_in": 12.0, "y_in": 5.0, "w_mm": 11.0, "h_mm": 11.0}, BIG_OBJ, {"x_in": 8.0, "y_in": 6.0, "w_mm": 7.0, "h_mm": 7.0}, {"x_in": 9.0, "y_in": 6.0, "w_mm": 19.0, "h_mm": 24.0}]
    assert pl.find_nut_anywhere(cc, table, objs, near_in=(11.0, 5.5)) == ([12.0, 5.0], "scene") and pl.find_nut_anywhere(cc, table, objs, near_in=(4.0, 8.0)) == ([5.0, 9.0], "scene")
    assert pl.find_nut_anywhere(cc, table, objs) == ([5.0, 9.0], "scene"), "no last spot: any candidate (the first)"
    assert [o["x_in"] for o in pl.nut_candidates(objs)] == [5.0, 12.0], "8 <= w <= 20 and h <= 20: the receptacle (60x40), a speck (7) and a 19x24 blob are not the nut"
    assert pl.nut_candidates([{"w_mm": "x"}, {}, None, {"w_mm": 12, "h_mm": 10, "x_in": 1, "y_in": 1}]) == [{"w_mm": 12, "h_mm": 10, "x_in": 1, "y_in": 1}], "junk objects are skipped, not raised on"
    # the dashboard path: /api/scene answers, or does not
    monkeypatch.setattr(pl, "fetch_scene_objects", real); monkeypatch.setattr(pl, "api", lambda path: {"ok": True, "objects": [NUT_OBJ]} if path == "/api/scene" else {}); assert pl.fetch_scene_objects() == [NUT_OBJ] and pl.find_nut_anywhere(cc, table)[1] == "scene"
    monkeypatch.setattr(pl, "api", lambda path: {"ok": False, "err": "top camera not live"}); assert pl.fetch_scene_objects() == []
    monkeypatch.setattr(pl, "api", lambda path: (_ for _ in ()).throw(OSError("down"))); assert pl.fetch_scene_objects() == []


def test_find_nut_anywhere_falls_back_to_the_masked_whole_mat_search(pl, cc, table, monkeypatch):
    monkeypatch.setattr(pl, "fetch_top", lambda: (_ for _ in ()).throw(AssertionError("jpeg given: the dashboard is not asked")))
    far = overhead_frame(cc, table, [(5.0, 9.0)])   # the nut far from its last spot (11.4, 6.3): find_nut's 70 px window misses it
    assert pl.find_nut(cc, table, PROBE_NUT_IN, jpeg=far) is None
    got = pl.find_nut_anywhere(cc, table, [], near_in=PROBE_NUT_IN, jpeg=far); assert got and got[1] == "mat" and math.hypot(got[0][0] - 5.0, got[0][1] - 9.0) * 25.4 < 3.0
    assert pl.find_nut_anywhere(cc, table, [BIG_OBJ], near_in=PROBE_NUT_IN, jpeg=far)[1] == "mat", "a scene map with no nut-sized object is no answer; the image is"
    two = overhead_frame(cc, table, [(5.0, 9.0), (13.0, 5.0)]); near = pl.find_nut_anywhere(cc, table, [], near_in=(12.0, 5.0), jpeg=two)[0]; assert math.hypot(near[0] - 13.0, near[1] - 5.0) * 25.4 < 3.0, "nearest to the last spot"
    centre = pl.find_nut_anywhere(cc, table, [], jpeg=two)[0]; assert math.hypot(centre[0] - 5.0, centre[1] - 9.0) * 25.4 < 3.0 or math.hypot(centre[0] - 13.0, centre[1] - 5.0) * 25.4 < 3.0
    # the marker squares are masked: a dark blob on marker 0's square (16, 10) is not a nut
    on_marker = overhead_frame(cc, table, [(16.0, 10.0)]); assert pl.find_nut_anywhere(cc, table, [], near_in=PROBE_NUT_IN, jpeg=on_marker) is None
    unmasked = pl.find_nut_mat(cc, table, PROBE_NUT_IN, jpeg=on_marker, markers_in=[]); assert unmasked and math.hypot(unmasked[0][0] - 16.0, unmasked[0][1] - 10.0) * 25.4 < 3.0, "without the mask the same blob is found: the mask is what hides it"
    edge = overhead_frame(cc, table, [(16.0, 8.9)]); assert pl.find_nut_anywhere(cc, table, [], jpeg=edge) is None, "the margin (10 mm) round the square counts as marker too"
    beside = overhead_frame(cc, table, [(16.0, 8.4)]); assert pl.find_nut_anywhere(cc, table, [], jpeg=beside)[1] == "mat", "just outside the margin it is a nut again"
    m = pl.mat_mask(cc, table, (table["image_size"][1], table["image_size"][0]), pl.marker_centres_in(table)); u, v = cc.table_to_pixel(table, 16 * 25.4, 10 * 25.4); assert m[int(v), int(u)] == 0 and m[int(v), int(u) + 60] == 255, "pixel u falls as mat x grows: +60 px is 2 in inboard of the marker, on the mat (-60 is past the 18 in edge)"
    # an empty mat, off-mat blobs, and garbage frames
    assert pl.find_nut_anywhere(cc, table, [], near_in=PROBE_NUT_IN, jpeg=overhead_frame(cc, table, [])) is None
    assert pl.find_nut_anywhere(cc, table, [], jpeg=overhead_frame(cc, table, [(-1.0, 6.0), (9.0, 13.0)])) is None, "blobs outside the mat area are not searched"
    assert pl.find_nut_anywhere(cc, table, [], jpeg=b"not a jpeg") is None and pl.find_nut_mat(cc, table, jpeg=b"") is None


def test_the_loop_uses_the_finder_when_the_near_search_fails_and_stops_lost_after_three_misses(pl, cc, table, monkeypatch, capsys, tmp_path):
    arm = ShortArm(cc, table, found=False, scene=[BIG_OBJ, NUT_OBJ]); out = drive(pl, monkeypatch, capsys, arm, ["--runs", "1", "--no-jaw-check", "--state", str(tmp_path / "s.json")])
    note, = [o for o in out if o.get("note")]; rec, = records(out)
    assert note["source"] == "scene" and note["nut_in"] == [round(v, 3) for v in PROBE_NUT_IN] and rec["nut_source"] == "scene" and rec["picked"] is True and len(arm.calls) == 3
    sleeps = []; monkeypatch.setattr(pl.time, "sleep", sleeps.append)
    arm2 = ShortArm(cc, table, found=False, scene=[BIG_OBJ]); out2 = drive(pl, monkeypatch, capsys, arm2, ["--runs", "4", "--state", str(tmp_path / "s2.json")])   # the header-only top JPEG decodes to nothing: the image search misses too
    misses = [o for o in out2 if "miss" in o]; s, = stops(out2)
    assert [m["miss"] for m in misses] == [1, 2, 3] and all(m["lost_after"] == 3 and m["run"] == 1 for m in misses) and s == {"run": 1, "stop": "lost", "misses": 3, "near": [9.0, 6.0]}
    assert arm2.calls == [] and not records(out2) and len(pl.rec.runs) == 1 and len(sleeps) == 2, "nothing moved, nothing recorded; a miss does not spend a run; it waits between looks"
    arm3 = ShortArm(cc, table, found=False); out3 = drive(pl, monkeypatch, capsys, arm3, ["--runs", "2", "--lost-after", "1", "--state", str(tmp_path / "s3.json")])
    assert stops(out3) == [{"run": 1, "stop": "lost", "misses": 1, "near": [9.0, 6.0]}] and arm3.calls == []
    # a miss between two finds resets the count
    arm4 = ShortArm(cc, table, found=False, scene=[]); seen = iter([[], [], [NUT_OBJ], [], [], [NUT_OBJ]]); monkeypatch.setattr(pl, "fetch_scene_objects", lambda: next(seen))
    monkeypatch.setattr(pl, "find_nut", arm4.find_nut); monkeypatch.setattr(pl, "api", arm4.api); monkeypatch.setattr(pl, "run_plan", arm4.run_plan)
    import sys; monkeypatch.setattr(sys, "argv", ["pick_loop.py", "--runs", "2", "--no-jaw-check", "--state", str(tmp_path / "s4.json")]); pl.main()
    out4 = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [o["miss"] for o in out4 if "miss" in o] == [1, 2, 1, 2] and [r["run"] for r in records(out4)] == [1, 2] and not stops(out4) and len(arm4.calls) == 6
