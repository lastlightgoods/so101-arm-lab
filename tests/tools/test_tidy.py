"""tools/tidy.py on fakes: the ring fit, the obstacle check, the route, the trims, the grip check, the miss path, keep/reject routing, the guardrails, the reset (lift height, set-down rules,
re-fit, spot sampling, the pile split) and the loop. No dashboard, no ssh, no arm: ShortArm (real kinematics) plus a small World that moves toys when the fake arm releases them."""
import argparse, json, math, os, random, types
import numpy as np, pytest
from pickhelpers import ShortArm, Recorder, solve_joints, mat_to_base, SEED, load, TOOLS
from autohelpers import Clock

RING = (9.0, 6.0)


@pytest.fixture(scope="session")
def td(): return load(os.path.join(TOOLS, "tidy.py"), "tidy_under_test")


@pytest.fixture(scope="session")
def gr(): return load(os.path.join(TOOLS, "guardrails.py"), "guardrails_for_tidy_test")


def tipped(cc, table):   # a fake tipped IK: the same vertical solve, at the asked height, with the asked roll
    def f(x, y, h_mm, roll=None):
        j = solve_joints(cc, table, SEED, mat_to_base(cc, table, x, y, h_mm)); j["wrist_roll"] = SEED["wrist_roll"] if roll is None else roll; return j
    return f


def toy(x, y, w=12, h=10, area=88, angle=0.0): return {"x_in": x, "y_in": y, "w_mm": w, "h_mm": h, "area_mm2": area, "angle_deg": angle}
def frog(x, y): return {"x_in": x, "y_in": y, "w_mm": 59.0, "h_mm": 57.0, "area_mm2": 2100.0, "angle_deg": 0.0}   # the live frogs: 59 x 57 / 67 x 57 mm, 2056-2444 mm2


class FakeRec:
    rejects_repo = "x/rej"
    def __init__(self, alive=True): self.calls = []; self.ep = None; self._alive = alive
    def alive(self): return self._alive
    def open(self, meta=None, t=None): self.calls.append(("open", meta)); self.ep = types.SimpleNamespace(meta=dict(meta or {}))
    def end(self, t=None): self.calls.append(("end",)); return {"ok": True, "reasons": []}
    def commit(self, keep, verdict=None): self.calls.append(("commit", keep, verdict)); self.ep = None; return {"dataset": "main" if keep else "rejects", "episode_index": 3, "frames": 100, "health": {"ok": True}}
    def abort(self, reason="aborted"): self.calls.append(("abort", reason)); self.ep = None; return True


class World:
    """Toys on the mat. The fake arm 'holds' (gripper reads 5 after the lift) or not; a held toy follows the tip to the waypoint that opens the jaws."""
    def __init__(self, cc, table, pl, arm, toys, hold=True, frogs=()):
        self.cc, self.table, self.pl, self.arm, self.toys, self.hold, self.holding, self.at, self.names, self.rows = cc, table, pl, arm, [dict(t) for t in toys], hold, None, None, [], []
        self.frogs = [frog(*f) for f in frogs]; self.clock = Clock()

    def frame(self, ring):   # the top camera: a light mat, the blue rim, and a dark disc for every toy
        import cv2
        w, h = self.table.get("image_size") or [640, 360]; img = np.full((h, w, 3), 190, np.uint8); u, v = self.cc.table_to_pixel(self.table, ring[0] * 25.4, ring[1] * 25.4)
        ppm = abs(self.cc.table_to_pixel(self.table, ring[0] * 25.4 + 10, ring[1] * 25.4)[0] - u) / 10.0; cv2.circle(img, (int(round(u)), int(round(v))), int(46 * ppm), (200, 60, 40), 5)
        for t in self.toys:
            a, b = self.cc.table_to_pixel(self.table, t["x_in"] * 25.4, t["y_in"] * 25.4); cv2.circle(img, (int(round(a)), int(round(b))), 4, (40, 40, 40), -1)
        for f in self.frogs:   # an orange frog with blue legs: colourful, and its blue must not pull the ring fit
            a, b = self.cc.table_to_pixel(self.table, f["x_in"] * 25.4, f["y_in"] * 25.4); cv2.circle(img, (int(round(a)), int(round(b))), int(28 * ppm), (30, 120, 240), -1); cv2.circle(img, (int(round(a)) + int(20 * ppm), int(round(b))), int(8 * ppm), (200, 60, 40), -1)
        return cv2.imencode(".png", img)[1].tobytes()

    def scene(self): return {"ok": True, "objects": [dict(t, id=i) for i, t in enumerate(self.toys + self.frogs)], "pending": 0, "n_markers_seen": 4, "calib_ok": True, "holding": False}
    def state(self): return {"jetson": {"online": True}, "arms": {"follower": {"pos": {"shoulder_pan": 0}}}}

    def run_plan(self, plan, max_deg_s):
        rows = self.arm.run_plan(plan, max_deg_s); self.names += [r["wp"] for r in rows]; self.rows += rows
        for r in rows:
            x, y, _ = self.pl.fk_mat(self.cc, self.table, r["reached"])
            if r["wp"] == "grasp": self.at = (x, y)
            if r["wp"] == "lift":
                r["reached"]["gripper"] = 5.0 if self.hold else 0.0
                self.holding = min(self.toys, key=lambda t: np.hypot(t["x_in"] - self.at[0], t["y_in"] - self.at[1])) if (self.hold and self.at and self.toys) else None
            if r["wp"] in ("open", "place") and self.holding is not None: self.holding["x_in"], self.holding["y_in"] = x, y; self.holding = None
        return rows


@pytest.fixture
def mk(td, gr, pl, cc, table, tmp_path, monkeypatch):
    def build(toys, hold=True, ring=RING, rec=None, spread_rec=None, frogs=(), **kw):
        arm = ShortArm(cc, table, shortfall_mm=2.0); w = World(cc, table, pl, arm, toys, hold, frogs)
        monkeypatch.setattr(pl, "api", arm.api); monkeypatch.setattr(pl, "run_plan", w.run_plan); monkeypatch.setattr(pl, "fetch_scene_objects", lambda: w.scene()["objects"]); monkeypatch.setattr(pl, "fetch_wrist", lambda: None)
        rig = td.Rig(pl, cc, table, ring, ik_free=tipped(cc, table)); notes = []; ctx = gr.Ctx(w.state, w.scene, w.clock.now, w.clock.sleep, notes.append)
        guard = gr.Guard(cc, table, gr.Limits(ceiling_in=4.6, stop_file=str(tmp_path / "stop")), ctx); out = []
        a = argparse.Namespace(jaw_open=None, no_roll=True, grip_check="own", session="t", lighting_tag=None, order="clean", temp_limit=60.0, out_dir=str(tmp_path / "out"), ring_box=list(td.RING_BOX_IN), spread_box=list(td.SPREAD_BOX_IN), nudge=True, blob_mask=False)
        for k, v in kw.items(): setattr(a, k, v)
        eps = {"ring": td.Episode(rec, "lastlight/so101-toys-to-ring"), "spread": td.Episode(spread_rec, "lastlight/so101-toys-spread")}; td.Episode.active["rec"] = None
        t = td.Tidy(pl, cc, table, rig, guard, ctx, Recorder(), a, random.Random(1), eps, out=lambda l: out.append(json.loads(l)), fetch_top=lambda: w.frame(rig.ring))
        monkeypatch.setattr(t, "refit_ring", lambda: rig.ring)
        return types.SimpleNamespace(t=t, w=w, arm=arm, rig=rig, guard=guard, out=out, stop=str(tmp_path / "stop"), a=a)
    return build


# ---------------------------------------------------------------- the ring in the frame
def synth_frame(cc, table, ring_in, thick=5):
    import cv2
    w, h = table.get("image_size") or [640, 360]; img = np.full((h, w, 3), 190, np.uint8); u, v = cc.table_to_pixel(table, ring_in[0] * 25.4, ring_in[1] * 25.4)
    ppm = abs(cc.table_to_pixel(table, ring_in[0] * 25.4 + 10, ring_in[1] * 25.4)[0] - u) / 10.0; cv2.circle(img, (int(round(u)), int(round(v))), int(46 * ppm), (200, 60, 40), thick)   # BGR: blue
    cv2.rectangle(img, (0, h - 30), (w, h), (210, 70, 50), -1)   # the camera's own blue tape on the bottom edge: must be ignored
    return cv2.imencode(".png", img)[1].tobytes()


def test_ring_fit_on_a_synthetic_frame_and_not_found_on_a_blank_one(td, cc, table):
    for ring in ((9.0, 6.0), (7.5, 5.0)):
        got = td.fit_ring(synth_frame(cc, table, ring), cc, table); assert got is not None and np.hypot(got[0] - ring[0], got[1] - ring[1]) * 25.4 < 3.0
    import cv2
    assert td.fit_ring(cv2.imencode(".png", np.full((360, 640, 3), 190, np.uint8))[1].tobytes(), cc, table) is None


def test_toy_kinds_and_the_ring_is_not_a_toy(td):
    assert td.toy_kind(toy(1, 1, 12, 10)) == ("nut", 4.0) and td.toy_kind(toy(1, 1, 20, 12)) == ("wing nut", 4.0) and td.toy_kind(toy(1, 1, 21, 8)) == ("spacer", 5.0) and td.toy_kind(toy(1, 1, 6, 6)) == ("spacer", 5.0) and td.toy_kind(toy(1, 1, 40, 30))[0] == "unknown"
    assert td.is_toy(toy(5, 5)) and not td.is_toy(toy(9, 6, 104, 104, 1500)) and not td.is_toy({"w_mm": 5})


# ---------------------------------------------------------------- the obstacle check and the route
def pose_at(cc, table, x, y, h_mm): return solve_joints(cc, table, SEED, mat_to_base(cc, table, x, y, h_mm))


def test_a_path_through_the_rim_is_caught_and_a_clean_one_passes(td, pl, cc, table, mk):
    r = mk([]).rig; w = 2.0   # the rim sits 2.05 in from the ring centre
    low_a, low_b = pose_at(cc, table, RING[0] - w, RING[1], 10.0), pose_at(cc, table, RING[0] + w, RING[1], 10.0)
    assert r.hits(low_a) and not r.path_clean(low_a, low_b), "a low tip on the rim, and a sweep straight across the ring, are both blocked"
    far_a, far_b = pose_at(cc, table, 5.0, 3.0, 10.0), pose_at(cc, table, 5.0, 9.0, 10.0)
    assert r.path_clean(far_a, far_b) and not r.hits(far_a), "a path well away from the ring is clean"
    over = r.ik_free(RING[0], RING[1], 110.0); assert not r.hits(over), "over the ring at 110 mm clears the 51 mm rim"
    assert r.droop_mm(*RING) > 4.0 and r.droop_mm(*r.base) == 4.0 and r.droop_mm(r.base[0] + 150 / 25.4, r.base[1]) == pytest.approx(4.0), "sag allowance: 4 mm at 150 mm"
    assert r.droop_mm(r.base[0] + 290 / 25.4, r.base[1]) == pytest.approx(24.0), "... and 24 mm at 290 mm"


def test_the_adaptive_route_goes_straight_when_clean_and_over_the_base_when_not(td, pl, cc, table, mk):
    m = mk([]); r = m.rig
    assert r.start_route((5.0, 3.0), 2.0) == [], "a toy away from the ring: rest -> straight to it"
    blocked = None
    for y in (9.0, 8.0, 7.5):   # a toy on the far side of the ring: the straight path from rest crosses the rim
        if r.start_route((RING[0], y), 1.0): blocked = y; break
    if blocked is None: pytest.skip("no far-side spot blocks the straight route with this fit")
    names = [n for n, _ in r.start_route((RING[0], blocked), 1.0)]; assert names == ["high_back", "high_back_up", "high", "above"]


def test_pieces_of_the_ring_wall_are_not_toys_but_toys_in_the_middle_and_well_outside_are(td):
    ring = (6.085, 6.658)   # 11 Nov 22:0x live: the scene map cut the ring into blobs at (5.5,4.62) (7.17,5.02) (4.56,5.77) (7.25,7.38) (5.53,7.98), 35-54 mm out
    wall = [toy(5.5, 4.62, 30, 30, 620), toy(7.17, 5.02, 53.2, 12, 384), toy(4.56, 5.77, 25.4, 8.7, 96), toy(7.25, 7.38, 23.2, 9.1, 120), toy(5.53, 7.98, 31.3, 30.7, 704)]
    assert not any(td.is_toy(o, ring=ring) for o in wall) and all(td.is_toy(o) for o in wall), "without a ring the old rule still applies"
    assert td.is_toy(toy(5.96, 5.9, 17.6, 10.5, 132), ring=ring) and td.is_toy(toy(2.37, 6.42, 20.4, 13.3, 168), ring=ring), "the wing nut inside (19 mm) and the spacer 95 mm out are toys"


def test_check_gives_a_verdict_per_toy(td, pl, cc, table, mk):
    r = mk([]).rig
    ok = r.check((5.0, 3.5)); assert ok["reachable"] and ok["ok"] and ok["direct"] is True and ok["n_bad"] == 0
    on_rim = r.check((RING[0] + 2.0, RING[1])); assert on_rim["reachable"] and not on_rim["ok"] and on_rim["n_bad"] > 0 and on_rim["why"] == "route passes the ring" and on_rim["worst"]


# ---------------------------------------------------------------- the trims
def test_the_trims_no_repeated_poses_and_the_carry_ends_over_the_ring(td, pl, cc, table, mk):
    m = mk([]); r = m.rig; orig_wp, orig_bp = pl.WP, pl.build_plan; r.end_direct = False
    with r.patched(2.0, 1.5):
        plans = [pl.build_plan([5.0, 3.5], list(RING), {}, names=nm, tilt_deg=pl.GRASP_TILT_DEG, cc=cc, table=table) for nm in (pl.STAGES[0], pl.STAGES[1], pl.STAGES[2])]
    assert pl.WP is orig_wp and pl.build_plan is orig_bp, "pick_loop is put back"
    names = [[w["name"] for w in p["waypoints"]] for p in plans]
    assert names[0] == ["hover_pre", "hover_nut"] and names[1] == ["hover_pre", "grasp"], "stage 2's hover is merged into the descent"
    assert "place" not in names[2] and names[2][-4:] == ["hover_tgt", "open", "lift2", "rest"]
    assert all(w["joints"] == r.high for w in plans[2]["waypoints"] if w["name"] in ("hover_tgt", "open")), "released over the ring at HIGH"
    assert [w["gripper"] for w in plans[2]["waypoints"] if w["name"] in ("hover_tgt", "open")] == [0, pl.JAW_OPEN], "closed over the ring, then open"
    seq = [w for p in plans for w in p["waypoints"] if not w["gripper_only"]]
    for a, b in zip(seq, seq[1:]): assert a["joints"] != b["joints"], f"{a['name']} -> {b['name']} repeats a pose"
    assert [w["settle"] for p in plans for w in p["waypoints"] if w["name"] in ("hover_pre", "pregrasp_up")] == [0.1, 0.1, 0.1]
    r.end_direct = True
    with r.patched(2.0, 1.5): p = pl.build_plan([5.0, 3.5], list(RING), {}, names=pl.STAGES[2], tilt_deg=pl.GRASP_TILT_DEG, cc=cc, table=table)
    assert "lift2" not in [w["name"] for w in p["waypoints"]], "straight home from over the ring when that path is clean"
    with r.patched(2.0, 1.5): ab = pl.build_plan([5.0, 3.5], list(RING), {}, names=pl.ABORT_STAGE, tilt_deg=pl.GRASP_TILT_DEG, cc=cc, table=table)
    assert [w["name"] for w in ab["waypoints"]] == ["hover_nut", "rest"], "the abort's back-off hover is dropped"


# ---------------------------------------------------------------- the grip check and the keep rule
@pytest.mark.parametrize("g,cam,kind,want", [(5.0, None, "nut", True), (5.0, False, "spacer", True), (2.01, False, "nut", True), (2.0, None, "nut", False), (0.4, False, "nut", False),
                                              (0.54, True, "spacer", False), (0.54, True, "nut", False), (1.2, True, "unknown", False), (None, True, "nut", None), (None, None, "nut", None),
                                              (0.4, True, "wing nut", True), (1.2, True, "wing nut", True), (0.4, False, "wing nut", False), (None, True, "wing nut", True), (None, None, "wing nut", None)])
def test_grip_decision_truth_table(td, g, cam, kind, want): assert td.grip_decision(g, cam, kind) is want, "the camera alone never passes a nut or a spacer"


@pytest.mark.parametrize("held,empty,placed,trip,health,keep", [(True, True, True, None, True, True), (False, True, True, None, True, False), (None, True, True, None, True, False), (True, False, True, None, True, False),
                                                              (True, True, False, None, True, False), (True, True, True, "motion", True, False), (True, True, True, None, False, False)])
def test_keep_decision(td, held, empty, placed, trip, health, keep):
    k, why = td.keep_decision(held, empty, placed, trip, health); assert k is keep and (not why) is keep


# ---------------------------------------------------------------- picks: held, miss, recording routing
def test_a_held_pick_carries_to_the_ring_and_is_kept(mk):
    rec = FakeRec(); m = mk([toy(5.0, 3.5)], hold=True, rec=rec); res = m.t.ring_pick(m.w.toys[0])
    assert res["held"] is True and res["spot_empty"] and res["in_ring"] and res["ok"] and res["keep"] and res["trip"] is None
    assert "hover_tgt" in m.w.names and "open" in m.w.names and "place" not in m.w.names
    kinds = [c[0] for c in rec.calls]; assert kinds == ["open", "end", "commit"] and rec.calls[2][1] is True
    meta = rec.calls[0][1]; assert meta["task"] == "Pick up the toy and put it in the tape ring" and meta["scene"]["toy_kind"] == "nut" and meta["scene"]["aim_mm"] == 4.0 and meta["scene"]["ring_in"] == [9.0, 6.0] and meta["scene"]["lighting_tag"] == "untagged"
    assert res["demo"]["kept"] and res["demo"]["dataset_repo"] == "lastlight/so101-toys-to-ring"
    assert any(f.endswith("timeline.json") for f in m.t.track.runs[-1].files) and os.path.exists(os.path.join(m.a.out_dir, "tidy_t", "pick_1.json"))


def test_a_miss_retries_once_then_goes_home_without_the_carry_and_is_rejected_not_dropped(mk):
    rec = FakeRec(); m = mk([toy(5.0, 3.5)], hold=False, rec=rec); res = m.t.ring_pick(m.w.toys[0])
    assert res["held"] is False and res["reason"] == "not_in_jaws" and not res["ok"] and not res["keep"] and "not_held" in res["keep_reasons"]
    assert "hover_tgt" not in m.w.names and "open" not in m.w.names or m.w.names.count("hover_tgt") == 0, "no carry"
    assert m.w.names[-1] == "rest" and m.w.names.count("lift") == 2, "one retry, then home"
    assert [c[0] for c in rec.calls] == ["open", "end", "commit"] and rec.calls[2][1] is False and res["demo"]["dataset"] == "rejects" and res["demo"]["discard_reason"] == "not_held"


def test_a_pick_that_leaves_the_toy_outside_the_ring_is_rejected(mk):
    rec = FakeRec(); m = mk([toy(5.0, 3.5)], hold=True, rec=rec)
    orig = m.w.run_plan
    def drop_en_route(plan, v):   # the toy slips on the way: it ends up elsewhere on the mat
        rows = orig(plan, v)
        if any(r["wp"] == "open" for r in rows): m.w.toys[0]["x_in"], m.w.toys[0]["y_in"] = 4.0, 8.0
        return rows
    m.t.pl.run_plan = drop_en_route; res = m.t.ring_pick(m.w.toys[0])
    assert res["held"] is True and not res["in_ring"] and not res["keep"] and "not_in_ring" in res["keep_reasons"] and rec.calls[-1][1] is False


def test_a_recorder_that_refuses_to_open_never_stops_the_arm(mk):
    class Bad(FakeRec):
        def open(self, meta=None, t=None): raise RuntimeError("no cameras")
    m = mk([toy(5.0, 3.5)], hold=True, rec=Bad()); res = m.t.ring_pick(m.w.toys[0])
    assert res["ok"] and res["demo"]["kept"] is False and res["demo"]["discard_reason"].startswith("recorder_open_failed")


def test_a_trip_mid_pick_aborts_the_episode_and_commits_nothing(mk):
    rec = FakeRec(); m = mk([toy(5.0, 3.5)], hold=True, rec=rec)
    orig = m.arm.run_plan
    def bad_row(plan, v):
        rows = orig(plan, v)
        for r in rows: r["max_err_deg"] = 30.0
        return rows
    m.arm.run_plan = bad_row; res = m.t.ring_pick(m.w.toys[0])
    assert res["trip"] == "motion" and not res["ok"] and ("abort", "motion") in rec.calls and not any(c[0] == "commit" for c in rec.calls) and res["demo"] is None


def test_without_record_there_is_no_episode_and_no_ticks_flag(td, mk, pl):
    m = mk([toy(5.0, 3.5)], hold=True); res = m.t.ring_pick(m.w.toys[0]); assert res["demo"] is None and getattr(pl, "PICK_SEQ_ARGS", "") == ""


def test_a_blocked_toy_is_refused_before_any_motion(mk):
    m = mk([toy(RING[0] + 2.0, RING[1])]); res = m.t.ring_pick(m.w.toys[0]); assert res.get("refused") and m.w.names == [] and not res["ok"]


# ---------------------------------------------------------------- the round: guardrails and skipping
def test_the_stop_file_ends_the_run_before_the_first_pick(mk):
    m = mk([toy(5.0, 3.5)]); open(m.stop, "w").write("x"); s = m.t.round()
    assert s["stop"] == "stop_file" and s["picked"] == 0 and m.w.names == []


def test_three_consecutive_failures_stop_the_run(mk):
    m = mk([toy(5.0, 3.5), toy(5.0, 8.0), toy(6.0, 4.5), toy(12.0, 3.0), toy(12.0, 9.0)], hold=False); s = m.t.round()
    assert s["stop"] == "consecutive_failures" and s["picked"] == 3 and s["failed"] == 3 and s["ok"] == 0


def test_toys_inside_the_ring_are_skipped_and_the_rest_are_tidied(mk):
    m = mk([toy(RING[0] + 0.6, RING[1] - 0.4), toy(5.0, 3.5)], hold=True); s = m.t.round()   # why: a toy already at the release point would hide the new one in the interior diff (false reject, the safe direction)
    assert s["stop"] == "all_tidy" and s["picked"] == 1 and s["ok"] == 1 and m.w.names.count("grasp") >= 1
    only = mk([toy(RING[0] + 0.3, RING[1] - 0.2)]); s2 = only.t.round(); assert s2["stop"] == "all_tidy" and s2["picked"] == 0 and only.w.names == []


def test_a_round_refuses_when_no_toy_has_a_clean_route(mk):
    m = mk([toy(RING[0] + 2.6, RING[1])]); s = m.t.round(); assert s["stop"].startswith("refused")   # 66 mm out: clear of the wall band (on_ring_wall), still behind the ring and s["picked"] == 0 and s["refused"] and m.w.names == []


def test_arm_busy_and_temperature_stop_the_run(td, mk):
    m = mk([toy(5.0, 3.5)]); m.t.ctx.arm_jobs = lambda: ["lerobot-teleoperate --x"]; assert m.t.round()["stop"] == "arm_busy"
    m2 = mk([toy(5.0, 3.5)]); m2.w.state = lambda: {"jetson": {"online": True}, "arms": {"follower": {"pos": {"a": 1}, "temps": {"shoulder": 71, "elbow": 40}}}}; m2.t.ctx.fetch_state = m2.w.state
    assert m2.t.round()["stop"] == "temperature"
    assert td.temp_trip({"arms": {"follower": {}}}, 60, lambda *a: a) is None and td.has_temp({"arms": {"follower": {"temp_c": 40}}})


def test_max_limits_the_picks(mk):
    m = mk([toy(5.0, 3.5), toy(5.0, 8.0), toy(6.0, 4.5)]); s = m.t.round(max_n=2); assert s["picked"] == 2 and s["stop"] == "max"


# ---------------------------------------------------------------- reset (a): the ring lift
def test_lift_height_after_the_sag(td):
    cmd, bottom = td.lift_height_mm(230.0, 150.0); assert cmd == td.RING_LIFT_CAP_MM and bottom == pytest.approx(110 - 24 - 15) and bottom >= td.MIN_BOTTOM_MM
    cmd, bottom = td.lift_height_mm(230.0, 70.0); assert cmd == 70.0 and bottom == pytest.approx(70 - 24 - 15) and bottom < td.MIN_BOTTOM_MM
    assert td.loaded_sag_mm(50.0) == 8.0 and td.loaded_sag_mm(290.0) > td.loaded_sag_mm(230.0), "the sag grows with reach, with a floor"
    assert td.ring_bottom_mm(78, 230.0, grip_h_mm=25) < td.ring_bottom_mm(78, 230.0), "gripping lower lifts the bottom higher for the same tip height"


def test_the_ring_plan_aims_at_40_mm_and_refuses_a_lift_that_cannot_clear(td, pl, cc, table, mk):
    m = mk([]); mover = td.RingMover(m.t); P = mover.plan(RING, (8.0, 6.5))
    assert P["bottom_mm"] >= td.MIN_BOTTOM_MM and P["lift_mm"] <= td.RING_LIFT_CAP_MM and P["lift_mm"] > 80
    base_free = m.rig._ik_free
    def low_ceiling(x, y, h, roll=None):
        if h > 60: raise ValueError("no tipped pose that high")
        return base_free(x, y, h, roll)
    m.rig._ik_free = low_ceiling
    with pytest.raises(SystemExit, match="ring bottom"): mover.plan(RING, (8.0, 6.5))


def test_set_down_spot_rules(td):
    markers = [(100.0, 80.0), (350.0, 220.0)]; pile = [RING, (RING[0] + 0.3, RING[1])]; others = [(6.0, 4.0), (12.0, 8.0)]
    for seed in range(40):
        p = td.sample_ring_spot(random.Random(seed), td.RING_BOX_IN, RING, pile, others, markers)
        if p is None: continue
        assert td.RING_BOX_IN[0] <= p[0] <= td.RING_BOX_IN[2] and td.RING_BOX_IN[1] <= p[1] <= td.RING_BOX_IN[3]
        assert all(td.dist_mm(p, q) >= 52.0 + 38.1 for q in pile + others), "1.5 in of clear floor between the ring's wall and the pile and the toys"
        assert all(np.hypot(p[0] * 25.4 - m[0], p[1] * 25.4 - m[1]) >= 52 + 35 for m in markers) and td.dist_mm(p, RING) >= 60.0
        assert min(p[0] * 25.4, p[1] * 25.4, 457.2 - p[0] * 25.4, 304.8 - p[1] * 25.4) >= 52 + 30
    assert any(td.sample_ring_spot(random.Random(s), td.RING_BOX_IN, RING, pile, [], markers) for s in range(10)), "some spot is found"
    assert td.sample_ring_spot(random.Random(0), td.RING_BOX_IN, RING, pile, [], markers, feasible=lambda p: False) is None, "an infeasible spot is never returned"
    assert td.sample_ring_spot(random.Random(0), (7.0, 4.5, 7.5, 5.0), RING, [(7.2, 4.7)], [], []) is None, "a box the pile fills has no spot"
    assert td.sample_ring_spot(random.Random(0), (0.2, 0.2, 0.6, 0.6), None, [], [], []) is None, "never within reach of the mat edge"


def ring_move_world(td, mk, hold=True, high_mm=0.0, drawn=None, cc=None, table=None):
    m = mk([toy(*RING), toy(RING[0] + 0.2, RING[1] + 0.1)], hold=hold); m.arm.high = high_mm; drawn = drawn or (7.6, 6.9)
    m.t.refit_ring = lambda: (m.rig.set_ring(drawn), drawn)[1]   # the top frame shows the ring a little off the commanded spot
    return m


def test_the_ring_lift_sets_it_down_and_refits_the_measured_position(td, mk, cc, table):
    m = ring_move_world(td, mk, drawn=(7.6, 6.9)); dest = (7.5, 7.0); r = td.RingMover(m.t).lift_ring(dest)
    assert r["ok"] and r["measured_in"] == [7.6, 6.9] and m.rig.ring == (7.6, 6.9) and r["err_mm"] == pytest.approx(td.dist_mm((7.6, 6.9), dest), abs=0.2), "the next round uses the measured centre, not the commanded one"
    assert m.w.names[:2] == ["hover", "pregrasp"] and m.w.names[-1] == "rest" and "carry" in m.w.names and m.w.names.index("lower") > m.w.names.index("carry")
    low = [x for x in m.w.rows if x["wp"] == "lower"][-1]["target"]; tip = m.t.pl.fk_mat(cc, table, low); assert tip[2] == pytest.approx(td.RING_GRIP_H_MM + 1.0, abs=1.5), "lowered to ~1 mm above the grip height"
    assert [x["gripper"] for x in m.arm.plans[-1]["waypoints"]][-3:] == [td.RING_JAW, td.RING_JAW, 0], "open, lift straight up, home"


def test_a_ring_that_is_not_gripped_is_not_carried(td, mk):
    m = ring_move_world(td, mk, hold=False); r = td.RingMover(m.t).lift_ring((7.5, 7.0))
    assert not r["ok"] and r["why"] == "ring not gripped" and "carry" not in m.w.names and m.w.names[-1] == "rest"


# ---------------------------------------------------------------- reset (b): spread the pile
def test_spread_spot_rules(td):
    markers = [(100.0, 80.0)]; others = [(6.0, 4.0)]; spots = []
    for seed in range(30):
        p = td.sample_spread_spot(random.Random(seed), td.SPREAD_BOX_IN, others + spots, markers, RING, (7.0, 5.0), avoid=[((10.0, 5.0), 67.0)])
        if p is None: continue
        assert all(td.dist_mm(p, q) >= 40.0 for q in others + spots) and td.dist_mm(p, RING) >= 52.0 + 40.0 and td.dist_mm(p, (10.0, 5.0)) >= 67.0 and td.dist_mm(p, (7.0, 5.0)) >= 60.0
        assert np.hypot(p[0] * 25.4 - 100.0, p[1] * 25.4 - 80.0) >= 50.0 and td.SPREAD_BOX_IN[0] <= p[0] <= td.SPREAD_BOX_IN[2]
        spots.append(p)
    assert len(spots) >= 4, "several toys fit"
    assert td.sample_spread_spot(random.Random(0), (8.9, 5.9, 9.1, 6.1), [], [], RING, None) is None, "inside the ring is never a spot"


def test_the_pile_split(td):
    import cv2
    m = np.zeros((60, 80), np.uint8); cv2.circle(m, (28, 30), 10, 255, -1); cv2.circle(m, (46, 30), 10, 255, -1)   # two touching toys, merged into one blob
    pieces = td.split_mask(m); assert len(pieces) == 2 and abs(pieces[0][0] - pieces[1][0]) > 8 and all(p[2] > 12 for p in pieces) and pieces[0][3] >= pieces[1][3], "most compact first"
    one = np.zeros((40, 40), np.uint8); cv2.circle(one, (20, 20), 9, 255, -1); assert len(td.split_mask(one)) == 1
    single = toy(8.0, 5.0); assert td.is_single(single) and td.pile_candidates(single) == [(8.0, 5.0)]
    blob = toy(8.0, 5.0, w=40, h=14, area=300, angle=0.0); assert not td.is_single(blob) and td.blob_count(blob) == 3
    ends = td.pile_candidates(blob); assert len(ends) >= 2 and all(td.dist_mm(e, (8.0, 5.0)) > 10 for e in ends) and max(abs(e[0] - 8.0) for e in ends) * 25.4 == pytest.approx(14.0, abs=0.6), "the long axis' ends, a toy radius in from the edge"
    assert td.pile_candidates(blob, mask_fn=lambda o: [(8.2, 5.1, 40.0, 0.9)]) == [(8.2, 5.1)], "the mask's best piece wins"
    assert td.pile_candidates(blob, mask_fn=lambda o: (_ for _ in ()).throw(ValueError("x"))) == ends, "a failing mask falls back to the geometry"


def test_nudge_sweeps_low_and_away_from_the_ring(td):
    calls = []
    wps = td.nudge_waypoints((8.0, 5.0), (10.0, 5.0), lambda x, y, h: calls.append((x, y, h)) or {"j": h}, 35)
    assert [w[0] for w in wps] == ["nudge_above", "nudge_in", "nudge_through", "nudge_up"] and calls[2][2] < 10 and calls[1][2] < 10 and calls[0][2] > 30
    assert calls[1][0] > calls[2][0] and calls[2][0] < 8.0, "the sweep runs away from the ring (the ring is at +x)"


def test_spread_picks_each_toy_to_a_clear_spot_and_stops_on_the_stop_file(td, mk):
    pile = (6.0, 5.0); m = mk([toy(pile[0], pile[1]), toy(pile[0] + 0.35, pile[1] + 0.1)], hold=True, ring=(12.0, 3.5), spread_rec=FakeRec())
    s = td.spread_pile(m.t, pile, td.SPREAD_BOX_IN, nudge=True)
    assert s["stop"] is None and s["moved"] == 2 and td.dist_mm(s["spots"][0], s["spots"][1]) >= 40.0 and all(td.dist_mm(p, pile) >= td.PILE_R_MM + 15 for p in s["spots"]) and "place" in m.w.names
    rec = m.t.eps["spread"].rec; assert [c[0] for c in rec.calls].count("commit") == 2 and rec.calls[0][1]["task"] == td.TASK_SPREAD
    m2 = mk([toy(pile[0], pile[1])], ring=(12.0, 3.5)); open(m2.stop, "w").write("x"); assert td.spread_pile(m2.t, pile, td.SPREAD_BOX_IN)["stop"] == "stop_file" and m2.w.names == []


def test_a_blob_that_fails_twice_is_nudged_once_then_picked(td, mk, monkeypatch):
    pile = (6.0, 5.0); m = mk([toy(*pile)], hold=False, ring=(12.0, 3.5)); nudges = []
    monkeypatch.setattr(td, "do_nudge", lambda t, blob: (nudges.append(blob["x_in"]), setattr(m.w, "hold", True))[0])
    s = td.spread_pile(m.t, pile, td.SPREAD_BOX_IN, nudge=True)
    assert len(nudges) == 1 and s["nudged"] == 1 and s["failed"] == 2 and s["moved"] == 1 and s["stop"] is None


# ---------------------------------------------------------------- the loop
def test_loop_sequence_and_stops(td, monkeypatch):
    seq = []
    class T:
        def __init__(s, rounds): s.rounds = list(rounds)
        def round(s, max_n=None): seq.append("tidy"); r = s.rounds.pop(0); return {"picked": 2, "ok": 2, "failed": 0, "stop": r}
    monkeypatch.setattr(td, "reset_mat", lambda t, dest=None: (seq.append("reset"), {"ok": True, "stop": None})[1])
    out = td.loop(T(["all_tidy"] * 3), 3); assert seq == ["tidy", "reset", "tidy", "reset", "tidy"] and out["stop"] is None and len(out["rounds"]) == 3, "tidy all -> reset -> tidy all, no reset after the last round"
    seq.clear(); out = td.loop(T(["all_tidy", "consecutive_failures"]), 3); assert seq == ["tidy", "reset", "tidy"] and out["stop"] == "round 2: consecutive_failures"
    seq.clear(); out = td.loop(T(["stop_file"]), 3); assert seq == ["tidy"] and "stop_file" in out["stop"]
    monkeypatch.setattr(td, "reset_mat", lambda t, dest=None: (seq.append("reset"), {"ok": False, "stop": "no free spot for the ring"})[1])
    seq.clear(); out = td.loop(T(["all_tidy", "all_tidy"]), 3); assert seq == ["tidy", "reset"] and "reset after round 1" in out["stop"]


def test_plan_report_lists_toys_verdicts_and_the_order(td, mk):
    m = mk([toy(*RING), toy(5.0, 3.5), toy(RING[0] + 2.0, RING[1]), toy(5.0, 3.0, 21, 8)]); r = td.plan_report(m.t.pl, m.rig, m.w.toys, RING, "clean")
    v = {tuple(x["toy_in"]): x for x in r["toys"]}
    assert v[(9.0, 6.0)]["verdict"] == "done" and v[(5.0, 3.5)]["verdict"] == "ok" and v[(11.0, 6.0)]["verdict"] == "blocked" and v[(5.0, 3.0)]["kind"] == "spacer" and v[(5.0, 3.5)]["route"] == "direct"
    assert r["n_outside"] == 3 and r["n_refused"] == 1 and [5.0, 3.5] in r["order"] and [11.0, 6.0] not in r["order"] and r["ring_in"] == [9.0, 6.0]
    near = td.plan_report(m.t.pl, m.rig, m.w.toys, RING, "near")["order"]; assert near[0] in ([6.0, 8.0], [5.0, 3.5])


# ---------------------------------------------------------------- the strict keep rule (11 Nov 21:32 pilot)
def test_the_pilot_false_positive_is_rejected(mk, pl, monkeypatch):
    """Missed grasp: the gripper read 0.54 after the lift, the camera swore in_jaws True (score 0.388, a false ring on the mat), and the spacer had been pushed off its spot: the old rule kept the episode."""
    rec = FakeRec(); m = mk([toy(5.0, 3.5, 21, 8)], hold=False, rec=rec)
    monkeypatch.setattr(pl, "jaw_check", lambda jpeg=None, window=None, out_path=None, min_score=None: {"in_jaws": True, "ring": {"uv": [340, 330], "r_px": 15, "score": 0.388}, "frame": None, "window": [], "note": None})
    orig = m.w.run_plan
    def push_it_off(plan, v):   # the closing jaws shove the toy off its spot
        rows = orig(plan, v)
        if any(r["wp"] == "grasp" for r in rows): m.w.toys[0]["x_in"], m.w.toys[0]["y_in"] = 4.0, 9.0
        return rows
    monkeypatch.setattr(m.t.pl, "run_plan", push_it_off)
    res = m.t.ring_pick(m.w.toys[0])
    assert res["held"] is False and res["spot_empty"] is True and res["placed"] is False and not res["keep"] and "not_held" in res["keep_reasons"] and "not_in_ring" in res["keep_reasons"]
    assert rec.calls[-1][0] == "commit" and rec.calls[-1][1] is False and res["demo"]["dataset"] == "rejects" and "hover_tgt" not in m.w.names, "rejected, no carry"
    g = res["grip"][-1]; assert g["gripper_after_lift"] == 0.0 and g["camera_in_jaws"] is True and g["camera_score"] == 0.388, "the readings are logged for tuning"
    teacher = rec.calls[0][1]["teacher"]; assert teacher["grip"] and teacher["interior"]["placed"] is False and teacher["interior"]["diff_px"] is not None, "and so are the interior counts, in the episode meta"
    r = m.t.track.runs[-1]; assert r.metrics_["gripper_after_lift"] == 0.0 and r.metrics_["camera_score"] == 0.388 and r.metrics_["placed"] == 0.0 and "interior_diff_px" in r.metrics_ and "interior_need_px" in r.metrics_


def test_the_ring_interior_diff_is_the_proof_of_placement(td, mk):
    rec = FakeRec(); m = mk([toy(5.0, 3.5)], hold=True, rec=rec); res = m.t.ring_pick(m.w.toys[0])
    iv = res["interior"]; assert res["placed"] and res["keep"] and iv["diff_px"] >= iv["need_px"] > 0 and iv["diff_px"] > 20, "a toy in the ring changes ~50 px of its interior; the threshold is 40 % of the toy's area in pixels"
    assert rec.calls[-1][1] is True and rec.calls[0][1]["task"].endswith("tape ring")
    # a toy that was already inside does not count for the next one; a frame with nothing new does not pass
    import cv2
    img = cv2.imdecode(np.frombuffer(m.w.frame(RING), np.uint8), cv2.IMREAD_GRAYSCALE); circ = (img.shape[1] / 2, img.shape[0] / 2, 60.0)
    assert td.interior_diff(img, img, circ) == 0 and not td.placed_verdict(img, img, circ, 88.0, 0.5)["placed"]
    b = img.copy(); cv2.circle(b, (int(circ[0]), int(circ[1])), 4, 0, -1); assert td.placed_verdict(img, b, circ, 88.0, 0.5)["placed"]
    c = img.copy(); cv2.circle(c, (int(circ[0] + 55), int(circ[1])), 4, 0, -1); assert not td.placed_verdict(img, c, circ, 88.0, 0.5)["placed"], "a change on the rim (outside 0.8 x r) is not a toy in the ring"


def test_a_wing_nut_may_pass_on_the_camera_but_still_has_to_be_placed(td, mk, pl, monkeypatch):
    m = mk([toy(5.0, 3.5, 20, 12)], hold=False, rec=FakeRec())   # wing nut: readback 0 (the thin wing), camera True
    monkeypatch.setattr(pl, "jaw_check", lambda jpeg=None, window=None, out_path=None, min_score=None: {"in_jaws": True, "ring": {"uv": [340, 330], "r_px": 15, "score": 0.7}, "frame": None, "window": [], "note": None})
    res = m.t.ring_pick(m.w.toys[0]); assert res["held"] is True and res["placed"] is False and not res["keep"] and "not_in_ring" in res["keep_reasons"]


# ---------------------------------------------------------------- overnight guards (tests/tools/test_overnight_guards.py has the thresholds)
def hot(mk, **kw):
    m = mk([toy(5.0, 3.5), toy(5.0, 8.0), toy(6.0, 4.5)], **kw); real = m.t.pl.run_plan; n = [0]
    def run(plan, max_deg_s):
        rows = real(plan, max_deg_s); n[0] += 1
        return [{**r, "tele": {"n": 3, "total_a_max": 1.0, "volt_min": 11.9, "flags": {}, "j": {"shoulder_lift": {"temp_max_c": 65, "load_abs_max_pct": 10.0, "cur_max_a": 0.5, "stall_s_max": 0.0}}}} for r in rows] if n[0] >= 2 else rows
    m.t.pl.run_plan = run; return m


def test_a_hot_servo_in_a_plan_row_stops_the_pick_and_sends_the_arm_home(mk):
    m = hot(mk); s = m.t.round()
    assert s["stop"] == "servo_hot" and s["picked"] == 1 and [o for o in m.out if o.get("event") == "trip"][0]["detail"]["temp_c"] == 65 and m.w.names[-1] == "rest" and m.t.in_motion is False


def test_max_tries_stops_the_round_cleanly_between_picks(td, mk):
    m = mk([toy(5.0, 3.5), toy(5.0, 8.0), toy(6.0, 4.5)]); m.guard.lim.max_tries = 2; s = m.t.round()
    assert s["stop"] == "max_tries" and s["picked"] == 2 and s["trip"]["clean"] is True
    assert td.loop(types.SimpleNamespace(round=lambda n=None: {"picked": 2, "ok": 2, "failed": 0, "stop": "max_tries"}), 3)["stop"] == "round 1: max_tries"


def test_spread_picks_count_as_tries(td, mk):
    pile = (6.0, 5.0); m = mk([toy(pile[0], pile[1]), toy(pile[0] + 0.35, pile[1] + 0.1)], hold=True, ring=(12.0, 3.5)); m.guard.lim.max_tries = 1
    s = td.spread_pile(m.t, pile, td.SPREAD_BOX_IN, nudge=False)
    assert s["stop"] == "max_tries" and s["moved"] == 1 and m.guard.tries == 1


def test_max_minutes_stops_between_picks(mk):
    m = mk([toy(5.0, 3.5), toy(5.0, 8.0)]); m.guard.lim.max_minutes = 1.0; m.guard.check_budget(m.t.ctx); m.w.clock.t += 61
    s = m.t.round(); assert s["stop"] == "max_minutes" and s["picked"] == 0 and m.w.names == []


def test_park_goes_to_rest_only_when_the_arm_is_out_of_rest_and_the_dashboard_reads_back(mk):
    m = mk([]); assert m.t.park("interrupted")["why"] == "already at rest" and m.w.names == []
    m.t.in_motion = True; r = m.t.park("exception"); assert r["parked"] is True and m.w.names[-1] == "rest" and m.t.in_motion is False
    m2 = mk([]); m2.t.in_motion = True; assert m2.t.park("run_plan")["parked"] is False and m2.w.names == [], "after so101-stop no further motion"
    m3 = mk([]); m3.t.in_motion = True; m3.t.ctx.fetch_state = lambda: (_ for _ in ()).throw(OSError("down")); assert m3.t.park("x")["parked"] is False and m3.w.names == []
    ev = [o for o in m.out if o.get("event") == "park"]; assert ev and ev[0]["reason"] == "interrupted"


def test_in_motion_follows_the_last_waypoint(mk):
    m = mk([toy(5.0, 3.5)]); m.t.round(max_n=1); assert m.t.in_motion is False, "a finished pick ends at rest"
    with m.t.guarded(): m.t.pl.run_plan({"waypoints": [dict(name="hover_nut", joints=dict(m.t.pl.REST), desired=dict(m.t.pl.REST), gripper=0, seconds=1.0, settle=0.1, gripper_only=False)]}, 50.0)
    assert m.t.in_motion is True

# ---------------------------------------------------------------- keep-outs: "be careful not to hurt the frogs" (TC, 11 Nov 2026)
def frog_frame(cc, table, ring_in, frogs_in):
    import cv2
    img = cv2.imdecode(np.frombuffer(synth_frame(cc, table, ring_in), np.uint8), cv2.IMREAD_COLOR)
    for f in frogs_in:
        u, v = cc.table_to_pixel(table, f[0] * 25.4, f[1] * 25.4); ppm = abs(cc.table_to_pixel(table, f[0] * 25.4 + 10, f[1] * 25.4)[0] - u) / 10.0
        cv2.circle(img, (int(round(u)), int(round(v))), int(28 * ppm), (200, 60, 40), -1)   # an all-blue frog: the worst case for the ring fit
    return img


def test_the_ring_fit_ignores_blue_frogs_and_a_lone_blue_blob_is_not_a_ring(td, cc, table):
    import cv2
    img = frog_frame(cc, table, (9.0, 6.0), [(13.0, 2.5), (11.0, 9.3)]); got = td.fit_ring(cv2.imencode(".png", img)[1].tobytes(), cc, table)
    assert got is not None and np.hypot(got[0] - 9.0, got[1] - 6.0) * 25.4 < 3.0, got   # 11 Nov 22:4x: the old all-blue fit landed 4 in off
    blank = np.full_like(img, 190); u, v = cc.table_to_pixel(table, 13.0 * 25.4, 2.5 * 25.4); ppm = abs(cc.table_to_pixel(table, 13.0 * 25.4 + 10, 2.5 * 25.4)[0] - u) / 10.0
    cv2.circle(blank, (int(u), int(v)), int(28 * ppm), (200, 60, 40), -1); assert td.fit_ring(cv2.imencode(".png", blank)[1].tobytes(), cc, table) is None


def test_keepouts_are_colourful_or_big_blobs_never_toys_or_pieces_of_the_ring(td, cc, table):
    import cv2
    ko = td.ko; ring = (9.0, 6.0); img = cv2.imdecode(np.frombuffer(World(cc, table, None, None, [], frogs=[(13.0, 2.5)]).frame(ring), np.uint8), cv2.IMREAD_COLOR)
    objs = [frog(13.0, 2.5), toy(7.0, 8.0), toy(12.0, 8.0, 50, 40, 1500), toy(9.0, 6.0 + 50 / 25.4, 30, 30, 620), toy(9.0, 6.0, 113, 106, 6100), dict(frog(9.0 - 45 / 25.4, 6.0))]
    k = ko.find_keepouts(objs, img, cc, table, ring); got = {(round(x["x_in"], 2), round(x["y_in"], 2)): x["why"] for x in k}
    assert got[(13.0, 2.5)] == "colourful" and got[(12.0, 8.0)] == "big" and (round(9.0 - 45 / 25.4, 2), 6.0) in got, got   # a frog against the ring's wall is still a keep-out
    assert (7.0, 8.0) not in got and (9.0, round(6.0 + 50 / 25.4, 2)) not in got and (9.0, 6.0) not in got, "a steel toy, a piece of the ring's wall and the ring itself are not keep-outs"
    assert all(x["r_mm"] > 0.5 * math.hypot(59, 57) and x["h_mm"] >= 51.0 for x in k if x["why"] == "colourful")


def test_the_obstacle_check_avoids_a_keepout_low_but_passes_over_it_high(td, pl, cc, table, mk):
    m = mk([]); r = m.rig; r.keepouts = [{"x_in": 6.0, "y_in": 4.0, "r_mm": 45.0, "h_mm": 60.0, "why": "colourful"}]
    low = r.above((6.0, 4.0), 0.4); hits = r.hits(low); assert hits and all(len(h) == 5 and h[4] == "keep-out" for h in hits if len(h) > 4) and any(len(h) > 4 for h in hits)
    assert not r.hits(r.ik_free(6.0, 4.0, 110.0)), "110 mm over a 60 mm keep-out is clear"
    assert not [h for h in r.hits(low, ring=False) if len(h) == 4], "ring=False checks keep-outs only"


def test_a_frog_is_never_picked_and_a_toy_beside_it_is_refused_while_a_far_one_is_tidied(mk):
    m = mk([toy(5.0, 3.5), toy(11.2, 3.3)], frogs=[(12.6, 3.0)]); s = m.t.round()
    picked_at = [m.w.at] if m.w.at else []; assert s["picked"] == 1 and s["ok"] == 1, s
    assert any(r["why"].startswith("route passes a keep-out") for r in s["refused"]), s["refused"]
    assert all(np.hypot(a[0] - 12.6, a[1] - 3.0) * 25.4 > 60 for a in picked_at), "the arm never went down at the frog"
    assert any(e.get("event") == "keepouts" and len(e["keepouts"]) == 1 for e in m.out)


def test_a_frog_inside_the_ring_stops_the_round_before_any_motion(mk):
    m = mk([toy(5.0, 3.5)], frogs=[(RING[0], RING[1])]); s = m.t.round(); assert s["stop"] == "keep-out inside the ring" and s["picked"] == 0 and m.w.names == []


def test_carries_and_spots_keep_clear_of_keepouts(td):
    ko = td.ko; k = [{"x_in": 8.0, "y_in": 6.0, "r_mm": 50.0, "h_mm": 60.0}]
    assert not ko.segment_clear((5.0, 6.0), (11.0, 6.0), k, 62.0) and ko.segment_clear((5.0, 1.0), (11.0, 1.0), k, 62.0) and ko.inside_ring(k, (8.2, 6.0)) and not ko.inside_ring(k, (12.0, 6.0))
    rng = random.Random(3); spots = [td.sample_spread_spot(rng, (5.5, 3.0, 12.5, 9.0), [], [], (3.0, 3.0), None, avoid=[((8.0, 6.0), 90.0)]) for _ in range(50)]
    assert all(s is None or np.hypot(s[0] - 8.0, s[1] - 6.0) * 25.4 >= 90.0 for s in spots) and any(s is not None for s in spots)
