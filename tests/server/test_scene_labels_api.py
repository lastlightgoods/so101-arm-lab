"""/api/scene with what each blob is: kind per object, the ring, the keep-outs (tools/scene_labels.py), computed in scene_step from the top frame the server already
holds, cached and rate-limited. Same hand-driven fixtures as test_scene_api (real 9 Nov frames; a toy frog is painted on the one with the hex nut)."""
import time
import pytest

cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
from test_scene_api import lab, img, IN, NUT_IN  # noqa: E402,F401  (lab: the fixture; importing registers it here)


def with_frog(srv, lab, px=(420, 280), r_mm=28.0):
    """obj.jpg (one hex nut) plus an orange toy frog with a blue leg at image pixel `px` (clear of the arm and its mask)."""
    cc = srv.cam_calib(); im = lab.frames["obj"].copy(); u, v = px; X, Y = cc.pixel_to_table(lab.table, u, v)
    ppm = abs(cc.table_to_pixel(lab.table, X + 10, Y)[0] - u) / 10.0
    cv2.circle(im, (int(round(u)), int(round(v))), int(r_mm * ppm), (30, 120, 240), -1); cv2.circle(im, (int(round(u + 20 * ppm)), int(round(v))), int(8 * ppm), (200, 60, 40), -1)
    return im


def settle(srv, n=None, now=None):
    for _ in range(n or srv.SCENE_MIN_HITS):
        now = (now or time.time()) + srv.SCENE_LABEL_EVERY_S + 0.05; srv.cam_frames["top"] = (srv.cam_frames["top"][0], now); srv.scene_step(now)
    return now


def test_scene_payload_carries_kinds_the_ring_and_the_keepouts(srv, http, lab):
    lab.learn_bg(http); lab.frame(with_frog(srv, lab)); settle(srv)
    s, _ = http.get_json("/api/scene"); assert s["ok"] is True
    kinds = {o["id"]: o for o in s["objects"]}; assert len(kinds) == 2, s["objects"]
    nut = next(o for o in s["objects"] if abs(o["x_in"] - NUT_IN[0]) * IN < 6); frog = next(o for o in s["objects"] if o is not nut)
    assert nut["kind"] in ("nut", "wing nut", "spacer", "toy?") and nut["keepout"] is None and nut["label_src"] == "rules"
    assert frog["kind"] == "keep-out" and frog["keepout"]["why"] == "colourful" and frog["keepout"]["h_mm"] == 60.0 and frog["keepout"]["colour_frac"] >= 0.10
    assert [{k: v for k, v in k.items() if k in ("id", "r_mm", "h_mm", "why")} for k in s["keepouts"]] == [{"id": frog["id"], "r_mm": frog["keepout"]["r_mm"], "h_mm": 60.0, "why": "colourful"}]
    assert abs(s["keepouts"][0]["x_in"] - frog["x_in"]) < 1e-6 and abs(s["keepouts"][0]["y_in"] - frog["y_in"]) < 1e-6
    assert s["ring"] is None, "no tape ring in this frame"
    assert {"id", "x_mm", "y_mm", "x_in", "y_in", "w_mm", "h_mm", "angle_deg", "area_mm2", "color", "px", "bbox_mm"} <= set(nut), "the scene map's own fields are all still there"


def test_a_ring_in_the_frame_is_served_at_its_fitted_centre(srv, http, lab):
    cc = srv.cam_calib(); lab.learn_bg(http); im = lab.frames["obj"].copy(); c = (7.0, 8.0)
    u, v = cc.table_to_pixel(lab.table, c[0] * IN, c[1] * IN); ppm = abs(cc.table_to_pixel(lab.table, c[0] * IN + 10, c[1] * IN)[0] - u) / 10.0
    cv2.circle(im, (int(round(u)), int(round(v))), int(46 * ppm), (200, 60, 40), 5); lab.frame(im); settle(srv)
    s, _ = http.get_json("/api/scene"); r = s["ring"]
    assert r is not None and r["r_mm"] == 52.0 and r["h_mm"] == 51.0 and ((r["x_in"] - c[0]) ** 2 + (r["y_in"] - c[1]) ** 2) ** 0.5 * IN < 4.0, r
    assert all(o["kind"] != "keep-out" for o in s["objects"] if o["kind"] in ("ring", "ring wall"))


def test_labels_are_cached_between_frames_and_rate_limited(srv, http, lab, monkeypatch):
    lab.learn_bg(http); lab.frame(with_frog(srv, lab)); now = settle(srv); runs0 = srv.SCENE["label_runs"]
    real = srv.scene_labels_mod().label_objects; calls = []
    monkeypatch.setattr(srv.scene_labels_mod(), "label_objects", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for dt in (0.05, 0.1, 0.2, 0.3, 0.4):                      # five detects inside 0.45 s: no new run
        srv.cam_frames["top"] = (srv.cam_frames["top"][0], now + dt); srv.scene_step(now + dt)
    assert calls == [] and srv.SCENE["label_runs"] == runs0
    first = http.get_json("/api/scene")[0]
    for _ in range(10): http.get_json("/api/scene")
    assert calls == [], "/api/scene itself never runs the labeller"
    monkeypatch.setattr(srv.scene_labels_mod(), "label_objects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert http.get_json("/api/scene")[0]["objects"] == first["objects"], "served from the cache"
    monkeypatch.setattr(srv.scene_labels_mod(), "label_objects", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    t = now + srv.SCENE_LABEL_EVERY_S + 0.01; srv.cam_frames["top"] = (srv.cam_frames["top"][0], t); srv.scene_step(t)
    assert calls == [1] and srv.SCENE["label_runs"] == runs0 + 1, "one run once the interval has passed"
    assert 0 < srv.SCENE["label_ms"] < 500


def test_a_labeller_failure_leaves_the_scene_working_and_unlabelled(srv, http, lab, monkeypatch, logs):
    lab.learn_bg(http); lab.frame(with_frog(srv, lab))
    monkeypatch.setattr(srv.scene_labels_mod(), "label_objects", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    settle(srv); s, _ = http.get_json("/api/scene")
    assert s["ok"] is True and len(s["objects"]) == 2 and all(o["kind"] is None and o["keepout"] is None for o in s["objects"]) and s["ring"] is None and s["keepouts"] == []
    assert srv.SCENE["labels_err"] == "ValueError: boom"


def test_an_object_newer_than_the_cached_labels_has_no_kind_yet(srv, http, lab):
    lab.learn_bg(http); lab.frame(lab.frames["bg"]); now = settle(srv)
    lab.frame(lab.frames["obj"])
    for i in range(srv.SCENE_MIN_HITS): srv.cam_frames["top"] = (srv.cam_frames["top"][0], now + (i + 1) * 0.01); srv.scene_step(now + (i + 1) * 0.01)   # all inside the label interval of the empty map
    s, _ = http.get_json("/api/scene"); assert len(s["objects"]) == 1 and s["objects"][0]["kind"] is None
    settle(srv, 1, now=now + 0.1); s, _ = http.get_json("/api/scene"); assert s["objects"][0]["kind"] in ("nut", "wing nut", "spacer", "toy?")


def test_scene_labels_move_nothing_and_reach_no_network(srv, http, lab, fake):
    lab.learn_bg(http); lab.frame(with_frog(srv, lab)); settle(srv); http.get_json("/api/scene")
    assert fake.calls == [] and fake.popens == []
