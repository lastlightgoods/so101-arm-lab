"""tools/scene_labels.py: what each mat-map blob is (nut / wing nut / spacer / toy? / ring / ring wall / keep-out), on synthetic top frames like test_tidy's
(a light mat, the blue ring, orange-with-blue-legs frogs, dark steel toys). No camera, no network. The module must stay light: the dashboard loads it, so it
may not pull in tidy or pick_loop."""
import math, os, subprocess, sys
import numpy as np, pytest
from pickhelpers import load, TOOLS
from test_tidy import synth_frame, toy, frog

IN = 25.4
RING = (9.0, 6.0)


@pytest.fixture(scope="module")
def sl(): return load(os.path.join(TOOLS, "scene_labels.py"), "scene_labels_under_test")


def ppm_at(cc, table, x, y):
    u, v = cc.table_to_pixel(table, x * IN, y * IN); return u, v, abs(cc.table_to_pixel(table, x * IN + 10, y * IN)[0] - u) / 10.0


def frame(cc, table, ring=RING, frogs=(), toys=()):
    """The synthetic top frame: ring (blue band), orange frogs with a blue leg, dark discs for toys."""
    import cv2
    img = cv2.imdecode(np.frombuffer(synth_frame(cc, table, ring), np.uint8), cv2.IMREAD_COLOR) if ring is not None else np.full((table.get("image_size") or [640, 360])[::-1] + [3], 190, np.uint8)
    for f in frogs:
        u, v, p = ppm_at(cc, table, *f); cv2.circle(img, (int(round(u)), int(round(v))), int(28 * p), (30, 120, 240), -1); cv2.circle(img, (int(round(u)) + int(20 * p), int(round(v))), int(8 * p), (200, 60, 40), -1)
    for t in toys:
        u, v, _ = ppm_at(cc, table, *t); cv2.circle(img, (int(round(u)), int(round(v))), 4, (40, 40, 40), -1)
    return img


def by_xy(objs): return {(round(o["x_in"], 2), round(o["y_in"], 2)): o for o in objs}


def scene_objs():
    """A frog, a nut, a wing nut, a spacer, an odd toy-sized thing, a piece of the ring's wall and the ring blob itself; ids as the scene map would."""
    objs = [frog(13.0, 2.5), toy(7.0, 8.0), toy(11.0, 9.0, 20, 12, 190), toy(5.0, 3.0, 22, 8, 100), toy(12.0, 4.0, 21, 10, 170),
            toy(RING[0], RING[1] + 40 / IN, 30, 30, 620), toy(RING[0], RING[1], 113, 106, 6100)]
    for i, o in enumerate(objs): o["id"] = i + 1
    return objs


def test_every_blob_gets_its_kind_and_the_frog_is_a_keepout(sl, cc, table):
    img = frame(cc, table, frogs=[(13.0, 2.5)], toys=[(7.0, 8.0)])
    out, ring = sl.label_objects(scene_objs(), img, cc, table); got = by_xy(out)
    assert ring is not None and math.hypot(ring["x_in"] - RING[0], ring["y_in"] - RING[1]) * IN < 3.0 and ring["r_mm"] == 52.0 and ring["h_mm"] == 51.0
    assert got[(13.0, 2.5)]["kind"] == "keep-out"
    ko = got[(13.0, 2.5)]["keepout"]; assert ko["why"] == "colourful" and ko["h_mm"] == 60.0 and ko["r_mm"] > 0.5 * math.hypot(59, 57) and ko["colour_frac"] >= 0.10
    assert got[(7.0, 8.0)]["kind"] == "nut" and got[(11.0, 9.0)]["kind"] == "wing nut" and got[(5.0, 3.0)]["kind"] == "spacer"
    assert got[(12.0, 4.0)]["kind"] == "spacer", "21 x 10 mm: ratio 2.1"
    assert got[(RING[0], round(RING[1] + 40 / IN, 2))]["kind"] == "ring wall" and got[(RING[0], RING[1])]["kind"] == "ring"
    assert all(o["label_src"] == "rules" and o["kind"] in sl.KINDS for o in out)
    assert all(o["keepout"] is None for o in out if o["kind"] != "keep-out"), "only keep-outs carry a keep-out"
    assert [o["id"] for o in out] == list(range(1, 8)), "order and ids kept; the inputs are not mutated"
    assert "kind" not in scene_objs()[0]


def test_a_toy_sized_blob_that_fits_no_kind_is_toy_question_mark(sl, cc, table):
    img = frame(cc, table); objs = [toy(7.0, 8.0, 24, 12, 228), toy(8.0, 3.0, 40, 30, 600)]
    for i, o in enumerate(objs): o["id"] = i
    out, _ = sl.label_objects(objs, img, cc, table); assert [o["kind"] for o in out] == ["toy?", "toy?"], "24 x 12 mm (live 11 Nov 2026) is longer than a nut, the toy rules say unknown"


def test_a_big_steel_blob_is_a_keepout_and_a_frog_on_the_ring_wall_still_is(sl, cc, table):
    img = frame(cc, table, frogs=[(RING[0] - 45 / IN, RING[1])]); big = toy(12.0, 8.0, 50, 40, 1500); wall_frog = frog(RING[0] - 45 / IN, RING[1])
    out, _ = sl.label_objects([big, wall_frog], img, cc, table)
    assert out[0]["kind"] == "keep-out" and out[0]["keepout"]["why"] == "big" and out[1]["kind"] == "keep-out" and out[1]["keepout"]["why"] == "colourful"


def test_no_frame_means_size_rules_only_and_no_ring(sl, cc, table):
    out, ring = sl.label_objects(scene_objs()[:4], None, cc, table); assert ring is None
    assert [o["kind"] for o in out] == ["keep-out", "nut", "wing nut", "spacer"] and out[0]["keepout"]["why"] == "big" and out[0]["keepout"]["colour_frac"] is None
    assert sl.label_objects([], None, cc, table) == ([], None) and sl.label_objects(None, None, cc, table) == ([], None)
    assert sl.label_objects([{"id": 1, "x_mm": 3}], None, cc, table)[0] == [], "a blob without a mat position is skipped, not a crash"


def test_a_frame_with_no_ring_gives_ring_none_and_no_ring_labels(sl, cc, table):
    img = frame(cc, table, ring=None); out, ring = sl.label_objects([toy(9.0, 6.0 + 40 / IN, 30, 30, 620)], img, cc, table)
    assert ring is None and out[0]["kind"] == "toy?"


def test_a_detection_within_15_mm_sets_the_kind_and_the_source(sl, cc, table):
    img = frame(cc, table, frogs=[(13.0, 2.5)], toys=[(7.0, 8.0)]); objs = scene_objs()
    dets = [{"label": "wing nut", "x_in": 7.0 + 10 / IN, "y_in": 8.0, "score": 0.91},      # 10 mm from the "nut": the detector wins
            {"label": "nut", "x_in": 11.0 + 16 / IN, "y_in": 9.0, "score": 0.99},           # 16 mm from the wing nut: out of range, rules stay
            {"label": "frog", "x_in": 5.0, "y_in": 3.0, "score": 0.8},                       # the spacer is called a frog: becomes a keep-out (why: detector)
            {"label": "nut", "x_in": 13.0, "y_in": 2.5, "score": 0.6},                       # the frog is called a nut: the detector wins on kind, the keep-out goes
            {"label": "banana", "x_in": 12.0, "y_in": 4.0, "score": 0.9}]                    # unknown label: ignored
    got = by_xy(sl.label_objects(objs, img, cc, table, detections=dets)[0])
    n = got[(7.0, 8.0)]; assert n["kind"] == "wing nut" and n["label_src"] == "detector" and n["score"] == 0.91
    w = got[(11.0, 9.0)]; assert w["kind"] == "wing nut" and w["label_src"] == "rules"
    s = got[(5.0, 3.0)]; assert s["kind"] == "keep-out" and s["label_src"] == "detector" and s["keepout"]["why"] == "detector" and s["keepout"]["h_mm"] == 60.0 and s["keepout"]["r_mm"] > 20.0
    f = got[(13.0, 2.5)]; assert f["kind"] == "nut" and f["label_src"] == "detector" and f["keepout"] is None
    assert got[(12.0, 4.0)]["label_src"] == "rules"
    assert sl.label_objects(objs, img, cc, table, detections=[])[0] == sl.label_objects(objs, img, cc, table)[0], "no detections = the rules"
    out = sl.label_objects(objs[:2], img, cc, table, detections=[{"label": "nut"}, {"label": "nut", "x_in": "x", "y_in": 1}])[0]; assert all(o["label_src"] == "rules" for o in out), "malformed detections are skipped"


def test_the_nearest_detection_wins(sl, cc, table):
    img = frame(cc, table); o = toy(7.0, 8.0); o["id"] = 1
    dets = [{"label": "spacer", "x_in": 7.0 + 12 / IN, "y_in": 8.0}, {"label": "wing nut", "x_in": 7.0 + 3 / IN, "y_in": 8.0}]
    assert sl.label_objects([o], img, cc, table, detections=dets)[0][0]["kind"] == "wing nut"


def test_the_shared_module_is_light_and_tidy_still_exports_the_same_functions(sl):
    code = "import sys, importlib.util; s = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(sorted(k for k in sys.modules if 'tidy' in k or 'pick_loop' in k))"
    r = subprocess.run([sys.executable, "-I", "-c", code, os.path.join(TOOLS, "scene_labels.py")], capture_output=True, text=True, timeout=60); assert r.returncode == 0, r.stderr; assert r.stdout.strip() == "[]"
    td = load(os.path.join(TOOLS, "tidy.py"), "tidy_for_scene_labels_test")
    for n in ("fit_circle", "blue_pixels", "ring_band", "fit_ring", "toy_kind", "on_ring_wall"): assert getattr(td, n) is getattr(td.sl, n), n
    assert (td.RING_R_MM, td.RING_H_MM) == (sl.RING_R_MM, sl.RING_H_MM) == (52.0, 51.0)
