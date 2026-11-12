"""tools/mat_detector.py with a fake model (no weights, no ultralytics): px box -> mat inches, label mapping, tape drop, graceful fallback."""
import importlib.util, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_s = importlib.util.spec_from_file_location("mat_detector_t", os.path.join(ROOT, "tools", "mat_detector.py")); md = importlib.util.module_from_spec(_s); _s.loader.exec_module(md)
_c = importlib.util.spec_from_file_location("cc_t", os.path.join(ROOT, "jetson", "cam_calib.py")); cc = importlib.util.module_from_spec(_c); _c.loader.exec_module(cc)
TABLE = md.load_table()


class _T:   # a torch-tensor stand-in
    def __init__(self, v): self.v = v
    def __getitem__(self, i): return self.v[i]
    def __iter__(self): return iter(self.v)


class _Box:
    def __init__(self, cls, conf, xyxy): self.cls, self.conf, self.xyxy = _T([cls]), _T([conf]), _T([xyxy])


class _Res:
    names = {0: "nut", 1: "wing_nut", 2: "spacer", 3: "ring", 4: "frog", 5: "toy"}
    def __init__(self, boxes): self.boxes = boxes


class FakeModel:
    def __init__(self, boxes): self.boxes, self.calls = boxes, []
    def predict(self, img, **kw): self.calls.append(kw); return [_Res(self.boxes)]


def det(boxes, **kw): return md.MatDetector(model=FakeModel(boxes), **kw).detect(np.zeros((360, 640, 3), np.uint8))


def test_pixel_box_becomes_mat_inches_like_tidy():
    d = det([_Box(0, 0.9, [300, 190, 312, 202])])[0]
    X, Y = cc.pixel_to_table(TABLE, 306, 196)
    assert d["label"] == "nut" and d["cls"] == "nut" and abs(d["x_in"] - X / 25.4) < 1e-9 and abs(d["y_in"] - Y / 25.4) < 1e-9 and d["box"] == [300, 190, 312, 202]


def test_label_mapping_matches_scene_labels_kinds():
    ds = det([_Box(i, 0.8, [100 + 40 * i, 100, 112 + 40 * i, 112]) for i in range(6)])
    assert {d["cls"]: d["label"] for d in ds} == {"nut": "nut", "wing_nut": "wing nut", "spacer": "spacer", "ring": "ring", "frog": "keep-out", "toy": "keep-out"}


def test_low_confidence_unknown_class_and_tape_zone_dropped():
    ds = det([_Box(0, 0.1, [100, 100, 112, 112]), _Box(1, 0.9, [300, 320, 330, 350]), _Box(2, 0.7, [200, 200, 212, 230])])
    assert [d["cls"] for d in ds] == ["spacer"]
    assert md.boxes_to_detections([("washer", 0.9, (1, 1, 5, 5))], cc, TABLE) == []


def test_sorted_by_score_and_conf_passed_to_model():
    m = FakeModel([_Box(0, 0.4, [100, 100, 112, 112]), _Box(2, 0.9, [200, 200, 212, 230])])
    ds = md.MatDetector(model=m, conf=0.3, imgsz=640).detect(np.zeros((360, 640, 3), np.uint8))
    assert [d["cls"] for d in ds] == ["spacer", "nut"] and m.calls[0]["conf"] == 0.3 and m.calls[0]["imgsz"] == 640


def test_no_weights_falls_back_to_no_detections(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(md.WEIGHTS_ENV, raising=False)
    d = md.MatDetector(weights=str(tmp_path / "missing.pt"))
    assert d.detect(np.zeros((360, 640, 3), np.uint8)) == [] and "no weights" in capsys.readouterr().err
    assert d.detect(np.zeros((360, 640, 3), np.uint8)) == []


def test_no_ultralytics_falls_back(tmp_path, capsys, monkeypatch):
    w = tmp_path / "best.pt"; w.write_bytes(b"x"); monkeypatch.setitem(sys.modules, "ultralytics", None)
    assert md.MatDetector(weights=str(w)).detect(np.zeros((360, 640, 3), np.uint8)) == [] and "ultralytics not importable" in capsys.readouterr().err


def test_inference_error_returns_empty(capsys):
    class Boom:
        def predict(self, *a, **k): raise RuntimeError("x")
    assert md.MatDetector(model=Boom()).detect(np.zeros((360, 640, 3), np.uint8)) == [] and "inference failed" in capsys.readouterr().err
    assert md.MatDetector(model=Boom()).detect(None) == []
