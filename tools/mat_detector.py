"""Mat object detector (fine-tuned YOLO11n on the top camera): top frame -> detections in mat inches, the `detections` input of tools/scene_labels.label_objects.

    from mat_detector import MatDetector
    det = MatDetector()                     # weights from $SO101_MAT_DETECTOR_WEIGHTS (never committed)
    dets = det.detect(img_bgr)              # [{label, x_in, y_in, score, box, cls}]
    labelled, ring = scene_labels.label_objects(objs, img_bgr, cc, table, detections=dets)

label: "nut" / "wing nut" / "spacer" / "ring" / "keep-out" (frog and toy both map to keep-out; unknown colourful things are keep-out, TC 11 Nov 2026).
cls is the raw detector class (nut, wing_nut, spacer, ring, frog, toy). box is [x0, y0, x1, y1] in 640x360 pixels. (x_in, y_in) = cam_calib.pixel_to_table of the box
centre, mm / 25.4 (the same call tools/tidy.py uses). No ultralytics, no weights or a failing model = no detections and one log line (callers fall back to the rules).
Inference only; training runs on Hugging Face (tools/mat_detector_train_job.py).
"""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
IN = 25.4
CLASSES = ("nut", "wing_nut", "spacer", "ring", "frog", "toy")
LABELS = {"nut": "nut", "wing_nut": "wing nut", "spacer": "spacer", "ring": "ring", "frog": "keep-out", "toy": "keep-out"}
WEIGHTS_ENV = "SO101_MAT_DETECTOR_WEIGHTS"
DEFAULT_CONF = 0.25
TAPE = (255, 296, 362, 360)   # the camera's own blue tape at the bottom edge (px x0, y0, x1, y1): detections centred in it are dropped


def log(*a): print("[mat_detector]", *a, file=sys.stderr, flush=True)


def _load_cc():
    spec = importlib.util.spec_from_file_location("cam_calib_for_mat_detector", os.path.join(ROOT, "jetson", "cam_calib.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load_table(path=None):
    return json.load(open(path or os.path.join(ROOT, "jetson", "calib", "top_table.json")))


def boxes_to_detections(boxes, cc, table, min_conf=DEFAULT_CONF):
    """boxes: [(cls_name, score, (x0, y0, x1, y1))] in frame pixels -> detections in mat inches, best score first. Unknown classes and the tape zone are dropped."""
    out = []
    for name, score, b in boxes:
        if name not in LABELS or float(score) < min_conf: continue
        x0, y0, x1, y1 = (float(v) for v in b); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if TAPE[0] <= cx <= TAPE[2] and cy >= TAPE[1]: continue
        try: X, Y = cc.pixel_to_table(table, cx, cy)
        except Exception: continue
        out.append({"label": LABELS[name], "cls": name, "x_in": X / IN, "y_in": Y / IN, "score": float(score), "box": [x0, y0, x1, y1]})
    return sorted(out, key=lambda d: -d["score"])


class MatDetector:
    """model: an object with .predict(img, conf=, imgsz=, verbose=) like ultralytics.YOLO (tests pass a fake); None = load the weights lazily."""
    def __init__(self, weights=None, conf=DEFAULT_CONF, imgsz=960, table=None, cc=None, model=None, device=None):
        self.weights = weights or os.environ.get(WEIGHTS_ENV); self.conf, self.imgsz, self.device = conf, imgsz, device
        self._table, self._cc, self._model, self._failed = table, cc, model, False

    def _ready(self):
        if self._model is not None: return True
        if self._failed: return False
        if not self.weights or not os.path.exists(self.weights):
            self._failed = True; log(f"no weights (set {WEIGHTS_ENV} or pass weights=; got {self.weights!r}): no detections, callers use the rules"); return False
        try: from ultralytics import YOLO
        except Exception as e:
            self._failed = True; log(f"ultralytics not importable ({type(e).__name__}): no detections, callers use the rules"); return False
        try: self._model = YOLO(self.weights)
        except Exception as e:
            self._failed = True; log(f"could not load {self.weights} ({e}): no detections"); return False
        return True

    def detect(self, img_bgr):
        if img_bgr is None or not self._ready(): return []
        try:
            kw = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
            if self.device: kw["device"] = self.device
            res = self._model.predict(img_bgr, **kw)[0]; names = res.names; boxes = []
            for b in res.boxes: boxes.append((names[int(b.cls[0])], float(b.conf[0]), [float(v) for v in b.xyxy[0]]))
            if self._cc is None: self._cc = _load_cc()
            if self._table is None: self._table = load_table()
            return boxes_to_detections(boxes, self._cc, self._table, self.conf)
        except Exception as e:
            log(f"inference failed ({type(e).__name__}: {e}): no detections"); return []


if __name__ == "__main__":   # python tools/mat_detector.py frame.jpg [weights.pt]
    import cv2
    img = cv2.imread(sys.argv[1]); d = MatDetector(sys.argv[2] if len(sys.argv) > 2 else None).detect(img)
    print(json.dumps(d, indent=1))
