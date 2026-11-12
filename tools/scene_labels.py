"""What each mat-map blob IS: light, shared by the dashboard (server.py) and tools/tidy.py (11 Nov 2026, "frogs and better object recognition in the display").

Only numpy (cv2 lazily) and tools/keepout.py; no pick_loop, so the dashboard can load it. Rules only for now (label_src "rules"); a future detector plugs in through
label_objects(detections=[{label, x_in, y_in, score}]), which sets the kind of the blob nearest each detection (within DET_MATCH_MM) and marks it label_src "detector".

Kinds: "nut" / "wing nut" / "spacer" (toy-sized, by size and aspect), "toy?" (toy-sized but unclear), "ring" (the tape ring's own blob), "ring wall" (a piece of its wall),
"keep-out" (colourful or big, tools/keepout.py: the toy frogs).
"""
import importlib.util, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN = 25.4
RING_R_MM, RING_H_MM = 52.0, 51.0                  # outer radius; 2 in tape (TC, 11 Nov 2026)
RING_WALL_IN_MM, RING_WALL_OUT_MM = 20.0, 10.0     # a blob centred this far inside / outside the ring's outer radius is the ring's own wall, not a toy
TOY_AREA_MM2, TOY_LONG_MM = 90.0, 22.0             # a nut reads ~12 x 10 mm, area ~88 (docs/mat-scene.md)
KIND_LONG_MAX_MM = 30.0   # longest blob toy_kind still names: lying spacers read 22-24 mm and wing nuts 24 mm (12 Nov 00:1x: at the old 22 mm cut both supervised picks got "unknown", a 4 mm aim and no wrist turn, and pushed the spacers instead of gripping them)
BLUE_ROWS = 5 / 6                                  # the bottom sixth of the frame holds the camera's own blue tape
DET_MATCH_MM = 15.0
KINDS = ("nut", "wing nut", "spacer", "toy?", "ring", "ring wall", "keep-out")
_DET_ALIASES = {"frog": "keep-out", "keepout": "keep-out", "keep out": "keep-out", "wingnut": "wing nut", "wing_nut": "wing nut", "hex nut": "nut", "washer": "spacer"}

_ko = []
def ko():
    """tools/keepout.py, loaded by path once (the dashboard has no tools/ on its import path)."""
    if not _ko:
        spec = importlib.util.spec_from_file_location("keepout_for_scene_labels", os.path.join(HERE, "keepout.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _ko.append(m)
    return _ko[0]


def dist_mm(a, b): return math.hypot(a[0] - b[0], a[1] - b[1]) * IN


# ---------------------------------------------------------------- the ring in the top frame
def fit_circle(xs, ys):
    """Least-squares circle through points -> (cx, cy, r)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float); A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]; s = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)[0]
    return float(s[0]), float(s[1]), float(math.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0.0)))


def blue_pixels(img):
    b, g, r = [img[:, :, i].astype(int) for i in range(3)]; m = ((b - r) > 60) & ((b - g) > 25) & (b > 90); m[int(img.shape[0] * BLUE_ROWS):, :] = False; return np.nonzero(m)


RING_BAND_R_MM = 40.5   # the blue band's image radius in mat mm: free fits on frames with the most band 39.4-42.0 (11-12 Nov 2026)
RING_ARC_FULL_DEG = 200.0   # a band covering less of the circle is fitted at RING_BAND_R_MM (the top camera sees ~100-110 deg of it)


def arc_span_deg(xs, ys, cx, cy):
    """How much of the circle around (cx, cy) the points cover, in degrees (360 minus the largest angular gap)."""
    if len(xs) < 3: return 0.0
    a = np.sort(np.degrees(np.arctan2(np.asarray(ys, float) - cy, np.asarray(xs, float) - cx))); return float(360.0 - np.max(np.diff(np.r_[a, a[0] + 360.0])))


def fit_circle_r(xs, ys, r, cx, cy, iters=25):
    """Centre of the circle of KNOWN radius r through the points (Gauss-Newton on the geometric distance), started at (cx, cy) on the arc's concave side."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for _ in range(iters):
        dx, dy = cx - xs, cy - ys; d = np.maximum(np.hypot(dx, dy), 1e-9); res = d - r; J = np.c_[dx / d, dy / d]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]; cx, cy = cx + step[0], cy + step[1]
        if math.hypot(*step) < 1e-3: break
    return float(cx), float(cy)


def ring_band(img, cc, table, min_px=150, r_mm=(30.0, 65.0), band_r_mm=RING_BAND_R_MM):
    """(ys, xs, (cx, cy, r_px)) of the ring's blue band: the largest connected blue region whose pixels lie on a ring-sized circle (an annulus, not a filled blob).
    The top camera sees only ~100-110 deg of the band, and a free circle through so short an arc drifts (12 Nov 00:5x: radius 31-37 mm and the centre 0.2 in off
    between frames under dim light). So a band covering less than RING_ARC_FULL_DEG is fitted at the measured radius band_r_mm; a fuller band keeps the free fit.
    Other blue things on the mat (the toy frogs) are separate regions and fail the annulus test. None when none passes."""
    b, g, r = [img[:, :, i].astype(int) for i in range(3)]; m = ((b - r) > 60) & ((b - g) > 25) & (b > 90); m[int(img.shape[0] * BLUE_ROWS):, :] = False
    for ys, xs in ko().blue_components(m, min_px):
        cx, cy, rp = fit_circle(xs, ys)
        if rp <= 0: continue
        if arc_span_deg(xs, ys, cx, cy) < RING_ARC_FULL_DEG:   # a short arc: known radius, centre only
            u0, v0 = cc.table_to_pixel(table, *cc.pixel_to_table(table, cx, cy)); ppm = abs(cc.table_to_pixel(table, *(np.add(cc.pixel_to_table(table, cx, cy), (10.0, 0.0))))[0] - u0) / 10.0
            rp = band_r_mm * ppm; cx, cy = fit_circle_r(xs, ys, rp, cx, cy)
        X, Y = cc.pixel_to_table(table, cx, cy); X2, Y2 = cc.pixel_to_table(table, cx + rp, cy); rmm = math.hypot(X2 - X, Y2 - Y)
        if r_mm[0] <= rmm <= r_mm[1] and float(np.median(np.abs(np.hypot(xs - cx, ys - cy) - rp))) < 0.2 * rp: return ys, xs, (cx, cy, rp)
    return None


def fit_ring_multi(fetch, cc, table, n=5, min_ok=3, min_px=100):
    """The ring's centre (x_in, y_in) as the median of up to n single-frame fits (fetch() -> a top frame), or None with fewer than min_ok good fits.
    why: 12 Nov 2026 the band was 200-227 px, near the old 150 px floor, so a frame now and then fell short and stopped a reset; single fits jittered 4 mm."""
    fits = [f for f in (fit_ring(fetch(), cc, table, min_px) for _ in range(n)) if f is not None]
    if len(fits) < min_ok: return None
    return (float(np.median([f[0] for f in fits])), float(np.median([f[1] for f in fits])))


def fit_ring(jpeg, cc, table, min_px=150, r_mm=(30.0, 65.0)):
    """(x_in, y_in) of the ring's centre from one top frame (ring_band), pixel -> mat. None when no ring-shaped blue band is seen.
    why: 11 Nov 22:4x two blue toy frogs pulled the old all-blue-pixels fit 4 in off the ring."""
    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if isinstance(jpeg, (bytes, bytearray)) else jpeg
    band = ring_band(img, cc, table, min_px, r_mm)
    if band is None: return None
    X, Y = cc.pixel_to_table(table, band[2][0], band[2][1]); return (X / IN, Y / IN)


# ---------------------------------------------------------------- the toys
def toy_kind(o):
    """(kind, aim_mm) from the scene map's size and aspect: nut (round, ~10-12 mm), wing nut (longer, ratio 1.3-1.8), spacer (a standing ~6 mm one or a lying cylinder, ratio >= 1.8), unknown (too big)."""
    w, h = float(o.get("w_mm") or 0), float(o.get("h_mm") or 0); long_, short = max(w, h), max(min(w, h), 1e-6)
    if long_ > KIND_LONG_MAX_MM or long_ <= 0: return "unknown", 4.0
    if long_ <= 8.0 or long_ / short >= 1.8: return "spacer", 5.0
    if long_ / short >= 1.3: return "wing nut", 4.0
    return "nut", 4.0


def on_ring_wall(o, ring, inner_mm=RING_WALL_IN_MM, outer_mm=RING_WALL_OUT_MM):   # why: 11 Nov 22:0x the scene map cut the standing ring into 5 toy-sized blobs 35-54 mm from its fitted centre; tidy would have tried to pick them
    return RING_R_MM - inner_mm < dist_mm((o["x_in"], o["y_in"]), ring) < RING_R_MM + outer_mm


def find_keepouts(objs, img_bgr, cc, table, ring=None):
    """tools/keepout.find_keepouts with the ring constants of this module."""
    return ko().find_keepouts(objs, img_bgr, cc, table, ring, RING_R_MM, RING_WALL_IN_MM, RING_WALL_OUT_MM)


# ---------------------------------------------------------------- the labels
def _match(o, detections, max_mm=DET_MATCH_MM):
    """The detection nearest to the blob's centre within max_mm, or None."""
    best, bd = None, max_mm
    for d in detections or ():
        try: dd = dist_mm((o["x_in"], o["y_in"]), (float(d["x_in"]), float(d["y_in"])))
        except (KeyError, TypeError, ValueError): continue
        if dd <= bd: best, bd = d, dd
    return best


def _kind_of(label):
    k = _DET_ALIASES.get(str(label).strip().lower(), str(label).strip().lower()); return k if k in KINDS else None


def label_objects(objs, img_bgr, cc, table, detections=None):
    """(labelled objs, ring). Each returned object is a copy of the scene object plus "kind" (one of KINDS), "keepout" ({r_mm, h_mm, why, colour_frac} or None) and
    "label_src" ("rules", or "detector" when a detection within DET_MATCH_MM of its centre set the kind). ring is {x_in, y_in, r_mm, h_mm} or None (no ring-shaped blue
    band, or no frame). img_bgr None = size rules only (no ring, no colour test). A detection labelled keep-out/frog makes a keep-out even where the rules saw none (why
    "detector"); a detection of any other kind removes the rules' keep-out (the detector wins on kind)."""
    objs = [o for o in (objs or ()) if isinstance(o, dict) and "x_in" in o and "y_in" in o]
    xy = None
    if img_bgr is not None:
        try: xy = fit_ring(img_bgr, cc, table)
        except Exception: xy = None
    ring = {"x_in": float(xy[0]), "y_in": float(xy[1]), "r_mm": RING_R_MM, "h_mm": RING_H_MM} if xy is not None else None
    kos = ko().find_keepouts(objs, img_bgr, cc, table, xy, RING_R_MM, RING_WALL_IN_MM, RING_WALL_OUT_MM)
    out = []
    for o in objs:
        o = dict(o); k = next((k for k in kos if dist_mm((o["x_in"], o["y_in"]), (k["x_in"], k["y_in"])) < 1.0), None)
        w, h = float(o.get("w_mm") or 0), float(o.get("h_mm") or 0); d = dist_mm((o["x_in"], o["y_in"]), xy) if xy is not None else None
        if xy is not None and max(w, h) >= ko().RING_BLOB_MM and d < ko().RING_CENTRE_MM: kind, k = "ring", None
        elif k is not None: kind = "keep-out"
        elif xy is not None and on_ring_wall(o, xy): kind = "ring wall"
        else:
            kind = toy_kind(o)[0]
            if kind == "unknown": kind = "toy?"
        o["kind"], o["label_src"] = kind, "rules"
        o["keepout"] = {"r_mm": round(k["r_mm"], 1), "h_mm": k["h_mm"], "why": k["why"], "colour_frac": k["colour_frac"]} if k is not None else None
        det = _match(o, detections)
        if det is not None and _kind_of(det.get("label")):
            kind = _kind_of(det["label"]); o["kind"], o["label_src"] = kind, "detector"
            if kind == "keep-out" and o["keepout"] is None: o["keepout"] = {"r_mm": round(0.5 * math.hypot(w, h) + ko().MARGIN_MM, 1), "h_mm": ko().HEIGHT_MM, "why": "detector", "colour_frac": None}
            elif kind != "keep-out": o["keepout"] = None
            if det.get("score") is not None: o["score"] = det["score"]
        out.append(o)
    return out, ring
