"""Keep-out objects on the mat: things the arm must never touch or drop anything on (TC 2026-11-11 22:3x: "be careful not to hurt the frogs").

A scene-map blob is a keep-out when it is not the tape ring and either
  - colourful: at least COLOUR_FRAC of the pixels around it in the top frame have HSV saturation above SAT_MIN (measured 11 Nov 22:4x on the two toy
    frogs: 0.46 and 0.36; on the steel nuts, wing nuts and spacers: at most 0.02), or
  - bigger than any toy: area at least BIG_AREA_MM2 (the frogs measured 2056-2444 mm2; the largest toy blob seen, a wing nut, 256 mm2).
Pieces of the ring's own wall (blobs centred on the wall band, smaller than BIG_AREA_MM2) are not keep-outs; a frog-sized blob on the wall band is.
Each keep-out is a vertical cylinder (centre, radius = half the blob's diagonal + MARGIN_MM, height HEIGHT_MM) that the obstacle checks avoid.
"""
import math
import numpy as np

IN = 25.4
SAT_MIN, COLOUR_FRAC = 120, 0.10
BIG_AREA_MM2 = 1000.0
MARGIN_MM = 20.0
HEIGHT_MM = 60.0          # assumed, not measured: taller than the 51 mm ring wall, so a route that clears a keep-out also clears the ring
RING_BLOB_MM = 80.0       # a blob at least this long, centred within RING_CENTRE_MM of the fitted ring centre, is the ring itself
RING_CENTRE_MM = 30.0


def dist_mm(a, b): return math.hypot(a[0] - b[0], a[1] - b[1]) * IN


def colour_frac(hsv, o, cc, table):
    """Share of the pixels in the blob's square (half-side = half its longer side + 2 px) with saturation > SAT_MIN. None when the square is off the frame."""
    u, v = cc.table_to_pixel(table, o["x_in"] * IN, o["y_in"] * IN); ppm = abs(cc.table_to_pixel(table, o["x_in"] * IN + 10.0, o["y_in"] * IN)[0] - u) / 10.0
    r = int(max(float(o.get("w_mm") or 0), float(o.get("h_mm") or 0)) / 2.0 * ppm) + 2; h, w = hsv.shape[:2]
    y0, y1, x0, x1 = max(int(v) - r, 0), min(int(v) + r + 1, h), max(int(u) - r, 0), min(int(u) + r + 1, w)
    if y1 <= y0 or x1 <= x0: return None
    s = hsv[y0:y1, x0:x1, 1]; return float((s > SAT_MIN).mean())


def find_keepouts(objs, img_bgr, cc, table, ring=None, ring_r_mm=52.0, wall_in_mm=20.0, wall_out_mm=10.0):
    """[{x_in, y_in, r_mm, h_mm, why, colour_frac, area_mm2}] from scene objects and the top frame (BGR array; None: size rule only)."""
    hsv = None
    if img_bgr is not None:
        import cv2
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out = []
    for o in objs:
        if "x_in" not in o or "y_in" not in o: continue
        w, h, area = float(o.get("w_mm") or 0), float(o.get("h_mm") or 0), float(o.get("area_mm2") or 0)
        if ring is not None:
            d = dist_mm((o["x_in"], o["y_in"]), ring)
            if max(w, h) >= RING_BLOB_MM and d < RING_CENTRE_MM: continue                                          # the ring itself
            if ring_r_mm - wall_in_mm < d < ring_r_mm + wall_out_mm and area < BIG_AREA_MM2: continue             # a piece of its wall
            if d < ring_r_mm and area < BIG_AREA_MM2: continue   # inside the ring only a big blob is a keep-out: 12 Nov 2026 the tape roll's tan cardboard core (hue 26-36, 40 mm2, 28 mm in) read as colourful and stopped a round; a frog inside is caught by size (2056-2444 mm2)
        cf = colour_frac(hsv, o, cc, table) if hsv is not None else None
        if ring is not None and d < ring_r_mm and cf is not None and cf < COLOUR_FRAC: continue   # a big, plain blob centred inside the ring is the ring itself: 12 Nov 2026 it was segmented 75 x 58 mm (under RING_BLOB_MM), 25 mm off the band fit, colour 0.06, and stopped a reset; a frog inside is big AND colourful (0.31-0.46)
        why = "colourful" if (cf is not None and cf >= COLOUR_FRAC) else ("big" if area >= BIG_AREA_MM2 else None)
        if why: out.append({"x_in": float(o["x_in"]), "y_in": float(o["y_in"]), "r_mm": 0.5 * math.hypot(w, h) + MARGIN_MM, "h_mm": HEIGHT_MM, "why": why,
                            "colour_frac": None if cf is None else round(cf, 3), "area_mm2": area})
    return out


def is_keepout_object(o, keepouts, tol_mm=1.0):
    """The scene object that a keep-out was made from (same centre)."""
    return any(dist_mm((o["x_in"], o["y_in"]), (k["x_in"], k["y_in"])) < tol_mm for k in keepouts)


def point_hits(x_in, y_in, h_mm, rad_mm, keepouts, droop_mm=0.0, margin_mm=5.0):
    """The keep-outs a body point of radius rad_mm at (x, y, h) touches: inside a cylinder's radius + rad_mm and below its height + margin (after droop)."""
    return [k for k in keepouts if dist_mm((x_in, y_in), (k["x_in"], k["y_in"])) - k["r_mm"] < rad_mm and h_mm - droop_mm < k["h_mm"] + margin_mm]


def segment_clear(a_in, b_in, keepouts, half_width_mm):
    """True when a straight carry from a to b (mat inches) keeps a body of half-width half_width_mm clear of every keep-out's radius (in plan view)."""
    ax, ay, bx, by = a_in[0] * IN, a_in[1] * IN, b_in[0] * IN, b_in[1] * IN; dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
    for k in keepouts:
        px, py = k["x_in"] * IN, k["y_in"] * IN; t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        if math.hypot(px - (ax + t * dx), py - (ay + t * dy)) < k["r_mm"] + half_width_mm: return False
    return True


def inside_ring(keepouts, ring, ring_r_mm=52.0):
    """Keep-outs whose disc reaches into the ring's opening: nothing may be released into the ring then."""
    return [k for k in keepouts if dist_mm((k["x_in"], k["y_in"]), ring) < ring_r_mm - 5.0]


def blue_components(mask, min_px=150):
    """[(ys, xs)] of the 8-connected components of a boolean mask with at least min_px pixels, largest first. why: 11 Nov 22:4x the blue legs of two toy
    frogs pulled the ring's least-squares circle 4 in off; the tape is one connected band, each frog a separate region (tidy.fit_ring keeps the first ring-shaped one)."""
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    idx = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [tuple(np.nonzero(lab == i)) for i in idx]
