"""tools/pick_loop.py's near search ignores dark blobs cut by its search box (11 Nov 2026 16:32): looking near (9, 6) it took a sliver of a printed
dark square, cut by the box's right edge at mat (6.74, 5.93), for the nut; the real nut sat on a light square at (10.17, 7.32) and the tilt test pick went to empty mat."""
import math


def _frame(cc, table, sliver=True, nut_px=None):
    import cv2, numpy as np
    w, h = table.get("image_size") or [640, 360]; img = np.full((h, w, 3), 200, np.uint8)
    cu, cv_ = cc.table_to_pixel(table, 9.0 * 25.4, 6.0 * 25.4); cu, cv_ = int(round(cu)), int(round(cv_))
    if sliver: cv2.rectangle(img, (cu + 64, cv_ - 12), (cu + 64 + 25, cv_ + 12), (20, 20, 20), -1)   # a 25 px dark square; the 70 px box keeps a 6 px strip of it
    if nut_px: cv2.circle(img, (cu + nut_px[0], cv_ + nut_px[1]), 4, (20, 20, 20), -1)
    return cv2.imencode(".png", img)[1].tobytes(), (cu, cv_)


def test_a_blob_cut_by_the_search_box_is_not_the_nut(pl, cc, table):
    frame, _ = _frame(cc, table)
    assert pl.find_nut(cc, table, [9.0, 6.0], jpeg=frame, joints=None) is None, "the strip of a printed square cut by the box edge is not a nut"


def test_the_nut_inside_the_box_wins_over_a_nearer_cut_sliver(pl, cc, table):
    frame, (cu, cv_) = _frame(cc, table, nut_px=(-58, 44))   # 73 px from the centre; the sliver's centroid is ~67 px
    got = pl.find_nut(cc, table, [9.0, 6.0], jpeg=frame, joints=None)
    assert got and math.hypot(got[1][0] - (cu - 58), got[1][1] - (cv_ + 44)) < 2.0


def test_a_whole_blob_inside_the_box_is_still_found(pl, cc, table):
    frame, (cu, cv_) = _frame(cc, table, sliver=False, nut_px=(20, -10))
    got = pl.find_nut(cc, table, [9.0, 6.0], jpeg=frame, joints=None)
    assert got and math.hypot(got[1][0] - (cu + 20), got[1][1] - (cv_ - 10)) < 2.0
