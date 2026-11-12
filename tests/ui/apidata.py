"""Fixture builders shaped like dashboard/server.py's JSON (state, hist rows, record session, calibration)."""
import time
import urllib.parse


def now():
    return time.time()


def jetson(online=True, emc=42, emc_mhz=665, gpu=35, cpu=22.5, temp=48.2, watts=7.9, ram_used=3000, ram_total=7600):
    j = {"online": online, "last": now()}
    if online:
        j.update(gpu=gpu, emc=emc, emc_mhz=emc_mhz, cpu=cpu, cpu_cores=[10, 20, 30, 40, 50, 60], ram_used=ram_used,
                 ram_total=ram_total, temp=temp, watts=watts, gpu_mhz=612, mode="MAXN", uptime="3 hours",
                 disk={"used": "40G", "total": "230G"})
    return j


def datasets(t=None):
    t = t or now()
    return {
        "alpha": {"host": "mac", "mtime": int(t - 3000), "episodes": 12, "frames": 9000, "fps": 30, "hours": 0.083},
        "bravo@jetson": {"host": "jetson", "mtime": int(t - 100), "episodes": 7, "frames": 5250, "fps": 30, "hours": 0.049},
        "charlie": {"host": "mac", "mtime": int(t - 50000), "episodes": 30, "frames": 22500, "fps": 30, "hours": 0.21},
    }


NEWEST_FIRST = ["bravo", "alpha", "charlie"]   # by mtime, '@jetson' stripped


def state(online=True, **jk):
    t = now()
    return {"jetson": jetson(online, **jk),
            "mac": {"cpu": 12.0, "ram_used": 16000, "ram_total": 32768, "disk_free_gb": 120.5},
            "training": {}, "datasets": datasets(t), "policies": {}, "procs": [],
            "events": [{"t": t, "msg": "stub online"}], "arms": {}, "spotify": {"running": False},
            "lock": {"locked": False}, "sim_arm": None, "now": t}


def hist(n, emc=True, step=2):
    """n tegrastats rows [t, gpu%, emc%|None, cpu%, ram%, temp, watts], newest = now, 2 s apart."""
    t = now()
    return [[round(t - (n - 1 - i) * step), 20 + (i * 7) % 60, (30 + (i * 5) % 50) if emc else None,
             25 + (i * 3) % 40, 40, 45.0 + (i % 10), 6.5 + (i % 5) * 0.5] for i in range(n)]


def record(**over):
    r = {"running": False, "name": "", "task": "", "episodes": 0, "episode": 0, "phase": "", "started": 0,
         "ep_started": 0, "episode_s": 25, "reset_s": 10, "done": False, "err": "", "log": "", "done_total": 0,
         "episodes_target": 0, "session_episode": 0, "paused_at": 0}
    r.update(over)
    return r


def record_running(phase="recording", elapsed=10, paused_at_elapsed=None, name="sessionB", episode=3, target=20):
    """A live session `elapsed` seconds into episode `episode`; paused_at_elapsed freezes it at that many seconds."""
    t = now()
    r = record(running=True, name=name, task="Pick up the lighter", episodes=target, episode=episode, phase=phase,
               started=t - 100, ep_started=t - elapsed, episodes_target=target, session_episode=episode)
    if paused_at_elapsed is not None:
        r["paused_at"] = r["ep_started"] + paused_at_elapsed
    return r


def record_done(err="", name="sessionB", episode=20):
    return record(done=True, name=name, episode=episode, phase="error" if err else "done", err=err,
                  episodes=20, episodes_target=20)


# ---- camera calibration (/api/calib, /api/locate): the real 2026-11-09 overhead homography, 640x360 frame ----
CALIB_H = [[-0.9079913357195868, 0.028177436289868022, 486.04602808376177],
           [0.020276156121766534, 0.9119972726007077, 2.0569471177725482],
           [-2.0239151283753944e-05, 2.8591617534012568e-05, 1.0]]          # raw px -> table mm
CALIB_MARKERS = {"0": [16.0, 10.0], "1": [2.0, 10.0], "2": [15.0, 4.0], "3": [3.0, 4.0]}
CALIB_ARM = {"base_mm": [229.6, -80.8], "base_in": [9.04, -3.18], "yaw_deg": 90.15, "table_z_mm": 3.3,
             "joint_offsets_deg": {"shoulder_lift": -3.91, "elbow_flex": -5.18, "wrist_flex": -0.69},
             "wrist_roll_deg": 0.46, "fitted": "2026-11-09 21:33", "residual_xy_mm_max": 1.09, "points": 4}


def calib_none():
    return {"ok": False, "err": "no calibration yet"}


def camera_pose(position_mm=(218.0, 162.0, 408.0), optical_axis=(0.0, 0.0, -1.0), up_table=(0.0, 1.0, 0.0), hfov_deg=70.4,
                source="nominal C920 hfov, height from the homography scale", **over):
    """camera.pose as /api/calib passes camera_pose_est through (cam_calib.py campose): table frame, z up from the mat."""
    p = {"position_mm": list(position_mm), "position_in": [round(v / 25.4, 3) for v in position_mm], "optical_axis": list(optical_axis),
         "up_table": list(up_table), "tilt_deg": 0.0, "axis_hits_mm": [position_mm[0], position_mm[1]], "hfov_deg": hfov_deg, "image_size": [640, 360],
         "reproj_rms_px": 0.41, "source": source, "created": "2026-11-10 09:00:00"}
    p.update(over)
    return p


# ---- FRONT camera (camera.front in /api/calib): a solved pose like jetson/calib/front_campose.json, 640x360, no distortion ----
# Camera centre FRONT_C in the table frame looking at FRONT_AIM on the mat (tilt 41.7 deg from straight down), image up roughly along table +Y
# (the real front camera is rolled that way: image-right climbs in the world). R rows = camera axes in the table frame (table -> camera, OpenCV),
# t = -R C, rvec = Rodrigues(R); hfov 68.88 deg -> f = 320 / tan(34.44 deg). Derived with cv2 once and pinned here, so the UI tests stay cv2-free.
FRONT_C = [-8.0, 354.0, 467.0]
FRONT_AIM = [370.0, 180.0]
FRONT_F = 466.649017432
FRONT_R = [[0.77728363, 0.0, 0.629150347], [-0.175015724, -0.96052957, 0.21622313], [0.604317512, -0.278177902, -0.74660391]]
FRONT_RVEC = [-2.708064241, 0.136020978, -0.958642463]
FRONT_TVEC = [-287.594942979, 237.651140387, 451.973543745]
FRONT_K = [[FRONT_F, 0.0, 320.0], [0.0, FRONT_F, 180.0], [0.0, 0.0, 1.0]]
FRONT_AXIS = [0.604317512, -0.278177902, -0.74660391]       # R^T (0, 0, 1)
FRONT_UP = [0.175015724, 0.96052957, -0.21622313]           # R^T (0, -1, 0)


def _mat3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


FRONT_H_T2P = _mat3(FRONT_K, [[FRONT_R[i][0], FRONT_R[i][1], FRONT_TVEC[i]] for i in range(3)])   # table mm -> px: K [r1 r2 t]


def _front_p2t():
    """px -> table mm: the inverse, scaled by a positive factor so |[2][2]| = 1 (as server.py serves it)."""
    hi = h_inv(FRONT_H_T2P)
    w = abs(hi[2][2])
    return [[v / w for v in row] for row in hi]


def front_pose(**over):
    """camera.front.pose as /api/calib passes front_campose.json through (plus position_in)."""
    p = {"position_mm": list(FRONT_C), "position_in": [round(v / 25.4, 3) for v in FRONT_C], "rvec": list(FRONT_RVEC), "tvec": list(FRONT_TVEC),
         "optical_axis": list(FRONT_AXIS), "tilt_deg": 41.702952, "axis_hits_mm": list(FRONT_AIM), "up_table": list(FRONT_UP), "reproj_rms_px": 1.91,
         "markers_used": [0, 2, 3], "camera_matrix": [list(r) for r in FRONT_K], "hfov_deg": 68.88, "image_size": [640, 360],
         "source": "lens height 18.375 in (tape measure) -> hfov 68.88 deg", "units_note": "position_mm in the table frame (z up from the mat surface)",
         "created": "2026-11-09 22:58:43"}
    p.update(over)
    return p


def front(**over):
    """camera.front as /api/calib emits it: the pose plus the mat-surface homography pair derived from it."""
    f = {"pose": front_pose(), "homography_px_to_table_mm": _front_p2t(), "homography_table_to_px": [list(r) for r in FRONT_H_T2P],
         "reproj_rms_px": 1.91, "created": "2026-11-09 22:58:43"}
    f.update(over)
    return f


def calib(ok=True, arm=True, pose=None, front=None, **over):
    """GET /api/calib as dashboard/server.py emits it: the overhead-camera table fit plus (arm=True) the arm pose fit,
    (pose=camera_pose()) the estimated top camera pose and (front=front()) the FRONT camera block. calib(ok=False) is the
    uncalibrated answer; keyword overrides patch the top-level payload."""
    if not ok:
        return calib_none()
    c = {"ok": True, "units": "in", "mat": {"width_in": 18, "depth_in": 12}, "image_size": [640, 360], "marker_size_in": 2.0,
         "layout_markers_in": dict(CALIB_MARKERS),
         "homography_px_to_table_mm": [list(r) for r in CALIB_H], "has_intrinsics": False,
         "camera": {"residual_mm_max": 0.12, "created": "2026-11-09 21:12:01", "markers_used": [0, 1, 2, 3], "pose": pose, "front": front},
         "arm": {**CALIB_ARM, "joint_offsets_deg": dict(CALIB_ARM["joint_offsets_deg"])} if arm else None,
         "files": {"table": "jetson/calib/top_table.json", "layout": "jetson/calib/top_layout.json", "table_mtime": 1.0},
         # self-healing camera solve (docs/mat-scene.md): idle values; calib(live=True, shift_px=10.6, live_at=...) is a bumped camera
         "stored_homography": [list(r) for r in CALIB_H], "live": False, "live_at": None, "shift_px": None,
         "holding": False, "calib_ok": None, "calib_note": None, "markers_seen": None, "live_solve": None}
    c.update(over)
    return c


# ---- mat scene (/api/scene): objects on the mat as dashboard/scene_detect.py reports them through server.py ----
def scene_none(err="top camera not live"):
    return {"ok": False, "err": err, "age_s": None}


def scene_obj(id=1, x_in=10.94, y_in=6.94, w_mm=12.0, h_mm=10.0, angle_deg=-90.0, color=(121, 126, 129), **over):
    """One detected object (default: the hex nut of tests/scene/fixtures/obj.jpg), mat inches in, the server's mm/in fields out."""
    x_mm, y_mm = x_in * 25.4, y_in * 25.4
    o = {"id": id, "x_mm": round(x_mm, 1), "y_mm": round(y_mm, 1), "x_in": round(x_in, 3), "y_in": round(y_in, 3), "w_mm": w_mm, "h_mm": h_mm,
         "area_mm2": round(w_mm * h_mm * 0.8), "angle_deg": angle_deg, "color": list(color), "px": [273.8, 209.2],
         "bbox_mm": [round(x_mm - w_mm / 2, 1), round(y_mm - h_mm / 2, 1), w_mm, h_mm]}
    o.update(over)
    return o


def scene(objects=(), background="ok", **over):
    """GET /api/scene as server.py emits it when the top camera is live and the scene thread has run."""
    s = {"ok": True, "t": now(), "age_s": 0.3, "objects": list(objects), "arm_in_view": False, "arm_mask_frac": 0.0, "background": background,
         "n_markers_seen": 4, "calib_err_mm": 0.68, "calib_ok": True, "calib_note": None, "refitted": False, "mat_shift_mm": 0.0, "mat_rot_deg": 0.0,
         "gain": 1.0, "px_per_mm": 0.5, "mat_mm": [457.2, 304.8], "calib_live": False, "calib_live_at": None, "shift_px": None, "holding": False}
    s.update(over)
    return s


def h_apply(m, x, y):
    w = m[2][0] * x + m[2][1] * y + m[2][2]
    return [(m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w]


def h_inv(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    A, B, C = e * i - f * h, f * g - d * i, d * h - e * g
    s = 1.0 / (a * A + b * B + c * C)
    return [[A * s, (c * h - b * i) * s, (b * f - c * e) * s], [B * s, (a * i - c * g) * s, (c * d - a * f) * s],
            [C * s, (b * g - a * h) * s, (a * e - b * d) * s]]


def table_to_arm(mx, my, arm=CALIB_ARM):
    import math
    y = math.radians(arm["yaw_deg"])
    dx, dy = mx - arm["base_mm"][0], my - arm["base_mm"][1]
    return [dx * math.cos(y) + dy * math.sin(y), -dx * math.sin(y) + dy * math.cos(y)]


def locate(path):
    """Callable stub route: answers /api/locate?u=&v=[&cam=front] like server.py would, from CALIB_H (top) or the front homography."""
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
    try:
        u, v = float(q["u"][0]), float(q["v"][0])
    except (KeyError, ValueError):
        return {"ok": False, "err": "u and v required"}
    cam = (q.get("cam") or ["top"])[0]
    mm = h_apply(_front_p2t() if cam == "front" else CALIB_H, u, v)
    am = table_to_arm(mm[0], mm[1])
    r = {"ok": True, "pixel": [u, v], "table_mm": mm, "table_in": [mm[0] / 25.4, mm[1] / 25.4],
         "arm_mm": am, "arm_in": [am[0] / 25.4, am[1] / 25.4]}
    if cam == "front":
        r["cam"] = "front"
    return r


# ---- wrist camera hand-eye pose (/api/calib camera.wrist = jetson/calib/wrist_campose.json from cam_calib.py handeye) ----
def _quat_xyzw(R):
    """Unit quaternion (x, y, z, w) of a 3x3 rotation given as rows (the same formula cam_calib._R_to_quat_xyzw uses)."""
    import math
    m = R; tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2; w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2; w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2; w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2; w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return [x / n, y / n, z / n, w / n]


def wrist_pose(position_mm=(1.3, 68.0, -11.2), optical_axis=(-0.095, -0.493, -0.865), up=(-0.105, 0.869, -0.484), hfov_deg=62.3, **over):
    """camera.wrist as the orchestrator serves wrist_campose.json: the wrist camera's centre and axes in URDF gripper_link
    (position_mm; rotation = camera axes x right / y down / z forward expressed in gripper_link, as quaternion_xyzw plus the
    optical axis and image-up unit vectors), the pinhole model and the fit quality. Defaults = the real 9 Nov 2026 solve."""
    import math
    n = math.sqrt(sum(v * v for v in optical_axis)); z = [v / n for v in optical_axis]
    d = sum(a * b for a, b in zip(up, z)); u = [a - d * b for a, b in zip(up, z)]; n = math.sqrt(sum(v * v for v in u)); u = [v / n for v in u]
    y = [-v for v in u]; x = [y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]]
    R = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]   # columns = camera axes in gripper_link
    fx = 320.0 / math.tan(math.radians(hfov_deg) / 2)
    p = {"frame": "gripper_link", "position_mm": list(position_mm),
         "rotation": {"quaternion_xyzw": _quat_xyzw(R), "optical_axis_gripper": z, "up_gripper": u, "note": "camera axes in gripper_link"},
         "camera_matrix": [[fx, 0.0, 320.0], [0.0, fx, 240.0], [0.0, 0.0, 1.0]], "dist_coeffs": [-0.41, 0.15, 0.0, 0.0, 0.0], "image_size": [640, 480],
         "hfov_deg": hfov_deg, "vfov_deg": math.degrees(2 * math.atan(240.0 / fx)), "tilt_from_gripper_axis_deg": math.degrees(math.acos(-z[2])), "off_axis_mm": math.hypot(position_mm[0], position_mm[1]),
         "reproj_rms_px": 8.07, "per_view_px": {"w1": 3.2, "w2": 11.3, "w3": 7.3, "w5": 10.3, "w6": 5.3}, "reproj_rms_px_corrected": 0.51,
         "leave_one_out_px": {"w1": 4.3, "w2": 15.7, "w3": 9.7, "w5": 13.1, "w6": 6.6}, "views_used": ["w1", "w2", "w3", "w5", "w6"], "markers_used": [1, 2, 3],
         "created": "2026-11-10 00:30:00", "source": {"views_dir": "jetson/calib/wristcam"}}
    p.update(over)
    return p


def tele_none():
    """/api/tele before the Jetson sends any servo telemetry: no rows, no flags."""
    return {"keys": [], "joints": [], "rows": [], "limits": {}, "src": None, "age": None, "n": 0, "now": now(), "t": None, "flags": [], "events": [], "dt_ms": None,
            "marks": {"temp_cut": 70, "temp_warn": 60, "temp_hot": 65, "volt_warn": 11.0, "volt_low": 10.0, "bus_a": 5.0}}


def default_routes():
    return {
        "/api/tele": tele_none(),
        "/api/state": state(),
        "/api/hist": hist(20),
        "/api/record": record(),
        "/api/record/ctl": {"ok": True},
        "/api/record/apply": {"ok": True},
        "/api/cams": [],
        "/api/term": {"up": False, "url": ""},
        "/api/shell": [],
        "/api/gallery": [],
        "/api/lock": {"locked": False, "status": ""},
        "/api/unlock": {"ok": True},
        "/api/sim": {"active": False, "gifs": [], "evals": [], "datasets": [], "procs": []},
        "/api/calib": calib_none(),
        "/api/arm/": {"ok": True},
        "/api/spotify/": {"ok": True},
        "/api/locate": locate,
        "/api/scene": scene_none(),
        "/api/scene/background": {"ok": True},
    }


def labelled_scene():
    """GET /api/scene with what each blob is (tools/scene_labels.py): the ring, a frog (keep-out), a nut, a wing nut, a spacer, a toy? and a piece of the ring's wall."""
    ring = {"x_in": 7.0, "y_in": 8.0, "r_mm": 52.0, "h_mm": 51.0}
    frog = scene_obj(id=20, x_in=12.5, y_in=3.0, w_mm=61.0, h_mm=54.0, area_mm2=2148.0, color=(220, 110, 30), kind="keep-out", label_src="rules",
                     keepout={"r_mm": 60.8, "h_mm": 60.0, "why": "colourful", "colour_frac": 0.41})
    objs = [scene_obj(id=10, x_in=7.0, y_in=8.0, w_mm=109.0, h_mm=109.0, area_mm2=6116.0, kind="ring", keepout=None, label_src="rules"), frog,
            scene_obj(id=11, x_in=10.9, y_in=7.1, kind="nut", keepout=None, label_src="rules"),
            scene_obj(id=12, x_in=4.0, y_in=3.5, w_mm=20.0, h_mm=12.0, kind="wing nut", keepout=None, label_src="rules"),
            scene_obj(id=13, x_in=11.5, y_in=5.0, w_mm=22.0, h_mm=9.0, kind="spacer", keepout=None, label_src="rules"),
            scene_obj(id=14, x_in=14.0, y_in=7.0, w_mm=24.0, h_mm=12.0, kind="toy?", keepout=None, label_src="rules"),
            scene_obj(id=15, x_in=7.0, y_in=8.0 + 40 / 25.4, w_mm=30.0, h_mm=30.0, kind="ring wall", keepout=None, label_src="rules")]
    return scene(objs, ring=ring, keepouts=[{"id": 20, "x_in": 12.5, "y_in": 3.0, **frog["keepout"]}])
