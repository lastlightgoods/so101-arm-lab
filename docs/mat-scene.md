# Mat scene: what is lying under the overhead camera

`dashboard/scene_detect.py` turns one frame of the overhead C920 into a list of objects on the Fiskars mat, in mat
millimetres and inches, so the wall dashboard can draw a live map of the workspace. It is cv2 + numpy only, has no
camera or network code, and runs in about 5-8 ms per 640x360 frame on the Mac (budget: 25 ms; the server polls at ~2 Hz).

```
from dashboard.scene_detect import MatScene          # or importlib on dashboard/scene_detect.py
scene = MatScene(json.load(open("jetson/calib/top_table.json")))
scene.set_background(empty_frame)                     # once, with nothing on the mat (arm parked)
res = scene.detect(frame)                             # ~2 Hz
```

Command line, for checking by eye:

```
.venv/bin/python dashboard/scene_detect.py --calib jetson/calib/top_table.json --bg empty.jpg frame1.jpg frame2.jpg --save-warp out.png
```

prints the `detect()` JSON per frame and writes `out_0.png`, `out_1.png`: the top-down mat with the arm mask tinted
red, foreground green, marker exclusion squares yellow, objects boxed with id and inches, and a 1 in grid.
`--bg` may be repeated (median of several empty frames) or point at a background PNG saved with `--save-bg`.

## Conventions

**Mat frame** (from `cam_calib.py`): X along the long edge, 0..457.2 mm (18 in); Y away from the arm, 0..304.8 mm
(12 in); origin at the mat corner on the arm's side, left as seen from the arm. The arm's base sits at `arm_base_mm`,
below the mat (y < 0), and the arm enters the mat across the y = 0 edge.

**Warped image** (`scene.warp(frame)`, `scene.last["warp"]`, `render()`): column = x * px_per_mm,
row = (304.8 - y) * px_per_mm. The origin is the BOTTOM-LEFT corner and the arm comes in from the bottom edge, so it
reads like a map drawn from the operator's side. Default `px_per_mm = 0.5` -> 229 x 152 px; one pixel is 2 x 2 mm.
(0.4 px/mm works too but a hex nut is then only ~17 px; 0.5 keeps small hardware comfortably above the speck limit at
no measurable cost: the warp itself takes 0.1 ms.)

**Raw pixels**: the calibration homography `homography_px_to_table` maps 640x360 camera pixels to mat mm; the camera
hangs upside down and the homography already contains that. A frame of another resolution is handled by scaling the
homography, and `px` in the result is in that frame's own pixels.

## detect() result

```
{"t": epoch, "objects": [ {"id": 3, "x_mm": 278.0, "y_mm": 176.1, "x_in": 10.945, "y_in": 6.932,
                            "w_mm": 12.0, "h_mm": 10.0, "angle_deg": -90.0, "area_mm2": 88, "color": [119,124,128],
                            "px": [273.8, 209.2], "bbox_mm": [x, y, w, h]} ...],
 "arm_in_view": true, "arm_mask_frac": 0.045, "arm_masked_by": "kinematics" | "edge" | "none", "background": "ok" | "missing",
 "n_markers_seen": 4, "calib_err_mm": 0.68, "calib_ok": true, "calib_note": null, "refitted": false,
 "mat_shift_mm": 0.0, "mat_rot_deg": 0.0, "gain": 1.008, "px_per_mm": 0.5, "mat_mm": [457.2, 304.8]}
```

* `x_mm, y_mm` (and `_in`): centroid of the blob on the mat surface. `w_mm, h_mm, angle_deg`: the minimum-area
  rectangle around the blob, `w` the long side, `angle_deg` its direction counter-clockwise from mat +X in (-90, 90].
  `bbox_mm` is the axis-aligned box `[x, y, w, h]` with `(x, y)` its lower-left corner in mat mm. `color` is the mean
  RGB of the blob. `px` is the raw-frame centroid for drawing on the camera view.
* `id`: stable while the object stays within 15 mm of where it was last seen; a track survives 3 missed calls (a
  flicker, the arm passing over) before the id is retired; new objects get fresh ids.
* `pending`: tracks seen but not yet listed. `MatScene(min_hits=N)` reports an object only once it has been seen in N
  consecutive calls (default 1); the server uses `SCENE_MIN_HITS = 3` (~1.5 s), which keeps one-frame shadows, the
  arm's motion blur and sensor flicker off the 3D map. The hit count lives on the track, so the id is the same when
  the object finally appears.
* `arm_in_view` / `arm_mask_frac`: whether any of the mat is under the arm mask and how much of it. The mask is the
  arm's kinematic silhouette when the caller passed one (`arm_masked_by: "kinematics"`, next section), plus every blob
  touching the arm-side edge (its attached shadow included; the only arm rule when no silhouette was passed:
  `"edge"`). Big blobs touching the OTHER edges (a hand reaching in, > 60 mm) count as intrusions: they go into the
  same mask and are not reported as objects. `"none"`: no background yet, nothing was masked.
* `background: "missing"`: `set_background` has not been called; `objects` is empty, everything else is still filled.
* Calibration health (see below): `n_markers_seen`, `calib_err_mm`, `calib_ok`, `calib_note`, `refitted`,
  `mat_shift_mm`, `mat_rot_deg`.
* `gain`: the exposure factor applied to the frame before differencing (the C920 auto-exposes).

## How it works

1. **Warp.** The frame is softened slightly (so the ~2x downscale does not alias the mat's printed grid) and warped
   with `A @ H` into the mat-space image above. Everything after this is in mat space, so a pixel is a known
   number of millimetres everywhere and "touches the arm edge" is one row test.
2. **Background.** `set_background(frame)` stores the warped empty mat (float32). Given a list of frames it takes the
   per-pixel median, which drops anything present in fewer than half of them (the arm, provided it is somewhere
   different in most frames; if it sits in the same place in most of them it gets baked in). A top-down image of the
   right shape is accepted as is. `save_background(png)` / `load_background(png)` persist it; the background
   lives in mat space, so it stays valid when the mat shifts under the camera (see re-fit).
3. **Difference.** Exposure match first: the ratio of the median grey of background and frame over the free mat area
   (clamped 0.6..1.6) scales the frame. Then per-channel absolute difference (max over B, G, R: a coloured object with
   the mat's luminance still shows), 3x3 median, threshold 28/255, 3x3 closing, and the exclusion mask: the four marker
   squares plus a 10 mm margin (they are printed, they never move relative to the mat, and the arm's shadow across
   them would otherwise flicker) and a 2 mm border.
4. **Arm and intrusions.** When the caller knows where the arm is (`detect(frame, arm_px=...)`, built by the server
   from the follower's joints: next section) its silhouette is masked outright. Then connected components. A blob
   reaching the last valid row on the arm side is the arm too; a blob reaching another edge is an intrusion if it is
   wider or taller than 60 mm. All of it forms the arm mask. The edge blobs' 10 mm halo swallows fragments hanging off
   them (cables, bits of shadow, an object the gripper is holding); anything touching the kinematic silhouette goes
   as well.
5. **Shadows.** Per remaining component, not per pixel: a blob is the arm's detached shadow when it is the
   background only darker (grey ratio between 0.40 and 0.97 at the 10th-90th percentile, ratio spread < 0.12, and the
   same colour once scaled). Per-component matters: a grey hex nut is also darker than the mat on average, but its
   bright rim and dark hole give it a ratio spread of 0.3+, so it survives where a pixel test killed it.
6. **Objects.** Components with 40 mm2 .. 35 % of the mat: centroid, min-area rectangle, mean colour, raw pixel
   through the inverse homography, then nearest-neighbour association with the previous call's tracks.

## Kinematic arm mask: the follower's joints say where the arm is

The edge rule alone has a hole TC saw on the wall on 9 Nov: it only knows a blob is the arm when the blob touches the
arm-side edge of the mat. When the arm reaches over the mat, the upper arm leaves the mat's edge region through the
air, the forearm casts its shadow somewhere else, and the gripper and its fingers become detached blobs in the middle
of the mat, which the detector duly reported as objects. The fix masks the arm from its known geometry instead of
guessing from the picture.

**What is masked.** `jetson/cam_calib.arm_mask_px_from_joints(joints_deg, calib)` takes the follower's joint angles
(LeRobot degrees, what `/api/state` serves under `arms.follower.pos` with `mode: "deg"`) and the table calibration,
adds the fitted joint zero offsets (`arm_fit.joint_offsets_deg`), runs the forward kinematics (`fk_chain`: the origins
of shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll and the gripper tip, in the base frame, mm) and
appends a "fingers" point 40 mm beyond the tip along the gripper axis. Each point gets a radius in mm
(`ARM_MASK_RADII_MM`: base 50, shoulder 40, elbow 30, wrist 30, wrist roll 35, tip 30, fingers 20; the SO-101 links are
40-60 mm across, the base plate ~100 mm, the gripper with its wrist camera mount ~70 mm, each with ~10 mm of slack
for cables and print) times 1.3 for the cast shadow that hugs the links. The wrist camera and its cable get no special
treatment: they are inside those radii. `MatScene.detect(frame, arm_px={"points": [[u, v], ...], "radii_px": [...]})`
rasterises them in RAW camera pixels as filled circles joined by strokes as thick as the larger of the two radii
(`kinematic_mask`), warps that mask into mat space with the very homography the frame goes through, ORs it into the
arm mask (so `arm_in_view` and `arm_mask_frac` count it), and drops every foreground component inside or touching it.
The edge rule keeps running underneath: the mask only ever grows. `render()` tints the whole arm mask red, so
`/api/scene.png` shows the silhouette.

**Why the camera pose and not the homography.** The homography is a map for the mat surface. The arm is not on the
mat surface: the elbow sits 200+ mm up under a camera 419 mm up, so the camera sees it displaced away from the point
where the optical axis meets the mat by `height x off-axis distance / (camera height - height)`, over 100 mm for the
elbow in the jogged (9, 7) pose, and about twice as large as its millimetres suggest. So the points are projected as
3D points: base frame -> table frame (`arm_to_table`, the inverse of `table_to_arm`; z above the mat =
z_base - `arm_fit.table_z_mm`) -> raw pixels through `camera_pose_est` (`rvec`, `tvec`, `camera_matrix` from
`cam_calib.py campose --lens-height`, no distortion: `arm_points_to_camera_px`), and the radius of each point is converted
to pixels at that point's own depth (`r_px = r_mm x 1.3 x fx / depth`): a link close to the camera masks the bigger
patch it really covers. Check: the tip of the follower jogged onto mat (9, 7) in projects within 1 px of where the table
homography puts (9, 7) (`tests/calib/test_arm_mask_geometry.py`).

**Fallback without a camera pose.** A table calibration that has never had `campose` run (no `camera_pose_est`) still
gets a mask: the points are projected at mat level through the homography at their table x, y only, px/mm is the
homography's local scale there, and the radii are inflated by a further 60 % (`source: "homography"`). Limitation:
parallax is ignored, so a link high above the mat and far from the camera axis appears outside its circle; the forearm
and gripper near the axis are covered, the elbow when it is high and far out may not be. Run `campose` with the tape-
measured lens height to get the real thing.

**Mat-level footprint** (real calibration, lens 419 mm up, jogged (9, 7) pose): tip 39 mm, wrist roll ~60 mm, wrist
flex ~63 mm, elbow ~92 mm around where each is seen. An object the gripper is hovering over vanishes from the map while
the arm is there (its track survives 3 missed calls) and comes back when the arm leaves. Objects 80 mm from the tip and
clear of the forearm are reported with the arm in place (`tests/scene/test_arm_mask.py`).

**Server wiring.** Each detect, `scene_loop` reads the latest follower joints the arm stream has left in memory
(`state["arms"]["follower"]`), uses them only when they arrived within 2 s (`SCENE_JOINTS_MAX_AGE_S`) and are in degrees
(`mode: "deg"`, i.e. the Jetson has a motor calibration), builds `arm_px` with the helper (the scene's current, possibly
re-fitted homography stands in for the file's in the fallback path) and passes it to `detect()` and to the next
`update_background()`, so the arm is never learned into the background either. No fresh joints, no arm pose in the
calibration file, or no cam_calib: the detector falls back to the edge rule (one LOG line per distinct reason,
`arm mask skipped: ...`). `/api/scene` carries `arm_masked_by`: `"kinematics"`, `"edge"` or `"none"`.

## Calibration drift: the markers re-fit the homography

Every call detects the four ArUco markers in the raw frame (2.6 ms) and measures how far their centres land from
`layout_markers` under the current homography (`calib_err_mm`). The mat is only taped down: it moved 13-19 mm during
the evening of 9 Nov between two frames half an hour apart. When the error exceeds 3 mm and at least 3 layout markers
are visible, a fresh homography is fitted from all their corners (same recipe as `cam_calib.solve_table`: affine
through the centres, Procrustes snap of each marker to an exact square, `findHomography` on 12-16 points) and used
from then on (`refitted: true`). The background does not need re-learning: it is an image of the mat, not of the
camera's view. With only 3 markers the perspective is weakly constrained outside their triangle (a few mm at the far
corner), which still beats a 15 mm offset; with fewer, the stored homography stays and `calib_ok` goes false.

What the re-fit cannot tell is a re-glued layout. On 9 Nov the mat was turned round and the markers re-glued in a
congruent rectangle: the four centres still fit the old layout as a rigid move, with sub-millimetre residual, and only
the ids' places differ. So the module also reports how far the mat now sits from where it was calibrated
(`mat_shift_mm`, `mat_rot_deg`, from where the calibrated mat centre and +X axis land under the current homography)
and clears `calib_ok` with a `calib_note` when that exceeds 150 mm or 30 deg: re-run `cam_calib.py table`. The
re-fit is still applied as the best available guess. Disable all of this with `MatScene(..., refit=False)`.

## Background adaptation

`update_background(frame, alpha=0.02, alpha_busy=0.0)` blends the current warp into the background on pixels that are
NOT flagged as object or arm (plus a halo), so slow lighting drift is followed while objects stay detected. The
trade-off is in `alpha_busy`:

* `alpha_busy = 0` (default): an object left on the mat is reported indefinitely; a persistent false positive (a shadow
  the test let through, a scuff on the mat, a lighting change the gain match did not cover) never heals either.
  Press "learn background" (`set_background`) when the mat is known to be empty.
* `alpha_busy = 0.002`: everything is absorbed after ~1/alpha_busy updates (about 8 minutes at 2 Hz): false positives
  heal, and objects that stay put vanish from the map after that long, then reappear as "new" when moved.

The exposure gain in `detect()` already covers global brightness changes, so `alpha` can stay small; calling
`update_background` once every few detects is enough.

## Limits

* **Height is unknown.** The homography is a plane-to-plane map for the mat surface (no intrinsics). Everything is
  placed at mat level: the top of a tall object reads displaced away from the camera axis by roughly
  `height x (distance from the axis) / (camera height)`; a 40 mm tall block 200 mm off-axis under a camera ~700 mm up
  reads ~11 mm too far out. The gripper's shadow and the arm's own parts are 3D for the same reason.
* **Objects touching the arm, in its shadow, under the kinematic silhouette, or within ~5 mm of the arm-side edge**
  merge with the arm mask and are not reported (they come back when the arm moves away). Objects within the marker
  squares (+10 mm) are invisible. The silhouette follows the camera pose the server serves (the live marker re-solve
  once the stand has been bumped by more than 5 mm / 1 deg, "Camera pose tracking" below; under that, the stored one,
  which the 30 % shadow margin covers).
* **Dark neutral flat objects** (a grey card) look like a shadow and are dropped by the component test; anything with
  internal contrast, colour, or a ratio below 0.4 (black) is kept.
* **Two touching objects** are one blob with one centroid.
* **Specks under 40 mm2** are ignored (a 3x3 mm2 blob is 2 pixels at 0.5 px/mm).
* **The empty-mat reference must be of the same mat print.** Turning the mat round changes the printed pattern under
  every pixel; learn a new background after moving the mat by more than the re-fit can explain.

## Fixtures (tests/scene/fixtures)

Real frames from 9 Nov 2026: `bg.jpg` (empty, arm parked at the top edge), `obj.jpg` (one M-size hex nut at
(10.97, 6.94) in, arm at the top), `top_now.jpg` (arm raised, same layout), each matching `calib_a.json`
(16:30 calibration), and `new.jpg` (mat turned round, markers re-glued, clean) matching `calib_b.json` (the 21:17
calibration that `jetson/calib/top_table.json` now holds). `bg.jpg` and `new.jpg` are therefore NOT a background/frame
pair: the tests use `bg.jpg` vs `top_now.jpg`/`obj.jpg`, `new.jpg` against itself, and synthetic squares
un-warped into `bg.jpg` through the inverse homography (`tests/scene/conftest.py: paste_square`).

## Server wiring (`dashboard/server.py`)

One `MatScene` is built from `CALIB_DIR/top_table.json` (`jetson/calib/` in the repo, `SO101_CALIB_DIR` to override) and
rebuilt whenever that file's mtime or size changes (the server's own write-back, below, excepted). A background thread
(`scene_loop`) runs `detect()` about twice a second on the newest JPEG the Jetson's ffmpeg MJPEG stream has already left in
memory for `/cam/top.jpg` (`cam_frames["top"]`, decoded with `cv2.imdecode`; the camera is never opened a second time), skips
the detect while the top camera is not live (no frame, or one older than 3 s), hands `detect()` the arm's silhouette from
the follower's joints when they are fresh ("Kinematic arm mask" above), calls `update_background(frame)` with the
module defaults on every 10th detect, and keeps the latest result in memory. No request handler ever waits for a detect.
cv2, numpy and `scene_detect` are imported lazily in that thread and in the handlers: without them the dashboard still
starts and every scene endpoint answers `{"ok": false, "err": "scene_detect unavailable: ..."}`.

| endpoint | returns |
|---|---|
| `GET /api/scene` | the `detect()` dict above (`arm_masked_by` included: `"kinematics"` while fresh follower joints drive the mask, `"edge"` otherwise) plus `ok: true`, `age_s` (seconds since it was computed), `background` (`"ok"` / `"missing"`), `calib_live`, `calib_live_at`, `shift_px`, `holding` (the self-healing status, below). `{"ok": false, "err": ...}` with no table calibration (`no table calibration yet (run so101-calib table)`), no live top frame (`top camera not live`, also when the camera drops out after a result) or no cv2. `Cache-Control: no-store`. |
| `POST /api/scene/background` | learns the empty mat: `set_background` from the live top frame (`{}`), or the per-pixel median of `{"frames": N}` frames taken ~0.5 s apart (N capped at 15, so the arm may be anywhere as long as it keeps moving), `save_background` to `CALIB_DIR/top_mat_bg.png` (loaded at start-up when present, so a restart does not forget the mat), one detect at once so the map updates immediately, and a LOG line. Answers `{"ok": true, "saved": path, "frames": n}`. |
| `GET /api/scene.png` | `render()` of the last detect (arm mask red, foreground green, marker squares yellow, objects boxed with id and inches, 1 in grid) as `image/png`, for checking by eye; JSON `ok: false` before the first detect. |

```
curl -s http://127.0.0.1:8765/api/scene | python3 -m json.tool
curl -s -X POST -d '{"frames": 5}' http://127.0.0.1:8765/api/scene/background
curl -s http://127.0.0.1:8765/api/scene.png > scene.png
```

## Self-healing calibration (the camera stand gets bumped)

The overhead camera hangs on a rickety stand the arm can reach, so the pixel-to-mat homography in `top_table.json` goes
stale without warning. `MatScene` already re-fits the homography from the markers in every frame (above); the server
turns that into a repair:

1. After each detect the server projects the layout's marker centres into the image through the stored solve and through
   the live one; the largest difference is `shift_px`. (MatScene only re-fits once the markers drift more than 3 mm, so
   `shift_px` is 0 while nothing has moved.)
2. When the re-fit is sound (`calib_ok`, at least 3 markers) and `shift_px > 3`, the live solve is what the dashboard
   works with: `GET /api/calib` serves it as `homography_px_to_table_mm` with `"live": true`, `"live_at"` (epoch of that
   re-solve), `"shift_px"`, and keeps the file's matrix under `"stored_homography"`; `/api/locate` converts pixels through
   it too; the top-feed overlay and the 3D stage follow (the stage's 2 Hz scene poll triggers `calibPoll()` the moment
   `calib_live_at` changes, instead of waiting for the 60 s calibration poll).
3. A LOG line (`overhead camera moved 11 px, re-solved from 4 markers (0.2 mm)`) is written to the dashboard's event stream,
   at most one per 30 s.
4. The live solve is written back into `top_table.json` at most once a minute (and only while `shift_px > 3`): an atomic
   replace that keeps everything else in the file, in particular `arm_base_mm`, `arm_yaw_deg`, `arm_fit` and `created`
   (the arm did not move when the camera did), replaces `homography_px_to_table`, sets `residual_mm_max` /
   `residual_in_max` to the re-fit's marker error, drops the old solve's per-marker `residual_mm` / `residual_in`, and adds a
   `live_solve` block (`at`, `shift_px`, `markers_seen`, `residual_mm_max`, `note`) that `/api/calib` echoes as `live_solve`
   so the CAL chip can say when it last re-solved. Nothing is copied to the Jetson by the dashboard (the orchestrator
   syncs it); the server's own write does not rebuild the `MatScene`, so object ids and the mat-move reference survive.
   Once the file holds the solve the next detect agrees with it and `live` goes back to `false`.
5. Fewer than 3 markers visible (a hand, the arm, a box over a marker): nothing to re-solve from, the last good solve
   (stored or live) stays in force and `/api/calib` and `/api/scene` report `"holding": true` with `markers_seen`.
6. `calib_ok` false (`mat_shift_mm > 150` or `|mat_rot_deg| > 30`: the mat itself moved or the markers were re-glued; or
   the markers are still more than 6 mm off after the re-fit): nothing is served live and nothing is written, and
   `calib_note` (`mat moved 43 mm / 178 deg since calibration (markers re-glued?): re-run cam_calib.py table`) is exposed on
   both endpoints for the page. Re-run `cam_calib.py table` (and the arm fit if the mat moved relative to the arm).

Thresholds live at the top of the scene section of `server.py` (`SCENE_SHIFT_PX`, `SCENE_NOTE_S`, `SCENE_PERSIST_S`,
`SCENE_BG_EVERY`, `SCENE_PERIOD_S`). The intrinsics path of `cam_calib.pixel_to_table` (rvec/tvec) is not touched by the
live solve: it applies to homography-only table calibrations, which is what `table` produces today.

### Camera pose tracking (the 3D stage follows the cameras)

The homography keeps the mat map right; the 3D stage also draws each camera where it is (`camera.pose` /
`camera.front.pose` from `/api/calib`, docs/camera-calibration.md section 6), and those poses went stale the same way. So the
server re-solves each camera's position and orientation from the markers whenever the camera moves, with the one thing that
does not move: **the field of view belongs to the lens, the pose to the mount.** `hfov_deg` in each pose
(`camera_pose_est` in `top_table.json`, `front_campose.json`) was pinned once by `cam_calib.py campose --lens-height` from a
tape-measured lens height (top 69.16 deg from 16.5 in, front 68.88 deg from 18.375 in, 9 Nov 2026) and is treated as a fixed
intrinsic: the live solve is `camera_pose_from_markers(gray, layout, nominal_K(w, h, hfov_deg))`, the marker corners alone.
It is never re-estimated by the dashboard; a pose without `hfov_deg` is served as it is and not tracked, with one LOG line
(`overhead camera pose has no hfov; run so101-calib campose --lens-height`, `front camera pose has no hfov; run so101-calib
campose --name front --lens-height`). Re-run `campose --lens-height` when the lens or the camera is replaced, or the
resolution changes, never because the mount moved.

**Overhead camera** (`scene_pose`, inside the scene loop). Whenever the live homography is accepted (step 2 above) the pose is
solved from the same frame, once per re-fit (cached against the live matrix, so a steady camera costs nothing more).
**Hysteresis**: the served pose is only replaced when the camera centre moved more than 5 mm or the optical axis turned more
than 1 degree (`POSE_MOVE_MM`, `POSE_TURN_DEG`) against the pose currently served, so solve noise never jitters the stage; a
4 px nudge re-fits the homography but leaves the frustum alone. The new pose carries `hfov_deg`, `image_size`, `created` and
`units_note` over from the stored one and marks `source` as `live marker re-solve, hfov from <the original source>`.
`/api/calib` then serves it as `camera.pose` with `camera.pose_live: true`, `camera.pose_at` (epoch) and
`camera.pose_moved_mm`; the stage rebuilds on its next calibration poll (the scene poll triggers it the moment
`calib_live_at` changes, so within a second). An event line `overhead camera pose re-solved: moved 9 mm` is written at most
once per 30 s, and the pose rides in the same rate-limited write-back of `top_table.json` as the homography, as
`camera_pose_est` (with a `live_solve.camera_pose` block: `at`, `moved_mm`, `markers_seen`); the arm fit and `created` stay.
Once the file holds it, `pose_live` goes back to `false` like `live` does. The kinematic arm mask reads the served pose, so
after a bump the silhouette is right again as soon as the pose is.

**Front camera** (`front_pose_step`, its own thread at ~1 Hz, `FRONT_POSE_PERIOD_S`). The front camera has no homography
to heal, only the pose in `front_campose.json`, so the loop decodes the newest front JPEG the Jetson stream left in memory
(skipped while the front feed is not live), detects the markers of the mat layout (from `top_table.json`) and, with at least
3 of them in view, solves the pose with the file's `hfov_deg` and the same hysteresis. `/api/calib` serves it as
`camera.front.pose` with the mat-surface homographies (`homography_table_to_px`, `homography_px_to_table_mm`) re-derived from
it, so the FRONT overlay and `/api/locate?cam=front` follow too, plus `pose_live` / `pose_at`, `live` / `live_at` /
`moved_mm` (mirroring the top's fields), `holding`, `markers_seen` and `live_solve`. Event line `front camera pose
re-solved: moved 42 mm` (one per 30 s); the live pose is written over `front_campose.json` at most once a minute
(`FRONT_POSE_PERSIST_S`, atomic, with a `live_solve` block: `at`, `moved_mm`, `markers_seen`, `note`). **Fewer than 3
markers** (the arm or the overhead stand often hides one or two in the oblique view; marker 1 sits past the frame edge in
the current aim): nothing to solve from, the last pose (file or live) stays and `camera.front.holding` is `true` with
`markers_seen`. The stage picks a new front pose up on its 60 s calibration poll.

Tests: `tests/server/test_pose_api.py` (the overhead camera from the real obj.jpg with a stored pose solved from the unmoved
frame: a 10 px bump moves the served pose ~8.5 mm with `pose_live`, the write-back carries `camera_pose_est` and keeps
`arm_fit`, a 0.5 px jitter and a 4 px sub-hysteresis nudge leave it alone, a pose without `hfov_deg` is not tracked and logs
once; the front camera from a frame synthesised by rendering the layout's markers through the real `front_campose.json`
pose with `cv2.projectPoints` + `warpPerspective`, which solves back within 5 mm / 1 deg: a 42 mm move is served live,
logged once, written once a minute, two markers hold the last pose, no `hfov_deg` logs once, the loop sleeps, nothing runs a
subprocess).

## Dashboard (`dashboard/index.html`)

* **3D stage.** The module polls `/api/scene` at 2 Hz (`scenePoll()`, global so the UI tests can drive it with the timers
  frozen) and draws every object in the `mat_frame` group as a `w_mm x h_mm x 8 mm` box at `(x_mm, y_mm)` (mm -> m from the
  mat corner), turned by `angle_deg` about the mat normal, in its mean colour with white edges, named `obj_<id>`, with a
  small `#id` sprite (`objtag_<id>`) beside it. An object missing from one poll is faded; missing from two in a row it is
  removed (so a flicker or the arm passing over does not blink the map). When the calibration changes and the mat group is
  rebuilt, the objects come back with the next poll. The TOP CAM frustum (`top_cam`) is not an object: it sits at
  `/api/calib`'s `camera.pose` in the mat group when there is one (docs/camera-calibration.md section 6), else over the
  mat centre at 0.5 m. A muted caption at the bottom of the stage (`#matcap`) says why the
  map is empty: `MAT MAP: NO BACKGROUND — tap SET BACKGROUND`, the `calib_note` when `calib_ok` is false, or
  `HOLDING THE LAST CAMERA SOLVE (n markers visible)`.
* **TOP feed.** A touch-sized `SET BACKGROUND` button (`#calbg`, top-right of the TOP · LIVE feed, clear of the CAL chip)
  POSTs `/api/scene/background` with `{"frames": 3}` and shows `SAVED` (or `FAILED`) for two seconds. Press it with the mat
  empty and the arm parked, or moving. The CAL chip (`#calchip`) gains `LIVE 11 px` while a re-solve is being served,
  `HOLDING` with fewer than 3 markers, and `CHECK MAT` when `calib_ok` is false (the note is in its tooltip, as is the
  time of the last write-back).

Tests: `tests/scene/test_arm_mask.py` (a blob pasted where the jogged (9, 7) gripper projects is an object without the
mask and arm with it, the same blob 80 mm away is reported, the nut survives an arm far from it, the no-camera-pose
fallback, the `arm_px` contract, the background never learns the masked arm); `tests/calib/test_arm_mask_geometry.py`
(`fk_chain` against `fk_tip`, `arm_to_table` round trips, the jogged tip through the camera pose within 6 px of the
homography's (9, 7), parallax direction and size, `arm_mask_px_from_joints` radii at height and the fallback);
`tests/server/test_scene_api.py` (fixture frames fed into `cam_frames` as JPEG: the nut in mat inches, background
learn / save / reload, the PNG, background adaptation, the shifted-frame live path with the single LOG line and the
write-back that keeps `arm_fit`, the holding path, the moved-mat path, fresh follower joints -> `arm_masked_by:
"kinematics"` while stale / raw / missing joints or a table without an arm pose fall back to `"edge"`, and that nothing
runs a subprocess);
`tests/ui/test_stage_scene.py` (`obj_<id>` at the expected world position via `__matToWorld` in the right colour, fade and
removal after two empty polls, the captions, the live-solve refresh, the chip parts, and the button's POST).
