# Overhead camera calibration

`jetson/cam_calib.py` (on the Jetson: `~/so101/cam_calib.py`, alias `so101-calib`) turns the overhead C920 into a
measuring instrument for the scripted jar-loading moves. Two results, both JSON under `~/so101/calib/`:

| file | what it holds | needed for |
|---|---|---|
| `top_intrinsics.json` | camera matrix, lens distortion, per-image error | undistorting, any pixel-to-mm at a height |
| `top_table.json` | camera pose over the table (or a flat homography without intrinsics), marker residuals, arm base | `locate` |

The learned policies (ACT, SmolVLA) do not use any of this. They learn pixels directly. Only the scripted path needs it.

## Frame sources

Every command that needs a picture takes `--source`:

- a `/dev/v4l/by-id/...` path or a camera index. The tool touches `~/so101/.cams_busy` while it has the camera so the
  Mac dashboard's stream lets go, and removes the flag when it exits. If the flag already existed it is left alone.
- a URL that returns a JPEG, such as the dashboard's `http://localhost:8765/cam/top.jpg` when running on the Mac.
- a folder of images, for offline re-runs. The tests work this way.

Default source is `$CAM_TOP` from `env.sh`, 640x360 MJPG, the same mode recording uses. Calibrate at the resolution
you will use. An intrinsics file for one size is refused for frames of another.

## 1. Print the targets

```
so101-calib board                       # calib/print_board.png, 9x6 inner corners, 25 mm squares
so101-calib markers --ids 0-7 --size-mm 40
```

Both are 300 dpi PNGs. Print at 100 percent, actual size, no fit-to-page, then measure one square and one marker with
calipers. If a square is not 25.0 mm, pass the measured value as `--square-mm` to `capture` and `intrinsics`, and put
the measured marker size in the layout file. Glue the board to something flat. A warped board is the usual cause of a
poor result.

## 2. Intrinsics

```
so101-calib capture --count 18          # auto-shoots whenever it sees the board from a new angle
so101-calib intrinsics                  # -> calib/top_intrinsics.json
```

Hold the board in front of the camera and keep moving: tilt it both ways, bring it close and far, push it into every
corner of the view. The tool only saves a frame when the board is found and has moved since the last shot, so you
cannot flood it with duplicates. The result prints RMS reprojection error and the worst image. Under 0.5 px is good,
above 1 px means reshoot or delete the worst captures from `calib/captures/` and rerun `intrinsics`.

## Units

The layout file takes `"units": "mm"` (default), `"cm"` or `"in"`. Every number in the layout is read in that unit:
marker centres, `marker_size`, `arm_base`. Results are always stored in mm and printed in both mm and the layout
unit (`residual_in_max`, `table_in`, `arm_in`, `error_in`). The Fiskars mat has an inch grid, so the lab layout is
in inches, origin at the mat corner on the arm's side, left, X along the 18 in edge, Y along the 12 in edge.

## 3. Table map

Tape four to eight markers to the mat, spread around the workspace, and measure each marker's centre in a table frame
you choose. Convention: X to the right and Y away from you as you stand at the operator's side, millimetres, origin at
a mat corner or a tape mark. Marker rotation does not matter because only centres are used.

`calib/top_layout.json`:

```json
{
 "dictionary": "DICT_4X4_50",
 "marker_size_mm": 40.0,
 "markers": {"0": [40, 40], "1": [360, 40], "2": [360, 260], "3": [40, 260], "4": [200, 30], "5": [200, 270]},
 "arm_base_mm": [200, 330],
 "arm_yaw_deg": -90
}
```

`arm_base_mm` is where the arm's base axis meets the table and `arm_yaw_deg` the direction its forward X axis points,
degrees counter-clockwise from table +X. With the arm at the far edge facing you, that is -90. Both are optional.

```
so101-calib table                       # -> calib/top_table.json and top_table_check.png
so101-calib check                       # new frame: markers seen vs layout, mm error per marker, table axes drawn
```

`residual_mm_max` is how far the worst marker lands from where the layout says it is. Under 2 mm is normal. Over
5 mm, re-measure that marker. Without an intrinsics file the tool falls back to a flat homography, which is only right
for points on the table surface.

## 4. Use it

```
so101-calib locate 412 188 --height 11
{"pixel": [412.0, 188.0], "height_mm": 11.0, "table_mm": [203.4, 151.2], "arm_mm": [178.8, -3.4]}
```

`--height` is the height above the table of the point you clicked on, in mm. For the top face of the Pack Tool box
that is the box height. Getting this wrong is the classic error: a 60 mm tall object read as if flat lands centimetres
off. From Python:

```python
import json, cam_calib
cal = json.load(open("calib/top_table.json"))
x, y = cam_calib.pixel_to_table(cal, u, v, height_mm=11)
ax, ay = cam_calib.table_to_arm(cal, x, y)
```

## 5. Where is the arm? (`fk` and `armfit`)

`locate` can only report arm-frame coordinates once the calibration knows where the arm base sits on the mat and
which way it faces. Guessing (`arm_base`, `arm_yaw_deg` in the layout) is fine to start. Measuring it is better:

1. Torque off (`so101-free`), move the gripper so the tip between the jaws sits exactly on a marker centre, and save
   the pose: `so101-mark m0`. Repeat for every marker (`m1`, `m2`, `m3`). Add a few extra points in the middle of
   the mat if you like (`so101-mark p5` with the mat coordinates written down).
2. Convert each pose to a tip position in the arm frame: `so101-calib fk --pose m0`. `tip_mm` is X forward, Y to the
   arm's left, Z up from the base plate. The forward kinematics come from the URDF and match the dashboard's 3D arm
   to 0.1 mm. `tip_in` is the same in inches.
3. Put the pairs in `calib/top_armfit.json`:

```json
{"arm_units": "in",
 "pairs": [{"marker": 0, "arm": [7.1, 6.9]}, {"marker": 1, "arm": [7.0, -7.0]},
           {"marker": 2, "arm": [12.4, 7.1]}, {"marker": 3, "arm": [12.3, -6.9]},
           {"table": [9, 6], "arm": [6.0, 0.0]}]}
```

4. `so101-calib armfit --pairs calib/top_armfit.json` fits a rigid rotation plus translation, writes `arm_base_mm`
   and `arm_yaw_deg` into `top_table.json`, and prints the residual per point. Under 3 mm is good. A "mirror image"
   error means one frame's Y axis points the wrong way; check the mat Y direction against the arm's left.

`so101-calib fk --reach` prints the farthest the tip can touch the table from the pan axis: with any wrist angle,
and with the gripper pointing down (within 25 degrees). Computed from the URDF with the base plate on the table.

### Joint zero offsets (why a plain rigid fit is not enough)

LeRobot reports degrees measured in physical motor ticks, but zeroed at the middle of each joint's calibrated range,
which is not exactly the URDF zero. A few degrees of offset in lift, elbow and wrist makes the kinematics read short
near the base and right far out, and a rigid fit of jogged points then leaves 15 to 30 mm residuals.
`fit_arm_and_offsets` (used by `tools/jog_capture.py --fit`) solves the base pose, the table height in the arm frame
and the three planar joint offsets together. On the real arm, 9 Nov 2026, six jogged points:

| quantity | value |
|---|---|
| base in mat frame | (8.81, -3.05) in, heading 89.3° |
| mat surface | 4 mm above the base_link origin |
| offsets | lift -3.9°, elbow -5.2°, wrist +-0.7° |
| fit residual | 0.3 to 4.1 mm in the plane, under 3 mm in height |
| blind prediction | 1.5 mm at (12, 7), 5.6 mm at the centre |

The offsets live in `top_table.json` under `arm_fit` and `fk` applies them automatically (`--raw` to see the
uncorrected kinematics). Any future IK or scripted move must apply them too.

### Jogging with `tools/jog_capture.py` (Mac)

Run teleop so the follower streams its joints to the dashboard, drive the closed jaw tip onto a known mat point,
then on the Mac: `tools/jog_capture.py --marker 0` (or `--table 9 6`). `--show` prints the tip without recording,
`--list`, `--drop 0`, and `--fit --hold-out 9,6` refits while predicting the named points blind. Count mat
coordinates from the markers, not from the mat's printed grid: the Fiskars labels run from the opposite corner.

## 6. Dashboard endpoints (read-only)

The Mac wall dashboard (`dashboard/server.py`, port 8765) serves the calibration to its front end from the repo copies
of the Jetson's files (`jetson/calib/top_table.json` + `top_layout.json`, synced after each calibration; override the
folder with `SO101_CALIB_DIR`). The files are re-read whenever they change. Nothing here talks to the Jetson or moves
the arm; `/api/ik` is a dry-run calculator. All three answer HTTP 200 with JSON; a failure is `{"ok": false, "err": ...}`.

| endpoint | needs | returns |
|---|---|---|
| `GET /api/calib` | stdlib only | `units`, `mat {width_in, depth_in}`, `image_size`, `marker_size_in`, `layout_markers_in {id: [x, y]}`, `homography_px_to_table_mm` (3x3 or null), `has_intrinsics`, `camera {residual_mm_max, created, markers_used, pose, pose_live, pose_at, pose_moved_mm, front}`, `arm {base_mm, base_in, yaw_deg, table_z_mm, joint_offsets_deg, wrist_roll_deg, fitted, residual_xy_mm_max, points}` (null before an arm fit), `files {table, layout, table_mtime}`, and the self-healing fields `stored_homography`, `live`, `live_at`, `shift_px`, `holding`, `calib_ok`, `calib_note`, `markers_seen`, `live_solve` (below) |
| `GET /api/locate?u=<px>&v=<px>&h_mm=<0>&cam=<top>` | cam_calib (cv2 + numpy) for `top`; stdlib for `front` | `pixel`, `height_mm`, `table_mm`, `table_in`, `arm_mm`, `arm_in` (arm fields null without an arm pose); `cam=front` adds `cam: "front"` |
| `GET /api/ik?x_in=<mat x>&y_in=<mat y>&h_in=<0>&roll=<deg>` | cam_calib + scipy | `target_mm [ax, ay, z]`, `joints_deg` (LeRobot degrees, offsets applied), `tip_mm`, `pos_err_mm`, `tilt_deg`, `wrist_roll_deg` |

`/api/ik` takes a mat point in inches, `h_in` above the mat surface; `z = arm.table_z_mm + h_in * 25.4`, the fitted
joint offsets are applied, and `roll` defaults to `arm.wrist_roll_deg` (the roll the fit was jogged at, which matters
because the fixed jaw sits off the roll axis). Unreachable targets come back `ok: false` with the `CalibError` text.
Without a table file every endpoint answers `{"ok": false, "err": "no table calibration yet (run so101-calib table)"}`.
`camera.pose` passes `camera_pose_est` through when `cam_calib.py campose` has written one (`position_mm` = the camera
centre in the table frame, z up from the mat, plus `position_in`; `optical_axis` and `up_table` unit vectors in the table
frame; `tilt_deg`, `axis_hits_mm`, `hfov_deg`, `image_size`, `reproj_rms_px`, `source`, `created`), `null` otherwise. The 3D
stage puts the TOP CAM frustum there, looking along the axis, sized by `hfov_deg` at 16:9, labelled "TOP CAM est." while
`source` says nominal (the lens height is not tape-measured yet). While the camera has moved since the file was written,
`camera.pose` is the dashboard's live marker re-solve instead (`camera.pose_live: true`, `camera.pose_at` epoch,
`camera.pose_moved_mm`; `source` reads `live marker re-solve, hfov from <the original source>`): "Live camera poses" below.

### The front camera (`camera.front`, `cam=front`)

The oblique FRONT camera (42° tilt, past the far edge of the mat) is not a table calibration: it only has a solved pose,
`jetson/calib/front_campose.json`, written by `so101-calib campose --name front --lens-height 18.375 --units in` on the Jetson
(marker corners + the tape-measured lens height, like the top camera's `camera_pose_est`, but saved in its own file because
there is no `front_table.json`). `/api/calib` serves it as `camera.front`:

| field | meaning |
|---|---|
| `pose` | the file passed through like `camera.pose` (`position_mm`, `rvec`, `tvec`, `camera_matrix`, `optical_axis`, `up_table`, `tilt_deg`, `axis_hits_mm`, `hfov_deg`, `image_size`, `reproj_rms_px`, `source`, `created`) plus `position_in`; the live marker re-solve instead while the camera has moved since the file was written (`pose_live`) |
| `homography_table_to_px` | `K [r1 r2 t]` with `R = Rodrigues(rvec)`, `t = tvec`: mat mm on the surface (z = 0) -> raw 640x360 px (derived from whichever pose is served) |
| `homography_px_to_table_mm` | its inverse, scaled so `[2][2] = 1` (a positive scale: the third row keeps the sign of 1/depth) |
| `reproj_rms_px`, `created` | from the served pose / the file |
| `pose_live`, `pose_at`, `live`, `live_at`, `moved_mm` | the live-pose state, `live` / `live_at` / `moved_mm` mirroring the top's `live` / `live_at` / `shift_px`: `true` + epoch + mm moved while the dashboard's re-solve is served, `false` / null otherwise ("Live camera poses" below) |
| `holding`, `markers_seen` | fewer than 3 layout markers in the last front frame (the arm or the overhead stand in the way): the last pose stays; how many were seen (null before the first front detect) |
| `live_solve` | the block the dashboard wrote into `front_campose.json` with its last write-back (`at`, `moved_mm`, `markers_seen`, `note`), null if it never did |

`camera.front` is `null` while the file is absent or unusable (no `rvec`/`tvec`/`camera_matrix`, singular matrix); the rest of
`/api/calib` is unchanged by it. Everything is derived in pure Python, so no cv2 is needed on the Mac. Only the mat SURFACE is
mapped: `GET /api/locate?cam=front&u=&v=` answers the mat point under that front pixel on the surface (same fields as the top
answer plus `cam: "front"`; `h_mm` must be 0, anything else is refused with `ok: false`; the arm fields come from the top table
file's arm pose and are null without it; a pixel above the horizon is refused). Without the file: `no front camera pose yet`.

The wall page draws the same calibration overlay on the FRONT feed as on the TOP feed (1-inch grid, mat outline, axes, the four
markers with ids, the arm base, tap-to-measure with a MAT / ARM readout confirmed via `cam=front`) through that homography, with
no assumption about the view's orientation; its chip reads `FRONT · <reproj_rms_px> px` (`UNCALIBRATED` without the pose) and
there is no SET BACKGROUND button on it. The 3D stage places the orange FRONT CAM frustum (`front_cam`) in the mat group at
`position_mm`, looking along `optical_axis`, sized by `hfov_deg`, with a dashed ray to `axis_hits_mm`; without a pose it keeps
its old static placement past the far edge.

```
curl -s 'http://127.0.0.1:8765/api/calib' | python3 -m json.tool
curl -s 'http://127.0.0.1:8765/api/locate?u=320&v=180&h_mm=0'
curl -s 'http://127.0.0.1:8765/api/locate?cam=front&u=320&v=180'
curl -s 'http://127.0.0.1:8765/api/ik?x_in=9&y_in=5&h_in=0'
```

Tests: `tests/server/test_calib_api.py` (temp calib dirs plus the checked-in real table file as a fixture; `camera.front` against
`cv2.projectPoints`, `cam=front` round trips), `tests/server/test_scene_api.py` (the live-solve fields); the overlays and the
stage: `tests/ui/test_feed_overlay.py`, `tests/ui/test_front_overlay.py`, `tests/ui/test_stage_calib.py`, `tests/ui/test_stage_front.py`.

### Self-healing fields (the camera moved)

The dashboard's mat-scene thread (docs/mat-scene.md) detects the ArUco markers in every overhead frame and re-fits the
homography when they have drifted. `GET /api/calib` reports that state next to the usual fields; the existing fields keep
their shapes, only `homography_px_to_table_mm` may carry the live solve instead of the file's:

| field | meaning |
|---|---|
| `stored_homography` | the matrix in `top_table.json` (always) |
| `live` | `true` while the markers sit more than 3 px from where the stored solve puts them and the re-fit is sound (`calib_ok`, >= 3 markers): `homography_px_to_table_mm` is then the live re-solve and `/api/locate` uses it; `false` otherwise (`homography_px_to_table_mm == stored_homography`) |
| `live_at` | epoch seconds of that re-solve (null when not live) |
| `shift_px` | the largest marker-centre displacement between the stored and the live solve, px (null when not live) |
| `holding` | fewer than 3 markers were visible in the last frame: the last good solve stays in force |
| `calib_ok`, `calib_note` | MatScene's verdict on the last frame; `calib_ok` false with a note means the mat itself moved (> 150 mm / 30 deg) or the markers are off even after the re-fit: nothing is served live or written, re-run `table` (null before the first detect) |
| `markers_seen` | layout markers detected in the last frame (null before the first detect) |
| `live_solve` | the block the dashboard wrote into `top_table.json` with its last write-back (`at`, `shift_px`, `markers_seen`, `residual_mm_max`, `note`), null if it never did |

The write-back happens at most once a minute, keeps `arm_base_mm` / `arm_yaw_deg` / `arm_fit` / `created` and is logged
(`overhead camera moved 11 px, re-solved from 4 markers (0.2 mm)`, one line per 30 s); the Jetson copy is synced by the
orchestrator, not by the dashboard. `/api/scene`, `/api/scene.png` and `POST /api/scene/background` are described in
docs/mat-scene.md.

### Live camera poses (the 3D stage follows the cameras)

The same marker detections re-solve each camera's **pose** (position and orientation in the table frame, what the stage's
frustums are placed from) whenever the camera moves, with `hfov_deg` held fixed. **Fixed-FOV rule:** the field of view
belongs to the lens and is pinned once by `campose --lens-height` (a tape-measured lens height: top 69.16 deg at 16.5 in,
front 68.88 deg at 18.375 in); the mount moving changes the pose, not the FOV, so the dashboard solves
`camera_pose_from_markers(gray, layout, nominal_K(w, h, hfov_deg))` and never re-estimates `hfov_deg`. A pose file without
`hfov_deg` is served as it is and not tracked (one LOG line: `... pose has no hfov; run so101-calib campose --lens-height`).
Re-run `campose --lens-height` only when the lens or the camera is replaced or the resolution changes, not when the mount moves.

| field | meaning |
|---|---|
| `camera.pose_live` | `true` while `camera.pose` is the dashboard's re-solve (the camera moved > 5 mm or turned > 1 deg against the stored pose; smaller moves are ignored so noise does not jitter the stage) and the file does not hold it yet |
| `camera.pose_at` | epoch seconds of that re-solve (null when not live) |
| `camera.pose_moved_mm` | how far the camera centre moved from the pose it replaced (null when not live) |
| `camera.front.pose_live`, `camera.front.pose_at` | the same for the front camera (`camera.front.live` / `live_at` / `moved_mm` repeat them in the top's naming) |

The overhead pose is solved in the scene loop whenever the live homography is accepted and travels with the same write-back
(`camera_pose_est` in `top_table.json`, plus `live_solve.camera_pose {at, moved_mm, markers_seen}`); event line `overhead
camera pose re-solved: moved 9 mm` (one per 30 s). The front pose is solved by its own ~1 Hz loop on the live front feed
(`front_campose.json` written at most once a minute with a `live_solve` block; `front camera pose re-solved: moved 42 mm`);
with fewer than 3 markers in the front view the last pose is held (`camera.front.holding`). Details, thresholds and the
tests: docs/mat-scene.md, "Camera pose tracking".

## 7. Wrist camera (hand-eye)

The icSpring UVC module (640x480) is bolted to the gripper, so its pose is fixed in the URDF `gripper_link` frame (the link after
the wrist_roll joint; the fingers point along its -Z). `so101-calib handeye` measures that pose from a few wrist-camera frames of
the mat markers plus the follower's joints at each frame, through the arm's forward kinematics: it is the classic hand-eye problem
with one camera and a known target. The result, `calib/wrist_campose.json`, is what the 3D stage uses to draw the WRIST CAM frustum
on the follower (`/api/calib` serves it as `camera.wrist`).

### Procedure

1. The table calibration and the arm fit must be current (`top_table.json` with `arm_base_mm`, `arm_yaw_deg`, `arm_fit.table_z_mm`
   and the joint offsets): the markers' positions in the arm frame come from them, so any error there goes straight into the wrist pose.
2. Pose the follower so the wrist camera sees ONE marker whole (the quiet zone too), from 120 to 250 mm away, and capture:
   `tools/jog_capture.py --wristcam w1` (saves `jetson/calib/wristcam/w1.jpg` + `w1.json` with the follower joints and prints the
   marker ids it sees; `enough` refers to the overhead-style 3-marker solve and does not matter here, one marker per view is the design).
3. Take at least 6 views, better 8 to 10, over different markers, and vary everything between them: height (close and far), tilt of
   the gripper (10 to 40 degrees off vertical, in different directions), wrist roll (spread over +-40 degrees) and pan. Views that
   only look straight down from the same height leave the camera's yaw about the gripper axis and its distance along the optical
   axis weakly determined. Keep the follower still when you capture: torque on, let it settle for a second, then shoot. A hand-held
   leader or a frame grabbed while the arm is still moving puts a degree of error into that view, which is 5 to 15 px at 150 mm.
4. `so101-calib handeye` (defaults: views `calib/wristcam`, layout `top_layout.json`, table `top_table.json`; `--views`, `--layout`,
   `--table`, `--out` override). It skips views without a complete layout marker, ignores foreign ids, prints a JSON summary and
   writes `calib/wrist_campose.json`. `so101-calib handeye --check w3` writes `calib/wrist_check_w3.png`: the detected marker in
   green, every layout marker reprojected through the arm and the solved pose in red, so you can see the fit with your own eyes.
   A previous `wrist_campose.json` is used as a starting point (`--fresh` to ignore it).

### What is solved

Unknowns: X = T_gripper_link<-camera (rotation vector + translation, 6), the focal length (fx = fy, principal point fixed at the
image centre; `--free-centre` frees it), the radial distortion k1, k2 of the OpenCV model, each marker's in-plane glue angle (the
90 degree step is chosen automatically, the small remainder is fitted with a prior of 0; `--overhead <frame>` measures the angles
through the overhead camera's table calibration instead), and a small rotation correction per view with a prior of
`--pose-sigma-deg` (default 1; 0 trusts the joints completely). The cost is the reprojection error of the marker corners
(soft-L1, 2 px scale), minimised with scipy from 96 camera placements around the gripper and several focal lengths, the best basins
refined. With a checkerboard calibration of the wrist camera (`so101-calib --name wrist capture/intrinsics` at 640x480,
then `handeye --intrinsics calib/wrist_intrinsics.json`) the camera model is fixed and only X is solved, which also removes the
trade-off between fx and the distance along the optical axis that single-marker views cannot break on their own.

The file holds `position_mm` (camera centre in gripper_link), `rotation` (`rvec`, `quaternion_xyzw`, `optical_axis_gripper`,
`up_gripper` = image up, all in gripper_link), `camera_matrix`, `dist_coeffs`, `image_size`, `hfov_deg`/`vfov_deg`,
`off_axis_mm`, `tilt_from_gripper_axis_deg`, and the quality: `reproj_rms_px` and `per_view_px` (RIGID reprojection from the saved
pose, what the stage or any consumer of the file will see), `reproj_rms_px_corrected` (with the per-view corrections, how well the
model explains the corners), `view_pose_corrections_deg` (which views were not still), `leave_one_out_px` (fit without each view,
predict its corners rigidly: the honest accuracy) and `leave_one_out_shift` (how far the mount estimate moves without that view).

### The 9 Nov 2026 result

Five usable views (w1, w2, w3, w5, w6; w4 showed no whole marker), markers 1, 2, 3:

| quantity | value |
|---|---|
| camera centre in gripper_link | (1.2, 67.9, -11.3) mm: 68 mm off the gripper axis (the STL mount says 69), 11 mm toward the wrist |
| optical axis | (-0.09, -0.49, -0.86): 30 degrees from the gripper axis, back toward it, looking past the fingertips |
| image up | toward the fingertips (the jaws enter the frame from the bottom) |
| focal length | 529 px, hfov 62 degrees, k1 -0.41 (strong barrel), k2 0.15 |
| corner fit with the per-view corrections | 0.5 px rms; corrections 0.4 to 1.4 degrees per view |
| rigid reprojection | 8 px rms (3 to 11 px per view) |
| leave-one-out | 4 to 16 px; mount moves 1 to 3 mm and 0.2 to 1.5 degrees without any one view |
| glue angles | marker 1 at -89, 2 and 3 at 181/182 degrees (the overhead camera says -90.5, 180, 180) |

So the mount is known to a few millimetres and about 1.5 degrees, good for the stage's frustum, but each view's joints are only good
to about a degree, which is why a rigid reprojection lands 5 to 15 px off. More views captured with the follower still (and a
checkerboard intrinsics file) are what will bring the rigid error toward the 2 to 3 px the corners support. A pan-offset unknown and a
fisheye lens model were tried on this data and change nothing.

### Redo when the mount changes

Any time the camera, its bracket or the gripper's printed parts are touched: re-capture 6+ views (the old `wristcam/*.{jpg,json}`
can be moved aside or overwritten; views are matched by file name), rerun `handeye` (`--fresh` if the mount moved a lot), look at a
`--check` image, sync `wrist_campose.json` to the repo copy the dashboard reads. Redo it as well after a new arm fit (the marker
positions in the arm frame changed) and after re-gluing the markers. Tests: `tests/calib/test_handeye.py` (a synthetic wrist camera at
a known pose on the real kinematics must come back within 2 mm / 0.5 degrees / 2 percent fx, the real captures must solve, the CLI
and the check image), `tests/ui/test_stage_wristcam.py` (the frustum on the stage).

## Redo when

- the camera or its mount moves: with a homography-only table the running dashboard re-solves it from the markers and writes it back (section 6), and re-solves the camera pose with the same markers (fixed FOV); rerun `table` when the dashboard was not running, when `calib_ok` stays false, or for an intrinsics calibration (intrinsics stay valid). `campose --lens-height` does NOT need re-running for a moved mount,
- the mat moves relative to the arm or the markers are re-glued (rerun `table`, then the arm fit: the dashboard refuses to heal that),
- the lens or resolution changes, or a camera is replaced (rerun everything, `campose --lens-height` included: that is the only time the FOV changes),
- `check` reports an error that has grown past a few mm.

## Tests

`tests/calib/` renders checkerboards and ArUco tables through a synthetic camera with known intrinsics, distortion and
pose, then checks what the tool recovers: lens model within 1.5 px over the covered frame, camera position within 6 mm,
pixel to table within 1.5 mm at several heights, inch layouts, the rigid arm fit (recovers base and heading, refuses a
mirror), forward kinematics against the dashboard URDF model, plus the CLI end to end and every error message.

```
.venv/bin/python -m pytest tests/calib -q
```
