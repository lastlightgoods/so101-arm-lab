# so101-arm-lab: fact sheet

Every fact below is taken from the repository; the source file is in parentheses. Dates are 2026. The
project started on 5 October 2026 (NEXT_STEPS.md, README.md) and this sheet was compiled on 10 November
2026, day 37. The person running the lab is TC, of Lastlight Goods in Bend, Oregon (README.md).

## 1. Hardware

- Arm: a prebuilt SO-ARM101, leader plus follower, six Feetech STS3215 servos per arm, ids 1-6,
  firmware 3.15, bus at 1 Mbaud (NEXT_STEPS.md "Status 5 Oct").
- Bus boards: Waveshare boards with a CH343 USB serial chip; the follower runs on 12 V, the leader on
  5-7.4 V (NEXT_STEPS.md "Old checklist"; jetson/jetson_setup.sh udev section). The boards need
  USB-A-to-C cables; a C-to-C cable dropped one board off the bus (NEXT_STEPS.md).
- Software: LeRobot 0.6.1 on both machines. The Mac runs Python 3.12 and trains on Apple MPS (an M1
  Max); the Jetson runs Python 3.10 with a syntax shim, `jetson/py310_compat.py`, that rewrites PEP
  695 syntax and `typing.Self` imports so LeRobot 0.6.1 loads on 3.10 (NEXT_STEPS.md, README.md).
- Edge computer: an NVIDIA Jetson Orin Nano developer kit, hostname `tc-nano`, JetPack 6.2.x (Ubuntu
  22.04, CUDA 12.6, torch 2.11 CUDA wheels), 8 GB of shared memory, 25 W power mode by default
  (jetson/JETSON_RUNBOOK.md, jetson/jetson_setup.sh, NEXT_STEPS.md "Jetson training gotchas"). The
  firmware was already 36.4.3, so the JetPack 5.1.3 bridge step was skipped (jetson/JETSON_RUNBOOK.md
  section C, 14 Oct).
- Division of labour: the Jetson owns the arms and the cameras (teleop, record, rollout); the Mac
  trains, keeps datasets and checkpoints, and runs the wall dashboard, reaching the Jetson over SSH
  (README.md, dashboard/server.py docstring).
- Cameras (jetson/env.sh, docs/camera-health.md):
  - top: Logitech C920, USB id 046d:08e5, serial 9D46D6AF, 640x360 MJPG at 30 fps, overhead;
  - front: a second Logitech C920, serial B6A1F66F, 640x360 MJPG at 30 fps, facing the arm from the
    far side at an oblique 42 degree tilt (jetson/calib/front_campose.json);
  - wrist: an icSpring UVC module, USB id 32e6:9005, 640x480 YUYV, bolted to the gripper on the
    arm's left as you face it from the front of the mat (NEXT_STEPS.md 28 Oct, jetson/99-so101-cams.rules).
  - The C920s arrived on 28 Oct; before that the overhead camera was an icSpring (NEXT_STEPS.md 28 Oct).
- Work surface: a Fiskars 18x12 in self-healing cutting mat with a printed inch grid
  (jetson/calib/top_layout.json, README.md).
- Gripper limits: the STS3215 trips its overload protection when it sits stalled at its torque limit
  for `Protection_Time` (2 s). Measured stall current against torque limit: 500 -> 541, 400 -> 370,
  300 -> 240, 250 -> 187. The busheal plugin sets the limits after every connect (NEXT_STEPS.md 6 Oct).
  Grippers were recalibrated on 29 Oct: follower range 1395-2865 ticks, leader 1384-2622 (NEXT_STEPS.md).
- A USB lesson from day one: both servo boards and both original cameras hung off one USB 2.0 hub;
  opening any camera stream stalled the CH343 serial device within about 0.3 s. Fix: servo boards on a
  different USB controller from the cameras (NEXT_STEPS.md "ROOT CAUSE of bus dropouts", 5 Oct 22:20).

## 2. Timeline

(NEXT_STEPS.md section headings and dated entries; `git log` for 9-10 Nov.)

- 5 Oct: arms answer on the bus, port mapping confirmed by a wiggle test, bus-dropout root cause found.
- 6 Oct: gripper "quits mid-grasp" traced to the STS3215 overload trip and fixed in the plugin.
- 7 Oct: first dataset, `bottle-pick`: 50 raw episodes, 40 kept, 28,288 frames; ACT trained 50k steps
  at batch 8 on MPS. Second dataset `jar-pocket`: 22 raw, 18 kept, 16,295 frames.
- 14 Oct: Jetson imaged with JetPack 6.2.1; the wall dashboard is born (gauges, live feeds, an embedded
  Jetson terminal, the 3D arm from the URDF).
- 15 Oct: `lighter-handoff` dataset, 30 teleop episodes, 16 kept, 7,349 frames. Synchronous inference
  reached only 13 Hz on the Jetson; the async (rtc) path holds 30 Hz. A policy that froze at rest was
  traced to 1-5.6 s of idle time at the start of 13 of 16 episodes.
- 16 Oct: dashboard turntable header, the self-playing Tempest attract screen, the retinal-scan lock,
  and a SmolVLA fine-tune in the cloud (10k steps, 61 minutes, about $2.55).
- 18 Oct: first git commit; 3D stage gets the mat, the camera frustums and solid follower meshes.
- 19-21 Oct: simulation rounds (section 9). Best sim policy reached 84% over 100 unseen layouts.
- 28 Oct: two C920s installed as top and front cameras; three-camera recording config.
- 29 Oct: grippers recalibrated, teleop repaired after the calibration JSON drifted from servo EEPROM.
- 9 Nov: the calibration day. Arm base jog-and-fit with joint zero offsets, camera settings lock plus
  guard and udev rule, camera health and drift tools, kinematic arm mask, live camera poses, the
  wrist camera hand-eye solve, and the single-session waypoint runner `pick_seq.py`.
- 10 Nov: mat-map persistence filter, wrist pose served to the 3D stage, and the iterative
  pick-and-place learner `tools/pick_loop.py` with its first four runs.

## 3. The wall dashboard

- `dashboard/server.py`: a stdlib HTTP server plus psutil on port 8765. It polls the Jetson over SSH
  (a `tegrastats` stream every 2 s, a slow status probe, `arm_state.py` for joints, ffmpeg MJPEG for
  the camera feeds), reads local datasets, training logs and checkpoints, and never opens a camera
  itself (dashboard/server.py docstring, NEXT_STEPS.md 14 Oct). A `getfqdn()` stall of 35 s on the Mac
  was fixed by overriding `server_bind` (NEXT_STEPS.md).
- `dashboard/index.html`: six gauges (GPU, memory bus, CPU, RAM, temperature, power) drawn on canvas
  from the tegrastats stream, with the bus gauge needing root `tegrastats` for the EMC figure
  (dashboard/index.html `refresh()`); live feeds from the three cameras; an embedded Jetson terminal
  (ttyd); ACT loss curves from Mac and Jetson overlaid; dataset and policy tables; an event log.
- The 3D arm: three.js 0.160 plus urdf-loader 0.12.6 render TheRobotStudio's `so101_new_calib.urdf`
  with its 13 STL meshes (16 MB under `dashboard/urdf/`). URDF zero equals LeRobot's calibrated zero,
  so degrees go straight in. Follower solid warm white with amber edges, leader as an orange ghost
  (NEXT_STEPS.md 14 and 18 Oct, dashboard/index.html importmap).
- The 3D stage: the real 18x12 in mat with its four ArUco markers at the measured layout, the TOP CAM
  and FRONT CAM frustums at their solved poses, the WRIST CAM frustum on `gripper_link` from the
  hand-eye file, and the live mat map: every detected object as a `w x h x 8 mm` box in its mean
  colour with an id sprite, faded after one missed poll and removed after two (docs/mat-scene.md
  "Dashboard"; docs/camera-calibration.md sections 6 and 7).
- Calibration overlays on the TOP and FRONT feeds: 1 inch grid, mat outline, axes, markers with ids,
  the arm base, tap-to-measure with a MAT / ARM readout, a CAL chip that says `LIVE n px`, `HOLDING` or
  `CHECK MAT`, and a SET BACKGROUND button (docs/camera-calibration.md section 6, docs/mat-scene.md).
- Read-only APIs: `/api/calib`, `/api/locate?u=&v=&h_mm=&cam=top|front`, `/api/ik?x_in=&y_in=&h_in=&roll=`,
  `/api/scene`, `/api/scene.png`, `POST /api/scene/background`, `/cam/top.jpg`, `/cam/front.jpg`,
  `/cam/wrist.jpg` (dashboard/server.py docstring, docs/mat-scene.md).
- Theatre: a turntable header whose tonearm drops whenever a LeRobot process runs, a self-playing
  Tempest in the centre strip, a retinal-scan lock that watches the wrist camera for an iris with
  HoughCircles ("theater, not security"), Spotify transport buttons, and spoken events through macOS
  `say` (NEXT_STEPS.md 16 Oct, dashboard/server.py `say_loop`).
- Skins (dashboard/skins/README.md): `tempest` (default, amber and cyan), `missioncontrol` (1969
  flight-controller console, cream phosphor, needle meters, teletype log), `nostromo` (green CRT,
  scanlines, motion-tracker attract screen), and `dystopian` (the Lastlight Goods DYSTOPIAN sub-brand:
  silver only, never gold, JetBrains Mono, facility signage, hazard striping). Selected with
  `?skin=name`, remembered in localStorage, cycled with the `k` key. Colours are CSS tokens on `:root`,
  so every gauge and sparkline recolours for free.
- The kiosk runs portrait at 1440x2560 on a second monitor; the compact layout fits in 2560 px exactly
  (NEXT_STEPS.md 14 and 18 Oct; dashboard/skins/README.md "Orientation").
- Dashboard mode buttons POST `/api/arm/<teleop|policy|vla|dance|wave|replay|stop>`; a KILL button
  runs `so101-stop`, which now covers every motion script and leaves the arm frozen with torque held
  (NEXT_STEPS.md 16 Oct; git log 9 Nov "KILL button").

## 4. Camera calibration

(jetson/cam_calib.py, docs/camera-calibration.md, jetson/calib/*.json, tools/jog_capture.py)

- `cam_calib.py` subcommands: `board`, `markers`, `capture`, `intrinsics`, `table`, `check`, `locate`,
  `fk`, `armfit`, `campose`, `handeye`, `ik`; alias `so101-calib` on the Jetson.
- Intrinsics: a printed 9x6 inner-corner checkerboard with 25 mm squares, 300 dpi, measured with
  calipers after printing; `capture --count 18` auto-shoots only when the board has moved; an RMS
  reprojection error under 0.5 px is good, above 1 px means reshoot (docs/camera-calibration.md 1-2).
- The learned policies (ACT, SmolVLA) use none of this; only the scripted path does
  (docs/camera-calibration.md intro).
- Table map: four 2 in (50.8 mm) ArUco markers from `DICT_4X4_50` glued to the mat at inch positions
  0:(16,10), 1:(2,10), 2:(15,4), 3:(3,4); origin at the arm-side left corner, X along the 18 in edge,
  Y away from the arm (jetson/calib/top_layout.json). The mat was re-squared on the evening of 9 Nov
  and the marker centres read off the printed grid by TC (top_layout.json `_note`).
- Current table solve, 9 Nov 23:06: a flat homography (no intrinsics file yet), 16 corner points, worst
  marker residual 0.19 mm (0.0073 in) (jetson/calib/top_table.json). Under 2 mm is normal; over 5 mm,
  re-measure that marker (docs/camera-calibration.md 3).
- `locate` turns a pixel plus a height above the mat into mat and arm coordinates. Getting the height
  wrong is the classic error: a 60 mm tall object read as flat lands centimetres off
  (docs/camera-calibration.md 4).
- Arm base by jog-and-fit: torque off, put the closed jaw tip on known mat points, record the joints
  with `tools/jog_capture.py`, convert through the URDF forward kinematics (which match the dashboard's
  3D arm to 0.1 mm), and fit. The current fit (9 Nov 21:49) used four points at mat (9,3), (6,5), (9,7),
  (12,5): base at (229.6, -80.8) mm, i.e. (9.04, -3.18) in, heading 90.15 degrees, mat surface 3.28 mm
  above the `base_link` origin, RMS 0.69 mm, planar residuals 0.71-1.09 mm (jetson/calib/top_table.json
  `arm_fit`, jetson/calib/top_armfit.json). An earlier six-point fit gave (8.81, -3.05) in, heading
  89.3 degrees, residuals 0.3-4.1 mm, blind prediction 1.5 mm at (12,7) and 5.6 mm at the centre
  (docs/camera-calibration.md "Joint zero offsets").
- Joint zero offsets: LeRobot zeroes each joint at the middle of its calibrated range, not at the URDF
  zero, so a rigid fit of jogged points left 15-30 mm residuals. `fit_arm_and_offsets` solves base pose,
  table height and three planar offsets together: shoulder_lift -3.9 degrees, elbow_flex -5.2 degrees,
  wrist_flex -0.7 degrees. `fk` and `/api/ik` apply them automatically; any future IK must too
  (docs/camera-calibration.md; jetson/calib/top_table.json `joint_offsets_deg`). Once fitted, the
  offsets are held fixed and three points suffice to refit after the mat or arm moves (git log 9 Nov
  "arm fit: hold fitted joint offsets fixed").
- Wrist-roll rule: the fixed jaw sits off the roll axis, so the contact point moves with wrist roll.
  The fit was jogged at wrist_roll 0.46 degrees (spread 2.11); jog and command with the roll near that
  value, and `/api/ik` defaults `roll` to it (jetson/calib/top_table.json `arm_fit.note`,
  docs/camera-calibration.md section 6).
- Camera poses with a fixed field of view: the FOV belongs to the lens and is pinned once from a
  tape-measured lens height; the mount moving changes only the pose. Top: lens 16.5 in up, hfov 69.16
  degrees, camera centre (219.5, 160.5, 419.1) mm in the mat frame, tilt 1.87 degrees, reprojection
  0.59 px. Front: lens 18.375 in up, hfov 68.88 degrees, centre (-7.6, 354.0, 466.7) mm, tilt 41.68
  degrees, reprojection 1.91 px from 3 markers (jetson/calib/top_table.json `camera_pose_est`,
  jetson/calib/front_campose.json).
- Self-healing when the stand is bumped (docs/mat-scene.md "Self-healing calibration",
  docs/camera-calibration.md section 6):
  - every overhead frame has its four markers detected (2.6 ms); when they land more than 3 mm from
    the layout under the stored homography and at least 3 are visible, a fresh homography is fitted;
  - when the largest marker displacement `shift_px` exceeds 3 px the live solve is served by
    `/api/calib` and used by `/api/locate`, the overlays and the 3D stage;
  - one event line per 30 s ("overhead camera moved 11 px, re-solved from 4 markers (0.2 mm)");
  - the file `top_table.json` is rewritten at most once a minute, keeping the arm fit (the arm did not
    move when the camera did);
  - fewer than 3 markers visible: the last good solve is held (`holding: true`);
  - mat moved more than 150 mm or 30 degrees, or markers still more than 6 mm off: `calib_ok` false,
    nothing served or written, re-run `table`;
  - camera poses are re-solved with the pinned FOV and a hysteresis of 5 mm / 1 degree so solve noise
    never jitters the frustums; the front camera has its own loop at about 1 Hz.
- The mat is only taped down: it moved 13-19 mm during the evening of 9 Nov between two frames half an
  hour apart (docs/mat-scene.md "Calibration drift").
- Redo rules: a moved camera heals itself; a moved mat or re-glued markers need `table` and the arm fit;
  a new lens or resolution needs everything including `campose --lens-height`
  (docs/camera-calibration.md "Redo when").

## 5. The camera settings lock

(docs/camera-settings.md, jetson/cam_lock.py, jetson/calib/cam_settings.json,
jetson/99-so101-cams.rules, jetson/so101-cam-lock.service)

- Why: with auto exposure, auto white balance and continuous autofocus, the camera re-exposes on the
  arm as it enters the frame, so brightness becomes a function of arm pose and a policy can learn the
  brightness cue instead of the scene. Rule: vary lighting between sessions deliberately, re-capture
  at the start of each session, keep settings fixed within a session.
- What is frozen on the C920s: `focus_automatic_continuous`, `auto_exposure` (1 = Manual Mode),
  `white_balance_automatic` and `exposure_dynamic_framerate` switched off; `focus_absolute`,
  `exposure_time_absolute`, `white_balance_temperature` and `gain` frozen; `power_line_frequency=2`
  (60 Hz) and `backlight_compensation=0` forced. On the icSpring wrist: `auto_exposure` (1) and
  `white_balance_automatic` off; `exposure_time_absolute` and `white_balance_temperature` frozen;
  `power_line_frequency=2` forced (it ships at 50 Hz).
- Exposure units on the C920 are 100 us, in 1/2^n-second stops (78 = 1/128 s, 156, 312). Recording at
  30 fps needs at most 333, so capture clamps at 312 and raises gain instead.
- Measured behaviour of this C920 revision (USB id 046d:08e5), 9 Nov:
  1. a manual control is flagged `inactive` while its auto flag is on; writes fail with EACCES, so the
     auto flag goes first;
  2. manual exposure snaps to the stop table and the same stop reads back as 77 or 78; `check` allows
     a tolerance of 2;
  3. about one second after `auto_exposure` goes 3 -> 1 the camera forces `gain` to 255 (the white-out
     on the first attempt); `apply` waits `SETTLE = 1.5` s after a transition and rewrites what moved
     (jetson/cam_lock.py line 67);
  4. at every stream start in Manual Mode the camera re-derives its exposure (78 came up as 156, then
     312); a write issued while the stream runs holds for that stream; since LeRobot opens the cameras
     itself, a one-shot apply is not enough;
  5. the top C920 re-enumerates on USB now and then and comes back with factory defaults.
- The fix has three layers: `cam_lock.py capture` (per camera: stream in the recording mode, let auto
  settle 4 s, read its choice and the picture's mean grey, lock, then tune gain or exposure until the
  locked picture matches auto's mean, clamped to 100-140 and within 8; about 15 s per camera; a locked
  picture outside mean grey 60-190 is put back on auto and not saved), `cam_lock.py guard` (resident,
  checks every 1 s, re-applies within about a second of any drift, run by the systemd user unit
  `so101-cam-lock.service` with `Restart=always`, `RestartSec=5`, plus `loginctl enable-linger` so it
  runs at boot), and the udev rule `99-so101-cams.rules` (on `ACTION=="add"` for 046d:08e5 and
  32e6:9005 capture nodes, a transient `systemd-run --on-active=3` unit re-applies the file 3 s after
  the camera appears).
- `so101-record` and `so101-record-more` run `cam_lock.py check --strict` first and refuse to start if
  any camera is unplugged, uncaptured or drifted; they also make sure a guard is running.
- Values captured 9 Nov 23:06 under the session lighting (jetson/calib/cam_settings.json): top C920
  exposure 38, gain 42, white balance 3377 K, focus 30 (auto had chosen exposure 83, gain 109); front
  C920 exposure 38, gain 85, white balance 3388 K, focus 35 (auto: 83 and 109); wrist exposure 35,
  white balance 4000 K (auto: exposure 200). Verified mean greys: top 112.6, front 103.0, wrist 119.6.
- Tests: `tests/jetson/test_cam_lock.py` runs the tool against a fake `v4l2-ctl` and a fake frame
  source that reproduce every behaviour above, plus the `so101-record` refusal through a sourced
  `env.sh`.

## 6. Camera health tools

(docs/camera-health.md, jetson/cam_health.py, jetson/cam_drift.py)

- `so101-cams-health [seconds]`: opens all three cameras at once in three threads exactly as
  `lerobot-record` does, 10 s by default after a 1 s warm-up, and reports fps, p50/p95/max frame
  interval, duplicate frames (every 8th pixel bit-identical), stalls over 100 ms, negotiated size and
  fourcc, and CPU load. Fails when fps < 28, any interval > 250 ms, or the size differs; exit 2 when a
  camera cannot be opened.
- Measured 9 Nov (load 1.1 on 6 cores, 12% CPU): top 30.00 fps, p50 32.1 ms, p95 36.7, max 39.7, no
  dups, no stalls; front 30.00 fps, 32.1 / 36.4 / 37.8; wrist 22.29 fps, p50 44.0 ms, max 88.1, FAIL.
- The wrist camera's 22.5 fps is firmware pacing, not bandwidth: it is 22 fps alone on the bus, the two
  C920s keep 30.00 beside it, and 30 ms exposure drops it to 18.7 fps. In a 30 Hz recording roughly one
  wrist frame in four is a repeat. ACT trains fine on that.
- `so101-cams-drift set|check`: saves a reference frame per camera and later compares with
  `cv2.phaseCorrelate` on Hanning-windowed grey plus an ECC Euclidean refinement (translation and
  rotation; a shifted copy of a real frame is recovered within 0.2 px, a 1.5 degree roll within 0.05
  degrees). For the top camera it also projects every marker through the calibration and compares
  with the detected ArUco centres. Threshold 6 px (about 4 mm on the mat), minimum response 0.3.
- Verdicts: `OK`, `MOVED`, `CAMERA MOVED` (frame and markers moved together: re-solve the table, arm
  fit kept), `MAT MOVED` (markers moved, background did not: put the mat back or refit the arm too),
  `MARKER MOVED` (markers no longer rigid among themselves), `LOW CONFIDENCE`, `BAD EXPOSURE`,
  `BAD REFERENCE`, `NO FRAME`.
- First real run, 9 Nov 22:40: top saved with 3 of 4 markers visible, 0.72 px from the calibration;
  the check a few seconds later showed 0.2 px shift and markers within 1.85 px. The front and wrist
  references were blown out because the C920s were on gain 255 from the camera-control work at that
  moment, which is why `set` now refuses such frames and streams two seconds before grabbing.
- Both tools hold `.cams_busy` so the dashboard feeds pause and come back by themselves, and both have
  `--json` and meaningful exit codes for a pre-recording checklist.

## 7. The live mat scene

(dashboard/scene_detect.py, docs/mat-scene.md, tests/scene/)

- `MatScene` is cv2 plus numpy only, 5-8 ms per 640x360 frame on the Mac against a 25 ms budget; the
  server runs it about twice a second on the newest JPEG the Jetson's MJPEG stream already left in
  memory, so the camera is never opened a second time.
- Mat frame 457.2 x 304.8 mm; the frame is warped top-down at 0.5 px/mm into a 229 x 152 image where
  one pixel is 2 x 2 mm and the arm enters from the bottom edge.
- Pipeline: exposure-matched per-channel difference against a stored empty-mat background, 3x3 median,
  threshold 28/255, closing, exclusion of the four marker squares plus 10 mm and a 2 mm border; the
  arm mask; a per-component shadow test (grey ratio 0.40-0.97 with spread under 0.12); objects from
  40 mm2 up to 35% of the mat with centroid, minimum-area rectangle, mean colour and raw pixel.
- Kinematic arm mask: the follower's joints (with the fitted zero offsets) run through the forward
  kinematics to the joint origins plus a point 40 mm past the fingertips; each gets a radius (base 50,
  shoulder 40, elbow 30, wrist 30, wrist roll 35, tip 30, fingers 20 mm) times 1.3 for the hugging
  shadow, projected as 3D points through the camera pose because parallax matters: an elbow 200+ mm up
  under a camera 419 mm up is seen over 100 mm away from its footprint. Mat-level footprints in the
  jogged (9,7) pose: tip 39 mm, wrist roll about 60 mm, wrist flex 63 mm, elbow 92 mm. The edge rule
  keeps running underneath; the mask only grows. This replaced the edge-only rule after TC saw the
  gripper reported as an object in the middle of the mat on 9 Nov.
- Tracking: an id is stable while the object stays within 15 mm of where it was; a track survives 3
  missed calls (the arm passing over); `min_hits=3` on the server (about 1.5 s) keeps one-frame
  shadows, motion blur and flicker off the map (git log 10 Nov).
- Calibration drift: markers detected every call; refit when the error exceeds 3 mm with at least 3
  markers; `mat_shift_mm` and `mat_rot_deg` flag a mat moved more than 150 mm or 30 degrees (a
  re-glued layout the refit cannot tell apart, as happened when the mat was turned round on 9 Nov).
- Background: learned from the live frame or the per-pixel median of up to 15 frames taken 0.5 s apart,
  saved to `jetson/calib/top_mat_bg.png` and reloaded at start-up; adapted slowly (alpha 0.02) on
  unmasked pixels every 10th detect.
- Live state when this sheet was compiled (10 Nov, `/api/scene`): one object, id 514, at
  (11.26, 6.33) in, 10 x 6 mm, with the arm masked by kinematics over 11.7% of the mat, 4 markers
  seen, calibration error 0.37 mm.
- Fixtures (tests/scene/fixtures, real frames from 9 Nov): `bg.jpg` (empty mat, arm parked), `obj.jpg`
  (one hex nut at (10.97, 6.94) in), `top_now.jpg` (arm raised), `new.jpg` (mat turned round and
  markers re-glued) with their two calibrations.

## 8. Wrist camera hand-eye

(jetson/calib/wrist_campose.json, docs/camera-calibration.md section 7, jetson/calib/wristcam/)

- Method: a few wrist-camera frames each showing one whole mat marker, plus the follower's joints at
  each frame, through the arm's forward kinematics; the unknowns are the camera pose in
  `gripper_link`, focal length, two radial distortion terms, each marker's glue angle and a small
  per-view rotation with a 1 degree prior. Multi-started from 96 camera placements.
- Result, 9 Nov 23:56, five usable views (w1, w2, w3, w5, w6; w4 showed no whole marker), markers 1, 2
  and 3, 20 points: camera centre (1.2, 67.9, -11.3) mm in `gripper_link`, 68 mm off the gripper axis
  (the STL mount says 69), optical axis 30.1 degrees from the gripper axis looking past the fingertips,
  focal length 529 px, hfov 62.3 degrees, vfov 48.8, k1 -0.41, k2 0.15.
- Quality: rigid reprojection 8.1 px RMS (3.0-11.3 per view); 0.51 px with the per-view corrections
  (0.3-1.4 degrees each); leave-one-out 4.5-15.7 px; the mount estimate moves 0.8-3.0 mm and
  0.2-1.5 degrees without any one view. So the mount is known to a few millimetres and about
  1.5 degrees; each hand-posed view's joints are only good to about a degree.
- The 3D stage attaches the WRIST CAM frustum at this pose (`/api/calib` `camera.wrist`), with the old
  STL guess as fallback (git log 9-10 Nov).
- Synthetic test: a wrist camera at a known pose on the real kinematics comes back within 2 mm,
  0.5 degrees and 2% focal length (tests/calib/test_handeye.py).

## 9. Scripted pick-and-place

(jetson/pick_seq.py, tools/pick_loop.py, jetson/calib/pick_state.json, README.md)

- `pick_seq.py` runs a list of joint waypoints on the follower in one bus session (the earlier
  `goto.py` reconnected per move, about 2 s each): half-cosine eased moves, stretched so no joint
  exceeds 80 degrees/s, joints read back after each move. Servo sag under load was measured at
  1-3.5 degrees on 9 Nov. It feeds forward only the last waypoint's error; summing errors acted like an
  integrator and overshot by 24 mm. It refuses to run while a LeRobot job holds the arm and freezes the
  arm with torque held on SIGTERM.
- `pick_loop.py` repeats "find the nut, pick it, place it" and learns: a 9-waypoint plan from `/api/ik`
  (hover 1.5 in, grasp at 0.12 in, close, lift, hover target, place at 0.3 in, open, lift, rest); each
  waypoint biased by the sag measured at the same waypoint last time (gain 0.7, capped at 6 degrees);
  the place target shifted by the camera-measured landing error (gain 0.5, capped at 0.6 in); targets
  alternate between mat (6,5) and (9,6) so every pick is unambiguous. The nut is found as the dark blob
  nearest its last position in a fresh overhead frame.
- First four runs, 10 Nov about 00:03-00:04, each about 18.3 s: all four picks succeeded; landing
  errors 31.2, 10.4, 39.3 and 21.2 mm. The sag learning converged: hover_nut error 3.17 -> 1.07 ->
  1.02 -> 0.97 degrees, lift 4.58 -> 0.99 -> 1.07 -> 0.97 degrees.
- The honest reading (README.md "Status, honestly"): the arm reaches within about 3 mm of a commanded
  point, but the hex nut lands 10-40 mm off because it shifts in the single-moving-jaw gripper. Next
  step: measure the part in the jaws with the gripper camera before placing.

## 10. Simulation

(sim/CLOUD_ROUNDS.md, sim/INFERENCE_TUNING.md, sim/ANALYSIS_v2_10k.md, sim/ANALYSIS_v3m.md,
NEXT_STEPS.md 19-21 Oct, env.sh)

- Setup: MuJoCo through `so101-nexus` (MuJoCo Menagerie `robotstudio_so101`) with a third-party
  LeRobot wrapper; a scripted IK expert writes LeRobot v3 datasets in `so101_follower` units with
  `wrist` and `overhead` cameras at 320x240, 50 Hz. Evaluation: 50 unseen layouts, seed 9100, goal
  radius 0.04 m; a 50-episode eval carries about +-7% noise.
- Round 1 (300 demos, ACT 20k steps on the Mac at 3.8 step/s): 20% at 10k, 20% at 15k, 30% at 20k.
- Round 2 (600 demos, 92,828 frames, warm start): 76% at 10k, 52% at 15k, 56% at 20k. Success peaks
  before the end of training: pick by rollout, not loss.
- Failure analysis of that policy: 11 of 12 failures never grasp (descent lands 1-3 cm off, closes on
  air, "phantom-carries" to the disc); cubes at radius >= 0.29 m fail 38% vs 21% elsewhere; a 600-step
  budget rescues none.
- Inference tuning: executing the full 100-step chunk open loop gives 76%; re-planning every 50 steps
  58%, every 25 steps 58%, every 10 steps 16%; temporal ensemble 0.01 gives 56%. Re-planning more often
  hurts this policy, which set the Jetson's `queue_threshold` recommendation to 10.
- Round 3 (recovery demos): 66%, a regression; it rescued 7 baseline failures but added 12, because
  the policy imitated re-opening after a good grasp and only 73 of 600 episodes contained a real
  recovery.
- Round 4 (the 600 clean episodes plus those 73 recovery episodes, warm start): 78% at 10k; continued
  to 20k: 90% on seed 9100 and 78% on seed 9200, 84% pooled over 100 unseen layouts. Best sim policy,
  staged on the Jetson for a sim-to-real attempt with a 25 mm red cube and a blue disc.
- Cloud rounds on an A100 at $2.50/h, about 3.1 step/s at batch 32: a 60k from-scratch run reached
  44/56/46/52/54/62% at 5k..30k before its 3 h cap; a warm start at batch 32 peaked at 72% at 2.5k and
  then degraded. Total about $10.9. Neither beat the Mac's batch-8 warm starts.

## 11. Real-robot datasets and policies

(NEXT_STEPS.md 7-16 Oct)

- `bottle-pick-clean`: 40 episodes, 28,288 frames; ACT 50k steps. Behaviour: reaches, then hovers,
  because training frames always showed the leader arm and a hand in the overhead view.
- `jar-pocket-trim`: 18 episodes, 16,295 frames, each cut 1 s after the arm's last motion; ACT 40k.
- `lighter-handoff-clean`: 16 of 30 episodes, 7,349 frames; two ACT runs (chunk 100 on the Mac, chunk 50
  on the Jetson) as an A/B, final losses 0.001 and 0.002; a SmolVLA fine-tune in the cloud.
- The Jetson record loop ran about 23 Hz (462 frames per 20 s episode) while the dataset is stamped
  30 fps, so playback is about 1.3x compressed; consistent between record and rollout.
- Jetson inference: synchronous 13 Hz; `so101-run-async` with the rtc engine holds 30 Hz with 4-5 steps
  (about 150 ms) of latency. SmolVLA loads in about 45 s and 0.93 GB of GPU memory after two patches.

## 12. Test suite

(tests/, tests/ui/README.md, README.md)

- 600 tests collected on 10 Nov (`python -m pytest tests -q`, about 3 minutes); all synthetic, no
  hardware. Folders: `busheal` (fake Feetech bus), `calib` (rendered checkerboards and ArUco tables
  through a synthetic camera; lens model within 1.5 px, camera position within 6 mm, pixel-to-table
  within 1.5 mm; the arm fit recovers base and heading and refuses a mirror; forward kinematics against
  the dashboard URDF), `jetson` (fake `v4l2-ctl`, fake capture source, synthetic timestamps, shifted and
  rotated copies of a real frame), `scene` (real fixture frames plus synthetic squares un-warped into
  them), `server` (temp calibration dirs, the real table file, fixture JPEGs fed into the frame buffer,
  checks that nothing runs a subprocess), `ui`.
- UI tests: no framework, npm or build step. A threaded Python stub server answers every `/api/*` route
  from mutable Python objects; a small Node driver opens the page in headless Chrome over the DevTools
  protocol, blocks all external traffic, freezes the page's timers, and fails any test whose page threw
  or logged a console error. About 45 tests in 35 s. The 3D-stage tests alone let the CDNs through and
  run WebGL on SwiftShader; they skip when offline.

## 13. LeRobot plugins

(plugins/*/pyproject.toml, NEXT_STEPS.md 5-15 Oct)

- `lerobot-robot-busheal`: wraps the Feetech bus; on "no status packet" it closes and reopens the port,
  pings a motor and retries once (give-up after 10 s); applies the gripper torque and overload settings
  after LeRobot's configure; clears a tripped overload flag; writes joint state to `/dev/shm` for the
  dashboard; relays LeRobot's `say()` lines to the Mac.
- `lerobot-camera-avf`: a macOS camera backend that selects cameras by device name and native format
  through AVFoundation, never by index; FaceTime and iPhone cameras are refused.
- `lerobot-policy-actasync`: lets LeRobot's async (rtc) inference engine drive an ACT policy by
  accepting the RTC call shape and ignoring the prefix.

## 14. Images

See `images/CAPTIONS.md`. Live frames `top.jpg`, `front.jpg`, `wrist.jpg` and the scene map `scene.png`
were pulled from the running dashboard on 10 Nov 2026; the fixture frames are the real 9 Nov captures
from `tests/scene/fixtures`; `top_mat_bg.png` is the learned empty-mat background; the two `wristcam_*`
frames are hand-eye captures. Dashboard screenshots were taken through Chrome; if any are missing,
`images/CAPTIONS.md` says so.
