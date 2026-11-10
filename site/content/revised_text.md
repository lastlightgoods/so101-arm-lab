========================================================================
PAGE: public/index.html
========================================================================

SO-101 Arm Lab
nav: Built | Notes | Gallery | Status | Timeline

Lastlight Goods · Bend, Oregon · Field notes
# SO-101 Arm Lab
Notes from five weeks of building a robot arm lab at home: an SO-101 arm, a Jetson, three cameras, and a cutting mat with four ArUco markers on it.
snapshot from 9 Nov 2026, 23:06 · cameras locked · 30.00 fps
[IMAGE: Render from the dashboard's 3D stage, front view: the SO-101 arm standing behind the 18 by 12 inch mat with its four ArUco markers, the overhead and front camera frustums drawn as thin wireframes above it, and the wrist camera's view cone at the gripper.]
caption: Render from the dashboard's 3D stage, front view, 10 Nov. Not a photo. The arm is posed from the follower's joint readings at the time, the mat from the overhead solve, and the three camera frustums from their measured poses. The wrist camera's cone hangs off the gripper.
- 600 tests, no hardware
- 0.19mm worst marker residual
- 30.00fps both C920s, locked
- 36days since 5 October

How it is built
## What runs where
The Jetson runs the arms and cameras and records the datasets. The Mac runs the wall dashboard. Training runs locally on the Mac or the Jetson, or on Hugging Face Jobs for the bigger runs.
[IMAGE: System diagram. The Mac runs the wall dashboard and local training, reaching the Jetson tc-nano over SSH. The Jetson runs the SO-ARM101 arm, three cameras, dataset recording and the camera lock guard. The arm works over an 18 by 12 inch cutting mat with four ArUco markers.]
diagram labels: 01 · DASHBOARD + TRAINING | MAC | M1 Max · Python 3.12 · MPS | ACT and SmolVLA training | datasets and checkpoints | wall dashboard, 3D arm, map | 600 tests against fakes | SSH | tegrastats | joints · mjpeg | 02 · EDGE | JETSON ORIN NANO | tc-nano · JetPack 6.2 · 25 W | arm: teleop, record, rollout | 3 cameras, 30 fps, frozen | cam_lock guard, every 1 s | calibration re-solves if bumped | USB | serial 1 Mbaud | 3 cameras | 03 · BENCH | SO-ARM101 + MAT | six STS3215 · 18 × 12 in mat | top | front | wrist | base
[IMAGE: System diagram. The Mac runs the wall dashboard and local training, reaching the Jetson tc-nano over SSH. The Jetson runs the SO-ARM101 arm, three cameras, dataset recording and the camera lock guard. The arm works over an 18 by 12 inch cutting mat with four ArUco markers.]
diagram labels: 01 · DASHBOARD + TRAINING | MAC | M1 Max · Python 3.12 · Apple MPS | ACT and SmolVLA training | datasets and checkpoints | wall dashboard, 3D arm, mat map | 600 tests against fakes | SSH · tegrastats | joints · mjpeg | 02 · EDGE | JETSON ORIN NANO | tc-nano · JetPack 6.2 · 25 W | arm: teleop, record, rollout | three cameras, 30 fps, frozen settings | cam_lock guard, checks every second | calibration re-solves when bumped | USB · serial 1 Mbaud | 3 cameras | 03 · BENCH | SO-ARM101 + MAT | 6 × STS3215 · 18 × 12 in · 4 markers | top | front | wrist | base
caption: Diagram. Thin lines are links; the gold ones carry data between machines. The mat is drawn in plan at the measured marker layout, with the arm base below the arm-side edge where the jog-and-fit put it.
### Arm
- SO-ARM101, leader and follower · prebuilt
- Six Feetech STS3215 per arm · firmware 3.15, 1 Mbaud
- Waveshare bus boards · follower 12 V
- LeRobot 0.6.1 on both machines
### Cameras
- Top: Logitech C920 · 640×360 MJPG, 30 fps, 16.5 in up
- Front: Logitech C920 · 42° tilt from the far side
- Wrist: icSpring UVC on the gripper · 640×480, 22 fps by firmware
- Settings captured per session, then frozen
### Bench
- Fiskars 18×12 in cutting mat · printed inch grid
- Four 2 in ArUco markers · DICT_4X4_50
- Jetson Orin Nano, tc-nano · 8 GB, 25 W
- Mac, M1 Max · wall dashboard, local training on MPS

Field notes
## Seven notes from five weeks
The source files for each note are listed at the end of it.
Field note 01
### Locking the C920's exposure
I used two Logitech C920s because they're cheap and common. I wanted to freeze their exposure so the arm sweeping through the frame would stop changing the picture. The first attempt came back pure white.
The revision I have (USB id 046d:08e5) does three things that a plain `v4l2-ctl` script does not account for. A manual control is flagged `inactive` while its auto flag is on, so the write fails with EACCES unless the auto flag goes first. About one second after `auto_exposure` goes from 3 to 1 the camera forces `gain` to 255 on its own, overwriting whatever was written in between; that was the white frame. And at every stream start in Manual Mode it re-derives the exposure. I watched 78 come up as 156, then 312.
The lock ended up as three layers. `cam_lock.py capture` lets auto settle for 4 s, reads what it chose, locks, waits 1.5 s for the gain reset to pass, then tunes gain until the locked picture matches auto's mean grey within 8 levels. A resident guard checks every second and re-applies within about a second of any drift, because LeRobot opens the cameras itself when a recording starts. A udev rule re-applies 3 s after a camera re-enumerates with factory defaults. `so101-record` refuses to start unless `check --strict` passes.
On 9 Nov at 23:06 the top camera locked at exposure 38 and gain 42 where auto had chosen 83 and 109, and the picture's mean grey stayed at 112.6 with the arm in and out of frame.
docs/camera-settings.md, jetson/cam_lock.py, jetson/calib/cam_settings.json · the full write-up
[IMAGE: Camera frame from the top camera at the locked exposure: the Fiskars 18 by 12 inch mat with four 2 inch ArUco markers, the follower parked at the top edge, the arm base in blue tape at the bottom centre, and the hex nut as a small dark dot left of centre.]
caption: Camera frame (top camera) at the locked exposure: the mat with its four markers, the follower parked at the top edge, the hex nut a small dark dot left of centre.

Field note 02
### Calibrating the table with four ArUco markers
The work surface is a Fiskars 18x12 in self-healing cutting mat. I chose it because it has an inch grid printed on it and costs very little. I glued four 2 in ArUco markers from `DICT_4X4_50` onto it at grid positions (16,10), (2,10), (15,4) and (3,4), with the origin at the mat corner nearest the arm and X running along the 18 in edge.
`cam_calib.py table` finds the four markers in one overhead frame, snaps each to an exact square, and fits a homography from 16 corner points to mat millimetres. That is all the table calibration involves. The current solve, from 9 Nov at 23:06, puts the worst marker 0.19 mm from where the layout says it is. Under 2 mm is normal; over 5 mm usually means a marker position was measured wrong.
A homography only knows the mat surface, so height matters. `locate` takes a pixel and a height above the mat; a 60 mm tall box read as if it were flat lands centimetres off. I also pinned each camera's field of view from a tape-measured lens height, 16.5 in for the top camera (69.16 degrees) and 18.375 in for the oblique front camera (68.88 degrees), and treat those numbers as belonging to the lens rather than the mount.
None of the learned policies use any of this. ACT and SmolVLA look at pixels. The calibration is for the scripted moves, for the map on the wall, and for being able to say where things are in inches.
docs/camera-calibration.md, jetson/calib/top_layout.json, jetson/calib/top_table.json
[IMAGE: Camera frame from the top camera of the empty mat on 9 Nov, before it was re-squared: the earlier marker layout, the arm parked at the top edge.]
caption: Camera frame (top camera): the empty-mat background the scene tests use, from 9 Nov before the mat was re-squared. The earlier marker layout, the arm parked at the top edge.

Field note 03
### Finding the arm's base on the mat
Knowing where the mat is in the camera does not say where the arm is on the mat. For that I switched torque off, pushed the closed jaw tip onto known grid points, and recorded the joints each time. Forward kinematics from the URDF, which match the dashboard's 3D arm to 0.1 mm, turn those joints into tip positions, and a rigid fit gives the base pose.
The first rigid fits left 15 to 30 mm of residual, far too much. The reason is a LeRobot convention: it zeroes each joint at the middle of its calibrated range, which is not exactly the URDF zero. A few degrees of offset in the lift, elbow and wrist make the kinematics read short near the base and right far out. So the fit now solves the base pose, the table height and three joint offsets together. On the real arm the offsets came out at shoulder_lift -3.9 degrees, elbow_flex -5.2 degrees and wrist_flex -0.7 degrees, and the residuals fell to 0.7 to 1.1 mm on the four points of the current fit, with an RMS of 0.69 mm. The base sits at (9.04, -3.18) in on the mat frame, heading 90.15 degrees.
One more rule came out of the jogging. The SO-101 gripper has one fixed jaw that is off the roll axis, so the contact point moves when the wrist rolls. I jogged everything at a wrist roll of 0.46 degrees and now command near that value; `/api/ik` defaults to it.
docs/camera-calibration.md "Joint zero offsets", jetson/calib/top_table.json, jetson/calib/top_armfit.json
[IMAGE: Dashboard screenshot, default tempest skin: turntable header, MISSION STATUS reading IDLE READY with tc-nano up 3 days, six Jetson gauges for GPU, bus, CPU, RAM, temperature and power, the three camera feeds with a CAL chip reading 0.2 mm and arm fit 1.1 mm, and the self-playing Tempest attract screen.]
caption: Dashboard screenshot, default `tempest` skin, landscape window. The CAL chip over the top feed reads 0.2 mm for the table and 1.1 mm for the arm fit; the portrait wall kiosk shows more rows below this.

Field note 04
### A live map of what's on the mat
The wall dashboard draws a top-down map of whatever is lying on the mat. `scene_detect.py` warps each overhead frame into mat space at 0.5 px per mm (229 x 152 pixels, 2 mm per pixel), subtracts a stored picture of the empty mat, and reports every blob between 40 mm² and a third of the mat with its centroid in inches, its size and its mean colour. It takes 5 to 8 ms a frame and runs twice a second.
Most of the false detections came from the arm itself. My first rule said any blob touching the arm-side edge is the arm. On 9 Nov I watched the arm reach across the mat and saw the gripper appear on the map as an object, because the forearm had left the edge through the air. The fix masks the arm from the joint readings rather than from the image: the follower's joints go through the forward kinematics, each joint origin gets a radius (30 to 50 mm, times 1.3 for the shadow), and the points are projected through the camera pose as 3D points. The projection is needed because an elbow 200 mm up under a camera 419 mm up is seen more than 100 mm away from the spot beneath it.
The other problem was flicker. A track now has to be seen in 3 consecutive detections, about 1.5 s, before it is listed, and it survives 3 missed ones, so an object the gripper passes over keeps its id.
When I fetched the map while writing this, it showed one object, id 514, at (11.26, 6.33) in, 10 by 6 mm. That is the hex nut.
docs/mat-scene.md, dashboard/scene_detect.py
[IMAGE: Generated by the software: the learned empty-mat background in mat space, a 229 by 152 pixel top-down image of the mat at 2 millimetres per pixel, shown enlarged with visible pixels.]
caption: Generated by the software (learned background), not a camera frame: the empty-mat background in mat space, all 229 × 152 pixels of it at 2 mm each. This is what the detector subtracts from every frame, shown here at its true resolution.

Field note 05
### Re-calibrating when the camera gets bumped
The overhead camera hangs on a stand the arm can reach, and the mat is only taped down. On the evening of 9 Nov the mat moved 13 to 19 mm between two frames half an hour apart, and the camera has been nudged more than once. So the calibration has to notice when it goes stale.
The four markers are detected in every frame, 2.6 ms each. When they land more than 3 mm from where the stored homography puts them, and at least 3 are visible, a fresh homography is fitted and used from then on. The server compares the stored and live solves; once the largest marker shift passes 3 px, `/api/calib` serves the live solve, the overlays and the 3D stage follow within a second, one event line is written ("overhead camera moved 11 px, re-solved from 4 markers (0.2 mm)"), and the file is rewritten at most once a minute, keeping the arm fit because the arm did not move when the camera did.
The camera poses are re-solved the same way, with one thing held fixed: the field of view belongs to the lens and was pinned once from a tape measure, so only position and orientation are re-solved, with a 5 mm / 1 degree hysteresis so noise does not jitter the frustums on the wall.
Limits: fewer than 3 markers visible means the last good solve is held. A mat moved more than 150 mm or 30 degrees, or markers re-glued, means `calib_ok` goes false and the chip on the feed reads CHECK MAT.
docs/mat-scene.md "Self-healing calibration", docs/camera-calibration.md section 6
[IMAGE: Camera frame from the top camera with the mat turned round and the markers re-glued: a congruent rectangle of four markers the refit cannot tell from a rigid move.]
caption: Camera frame (top camera), 9 Nov: the mat turned round and the markers re-glued, a congruent rectangle the marker refit cannot tell from a rigid move. This is why `mat_shift_mm` and `mat_rot_deg` exist.

Field note 06
### Why the nut lands off target
With a calibrated camera and a fitted arm I wrote the obvious script: find the hex nut in the overhead frame, plan nine waypoints through `/api/ik`, pick it up, carry it to a target, set it down, measure where it landed, repeat and learn.
Two kinds of learning are in the loop. Servo sag under load, which I measured at 1 to 3.5 degrees, is corrected per waypoint from last time's error with a gain of 0.7. I tried summing the errors first and overshot by 24 mm, so only the last error is fed forward. The landing error, measured by the camera, shifts the place target with a gain of 0.5.
The sag learning worked. Over the first four runs on 10 Nov the hover error fell from 3.17 to 0.97 degrees and the lift error from 4.58 to 0.97. All four picks succeeded, each run took about 18 s, and the arm reaches within about 3 mm of a commanded point.
The nut still landed 31, 10, 39 and 21 mm from the target. The reason is mechanical: the SO-101 gripper has one moving jaw, and a 10 mm nut shifts inside it during the grasp, so the part is no longer where the tip is. The next step is to look at the part in the jaws with the wrist camera before placing, which is why that camera now has a solved hand-eye pose (1.2, 67.9, -11.3) mm from the gripper link, good to a few millimetres and about 1.5 degrees.
tools/pick_loop.py, jetson/pick_seq.py, jetson/calib/pick_state.json, README.md, jetson/calib/wrist_campose.json
[IMAGE: Camera frame from the icSpring wrist camera: the gripper jaws enter from the bottom of the frame, the hex nut sits just right of the fingertips on the mat grid, with strong barrel distortion and a green-tinted white balance.]
caption: Camera frame (wrist camera): the jaws enter from the bottom, the hex nut sits just right of the fingertips. Strong barrel distortion (k1 −0.41 in the hand-eye solve) and the green-tinted white balance before the gain match.

Field note 07
### Testing without the robot
Most evenings the arm is busy, or the Jetson is training, or the cameras are being re-aimed. The code still has to be checked, so the test suite does not touch any of it. `python -m pytest tests -q` collects 600 tests and takes about 3 minutes with nothing plugged in.
The calibration tests render checkerboards and ArUco tables through a synthetic camera with known intrinsics, distortion and pose, then check what the tool gets back: the lens model within 1.5 px, the camera position within 6 mm, pixel-to-mat within 1.5 mm, and that the arm fit refuses a mirror image. The hand-eye solver is run on a synthetic wrist camera placed on the real kinematics and must come back within 2 mm, 0.5 degrees and 2 percent of focal length. The camera lock is driven against a fake `v4l2-ctl` that reproduces every behaviour I measured on the real C920, the gain reset included. The bus plugin runs on a fake Feetech bus that drops packets on cue.
The dashboard page gets the same treatment with no framework and no npm: a Python stub server answers every API route from objects a test can mutate, a small Node driver opens the page in headless Chrome, blocks every external request, freezes the page's timers, and fails any test whose page threw or logged an error. Forty-five of those run in about 35 s. The camera frames in the fixtures are from 9 Nov: an empty mat, one hex nut at (10.97, 6.94) in, and the arm raised.
tests/, tests/ui/README.md, README.md
[IMAGE: Camera frame from the top camera with the arm raised over the mat: the forearm leaves the mat edge through the air, the frame that exposed the edge-only arm rule.]
caption: Camera frame (top camera), one of the test fixtures from 9 Nov: the arm raised over the mat, the forearm leaving the edge through the air. The kinematic mask was written for this case.

Gallery
## Frames and screenshots
Three kinds of image here, and each caption says which: camera frames and dashboard screenshots taken from the running wall dashboard shortly after midnight on 10 Nov 2026; one mat map generated by the software; and one fixture frame, a 9 Nov camera capture checked into the tests. Each opens at full size.
[IMAGE: Camera frame (top camera): the mat with four ArUco markers, the follower parked at the top edge.]
caption: Top · camera frame from the overhead C920 at the locked exposure, markers 0 to 3 at their inch positions, the arm base in blue tape at bottom centre.
[IMAGE: Camera frame (front camera): the mat seen obliquely with three markers in view and the black articulated lamp stand across the frame.]
caption: Front · camera frame from the second C920, 42 degrees down from past the far edge. The black articulated arm across the frame is the lamp stand, not the robot.
[IMAGE: Camera frame (wrist camera): the jaws enter from the bottom, the hex nut just right of the fingertips.]
caption: Wrist · camera frame from the icSpring module bolted to the gripper: jaws from the bottom, the hex nut just right of the fingertips, one marker corner at lower right.
[IMAGE: Mat map generated by the software: the red kinematic arm mask, yellow marker squares and one green object box.]
caption: Mat map · generated by `scene_detect.py`, not a camera frame: a 1 inch grid, the arm's silhouette from the follower's joints, object id 514 at (11.26, 6.33) in.
[IMAGE: Camera frame (wrist camera), hand-eye capture w1: a single mat marker.]
caption: Hand-eye w1 · camera frame from the wrist camera looking at a single mat marker, with the follower's joints recorded; one of five views that solved the camera's pose. Rigid residual 3.0 px.
[IMAGE: Camera frame (wrist camera), hand-eye capture w3: another single-marker view from a different height and tilt.]
caption: Hand-eye w3 · camera frame from the wrist camera, another single-marker view from a different height and tilt. Rigid residual 7.5 px.
[IMAGE: Camera frame (top camera): one hex nut on the mat at (10.97, 6.94) inches, the arm parked at the top.]
caption: Fixture: one nut · camera frame (top camera) with a hex nut at (10.97, 6.94) in, the frame the object tests measure.
[IMAGE: Dashboard screenshot, default tempest skin: gauges, camera feeds and the Tempest attract screen.]
caption: Dashboard, tempest · dashboard screenshot of the default skin in a landscape window: mission status, the six Jetson gauges, three camera feeds, the attract screen.
[IMAGE: Dashboard screenshot, Lastlight Goods DYSTOPIAN skin: silver only, JetBrains Mono, facility signage, and the card ARM BAY. IDLE.]
caption: Dashboard, dystopian · dashboard screenshot of the Lastlight Goods DYSTOPIAN skin: silver only, JetBrains Mono, facility signage, and the follower pose at the time as a silhouette.

Status
## The pick loop isn't finished yet
The arm reaches within about 3 mm of a commanded point. The hex nut still lands 10 to 40 mm from where it was meant to go. The reason is the gripper: the SO-101 has one moving jaw, and a 10 mm nut shifts inside it during the grasp, so the part is no longer where the fingertip is. Sag learning does not fix that. In the first four runs the joint errors converged in one run and the landing error did not.
The parts underneath do work. The table calibration holds to a fifth of a millimetre and re-solves itself when the stand is bumped. Both C920s stream at a true 30.00 frames per second with frozen settings, and a guard puts them back within a second if anything changes them. The mat map runs in under 8 ms a frame and no longer lists the gripper as an object. Six hundred tests cover all of it without a robot plugged in.
Open items, in order: look at the part in the jaws with the wrist camera before placing, using the hand-eye pose solved on 9 Nov; account for the wrist camera pacing itself at about 22 fps by firmware, so one frame in four is a repeat in a 30 Hz recording; and keep working on the real-robot learned policies, which are still early. The first one learned to reach and then hover, because every training frame showed the leader arm and a hand.
### First four runs · 10 Nov, about 00:03
picks: 4 of 4 succeeded, about 18.3 s each
reach: within about 3 mm of a commanded point
landing: 10–40 mm off, the nut shifting in the jaws
table: Per run
run | hover err ° | lift err ° | landing mm
1 | 3.17 | 4.58 | 31.2
2 | 1.07 | 0.99 | 10.4
3 | 1.02 | 1.07 | 39.3
4 | 0.97 | 0.97 | 21.2
jetson/calib/pick_state.json, tools/pick_loop.py

Timeline
## 5 October to 10 November 2026
- 5 Oct Both arms on the bus. Port mapping confirmed with a wiggle test. The cause of the bus dropouts found the same night: cameras and servo boards shared one USB 2.0 hub, and opening a stream stalled the serial chip in 0.3 s.
- 6 Oct Gripper dropping out mid-grasp traced to the STS3215 overload trip and fixed in the bus plugin.
- 7 Oct First dataset, `bottle-pick`: 50 raw episodes, 40 kept, 28,288 frames; ACT trained 50k steps on the Mac. A second dataset, `jar-pocket`, followed.
- 14 Oct The Jetson arrives and is imaged with JetPack 6.2.1. First version of the wall dashboard: gauges, camera feeds, an embedded terminal, the 3D arm from the URDF.
- 15 Oct Synchronous inference reaches only 13 Hz on the Jetson; the async path holds 30 Hz. A policy that froze at rest is traced to idle time at the start of 13 of 16 episodes.
- 16 Oct Dashboard additions. The turntable header, the self-playing Tempest attract screen and the retinal-scan lock. Also a SmolVLA fine-tune in the cloud: 10k steps, 61 minutes, about $2.55.
- 18 Oct First git commit. The 3D stage gets the mat, the camera frustums and solid follower meshes.
- 19–21 Oct Simulation rounds. The best sim policy reaches 84% over 100 unseen layouts; re-planning more often made this policy worse.
- 28 Oct Two C920s installed as top and front cameras; three-camera recording config.
- 29 Oct Grippers recalibrated; teleop repaired after the calibration JSON drifted from servo EEPROM.
- 9 Nov Calibration day. Arm base jog-and-fit with joint zero offsets, the camera settings lock with guard and udev rule, camera health and drift tools, the kinematic arm mask, camera poses that re-solve themselves, the wrist camera hand-eye solve, and the single-session waypoint runner.
- 10 Nov First pick-loop runs. Mat-map persistence filter, wrist pose on the 3D stage, and the iterative pick-and-place learner with its first four runs. This page.

A Lastlight Goods project
Bend, Oregon
Figures as of 10 November 2026.
- github.com/lastlightgoods/so101-arm-lab (repo, public soon)
- Note: locking the C920's exposure
- Back to top


========================================================================
PAGE: public/notes/c920-gain-lock.html
========================================================================

SO-101 Arm Lab
nav: Notes | Gallery | Status

← back to the lab
Note · Cameras
# Locking a Logitech C920's exposure: gain 255, re-derived exposure, and keeping it frozen for LeRobot
Measured 9 Nov 2026 on a Jetson Orin Nano · USB id 046d:08e5 · ~6 minutes

If you record SO-101 datasets with a Logitech C920 and have tried to pin its exposure so the arm moving through the frame stops changing the picture, this is for you. I spent an evening on it and measured some firmware behaviour worth knowing before writing the three `v4l2-ctl` lines everyone writes first.
[IMAGE: Camera frame from the top camera at the locked exposure: the cutting mat with four ArUco markers, the follower parked at the top edge.]
caption: Camera frame (top camera) after the lock: exposure 38, gain 42, mean grey 112.6 with the arm in or out of the frame.
## Why lock at all
Out of the box a C920 runs auto exposure, auto white balance and continuous autofocus. When the arm enters the frame the camera re-exposes on the arm, so brightness becomes a function of arm pose, and a policy can learn the brightness cue instead of the scene. My rule: vary lighting between sessions deliberately, re-capture settings at the start of each session, keep them fixed within a session.
## What this revision does
The controls, by their v4l2 names: `auto_exposure` (3 = Aperture Priority, 1 = Manual Mode), `exposure_time_absolute`, `exposure_dynamic_framerate`, `gain`, `white_balance_automatic`, `white_balance_temperature`, `focus_automatic_continuous`, `focus_absolute`, `power_line_frequency`, `backlight_compensation`.
- Manual controls are `inactive` while their auto flag is on. A write to `exposure_time_absolute` or `gain` in that state fails with EACCES. Set `auto_exposure=1` first, then the manual values.
- Exposure snaps to a stop table. `exposure_time_absolute` is in 100 µs units and the camera only exposes in 1/2n-second stops: 78 (1/128 s), 156, 312. The same stop reads back as 77 or 78, so verify with a tolerance of 2. At 30 fps exposure must stay at or below 333, so 312 is the longest usable stop; if auto wanted more, start at 312 and raise gain.
- About one second after `auto_exposure` goes 3 → 1, the camera forces `gain` to 255. This was the one that took longest to find. The sequence I saw: write `auto_exposure=1`, write `gain=42`, read back 42, and roughly a second later read back 255 and a white picture. Any gain written inside that window is clobbered. The tool now waits 1.5 s after the transition and rewrites whatever moved.
- At every stream start in Manual Mode the camera re-derives its exposure. I had 78 locked; opening the stream brought it up as 156, the next open 312. A write issued while the stream runs holds for that stream. LeRobot opens the cameras itself when `lerobot-record` starts, so a one-shot apply before recording is not enough.
- The camera occasionally re-enumerates on USB and comes back with factory defaults, auto on.
## The fix, in three layers
Capture once per lighting session. `cam_lock.py capture` streams each camera in the exact mode recording uses (640x360 MJPG 30 fps), hands it to auto, waits 4 s, reads what auto chose and the picture's mean grey, locks, then tunes `gain` until the locked picture matches auto's mean within 8 grey levels (clamped to 100-140). A locked picture outside 60-190 is put back on auto and not saved. The result is one JSON block per camera keyed by its `/dev/v4l/by-id` name. On 9 Nov the result was: top C920 exposure 38, gain 42, white balance 3377 K, focus 30; front C920 exposure 38, gain 85, white balance 3388 K, focus 35. Auto had chosen exposure 83 and gain 109 on both. The capture also forces `power_line_frequency=2` and `backlight_compensation=0`, and switches off `exposure_dynamic_framerate` so the camera cannot halve the frame rate to buy light.
A resident guard. `cam_lock.py guard` runs as a systemd user unit (`Restart=always`, with `loginctl enable-linger` so it starts at boot). It checks every second and re-applies the saved values within about a second of any drift, logging each fix. That catches the stream-start jump.
A udev rule for re-plugs. On `ACTION=="add"` for the C920's capture node, a transient `systemd-run --on-active=3` unit re-applies the file 3 s later, because udev kills `RUN+=` programs after a few seconds and the camera is not always ready to take controls the instant its node exists.
The `so101-record` wrapper runs `cam_lock.py check --strict` first and refuses to start if any camera is unplugged, uncaptured or drifted. A dataset recorded on auto is expensive to redo, so the tool refuses rather than warns.
## How to tell it worked
Pull a frame with the arm out of the picture and one with the arm waved through it; the mean grey should not move (mine stays at 112.6 on the top camera). This should show the auto flags off and no `inactive` anywhere:
    `v4l2-ctl -d /dev/video0 --list-ctrls | grep -E "auto|absolute|gain"`
`journalctl --user -u so101-cam-lock -n 20` shows one line per re-lock, typically right after LeRobot opens a camera. The whole tool is covered by tests against a fake `v4l2-ctl` that reproduces every behaviour above, so it can be worked on without the camera.
## Where the code is
`jetson/cam_lock.py`, `jetson/99-so101-cams.rules`, `jetson/so101-cam-lock.service` and `docs/camera-settings.md` in the so101-arm-lab repository at github.com/lastlightgoods/so101-arm-lab. The repo will be public soon. If your C920 revision behaves differently, I would like to hear the numbers.
← back to field note 01

A Lastlight Goods project
Bend, Oregon
- github.com/lastlightgoods/so101-arm-lab (repo, public soon)
- SO-101 Arm Lab


========================================================================
PAGE: public/404.html
========================================================================

SO-101 Arm Lab

Fewer than 3 markers visible
404
# Holding the last good solve.
That page is not on the mat. The calibration is held until something recognisable comes back into frame.
← back to the lab


