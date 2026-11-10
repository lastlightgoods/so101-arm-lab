# Field notes from the arm lab

Seven short notes from five weeks with an SO-ARM101. Every number comes from the repository; the
source files are named at the end of each note.

## 1. The camera that fought back

We bought two Logitech C920s because they are the camera everyone has. Then we tried to freeze their
exposure so the arm sweeping through the frame would stop changing the picture, and the first attempt
came back pure white.

The revision we have (USB id 046d:08e5) does three things that a plain `v4l2-ctl` script cannot see
coming. A manual control is flagged `inactive` while its auto flag is on, so the write fails with
EACCES unless the auto flag goes first. About one second after `auto_exposure` goes from 3 to 1 the
camera forces `gain` to 255 on its own, clobbering whatever we wrote in between: that was the white
frame. And at every stream start in Manual Mode it re-derives the exposure; we watched 78 come up as
156, then 312.

So the lock became three layers. `cam_lock.py capture` lets auto settle for 4 s, reads what it chose,
locks, waits 1.5 s for the gain clobber to pass, then tunes gain until the locked picture matches
auto's mean grey within 8 levels. A resident guard checks every second and re-applies within about a
second of any drift, because LeRobot opens the cameras itself when a recording starts. A udev rule
re-applies 3 s after a camera re-enumerates with factory defaults. And `so101-record` refuses to
start unless `check --strict` passes.

On 9 Nov at 23:06 the top camera locked at exposure 38 and gain 42 where auto had chosen 83 and 109,
and the picture's mean grey stayed at 112.6 with the arm in and out of frame.

(docs/camera-settings.md, jetson/cam_lock.py, jetson/calib/cam_settings.json)

## 2. Four stickers and a cutting mat

The work surface is a Fiskars 18x12 in self-healing cutting mat, chosen because it has an inch grid
printed on it and costs very little. We glued four 2 in ArUco markers from `DICT_4X4_50` onto it at
grid positions (16,10), (2,10), (15,4) and (3,4), with the origin at the mat corner nearest the arm
and X running along the 18 in edge.

That is the whole table calibration. `cam_calib.py table` finds the four markers in one overhead
frame, snaps each to an exact square, and fits a homography from 16 corner points to mat millimetres.
The current solve, from 9 Nov at 23:06, puts the worst marker 0.19 mm from where the layout says it
is. Under 2 mm is normal; over 5 mm means someone measured a marker wrong.

A homography only knows the mat surface, so height matters. `locate` takes a pixel and a height above
the mat; a 60 mm tall box read as if it were flat lands centimetres off. We also pinned each camera's
field of view from a tape-measured lens height, 16.5 in for the top camera (69.16 degrees) and
18.375 in for the oblique front camera (68.88 degrees), and treat those numbers as belonging to the
lens, not the mount.

Nothing learned uses any of this. ACT and SmolVLA look at pixels. The calibration exists for the
scripted moves, for the map on the wall, and for being able to say where things are in inches.

(docs/camera-calibration.md, jetson/calib/top_layout.json, jetson/calib/top_table.json)

## 3. Teaching the arm where it is

Knowing where the mat is in the camera does not tell us where the arm is on the mat. For that we
switched torque off, pushed the closed jaw tip onto known grid points, and recorded the joints each
time. Forward kinematics from the URDF, which match the dashboard's 3D arm to 0.1 mm, turn those
joints into tip positions, and a rigid fit gives the base pose.

The first rigid fits left 15 to 30 mm of residual, far too much. The reason is a LeRobot convention:
it zeroes each joint at the middle of its calibrated range, which is not exactly the URDF zero. A few
degrees of offset in the lift, elbow and wrist make the kinematics read short near the base and right
far out. So the fit now solves the base pose, the table height and three joint offsets together. On
the real arm the offsets came out at shoulder_lift -3.9 degrees, elbow_flex -5.2 degrees and
wrist_flex -0.7 degrees, and the residuals fell to 0.7 to 1.1 mm on the four points of the current fit,
with an RMS of 0.69 mm. The base sits at (9.04, -3.18) in on the mat frame, heading 90.15 degrees.

One more rule came out of the jogging. The SO-101 gripper has one fixed jaw that is off the roll axis,
so the contact point moves when the wrist rolls. We jogged everything at a wrist roll of 0.46 degrees
and now command near that value; `/api/ik` defaults to it.

(docs/camera-calibration.md "Joint zero offsets", jetson/calib/top_table.json, jetson/calib/top_armfit.json)

## 4. A map of the mat

The wall dashboard draws a live top-down map of whatever is lying on the mat. `scene_detect.py` warps
each overhead frame into mat space at 0.5 px per mm (229 x 152 pixels, 2 mm per pixel), subtracts a
stored picture of the empty mat, and reports every blob between 40 mm2 and a third of the mat with
its centroid in inches, its size and its mean colour. It takes 5 to 8 ms a frame and runs twice a second.

The hard part was the arm. Our first rule said any blob touching the arm-side edge is the arm. On
9 Nov TC watched the arm reach across the mat and saw the gripper appear on the map as an object,
because the forearm had left the edge through the air. The fix masks the arm from what we know rather
than from what we see: the follower's joints go through the forward kinematics, each joint origin gets
a radius (30 to 50 mm, times 1.3 for the shadow), and the points are projected through the camera pose
as 3D points. That last part matters; an elbow 200 mm up under a camera 419 mm up is seen more than
100 mm away from the spot beneath it.

Flicker was the other enemy. A track now has to be seen in 3 consecutive detections, about 1.5 s,
before it is listed, and it survives 3 missed ones, so an object the gripper passes over keeps its id.

When we fetched the map while writing this, it showed one object, id 514, at (11.26, 6.33) in,
10 by 6 mm. That is the hex nut.

(docs/mat-scene.md, dashboard/scene_detect.py)

## 5. When the stand gets bumped

The overhead camera hangs on a stand the arm can reach, and the mat is only taped down. On the
evening of 9 Nov the mat moved 13 to 19 mm between two frames half an hour apart, and the camera has
been nudged more than once. A calibration that goes stale without saying so is worse than none.

So the four markers are detected in every frame, 2.6 ms each. When they land more than 3 mm from
where the stored homography puts them, and at least 3 are visible, a fresh homography is fitted and
used from then on. The server compares the stored and live solves; once the largest marker shift
passes 3 px, `/api/calib` serves the live solve, the overlays and the 3D stage follow within a second,
one event line is written ("overhead camera moved 11 px, re-solved from 4 markers (0.2 mm)"), and the
file is rewritten at most once a minute, keeping the arm fit because the arm did not move when the
camera did.

The camera poses heal the same way, with one thing held fixed: the field of view belongs to the lens
and was pinned once from a tape measure, so only position and orientation are re-solved, with a
5 mm / 1 degree hysteresis so noise never jitters the frustums on the wall.

There are limits, and they are reported rather than hidden. Fewer than 3 markers visible means the
last good solve is held. A mat moved more than 150 mm or 30 degrees, or markers re-glued, means
`calib_ok` goes false and the chip on the feed reads CHECK MAT.

(docs/mat-scene.md "Self-healing calibration", docs/camera-calibration.md section 6)

## 6. The nut moves in the jaws

With a calibrated camera and a fitted arm we wrote the obvious script: find the hex nut in the
overhead frame, plan nine waypoints through `/api/ik`, pick it up, carry it to a target, set it down,
measure where it landed, repeat and learn.

Two kinds of learning are in the loop. Servo sag under load, which we measured at 1 to 3.5 degrees,
is corrected per waypoint from last time's error with a gain of 0.7. We tried summing the errors
first and overshot by 24 mm, so only the last error is fed forward. The landing error, measured by the
camera, shifts the place target with a gain of 0.5.

The sag learning worked. Over the first four runs on 10 Nov the hover error fell from 3.17 to 0.97
degrees and the lift error from 4.58 to 0.97. All four picks succeeded, each run took about 18 s, and
the arm reaches within about 3 mm of a commanded point.

The nut still landed 31, 10, 39 and 21 mm from the target. The reason is mechanical and we are not
going to pretend otherwise: the SO-101 gripper has one moving jaw, and a 10 mm nut shifts inside it
during the grasp, so the part is no longer where the tip is. The next step is to look at the part in
the jaws with the wrist camera before placing, which is why that camera now has a solved hand-eye
pose (1.2, 67.9, -11.3) mm from the gripper link, good to a few millimetres and about 1.5 degrees.

(tools/pick_loop.py, jetson/pick_seq.py, jetson/calib/pick_state.json, README.md, jetson/calib/wrist_campose.json)

## 7. 600 tests and no robot

Most evenings the arm is busy, or the Jetson is training, or the cameras are being re-aimed. The code
still has to be checked, so the test suite does not touch any of it. `python -m pytest tests -q`
collects 600 tests and takes about 3 minutes with nothing plugged in.

The calibration tests render checkerboards and ArUco tables through a synthetic camera with known
intrinsics, distortion and pose, then check what the tool gets back: the lens model within 1.5 px,
the camera position within 6 mm, pixel-to-mat within 1.5 mm, and that the arm fit refuses a mirror
image. The hand-eye solver is run on a synthetic wrist camera placed on the real kinematics and must
come back within 2 mm, 0.5 degrees and 2 percent of focal length. The camera lock is driven against a
fake `v4l2-ctl` that reproduces every behaviour we measured on the real C920, the gain clobber
included. The bus plugin runs on a fake Feetech bus that drops packets on cue.

The dashboard page gets the same treatment with no framework and no npm: a Python stub server answers
every API route from objects a test can mutate, a small Node driver opens the page in headless Chrome,
blocks every external request, freezes the page's timers, and fails any test whose page threw or
logged an error. Forty-five of those run in about 35 s. The real frames in the fixtures are from
9 Nov: an empty mat, one hex nut at (10.97, 6.94) in, and the arm raised.

(tests/, tests/ui/README.md, README.md)
