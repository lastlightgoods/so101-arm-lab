# Camera health: frame rate and drift checks

Two small tools on the Jetson keep an eye on the three recording cameras (top and front Logitech C920 at
640x360 MJPG 30 fps, icSpring wrist at 640x480 YUYV 30 fps, the `SO101_CAMS` configuration in `jetson/env.sh`):

| tool | question it answers | command |
|---|---|---|
| `jetson/cam_health.py` | Do all three cameras really deliver 30 fps when opened together, without stalls or repeated frames, at the size we asked for? | `so101-cams-health [seconds]` |
| `jetson/cam_drift.py` | Has a camera (or the mat under the overhead camera) moved since the reference picture was taken? | `so101-cams-drift set` / `so101-cams-drift check` |

Both hold `~/so101/.cams_busy` while they open a device, so the Mac dashboard's ffmpeg feeds pause for the run and come
back by themselves afterwards. Both print a human table by default and a machine-readable report with `--json`, and both
use the exit status (0 fine, 1 something is wrong, 2 could not even get a frame) so they can sit in a pre-recording checklist.

Tests: `tests/jetson/test_cam_health.py` and `tests/jetson/test_cam_drift.py` (synthetic timestamps, a fake capture
source, shifted and rotated copies of the real overhead frame `tests/scene/fixtures/new.jpg` with its calibration
`calib_b.json`, and a tiny http server standing in for the dashboard). No hardware needed.

## cam_health: frame rate and dropped frames

`so101-cams-health` (default 10 s, after a 1 s warm-up that is not counted) opens the three cameras **at the same time**
in three threads, exactly as `lerobot-record` does (fourcc, then width/height, then fps), and timestamps every frame as it
arrives. It never records anything.

Per camera it reports:

- **fps**: frames in the measurement window divided by the time between the first and the last of them.
- **p50 / p95 / max**: percentiles of the interval between consecutive frames, in ms. A healthy 30 fps camera shows p50 of
  about 33 ms (the C920 shows 32.1 because of how the driver paces) and a max under 40 ms.
- **dups**: consecutive frames whose strided sample (every 8th pixel) is bit-identical. A live sensor never produces two
  identical frames, so any duplicate means the driver handed the same buffer out twice. `--dup-mad 1.5` also counts frames
  that differ by less than that many grey levels on average, for a frozen camera with a noisy sensor.
- **stalls**: intervals longer than `--stall-ms` (100 ms, three missed frames).
- **size/fourcc**: what the driver negotiated, and the size of the frames actually received.
- **cpu**: load average before and after, and the overall CPU busy percentage during the run (psutil).

Fail (exit 1) when any camera's fps < `--min-fps` (28), any interval is longer than `--max-stall-ms` (250) or the
delivered size differs from the requested one. Exit 2 when a camera cannot be opened at all (unplugged, or held by a
lerobot job that did not release it; `so101-stop` clears that).

Measured 9 Nov 2026 (dashboard feeds paused, nothing else running, load 1.1 on 6 cores, 12 % CPU during the run):

```
cam    size/fourcc         fps     p50     p95     max  dups stalls  verdict
top    640x360 MJPG      30.00    32.1    36.7    39.7     0      0  OK
front  640x360 MJPG      30.00    32.1    36.4    37.8     0      0  OK
wrist  640x480 YUYV      22.29    44.0    48.0    88.1     0      0  FAIL fps 22.29 < 28.0
```

### The wrist camera runs at 22.5 fps, not 30

The icSpring advertises 30 fps at 640x480 YUYV (its only format) but delivers a steady 44 ms frame period (22.1-22.6
fps) with no stalls and no duplicates. Checked and ruled out:

- **USB bandwidth / topology**: all three cameras sit on the same USB 2.0 hub chain (`lsusb -t`: Bus 01 480 Mb/s, Realtek
  4-port hub, then a Huasheng 4-port hub with the cameras on ports 1-3; the USB 3.0 hub on Bus 02 carries nothing). Raw
  YUYV 640x480x30 is 18.4 MB/s and each camera's biggest isochronous alternate setting asks for 3x1024 bytes per
  microframe, so three streams are tight on one 480 Mb/s bus in principle. But the wrist is at 22 fps **alone** on the bus
  as well (top and front closed), and the two C920s keep 30.00 fps beside it, so bandwidth is not what limits it.
- **Exposure**: with manual exposure at 15.6 ms it is still 22.6 fps; with 30 ms it drops to 18.7 fps (52 ms period), so
  the sensor adds its read-out time on top of the exposure and never gets under 44 ms. Anti-flicker 50/60 Hz/off makes no
  difference. Controls were restored to their defaults afterwards (aperture-priority auto exposure, 50 Hz).
- **CPU**: 12 % busy on 6 cores while all three stream; `top`/`front` decode MJPG at 30 fps without trouble.

So it is the camera's own firmware pacing, and nothing in the cabling or the Jetson. What it means for datasets:
`lerobot-record` runs its loop at 30 Hz and takes the newest wrist frame each tick, so roughly one wrist frame in four in a
recorded episode is a repeat of the previous one (the top and front views are genuine 30 fps). ACT trains fine on that,
it just means the wrist view carries 22 fps of information.

Recommendation, in order of effort: (1) accept it and note it in the dataset card; (2) if the wrist view matters for a
fast task, replace the icSpring with a small UVC module that offers MJPG at 640x480 30 fps or better (any 1080p "USB
camera module" with an OV2710/IMX sensor does; keep the same cable length), then update `CAM_WRIST`, `SO101_CAMS` and
`JETSON_CAMS` in `dashboard/server.py`; (3) moving the wrist camera to the empty USB 3.0 hub will not help this camera
(it is USB 2.0 and not bandwidth-bound) but would free headroom on the shared bus if a fourth camera is ever added.

## cam_drift: has a camera moved?

`so101-cams-drift set` saves one frame per camera to `~/so101/calib/ref_<cam>.jpg` (plus `ref_meta.json` with the time,
source and, for the top camera, where the ArUco markers were seen). Take it with the arm at rest, right after a calibration
or whenever the cameras are known to be right. `so101-cams-drift check` grabs a fresh frame per camera and compares.

### What it measures

For **every camera**: the whole-frame shift between the reference and the fresh frame. `cv2.phaseCorrelate` on
Hanning-windowed grayscale gives a translation and a **response** (1.0 = the same picture, 0.0 = nothing in common; two
live frames of the same scene a few seconds apart score 0.96-1.08, a different scene 0.0-0.15). An ECC Euclidean fit
seeded with that translation refines it to sub-pixel and adds the in-plane **rotation** (degrees, counter-clockwise
positive). `dx, dy` are the displacement of the image centre, positive when the scene content moved right / down in the
new frame. A shifted copy of a real frame is recovered within 0.2 px, a 1.5 degree roll within 0.05 degree.

For the **top camera** additionally: every marker in `~/so101/calib/top_table.json` (`layout_markers`) is projected to
the pixel where the calibration says it should be (inverse `homography_px_to_table`, or the full pose when intrinsics
are present) and compared with the ArUco centre actually detected. Per marker: expected, detected, displacement and its
norm; overall: mean displacement, max norm, and the spread between markers. This is the precise number for the overhead
camera (sub-pixel; the calibration itself fits to 0.12 mm) and it is what the dashboard's mat map and `scene_detect`
depend on.

### Reading the verdicts

Threshold `--threshold` 6 px (about 4 mm on the mat for the overhead camera), `--min-response` 0.3.

| status | meaning | what to do |
|---|---|---|
| `OK` | shift within the threshold; for top, markers within the threshold of the calibration | nothing |
| `OK ... frame shift comes from scene changes` | top only: the frame shifted but the markers are exactly where the calibration projects them | nothing: the camera has not moved relative to the mat; the arm, objects or lighting changed the picture |
| `MOVED` | front / wrist: the frame shifted more than the threshold | re-aim the camera to match `ref_<cam>.jpg`, or `set` a new reference if the new aim is intended |
| `CAMERA MOVED` | top: frame and markers moved together by the same shift (and roll) | `so101-calib table` (a camera-only re-solve keeps the arm fit), then `so101-cams-drift set` |
| `MAT MOVED` | top: the markers moved as one body but the frame around them did not follow | put the mat back on its outline, or re-solve the table **and** refit the arm (`cam_calib.py armfit`) |
| `MARKER MOVED` | top: the markers no longer move as one rigid body (worst id named) | re-stick that marker on its measured centre, or re-solve the table |
| `LOW CONFIDENCE` | response below 0.3 (and no usable markers): the scene changed too much to compare, or the camera looks somewhere else entirely | look at the picture; re-aim, or `set` a new reference if the scene legitimately changed |
| `BAD EXPOSURE` | the fresh frame is blown out (mean grey > 240), black (< 12) or flat (contrast < 8): exposure/gain stuck or the lens covered | `v4l2-ctl -d $CAM_x --list-ctrls`: `auto_exposure` should be 3 (aperture priority) and `gain` 0 on the C920s; fix, then check again |
| `BAD REFERENCE` | the saved reference itself is blown out / black / flat | `set` again with the camera exposing properly (`set` now refuses to save such a frame) |
| `NO FRAME` / `NO REFERENCE` | nothing to compare (exit 2) | check the camera / run `set` |

Why the markers outrank the frame for the top camera: a moved camera moves *everything* in the picture, markers
included, by one rigid transform; a moved mat moves the markers but not the table, arm base and background behind them,
so the frame's correlation peak stays put (or weakens); a knocked marker breaks the markers' rigidity among themselves.
A frame shift *without* a marker shift therefore means the camera did not move relative to the mat: either the scene
content changed, or camera and mat moved as one unit (someone bumped the table), and in both cases the pixel-to-mat
calibration is still right. The only case the markers cannot see is the mat and camera moving together while the arm
did not, which only happens when the arm is not on the same table.

The **wrist camera** rides on the arm, so its reference and check only compare at the same arm pose. Run both with the
arm at rest; a wrist `MOVED` after a teleop session usually means the arm did not come back to exactly the same rest pose
(check the dashboard's joint readout), not that the camera slipped on its mount.

### What the first real run found (9 Nov 2026, 22:40)

`set` from the devices, arm at rest: top saved with 3 of 4 markers visible (one under the gripper), 0.72 px from the
calibration; `check` a few seconds later: top shift 0.2 px, markers within 1.85 px, OK. The same reference checked
against a dashboard still taken half an hour earlier: 4/4 markers within 1.16 px, shift 0.25 px, so the dashboard
path and the device path agree. The front and wrist references from that run were blown out (both C920s had been
switched to manual exposure with gain 255 by the camera-control work going on at the same time, and the icSpring
opens white and needs ~1.5 s to settle): that is why `set` now refuses such frames and streams two seconds before
grabbing. The wrist reference was re-taken from the device at 22:47 (mean grey 120, check: 1.5 px, response 0.98,
OK). The front reference was re-taken from the last correctly aimed dashboard still: between 22:15 and 22:40
the front camera was physically re-aimed (it now looks at the pegboard and the lamp instead of the arm), which the
check reports as `LOW CONFIDENCE` until it is pointed back at the arm, then as `MOVED (dx, dy)` to guide the last bit.

### Running it

On the Jetson (devices, busy flag held automatically; the dashboard feeds pause for about two seconds per camera):

```
so101-cams-drift set            # new reference, arm at rest
so101-cams-drift check          # exit 0 = all in place
so101-cams-drift check --json
```

From the Mac, without touching the devices, using the dashboard's stills (the references must be local; copy them once
with `scp jetson:~/so101/calib/ref_\*.jpg jetson:~/so101/calib/top_table.json ~/so101-calib/`):

```
python jetson/cam_drift.py check --url http://localhost:8765 --calib-dir ~/so101-calib
```

The dashboard's frames are the same cameras at the same resolution (its ffmpeg streams 640x360 / 640x480 at 15 fps,
re-encoded to JPEG), so the shift numbers match the device path to within the JPEG noise, about 0.1 px.

Offline, against saved frames (`DIR/<cam>.jpg`, e.g. a frame pulled out of a dataset): `--from-dir DIR`.

Options: `--cams top,front`, `--threshold 6`, `--marker-threshold` (defaults to `--threshold`), `--min-response 0.3`,
`--no-rotation` (translation only, faster), `--calib PATH` (another table calibration), `--calib-dir DIR`.
