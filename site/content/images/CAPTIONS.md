# Image captions

All images are real captures from the lab. Live frames and screenshots were taken from the running
wall dashboard on 10 Nov 2026 shortly after midnight, and the SO-101 Panel screenshots on 18 Nov 2026 at 11:43 from a separate headless Chrome at the wall screen's size; fixture frames are the 9 Nov 2026 captures
checked into `tests/scene/fixtures`. No image shows a LAN address, an email, or a person.

| file | size | caption | what it shows |
|---|---|---|---|
| `top.jpg` | 640x360 | The overhead C920 at the locked exposure: the Fiskars 18x12 in mat with four 2 in ArUco markers, the follower parked at the top edge. | Live `/cam/top.jpg`. Markers 0-3 at their inch positions, the printed mat grid, the arm base in the blue tape at the bottom centre (arm side), the hex nut as a small dark dot left of centre. |
| `front.jpg` | 640x360 | The front C920, 42 degrees down from past the far edge of the mat. | Live `/cam/front.jpg`. The mat seen obliquely with three markers in view; the black articulated arm across the frame is the lamp/camera stand, not the robot; the follower's gripper and blue-taped base at the bottom edge. |
| `wrist.jpg` | 640x480 | The icSpring wrist camera: the jaws enter from the bottom, the hex nut sits just right of the fingertips. | Live `/cam/wrist.jpg`. Strong barrel distortion (k1 -0.41 in the hand-eye solve), the mat grid, one marker corner at the lower right, the camera's green-tinted white balance before the gain match. |
| `scene.png` | 916x608 | The live mat map: the kinematic arm mask in red, the four marker exclusion squares in yellow, and one tracked object boxed in green. | `/api/scene.png` rendered by `scene_detect.py`: top-down mat with a 1 inch grid, the arm's silhouette from the follower's joints projected through the camera pose, object id 514 at (11.26, 6.33) in, the arm side labelled at the bottom. |
| `panel.jpg` | 1440x2560 | The SO-101 Panel on the portrait wall screen, default `tempest` skin (18 Nov 2026). | Headless Chrome render of `http://127.0.0.1:8765/` at the wall screen's 1440x2560: turntable header, MISSION STATUS (IDLE READY, Jetson up 4 days), six Jetson gauges, three live feeds with the CAL chip (0.1 mm, arm fit 1.1 mm, roll 0.5 degrees), the Tempest attract screen, the 3D arm with joint temperatures, the joint table, training loss, datasets and policies, the log. The Jetson terminal and Claude shell panels and the host label are blurred. |
| `panel_top.jpg` | 1440x960 | The top of `panel.jpg`: header, mission status, gauges and the three camera feeds. | Crop of `panel.jpg` used in field note 03. |
| `panel_dystopian.jpg` | 1440x2560 | The same panel in the Lastlight Goods DYSTOPIAN skin: silver only, JetBrains Mono, facility signage. | Render of `?skin=dystopian` at 1440x2560: SECTOR 01 STATUS, COMPUTE, the live feeds, the attract card "TRANSPORT. IDLE." with the silver ridge and the follower pose as a silhouette, ARM BAY, DATA VAULT. Terminal panels and host label blurred. |
| `panel_telemetry.jpg` | 1440x860 | The panel's servo telemetry view (M key). | Render with the telemetry overlay open, cropped to its content: temperature, load and current per joint, bus current against the 5 A board limit, supply voltage, commanded vs actual, the joint table, and the error-flag list (gripper temperature warnings from the 18 Nov morning teleop session). |
| `fixture_bg_empty_mat.jpg` | 640x360 | The empty-mat background frame used by the scene tests (9 Nov, before the mat was re-squared). | `tests/scene/fixtures/bg.jpg`: the earlier marker layout, arm parked at the top edge; pairs with `calib_a.json`. |
| `fixture_hex_nut.jpg` | 640x360 | One M-size hex nut on the mat at (10.97, 6.94) in, the frame the object tests measure. | `tests/scene/fixtures/obj.jpg`: same layout as the background frame, arm at the top. |
| `fixture_arm_raised.jpg` | 640x360 | The arm raised over the mat, the frame that exposed the edge-only arm rule. | `tests/scene/fixtures/top_now.jpg`: the forearm leaves the mat edge through the air; the kinematic mask was written for this case. |
| `fixture_mat_turned.jpg` | 640x360 | The mat turned round and the markers re-glued: a congruent rectangle the marker refit cannot tell from a rigid move. | `tests/scene/fixtures/new.jpg`, matching `calib_b.json` (the 21:17 calibration); the reason `mat_shift_mm` / `mat_rot_deg` exist. |
| `top_mat_bg.png` | 229x152 | The learned empty-mat background in mat space, 2 mm per pixel. | `jetson/calib/top_mat_bg.png` as saved by `POST /api/scene/background`: the warped top-down mat the detector subtracts from every frame; reloaded at server start. |
| `wristcam_view_w1.jpg` | 640x480 | Hand-eye capture w1: the wrist camera looking at a single mat marker with the follower's joints recorded. | `jetson/calib/wristcam/w1.jpg`, one of the five views that solved the wrist camera's pose (rigid residual 3.0 px on this view). |
| `wristcam_view_w3.jpg` | 640x480 | Hand-eye capture w3: another single-marker view from a different height and tilt. | `jetson/calib/wristcam/w3.jpg` (rigid residual 7.5 px on this view). |

Notes for the site builder:

- `top.jpg`, `front.jpg`, `wrist.jpg` and `scene.png` are the camera frames and map at the moment of
  capture; the front camera was aimed at the stand, not the arm, at that moment, which is also what the
  drift tool would report as `LOW CONFIDENCE` or `MOVED` (docs/camera-health.md).
- The dashboard screenshots are of a 1200x758 landscape browser window, not the 1440x2560 portrait
  wall kiosk; the ARM stage, training column and event log sit below the fold in this window.
- The two half-scale screenshots taken during the session were discarded; only the full-resolution
  PNGs are kept.
