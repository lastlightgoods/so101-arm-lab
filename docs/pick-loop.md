# The closed-loop pick

`tools/pick_loop.py` picks the nut off the mat and places it on a target, learning a little each run (per-waypoint servo sag,
landing offset). Since 10 Nov 2026 it is closed loop by default: the arm is commanded, read back, and corrected twice before the
jaws close. This page is the rule, the numbers behind it, and what gets logged.

## The rule

**Command, read back, correct twice, jaws at 35, fixed jaw beside the nut.**

1. Hover over the commanded nut point. Read the reached joints back from `pick_seq.py`'s rows, run them through forward kinematics
   with the arm fit's joint offsets (`top_table.json: arm_fit.joint_offsets_deg`, `tip_offset_mm` when present) into mat inches.
   Correction c1 = aim point - reached. The aim point is the nut moved 10 mm back along the jaw-opening direction, so the fixed jaw
   lands beside the nut and the nut sits inside the open gap (see "The grasp aim" below).
2. Hover at the corrected point, descend to grasp height with the jaws open, read back again: correction c2 (the residual), and the
   tip height h above the mat. If h > 6 mm the grasp target is lowered by min(h - 3, 6) mm through `/api/ik`.
3. Grasp at the twice-corrected point, close, lift, carry, place, open, lift, rest. The carry and place are unchanged: the moving jaw
   pushes the nut back against the fixed jaw, where it was held before.

Each stage is one `pick_seq.py` bus session over ssh, at 50 deg/s. The wrist camera is not in the loop: the readback is the sensor.

## Why (measured 10 Nov 2026, real arm, nut at about mat (11.1-11.6, 6.3-6.5) in)

| what | measured |
|---|---|
| open-loop hover vs command (FK of the reached joints) | commanded (11.44, 6.26) -> reached (11.15, 6.35); commanded (11.63, 6.29) -> reached (11.32, 6.42): **5-8 mm short along the reach direction** |
| wrist camera prediction of the nut's pixel from the hand-eye pose + the REACHED joints | within about 10 px of the real nut at 147 mm (17 px on the frame kept as a test fixture), so readback + arm fit + hand-eye agree; it is the open-loop command that is off |
| after two readback corrections, FK of the final grasp pose vs the commanded nut | within 0.1 mm |
| jaws opened to 35 | nut grasped; landed 6.2 mm from the target |
| jaws opened to 45, same corrected pose | failed twice: the moving jaw's closing arc flicked the nut 5-13 mm before the fixed jaw met it |
| dark-blob nut finder on the wrist frame | picked the gripper's shadow; the nut is a small bright ring with a dark hole, ~36 px across at 147 mm |

The first correction is the whole shortfall (7 mm or so); the second trims what is left (well under a millimetre on the fake arm
in `tests/tools/test_pick_closed_loop.py`, 0.1 mm on the real one). The shortfall is systematic enough that one correction would
nearly do, but the second pass is also where h is measured at grasp height, which the hover cannot tell you.

## The grasp aim (10 Nov 2026)

The arm fit's tip is the point where the closed jaws meet, which is the fixed jaw's inner face. The moving jaw swings away from it
when the jaws open. Aiming that tip at the nut's centre put the fixed jaw ("thumb") on the nut. TC watched it push the nut forward
and, on a later cycle, land on top of it. In the first10b session every pick from the target spot failed, and that is where the jaws
open straight forward.

| what | measured |
|---|---|
| wrist frame at the grasp, jaws at 35, old aim | the nut's centre 23 px past the fixed jaw's tip, overlapping its edge |
| jaw-opening direction on the mat | gripper_link's +x axis; within 2 deg of the URDF moving-jaw joint rotated to 35 (`tests/tools/test_pick_aim.py`) |
| URDF moving-jaw tip at jaws 35, from the fixed tip | 58-61 mm on the mat |
| hand-eye projection, probe pose | fixed tip at pixel (317, 317), moving tip at (587, 346); the real frame shows the jaws there |
| scale at the grasp depth | 10 mm along the jaw direction = 48 px; the nut's ring is about 20 px (4 mm) in radius |

`GRASP_AIM_MM = 10`: about half the nut plus 6 mm of clearance, more than first10b's 5.1 mm median landing error. The direction is
computed from the hover readback each pick (`jaw_dir_mat`). `--grasp-aim-mm 0` restores the old aim. Every record and MLflow run
carries `aim_mm`, `jaw_dir` and `aim_in`; the corrections and the final residual are measured against the aim point.

## The wrist tilt (11 Nov 2026)

The grasp height is `GRASP_H_IN` = 1.0 mm by the IK. At 0.12 in (3 mm) the jaws closed about 2 mm above the mat over a ~3 mm thick nut and rode over it; a
touch-off put mat contact at an IK height of about 0.5 mm. At 1.0 mm (16:20) the jaws gripped the nut, but TC saw the front finger (the moving jaw, on the side
away from the base) drag on the mat and pull the fixed thumb forward: "the wrist needs to tilt a bit so the front finger doesn't drag".

| what | measured |
|---|---|
| tip swing per degree of wrist_flex at the grasp | about 2.8 mm; negative swings the jaw tips away from the base |
| 16:35 test pick, tilt -3 deg, grasp at 1.0 mm | picked and placed, landed 5.9 mm from the target |

`GRASP_TILT_DEG = -3` is added to wrist_flex on pregrasp, grasp, close1, close2 (and the `--fast-close` close), not on the hovers, lift, place, open or rest. Each tilted
waypoint's IK target is shifted back by the tilt's own tip shift (forward kinematics through the arm fit, two iterations), so the tilted tip lands where the vertical one
would; the wrist_flex clamp of 93 deg still applies after the tilt. The plan's `desired` carries the tilt, so the sag learning does not learn it as sag.
`--grasp-tilt-deg 0` gives the vertical gripper of before; the value is an MLflow param (`grasp_tilt_deg`). `build_plan` takes `tilt_deg` and, for the FK, `cc`/`table`
(the closed loop and the open loop pass theirs; otherwise `calib()` loads `jetson/calib/top_table.json` once).

The tilt shifts the tip in x/y only by design; with the arm fit it also moves the tip height by a millimetre or two (the 9 Nov fit: about -1 to -3 mm near, about +2 mm far), which the closed loop's raise/lower rule then sees in the readback.

### pregrasp_up

The final stage starts with `pregrasp_up`, a waypoint in joint space: pregrasp's final command (IK, tilt, bias) with `shoulder_lift` and `elbow_flex` each `APPROACH_DEG` = 8 deg lower. Lifting is the servo's bad side (P-only, no integral: it stops short), and
the 11:56 sweep on 11 Nov 2026 measured 1.7-4.7 deg of droop on the shoulder and 3 deg on the elbow. A tilt pick whose pregrasp sagged 4.35 deg then went down by a commanded ~1.4 deg on both joints, less than the droop, so the real shoulder move was a reversal from the bad side
and the grasp landed 6.4 mm high and 4 mm past the aim, the fixed thumb on top of the nut. A final move that increases both joints landed within 0.04-0.6 deg. With the arm fit at 5 mat spots `pregrasp_up` sits 50-63 mm above the mat and 33-40 mm back, the straight joint path down to
pregrasp never goes below pregrasp's 10 mm, and 8 deg beats the largest measured droop. It is skipped by the bias learning (`NO_LEARN`), so it never changes a bias; the guard checks it like every waypoint.

### The pan back-off (`hover_pre`, and `pregrasp_up`'s pan)

The `shoulder_pan` servo has stiction: every move stops short in the direction it was moving. The live pick of 11 Nov 2026 17:17 (closed loop, -3 deg tilt) read back, on pan mostly: stage 1 hover commanded 9.280 in, reached 8.830 (11 mm short); stage 2 grasp pose after c1
commanded 9.728, reached 9.613 (3 mm short); stage 3 final grasp after c2 commanded 9.393, reached 9.548: the arm moved 1.6 mm of a 5.6 mm correction, about 4 mm (roughly 1 deg at ~230 mm reach) short, and the grasp landed 7.5 mm off the aim. The readback corrections c1 and c2 assume the
next arrival behaves like the last, which only holds when every arrival at a pan target comes from the same side. Pan has no gravity side, so the plan picks one: every nut-side target is approached from `PAN_APPROACH_DEG` = 3 deg lower pan, so the last pan move is always an increase (3 deg beats the ~1 deg stiction band).

- stages 1 and 2 start with `hover_pre`: hover_nut's command (IK, bias) with `shoulder_pan` 3 deg lower, same height, jaws as at the hover (it carries the long move from rest, 1.6 s; `hover_nut` then takes 0.5 s). The abort stage opens over the spot the same way (`hover_pre`, `hover_nut`, `rest`);
- `pregrasp_up` also takes the 3 deg off pan, so `pregrasp_up` -> `pregrasp` increases pan, shoulder_lift and elbow_flex together;
- `hover_pre` and `pregrasp_up` are in `NO_LEARN` (table `BACKOFF`): they never change a bias. The guard checks both like every waypoint; with the 9 Nov arm fit the 3 deg is 17-20 mm sideways at the hover, so a nut within about 1 in of the mat's bottom edge (hover_pre) or of the left edge / 2 in of the top edge (pregrasp_up) can now trip the workspace check where the nut itself passes.

## Gripping an elongated object across its long axis (`--grasp-roll`, 12 Nov 2026)

The jaws close along `jaw_dir_mat` (gripper_link +x on the mat, fixed jaw toward the moving jaw). With wrist_roll fixed that is roughly the reach direction, so a ~6 x 13 mm
spacer lying along the reach was closed on lengthwise and pushed 1.8 in (11 Nov, 18:2x). `--grasp-roll auto|off|DEG` (default `off`, nothing changes) turns the wrist so
`jaw_dir` is perpendicular to the object's long axis.

* **The axis.** `/api/scene` objects carry `w_mm >= h_mm` and `angle_deg` from `cv2.minAreaRect` on the rectified mat image. `detect()` negates cv2's image-frame angle (image rows run
  against mat +Y), so `angle_deg` is the long side's direction **counter-clockwise from mat +X, in mat millimetres**: the frame of `x_in / y_in` and `jaw_dir_mat`. `long_axis_mat(obj)`
  (`dashboard/scene_detect.py`) returns `(cos a, sin a)`, or `None` when long/short < 1.3 (round: no roll). `tests/scene/test_long_axis.py` draws bars at known mat angles and runs them through
  `MatScene.detect`.
* **The roll.** `roll_to_perpendicular` scans the roll offset over -90..90 deg and bisects the zero of `jaw_dir . axis`, with `jaw_dir` from forward kinematics through the arm fit at the IK pose of
  the grasp (tilt included): the smaller of the two solutions, `|offset| <= 90`. Measured with the 9 Nov fit: **+wrist_roll turns `jaw_dir` counter-clockwise on the mat, about 1.02 deg per degree**.
  `auto` uses the scene object nearest the nut position (within 15 mm); no object or a round one means no roll. A number is the offset in degrees from the arm fit's roll.
* **The plan.** `build_plan(..., roll_deg=)` passes `&roll=` (the fit's roll plus the offset) to `/api/ik` on hover_pre, hover_nut, pregrasp_up, pregrasp, grasp, close1/close2/close and lift (hover_pre and pregrasp_up
  copy hover_nut and pregrasp, so the pan and approach back-offs and the biases work as before). The tilt's tip-shift loop passes the roll too, so the tilted, rolled tip still lands on the commanded point
  (0.3 mm xy and height, tested at six mat spots). The carry (hover_tgt onward) and rest keep the fit's roll. Roll off sends no `&roll=` and gives exactly the old plan.
* **The aim.** `closed_loop_pick` reads `jaw_dir` from the hover readback, which is rolled, so the nut still sits `GRASP_AIM_MM` inside the open gap along the rolled jaw direction.
* **Logging.** The `grasp_roll` param (`auto`, `off` or the number) and the `grasp_roll_deg` metric go to MLflow; a `grasp_roll` stage line shows the object, axis and roll. `tools/autoloop.py --grasp-roll` passes it on (default `off`).

## Running it

```
~/so101/.venv/bin/python ~/so101/tools/pick_loop.py --runs 4                    # closed loop, jaws 35, 50 deg/s
~/so101/.venv/bin/python ~/so101/tools/pick_loop.py --runs 4 --wrist-check      # plus the camera check at hover (below)
~/so101/.venv/bin/python ~/so101/tools/pick_loop.py --runs 4 --open-loop        # the single-run plan of before, 80 deg/s
```

| flag | default | meaning |
|---|---|---|
| `--closed-loop` / `--open-loop` | closed | staged runs with readback corrections, or one run of the whole plan |
| `--grasp-aim-mm MM` | 10 | put the nut this far inside the open gap; 0 aims the fixed jaw's tip at the nut (before 10 Nov) |
| `--jaw-open N` | 35 | gripper opening at hover / grasp / open / lift2. Keep 35: 45 flicks the nut |
| `--max-deg-s` | 50 closed, 80 open | `pick_seq.py` speed cap |
| `--wrist-check` | off | at hover, fetch the wrist frame and report where the camera sees the nut vs the command (a check only) |
| `--no-jaw-check` | check on | skip the nut-in-jaws check after the lift |
| `--jaw-window x0 y0 x1 y1` | 300 300 380 360 | wrist-frame window (640x480 px) where the ring detector (`wrist_check.find_ring`) must find the nut between the jaw tips |
| `--fast-close` | off | one `close` waypoint (gripper 0) instead of the two-step close |
| `--reset` | off | one pick that moves the nut to a random clear spot (see Reset) instead of the alternating targets; `--runs` is ignored |
| `--safe-box x0 y0 x1 y1` | 6 4 13 8 | mat inches the reset may drop the nut in; `--seed N` makes it repeatable |
| `--lost-after N` | 3 | consecutive misses of the nut before the loop stops with `lost` |
| `--target X Y`, `--alt X Y`, `--runs`, `--gain`, `--outcome-gain`, `--state` | as before | targets alternate alt / target; sag and landing learning; the state file |

The closed loop needs the arm fit in `jetson/calib/top_table.json` (`tools/jog_capture.py --fit`); without it the tool says so and
stops (use `--open-loop`). Per pick it prints one JSON line per stage, then the history record. This is the fake arm of the
test suite (reaches 7 mm short along base-to-tip; the real arm's first correction on 10 Nov was (7.4, -2.3) mm):

```
{"run": 1, "stage": "hover", "cmd_in": [11.44, 6.26], "reached_in": [11.36, 5.996], "h_mm": 38.1, "corr1_mm": [2.0, 6.7]}
{"run": 1, "stage": "grasp_pose", "cmd_in": [11.52, 6.524], "reached_in": [11.44, 6.26], "h_mm": 3.0, "corr2_mm": [-0.0, -0.0], "lowered_mm": 0.0}
{"run": 1, "stage": "final", "cmd_in": [11.52, 6.524], "reached_in": [11.44, 6.26], "h_mm": 3.0, "residual_mm": [0.0, 0.0], "sag_deg": {"grasp": 1.61, "close1": 0.0, "close2": 0.0, "lift": 1.77, "hover_tgt": 1.66, "place": 1.49, "open": 0.0, "lift2": 1.66, "rest": 0.0}}
{"run": 1, "target_in": [9.0, 6.0], "nut_in": [11.44, 6.26], "picked": true, "landed_in": [9.1, 5.95], "err_mm": 2.8, ..., "mode": "closed", "jaw_open": 35, "corr1_mm": [2.0, 6.7], "corr2_mm": [-0.0, -0.0], "grasp_h_mm": 3.0, "grasp_lowered_mm": 0.0, "final_residual_mm": 0.0, "wrist_offset_mm": null, ...}
```

Sag learning (`biases` in the state file) is driven by desired-vs-reached of the FINAL stage only; the hover readings of stages 1 and 2
are already consumed by the corrections and would double-count. The history record carries `mode`, `jaw_open`, `corr1_mm`, `corr2_mm`,
`grasp_h_mm`, `grasp_lowered_mm`, `final_residual_mm` and `wrist_offset_mm`; the same numbers go to MLflow as metrics of the pick's run
in `so101-picks` ([mlops/tracking.md](mlops/tracking.md)).

## Two-step close, jaw check, finder, reset

- **Close**: waypoints `close1` (gripper 20, settle 0.4 s) then `close2` (gripper 0, settle 0.6 s): the moving jaw meets the nut slowly and does not flick it. `--fast-close` keeps the single `close`. Gripper-only waypoints are skipped by sag learning (the arm does not move there), so only arm-moving waypoints get a bias.
- **Jaw check** (after `lift`; the grip check below decides `in_jaws`): one wrist frame, ring detector inside `--jaw-window`. Held -> carry. Not held -> open, back to the hover, redo the readback-corrected grasp ONCE. Not held again -> open over the spot, rest, record `picked=false, reason="not_in_jaws"`, and the loop stops. A frame that cannot be fetched or decoded (including an empty buffer) is "unknown": the nut is carried anyway, the note is logged, and the check never raises. `--no-jaw-check` disables it.
- **Finder** (`find_nut_anywhere`): when the 70 px search near the last spot misses, the scene map (`/api/scene`) is asked for objects with 8 <= w_mm <= 20 and h_mm <= 20 (nearest to the last spot first); failing that, a dark-blob search over the WHOLE mat with the marker squares (+10 mm margin) masked. Markers come from `jetson/calib/top_layout.json` when present, else the table calibration's `layout_markers`. After `--lost-after` consecutive misses the loop stops with `stop: "lost"`; any find resets the count.
- **Reset** (`--reset`, `reset_nut`): finds the nut anywhere, picks a random target inside the safe box at least 40 mm from every other scene object, 50 mm from every marker square centre and 60 mm from where the nut is now, picks it up (closed loop, jaw check included) and places it there. Rejection sampling, 200 tries, then `SystemExit` with the reason (clear the mat or widen `--safe-box`). The record has `mode: "reset"`; sag learning carries over, place offsets are left alone.

## The final-pose correction (`final_corr`, 11 Nov 2026)

Stage 2 reads the tip height at the grasp and the lower/raise rule decides from it, but the final stage arrives differently (`pregrasp_up` -> `pregrasp` -> `grasp`) and its readback height differed from stage 2's by up to 4 mm (live, 11 Nov 22:2x: stage 2 read 1.6 mm, final 5.4; stage 2 read 5.4 (lowered 4.4), final 2.5). Final xy residuals were 1-4 mm, and several misses closed the jaws 3.4-7 mm above a ~3 mm nut. So the correction measured in stage 2 does not transfer, and the final stage is read where it matters.

With the persistent pick_seq session (`--session`, the default; a stage costs ~0.2 s instead of 5.8 s) the final stage is split at the grasp:

1. `pregrasp_up, pregrasp, grasp`: read the tip back (`fk_mat` of the grasp row); stage line `final`.
2. If the height is outside 0.5-2.0 mm, or the xy residual against the aim is over `FINAL_XY_TOL_MM` (1.5 mm): ONE corrective move, the `grasp` waypoint alone with the jaws open, at the corrected point. xy = commanded + (aim - reached), capped at 4 mm (`FINAL_XY_CAP_MM`); height target 1.0 mm (`FINAL_H_TARGET_MM`), lowered/raised from the current command within the stage-2 caps (6 mm lower, 5 mm raise, `LOWER_CAP_MM`). The move starts at the grasp pose and travels a few mm at grasp height: no sweep at the mat. Never more than one per attempt (a retry gets its own). Read back again; stage line `final_corr` with `corr_mm`, `lowered_mm`, `h_before`, `h_after`, `residual_mm`.
3. `close1, close2, lift` at the final (corrected) pose, then the grip check, then the carry (or the abort).

Session off (`--no-session`, or a dead session): the split would cost a connect per stage, so the single `pregrasp_up .. lift` stage of before runs and nothing is corrected. `--no-jaw-check` with the session splits the same way (`close1, close2, lift` plus the carry in one stage after the correction). Record and MLflow: `final_corr_mm` (0.0 = checked, no move; None = not split), `final_h_before`, `final_h_after`; `final_residual_mm` is the reading after the correction. The learned grasp bias comes from the corrected row (the pre-correction grasp row is not learned).

## The grip check (object-agnostic, 11 Nov 2026)

`held = (gripper readback after the lift > GRIP_EMPTY_MAX) OR (the wrist-camera ring check says in_jaws)`, `GRIP_EMPTY_MAX = 2.0` on the 0-100 gripper scale. Measured live 11 Nov 2026:

| holding | gripper after the lift |
|---|---|
| nothing | 0.5-1.5 |
| a nut across its flats, or a 6 mm spacer | 4.0-4.8 |
| a wing nut by a thin wing | 1.3-1.5 (looks empty; the camera has to say held) |

Only the reading after the lift counts: once `close2` read 4.49 and the lift read 0.54 (the nut slipped out on the lift). `jaw_check(..., rows=)` takes the stage rows and reads the `lift` row; the camera check itself is unchanged. The `jaw_check` stage line carries `in_jaws` (the verdict), `gripper` and `camera` (the ring check's True/False/None); the record has `gripper` and `camera`, MLflow `gripper_after_lift`. Gripper low and camera unknown (no frame) stays unknown and is carried, as before. A False verdict takes the same retry and abort paths.

## The wrist-camera check (`tools/wrist_check.py`)

Not a controller: it says whether camera, arm fit and command agree. At hover the commanded nut point is projected through the arm's
FK of the REACHED joints, the hand-eye pose (`jetson/calib/wrist_campose.json`, [camera-calibration.md](camera-calibration.md) section 7)
and the camera model to a pixel. The nut is searched for near that pixel (80 px) as a **bright ring with a dark centre**: a mean-free
annulus template (ring +1, hole -1) correlated over the ROI at radii 12-26 px; the gripper's shadow anti-correlates and a bare mat square
scores about zero, so only a ring passes the 0.35 threshold (the real nut scored 0.49). Its centre is back-projected onto the mat plane
and the offset to the command reported, with an annotated JPEG in `outputs/run/wrist_check_<session>_r<n>.jpg` (expected = yellow,
found = red). No ring, no frame, no pose: `None`, the pick goes on. Offline, on a saved frame:

```
tools/wrist_check.py --image frame.jpg --joints '{"shoulder_pan": 14.9, "shoulder_lift": 17.5, "elbow_flex": 1.7, "wrist_flex": 80.4, "wrist_roll": 0.6}' --nut 11.44 6.26
{"ok": true, "uv_expected": [334.5, 255.8], "uv_found": [350.0, 262.0], "r_px": 18, "score": 0.495, "px_off": 16.7, "depth_mm": 146.7, "nut_seen_in": [11.547, 6.41], "offset_mm": [2.7, 3.8], "offset_norm_mm": 4.7, ...}
```

(that is the 10 Nov frame; the nut sat at about (11.5, 6.4), the command said (11.44, 6.26).)

## Tests

`tests/scene/test_long_axis.py` (angle convention) and `tests/tools/test_grasp_roll.py` (roll, plan, aim, CLI, autoloop) cover --grasp-roll.

`tests/tools/test_pick_closed_loop.py` (fixtures in `tests/tools/pickhelpers.py`): a fake arm on the real kinematics answers `/api/ik`
and "reaches" every waypoint 7 mm short along the base-to-tip direction; the staged corrections must converge (c1 is the 7 mm, c2 under
1 mm, the final grasp within 1 mm by FK), the grasp lowering triggers only above 6 mm and caps at 6, open loop makes one `pick_seq` call
at 80 deg/s, the history record and the MLflow metrics carry the new fields, jaws default to 35. For the check: the ring finder on a
synthetic frame (ring plus a dark shadow blob) and on the real 10 Nov frame, the project / back-project round trip within 0.5 mm at three
poses, the CLI, and that a bad frame, a missing pose or a dead feed never raise into the loop.

`tests/tools/test_pick_final_corr.py`: the correction decision and caps, one corrective move at grasp height (and never two), session off and `--no-jaw-check`, the grip check truth table, the retry and abort paths with it.

`tests/tools/test_pick_jaw_reset.py`: jaw check (found, one retry, twice -> `not_in_jaws` with exact `run_plan` call counts, never raises), two-step and fast close, reset sampling over 500 seeded draws, the whole-mat finder (scene first, markers masked, empty mat) and `lost` after three misses.

```
.venv/bin/python -m pytest tests/tools/test_pick_closed_loop.py -q
```
