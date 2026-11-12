# The session runner: an arm that practises on its own

`tools/autoloop.py` (Mac shell: `so101-autoloop`) runs the closed-loop pick of [`pick-loop.md`](pick-loop.md) over and over without
anyone resetting the mat, and `tools/guardrails.py` decides, before every step, whether it is still safe to go on. It never talks to the
arm itself: every motion goes through `pick_loop.run_plan` (one `pick_seq.py` run over ssh), so the arm's own limits still apply.

```
so101-autoloop --cycles 10 [--minutes 60] [--target X Y | --random-target] [--safe-box X0 Y0 X1 Y1] [--radius-mm 15] [--session NAME] [--dry-run]
```

## The loop

One cycle, repeated until a cap or a trip ends the session:

1. **guard.before**: stop file, then `GET /api/state` (the panel must be alive), then the perception watchdogs.
2. **TRY**: locate the nut (`pick_loop.find_nut_anywhere` when it exists; else the overhead search near the last spot and the target; else
   the `/api/scene` object nearest the last spot), then `closed_loop_pick` to the target: hover, read back, correct; grasp pose, read back,
   correct; pick, carry, place, rest. If the nut was **not picked**, one retry (`--max-retries`, default 1, never more).
   **A nut already within 20 mm of the target is reset away first** (a pre-reset, judged like any reset): placing it on the target again
   could not move it 20 mm, so first10b scored good picks as misses in cycles 5 and 6. If the pre-reset leaves it there, the cycle fails
   with `error: nut on the target ...` and no try.
3. **JUDGE**, after `--settle` seconds (1.0) from `/api/scene`, then polling every 0.5 s while the map reports objects `pending`
   (it shows an object only after 3 detects in a row, about 1.5 s), up to `--judge-wait` (4 s): the nut is the scene object nearest the target.
   `picked` = it moved more than 20 mm from where the TRY found it, and, if the pick record or the scene object has `in_jaws`, that is true.
   `landed_err_mm` = its distance to the target; `landed` = picked and err <= `--radius-mm` (15). A cycle is a **success** when landed.
   A nut that was picked but dropped elsewhere is a failure and is not retried (it is no longer where the retry would look).
4. **RESET** (then judged): find the nut (`pick_loop.find_nut_anywhere`: the scene map first, then the whole-mat image search) and run the same closed-loop pick to a spot sampled inside the safe box: at least 40 mm from other objects, 50 mm from marker centres and 60 mm from where the nut was. The runner deliberately does not call `pick_loop.reset_nut`: that function saves its history over the pick loop's own state file and samples its own spot. The pick loop's learned sag biases are read (never written) from `jetson/calib/pick_state.json`.
   The reset is then **judged** the same way as the landing: the nut-like object nearest the spot within `--radius-mm` is `at_spot` (ok);
   else one still within 20 mm of where the nut was is `not_moved`; else the one nearest the spot is `elsewhere`; none is `not_seen`.
   The next cycle looks for the nut where the map saw it, not where the reset meant to put it (first10b cycle 4: the reset's jaw check saw
   nothing held, the nut stayed on the target, and the runner believed the planned spot).
5. **guard.after**: the consecutive counters (below).

`--random-target` samples the target each cycle with the same rules instead of using `--target` (default 9 6). The safe box defaults to
`6 4 13 8` (mat inches, the approved box). Both the target and the reset go through the workspace box guard as well.

Each cycle prints one JSON line (`event: cycle`; `cycle, found, picked, landed, ok, landed_err_mm, moved_mm, retries, target_in, nut_in,
reset_to_in, reset_ok, reset_err_mm, reset_how, reset_found_in, judge_wait_s, pre_reset, locates, cycle_s, trip`). `locates` lists every
nut search of the cycle: where it looked, how many objects and nut-like candidates the map had, up to 8 candidates, and the one chosen.
Every pick stage (hover, grasp_pose, final, jaw_check, with corrections, residuals, sag and the aim) is printed too, as `event: stage`
lines, and kept in MLflow. On the real arm the overhead frame at each judgement is saved to
`outputs/run/autoloop_<session>_frames/c<N>_try<k>.jpg`, `c<N>_reset.jpg` and `c<N>_prereset.jpg`, and attached to the cycle's MLflow run.

## Guardrails (tools/guardrails.py)

A check returns a `Trip(reason, detail)` or nothing. Any trip ends the session after one JSON line (`event: trip`, `reason`, `detail`,
`cycle`, `rest`). Thresholds are the defaults; the flag in brackets changes one.

| Reason | Fires when | Where checked |
| --- | --- | --- |
| `workspace` | any waypoint tip of a plan (FK of the planned joints through `cam_calib.fk_tip`, the arm fit's joint offsets and tip offset, then `arm_to_table`) is outside the mat shrunk by 20 mm (`--margin-mm`), above 4 in over the mat (`--ceiling-in`), or more than 10 mm below the mat plane. Edges are inside. `rest` is not checked (it is folded behind the mat on purpose). | before **every** `run_plan`: a violating plan is never run |
| `motion` | a `pick_seq` row has `max_err_deg` > 8 (`--max-err-deg`); equal is fine | right after each `run_plan` |
| `markers` | `n_markers_seen` < 3 continuously for 10 s (polled every second) | before each cycle |
| `holding` | the scene's `holding` flag (calibration held with too few markers) true for more than 10 s | before each cycle |
| `calib` | `/api/scene` `calib_ok` is false (null = unknown, no trip) | before each cycle |
| `perception` | `/api/scene` is not ok (top camera not live, no detection) or does not answer | before each cycle |
| `nut_lost` | the nut was not found for 3 cycles in a row (`--lost-after`) | after each cycle |
| `consecutive_failures` | 3 cycles in a row did not land the nut (`--max-consecutive-failures`) | after each cycle |
| `panel` | `GET /api/state` does not answer within 3 s, is an error, or reports the Jetson offline | before each cycle |
| `run_plan` | the ssh `pick_seq` run failed or returned no rows (what `so101-stop` causes) | inside a pick; no rest is sent |
| `overload`, `fps` | see "Not exposed yet" | before each cycle |
| `stop_file` | the stop file exists (a **clean** stop: `event: stop`, not a fault) | before each cycle and before every `run_plan` |

Caps are not trips: `--cycles N` and `--minutes T` end the session normally (`end: cycles` / `minutes`); the minutes cap is checked
before a new cycle starts, so a cycle in progress finishes. `nut_lost` is checked before `consecutive_failures`.

**Rest.** After a trip the arm goes to the rest pose only if it may be mid-motion **and** a fresh `/api/state` poll answers with the Jetson
up and a live follower stream. A dead panel, a `run_plan` failure (an owner's STOP) and a stop between cycles (the plan already ended at
rest) send no motion command.

## Not exposed yet

The dashboard publishes `arms.follower.pos`, `procs`, `jetson.online` and the scene fields, but no motor overload/error flag and no camera
frame rate. So:

* **overload**: `Guard` looks for an `overload` key in `arms.follower` or at the top of `/api/state` and trips on a truthy value; today there
  is none, and the session logs `overload not checked` once. The `motion` watchdog (commanded vs reached) is the only stand-in.
* **fps < 25**: `Guard` reads `fps` (a number, or a dict of camera -> fps) from `/api/state` and trips below 25; today there is none, and the
  session logs `fps not checked` once. `calib_ok` / `n_markers_seen` going bad is the practical symptom of a stalled overhead camera.
* "holding" in the dashboard is derived from the marker count, so `holding` and `markers` see the same condition; `markers` (10 s, `>=`)
  fires first.

## Stopping it

* **`so101-stop`** (on the Jetson) kills `pick_seq`; the runner sees the dead run, trips `run_plan` and sends nothing more. From the Mac:
  `ssh jetson 'source ~/so101/env.sh; so101-stop'`.
* **The stop file**: `touch ~/so101/outputs/run/autoloop.stop` (or `--stop-file PATH`). Checked between cycles and before every `run_plan`,
  so the session ends within one waypoint stage; if the arm is mid-pick it is sent to rest first. Delete the file before the next session.
* **The wall STOP** (the panel's emergency button) cuts the arm; the runner finds out at its next `/api/state` poll or when `pick_seq`
  fails, and ends with `panel` or `run_plan`.
* Ctrl-C ends the session (`end: interrupted`) and still writes the summary. Since the overnight guards (below) it, SIGTERM (`kill <pid>`) and a crash also send the arm to rest first, under the same rule as a trip.
* The dashboard STOP (wall, iPad, phone page) now also writes the stop file (`dashboard/server.py write_stop_file`): `so101-stop` only kills what is running that second, and a STOP pressed between two plans used to be missed (the runner restarted `pick_seq`). The file stays until you delete it; the next run then stops at once with `stop_file`.

## Overnight guards (tools/guardrails.py, shared with tools/tidy.py)

`--unattended` (TC's decisions, 11 Nov): caps default to `--max-tries 60` and `--max-minutes 240` (overridable); turns on `--require-health`; refuses to start while `outputs/run/autoloop.stop` exists (it prints the `rm` command, never deletes the file); keeps the Mac awake with `caffeinate -dims -w <own pid>` and refuses to start if that fails (you may also wrap the command in `caffeinate -dims`); and at the end, for any exit reason, after the park, releases the servo torque (`jetson/free_arm.py` over ssh, the lab's own so101-free path without its pkill) ONLY if a fresh `/api/state` readback (src idle-poll, under 3 s old) shows every arm joint within 5 deg of rest. 5 deg: above the 1 to 3.5 deg sag measured 9 Nov and the 4.7 deg worst loaded sag, under the 8 deg motion trip. Otherwise the torque stays ON and a `torque_left_on` line says so. Trips and their sources:

| trip | when | source of the number |
|---|---|---|
| `servo_hot` | any servo >= `--temp-limit` (65 C); also no start at 60 C or more (`servo_hot_start`) | servo cut 70 C and idle 52 to 57 C, both read from the real arm on 11 Nov 22:30 |
| `servo_current` | one arm joint >= 2.7 A, or all servos > 5.0 A | STS3215 stall 2.7 A at 12 V (docs/stronger-arm.md); 5 A supply and board rating. The gripper is exempt |
| `servo_voltage` | supply < 10.0 V | lab choice (servo_telemetry.VOLT_LOW_V) |
| `servo_stall` | position error > 6 deg with load >= 50 % for >= 3 s | 6 deg: above the 4.7 deg worst loaded sag, below the 8 deg motion trip; 50 %: assumption, never measured on the arm |
| `servo_error` | any servo error bit (voltage, sensor, temp, current, overload) | the servo's Status register |
| `no_servo_telemetry` | `--require-health` and a plan came back without telemetry | |
| `max_tries`, `max_minutes` | `--max-tries`, `--max-minutes`; clean stops, checked between picks | |

Rows from `pick_seq` carry the servo summary of each waypoint, so heat, current, supply and stall are judged after every plan, not only between cycles; a plan already running is not interrupted. Live `/api/tele` is also read between cycles (only when it has a sample younger than 5 s).

## What a session leaves behind

* stdout: the stage and cycle lines, at most one trip/stop line, and the `summary` line.
* `outputs/run/autoloop_<session>.json` (`--out-dir`): `session, cycles, landed, picked, success_rate, mean_err_mm, median_err_mm, retries,
  stops` (each with `reason`, `detail`, `clean`, `cycle`), `end` (`cycles`, `minutes`, `interrupted` or the trip reason), `duration_s,
  radius_mm, dry_run, resets, resets_ok, reset_median_err_mm, pre_resets`. Errors are over cycles where the nut was found at the end.
* MLflow (`jetson/labtrack.py`, experiment **`so101-autoloop`**): one run per cycle (params `session, cycle, target, mode`
  (`fixed`/`random`), `radius_mm, dry_run`; metrics `landed_err_mm, picked, landed, cycle_s, retries, trip, reset_err_mm, reset_ok, judge_wait_s,
  pre_reset`; artifacts `cycle.json` with the record and the pick stages, and `frames/`) and a parent run `autoloop-<session>` (the cycle runs carry its id in `mlflow.parentRunId`) with the summary
  as metrics and `summary.json`. Dry runs log to `so101-autoloop-dry`. No tracking server: the loop runs unchanged.

## Dry run

`--dry-run` replaces the dashboard with `FakeWorld` (state, scene, a nut that picks move) and the arm with `DryArm` (the real kinematics,
every waypoint reached 6 mm short, `--dry-miss` 0.1 chance of a miss). The whole loop, guardrails, retries and files run; nothing moves
and nothing is contacted. `tests/tools/test_autoloop.py` and `test_guardrails.py` run it in CI against the `ShortArm` of
`tests/tools/pickhelpers.py` and a fake clock.

## First milestone: "10 in a row, attended"

1. The mat is empty except the nut. Arm parked at rest. Lights as for calibration; `so101-cams-check` passes.
2. Take the nut off, press **SET BACKGROUND** on the dashboard with the mat empty, put the nut back at a spot inside the safe box. Check
   `/api/scene`: `calib_ok: true`, `n_markers_seen: 4`, exactly one object (the nut).
3. Choose the safe box (mat inches) where you are happy for the nut to land and be picked from, e.g. `--safe-box 6 4 13 8` (the default).
4. `so101-autoloop --dry-run --cycles 10` first: it must finish with `end: cycles`.
5. The owner stands at the wall STOP, with the stop file command ready. Run
   `so101-autoloop --cycles 10 --target 9 6 --session first10`.
6. **Pass** = at least 9 of the 10 cycles `landed` with `landed_err_mm` <= 15, **and zero trips** (`stops: []`, `end: cycles`). Anything
   else: read the cycle lines and the MLflow runs, fix, and start a new 10.

## Keep-outs (the toy frogs)

Every cycle the runner takes the keep-outs from the scene map and the top frame (`tools/keepout.py`, same rule as tidy). The nut finder never returns a point inside a keep-out and never sees a keep-out object; random targets and reset spots stay at least the keep-out's radius + 40 mm away; a fixed target inside one is refused (`keepout_target`); `guard.check_plan` refuses (`keepout`) any plan whose arm body points (wrist, gripper, tip, with droop) touch a keep-out cylinder on the way between waypoints. `--unattended` stops (`keepout_unknown`) when there is no top frame; attended runs fall back to the size rule and say so once. The keep-out height (60 mm) is an assumption, not measured. The first waypoint's move from wherever the arm currently is (normally rest) is checked only at the waypoint itself.
