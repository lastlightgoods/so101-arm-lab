# Scripted demos for training

The autonomous loop (`tools/autoloop.py` with `tools/pick_loop.py`'s closed-loop pick) can record its own successful picks as LeRobot
training episodes. This page is phase 1: everything is built and tested against fakes (the fake arm, `FakeWorld`, a fake camera
source). Nothing here has run on the arm, the Jetson or a real camera, and nothing was pushed to Hugging Face.

## What was built

| Piece | File | What it does |
|---|---|---|
| Per-tick sink | `jetson/pick_seq.py` | `--tick-sink [PATH]`, OFF by default. While a plan runs it appends one JSON line per command tick: the action `send_action` returned and the servo state read just before it, with `time.monotonic()` and wall time. |
| Easing helpers | `jetson/pick_seq.py` | `move_secs`, `ease_n`, `ease_cmd` factored out of `move()`, so the fake arm sends the same command sequence as the real one. |
| Labels | `jetson/lab_config.py` | `LabConfig`: config id, gripper type, arm version, robot id, servo set, camera mount, lighting tag, jaw max, optional pinned calibration hashes. `env_vector()` is the `observation.environment_state` feature; `calib_hashes()` hashes the calibration files. |
| Episode writer | `jetson/demo_recorder.py` | `DemoRecorder`: samples at 30 Hz from a camera source and the tick stream, writes LeRobot v3.0 datasets with LeRobot's own API (`create` / `resume`, `add_frame`, `save_episode`, `finalize`), keeps a main dataset and a separate rejects dataset, writes the sidecar. |
| Keep rule and glue | `tools/demo_session.py` | `keep_decision()` (pure), `DemoSession` (open / end / commit around each try, MLflow links, dataset-version run), `FakeRig` (the simulated arm and tick sink). |
| Runner integration | `tools/autoloop.py` | `--record-demos` (OFF by default), `--demo-root`, `--demo-repo`, `--lab-config`, `--lighting-tag`. |

## The dry run

```
.venv/bin/python tools/autoloop.py --dry-run --record-demos --cycles 5 --seed 4 --session demo-dry \
    --out-dir /tmp/demo-dry --settle 0 --dry-miss 0.2
```

It drives the whole loop on the fake arm and fake dashboard, synthesises each plan's 40 Hz ticks with `pick_seq`'s easing, samples
synthetic cameras at 30 Hz and writes real datasets under `<out-dir>/demos/<session>/`:

```
so101-nut-pick/            kept episodes  (repo id lastlight/so101-nut-pick, change with --demo-repo)
so101-nut-pick-rejects/    everything the keep rule refused
  meta/lab/episodes.jsonl  one row per episode (the sidecar)
  meta/lab/frames/episode_NNNNNN.json   per-frame sample time, tick age, camera ages, stage name
  meta/lab/links.jsonl     MLflow cycle run id per episode (appended once the cycle run exists)
```

Load one back with `LeRobotDataset("lastlight/so101-nut-pick", root="<path>")`. `--record-demos` without `--dry-run` exits with a message
until phase 2 exists, so the flag cannot start a half-working recording on the real arm.

Measured on the Mac with the tiny synthetic cameras (64x36 and 64x48): a 5-cycle session takes 25 s with `--record-demos` and 2 s without
(the difference is video encoding of three cameras per episode). The first full run of the new tests shows the same cost: `tests/tools/test_autoloop_demos.py` (15 tests) takes about 85 s.

## The format

Same features `lerobot-record` writes for a `so_follower`, built with `hw_to_dataset_features` and `combine_feature_dicts`:

- `action` and `observation.state`: float32 [6], names `<motor>.pos`
- `observation.images.top`, `.front`, `.wrist`: video
- `observation.environment_state`: float32 [5] = gripper one-hot (stock, robonine_parallel), arm-version one-hot (stock, special_edition_1), `jaw_max_mm / 100`; constant within an episode
- `task`: one fixed string, "Pick up the nut and place it on the target"
- fps 30, `robot_type` `so_follower`, codebase version v3.0

LeRobot sets `timestamp = frame_index / fps`, so a slow session would still claim 30 Hz. The real sample times, tick ages and camera ages
go to `meta/lab/frames/`, and an episode is reported unhealthy (and so rejected) when more than 5 percent of its frames were sampled
more than one frame period late, a camera frame was older than 0.25 s at any sample, there are fewer than 30 frames, or top/front banding
is above 8 grey levels (the 8 is my choice; measured with the lock: 3.3 and 2.2).

Idle gaps (the Mac thinking between two staged `pick_seq` runs, or the settle after the last waypoint) have no tick younger than 60 ms;
those 1/30 s boundaries are dropped, not recorded.

## Keep / discard

`tools/demo_session.keep_decision()` keeps a try only if all of these hold, and returns every reason it did not:

1. picked, and the nut ended within the radius (`--radius-mm`, default 15) of the target; the radius is rechecked, the judge's `landed` flag alone is not trusted
2. no guardrail trip in the cycle (a trip from `check_after_cycle` counts, so a cycle that ends the session with `consecutive_failures` is rejected)
3. the final jaw check held (the pick record and the scene object both say so; unknown counts as not held)
4. recording health ok
5. no cycle error (including "nut on the target and the reset did not move it away")

Tier `A` = no retry, tier `R` = success after the one retry (the episode holds both attempts). Rejected tries go to the rejects dataset with
the reasons in the sidecar. A trip or stop in the middle of the try saves nothing anywhere (`aborted`). The pre-reset and the reset pick
are never recorded: the episode opens just before the first pick of the try and ends when the last attempt returns.

## Sidecar and MLflow

Each row of `meta/lab/episodes.jsonl` carries: `configuration` (the `LabConfig` labels plus the calibration file hashes), `scene` (start and
target in mat inches; `start_region` is `null` until regions are defined), `teacher` (session, cycle, attempts, jaw checks, stage
corrections, max error in degrees, git sha, trip), `quality` (tier and the health numbers), `verdict`, `links` (the MLflow cycle run name
`autoloop-<session>-c<N>` and the session run id). The row is written after `save_episode`; if the process dies between the two,
`reconcile_sidecar()` adds a placeholder row for the episode on the next start.

MLflow (all through `jetson/labtrack.py`, so a missing server changes nothing): each cycle run gets params `dataset_repo`, `kept`,
`discard_reason`, `config_id`, `tier`, `dataset`, metrics `episode_frames`, `episode_index`, `cadence_over_budget_pct` and an
`episode_meta.json` artifact; the run id goes back into `links.jsonl`. At the end of the session a run of kind `dataset-version` goes to
the experiment `so101-datasets` (`so101-datasets-dry` for a dry run, like `so101-autoloop-dry`) with the repo id, a dataset hash,
a features hash, episode and frame counts, the episode count per config id, kept / rejected / aborted counts and the sidecar as an artifact.

## Where this differs from the plan

- **State read before the command.** The plan read the state after each `send_action`. The sink reads it before, like `lerobot-record` (observation first, then the action of the same tick). Same cost: one 6-motor read per tick.
- **Tick age limit 60 ms, not 25 ms.** Ticks are 25 ms apart by construction (`sleep(secs/n)`) plus the readback, so a 25 ms limit would drop most frames. 60 ms is two ticks.
- **Rejects dataset instead of `clear_episode_buffer`.** The plan discarded rejected tries. Because the verdict arrives after the try, frames wait in a spool on disk (raw `.npy` per frame and camera) and are added to whichever dataset the verdict picks. `clear_episode_buffer` would only have deleted them.
- **Commit waits for the end of the cycle.** The plan committed after the judge; the commit is after `check_after_cycle`, because rule 2 includes trips that only that check can raise.
- **Recorder in the same process.** The plan had `tools/demo_session.py` as a socket client. In phase 1 the recorder runs inside the session object; the interface (`open_try`, `end_try`, `commit`, `abort`, `alive`) is the one a client would have.
- **No lab_config.json registry file** yet: `LabConfig()` defaults to the stock-v1 labels, `--lab-config` reads a JSON of the same fields. `jaw_max_mm` is 0.0 (not measured) and the camera mount label is a placeholder.
- **The `recorder` guardrail** is added in the runner (`guard.trip("recorder", ...)` before a cycle when `alive()` is false). `tools/guardrails.py` itself is unchanged.
- **Not built (later steps of the plan):** `tools/demo_push.py`, `tools/demo_split.py`, the `env.sh` / `so101-stop` change, the dashboard fallback.

## Tests

`tests/demos/` (pick_seq sink and golden tests, recorder, keep rule), `tests/tools/test_autoloop_demos.py` (runner integration, command
line, flag off), `tests/mlops/test_demo_tracking.py` (real SQLite store and an unreachable server). The golden tests pin sha256
hashes of the exact command floats computed with the `pick_seq.py` and `autoloop.py` of main before the change: with the flag off the
124 commands of the test plan and the 63 output lines of a 6-cycle dry run are identical.

## Phase 2: what is needed on hardware

1. **Camera source.** A real `CameraSource` (`jetson/demo_recorder.py` has the interface and a `TODO`): lerobot `OpenCVCamera` async readers for top, front and wrist opened once per session so the locked exposure holds, plus the 15 fps re-serve to the dashboard so `perception` never goes dark, with the fallback to today's ffmpeg when the recorder is down. This is the biggest risk in the plan (a new process in the dashboard's perception path). Run `so101-cams-health` with the tee active first.
2. **Resident process.** Calls `pump()` at 30 Hz, polls the tick file (`TickBuffer.poll_file`), exposes open / end / commit / abort / alive on a control socket; `DemoSession` then talks to it and `fake` is `None`. `so101-stop` should stop it and abort its open episode. Refuse to start when a `lerobot-*` arm job or `.cams_busy` exists.
3. **Pass `--tick-sink` to `pick_seq`** in `pick_loop.run_plan`'s ssh command (and set `SO101_TICK_SINK` if the default `/dev/shm/so101_ticks.jsonl` is not wanted), only when recording. Check the sink's extra servo read per tick on the real bus: the extra reads lengthen each tick, which was not measured (the fake robot costs nothing).
4. **The spool is a risk.** Raw frames for top, front and wrist are 2,304,000 bytes per frame, so a 1,370-frame episode (the length of the newest 3-camera set) would spool about 3.2 GB at about 69 MB/s. Decide before the pilot: JPEG or PNG spool, or a bounded in-memory ring, and check the disk free before each session.
5. **Measure what the fakes cannot:** sink line size is 485 bytes (about 19 KB/s at 40 Hz, 9.7 MB per full 20,000-line file; measured), but the tick spacing, the sampling lateness and the Orin's CPU with three cameras, the encode and the JPEG re-serve are unmeasured. The length of a recorded try (the plan estimates about 40 s) is also unmeasured.
6. **Open parts of the plan:** start regions (5 x 10 episodes), a visible target cue before any target variety, the wrist banding (about 33 grey levels, the recorder only labels it), servo temperature, disk and minute caps, `demo_push.py` (Mac side, private repo first, never the rejects).
7. **First pilot, attended:** TC at the wall STOP with the stop-file command ready, `--cycles 5`, a scratch dataset name. Order from the plan: `so101-cams-health` with the tee, then one dry cycle with the recorder writing to the scratch dataset, then the 5 cycles. The "10 in a row" milestone is not required for an attended pilot (the keep filter protects the data) and is for anything unattended. Grasp retention should be re-measured first; the plan's evidence on the stock gripper is mixed.
