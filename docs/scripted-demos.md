# Scripted demos for training

The autonomous loop (`tools/autoloop.py` with `tools/pick_loop.py`'s closed-loop pick) can record its own successful picks as LeRobot
training episodes. Phase 1 (this page, up to the last section) is built and tested against fakes (the fake arm, `FakeWorld`, a fake camera
source); phase 2 (last section) adds the real camera reader and the resident recorder on the Jetson, also tested only against fakes. Nothing here has run on the arm, the Jetson or a real camera, and nothing was pushed to Hugging Face.

## What was built

| Piece | File | What it does |
|---|---|---|
| Per-tick sink | `jetson/pick_seq.py` | `--tick-sink [PATH]`, OFF by default. While a plan runs it appends one JSON line per command tick: the action `send_action` returned and the servo state read just before it, with `time.monotonic()` and wall time. |
| Easing helpers | `jetson/pick_seq.py` | `move_secs`, `ease_n`, `ease_cmd` factored out of `move()`, so the fake arm sends the same command sequence as the real one. |
| Labels | `jetson/lab_config.py` | `LabConfig`: config id, gripper type, arm version, robot id, servo set, camera mount, lighting tag, jaw max, optional pinned calibration hashes. `env_vector()` is the `observation.environment_state` feature; `calib_hashes()` hashes the calibration files. |
| Episode writer | `jetson/demo_recorder.py` | `DemoRecorder`: samples at 30 Hz from a camera source and the tick stream, writes LeRobot v3.0 datasets with LeRobot's own API (`create` / `resume`, `add_frame`, `save_episode`, `finalize`), keeps a main dataset and a separate rejects dataset, writes the sidecar. |
| Keep rule and glue | `tools/demo_session.py` | `keep_decision()` (pure), `DemoSession` (open / end / commit around each try, MLflow links, dataset-version run), `FakeRig` (the simulated arm and tick sink). |
| Runner integration | `tools/autoloop.py` | `--record-demos` (OFF by default), `--demo-root`, `--demo-repo`, `--demo-vcodec` (dry run default h264, because LeRobot's default SVT-AV1 aborts at teardown on macOS; `DemoRecorder` itself defaults to `libsvtav1`), `--lab-config`, `--lighting-tag`. |

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

Load one back with `LeRobotDataset("lastlight/so101-nut-pick", root="<path>")`. `--record-demos` without `--dry-run` uses the phase 2 recorder on the Jetson (last section) and refuses to start when it is not there.

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

## Phase 2: the resident recorder on the Jetson

Built and tested against fakes only (a fake ffmpeg for camd, a live-written tick file, an in-process daemon behind an injected transport). Nothing here has run on the Jetson, a real camera or the arm.

### Who owns the cameras today (read from the code, 11 Nov 2026)

- There is no camera process on the Jetson. `dashboard/server.py` `jetson_cam_loop` (one thread per camera, started from `JETSON_CAMS` in `main`) runs, over ssh, **one `ffmpeg` per camera that opens `/dev/v4l/by-id/<camera>`** and pipes 15 fps MJPEG (`-q:v 6`; 640x360 for the two C920s, 640x480 YUYV for the wrist) back on stdout. The Mac thread splits the JPEGs and stores the newest in `cam_frames[name] = (jpeg, Mac receive time)`, which `/cam/<name>.jpg`, `/api/cams` ("live feed: top camera (jetson)") and the perception threads (`scene_loop`, the front pose loop, `fetch_top`) read.
- The only existing hand-off is `~/so101/.cams_busy`: the remote shell loop in that command waits while the file exists and kills ffmpeg when it appears (`so101-cams-take` in `jetson/env.sh`, `so101-record`, `cam_lock.py`, `cam_health.py`, `cam_calib`).
- A UVC device has one streaming owner. A second process that opens the same camera while the dashboard's ffmpeg runs gets EBUSY, and a recorder that took the device would blind the dashboard and `perception`. The Mac's copy of the stream is no help either: 15 fps, q6, stamped with the Mac's receive time, on the wrong machine for the tick clock.

### Design: one camera owner, readers

```
V4L2 devices <-- jetson/camd.py (one ffmpeg per camera, 30 fps, newest JPEG + time.monotonic()) --unix socket /dev/shm/so101_camd.sock-->
      |-- camd.py tap NAME   (the dashboard's remote command, 15 fps, the same MJPEG stream as before) -> ssh -> dashboard cam_frames (unchanged)
      '-- jetson/demo_cams.py CamdCameraSource (30 fps ring of JPEG + capture time) -> jetson/demo_recorderd.py
pick_seq --tick-sink /dev/shm/so101_ticks.jsonl ----------------------------------------> demo_recorderd.py (TickBuffer.poll_file)
Mac: autoloop --record-demos -> tools/demo_session.RemoteRecorder -> ssh "demo_recorderd.py ctl OP" -> unix socket /dev/shm/so101_demorec.sock
```

- **`jetson/camd.py`** (stdlib only) owns the three devices with the dashboard's own ffmpeg command line (plus `-fflags nobuffer -flags low_delay`, 30 fps, `-q:v 4`), yields them within 0.5 s while `.cams_busy` exists, restarts ffmpeg when it dies. Readers get the newest frame with its capture time; a slow reader loses frames and cannot slow the camera or another reader; a reader whose camera is silent for 5 s is disconnected.
- **`camd.py tap NAME`** is the dashboard's reader. It exits 3 when no camd answers, which is how the dashboard's command falls back to its own ffmpeg (below). A dead camd costs one reconnect, not the live feed.
- **`jetson/demo_cams.py`** `CamdCameraSource` is the recorder's `CameraSource`: a subscriber only, it never opens a device. It spools the JPEGs as they came (tens of KB a frame instead of 0.9 MB of raw pixels, which answers phase 1's spool risk) and decodes at commit. A camera that never delivered returns a black frame stamped 1000 s old, so the episode fails its `camera_frozen` check and goes to the rejects.
- **`jetson/demo_recorderd.py`** is the resident process: one thread wakes every 1/30 s, polls the tick file, pumps the open episode and answers the control socket in between (`ping`, `status`, `configure`, `open`, `end`, `commit`, `abort`, `link`, `summary`, `finish`). It imports no robot code and never commands the arm (a test greps for it). An exception while sampling aborts the open episode; a handler exception becomes an error reply; the process dying makes `alive()` false on the Mac and the existing `recorder` guardrail stops the session before the next cycle (never in the middle of a pick). The next start deletes the half-written spool. `open` is refused only while `.cams_busy` exists (not `.arms_busy`: `pick_seq --serve` holds that for its whole session); a missing camera does not refuse it (the episode is rejected by its health check; the arm never waits).
- **Mac side**: `tools/demo_session.RemoteRecorder` has the interface `DemoSession` already used, over an injected `transport(op, payload, timeout)`; `ssh_transport()` is one ssh call per request (JSON on stdin, reply on stdout; about 4 per cycle plus a `ping` before each cycle). If the recorder refuses `open`, that try is not recorded (`recorder_open_failed:...`) and the cycle goes on; `end` / `commit` / `abort` never raise; `finish` tolerates a dead recorder. The keep rule, the rejects dataset and the MLflow links are unchanged. `DemoSession.finish` now reads a `summary()` (hashes, sidecar rows, text of `info.json` and `episodes.jsonl`) so the datasets can live on the Jetson.
- **`pick_seq --tick-sink`** is switched on by `tools/pick_loop.PICK_SEQ_ARGS` (empty by default), which `autoloop --record-demos` sets before the pick session is opened and nothing else does. It is appended both to `PickSession`'s `pick_seq.py --serve ...` command (one sink file, with its header line, per served plan, as `--plan` runs write) and to the `run_plan_ssh` fallback. That is the one hook in `pick_loop.py`; `pick_seq.py` needed no change.

Datasets are written on the Jetson, by default under `~/so101/outputs/demos/<session>/so101-nut-pick` (kept) and `.../so101-nut-pick-rejects`, with the same sidecar as phase 1. Copying them to the Mac is manual for now (`tools/demo_push.py` is still not built).

### Command line

```
tools/autoloop.py --record-demos --cycles 5 --session pilot1 --lighting-tag lamp-a --demo-repo lastlight/so101-pilot     # real arm, scratch dataset name
   [--demo-host jetson] [--demo-tick-sink /dev/shm/so101_ticks.jsonl] [--demo-root ~/so101/outputs/demos/<session>] [--demo-vcodec h264] [--lab-config FILE]
```

Before any arm motion it requires: the recorder answers, all three cameras delivered a frame in the last 0.25 s, no `.cams_busy`, and `configure` succeeded; otherwise it exits with the reason and the start command. `--dry-run --record-demos` is unchanged (fakes, local, h264).

### Exact deploy steps

Files to the Jetson, flat in `~/so101/` like the other scripts: `jetson/camd.py`, `jetson/demo_cams.py`, `jetson/demo_recorder.py` (changed), `jetson/demo_recorderd.py`, `jetson/lab_config.py`, `jetson/pick_seq.py` (already has `--tick-sink`). On the Mac, this branch's `tools/` (`autoloop.py`, `demo_session.py`, `pick_loop.py`).

```
scp jetson/camd.py jetson/demo_cams.py jetson/demo_recorder.py jetson/demo_recorderd.py jetson/lab_config.py jetson/pick_seq.py jetson:~/so101/
```

1. **Dashboard patch (already in this branch's `dashboard/server.py`; the running dashboard needs a restart by you to pick it up).** `jetson_cam_loop`'s remote command now runs, after the `.cams_busy` wait and before its ffmpeg: `[ -f ~/so101/camd.py ] && { python3 ~/so101/camd.py tap NAME; rc=$?; [ $rc -eq 3 ] || exit $rc; };` (the text is `camd.DASH_PREFIX`; `tests/jetson/test_camd.py` runs it in sh for: no camd.py deployed, camd not running, camd streaming then ending, and checks server.py carries it). With no camd the dashboard behaves exactly as today, so this can go in first.
2. **Hand the cameras to camd (Jetson, with the patched dashboard running).** `touch ~/so101/.cams_busy` (the dashboard's ffmpeg exits within 0.5 s), `nohup python3 ~/so101/camd.py serve > ~/so101/outputs/run/camd.log 2>&1 &` (it idles while the flag exists), `rm ~/so101/.cams_busy`. camd opens the devices and the dashboard's taps connect. **Never start camd next to an unpatched dashboard that is streaming**: its ffmpeg holds the devices and camd cannot open them. Check: `python3 ~/so101/camd.py status` shows three cameras with `age_s` under 0.1 and `restarts` 0; the dashboard shows three live feeds at the quality it had; `so101-cams-check` still passes (the exposure lock is a device setting, independent of who has the device open).
3. **Start the recorder (Jetson).** `nohup nice -n 5 ~/so101/.venv/bin/python ~/so101/demo_recorderd.py serve > ~/so101/outputs/run/demo_recorderd.log 2>&1 &`. Check: `~/so101/.venv/bin/python ~/so101/demo_recorderd.py ctl status` replies `"ok": true`, `cams_missing: []`, `busy: []`.
4. **Control path and cameras without arm motion.** On the Jetson: `D="$HOME/so101/.venv/bin/python $HOME/so101/demo_recorderd.py ctl"`; `$D configure '{"repo":"lastlight/so101-scratch","root":"~/so101/outputs/demos/scratch/so101-scratch","vcodec":"h264"}'`; `$D open '{}'`; wait 2 s; `$D end` (no ticks, so `frames` is 0 and `reasons` lists `too_few_frames`, but `cam_age_max_s` must be empty of freezes); `$D abort`.
5. **One recorded pick, attended (TC at the wall STOP, stop-file command ready).** `so101-cams-check`, then from the Mac `.venv/bin/python tools/autoloop.py --record-demos --cycles 1 --session pilot1 --demo-repo lastlight/so101-pilot --demo-vcodec h264 --start-near X Y`. Expect the preflight to pass and the cycle line to carry `"demo": {"kept": true, ...}` if it landed (or a `discard_reason`); `$D status` then shows `counts`. On the Jetson look at `~/so101/outputs/demos/pilot1/so101-pilot/meta/lab/episodes.jsonl` (`quality.over_budget_pct`, `cam_age_max_s`) and load it with `LeRobotDataset("lastlight/so101-pilot", root=...)`; check the video by eye, and compare the pick's duration with and without `--record-demos` (the extra servo read per tick has never been measured on the real bus). Only then 5 cycles.
6. **Roll back.** `pkill -f demo_recorderd.py` (SIGTERM aborts the open episode); without `--record-demos` autoloop and `pick_seq` behave as before (no sink file is written). To give the cameras back to the dashboard's own ffmpeg: `pkill -f "camd.py serve"; rm -f /dev/shm/so101_camd.sock` and the patched dashboard falls back on its next reconnect (a few seconds), or revert the one dashboard line and restart it. Datasets already written stay on the Jetson.

### Codec and CPU

`libsvtav1` is the Jetson default (Linux). It encodes at commit, after the reset, while the arm is idle, so it cannot slow a pick; it only lengthens the cycle. To decide on the real Orin: (1) time the first `commit` (`time $D commit ...`, or the gap between the cycle's judge line and the next cycle); (2) run `tegrastats --interval 1000` during a recording and during the encode: the question is whether `quality.over_budget_pct` in the sidecar stays under 5 and the sink's tick spacing stays near 25 ms; (3) re-run the same pilot with `--demo-vcodec h264` (or `serve --vcodec h264`) and compare commit wall time and CPU. Switch to h264 if the encode makes the wait between cycles too long or the cadence suffers.

### Still unmeasured or open

- camd's capture-to-pipe latency (`camd.py serve --latency-ms` subtracts it from the frame times; measure it by flashing a light in view of one camera and finding the first frame that shows it), three cameras at 30 fps on one USB bus (the dashboard asked for 15), the real tick spacing and the extra servo read per tick, the length of a real try, the JPEG spool and dataset sizes, and the Orin CPU with everything running.
- The wrist camera banding (about 33 grey levels) is still only labelled: the real source reports no banding value.
- The frame in the dataset is the JPEG camd received at `-q:v 4` (the C920s are MJPEG natively, so this is one decode and one re-encode of the camera's own stream, then AV1 or h264).
- Start regions, a target cue, servo temperature and disk and minute caps from the plan are still open; `so101-stop` is unchanged (it kills `pick_seq`, autoloop then aborts the episode on the recorder).
