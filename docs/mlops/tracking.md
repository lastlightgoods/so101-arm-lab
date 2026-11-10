# MLflow tracking in the arm lab

What gets logged, where it lands, how to read it, and what happens when the server is not there.
Companion to [`docs/mlflow_field_guide.html`](../mlflow_field_guide.html) (the MLflow tour and the
`so101-act` training backfill). Everything here is measured from `python -m pytest tests/mlops -q`
and the smoke run described at the end (10 Nov 2026, mlflow 3.17.2 on the Mac).

## The pieces

| File | Role |
| --- | --- |
| `jetson/labtrack.py` | The one wrapper every logger uses: `LabTrack(experiment)`, `with track.run(name, tags) as r: r.params / r.metrics / r.artifact / r.json / r.file`. Resolves the URI, probes it, never raises, never stalls. Lives in `jetson/` because that folder is what gets copied to the Jetson; Mac tools load it by path. `python labtrack.py [uri]` is a self-test that logs one run. |
| `tools/pick_loop.py` | One run per pick in **`so101-picks`**. |
| `jetson/cam_calib.py` | One run per written calibration file in **`so101-calibration`** (`table`, `campose`, `armfit`, `handeye`). |
| `jetson/record_track.py` | One run per `so101-record` / `so101-record-more` session in **`so101-recordings`**, called from `jetson/env.sh` around `lerobot-record`. |
| `tools/mlflow_ui.sh` | The server: `start [--lan] | stop | status | uri` (alias `so101-mlflow` in the Mac `env.sh`). |
| `tests/mlops/` | 34 tests, all against a temp SQLite store or a closed localhost port; about a minute (most of it the `cam_calib.py` subprocesses). |

## Where things go

The tracking URI is resolved once per process, in this order:

1. `$SO101_MLFLOW_URI` — set but empty means **off** (the test suite does this in `tests/conftest.py`);
2. `$MLFLOW_TRACKING_URI`;
3. the Mac's own store, `sqlite:///~/so101/mlflow/mlflow.db`, when `~/so101/mlflow/` exists;
4. otherwise off.

Both `env.sh` files export `SO101_MLFLOW_URI` from `~/so101/.mlflow_uri` when that one-line file exists, so the
Jetson points at the Mac without an address living in the repo. A client that writes straight into the SQLite file
puts artifacts in `~/so101/mlflow/artifacts/<experiment>/` beside it (not in `./mlruns`, which is MLflow's default
and depends on the cwd that first created the experiment); a client that goes through the server ships artifacts to
the server, which keeps them under the same `artifacts/` root. The UI serves both.

## What each run carries

Measured on the smoke run (counts are the number of keys MLflow holds for the run).

### `so101-picks` (pick_loop.py, host mac): 9 params, 14 metrics, 4 artifacts

| | Keys |
| --- | --- |
| run name | `pick-<session>-r<n>`; tags `host=mac`, `run`, `session`, `nut_in` (where the nut was found, mat inches), `target_in` |
| params | `target_in`, `alt_in`, `place_in` (target plus the learned offset), `gain`, `outcome_gain`, `max_deg_s`, `runs`, `state` (the state file), `dash` |
| metrics | `landing_err_mm`, `before_mm`, `seconds`, `picked` (0/1), `found_at_landing` (0/1), `sag_<waypoint>` for all nine waypoints (`max_err_deg` from pick_seq: `sag_hover_nut`, `sag_grasp`, `sag_close`, `sag_lift`, `sag_hover_tgt`, `sag_place`, `sag_open`, `sag_lift2`, `sag_rest`) |
| artifacts | `frames/landing.jpg` (the dashboard's overhead frame the landing was measured in), `plan.json`, `waypoints.json` (the per-waypoint target/reached rows), `state/pick_state.json` |

If `pick_seq` dies over ssh the run is marked FAILED with the plan attached and the loop exits exactly as before.
A lost nut is a FINISHED run with `picked=0`, `found_at_landing=0` and no `landing_err_mm`.

### `so101-calibration` (cam_calib.py): tag `kind`, the written JSON attached

| kind | params | metrics | artifacts |
| --- | --- | --- | --- |
| `table` | `camera`, `layout`, `markers` ("0,1,2,3"), `units`, `intrinsics` (path or `none`), `source`, `image_size`, `arm_fit` (carried-over date or `none`), `out` | `residual_mm_<id>` per marker, `residual_mm_max`, `reproj_rms_px` and `camera_height_mm` (with intrinsics only), `points_used`, `markers`, `arm_fit_rms_mm` (when carried over) | `calib/top_table.json`, `calib/top_table_check.png` |
| `campose` | `camera`, `layout`, `markers`, `units`, `hfov_source`, `lens_height` (if given), `source`, `image_size`, `out` | `height_mm`, `tilt_deg`, `hfov_deg`, `reproj_rms_px`, `markers` | the file it wrote (the table file when it exists, else `<name>_campose.json`) |
| `armfit` | `camera`, `pairs_file`, `units`, `markers`, `out` | `residual_mm_<i>` per pair, `residual_mm_max`, `pairs`, `arm_yaw_deg`, `arm_base_x_mm`, `arm_base_y_mm` | the table file and the pairs file |
| `handeye` (camera `wrist`) | `views_dir`, `layout`, `table`, `intrinsics`, `units`, `markers`, `views`, `pose_sigma_deg`, `free_centre`, `overhead`, `out` | `reproj_rms_px`, `reproj_rms_px_corrected`, `off_axis_mm`, `tilt_from_gripper_axis_deg`, `hfov_deg`, `views`, `markers`, `points_used`, `dropped_corners`, `per_view_px_<view>`, `leave_one_out_px_<view>` | `calib/wrist_campose.json` |

Only commands that write a file log; `check`, `locate`, `fk`, `ik` and `handeye --check` do not. The numbers are the
ones the command printed and saved; the logger runs after the print, reads the result and changes nothing
(`tests/mlops/test_calib_tracking.py::test_tracking_off_or_dead_server_changes_nothing` diffs the file and stdout
with tracking off, on and failing).

### `so101-recordings` (record_track.py via env.sh): 17 params, 11 metrics, 4 artifacts

| | Keys |
| --- | --- |
| run name | `record-<dataset>-<time>`; tags `kind=record`, `dataset`, `resume` (0/1), `host` |
| params at `start` | `repo_id`, `task`, `episodes_requested`, `resume`, `root`, `cameras` and `fps` (parsed from `$SO101_CAMS`), `episode_time_s`, `reset_time_s`, `episodes_before`, `cam_settings_captured` and one `cam_<role>_<control>` per locked control in `~/so101/calib/cam_settings.json` |
| metrics at `start` | `cams_locked` (0/1 from `cam_lock.py check --json`), and when `~/so101/outputs/run/cam_health.json` is under 10 min old: `cam_health_ok`, `cam_health_fps_<cam>`, `cam_health_age_s` |
| metrics at `finish` | `episodes_total` (the dataset's `meta/info.json`), `episodes_new`, `duration_s`, `rc`, `ok` |
| artifacts | `cameras/cam_settings.json`, `cameras/cam_check.json` (the live check), `cameras/cam_health.json` (if fresh), `dataset/info.json` |

`so101-cams-health` now also saves its JSON report to `~/so101/outputs/run/cam_health.json` (`cam_health.py --save`),
which is how the pre-recording check ends up attached to the recording. The run is FINISHED when `lerobot-record`
exits 0 and FAILED otherwise; the shell functions return what they always returned and `lerobot-record`'s argv is
unchanged (`tests/mlops/test_record_track.py`, driven through a sourced `jetson/env.sh` with the same fakes
`tests/jetson/test_cam_lock.py` uses).

## Setting it up (the two lines)

On the Mac, once (the server binds the LAN; there is no auth, which is fine on the lab's trusted home network behind
the router and the Mac firewall, and not fine anywhere else, where you would bind local and tunnel instead):

```sh
~/so101/tools/mlflow_ui.sh start --lan                                  # or: so101-mlflow start --lan
~/so101/tools/mlflow_ui.sh uri | ssh jetson 'cat > ~/so101/.mlflow_uri' # the Mac's LAN address, one line, on the Jetson
```

On the Jetson, once: `~/so101/.venv/bin/pip install mlflow-skinny` (no server, no UI, no pandas; the client API is
all `labtrack.py` uses), then check the link:

```sh
ssh jetson 'cd ~/so101 && source env.sh && python labtrack.py'
# {"uri": "http://<mac-ip>:5050", "enabled": true, "why": ""}  {"run_id": "...", "experiment": "so101-labtrack-selftest"}
```

Use an IP address in `.mlflow_uri`, not a `.local` name: the 1 s probe cannot bound a DNS lookup. The Mac's own tools
(`pick_loop.py`, the backfill) write straight into the SQLite file by default; put the same URL in the Mac's
`~/so101/.mlflow_uri` if you would rather they go through the server too.

## Looking at it

Open `http://127.0.0.1:5050` on the Mac (or the LAN URL `mlflow_ui.sh start --lan` prints, from the iPad). Each
experiment is a table of runs; useful searches in the box at the top:

```
metrics.landing_err_mm < 10 and metrics.picked = 1          # so101-picks: the good picks
tags.target_in = "9.00,6.00"                                 # one place target
tags.kind = "table" and metrics.residual_mm_max > 2          # so101-calibration: table solves that drifted
tags.kind = "handeye"                                        # every wrist-camera solve, compare per_view_px across them
tags.dataset = "cube" and metrics.ok = 1                     # so101-recordings: sessions that finished
params.cam_top_exposure_time_absolute != "156"               # sessions recorded with a different exposure lock
```

Switch a `so101-picks` table to the chart view with `landing_err_mm` on the y axis and the x axis set to the run's
start time to see whether the loop is learning; tick several `handeye` runs and press Compare for the parameter diff.
Artifacts (the landing frame, the calibration JSON, the camera check) preview in the browser from the run page.

## How it degrades

Every rule is a test in `tests/mlops/test_labtrack.py`:

| Situation | What happens | Measured |
| --- | --- | --- |
| No URI at all (fresh machine, no `.mlflow_uri`, no `~/so101/mlflow`) | silent no-op; `track.enabled` is False, `track.why` says `no tracking URI` | nothing on stderr |
| http URI, server down (connection refused) | one line on stderr, `labtrack: tracking off (<experiment>): ... not reachable within 1 s`, then no-op | 55 ms for the whole `labtrack.py` self-test; the tests assert < 3 s |
| http URI, host asleep (packets dropped) | the 1 s TCP probe gives up, same line, no-op | bounded by the probe (test with a 0.3 s probe: < 3 s) |
| `mlflow` not installed | same line (ImportError), no-op | |
| the server dies mid-run | the failing call prints the one line, every later call is a no-op; the run stays RUNNING server-side | `MLFLOW_HTTP_REQUEST_MAX_RETRIES=1`, `MLFLOW_HTTP_REQUEST_TIMEOUT=5` are set by `labtrack.py` unless you set them, so one dead request costs at most 5 s, not MLflow's default seven retries with backoff |
| the caller's block raises | the run is marked FAILED and the exception propagates (Ctrl-C included) | |
| a param is `None`, a metric is `None`/text/NaN | skipped | |

`pick_loop.py`'s stdout is byte-identical with tracking off, on, and failing
(`test_output_lines_identical_with_tracking_off_and_dead_server`), `cam_calib.py`'s files and stdout likewise, and a
recording still starts and finishes with the server down
(`test_dead_server_does_not_slow_or_fail_the_recording`, < 15 s end to end including two Python start-ups).

## Costs

| Step | Time | Where measured |
| --- | --- | --- |
| `labtrack.py` self-test, SQLite, first use (creates the tables) | 2.2 s | `time python jetson/labtrack.py sqlite:///tmp/t.db` |
| same, existing store | 1.7 s | second run |
| `cam_calib.py table` on the synthetic scene, with logging | 2.2 s wall | smoke run |
| 3 stubbed picks, logging only (no arm, no ssh) | 0.8 s | smoke run |
| `so101-record` end to end through env.sh with the fakes (two `record_track.py` processes + the stub recorder) | 5.5 s | smoke run |
| `so101-record` with the server down | 1 s probe twice, nothing else | test |

The pick loop pays this between the arm finishing and the next pick, never during a move: the run is opened before
`build_plan`, the metrics and files go in after the landing is measured.

## Exact commands

```sh
# Mac: server
so101-mlflow start --lan        # tools/mlflow_ui.sh; http://127.0.0.1:5050 and the LAN URL it prints
so101-mlflow status | stop | uri

# Mac: picks (logs by default into ~/so101/mlflow/mlflow.db, or through the server if ~/so101/.mlflow_uri exists)
~/so101/.venv/bin/python ~/so101/tools/pick_loop.py --runs 4

# Jetson: calibration and recording log automatically once ~/so101/.mlflow_uri exists
so101-calib table --layout calib/top_layout.json
so101-cams-health 10 && so101-record cube "pick the cube" 10

# anywhere: turn it off for one command, or point it somewhere else
SO101_MLFLOW_URI= so101-calib table ...
SO101_MLFLOW_URI=sqlite:///$HOME/scratch/mlflow.db ~/so101/.venv/bin/python ~/so101/tools/pick_loop.py --runs 2

# check what a store holds without the UI
~/so101/.venv/bin/python - <<'PY'
import mlflow; c = mlflow.MlflowClient("sqlite:////Users/you/so101/mlflow/mlflow.db")
for e in c.search_experiments(): print(e.name, len(c.search_runs([e.experiment_id])))
PY

# tests
~/so101/.venv/bin/python -m pytest tests/mlops -q        # 34 passed in 68 s on the Mac
```

## The smoke run behind the numbers above

One temp SQLite store, every logger once, from the worktree root (the script is not in the repo; it is the four test
drivers run by hand): `cam_calib.py table` and `campose` on the synthetic ArUco scene from `tests/calib/synth.py`,
`armfit` on the real 9 Nov pairs (`jetson/calib/top_armfit.json`, copied to a temp calib dir so nothing in the repo
changes), `handeye` on the real `jetson/calib/wristcam` captures, `handeye --check w3` (must not log), three stubbed
picks through `pick_loop.main()`, and `so101-record` + `so101-record-more` through the sourced `jetson/env.sh`.
Result: `so101-calibration` 4 runs, `so101-picks` 3 runs, `so101-recordings` 2 runs, all FINISHED, 0 in `Default`,
and every per-run key listed in the tables above present.
