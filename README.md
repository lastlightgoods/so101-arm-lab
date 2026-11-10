# so101-arm-lab

A small, well-instrumented robot-arm lab built around a prebuilt **SO-ARM101** (leader + follower,
Feetech STS3215 servos) running **LeRobot 0.6.1**. A Jetson Orin Nano drives the arm and the cameras,
a Mac trains policies, records, and shows a wall dashboard, and this repo holds everything that glues
the two together: calibration, camera tooling, a live map of the work mat, scripted picking, LeRobot
plugins, and a 600-test harness that runs without any hardware.

Started 5 October 2026 by [Lastlight Goods](https://lastlightgoods.com) in Bend, Oregon.

## What is in here

| Path | What |
| --- | --- |
| `jetson/` | Jetson side. `jetson_setup.sh` (JetPack 6.2 + CUDA torch + LeRobot), `env.sh` with every `so101-*` command, `cam_calib.py` (camera and arm calibration), `cam_lock.py` / `cam_health.py` / `cam_drift.py` (camera settings lock, frame-rate and drift checks), `goto.py` / `pick_seq.py` (scripted motion), `calib/` (the live calibration files), `JETSON_RUNBOOK.md` |
| `dashboard/` | The wall dashboard: `server.py` (stdlib HTTP, SSH streams, calibration and scene APIs), `index.html` (gauges, live feeds, 3D arm and cameras, mat map, skins), `scene_detect.py` (objects on the mat from the overhead camera), `skins/`, `urdf/` |
| `tools/` | Mac-side helpers: `jog_capture.py` (jog-and-fit of the arm base), `pick_loop.py` (iterative pick-and-place learning), `cams.sh` / `cams_avf.py` (camera wall on monitor 2), `teleop_robust.py` (teleop that survives a stalled bus), `ping_arm.py` (raw servo ping), `clean_dataset.py` / `trim_dataset.py` (dataset repair), `film.py` (episode films), `mlflow_backfill.py` / `mlflow_ui.sh` (MLflow), `cf_token.py` (Cloudflare token helper) |
| `plugins/` | LeRobot plugins: `lerobot-robot-busheal` (bus recovery, gripper limits, state relay), `lerobot-camera-avf` (AVFoundation cameras by name), `lerobot-policy-actasync` (ACT with the async engine) |
| `sim/` | Simulation rounds: ACT on a simulated pick task, cloud training notes, success tables |
| `tests/` | 600 tests, all synthetic (rendered boards and markers, fake `v4l2-ctl`, fake bus, stub HTTP server). `python -m pytest tests -q`, about 3 minutes |
| `docs/` | The four guides below, plus CNC panel notes from a side project |
| `env.sh` | Mac shell commands (`so101-train`, `so101-finetune`, `so101-dash`, ...) |
| `NEXT_STEPS.md` | The running log: what works, what broke and why, what is next |
| `pyproject.toml`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Lint (ruff) and test configuration; CI runs `ruff check .` and the suite on every push |
| `attic/` | Unreferenced leftovers kept for the record (see `attic/README.md`) |

## The guides

- [`docs/camera-calibration.md`](docs/camera-calibration.md): checkerboard intrinsics, four 2 in ArUco
  markers on a Fiskars 18x12 in cutting mat, the arm base pose by jogging to grid points, the LeRobot
  joint zero offsets, and why wrist roll must be held constant. Also the self-healing re-solve when the
  camera stand gets bumped.
- [`docs/camera-settings.md`](docs/camera-settings.md): locking exposure, white balance and focus, and
  the measured behaviour of the Logitech C920 revision 046d:08e5 that makes a naive lock white out
  (gain forced to 255 after auto to manual, exposure re-derived at every stream start). A guard service
  and a udev rule keep the lock; recording refuses to start unless the cameras are locked.
- [`docs/camera-health.md`](docs/camera-health.md): frame-rate, duplicate-frame and drift checks for
  the three cameras, usable in a pre-recording checklist.
- [`docs/mat-scene.md`](docs/mat-scene.md): the live map of the mat. Background model, kinematic arm
  mask from the follower's joints, marker-based drift refit, persistence filter, and the APIs the
  dashboard draws from.

## Hardware

SO-ARM101 leader and follower (Waveshare bus boards, 12 V follower supply), Jetson Orin Nano on
JetPack 6.2, a Mac for training and the dashboard, two Logitech C920 cameras (overhead and front) and
an icSpring gripper camera, a Fiskars 18x12 in self-healing cutting mat as the work surface.

## Status, honestly

Calibration, camera tooling, the mat map and the dashboard are solid and tested. Scripted picking is
work in progress: the arm reaches within about 3 mm of a commanded point, but a hex nut lands 10 to
40 mm off because it shifts in the single-moving-jaw gripper. The next step is to measure the part in
the jaws with the gripper camera before placing.

## Not in the repo

Datasets, checkpoints, JetPack images and the Python environments are deliberately absent (see
`.gitignore`). They live in `~/.cache/huggingface/lerobot`, `outputs/` and on the Jetson.

## Licence

MIT, see [`LICENSE`](LICENSE). The SO-101 URDF and meshes under `dashboard/urdf/` come from
TheRobotStudio's SO-ARM100 project under Apache-2.0, see [`dashboard/urdf/README.md`](dashboard/urdf/README.md).
