# Deploying to the Jetson

One tool, `tools/jetson_deploy.py`, puts the repo's `jetson/*.py`, `jetson/*.sh` and the two Jetson-side plugins onto the Jetson's flat `~/so101/`, and records what is there. Before it, deploys were single-file `scp` with no record, and the two sides drifted.

```
tools/jetson_deploy.py status   [--ref REF]                       # read only; exit 0 only when the Jetson is at REF
tools/jetson_deploy.py deploy   [--ref REF] [--files a b ..] [--yes] [--restart recorder|camd] [--allow-system]
tools/jetson_deploy.py rollback [--to BACKUP] [--yes]
```

## What it does

- Files come from `git show <sha>:path`, so uncommitted edits never deploy. `--yes` without `--ref` refuses when tracked files under `jetson/` or `plugins/` have uncommitted changes.
- `deploy` is a dry run (it prints the plan, who uses each file and the restart step) until you add `--yes`.
- It refuses to write while anything is busy: `~/so101/.arms_busy` or `.cams_busy`, a `lerobot-(teleoperate|record|rollout|calibrate|replay)` job, a `pick_seq.py` or other arm script, a recorder with an open episode (asked via `demo_recorderd.py ctl status`), or `tools/tidy.py` / `autoloop.py` / `pick_loop.py` running on the Mac. A Jetson it cannot read counts as busy.
- Each replaced file is copied to `~/so101/deploy_bak/<UTC time>_<shortsha>/` (with a `BACKUP.json`), written as `name.deploy_tmp` then `mv`'d, and checked by md5. `~/so101/DEPLOYED.json` then records the commit, time and the md5 of every deployable file. `status` calls the Jetson **dirty** when a file differs from that record (hand edit since the last deploy), **behind** when files differ from REF, **at** REF otherwise.
- Nothing is ever deleted by a deploy (files only on the Jetson, such as `ping_arm.py` and `servo_health.py`, are listed and left alone). `rollback` restores the newest backup (or `--to`), removes files that deploy added only if still unedited, and rewrites `DEPLOYED.json`. Same busy checks.

## What is deployable

| Class | Files | Rule |
|---|---|---|
| deploy | `jetson/*.py`, `jetson/*.sh` (not the system list), `plugins/lerobot-robot-busheal`, `plugins/lerobot-policy-actasync` | normal deploy. The plugins are editable installs in the Jetson venv, so a copied file is live at the next job start. `lerobot-camera-avf` is the Mac's and is not deployed. `jetson_setup.sh` is copied but never run. |
| calib | `jetson/calib/*` | The Jetson is the source of truth (`cam_calib.py`, `cam_lock.py` write `~/so101/calib` there; the repo copy is the synced mirror). Never in a plan; only `--files calib/NAME --allow-system`. |
| system | `99-so101-cams.rules`, `so101-cam-lock.service`, `90-so101-hotspot.dispatcher`, `so101-hotspot.nft`, `hotspot.sh` | Need root or change the network. Never part of a normal deploy. `--files NAME --allow-system` only stages the file in `~/so101` and prints the sudo lines; the tool never runs sudo. |

## Which process picks up what

| Changed file reaches | How it is picked up |
|---|---|
| `arm_state.py` (and `servo_telemetry.py`) | Started over ssh by the Mac dashboard; the next poller start loads it. Force: restart `so101-dash`. |
| `pick_seq.py`, `servo_telemetry.py` | Next `pick_seq.py --serve` session (started per session by the Mac). Running sessions keep the old code. |
| `camd.py` | `camd.py serve` restart; also imported by `demo_cams.py`, so the recorder needs one too. |
| `demo_recorderd.py`, `demo_recorder.py`, `demo_cams.py`, `lab_config.py` | `demo_recorderd.py serve` restart. |
| `env.sh` | New shells only; an open shell needs `source ~/so101/env.sh`. |
| `cam_lock.py` | Resident `cam_lock.py guard`, if running: manual restart with `.cams_busy` absent. |
| everything else, plugins | Next run / next lerobot job. |

The mapping is computed from the imports of the files at REF, so it follows the code.

## The only restarts the tool will do (`--restart NAME`, after a successful deploy)

- `recorder`: only if `demo_recorderd.py serve` is running and `ctl status` shows no open episode (configured but idle is allowed; the Mac re-sends `configure` when `autoloop --record-demos` starts). Stops it with SIGTERM, waits, starts `nohup nice -n 5 .venv/bin/python demo_recorderd.py serve` with the log in `outputs/run/`.
- `camd`: only if `camd.py serve` is running and no recorder is configured. Dashboard camera tiles reconnect.

Both re-check the busy flags. Nothing else (arm_state, pick_seq, anything with the arm) is restarted by the tool.

## Overnight preflight

An unattended run should refuse to start when the Jetson is not at the commit it was tested on. Read-only, no arm motion:

```python
from tools.jetson_deploy import deployed_matches; ok, why = deployed_matches("<tested sha>")   # (False, reason) also when the Jetson is unreachable
```

It is not wired into `tidy.py` or `autoloop.py`; whoever owns those calls it.

## Tests

`tests/tools/test_jetson_deploy.py` uses a fake git and an in-memory Jetson (the transport is injected). Nothing there can reach a real machine.
