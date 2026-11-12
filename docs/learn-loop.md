# The learn loop: record, train, register, deploy, evaluate, report

`tools/learn_loop.py` (`so101-learn-loop` in `env.sh`) is one resumable routine for the whole round trip: the scripted pick records its own successful picks as a LeRobot dataset, an ACT policy is trained on them, the policy goes through the model registry to the Jetson, and the Jetson runs it on the real task for M trials judged by the same overhead-camera judge the autoloop uses. The last stage puts the success rate next to the scripted baseline. Everything is tracked in MLflow and linked: eval trial -> registry version -> train run -> dataset hash.

Status (11 Nov 2026): built and tested on fakes only. No stage has run on the arm, the Jetson or a real `lerobot-train`. The first live pass is the checklist at the bottom.

```
tools/learn_loop.py plan      --name nut1                      # stages, state, estimates; touches nothing
tools/learn_loop.py all       --name nut1 --cycles 40 --trials 10 --trial-s 25
tools/learn_loop.py record    --name nut1 --cycles 40          # one stage at a time works too
tools/learn_loop.py train     --name nut1 --steps 20000
tools/learn_loop.py register  --name nut1
tools/learn_loop.py deploy    --name nut1
tools/learn_loop.py evaluate  --name nut1 --trials 10 --trial-s 25
tools/learn_loop.py report    --name nut1
tools/learn_loop.py all       --name smoke --dry-run           # everything on fakes, ~10 s, nothing leaves the Mac
```

## What already existed (and what this builds on)

| Need | What was there | Where |
|---|---|---|
| Record scripted picks | `autoloop.py --record-demos` + `DemoSession.keep_decision` + `DemoRecorder` write a kept-only LeRobot v3 dataset (rejects in a sibling dataset), a `dataset-version` MLflow run with a dataset hash | `tools/autoloop.py`, `tools/demo_session.py`, `jetson/demo_recorder.py`, `docs/scripted-demos.md` |
| Train | `so101-train` (Mac, MPS), `so101-finetune` and the Jetson's `so101-train` (CUDA, batch 4), `so101-sim-train`; logs in `outputs/train/<job>.log`; no MLflow while training, `tools/mlflow_backfill.py` imports them afterwards into `so101-act` | `env.sh`, `jetson/env.sh`, `tools/mlflow_backfill.py` |
| Registry and deploy | `so101-policy register / deploy / rollback / status`: MLflow registry `so101-registry`, rsync to `~/so101/policies/<name>/v<N>/` with `current` -> `v<N>`, `DEPLOYED.json`, sha256 checked end to end, alias `jetson` | `tools/so101_policy.py`, `docs/mlops/registry-and-deploy.md` |
| Run a policy | the dashboard's POLICY button: `so101-nudge; so101-run-async 90 <ckpt> 90` = `lerobot-rollout --inference.type=rtc --policy.path=<ckpt> --policy.device=cuda ...` with the cameras taken from the dashboard (`so101-cams-take`); VLA button: `so101-run-vla`; stop: `so101-stop` | `dashboard/server.py` (`ARM_MODES`), `jetson/env.sh` |
| Judge, reset, guardrails | `Runner.judge` (overhead camera after a settle, picked = moved > 20 mm and jaws, landed = within the radius), `Runner.reset_and_judge` (scripted pick to a fresh spot), `Guard` (stop file, panel, markers, arm_busy, workspace box, consecutive failures, nut lost) | `tools/autoloop.py`, `tools/guardrails.py` |

Three things the existing pieces did NOT give, found by reading the code, and what this does about each:

1. **`lerobot-rollout` has no workspace box.** It has a joint-step cap (`--robot.max_relative_target`, `SOFollower.send_action`) and nothing else. The scripted pick checks every waypoint's FK tip against the mat box before the plan runs; a policy picks its own waypoints at 30 Hz. See "The safety envelope" below.
2. **The recorder writes `observation.environment_state`** (the lab-config one-hot, constant per episode); `lerobot-rollout` cannot supply it at run time, so a policy trained on the dataset as it is would fail on the Jetson. `train` builds `--policy.input_features` from the dataset's `meta/info.json` WITHOUT it (state + the camera images) and logs the dropped keys as a run param. A test compares the generated features with LeRobot's own `dataset_to_policy_features`.
3. **The arm slumps when `lerobot-rollout` exits** (`disable_torque_on_disconnect` defaults to true). The rollout script passes `--robot.disable_torque_on_disconnect=false` (what `nudge.py` already does) so the arm holds its pose until the scripted park.

## The stages

State is `<out-dir>/<name>/state.json` (default `outputs/learn_loop/<name>/`, git-ignored). Repeat a command to resume: a stage marked `done` is skipped, a failed or interrupted one runs again, `--force` redoes a stage and marks every later stage pending.

| Stage | Does | Output / link |
|---|---|---|
| `record` | runs `autoloop.py --cycles N --record-demos --demo-root <run>/demos --demo-repo lastlight/so101-nut-pick`; only kept episodes are in the main dataset; fails under `--min-kept` (10) kept episodes; hashes the dataset (`demo_recorder.dataset_hash`, must equal the recorder's own hash when it reports one) | dataset path, hash, episode and frame counts, the session's scripted success rate (the baseline), the `dataset-version` run id |
| `train` | `lerobot-train` ACT: chunk 100, batch 8, 20000 steps, MPS, input features from the dataset (no environment_state); resumes from `checkpoints/last` after a crash | run in `so101-act` (same experiment as the backfilled runs) with params, loss curve, steps/s, tags `dataset_hash`, `dataset_run_id`, `learn_loop_run` |
| `register` | `so101-policy register` into `so101-registry`, name `so101-nutpick` (`-dry` for dry runs), alias `candidate`; adds tags `dataset_hash`, `train_run_id`, `record_session` to the version and its run and the metrics (final loss, steps, episodes, scripted success rate) to the registry run; reuses a version with the same weights and dataset hash instead of making a second one | model version, the train run gets `registry_version` back |
| `deploy` | `so101-policy deploy name@version` (rsync, sha256 verified, `current` flipped, alias `jetson`), then `so101-policy status` must agree | `~/so101/policies/so101-nutpick/current` on the Jetson |
| `evaluate` | M trials, below | `eval.json`, an `eval-session` run and one `eval-trial` run per trial, each tagged with the whole chain |
| `report` | success rate with a 95 % Wilson interval, median placement error, against (a) this loop's own scripted session and (b) the last 5 real scripted sessions pooled from `so101-autoloop`; `report.md`, `report.json`, metrics on the loop's parent run | `learnloop-<name>` run in `so101-learn-loop` |

### One evaluate trial

1. `guard.check_before_cycle`: stop file, dashboard answering, Jetson up, no other arm job (`lerobot-*` on the Jetson), markers visible.
2. Locate the nut (the autoloop's `locate`); if it sits on the target, scripted pre-reset away from it first.
3. Rollout: the remote script (nudge, `so101-cams-take`, `lerobot-rollout` rtc, speed cap, `--duration` T, `so101-cams-release`) goes over `ssh <host> bash -s`; the Mac polls every 0.25 s while it runs (below).
4. Park: the scripted `rest` plan through the guarded `run_plan`.
5. Wait for the cameras to come back (`/api/scene` `age_s` < 2 s), then the autoloop's judge: `picked and landed within --radius-mm` (15), nothing different for the policy.
6. Scripted reset to a new spot (`reset_and_judge`), then `check_after_cycle`: 3 failed trials in a row (`--max-consecutive-failures`) or 3 lost nuts end the session.

The policy gets no retry; the scripted pick gets one per cycle. The report says so.

### The safety envelope

| Guard | Scripted pick | Policy rollout |
|---|---|---|
| Speed cap | `pick_loop.SPEED` 50 deg/s closed, 80 open | `--robot.max_relative_target=2` = 2 deg per 30 Hz tick = 60 deg/s (`--max-rel-deg`), enforced inside lerobot |
| Workspace box (mat shrunk 20 mm, 4 in ceiling, -10 mm floor) | checked on the FK tip of every waypoint BEFORE the plan runs | `RolloutWatch` on the Mac: the follower's read-back joints from `/api/state` (fed by the busheal plugin during a rollout) through the same `fk_mat_mm` + `box_check`, every 0.25 s; on a violation `so101-stop` over ssh, trip `workspace`. A tripwire with a latency of the state chain (arm_state over ssh -> dashboard -> poll, roughly 0.3 to 1 s), not a clamp; the speed cap bounds how far the arm gets in that time |
| Stop file | checked before every plan | checked every 0.25 s during the rollout: `so101-stop`, clean stop |
| State lost | n/a | no usable follower fix for 2.5 s (or none in 30 s at the start, or none at all during a finished rollout: trip `unwatched`) stops the arm; joints not in degrees count as lost |
| Consecutive failures, nut lost | `Guard.check_after_cycle` | same |
| Rollout hung / crashed / killed | n/a | no exit within T + 40 s: `so101-stop`, trip `rollout_hung`; exit before T that is not SIGTERM: `rollout_failed`; SIGTERM (TC at STOP, `so101-stop`): clean `stop_arm` |
| After a trip | rest only if the readback is healthy | after a policy trip NO further arm command (not even rest): the arm holds torque where it is and the session ends; you look first |

What is missing and where it would go: an in-process clamp (FK of the commanded action against the box inside the rollout loop, as the scripted plan check does). That needs a small wrapper around `lerobot-rollout`'s robot (`robot_wrapper.py`) on the Jetson; it was not written because it cannot be tested without the arm. Until it exists the tripwire above plus the speed cap are the envelope, and the first live trials should be run with a hand on STOP (checklist).

## Where training should run: the Mac (MPS)

Evidence from the repo's own logs (`outputs/train/*.log`, `env.sh`, `NEXT_STEPS.md`):

| Run | Machine | Batch | Steps | Speed | Wall |
|---|---|---|---|---|---|
| `act_lighter` (real, 16 episodes, 2 cameras) | Mac MPS | 8 | 25 000 | 1.45 step/s (updt 0.67 s, 12 samples/s) | 4 h 38 min |
| `ft_lighter` (fine-tune) | Mac MPS | 8 | 8 000 | 1.49 step/s | 1 h 34 min |
| `act_sim_pick_v1` (sim, 320x240) | Mac MPS | 8 | 20 000 | 3.97 step/s | 1 h 26 min |
| `act_sim_pick_v4b` | Mac MPS | 8 | 10 000 | 2.98 step/s | 54 min |
| `so101-finetune` (header comment) | Orin Nano CUDA | 4 | - | ~1.9 step/s (7.6 samples/s) | - |
| SmolVLA 10k steps | HF Jobs A100 | 32 | 10 000 | - | 61 min, about 2.55 USD |

- The Mac does 12 samples/s at batch 8; the Orin 7.6 at batch 4, and batch 4 is its ceiling (`NEXT_STEPS.md`: the NVML allocator assert at the 8 GB shared memory limit above that, dashboard camera streams must be paused). The Mac is faster per sample and has the headroom.
- The Jetson is also the box that holds the arm, the cameras and the recorder. Training there blocks the next record or evaluate pass (one 8 GB, CUDA contended with `lerobot-rollout`).
- The recorded dataset is on the Mac today (the recorder runs in the autoloop process); when the recorder moves to the Jetson (phase 2) `record --pull-from jetson:<path>` rsyncs it to the Mac.
- SmolVLA does not fit on the Jetson (8 GB) and is untested on MPS; the only measured run is the A100 Job. `--policy smolvla --device cuda` builds the command from the NEXT_STEPS recipe (base weights, top/wrist renamed to camera1/2, one empty camera) for a CUDA host; it has not been run from here. Default is ACT.

`--device cuda` trains on a CUDA machine you run the command on (the Jetson itself: run learn_loop there, with `--batch 4 --workers 1`); the code has no remote-train mode and does not ssh to start training.

## How long each stage takes

| Stage | Estimate | Evidence |
|---|---|---|
| record, N cycles | N x 86 s (30 cycles: 43 min; 100: 2.4 h) | median `cycle_s` of 19 real cycle runs in `so101-autoloop` (86 s, mean 101 s). About half of the tries are kept (`first10b` 5/10, `first10` 3/4), so 30 cycles give 12 to 22 episodes. `act_lighter` learned from 16 episodes (435 episodes seen in 27 epochs); 40 cycles for about 20 episodes is the first target, more is better |
| train, 20 000 steps, batch 8, MPS | 3.8 h (`plan` prints it) | 1.45 step/s from the logs above; `--steps 8000` is 1.5 h (the `ft_lighter` fine-tune ran 8k steps) |
| register | 30 to 90 s | the checkpoint is 197 MB (`act_lighter`), copied into the artifact store |
| deploy | about 1 to 2 min | rsync of 197 MB over the LAN, sha256 on both sides |
| evaluate, M trials of T s | M x (T + ~45 s): 10 x 70 s = 12 min | the 45 s (park, cameras back, settle, judge, scripted reset ~ 86 s cycle minus its pick) is an estimate until the first live run |
| report | seconds | |

## MLflow

One URI for everything: `--mlflow-uri`, else `SO101_MLFLOW_URI` / `MLFLOW_TRACKING_URI`, else the Mac store; the registry tool is pointed at the same one. `--dry-run` uses a private store in the run folder and `-dry` experiment and model names.

| Experiment | Run | Key fields |
|---|---|---|
| `so101-datasets` | `dataset-<name>-<hash8>` (the recorder's, kind `dataset-version`) | `dataset_hash`, counts |
| `so101-act` | the train job | params policy, steps, batch, device, `dataset_hash`, `dropped_features`; loss series, `final_loss`, `steps_per_s`; tags `dataset_run_id`, `learn_loop_run`, later `registry_version` |
| `so101-registry` | `<name>:register`, model version | tags `dataset_hash`, `train_run_id`, `sha256`; metrics from the training |
| `so101-learn-loop` | `learnloop-<name>` (parent), `eval-<name>` (session), `eval-trial-<n>` | every trial tagged `policy_name`, `policy_version`, `registry_run_id`, `train_run_id`, `dataset_hash`, `dataset_run_id`; the parent gets the report numbers |
| `so101-autoloop` | the scripted baseline (read only) | session runs: `cycles`, `success_rate` |

## The live checklist (hardware, with TC possibly at STOP)

Before the first real run:

1. `tools/learn_loop.py all --name smoke --dry-run` passes on the Mac (about 10 s). `ruff` and the suite are green.
2. The other agents' pieces are on this branch's base: `autoloop --record-demos` without `--dry-run` actually records (phase 2), and the persistent `pick_seq --serve` session. Hooks I need from them, none exist yet: (a) a way to RELEASE the arm port from a `pick_seq --serve` session before `lerobot-rollout` starts and take it back after (the rollout and a resident serial session cannot both hold `/dev/so101_follower`; `jetson_arm_jobs` does not look for `pick_seq`); (b) where the Jetson-side dataset lives, so `--pull-from` can fetch it.
3. On the Jetson: `lerobot-rollout --help | grep max_relative_target` shows the flag (lerobot 0.6.1 has it on `SOFollowerConfig`; the rtc engine's use of it is the thing to confirm), `so101-cams-lock` ok, `so101-cams-check` exit 0, `so101-perf` (max clocks) for the rollout.
4. `touch ~/so101/outputs/run/autoloop.stop` works as the stop file (the same one the autoloop uses); you know where the physical STOP and the dashboard STOP are. `so101-stop` on the Jetson stops the rollout.
5. Mac store reachable (`curl -s http://127.0.0.1:5050/health`) or `--mlflow-uri` set.

Then, in this order, each a separate command so a surprise costs one stage:

1. A throw-away first pass under its own name: `record --name nut-smoke --cycles 4 --min-kept 1` and watch the 4 cycles. Check the dataset hash and the episode count in `outputs/learn_loop/nut-smoke/state.json`. (Do not reuse that name for the real run: a second `record` in the same loop would need `--force` and is not tested to append to the first dataset.)
2. The real run under a new name: `record --name nut1 --cycles 40` (about an hour). `--min-kept` 10 holds the next stage back if the keep rate is bad.
3. `train --steps 2000` for a smoke pass on the real data (about 25 min) to see loss fall and steps/s; then `train --force --steps 20000` (about 3.8 h; `--force` starts over, a plain repeat of the command after a crash resumes). It uses the Mac GPU: close other GPU-heavy things.
4. `register`, `deploy`: check `so101-policy status` prints OK and `~/so101/policies/so101-nutpick/current` exists on the Jetson.
5. First evaluation with a person at STOP: `evaluate --trials 2 --trial-s 15`. Watch the first rollout from start to finish: the arm should leave the nudge pose, reach toward the nut, stay inside the mat. Expect nothing good from an under-trained policy; the point is the guard: you can test the tripwire by touching the stop file during the rollout and checking the arm stops within a second or two.
6. `evaluate --trials 10 --trial-s 25`, then `report`. Read `report.md`: with 10 trials the interval is wide (5/10 is 24 to 76 percent); the verdict line says when the difference is not distinguishable.

During any rollout: hand near STOP. A trip prints one JSON line, ends the session, and leaves the arm holding torque in place (policy trips). To continue after fixing the cause repeat the same `evaluate` command: finished trials are kept, it continues at the next. Exit codes: 0 done, 1 a stage failed, 2 a guardrail tripped or the stop file was set.

## What still needs the real hardware (not exercised by any test)

- `autoloop --record-demos` without `--dry-run` (phase 2 of `docs/scripted-demos.md`) and a real `lerobot-train` on the generated command (the `--policy.input_features` JSON is parsed by draccus in a check done while writing this, not by a training run).
- The remote rollout script: `so101-nudge`, `lerobot-rollout --robot.max_relative_target` under rtc, camera hand-over to and from the dashboard, how long the cameras take to come back (`--fresh-s 20`), the arm's pose after `return_to_initial_position`.
- `RolloutWatch` latency and the follower joints during a rollout (busheal plugin state in `/dev/shm`, `src` `busy-tele`), including `mode: deg`.
- That a ACT policy trained on 15 to 20 scripted episodes does anything useful. The sim policy trained on 300 demos reached 30 % on unseen layouts at 20k steps (`NEXT_STEPS.md`); a first live success rate near zero is a likely and uninformative result, and says to record more.
- Checkpoint compatibility: Mac and Jetson both run lerobot 0.6.1, so no `jetson/prep_ckpt.py` step is added; if the Mac is upgraded first, run `prep_ckpt.py` on the deployed checkpoint (and `prep_vla_ckpt.py` for SmolVLA).

## Files

`tools/learn_loop.py` (CLI, stages, state, report), `tools/learn_loop_eval.py` (`rollout_script`, `RolloutWatch`, `SshTransport`, `PolicyEval`), `tools/learn_loop_fakes.py` (dry-run and test fakes), `tests/tools/test_learn_loop.py`, `tests/tools/test_learn_loop_eval.py`, `tests/tools/learn_loop_helpers.py`, the `so101-learn-loop` function in `env.sh`. Nothing in `autoloop.py`, `pick_loop.py`, `pick_seq.py`, `demo_recorder.py`, `demo_session.py` or the dashboard was changed.
