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
| `record` | runs `autoloop.py --cycles N --record-demos --demo-repo lastlight/so101-nut-pick`; the recorder writes on the Jetson (`~/so101/outputs/demos/<session>/so101-nut-pick`, no copy made here; the hash is the recorder's); only kept episodes are in the main dataset; fails under `--min-kept` (10) kept episodes; hashes the dataset (`demo_recorder.dataset_hash`, must equal the recorder's own hash when it reports one) | dataset path, hash, episode and frame counts, the session's scripted success rate (the baseline), the `dataset-version` run id |
| `train` | `--where hf` (default) or `--where jetson`, never the Mac; see "Where training runs". ACT, chunk 100, input features built from the dataset (no environment_state); tracked in MLflow either way | run in `so101-act` with params, loss curve, steps/s, tags `where`, `dataset_hash`, `dataset_run_id`, `learn_loop_run` (and `hf_job`, `hf_repo`) |
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

## Where training runs: Hugging Face or the Jetson, not the Mac

TC, 11 Nov 2026: "the training should take place on hugging face or the jetson". The Mac is no training target in `learn_loop.py`; `--where local` exists only with `--dry-run`, for the fakes, and is refused otherwise. `train --where hf|jetson`, default `hf`. `plan` and the start of `train` print the same numbers as below.

| | `--where hf` (default) | `--where jetson` |
|---|---|---|
| What | an HF Job: `hf jobs run --flavor a100-large --detach --secrets HF_TOKEN <image> bash -c "lerobot-train ... --policy.push_to_hub=true"` | `lerobot-train` over ssh, CUDA, detached with nohup and polled |
| Defaults | 10 000 steps, batch 32, save every 2 500 | 20 000 steps, batch 4, 1 worker, save every 2 000 |
| Speed (measured) | ~3.1 step/s, ACT batch 32 on an A100 (`sim/CLOUD_ROUNDS.md`, 21 Oct 2026) | ~1.9 step/s at batch 4 (`so101-finetune`, `jetson/env.sh`); batch 4 is its ceiling on the shared 8 GB |
| Time | 10 000 steps: 54 min + 5 min startup (an estimate, never measured) = about 1 h | 20 000 steps: 2.9 h |
| Cost | $2.50/h: about $2.45; the job `--timeout` (1.5 x the estimate, whole hours) caps it at $5.00 | none, but the Jetson is busy that long |
| Evidence for the rate | the sim jobs: long 10 801 s = $7.50, v3 4 911 s = $3.41; the SmolVLA job (16 Oct, batch 32, 10k steps) took 61 min = about $2.55 | the `so101-finetune` header comment (one number, not a log); the Mac (not used any more) did 1.45 step/s at batch 8 |
| Dataset | pushed first to a PRIVATE Hub dataset repo `<hf-user>/so101-nut-pick-<hash8>` with a `v3.0` tag (LeRobot needs it); if the dataset is only on the Jetson it is rsynced to the Mac first and its hash checked against the recorder's | read where the recorder wrote it on the Jetson; nothing copied |
| Checkpoint back | `hf download <hf-user>/<job>` (the model repo the job pushed to), then `jetson/prep_ckpt.py` on it (cloud checkpoints carry keys our lerobot 0.6.1 rejects), then `register` with `--hf-repo`/`--hf-job` | one rsync of `checkpoints/last/pretrained_model` to the Mac for `register` |
| Guard | does nothing, uploads nothing, launches nothing without `--yes`; prints the flavour, wall time, cost first. A job already launched (id in `state.json`) is polled by a repeated command without `--yes`, never launched twice | refuses while: `.arms_busy` or `.cams_busy` exist, a `lerobot-*` job runs, a `pick_seq --serve` session or other arm script runs, a configured recorder has an open episode, or `autoloop.py`/`tidy.py` runs on the Mac. While training it holds `.cams_busy` (frees the dashboard's camera memory, the NEXT_STEPS gotcha) and releases it when the process ends |
| Resume | repeat the command: polls the job; after an ERROR/TIMEOUT `train --force --yes` starts a new one (the model repo keeps the checkpoints it pushed) | repeat the command: relaunches `--resume=true` from `checkpoints/last` |

The exact commands, the live ones included:

```
tools/learn_loop.py train --name nut1                      # prints the plan, refuses: "not launching ... repeat with --yes"
tools/learn_loop.py train --name nut1 --yes                # HF: about 1 h, about $2.45 (cap $5.00); needs `hf auth login` and HF_TOKEN in the environment
tools/learn_loop.py train --name nut1 --where jetson       # Jetson: 2.9 h, free; run `so101-stop` and stop autoloop/tidy first
```

Things in the HF path that nothing in the repo records and that the first live launch must confirm: the job image (`--hf-image`, default `huggingface/lerobot-gpu`; `hf jobs run --dry-run` prints the resolved configuration without submitting), and that the image's `lerobot-train` accepts the same flags (the earlier jobs were launched by hand and their command line was not kept; NEXT_STEPS.md only has the failure notes: dataset repo needs the `v3.0` tag, SmolVLA needs the camera rename). SmolVLA (`--policy smolvla`) builds the NEXT_STEPS recipe for the HF backend only; it does not fit on the Jetson and has not been launched from here.

Past Mac numbers (`outputs/train/*.log`), for reference only: `act_lighter` 25 000 steps at batch 8 in 4 h 38 min (1.45 step/s), `ft_lighter` 8 000 steps in 1 h 34 min, sim runs 3 to 4 step/s.

## How long each stage takes

| Stage | Estimate | Evidence |
|---|---|---|
| record, N cycles | N x 86 s (30 cycles: 43 min; 100: 2.4 h) | median `cycle_s` of 19 real cycle runs in `so101-autoloop` (86 s, mean 101 s). About half of the tries are kept (`first10b` 5/10, `first10` 3/4), so 30 cycles give 12 to 22 episodes. `act_lighter` learned from 16 episodes (435 episodes seen in 27 epochs); 40 cycles for about 20 episodes is the first target, more is better |
| train, HF Job, 10 000 steps batch 32 | about 1 h, about $2.45 (`--steps 20000`: 1.9 h, $4.7) | 3.1 step/s on an A100 (`sim/CLOUD_ROUNDS.md`) + 5 min startup; $2.50/h |
| train, Jetson, 20 000 steps batch 4 | 2.9 h, free | 1.9 step/s (`so101-finetune`); `--steps 8000` is 1.2 h |
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
2. The other agents' pieces are on this branch's base: `autoloop --record-demos` without `--dry-run` actually records (phase 2), and the persistent `pick_seq --serve` session. Hooks I need from them, none exist yet: (a) a way to RELEASE the arm port from a `pick_seq --serve` session before `lerobot-rollout` starts and take it back after (the rollout and a resident serial session cannot both hold `/dev/so101_follower`; `jetson_arm_jobs` does not look for `pick_seq`); (b) none now: the recorder's folder on the Jetson is read as it is (`~/so101/outputs/demos/<session>/<name>`, or `--jetson-dataset`).
3. On the Jetson: `lerobot-rollout --help | grep max_relative_target` shows the flag (lerobot 0.6.1 has it on `SOFollowerConfig`; the rtc engine's use of it is the thing to confirm), `so101-cams-lock` ok, `so101-cams-check` exit 0, `so101-perf` (max clocks) for the rollout.
4. `touch ~/so101/outputs/run/autoloop.stop` works as the stop file (the same one the autoloop uses); you know where the physical STOP and the dashboard STOP are. `so101-stop` on the Jetson stops the rollout.
5. For `--where hf`: `hf auth whoami` works on the Mac and `HF_TOKEN` is set (the Job gets it with `--secrets HF_TOKEN`); you accept the spend. For `--where jetson`: arm and cameras idle (the command checks and refuses). Mac store reachable (`curl -s http://127.0.0.1:5050/health`) or `--mlflow-uri` set.

Then, in this order, each a separate command so a surprise costs one stage:

1. A throw-away first pass under its own name: `record --name nut-smoke --cycles 4 --min-kept 1` and watch the 4 cycles. Check the dataset hash and the episode count in `outputs/learn_loop/nut-smoke/state.json`. (Do not reuse that name for the real run: a second `record` in the same loop would need `--force` and is not tested to append to the first dataset.)
2. The real run under a new name: `record --name nut1 --cycles 40` (about an hour). `--min-kept` 10 holds the next stage back if the keep rate is bad.
3. `train --steps 1000` for a smoke pass on the real data (HF: about 8 min and $0.32 with `--yes`; Jetson: about 9 min) to see the whole path work and the loss fall; then `train --force --yes` for the real one (HF: about 1 h, $2.45, cap $5.00; `--where jetson` instead: 2.9 h, after `so101-stop` and stopping autoloop/tidy). `--force` starts over, a plain repeat after a crash resumes or polls. It uses the Mac GPU: close other GPU-heavy things.
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

`tools/learn_loop.py` (CLI, stages, state, report), `tools/learn_loop_train.py` (HF and Jetson training backends, the busy check, the numbers), `tools/learn_loop_eval.py` (`rollout_script`, `RolloutWatch`, `SshTransport`, `PolicyEval`), `tools/learn_loop_fakes.py` (dry-run and test fakes), `tests/tools/test_learn_loop.py`, `tests/tools/test_learn_loop_train.py`, `tests/tools/test_learn_loop_eval.py`, `tests/tools/learn_loop_helpers.py`, the `so101-learn-loop` function in `env.sh`. Nothing in `autoloop.py`, `pick_loop.py`, `pick_seq.py`, `demo_recorder.py`, `demo_session.py` or the dashboard was changed.
