# Learn MLflow in this lab: an evening of exercises

Ten exercises, in order, each ten to twenty minutes. They use the real store (`~/so101/mlflow/mlflow.db`),
the real tools, and a fake Jetson (a local directory `so101-policy` treats as the Jetson's `$HOME`), so
nothing moves the arm and nothing needs the Jetson switched on. The field guide
([`docs/mlflow_field_guide.html`](../mlflow_field_guide.html)) taught the UI on the `so101-act` training runs;
this evening is about the four new experiments: `so101-picks`, `so101-calibration`, `so101-recordings`,
`so101-registry`.

Every answer was measured on 10 Nov 2026 on a copy of the real store (mlflow 3.17.2, the Mac venv), running
exactly the commands below from the repo. Timings are from `time`; counts are from the MLflow client. Your
run ids, timestamps and hostnames will differ; the numbers that matter will not, because the fake arm, the
synthetic calibration scene and the fake checkpoints are deterministic.

**Where the practice data goes.** Exercises 3 to 7 write into the real store, which is the point: you learn
on the thing you will use. Exercise 10 removes it all. Note the time before you start, so the cleanup can
find what you made:

```sh
export SO101_PRACTICE_T0=$(($(date +%s) * 1000))     # MLflow start_time is milliseconds since the epoch
```

Two environment facts to keep in mind all evening. The loggers (`labtrack.py`, so `pick_loop.py`,
`cam_calib.py`, `record_track.py`) read `SO101_MLFLOW_URI` and fall back to the SQLite file;
`so101-policy` and the backfill read `MLFLOW_TRACKING_URI` and fall back to the same file. With neither set,
everything lands in `~/so101/mlflow/mlflow.db`, and the bare server shows it at once because it reads the same
file. Keep both unset tonight.

## 1. Start the server and take inventory

**Goal.** Know what the store holds before you add to it, and where the bytes are.

```sh
so101-mlflow start                                    # Mac only; --lan is for the Jetson, not needed tonight
so101-mlflow status
sqlite3 ~/so101/mlflow/mlflow.db "select experiment_id, name, artifact_location from experiments"
sqlite3 ~/so101/mlflow/mlflow.db "select key, count(*) from metrics group by key order by 2 desc"
```

**UI.** `http://127.0.0.1:5050`. The left column lists the experiments; `so101-act` has ten rows. The other
four experiments do not exist yet unless a real session already ran: experiments are created on first use.

**Question.** How many experiments are there, how many metric points does `loss` have across all runs, and
where do `so101-act`'s artifacts live on disk?

<details><summary>Answer</summary>
Three experiments (`Default`, `so101-act`, `MLflow Demo`), `loss` has 843 points (`lr` 190, `final_loss` 8).
`so101-act`'s artifact location is `~/so101/mlruns/1`, not `~/so101/mlflow/artifacts/`: the backfill created
the experiment with MLflow's default location (`./mlruns` of the directory it ran from) before
`labtrack.py` existed. Experiments the wrapper creates get `~/so101/mlflow/artifacts/<experiment>/` explicitly.
</details>

## 2. Loss is not success (the field guide's lesson, with the query that proves it)

**Goal.** Use the search box and sort the way you will when picking a checkpoint.

In `so101-act`, type each of these in the search box and sort by `final_loss`:

```
metrics.final_loss < 0.1
tags.family = "sim_pick"
params.warm_start != "scratch"
metrics.sim_success_pooled > 0
```

**UI.** The table filters as you type; param values need quotes even when they look like numbers; a metric
comparison uses the last logged value. Add the `sim_success_pooled` column with the Columns button.

**Question.** Which run has the lowest `final_loss`, and which is the only run with a measured rollout
success? Are they the same run?

<details><summary>Answer</summary>
`act_sim_pick_v2` has the lowest `final_loss` (0.072); `act_sim_pick_v4b` is the only run with
`sim_success_pooled` (0.84) and ranks second on loss (0.090). `metrics.final_loss < 0.1` returns exactly
those two runs; `tags.family = "sim_pick"` returns 5; `params.warm_start != "scratch"` returns 5 (every run
warm-started from another). The run you would deploy is not the one loss picks.
</details>

## 3. Three picks with the fake arm

**Goal.** See a `so101-picks` run appear from the real `pick_loop.py`, driven by the scripted arm the tests
use (`FakeArm` in `tests/mlops/test_pick_loop_tracking.py`: the nut is found where the last pick landed,
every waypoint sags 1.5 degrees, the nut lands a fixed offset from the target). This is what the smoke run
in [tracking.md](tracking.md) did by hand. A real session is `pick_loop.py --runs 4` with the arm and
dashboard up.

```sh
~/so101/.venv/bin/python - <<'PY'
import os, sys, json, tempfile
sys.path.insert(0, os.path.expanduser("~/so101/tests/mlops"))
import mlhelpers as H; from test_pick_loop_tracking import FakeArm, JPEG   # the scripted arm the tests use
pl = H.load(os.path.join(H.TOOLS, "pick_loop.py"))
tmp = tempfile.mkdtemp(prefix="picks_"); os.makedirs(f"{tmp}/jetson/calib")
json.dump({"units": "in"}, open(f"{tmp}/jetson/calib/top_table.json", "w")); os.symlink(os.path.join(H.JETSON, "labtrack.py"), f"{tmp}/jetson/labtrack.py")
pl.ROOT = tmp; pl.cam_calib = lambda: object(); pl.time.sleep = lambda s: None; pl.fetch_top = lambda: JPEG
arm = FakeArm(); pl.find_nut, pl.api, pl.run_plan = arm.find_nut, arm.api, arm.run_plan
sys.argv = ["pick_loop.py", "--runs", "3", "--target", "9", "6", "--alt", "6", "5", "--state", f"{tmp}/pick_state.json", "--gain", "0.5"]
pl.main()
PY
```

You should see three JSON lines (`"run": 1` to `3`, all `"picked": true`) and nothing on stderr.

**UI.** `so101-picks` now exists with three runs `pick-<session>-r1..r3`. Open r1: Parameters (9), Metrics
(14), Tags (`host`, `run`, `session`, `nut_in`, `target_in`), Artifacts (`frames/landing.jpg`, `plan.json`,
`waypoints.json`, `state/pick_state.json`; the JPEG is header-only here and will not render). Switch to the
chart view with `landing_err_mm` on y.

**Question.** What are the three landing errors, and which run does
`metrics.landing_err_mm < 2 and metrics.picked = 1` return? Why do r2 and r3 share a `target_in` tag that r1
does not?

<details><summary>Answer</summary>
r1 2.84 mm, r2 2.84 mm, r3 1.42 mm (the loop halves the sag bias each time and shifts the place target by
the measured error; the fake arm's offset is constant, so the third pick lands closer). The filter returns
only r3. `tags.target_in` is `6.00,5.00` for r1 and `9.00,6.00` for r2 and r3: odd picks go to `--alt`, even
picks and the last pick to `--target`, and the tag records where that pick was told to place. Each run has 9
params and 14 metrics; `plan.json` is 3903 bytes and `waypoints.json` 3506 on this run.
</details>

## 4. A calibration solve, and a command that must not log

**Goal.** Watch `cam_calib.py table` leave one run in `so101-calibration`, then watch `check` leave none.
The scene is the rendered ArUco table from `tests/calib/synth.py`; a real redo is `so101-calib table` on the
Jetson with the overhead camera.

```sh
~/so101/.venv/bin/python - <<'PY'
import json, os, sys, subprocess, tempfile, cv2
sys.path.insert(0, os.path.expanduser("~/so101/tests/calib")); import synth
d = tempfile.mkdtemp(prefix="calib_"); os.mkdir(f"{d}/scene")
layout = synth.default_layout(); rvec, tvec = synth.table_camera_pose()
cv2.imwrite(f"{d}/scene/table.png", synth.render_table(layout, rvec, tvec))
json.dump({**layout, "markers": {str(k): v for k, v in layout["markers"].items()}}, open(f"{d}/top_layout.json", "w"))
json.dump({"image_size": [synth.W, synth.H], "camera_matrix": synth.K_TRUE.tolist(), "dist_coeffs": synth.DIST_TRUE.tolist()}, open(f"{d}/top_intrinsics.json", "w"))
cc = os.path.expanduser("~/so101/jetson/cam_calib.py")
for cmd in (["table", "--source", f"{d}/scene"], ["check", "--source", f"{d}/scene"]):
    r = subprocess.run([sys.executable, cc, "--calib-dir", d, *cmd], capture_output=True, text=True); print(cmd[0], "->", r.stdout.strip()[:400], r.stderr.strip()[:200])
PY
```

**UI.** `so101-calibration` has one run, `table-top-<time>`, tags `kind=table`, `camera=top`. Metrics:
`residual_mm_0` to `_5`, `residual_mm_max`, `reproj_rms_px`, `camera_height_mm`, `points_used`, `markers`.
Artifacts: `calib/top_table.json` and `calib/top_table_check.png` (this one renders: the detected markers
drawn on the synthetic table).

**Question.** What is `residual_mm_max`, how tall does the solve think the camera is, and how many runs did
`check` add?

<details><summary>Answer</summary>
`residual_mm_max` 0.34 mm (0.336 in the metric), `camera_height_mm` 615.4, 24 `points_used` from 6 markers;
9 params, 11 metrics, 2 artifacts. `check` added nothing: the run count stays at one, because only commands
that write a calibration file log (`test_campose_armfit_and_check_locate` asserts exactly this). The search
`tags.kind = "table" and metrics.residual_mm_max > 2` returns 0 runs, which is the query you will use to
spot a drifted table solve.
</details>

## 5. A recording session without a camera

**Goal.** Run `so101-record` from the real `jetson/env.sh` on the Mac, against a fake `$HOME` with a stub
`lerobot-record` and a stub `cam_lock.py` (`RecordShell` in `tests/mlops/mlhelpers.py`), and read the run it
leaves in `so101-recordings`. A real session is `so101-cams-health 10 && so101-record cube "pick the cube" 10`
on the Jetson.

```sh
~/so101/.venv/bin/python - <<'PY'
import os, sys, tempfile, pathlib
sys.path.insert(0, os.path.expanduser("~/so101/tests/mlops")); import mlhelpers as H
uri = os.environ.get("SO101_MLFLOW_URI") or "sqlite:///" + os.path.expanduser("~/so101/mlflow/mlflow.db")
sh = H.RecordShell(pathlib.Path(tempfile.mkdtemp(prefix="rec_")), uri); sh.lock()
rc, out = sh.run('so101-record cube "pick the cube" 2'); print("rc", rc); print(out)
PY
```

You should see `cams locked: top ok; front ok; wrist ok`, then a `{"tracking": true, ...}` line from
`record_track.py start` and another from `finish` with `"episodes_total": 2`.

**UI.** `so101-recordings`, run `record-cube-<time>`, tags `kind=record`, `dataset=cube`, `resume=0`.
Parameters include `repo_id`, `task`, `episodes_requested`, `fps`, `cameras`, and one `cam_<role>_<control>`
per locked control from the stub `cam_settings.json`. Artifacts: `cameras/cam_settings.json`,
`cameras/cam_check.json`, `dataset/info.json`.

**Question.** How many params does the run have, what is `cam_top_exposure_time_absolute`, and why are there
only 6 metrics when [tracking.md](tracking.md) lists 11?

<details><summary>Answer</summary>
17 params; `cam_top_exposure_time_absolute` is `156` (so the search
`params.cam_top_exposure_time_absolute != "156"` returns 0 runs: that query finds sessions recorded with a
different exposure lock). The 6 metrics are `cams_locked`, `episodes_total` (2), `episodes_new` (2),
`duration_s` (2.0 here), `rc` (0) and `ok` (1). The other five (`cam_health_ok`, three `cam_health_fps_*`,
`cam_health_age_s`) only appear when `so101-cams-health` wrote `cam_health.json` less than 10 minutes
earlier; the stub session did not, and the fake `lerobot-record` wrote `meta/info.json` directly.
</details>

## 6. Register two fake checkpoints

**Goal.** Put a checkpoint into the Model Registry and see what a registered version is made of. The
checkpoints come from `make_ckpt` in `tests/tools/toolhelpers.py` (deterministic bytes, so the sha256 values
below are yours too). A real one is `so101-policy register ~/so101/outputs/train/<job>/checkpoints/last/pretrained_model --name so101-<task>`.

```sh
P=~/so101/outputs/practice; mkdir -p $P
~/so101/.venv/bin/python - <<'PY'
import os, sys; sys.path.insert(0, os.path.expanduser("~/so101/tests/tools")); import toolhelpers as T
for i in (1, 2): print(T.make_ckpt(os.path.expanduser(f"~/so101/outputs/practice/fake_train/act_fake{i}"), seed=i, step="005000"))
PY
so101-policy register $P/fake_train/act_fake1/checkpoints/005000/pretrained_model --name so101-fake --note "practice v1"
so101-policy register $P/fake_train/act_fake2/checkpoints/005000/pretrained_model --name so101-fake --note "practice v2"
so101-policy list
```

You should see `registered so101-fake v1  run <id>  alias candidate  sha256 ed796134b500  12 params  from ...`,
then the same for v2 with `sha256 d77158893dba`, and a list with `so101-fake  (2 versions; candidate -> v2)`.

**UI.** Models page: `so101-fake`, versions 1 and 2, alias `candidate` on 2. Click a version: its tags
(`sha256`, `checkpoint_step=005000`, `dataset`, `git_commit`, `host`, `source_dir`) and a link to the source
run. Follow it to `so101-registry`: run `so101-fake:register`, 12 params, artifacts under `model/`
(`config.json`, `model.safetensors`, `policy_preprocessor.json`, `train_config.json`).

**Question.** Which params does the register run share with the backfilled `act_lighter` training run, and
how many are they? Why does the tool bother?

<details><summary>Answer</summary>
All 12, and they are identical sets: `batch_size, chunk_size, dataset, device, dim_model, lr, n_action_steps,
policy, seed, steps, use_vae, warm_start`. `ckpt_params` in `tools/so101_policy.py` deliberately writes the
same keys as `params_for` in `tools/mlflow_backfill.py`, so a registry run and a training run compare in
one table and the Compare view's parameter diff works across them.
</details>

## 7. Deploy to the fake Jetson, roll back, check

**Goal.** The full lifecycle with no ssh: `SO101_DEPLOY_LOCAL_ROOT` makes `so101-policy` run its remote
script with `bash -s` against that directory as `$HOME` and copy with a local `rsync`. The files it leaves
are byte-for-byte what the real Jetson gets.

```sh
export SO101_DEPLOY_LOCAL_ROOT=$P/fake_jetson         # unset this before a real deploy
so101-policy deploy so101-fake@1
so101-policy deploy so101-fake@candidate
so101-policy deploy so101-fake@2                      # again: a no-op
so101-policy rollback so101-fake
so101-policy status; echo "exit $?"
cat $P/fake_jetson/so101/policies/DEPLOYED.json
ls -la $P/fake_jetson/so101/policies/so101-fake/
so101-policy list so101-fake
unset SO101_DEPLOY_LOCAL_ROOT
```

You should see, in order: `alias jetson -> v1` ... `v1 is live (deploy)`; `alias jetson -> v2 (was v1)`;
`so101-fake v2 is already deployed on local stand-in ... nothing to do`; `rollback so101-fake: v2 -> v1`
ending `v1 is live (rollback)`; a status block ending `so101-run default -> ~/so101/policies/so101-fake/current (so101-fake v1)`
and `exit 0`.

**UI.** The version with alias `jetson` moves 1, 2, 1; that version gains tags `deployed_at`, `deployed_to`,
`deployed_by`. `candidate` stays on 2 throughout (it means "newest registered", not "live").

**Question.** What is the `history` list in `DEPLOYED.json` after these steps, what is on disk under
`so101-fake/`, and what would `status` do if you moved the `jetson` alias to v2 by hand in the UI?

<details><summary>Answer</summary>
`history` has three entries with versions `[1, 2, 1]` and actions `deploy, deploy, rollback`; the top-level
fields repeat the last one (`version: 1`, `action: rollback`, `path: ~/so101/policies/so101-fake/current`).
On disk: `v1/`, `v2/` and `current -> v1` (the tool keeps the last three distinct versions; nothing is pruned
yet). Moving the alias by hand makes `status` print `registry jetson alias -> v2  MISMATCH` and exit 1
(`test_status_agrees_then_catches_a_moved_alias_and_a_vla_default`); the Jetson's `current` is the truth
the tool checks the registry against, and the fix is `so101-policy deploy so101-fake@2` or moving the alias back.
</details>

## 8. The server is down (and the HTTP path when it is up)

**Goal.** See the degrade path cost nothing, then see the same self-test go through the server over HTTP.

```sh
so101-mlflow stop
time ~/so101/.venv/bin/python ~/so101/jetson/labtrack.py http://127.0.0.1:5050; echo "exit $?"
so101-mlflow start
time ~/so101/.venv/bin/python ~/so101/jetson/labtrack.py http://127.0.0.1:5050; echo "exit $?"
time ~/so101/.venv/bin/python ~/so101/jetson/labtrack.py                       # no argument: the SQLite file directly
```

**UI.** After the second and third commands, `so101-labtrack-selftest` has two runs: one whose artifact
`hello.json` was uploaded through the server, one written straight into the folder. They look the same,
which is the point of `--default-artifact-root` being the same folder.

**Question.** How long does the self-test take with the server down, and what does it print? How long over
HTTP, and what extra lines appear?

<details><summary>Answer</summary>
Down: 62 ms wall, one stderr line `labtrack: tracking off (so101-labtrack-selftest): http://127.0.0.1:5050 not reachable within 1 s`,
a JSON line with `"enabled": false`, exit 1. The 1 s probe never fires because a closed local port refuses at
once; an asleep Mac on the LAN would cost the full second (`test_probe_timeout_path_is_bounded`). Over HTTP:
1.42 s wall, exit 0, and MLflow 3 adds two lines of its own (`View run ... at: http://127.0.0.1:5050/#/experiments/...`)
after the JSON. Straight into SQLite: 1.75 s on an existing store (most of it importing mlflow). The pick loop
pays the SQLite cost between picks, never during a move.
</details>

## 9. Read the store without the UI: REST and SQL

**Goal.** The dashboard and any future tooling will read MLflow over HTTP; know the two calls you need.

```sh
curl -s "http://127.0.0.1:5050/api/2.0/mlflow/experiments/get-by-name?experiment_name=so101-picks" | python3 -m json.tool
EXP=<the experiment_id from above>
curl -s -X POST http://127.0.0.1:5050/api/2.0/mlflow/runs/search -H "Content-Type: application/json" \
  -d "{\"experiment_ids\":[\"$EXP\"],\"filter\":\"metrics.landing_err_mm < 2 and metrics.picked = 1\",\"max_results\":20}" | python3 -m json.tool | head -40
RID=<a run_id from above>
curl -s "http://127.0.0.1:5050/api/2.0/mlflow/artifacts/list?run_id=$RID" | python3 -m json.tool
curl -s "http://127.0.0.1:5050/api/2.0/mlflow/registered-models/search" | python3 -m json.tool
sqlite3 ~/so101/mlflow/mlflow.db "select e.name, count(r.run_uuid) from experiments e left join runs r on r.experiment_id = e.experiment_id group by e.name"
```

**Question.** How many runs does the search return and what is in its `data.metrics` list? What does
`artifacts/list` show for a pick run, and what does the model search show for `so101-fake`?

<details><summary>Answer</summary>
One run (r3); `data.metrics` is a list of `{key, value, timestamp, step}` objects, 14 of them, each with
`step: 0` because the pick loggers log one value per key (a training run's `loss` would have a step per
point: `metrics/get-history?run_id=...&metric_key=loss` on `act_sim_pick_v4b` returns 76 points).
`artifacts/list` shows `frames/` and `state/` as directories and `plan.json` (3903 bytes), `waypoints.json`
(3506 bytes) as files. The model search shows `so101-fake` with `aliases: [{candidate: "2"}, {jetson: "1"}]`.
The SQL join gives the same per-experiment counts the UI sidebar does.
</details>

## 10. Clean up: soft delete, restore, garbage-collect

**Goal.** Learn that deletion is two-stage, then leave the store as you found it.

```sh
~/so101/.venv/bin/python - <<'PY'
import os; os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
c = mlflow.MlflowClient("sqlite:///" + os.path.expanduser("~/so101/mlflow/mlflow.db"))
t0 = int(os.environ["SO101_PRACTICE_T0"])
exps = [e.experiment_id for e in c.search_experiments() if e.name in ("so101-labtrack-selftest", "so101-picks", "so101-calibration", "so101-recordings", "so101-registry")]
runs = c.search_runs(exps, filter_string=f"attributes.start_time >= {t0}")
for r in runs: c.delete_run(r.info.run_id)
print("soft-deleted", len(runs), "runs; restoring one to prove it:", runs[0].info.run_name)
c.restore_run(runs[0].info.run_id); print(c.get_run(runs[0].info.run_id).info.lifecycle_stage); c.delete_run(runs[0].info.run_id)
c.delete_registered_model("so101-fake"); print("models left:", [m.name for m in c.search_registered_models()])
PY
MLFLOW_TRACKING_URI=sqlite:///$HOME/so101/mlflow/mlflow.db ~/so101/.venv/bin/mlflow gc --backend-store-uri sqlite:///$HOME/so101/mlflow/mlflow.db
rm -rf ~/so101/outputs/practice
```

**UI.** Before `gc`: the runs are gone from the tables but the state filter "Deleted" shows them, with a
Restore button. After `gc`: gone for good, their artifact folders removed, the five experiments still listed
(empty); delete `so101-labtrack-selftest` from the UI if you like, the other four will fill with real
sessions. Models page: empty again.

**Question.** How many runs did the evening make, what does `gc` print, and what is left on disk under
`~/so101/mlflow/artifacts/` afterwards?

<details><summary>Answer</summary>
9 runs (2 self-tests from exercise 8, one over HTTP and one straight into SQLite, 3 picks, 1 calibration,
1 recording, 2 register runs; the server-down call made none). `gc` prints one
`Run with ID <id> has been permanently deleted.` line per run and took 2.2 s for the 10 runs of the measured
evening (which had one extra self-test). The artifact folders for the five practice experiments are emptied
(0 files; measured on the copy), while `~/so101/mlruns/1` and the `MLflow Demo` folder are untouched. Note the invocation: in 3.17.2 `mlflow gc` insists on `MLFLOW_TRACKING_URI`
in the environment as well as `--backend-store-uri`; with only `--tracking-uri` it exits 0 having deleted nothing.
</details>

## What to do next with real data

The same ten moves on real sessions: a pick-loop afternoon (exercise 3's chart becomes a learning curve), a
calibration redo after bumping the camera stand (exercise 4's Compare shows which marker moved), a recording
session (exercise 5's `cam_*` params become the audit trail for a dataset), and the next real checkpoint
(exercises 6 and 7 with `so101-lighter` instead of `so101-fake`, and no `SO101_DEPLOY_LOCAL_ROOT`). Then the
gaps in [README.md](README.md): live training logs, rollout success as nested runs, the dashboard buttons
resolving `current`.
