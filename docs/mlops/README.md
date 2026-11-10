# MLOps in the arm lab: the map

The lab's MLOps stack is one MLflow server and the tools that feed it. Every pick, every calibration solve,
every recording session and every registered checkpoint becomes a run in that server, with its inputs as
params, its results as metrics, and the files behind the numbers (the landing frame, the calibration JSON,
the camera check, the checkpoint itself) attached as artifacts. The Model Registry on the same server holds
the policies the Jetson has run, numbered, with two aliases (`candidate`, `jetson`) that say which one is next
and which one is live, and `so101-policy` moves them to the Jetson and back. Underneath, the server runs
today as a bare process over a SQLite file on the Mac (`tools/mlflow_ui.sh`), with a container stack
(Postgres + S3-compatible store, `infra/`) and a Terraform mirror on AWS ready but not yet used. Everything
degrades to a silent no-op when the server is absent, so the arm never waits on its bookkeeping. Counts and
timings below come from the three detailed docs and from `pytest --collect-only` on 10 Nov 2026
(mlflow 3.17.2 in the Mac venv).

## The pieces

| Piece | What it does for the lab | The command you type | Where its data lives | Doc |
| --- | --- | --- | --- | --- |
| Tracking server (bare) | The MLflow UI and API on port 5050 over the Mac's SQLite file | `so101-mlflow start [--lan] \| stop \| status \| uri` (alias for `tools/mlflow_ui.sh`) | `~/so101/mlflow/mlflow.db`, artifacts in `~/so101/mlflow/artifacts/<experiment>/` (`so101-act`, backfilled before the wrapper existed, keeps its files in `~/so101/mlruns/1`); log in `/tmp/so101_mlflow.log` | [tracking.md](tracking.md), [field guide](../mlflow_field_guide.html) |
| `jetson/labtrack.py` | The one fail-safe wrapper every logger uses: resolves the URI, probes it for 1 s, never raises | `~/so101/.venv/bin/python ~/so101/jetson/labtrack.py [uri]` (self-test, logs one run) | experiment `so101-labtrack-selftest` | [tracking.md](tracking.md) |
| Pick logging | One run per pick: landing error, per-waypoint sag, the overhead frame, the plan | `~/so101/.venv/bin/python ~/so101/tools/pick_loop.py --runs 4` | experiment `so101-picks` | [tracking.md](tracking.md) |
| Calibration logging | One run per written calibration file (`table`, `campose`, `armfit`, `handeye`), residuals and the JSON | `so101-calib table --layout calib/top_layout.json` (Jetson) | experiment `so101-calibration` | [tracking.md](tracking.md) |
| Recording logging | One run per `lerobot-record` session: dataset, task, locked camera controls, health check, episode count | `so101-cams-health 10 && so101-record cube "pick the cube" 10` (Jetson) | experiment `so101-recordings`; `~/so101/outputs/run/cam_health.json` on the Jetson | [tracking.md](tracking.md) |
| Training backfill | Past LeRobot training runs (loss curve, config) imported as runs; re-runnable | `~/so101/.venv/bin/python ~/so101/tools/mlflow_backfill.py` | experiment `so101-act` (10 runs on 10 Nov 2026) | [field guide](../mlflow_field_guide.html) |
| Model registry and deployment | Checkpoint to registered version (alias `candidate`), rsync to the Jetson (alias `jetson`), sha256-verified, one-command rollback | `so101-policy register \| list \| deploy \| rollback \| status` (function in the Mac `env.sh`) | registry in the same `mlflow.db`; experiment `so101-registry`; on the Jetson `~/so101/policies/<name>/v<N>/`, `current -> v<N>`, `~/so101/policies/DEPLOYED.json` | [registry-and-deploy.md](registry-and-deploy.md) |
| Jetson runners | `so101-run`, `so101-run-async`, `so101-run-vla` default to the deployed `current` and print which version runs | `so101-run 60` (Jetson) | reads `DEPLOYED.json` and the `current` symlink | [registry-and-deploy.md](registry-and-deploy.md) |
| Dashboard policy panel | `GET /api/policy` and the `v<N>` badge in the POLICIES panel | `so101-dash` (Mac) | cached copy of `DEPLOYED.json` from the 15 s ssh probe | [registry-and-deploy.md](registry-and-deploy.md) |
| Container stack | The same server with Postgres as the backend store and an S3-compatible store (RustFS) for artifacts | `so101-mlflow up \| down \| logs [service]` (needs Docker Desktop or OrbStack; neither is installed on the Mac on 10 Nov 2026) | docker volumes `so101-mlflow_pgdata`, `so101-mlflow_s3data`; settings in `infra/.env` (gitignored) | [infra.md](infra.md) |
| Store migration | Copies the SQLite store into the container stack (or later into AWS) through the REST API, idempotently | `infra/migrate_sqlite.sh [target] [--dry-run]` | the target server | [infra.md](infra.md) |
| AWS mirror (IaC) | The container stack as RDS Postgres + S3 bucket + one ECS Fargate task, image in ECR, logs in CloudWatch | `terraform init` then `terraform apply` in `infra/terraform/` (init has been run; apply never has) | nothing yet; would be RDS, the bucket, CloudWatch `/ecs/so101-mlflow` | [infra.md](infra.md) |
| Tests | 80 hardware-free tests for the stack: `tests/mlops` 34, `tests/tools` 29, `tests/infra` 13, `tests/server/test_policy_api.py` 4 (whole repo: 702 collected) | `~/so101/.venv/bin/python -m pytest tests/mlops tests/tools tests/infra -q` | temp SQLite stores and a local directory standing in for the Jetson | each doc's Tests section |

## Day to day

Every sequence is on the Mac unless it says Jetson. "Should see" lines are the real outputs of the smoke
runs in [learn-mlflow-in-this-lab.md](learn-mlflow-in-this-lab.md); run ids, hostnames and sha256 prefixes
will differ.

### A pick-loop session

```sh
so101-mlflow status                                   # "running · http 200 at http://127.0.0.1:5050/"; else: so101-mlflow start
~/so101/.venv/bin/python ~/so101/tools/pick_loop.py --runs 4
```

You should see one JSON line per pick on stdout, exactly as before tracking existed
(`{"run": 1, "target_in": [...], "picked": true, "landed_in": [...], "err_mm": 2.8, ...}`), and nothing on
stderr. In the UI, `so101-picks` has four new runs named `pick-<session>-r1` to `-r4`, each with 9 params,
14 metrics and 4 artifacts (`frames/landing.jpg`, `plan.json`, `waypoints.json`, `state/pick_state.json`).
Chart view, y = `landing_err_mm`, x = start time, tells you whether the loop is learning. If instead the
first line on stderr is `labtrack: tracking off (so101-picks): ...`, the loop still ran; see "the server is
down" below.

### A calibration redo

```sh
ssh jetson
cd ~/so101 && source env.sh
so101-calib table --layout calib/top_layout.json     # or campose / armfit / handeye: whatever got bumped
```

The command prints and saves what it always did; the logger runs after the save. In the UI, `so101-calibration`
has one new run tagged `kind=table`, `camera=top`, with `residual_mm_<id>` per marker, `residual_mm_max`,
`reproj_rms_px` (when intrinsics were given) and the written `calib/top_table.json` and check image attached.
Tick the new run and the previous `kind=table` run and press Compare to see which marker moved. `check`,
`locate` and `handeye --check` never log: only commands that write a file do. The Jetson logs only when
`~/so101/.mlflow_uri` holds the Mac's URL (one line, written once by
`so101-mlflow uri | ssh jetson 'cat > ~/so101/.mlflow_uri'`); without it, nothing is logged and nothing waits.

### A recording session

```sh
ssh jetson
cd ~/so101 && source env.sh
so101-cams-health 10                                  # saves ~/so101/outputs/run/cam_health.json; used if < 10 min old
so101-record cube "pick the cube" 10                  # or: so101-record-more cube 5
```

Before `lerobot-record` starts you should see `cams locked: top ok; front ok; wrist ok` and then
`{"tracking": true, "run_id": "...", "experiment": "so101-recordings", "why": ""}`; after it exits,
`{"tracking": true, "run_id": "...", "episodes_total": 10, "duration_s": ..., "rc": 0}`. In the UI the run
`record-cube-<time>` carries the dataset, task, fps, every locked camera control as `cam_<role>_<control>`,
`cams_locked`, the health metrics, and at finish `episodes_total`, `episodes_new`, `duration_s`. A non-zero
exit from `lerobot-record` makes it a FAILED run. The refusal when the cameras are not locked logs nothing.

### Training, then deploying a policy

```sh
so101-train lighter-handoff-clean act_lighter2 25000  # Mac; writes ~/so101/outputs/train/act_lighter2/ as before
~/so101/.venv/bin/python ~/so101/tools/mlflow_backfill.py     # training is not live-logged yet: this imports the log + config into so101-act
so101-policy register ~/so101/outputs/train/act_lighter2/checkpoints/last/pretrained_model --name so101-lighter --note "25k steps"
#   registered so101-lighter v1  run <id>  alias candidate  sha256 <12 hex>  12 params  from <dir>
so101-policy list so101-lighter
so101-policy deploy so101-lighter@candidate --dry-run # prints the plan, touches nothing
so101-policy deploy so101-lighter@candidate
#   deploy so101-lighter v1 -> ssh jetson: rsync <dir> -> so101/policies/so101-lighter/v1/
#     current -> v1; DEPLOYED.json; keep v1
#     alias jetson -> v1
#     sha256 <12 hex> verified; so101-lighter v1 is live (deploy)
ssh jetson 'cd ~/so101 && source env.sh && so101-run 60'     # first line: policy: so101-lighter v1 (~/so101/policies/so101-lighter/current)
so101-policy rollback so101-lighter                   # worse than before: current -> the previous version, alias jetson follows
so101-policy status                                   # exit 0 when Jetson and registry agree, 1 on any MISMATCH
```

In the UI: Models page shows `so101-lighter` with the version, its two aliases, and a link back to the
`so101-lighter:register` run in `so101-registry` (12 params, the same keys the backfilled training runs
have, and the checkpoint under `model/`). Registering again makes v2 and moves `candidate`; deploying it moves
`jetson`; the dashboard's POLICIES panel shows `v2`.

### The server is down

Symptoms: one line on stderr per process, `labtrack: tracking off (<experiment>): http://<mac>:5050 not
reachable within 1 s`, then the tool runs exactly as it would without tracking (measured: the whole
`labtrack.py` self-test against a closed port takes 62 ms; a recording with the server down still starts and
finishes, < 15 s end to end in `test_dead_server_does_not_slow_or_fail_the_recording`). `so101-policy`
is the exception: it needs the store and says so (`so101-policy: ...`, exit 2).

```sh
so101-mlflow status                                   # "not running (bare)" or "running · http 000" (process alive, not answering)
so101-mlflow stop; so101-mlflow start --lan           # the Jetson needs --lan; the Mac alone does not
so101-mlflow uri                                      # must equal the one line in the Jetson's ~/so101/.mlflow_uri (an IP, not a .local name)
ssh jetson 'cd ~/so101 && source env.sh && python labtrack.py'   # {"uri": "http://<mac-ip>:5050", "enabled": true, "why": ""}
tail -20 /tmp/so101_mlflow.log                        # when start printed "http 000"
```

Runs that happened during the outage are gone: there is no offline queue, by design (the arm is never made
to wait, and a pick that was not logged is still a pick). A run in progress when the server died stays
RUNNING server-side; delete it from the UI.

## How it maps to standard MLOps practice

| The posting says | What it means in general | What it is in this lab | What is deliberately NOT done here, and why |
| --- | --- | --- | --- |
| Experiment tracking | Every training or evaluation run records its inputs, outputs and files in a queryable store | `jetson/labtrack.py` and the four loggers; experiments `so101-picks`, `so101-calibration`, `so101-recordings`, `so101-registry`, plus `so101-act` from `tools/mlflow_backfill.py`; `tests/mlops` (34) | Training itself is not live-logged: `so101-train` writes a log file and the backfill imports it afterwards. The LeRobot trainer plugin is the next step in the field guide. No cloud (HF Jobs) runs log anywhere yet. |
| Model registry | A catalogue of named models with numbered, immutable versions, each tied to the run that made it | `so101-policy register` (`tools/so101_policy.py`): one run + one version per checkpoint, sha256 and git commit as tags; `tests/tools/test_policy_registry.py` | Nothing registers automatically at the end of training; you choose the checkpoint and register it by hand. No model signature or input example is logged (LeRobot checkpoints are plain folders, not MLflow "models"). |
| Model promotion | Moving a version through stages (staging to production) by rule or review | Two aliases: `candidate` (newest registered) and `jetson` (what is live); `deploy` moves `jetson` only after the copied weights' sha256 matches | No approval gate, no evaluation threshold: `deploy` trusts the human. MLflow stages are not used (deprecated upstream; aliases replace them). |
| Deployment / rollout | Getting the chosen version onto the machine that serves it, verifiably | `so101-policy deploy name@version`: rsync to `~/so101/policies/<name>/v<N>/`, `current` symlink, `DEPLOYED.json`, remote sha256 check, idempotent re-run ("already deployed"); `tests/tools/test_policy_deploy.py` | One target, one `current`: no canary, no blue/green, no traffic split. The dashboard buttons still pass hard-coded folders rather than resolving `current`. |
| Rollback | Returning to the previous known-good version in one step | `so101-policy rollback name`: previous distinct version from the history in `DEPLOYED.json`, same verified path, history records `rollback` | Nothing rolls back automatically on a bad metric; no rollout success is logged yet to decide from. |
| Artifact store | Durable storage for the files runs produce (frames, JSON, checkpoints) | Bare server: a folder beside the SQLite file; containers: an S3-compatible bucket (`infra/docker-compose.yml`, RustFS); AWS: S3, versioned, old versions expire after 30 days | The containers have not been run on this Mac (no Docker installed, 10 Nov 2026) and AWS has not been applied: today's artifacts are plain files in `~/so101/mlflow/artifacts/` and `~/so101/mlruns/1` with no backup beyond the Mac. |
| Metadata store | The database of experiments, runs, params, metrics, tags, model versions | `~/so101/mlflow/mlflow.db` (SQLite); Postgres 16 in the stack; RDS `db.t4g.micro` on AWS with 7 daily backups | SQLite is one writer at a time and lives on one laptop. The migration path (`infra/migrate_sqlite.sh`) exists but has not been exercised against a live stack here. |
| Reproducibility | Being able to say exactly what produced a result and do it again | Params from `train_config.json` (12 keys shared by training and registry runs), `git_commit`, `sha256`, `source_dir`, `checkpoint_step` tags; recordings carry every locked camera control; calibration runs carry the layout and intrinsics paths | Datasets are referenced by Hub repo id, not pinned by hash or version. The Python environment (venv, JetPack, LeRobot 0.6.1) is described in `jetson_setup.sh`, not captured per run. |
| Data lineage | Tracing a model back to the data and process that made it, and forward to where it ran | Registry run tags `dataset`, `hf_repo`, `hf_job`; version tags `deployed_at`, `deployed_to`, `deployed_by`; recordings log `repo_id` and `episodes_total`; `DEPLOYED.json` history | The links are tags, not enforced references: nothing stops registering a checkpoint whose dataset was never recorded through `so101-record`. No rollout runs nest under the deployed version yet. |
| IaC | Infrastructure described in code and created by a tool, not by clicking | `infra/terraform/` (S3, ECR, CloudWatch, security groups, RDS, SSM secret, IAM, ECS cluster/task/service); variables mirror `infra/.env.example`; `tests/infra` checks the shape (13 tests) | `terraform apply` has never been run: there is no state file, no AWS resource and no cost. No remote state (only `backend.tf.example`), no CI plan/apply, no environments beyond one. |
| Containerisation | The server packaged as an image that runs the same way everywhere | `infra/Dockerfile.mlflow` (python slim + mlflow 3.17.2 pinned to the venv) and `entrypoint.sh`, used unchanged by compose and by the Fargate task definition | Docker is not installed on the Mac, so the image has never been built here; the compose file is validated by tests only. No image is published anywhere. |
| Observability | Knowing whether the system is healthy and what it did | Healthchecks on all three services (`healthcheck.py`, `pg_isready`, RustFS `/health`); `so101-mlflow status`; CloudWatch logs with 14-day retention in the Terraform; `so101-policy status` exits 1 on a mismatch | No metrics or alerting on the server itself, no uptime monitor, no auth logs (there is no auth: the LAN bind relies on the trusted home network, the AWS build on a /32 security-group rule). |
| Multimodal data | Images, video, joint streams and text handled in one pipeline | Each pick run attaches the overhead JPEG at landing next to the numbers; recordings attach the camera check and health JSON next to `episodes_total`; calibration runs attach the check PNG; LeRobot datasets themselves (video + joint state) live in the HF cache | The datasets (video + state) are not stored in or referenced by hash from MLflow; only their id and episode count are. No per-episode video or frame is attached to a run. |
| CI/CD of models | Pipelines that test, build, promote and deploy models automatically | Not done. GitHub Actions runs `ruff` and the test suite on every push (`.github/workflows/tests.yml`); nothing in CI touches models or infrastructure | Every register/deploy/rollback is a typed command. Also not done: a feature store (no features exist to share), any form of authentication on the MLflow server, TLS, a domain name. |
