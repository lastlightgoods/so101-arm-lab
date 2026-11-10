# Résumé notes: what this work supports, and what it does not yet

One page. Each claim is paired with the file or command that backs it, so you can answer the follow-up
question in an interview from memory. Numbers are measured; the source is named each time. Target wording:
a "Senior Software Engineer, AI/ML Platform" posting (data collection platform, MLOps, cloud-native, containers,
IaC, multimodal data).

## You can truthfully claim

- **Designed and built an MLflow-based experiment-tracking layer for a robot-learning lab, logging every pick,
  calibration solve and data-collection session as a structured run (params, metrics, images, JSON) from two
  machines to one server.** Backed by `jetson/labtrack.py`, `tools/pick_loop.py`, `jetson/cam_calib.py`,
  `jetson/record_track.py`, experiments `so101-picks`, `so101-calibration`, `so101-recordings`
  ([tracking.md](tracking.md)).
- **Made tracking fail-safe for a real-time system: a 1 s reachability probe, bounded HTTP retries, every call
  wrapped, one stderr line then silent no-op; the robot never waits on its bookkeeping.** Backed by
  `jetson/labtrack.py` and `tests/mlops/test_labtrack.py`; measured 62 ms for the whole self-test against a
  closed port, stdout byte-identical with tracking on, off and failing
  (`test_output_lines_identical_with_tracking_off_and_dead_server`), a recording with the server down < 15 s
  end to end (`test_dead_server_does_not_slow_or_fail_the_recording`).
- **Built a model registry and versioned deployment path: checkpoint to numbered registry version with
  sha256, git commit and dataset lineage, alias-based promotion (`candidate` to `jetson`), rsync deployment
  with end-to-end checksum verification, idempotent re-deploys, three-version retention and one-command
  rollback with an audit history.** Backed by `tools/so101_policy.py`, `jetson/env.sh`
  (`so101-policy-current`), `DEPLOYED.json` ([registry-and-deploy.md](registry-and-deploy.md)).
- **Exposed deployment state to the operator UI** (`GET /api/policy`, version badge in the dashboard).
  Backed by `dashboard/server.py`, `dashboard/index.html`, `tests/server/test_policy_api.py` (4 tests).
- **Containerised the tracking server with Postgres and an S3-compatible artifact store, one image and one
  entrypoint shared by local compose and cloud, with health checks and secrets kept out of the compose file
  and task definition.** Backed by `infra/Dockerfile.mlflow`, `infra/entrypoint.sh`, `infra/docker-compose.yml`
  ([infra.md](infra.md)).
- **Wrote the AWS equivalent as Terraform: S3 (versioned, encrypted, lifecycle), ECR, CloudWatch, RDS Postgres,
  least-privilege IAM (task role scoped to one bucket), SSM SecureString for the password, security groups
  admitting one CIDR, ECS Fargate on ARM64; priced it (about USD 25 a month running, 14 paused, 0 destroyed,
  rates dated 10 Nov 2026 in `infra.md`).** Backed by `infra/terraform/*.tf`.
- **Wrote a store-migration tool that moves experiments, runs, metric histories, artifacts and registered
  models between MLflow backends through the public API, idempotently.** Backed by `infra/migrate_runs.py`,
  `infra/migrate_sqlite.sh`.
- **Covered the stack with 80 hardware-free, network-free tests (temp SQLite stores, a closed localhost port
  for the dead server, a local directory standing in for the Jetson, a sourced shell environment with stub
  binaries), inside a 702-test repository suite run by GitHub Actions on every push.** Counts from
  `pytest --collect-only -q` on 10 Nov 2026: `tests/mlops` 34, `tests/tools` 29, `tests/infra` 13,
  `tests/server/test_policy_api.py` 4; runtimes from the docs: 68 s and about 30 s for the first two.
- **Handled multimodal data in the tracking design: camera frames, calibration images and JSON, camera control
  state and health reports attached to the runs whose numbers they explain.** Backed by the artifact lists in
  [tracking.md](tracking.md) (`frames/landing.jpg`, `calib/top_table_check.png`, `cameras/cam_health.json`).
- **Documented it as runbooks with measured numbers** ([README.md](README.md), the three docs, the exercises).

Suggested résumé lines, in posting language:

- Built the MLOps layer for a robot-learning lab: MLflow experiment tracking from edge (Jetson) and workstation
  into one server, a model registry with alias-based promotion, checksum-verified deployment and one-command
  rollback; 80 tests, no hardware required.
- Containerised the tracking stack (MLflow, Postgres, S3-compatible store) and wrote its AWS mirror in Terraform
  (ECS Fargate, RDS, S3, ECR, IAM, SSM), costed and documented; not yet deployed.
- Designed fail-safe telemetry for a real-time control loop: bounded probes and retries, zero behavioural change
  when the server is absent (measured, tested).

## Do not claim yet

- **"Deployed to AWS" or "cloud-native in production".** `terraform init` has been run (lock file present,
  untracked); `terraform apply` has not. No AWS resource exists, no state file, no cost incurred. Say
  "written and tested, not applied".
- **"Ran the containerised stack".** Docker Desktop or OrbStack is not installed on the Mac (10 Nov 2026), so
  the image has never been built and `so101-mlflow up` has never run here. The compose file and Dockerfile
  are validated by `tests/infra` (13 tests), not by a running stack. Say "containerised; validated by tests".
- **"Production" anything.** One user, one arm, a home LAN, a SQLite file on a laptop, no backups beyond the
  laptop, no auth on the server. Say "lab" or "single-operator".
- **"CI/CD for models" or "automated promotion".** CI runs lint and tests only. Registering, deploying and
  rolling back are typed commands with a human deciding. Say "manual promotion with an audit trail".
- **"Live training tracking".** Training runs enter MLflow through `tools/mlflow_backfill.py` after the fact
  (10 runs in `so101-act`); the trainer does not log as it goes. Say "backfilled training history".
- **"Rollout evaluation tracked".** Success per episode under the deployed version is the next step in the field
  guide, not built. The only rollout numbers in the store are three hand-entered sim metrics on `act_sim_pick_v4b`.
- **"Feature store", "data versioning", "authentication".** None exist. Datasets are referenced by Hub id, not
  pinned by hash; the server has no login (LAN trust today, a /32 security group on AWS).
- **Any run count for the four new experiments.** On 10 Nov 2026 the real store held `so101-act` (10 runs),
  `Default` and `MLflow Demo` only; `so101-picks`, `so101-calibration`, `so101-recordings` and `so101-registry`
  are created on first real use. The smoke numbers in the docs (3 picks, 1 calibration, 1 recording, 2
  registrations) come from fakes.

## The three questions an interviewer will ask, with the honest answer

1. *How do you know it does not slow the robot?* The probe is 1 s and the test asserts < 3 s for every failure
   path; retries are capped at one with a 5 s timeout; the pick loop opens its run before planning and writes
   after the landing is measured, never during a move. `tests/mlops/test_labtrack.py`.
2. *What happens if the deploy half-fails?* The alias moves only after the remote sha256 matches the local
   one; a mismatch exits 2 with the alias untouched (`test_deploy_verifies_the_remote_sha`), and
   `so101-policy status` exits 1 whenever the Jetson and the registry disagree.
3. *Why SQLite and a bare process instead of the containers?* Because one writer on one laptop did not need
   more, and the containers and Terraform exist for the day it does; the migration tool is the bridge and
   has a dry run. The honest gap: that migration has not been exercised against a live stack on this machine.
