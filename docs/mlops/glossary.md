# Glossary

Forty terms you will meet in the three MLOps docs, in job postings, and in the MLflow UI. One or two sentences
each; the lab example is in italics. MLflow terms first, then storage and cloud, then infrastructure practice,
then the lab's own words.

## MLflow

**Run.** One execution of something worth comparing: it has a start, an end, a status (RUNNING, FINISHED,
FAILED) and carries params, metrics, tags and artifacts. *One pick in `so101-picks`; one `so101-record`
session; one `so101-policy register`.*

**Experiment.** A named folder of runs, one per question you are asking, not one per day. *`so101-picks`,
`so101-calibration`, `so101-recordings`, `so101-registry`, and `so101-act` for training.*

**Param.** An input to the run: a fact about how it was configured, written once, stored as a string. *`chunk_size`
on a training run; `target_in` on a pick; `cam_top_exposure_time_absolute` on a recording.*

**Metric.** A number the run produced, optionally with an integer step so it forms a series; comparisons in the
search box use the last value. *`landing_err_mm` (one value per pick, step 0); `loss` (one value per 100
training steps).*

**Tag.** A free label you can set or change after the fact, used for filtering. *`kind=table` on a calibration
run; `host=jetson`; the run name itself is the tag `mlflow.runName`.*

**Artifact.** A file attached to a run, stored in the artifact store and previewed in the UI. *`frames/landing.jpg`,
`calib/top_table.json`, the checkpoint under `model/` on a register run.*

**Nested run.** A run started inside another, shown indented under its parent. *Planned: each rollout's
success per episode under the deployed version's run; not built yet.*

**Soft delete.** Deleting a run in the UI or the client marks it `deleted`; it stays in the database until
`mlflow gc` removes it, so Restore works. *Exercise 10 in the learning doc deletes, restores, then collects.*

**Tracking URI.** The address a client logs to: `http://host:port` for a server, `sqlite:///path` for a file.
*`labtrack.py` reads `SO101_MLFLOW_URI`, then `MLFLOW_TRACKING_URI`, then uses `sqlite:///~/so101/mlflow/mlflow.db`;
an `http` URI gets a 1 s TCP probe first.*

**Tracking server.** The process that serves the UI and the REST API and owns the two stores. *`so101-mlflow start`
on port 5050; in containers, the `mlflow` service.*

**Backend store.** The database of experiments, runs, params, metrics, tags and model versions. *`mlflow.db`
(SQLite) today; Postgres in the container stack; RDS on AWS.*

**Artifact store.** Where artifact files live, separate from the database; a bare server puts new experiments'
files under its `--default-artifact-root`. *`~/so101/mlflow/artifacts/` today; an S3-compatible bucket in the
stack; an S3 bucket on AWS.*

**Serve-artifacts (proxied artifact access).** The server mode where clients upload and download artifacts through
the server instead of touching the store themselves, so a client needs only the server URL. *`--serve-artifacts`
in `infra/entrypoint.sh`; the reason the Jetson needs no S3 keys.*

**Registered model.** A named entry in the Model Registry that collects versions of one model. *`so101-lighter`;
`so101-fake` in the exercises.*

**Model version.** A numbered, immutable entry under a registered model, pointing at the run and artifact path
that hold its files. *`so101-fake` v1, source `runs:/<run id>/model`.*

**Alias.** A named pointer to one version of a registered model that moves when you choose a different version;
it replaces MLflow's older, deprecated "stages" (None, Staging, Production, Archived), which this lab does not use.
*`candidate` (newest registered) and `jetson` (live on the Jetson); `so101-policy deploy name@candidate`.*

**Search filter.** The small SQL-like language of the search box and `search_runs`: `metrics.x < 1`,
`params.y = "100"`, `tags.z != "a"`, `attributes.start_time >= <ms>`, joined with `and`.
*`metrics.landing_err_mm < 10 and metrics.picked = 1`.*

**mlflow-skinny.** The client-only package: logging and the registry API without the server, UI or pandas.
*What `jetson_setup.sh` installs on the Jetson.*

**Backfill.** Importing results that happened before tracking existed, from logs and config files. *`tools/mlflow_backfill.py`
builds `so101-act` from `outputs/train/*.log` and each `train_config.json`.*

## Storage and cloud

**S3-compatible.** Any object store that speaks Amazon's S3 API (buckets, keys, GET/PUT), so one client works
against all of them. *RustFS in the compose stack; real S3 on AWS; the MLflow image only ever talks S3.*

**Bucket.** The top-level container of an object store; names are global on AWS, and a versioned bucket keeps
old copies of overwritten objects until a lifecycle rule expires them. *`mlflow` locally; `so101-mlflow-<suffix>`
on AWS, versioned, old versions expire after 30 days.*

**RDS.** Amazon's managed relational database: you choose the engine and size, it does backups and patches.
*Postgres 16 on `db.t4g.micro`, 20 GB, 7 daily backups, the one real recurring cost (about USD 12 a month).*

**ECR.** Amazon's container image registry, where the image is pushed so the task can pull it. *`terraform output push_image`
prints the three commands.*

**Task definition.** The ECS description of a container to run: image, CPU and memory, environment, secrets,
ports, logging, health check. *`aws_ecs_task_definition.mlflow`: 0.25 vCPU, 0.5 GB, the same variables as the
compose file.*

**Fargate.** Running an ECS task without managing the server underneath; you pay per vCPU-hour and GB-hour,
and an ECS service keeps the desired number of copies alive. *One ARM64 task, about USD 7 a month;
`desired_count = 0` pauses it.*

**Security group.** A virtual firewall: rules for which addresses and ports may reach a resource. *The server
admits `var.allowed_cidr` on the MLflow port only; the database admits only the server's group.*

**CIDR.** A block of IP addresses written `address/prefix`; `/32` is exactly one address. *The lab's public IP
as a /32 is the only lock on the AWS server.*

**SSM parameter (SecureString).** An encrypted named value in AWS Systems Manager that a task's execution role
may read at start; the separate task role is what the running container may do. *The Postgres password: in the
task as a secret, never in the task definition JSON; the task role may touch one bucket and nothing else.*

## Infrastructure practice

**IaC (infrastructure as code).** Declaring resources in files a tool applies, so the environment is reviewable,
repeatable and deletable. *`infra/terraform/*.tf`; `tests/infra` checks their shape without an AWS account.*

**State file.** Terraform's record of what it created and the ids it got back; losing it means Terraform no
longer knows what it owns. *`terraform.tfstate`, gitignored; `backend.tf.example` shows keeping it in a bucket.
No state exists yet because `apply` has not been run.*

**Idempotent.** Running the same operation again changes nothing the second time. *`so101-policy deploy x@3`
twice: "already deployed, nothing to do"; `migrate_runs.py` skips runs the target already has.*

**Compose.** A file describing several containers (each from an image), their volumes, health checks and start
order, run with one command. *`infra/docker-compose.yml`: `mlflow`, `postgres`, `s3`, `s3-init`; `so101-mlflow up`
waits for all three health checks.*

**Rollback.** Returning to the previous known-good version after a bad deploy. *`so101-policy rollback name`:
the previous distinct version in `DEPLOYED.json`'s history, copied, verified, `current` re-pointed.*

## The lab's own words

**Deploy (here).** Copy a registered version to `~/so101/policies/<name>/v<N>/` on the Jetson, point `current`
at it, record it in `DEPLOYED.json`, verify the sha256 on both ends, move the `jetson` alias. *`so101-policy deploy so101-lighter@candidate`.*

**`current` symlink and `DEPLOYED.json`.** The link the Jetson runners load by default, and the one JSON file
that says what is deployed and the history of every deploy and rollback (newest last, capped at 50).
*`so101-run 60` prints `policy: so101-lighter v3 (~/so101/policies/so101-lighter/current)`.*

**Checkpoint (`pretrained_model`).** A folder with `config.json`, `model.safetensors`, `train_config.json` and
the processors: the unit `so101-policy register` accepts and the Jetson loads. *`outputs/train/<job>/checkpoints/last/pretrained_model`.*

**Hand-eye calibration.** Solving where a camera sits relative to the arm link it is mounted on, from views of
known markers at known joint angles. *`cam_calib.py handeye`: the wrist camera's pose on `gripper_link`,
logged as `kind=handeye` with `reproj_rms_px` and `off_axis_mm`.*

**Homography.** The 3x3 map between the overhead image plane and the flat mat, from four or more marker
corners; it is how a pixel becomes a mat coordinate. *The `table` solve: `residual_mm_<id>` is how far each
marker lands from its layout position after the map.*

**Reprojection error.** How far projected 3D points land from where the camera actually saw them, in pixels;
the standard quality number for a camera solve. *`reproj_rms_px` on `table`, `campose` and `handeye` runs.*

**Sag (iterative learning).** The per-waypoint angular error the servos leave under load, measured and fed back
as a bias on the next pick. *`sag_<waypoint>` metrics, nine per pick run.*
