# MLflow infrastructure: containers today, AWS when wanted

The lab's MLflow server has run as a bare process on the Mac (`tools/mlflow_ui.sh start`: SQLite file plus an artifacts
folder under `~/so101/mlflow/`, port 5050). `infra/` is the same server with the two stores that scale and move: Postgres
for the experiments, runs, params, metrics and tags; an S3 bucket for the artifacts. Locally the bucket is a container; on
AWS it is S3. Same image, same `entrypoint.sh`, same variable names. Nothing in here costs anything until `terraform apply`,
which nothing runs for you.

## The pieces

| Container | What it is | Local image | On AWS |
| --- | --- | --- | --- |
| `mlflow` | The tracking server: `mlflow server --backend-store-uri postgresql://... --artifacts-destination s3://<bucket>/ --serve-artifacts --host 0.0.0.0 --port 5050`, built from `infra/Dockerfile.mlflow` (python slim + mlflow 3.17.2, the venv's version, + psycopg2-binary + boto3) | `so101-mlflow:3.17.2` | one ECS Fargate task (0.25 vCPU, 0.5 GB) pulling the same image from ECR |
| `postgres` | The backend store | `postgres:16-alpine`, volume `pgdata` | RDS Postgres 16, `db.t4g.micro`, 20 GB |
| `s3` | The artifact store, S3 API on 9000, browser console on 9001 | `rustfs/rustfs:1.0.4`, volume `s3data` | an S3 bucket (versioned, private, old versions expire after 30 days) |
| `s3-init` | One-shot: creates the bucket on the local store, then exits | the `mlflow` image running `s3_init.py` | nothing; Terraform makes the bucket |

`--serve-artifacts` matters: clients upload and download artifacts through the server, so the Jetson, a cloud job or the
dashboard needs only the server's URL and the plain `mlflow` package. No S3 keys, no boto3, no bucket name anywhere but on
the server.
One wrinkle, found on the first bring-up: an mlflow 3.17 client asks the server whether the store does presigned URLs and,
since S3 does, would move every artifact through a URL signed for `http://s3:9000`, which only the containers resolve. Lab
clients set `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` and `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false` (`labtrack.py`,
`so101_policy.py`, both `env.sh`); anything else talking to the local stack needs the same two lines. Details and
measurements in [`colima.md`](colima.md).

Why not MinIO: MinIO archived its repository in April 2026 and removed its Docker Hub images in September 2026, so
`minio/minio` no longer pulls. RustFS has the same shape (S3 on 9000, console on 9001, one data volume, two key variables)
and is maintained. The MLflow container only ever speaks S3, so the choice of local store never reaches the clients or the
AWS build. The community fork `pgsty/silo` is a one-line swap if the MinIO console is missed.

## One-command start

```
cp infra/.env.example infra/.env     # then change the two change-me passwords
tools/mlflow_ui.sh up                # builds the image, starts the four containers, waits for health, prints the URL
tools/mlflow_ui.sh logs [service]    # follow logs
tools/mlflow_ui.sh down              # stop; the data stays in the docker volumes
```

The UI is at `http://127.0.0.1:5050` as before, so the field guide and `MLFLOW_TRACKING_URI=http://127.0.0.1:5050` stay
true. The bare server and the stack both want port 5050: `tools/mlflow_ui.sh stop` first, or set `MLFLOW_PORT` in `.env`.
A Docker engine must be running: on this Mac it is Colima, installed and first brought up on 10 Nov 2026 ([`colima.md`](colima.md):
install, VM size, start and stop, the measured bring-up, what broke and the fixes); Docker Desktop or OrbStack work the same
way, and `up` says so when `docker` is missing. One difference from the bare server: a compose
port mapping `"5050:5050"` publishes on every interface, the same as `start --lan`, so the no-auth caveat in `tracking.md`
applies to the stack as soon as it is up; write `"127.0.0.1:${MLFLOW_PORT}:5050"` in the compose file to keep it Mac-only.

## Where the data lives

| | Bare server (today) | Containers | AWS |
| --- | --- | --- | --- |
| Runs, metrics, params | `~/so101/mlflow/mlflow.db` | docker volume `so101-mlflow_pgdata` | RDS, 7 daily automated backups |
| Artifacts | `~/so101/mlflow/artifacts/` and `~/so101/mlruns/` | docker volume `so101-mlflow_s3data`, bucket `mlflow` | the S3 bucket |
| Settings | the command line in `mlflow_ui.sh` | `infra/.env` (gitignored) | Terraform variables; the password in SSM |
| Logs | `/tmp/so101_mlflow.log` | `tools/mlflow_ui.sh logs` | CloudWatch `/ecs/so101-mlflow`, 14 days |

## Moving the existing runs in

```
tools/mlflow_ui.sh up
infra/migrate_sqlite.sh                 # or: infra/migrate_sqlite.sh http://127.0.0.1:5050 --dry-run
```

The script copies `~/so101/mlflow/mlflow.db` to a temp folder (the live file is never opened), runs `mlflow db upgrade` on
the copy in case it predates the venv's MLflow, then `infra/migrate_runs.py` reads the copy through `sqlite:///` and writes
every experiment, run, full metric history and artifact tree to the target through the REST API. Run ids change (a client
cannot choose them), so each copied run carries `source_run_id`; re-running skips what is already there. Deleted runs,
traces and the MLflow demo's prompt registry are not copied. The same script moves the containers' data to AWS later:
point `--source` at the local server and `--target` at the AWS one.

## Backing up

- Containers: `docker compose -p so101-mlflow -f infra/docker-compose.yml exec postgres pg_dump -U mlflow mlflow > mlflow.sql`
  for the store; the artifacts are plain objects, copy them with any S3 client against `http://127.0.0.1:9000` or copy the
  `s3data` volume. `tools/mlflow_ui.sh down` keeps both volumes; only `docker compose ... down -v` deletes them.
- AWS: RDS keeps 7 days of automated backups; the bucket is versioned, so an overwritten or deleted artifact is recoverable
  for 30 days. `terraform destroy` deletes everything including the bucket contents (`force_destroy`), which is the point:
  nothing lingers to be billed.
- Either way, `migrate_runs.py` is also the export: aim it at a fresh local SQLite target and you have a copy on disk.

## How the AWS build maps 1:1

`infra/terraform/` uses the compose variable names in lower case (`MLFLOW_PORT` is `var.mlflow_port`, `POSTGRES_DB` is
`var.postgres_db`, ...). The task definition sets the same environment the compose file sets, except that there is no
`MLFLOW_S3_ENDPOINT_URL` and no access keys: boto3 finds real S3 by region and gets credentials from the task role, which
is allowed to read and write that one bucket and nothing else. The password goes in as an ECS secret from an SSM
SecureString, so it is never in the task definition or in state output.

Networking is the account's default VPC. The task gets a public IP; a security group admits `var.allowed_cidr` (the lab's
public IP as a /32) on the MLflow port and nothing else; RDS admits only the task. MLflow itself has no login, so that CIDR
is the lock. The public IP changes when the task restarts; `terraform output server_ip_command` prints how to find it.

To bring it up, once: `terraform init`, `terraform apply` (asks for the password, the bucket name and the CIDR), then the
commands from `terraform output push_image` to build the image on the Mac (ARM64 by default, which Apple silicon builds
natively and which Fargate charges 20 percent less for) and push it. `terraform apply -var desired_count=0` pauses the
server and its charge while keeping the data; `terraform destroy` removes it all. `backend.tf.example` shows how to keep
the state in a bucket instead of on one laptop.

## What it would cost

Prices looked up 10 Nov 2026 for US West (Oregon), on-demand, 730-hour month. AWS's own pages render the tables with
scripts, so the RDS rate comes from three price mirrors that agree ([sparecores](https://sparecores.com/database/aws/db.t4g.micro),
[holori](https://calculator.holori.com/aws/rds), [aiven](https://aiven.io/tools/instances/db.t4g.micro)); the Fargate
x86 rate from [aws.amazon.com/fargate/pricing](https://aws.amazon.com/fargate/pricing/) (its example is us-east-1, which
historically matches Oregon) and the ARM rate from [vantage](https://www.vantage.sh/blog/fargate-pricing); the rest from
memory of the public price lists. Confirm in the AWS Pricing Calculator before trusting the total to the dollar.

| Item | Rate | Per month |
| --- | --- | --- |
| RDS `db.t4g.micro`, single-AZ (the one real recurring cost) | USD 0.016/h | 11.68 |
| RDS storage, 20 GB gp3 | USD 0.115/GB-month | 2.30 |
| Fargate 0.25 vCPU, ARM | USD 0.03238/vCPU-h | 5.91 |
| Fargate 0.5 GB, ARM | USD 0.00356/GB-h | 1.30 |
| Public IPv4 address for the task | USD 0.005/h | 3.65 |
| S3, about 1 GB of artifacts | USD 0.023/GB-month | 0.02 |
| ECR, one image of about 1 GB | USD 0.10/GB-month | 0.10 |
| CloudWatch Logs, under 0.1 GB | USD 0.50/GB ingested | 0.05 |
| **Running** | | **about 25** |
| Paused (`desired_count=0`: RDS and storage only) | | about 14 |
| Destroyed | | 0 |

On x86 Fargate the two compute lines are 7.39 and 1.62 instead (USD 0.04048 and 0.004445 per hour). An account in its
first year may have RDS free-tier credit that covers the micro instance; that is a bonus, not the plan.

## Deliberately not built

- No load balancer, no TLS, no domain name. An ALB alone is about USD 16 a month, more than the server. The lab reaches the
  task by IP from one CIDR; if the Jetson or a cloud job needs in from elsewhere, the Cloudflare Tunnel the field guide
  already plans for is the cheaper door.
- No NAT gateway (USD 32 a month and up); the task has a public IP instead.
- No Multi-AZ, no read replica, no autoscaling: one person, one writer.
- No Kubernetes, no Helm, no service mesh.
- No MLflow authentication. The `mlflow` basic-auth app exists; it can be turned on later by adding `--app-name basic-auth`
  to `entrypoint.sh` without changing anything else here.
- No CI that builds or pushes the image; `terraform output push_image` is three commands run by hand when the version pin
  changes.
- No automatic migration from local containers to AWS: `migrate_runs.py` does it when asked.
