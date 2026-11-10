# Colima: the Docker runtime for the MLflow stack on the Mac

How the container stack in [`infra.md`](infra.md) runs on this Mac (M1 Max, 10 cores, 32 GB) without Docker Desktop, what it cost
to bring up the first time, and the handful of commands that matter. Everything below was measured on 10 Nov 2026, the day it was
first installed; the method is named next to each number.

## What is installed

```
brew install colima docker docker-compose        # 14 s from bottles; pulls lima as a dependency
```

| Piece | Version | Role |
| --- | --- | --- |
| `colima` | 0.10.3 (lima 2.3.0) | runs a small Linux VM with the Docker engine inside, and puts its socket where the `docker` CLI looks |
| `docker` | CLI 29.9.3 | the client; talks to `~/.colima/default/docker.sock` through the `colima` context it creates |
| `docker-compose` | 5.7.0 | the `docker compose` plugin that `tools/mlflow_ui.sh up|down|logs` drives |
| Docker engine in the VM | 29.5.2 on Ubuntu 24.04, aarch64 | what actually builds and runs the four containers |

Homebrew's `docker-compose` is a plugin, not a `docker-compose` binary on `PATH` in the old sense. The `docker` CLI finds it only
after this file exists (Homebrew prints the same snippet in its caveats):

```json
// ~/.docker/config.json
{ "cliPluginsExtraDirs": [ "/opt/homebrew/lib/docker/cli-plugins" ] }
```

`docker compose version` printing `v5.7.0` is the check. `docker buildx` is not installed, so compose builds with the classic
builder and says so in a warning; the build works and nothing here needs BuildKit (`brew install docker-buildx` and the same
config line add it if ever wanted).

## The VM

```
colima start --cpu 4 --memory 8 --disk 60 --vm-type vz --mount-type virtiofs
```

Apple's Virtualization.framework (`vz`) and `virtiofs` were accepted on macOS 26.6; no fallback was needed. `colima list` shows
`default  Running  aarch64  4 CPUs  8GiB  60GiB  docker`. Inside, `/var/lib/docker` is a 59 GB volume on its own disk and the
root filesystem is 19 GB. The four containers plus the image use about 1.3 GB of it. The settings are remembered: a plain
`colima start` later reuses them.

Measured: 42 s from typing `colima start` to `docker info` answering, on the first start (the VM image was downloaded during the
install). `colima stop` takes a few seconds; `colima status` says whether it is running.

The VM does not start at login. The one command that would make it, which nobody has run and which is the owner's call:

```
brew services start colima        # start the VM at login (brew services stop colima undoes it)
```

Without it, after a reboot the stack needs `colima start` and then `tools/mlflow_ui.sh up` (the containers are `restart:
unless-stopped`, so once the engine is up they come back by themselves; `up` is then a no-op that reports the health).

## Day to day

```
colima start                      # after a reboot (42 s first time, less after)
tools/mlflow_ui.sh up             # build if needed, start, wait for health, print the URL   (12 to 15 s when the image exists)
tools/mlflow_ui.sh status         # bare server state + the compose table
tools/mlflow_ui.sh logs [service] # mlflow | postgres | s3 | s3-init
tools/mlflow_ui.sh down           # stop the containers, keep the data (1 s)
colima stop                       # stop the VM too
```

The UI is `http://127.0.0.1:5050`, the RustFS console `http://127.0.0.1:9001` (login: `S3_ACCESS_KEY` / `S3_SECRET_KEY` from
`infra/.env`). The two passwords in `infra/.env` were set with `openssl rand -hex 12` (24 characters) before the first `up`,
which matters: Postgres bakes `POSTGRES_PASSWORD` into its volume on first start, so changing it later means wiping `pgdata`.
`infra/.env` is gitignored (`git check-ignore infra/.env` prints it).

## What it measured the first time

| Step | Measured | How |
| --- | --- | --- |
| `brew install colima docker docker-compose` | 14 s | `date` before and after, bottles already built for arm64 |
| `colima start` to `docker info` answering | 42.4 s | wall clock around the command plus a poll loop on `docker info` |
| `docker compose build` of `so101-mlflow:3.17.2` | 86.6 s | wall clock; classic builder, `pip install mlflow psycopg2-binary boto3` is most of it |
| image size | 1.38 GB (`python:3.12-slim` base 205 MB) | `docker images` |
| `tools/mlflow_ui.sh up` to `curl /health` = `OK`, image built, Postgres and S3 already running | 14.8 s | wall clock around `up` plus a 0.5 s poll on `/health` |
| the same from cold (`down`, then `up`) | 11.9 s | same method, second start |
| `tools/mlflow_ui.sh down` | 1.1 s | wall clock |
| `infra/migrate_sqlite.sh` (11 runs, 108 params, 1044 metric points, 17 artifact files) | 7.1 s | wall clock; a second run skipped all 11 in 3.9 s |
| `jetson/labtrack.py` self-test against the server | 1.5 s | wall clock |
| `cam_calib.py table` on the synthetic scene, logging to the server | 1.8 s | wall clock around the subprocess |

The migration was checked both ways: `sqlite3` on the source (`select count(*)` on `runs`, `params`, `metrics`, `find` on
`~/so101/mlruns`) and the MLflow client on the target (`search_runs`, `get_metric_history` per key, `list_artifacts` per run).
Both say 11 runs, 108 params, 1044 metric points, 17 files. The bucket (`boto3` from inside the `mlflow` container,
`list_objects_v2`) holds those 17 objects under `2/` plus whatever has been logged since (`3/` the self-test, `4/` calibration).
The demo experiment's one run is `RUNNING` on both sides (it was never closed); the three `mlflow-demo.prompts.*` registry
entries have no run behind them and were skipped, as `infra/migrate_runs.py` documents.

## What broke the first time, and the fixes

1. **The `mlflow` container crash-looped: `No module named 'psycopg'`.** MLflow 3.17 pulls SQLAlchemy 2.1, which maps a bare
   `postgresql://` URI to the psycopg 3 driver; the image ships `psycopg2-binary`. `infra/entrypoint.sh` now says
   `postgresql+psycopg2://`. The same image goes to Fargate, so AWS gets the fix for free.
   (`tests/infra/test_compose.py::test_entrypoint_names_the_postgres_driver_the_image_installs`)
2. **Downloading an artifact through the server hung for minutes.** A 3.17 client asks `/api/3.0/mlflow/server-info`, learns the
   store behind `--serve-artifacts` is S3, and from then on fetches every artifact from a presigned URL the server signs for
   `http://s3:9000`, a name only the containers resolve; the client retries for about a minute per file and then fails. Uploads
   over 500 MB would take the same road. Measured: 64 s to fail by default, 1.4 s with
   `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`. `jetson/labtrack.py` and `tools/so101_policy.py` now default
   `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` and `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false` before importing mlflow,
   `infra/migrate_sqlite.sh` exports them (a later local-to-AWS migration reads from this server), and both `env.sh` files export
   them for anything run by hand. Any other client of the local stack (a notebook, `mlflow artifacts download`) needs the same
   two variables; the UI's own artifact viewer is unaffected (it uses `/get-artifact`, served by the server). On AWS the presigned
   URLs point at real S3 and would work from anywhere the bucket is reachable, so this is a local-stack wrinkle, not a design flaw.
   (`tests/infra/test_compose.py::test_clients_of_the_proxied_s3_store_turn_presigned_transfers_off`,
   `tests/mlops/test_labtrack.py::test_module_sets_the_client_defaults_a_proxied_s3_server_needs`,
   `tests/tools/test_policy_registry.py::test_tool_turns_presigned_transfers_and_url_printing_off_before_mlflow_loads`)
3. **Two `View run ... at: http://...` lines appeared on stdout** of `cam_calib.py` and the self-test. mlflow prints them when a
   run ends against an http server (never against sqlite, which is why no test had seen it); callers parse that stdout as JSON.
   `labtrack.py` (and `so101_policy.py`, `migrate_sqlite.sh`) now set `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=1`.
   (`tests/mlops/test_labtrack.py::test_selftest_against_a_real_server_keeps_stdout_json_and_artifacts_fetchable`, which starts a
   real `mlflow server --serve-artifacts` on a free port, the client-facing shape of the stack, no Docker needed)

## Wipe and rebuild

```
tools/mlflow_ui.sh down                                                                     # containers off, data kept
docker compose -p so101-mlflow --env-file infra/.env -f infra/docker-compose.yml down -v    # ALSO deletes pgdata, s3data, s3logs
docker rmi so101-mlflow:3.17.2                                                              # force the image to rebuild (optional)
tools/mlflow_ui.sh up                                                                       # rebuild and start empty
infra/migrate_sqlite.sh                                                                     # refill from ~/so101/mlflow/mlflow.db
```

`down -v` is the only command here that loses data; `pg_dump` first if the runs since the migration matter (see "Backing up" in
`infra.md`). To throw away the VM as well: `colima delete` (asks), then the `colima start` line above; the images and volumes go
with it, the files under `infra/` and `~/so101/mlflow/` do not.

## Pointing the Jetson at it

Unchanged from [`tracking.md`](tracking.md): the stack already binds `0.0.0.0:5050` inside the VM and Colima publishes it on the
Mac's interfaces, so the Jetson reaches it at the Mac's LAN address without `--lan`:

```sh
~/so101/tools/mlflow_ui.sh uri | ssh jetson 'cat > ~/so101/.mlflow_uri'   # the Mac's LAN address, one line, on the Jetson
ssh jetson 'cd ~/so101 && source env.sh && python labtrack.py'            # {"uri": ..., "enabled": true} and one run
```

The Jetson's `env.sh` exports the two presigned switches too, so copying the current `jetson/` folder over first is part of the step.
