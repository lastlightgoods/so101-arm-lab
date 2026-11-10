#!/bin/sh
# entrypoint.sh: the one `mlflow server` line, assembled from environment variables so that compose (Postgres + RustFS) and
# ECS Fargate (RDS + S3) run the same image with nothing but settings changed.
# why: the backend URI carries the password, so it is built here at start instead of being written into a compose file or a task
# definition (where `docker inspect` / the ECS console would show it).
set -eu
: "${POSTGRES_HOST:=postgres}" "${POSTGRES_PORT:=5432}" "${POSTGRES_DB:=mlflow}" "${POSTGRES_USER:=mlflow}"
: "${MLFLOW_ARTIFACT_BUCKET:=mlflow}" "${MLFLOW_PORT:=5050}" "${MLFLOW_WORKERS:=2}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}"
# why: MLflow 3 rejects Host headers outside localhost and private ranges unless --allowed-hosts says otherwise; on Fargate the
# lab reaches the task by its public IP, which changes on every restart, so the task definition sets MLFLOW_ALLOWED_HOSTS="*"
# and relies on the security group CIDR as the lock. Unset locally, where the default (localhost + LAN addresses) is right.
# why: a password containing @ / or # would corrupt the URI; percent-encode it
PW=$(python -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["POSTGRES_PASSWORD"], safe=""))')
exec mlflow server \
  --backend-store-uri "postgresql://${POSTGRES_USER}:${PW}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}" \
  --artifacts-destination "s3://${MLFLOW_ARTIFACT_BUCKET}/" --serve-artifacts \
  --host 0.0.0.0 --port "${MLFLOW_PORT}" --workers "${MLFLOW_WORKERS}" \
  ${MLFLOW_ALLOWED_HOSTS:+--allowed-hosts "${MLFLOW_ALLOWED_HOSTS}"}
