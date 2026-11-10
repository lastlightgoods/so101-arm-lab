#!/bin/bash
# migrate_sqlite.sh [target_url] [migrate_runs.py options]
# Copy the lab's SQLite MLflow store and the artifacts it points at into the containerised server (default http://127.0.0.1:5050,
# or http://127.0.0.1:$MLFLOW_PORT). The live file ~/so101/mlflow/mlflow.db is never opened: it is copied to a temp folder first.
#   1 checks: venv, source file, a healthy target server      2 copy the .db      3 `mlflow db upgrade` on the COPY
#   4 infra/migrate_runs.py reads the copy through sqlite:/// and writes the target through its REST API (artifacts included)
# Safe to re-run: runs already on the target (tag source_run_id) are skipped. Add --dry-run to see the plan.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); VENV=$HOME/so101/.venv/bin; SRC_DB=${SO101_MLFLOW_DB:-$HOME/so101/mlflow/mlflow.db}
TARGET=${1:-http://127.0.0.1:${MLFLOW_PORT:-5050}}; export MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false
[ -x "$VENV/python" ] || { echo "venv not found at $VENV"; exit 1; }
[ -f "$SRC_DB" ] || { echo "source store not found: $SRC_DB (set SO101_MLFLOW_DB to point elsewhere)"; exit 1; }
curl -fsS --max-time 5 "$TARGET/health" >/dev/null || { echo "no MLflow server answering at $TARGET: run tools/mlflow_ui.sh up first"; exit 1; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/so101-mlflow-migrate.XXXXXX")
cp "$SRC_DB" "$WORK/mlflow.db"   # why: alembic (db upgrade) and the reader must never touch the file the bare server may be using
echo "copied $SRC_DB -> $WORK/mlflow.db ($(du -h "$WORK/mlflow.db" | cut -f1))"
# why: a store written by an older mlflow cannot be opened by a newer client; the upgrade is a no-op when the schema already matches
"$VENV/mlflow" db upgrade "sqlite:///$WORK/mlflow.db"
"$VENV/python" "$HERE/migrate_runs.py" --source "sqlite:///$WORK/mlflow.db" --target "$TARGET" "${@:2}"
echo "finished; the working copy $WORK can be deleted"
