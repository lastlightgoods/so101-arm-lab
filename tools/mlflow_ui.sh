#!/bin/bash
# mlflow_ui.sh start|stop|status · the lab's MLflow tracking server (SQLite store + local artifacts) on 127.0.0.1:5050
DB=$HOME/so101/mlflow/mlflow.db; ART=$HOME/so101/mlflow/artifacts; HOST=${MLFLOW_HOST:-127.0.0.1}; PORT=${MLFLOW_PORT:-5050}; LOG=/tmp/so101_mlflow.log
case "${1:-status}" in
  start) pgrep -f "mlflow server" >/dev/null && { echo "already running on $HOST:$PORT"; exit 0; }
         mkdir -p "$ART"; nohup $HOME/so101/.venv/bin/mlflow server --backend-store-uri "sqlite:///$DB" --default-artifact-root "$ART" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 &
         sleep 3; curl -s -o /dev/null -w "mlflow http %{http_code} at http://$HOST:$PORT/ (log $LOG)\n" "http://$HOST:$PORT/";;
  stop)  pkill -f "mlflow server" && echo stopped || echo "not running";;
  status) pgrep -f "mlflow server" >/dev/null && curl -s -o /dev/null -w "running · http %{http_code} at http://$HOST:$PORT/\n" "http://$HOST:$PORT/" || echo "not running";;
  *) echo "usage: $0 start|stop|status";;
esac
