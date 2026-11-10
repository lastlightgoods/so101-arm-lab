#!/bin/bash
# mlflow_ui.sh · the lab's MLflow tracking server on port 5050, two ways:
#   start [--lan] | stop | status | uri   bare process from the venv (SQLite store + local artifacts folder)
#   up | down | logs [service]            the container stack in infra/docker-compose.yml (Postgres + S3-compatible artifacts), same port; docs/mlops/infra.md
#   start        binds 127.0.0.1: the Mac only
#   start --lan  binds 0.0.0.0 so the Jetson (and the iPad) can log to it and open the UI. The server has no auth: this is fine
#                here because the lab LAN is a trusted home network behind the router and the Mac firewall; do not do this on a
#                network you share with strangers (bind local and use an ssh tunnel instead).
#   uri          prints the URL the Jetson should log to (the Mac's LAN address), for ~/so101/.mlflow_uri. See docs/mlops/tracking.md.
DB=$HOME/so101/mlflow/mlflow.db; ART=$HOME/so101/mlflow/artifacts; HOST=${MLFLOW_HOST:-127.0.0.1}; PORT=${MLFLOW_PORT:-5050}; LOG=/tmp/so101_mlflow.log
[ "$2" = "--lan" ] && HOST=0.0.0.0
lan_ip() { for i in en0 en1 en2; do ipconfig getifaddr $i 2>/dev/null && return 0; done; hostname -I 2>/dev/null | awk '{print $1}'; }
INFRA=$(cd "$(dirname "$0")/../infra" && pwd); COMPOSE="docker compose --project-name so101-mlflow --env-file $INFRA/.env -f $INFRA/docker-compose.yml"
stack() {   # why: the two ways the stack commands fail (no Docker, no .env) should each read as one plain line, not a compose stack trace
  command -v docker >/dev/null || { echo "docker not found: install Docker Desktop or OrbStack, then retry"; exit 1; }
  [ -f "$INFRA/.env" ] || { cp "$INFRA/.env.example" "$INFRA/.env"; echo "created infra/.env from .env.example: change the two change-me passwords before trusting it"; }
  SPORT=${MLFLOW_PORT:-$(grep -E '^MLFLOW_PORT=' "$INFRA/.env" | cut -d= -f2)}   # the shell overrides .env, as compose itself does
}
case "${1:-status}" in
  start) pgrep -f "mlflow server" >/dev/null && { echo "already running (stop it first to change the bind address)"; exit 0; }
         mkdir -p "$ART"; nohup $HOME/so101/.venv/bin/mlflow server --backend-store-uri "sqlite:///$DB" --default-artifact-root "$ART" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 &
         sleep 3; curl -s -o /dev/null -w "mlflow http %{http_code} at http://$HOST:$PORT/ (log $LOG)\n" "http://127.0.0.1:$PORT/"
         [ "$HOST" = "0.0.0.0" ] && echo "LAN: http://$(lan_ip):$PORT/  (trusted home LAN; no auth)";;
  stop)  pkill -f "mlflow server" && echo stopped || echo "not running";;
  status) pgrep -f "mlflow server" >/dev/null && curl -s -o /dev/null -w "running · http %{http_code} at http://127.0.0.1:$PORT/\n" "http://127.0.0.1:$PORT/" || echo "not running (bare)"
          command -v docker >/dev/null && [ -f "$INFRA/.env" ] && [ -n "$($COMPOSE ps --status running -q 2>/dev/null)" ] && $COMPOSE ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}';;
  uri)   echo "http://$(lan_ip):$PORT";;
  up)    stack; pgrep -f "mlflow server" >/dev/null && echo "note: the bare server is running too; '$0 stop' frees port $PORT if the stack wants it"
         $COMPOSE up -d --build --wait && curl -s -o /dev/null -w "stack up · http %{http_code} at http://127.0.0.1:$SPORT/ (console http://127.0.0.1:9001/)\n" "http://127.0.0.1:$SPORT/health";;
  down)  stack; $COMPOSE down && echo "stack down; data kept in the docker volumes ('$COMPOSE down -v' wipes it)";;
  logs)  stack; $COMPOSE logs -f --tail=100 "${@:2}";;
  *) echo "usage: $0 start [--lan] | stop | status | uri (bare) | up|down|logs [service] (containers)";;
esac
