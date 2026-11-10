"""Container health check: GET /health on the local MLflow server, exit 0 on 200. Used by docker-compose.yml and the ECS task definition.
why: python:slim ships no curl, and the same one-liner must work in both places."""
import os, sys, urllib.request
try:
    sys.exit(0 if urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('MLFLOW_PORT', '5050')}/health", timeout=4).status == 200 else 1)
except Exception as e:
    print(f"unhealthy: {e}"); sys.exit(1)
