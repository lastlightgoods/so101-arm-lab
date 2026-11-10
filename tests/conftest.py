"""Root pytest shim: tracking off for every test by default. jetson/labtrack.py falls back to the Mac's real MLflow store
(~/so101/mlflow/mlflow.db) when no URI is set, so without this a CLI test run from the Mac would write into TC's real
experiments. Tests that exercise tracking set SO101_MLFLOW_URI themselves (monkeypatch or the subprocess env)."""
import os as _os
_os.environ["SO101_MLFLOW_URI"] = ""
