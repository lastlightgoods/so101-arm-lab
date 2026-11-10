"""tools/mlflow_ui.sh: the bits that run without a server (syntax, usage, the LAN bind flag, the `uri` line the Jetson copies)."""
import os, re, subprocess
from mlhelpers import TOOLS

SH = os.path.join(TOOLS, "mlflow_ui.sh")


def test_script_parses_and_documents_lan():
    assert subprocess.run(["bash", "-n", SH], capture_output=True).returncode == 0
    r = subprocess.run(["bash", SH, "help"], capture_output=True, text=True); assert "start [--lan]" in r.stdout and "uri" in r.stdout
    src = open(SH).read(); assert '"$2" = "--lan"' in src and "HOST=0.0.0.0" in src and "trusted" in src and "no auth" in src


def test_uri_is_one_http_line_on_port_5050():
    r = subprocess.run(["bash", SH, "uri"], capture_output=True, text=True, env={**os.environ, "MLFLOW_PORT": "5050"})
    assert r.returncode == 0 and re.fullmatch(r"http://[0-9A-Za-z.\-]*:5050\n", r.stdout), r.stdout   # the host part may be empty on a box with no LAN address (CI)
