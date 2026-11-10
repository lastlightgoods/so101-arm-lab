#!/usr/bin/env python3
"""labtrack.py - a tiny, fail-safe wrapper around MLflow for the arm lab (docs/mlops/tracking.md).

    track = LabTrack("so101-picks")                       # resolves the URI, probes it, opens/creates the experiment
    with track.run("pick-3", tags={"host": "mac"}) as r:  # one MLflow run; FINISHED on exit, FAILED if the block raises
        r.params({"target": "9,6", "gain": 0.7})          # inputs, strings, written once
        r.metrics({"landing_err_mm": 12.3}, step=None)    # numbers (None / NaN skipped); step builds a series
        r.artifact("/tmp/frame.jpg", "frames")            # a file, under an optional sub-folder
        r.json(plan, "plan.json")                         # any JSON-able object as a file
        r.file("landing.jpg", jpeg_bytes, "frames")       # bytes / text as a file
    track.enabled                                         # False = everything above is a no-op; track.why says why
    track.attach(run_id)                                  # a Run bound to an existing run (another process started it); call .end() yourself

Tracking URI: $SO101_MLFLOW_URI (set but empty = off), else $MLFLOW_TRACKING_URI, else the Mac's sqlite store
when ~/so101/mlflow exists, else off. http(s) URIs get a 1 s TCP probe first so an unplugged Mac never holds the
arm. Every mlflow call is wrapped: the first failure prints one `labtrack:` line on stderr and turns tracking
off for the rest of the process; nothing ever raises into the caller. mlflow is imported lazily, so a machine
without it (or with only mlflow-skinny, which is all the Jetson needs) runs the callers unchanged.

Self-test (logs one run and prints where it went):   python labtrack.py [uri]
This file lives in jetson/ because that folder is what gets copied to the Jetson; Mac tools load it by path.
"""
import contextlib, json, math, os, re, socket, sys, tempfile, time, urllib.parse

ENV_URI = "SO101_MLFLOW_URI"
MAC_DB = os.path.expanduser("~/so101/mlflow/mlflow.db")
PROBE_S = 1.0
# why: mlflow 3.x prints a tracing hint on import, and its HTTP client retries 7 times with backoff by default; a server
# that vanishes mid-pick must cost one short request, not a minute of the arm hanging between waypoints
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
# why: against an http server (never against sqlite) mlflow writes "View run ... at: http://..." to STDOUT when a run ends; cam_calib.py
# and pick_loop.py print JSON on stdout that callers parse, and tracking must not change it (seen on the first container server, 10 Nov 2026)
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "1")
# why: mlflow >= 3.17 clients ask the server whether its artifact store can hand out presigned URLs and, when it can (S3, which the
# container stack is), download EVERY artifact straight from that URL. The local stack signs them for http://s3:9000, a name only the
# containers can resolve, so downloads hang and retry for minutes; large uploads would take the same presigned road. "false" keeps
# both streaming through the server (10 Nov 2026).
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
_KEY_BAD = re.compile(r"[^A-Za-z0-9_\-./ ]")


def resolve_uri(env=None):
    env = os.environ if env is None else env
    if ENV_URI in env: return env[ENV_URI].strip() or None
    if env.get("MLFLOW_TRACKING_URI"): return env["MLFLOW_TRACKING_URI"].strip() or None
    return "sqlite:///" + MAC_DB if os.path.isdir(os.path.dirname(MAC_DB)) else None


def reachable(uri, timeout=PROBE_S):
    """True unless `uri` is http(s) and a TCP connect to it does not succeed within `timeout` seconds."""
    u = urllib.parse.urlsplit(uri)
    if u.scheme not in ("http", "https"): return True
    try: socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout=timeout).close(); return True
    except (OSError, ValueError): return False


def key(k): return _KEY_BAD.sub("_", str(k))[:250]


def artifact_root(uri, experiment):
    """Where a sqlite-backed experiment keeps its files: an artifacts/ folder beside the .db (mlflow's own default is ./mlruns of whatever cwd first created the experiment)."""
    if not uri.startswith("sqlite:///"): return None
    db = uri[len("sqlite:///"):]; return os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(db))), "artifacts", experiment)


class LabTrack:
    def __init__(self, experiment, uri=None, probe_s=PROBE_S, env=None):
        self.experiment = experiment; self.uri = resolve_uri(env) if uri is None else (uri.strip() or None)
        self.enabled = False; self.why = ""; self._said = False; self._client = self._mlflow = self._exp_id = None
        if not self.uri: self.why = "no tracking URI"; return
        if not reachable(self.uri, probe_s): self._fail(f"{self.uri} not reachable within {probe_s:g} s"); return
        try:
            import logging; logging.getLogger("mlflow").setLevel(logging.WARNING); logging.getLogger("alembic").setLevel(logging.WARNING)   # why: "Creating initial MLflow database tables" is not something the pick loop's console should show
            import mlflow; from mlflow import MlflowClient
            self._mlflow = mlflow; self._client = MlflowClient(tracking_uri=self.uri)
            e = self._client.get_experiment_by_name(experiment)
            self._exp_id = e.experiment_id if e else self._client.create_experiment(experiment, artifact_location=artifact_root(self.uri, experiment))
            self.enabled = True
        except BaseException as ex:   # why: ImportError (no mlflow on this box), a locked sqlite file, a 500 from the server: all mean "log nowhere", never "stop the tool"
            if isinstance(ex, (KeyboardInterrupt, SystemExit)): raise
            self._fail(f"{self.uri}: {type(ex).__name__}: {str(ex).splitlines()[0][:160] if str(ex) else ''}")

    def _fail(self, why):
        self.enabled = False; self.why = why
        if not self._said: self._said = True; print(f"labtrack: tracking off ({self.experiment}): {why}", file=sys.stderr, flush=True)

    def _call(self, what, fn, *a, **k):
        if not self.enabled: return None
        try: return fn(*a, **k)
        except BaseException as ex:
            if isinstance(ex, (KeyboardInterrupt, SystemExit)): raise
            self._fail(f"{what}: {type(ex).__name__}: {str(ex).splitlines()[0][:160] if str(ex) else ''}"); return None

    def start(self, name, tags=None):
        """A Run (no-op when disabled). Default tag host=<short hostname>; callers' tags win."""
        tags = {"host": socket.gethostname().split(".")[0], **{key(k): str(v) for k, v in (tags or {}).items()}}
        run = self._call("start_run", lambda: self._client.create_run(self._exp_id, tags=tags, run_name=name))
        return Run(self, run.info.run_id if run is not None else None)

    def attach(self, run_id): return Run(self, run_id if (self.enabled and run_id) else None)

    @contextlib.contextmanager
    def run(self, name, tags=None):
        r = self.start(name, tags)
        try: yield r
        except BaseException: r.end("FAILED"); raise
        else: r.end("FINISHED")


class Run:
    def __init__(self, track, run_id): self.track, self.id = track, run_id

    @property
    def enabled(self): return self.track.enabled and self.id is not None

    def _ents(self):
        from mlflow.entities import Metric, Param, RunTag
        return Metric, Param, RunTag

    def params(self, d):
        if not self.enabled or not d: return
        def go():
            _, Param, _ = self._ents()
            try: from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH as lim   # 6000 on 3.x, 500 on old 2.x
            except ImportError: lim = 500
            self.track._client.log_batch(self.id, params=[Param(key(k), str(v)[:lim]) for k, v in d.items() if v is not None])
        self.track._call("log_params", go)

    def metrics(self, d, step=None):
        if not self.enabled or not d: return
        def go():
            Metric, _, _ = self._ents(); ts = int(time.time() * 1000); ms = []
            for k, v in d.items():
                try: f = float(v)
                except (TypeError, ValueError): continue
                if math.isfinite(f): ms.append(Metric(key(k), f, ts, int(step or 0)))
            if ms: self.track._client.log_batch(self.id, metrics=ms)
        self.track._call("log_metrics", go)

    def tags(self, d):
        if not self.enabled or not d: return
        def go():
            _, _, RunTag = self._ents(); self.track._client.log_batch(self.id, tags=[RunTag(key(k), str(v)) for k, v in d.items() if v is not None])
        self.track._call("set_tags", go)

    def artifact(self, path, subdir=None):
        if not self.enabled or not path or not os.path.isfile(path): return
        self.track._call("log_artifact", self.track._client.log_artifact, self.id, path, subdir)

    def file(self, name, data, subdir=None):
        """Log bytes or text as an artifact called `name` (a temp file on the way)."""
        if not self.enabled: return
        d = tempfile.mkdtemp(prefix="labtrack_"); p = os.path.join(d, os.path.basename(name))
        try:
            with open(p, "wb") as f: f.write(data if isinstance(data, bytes) else str(data).encode())
            self.artifact(p, subdir)
        finally:
            try: os.remove(p); os.rmdir(d)
            except OSError: pass

    def json(self, obj, name, subdir=None):
        if not self.enabled: return
        try: self.file(name, json.dumps(obj, indent=1, default=str), subdir)
        except (TypeError, ValueError) as ex: self.track._fail(f"json {name}: {ex}")

    def end(self, status="FINISHED"):
        if not self.enabled: return
        self.track._call("end_run", self.track._client.set_terminated, self.id, status, int(time.time() * 1000)); self.id = None


if __name__ == "__main__":
    t = LabTrack("so101-labtrack-selftest", sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({"uri": t.uri, "enabled": t.enabled, "why": t.why}))
    if t.enabled:
        with t.run("selftest-" + time.strftime("%Y%m%d-%H%M%S"), {"kind": "selftest"}) as r:
            r.params({"cwd": os.getcwd()}); r.metrics({"ok": 1}); r.json({"hello": "lab"}, "hello.json"); print(json.dumps({"run_id": r.id, "experiment": t.experiment}))
    sys.exit(0 if t.enabled else 1)
