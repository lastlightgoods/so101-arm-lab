#!/usr/bin/env python3
"""so101_policy.py · the lab's model registry and deployment path: a LeRobot checkpoint -> MLflow Model Registry -> the Jetson.
    so101-policy register <pretrained_model dir> --name so101-<task> [--hf-repo tclastlight/x] [--hf-job <id>] [--dataset lastlight/x] [--note ...]
    so101-policy register --from-hf tclastlight/<repo> --name so101-<task> [--step 005000]      (snapshot_download into ~/so101/policies_hub/<repo>)
    so101-policy list [name]                      versions, aliases, origin, which one the Jetson runs
    so101-policy deploy <name>@<version|alias> [--host jetson] [--dest ~/so101/policies/<name>] [--dry-run]
    so101-policy rollback <name> [--dry-run]      back to the previous `jetson` version, one command
    so101-policy status [name]                    Jetson (DEPLOYED.json, current symlinks, so101-run default) vs the registry
Lifecycle: train -> register (run + version, alias `candidate`) -> deploy (rsync to ~/so101/policies/<name>/v<N>/, `current` -> v<N>,
DEPLOYED.json, alias `jetson`, version tagged deployed_at) -> rollback (previous entry in DEPLOYED.json's history). The last 3 versions stay on
the Jetson. docs/mlops/registry-and-deploy.md has the walkthrough.
Env: MLFLOW_TRACKING_URI (default sqlite:///~/so101/mlflow/mlflow.db, the same store mlflow_ui.sh serves), SO101_MLFLOW_ARTIFACTS (artifact
root for the registry experiment when the URI is not an http server), SO101_JETSON (ssh host, default jetson), SO101_DEPLOY_LOCAL_ROOT (a local
directory standing in for the Jetson's $HOME: ssh becomes `bash -s` and rsync a local copy; the tests use it), HF_HUB_OFFLINE (honoured)."""
import argparse, getpass, glob, hashlib, json, os, platform, shlex, subprocess, sys, time
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1"); os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
HOME = os.path.expanduser("~"); HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
EXPERIMENT = "so101-registry"; REQUIRED = ("train_config.json", "model.safetensors", "config.json"); KEEP_VERSIONS = 3
DEPLOYED = "$HOME/so101/policies/DEPLOYED.json"; SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
def tracking_uri(): return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{HOME}/so101/mlflow/mlflow.db"
def artifact_root(): return os.environ.get("SO101_MLFLOW_ARTIFACTS") or f"{HOME}/so101/mlflow/artifacts"
def hub_root(): return os.environ.get("SO101_POLICIES_HUB") or f"{HOME}/so101/policies_hub"
def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%S%z")
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def git_commit():
    try: return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception: return "unknown"
class Fail(Exception): pass
# ---------- checkpoints ----------
def check_checkpoint(d):
    d = os.path.abspath(os.path.expanduser(d))
    missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(d, f))]
    if missing: raise Fail(f"{d}: not a pretrained_model dir, missing {', '.join(missing)}")
    return d
def ckpt_params(cfg):
    """# why: the same keys as mlflow_backfill.params_for, so registry runs and backfilled training runs line up in one MLflow table"""
    p = cfg.get("policy") or {}
    params = {"policy": p.get("type"), "chunk_size": p.get("chunk_size"), "n_action_steps": p.get("n_action_steps"), "lr": p.get("optimizer_lr"), "dim_model": p.get("dim_model"),
              "use_vae": p.get("use_vae"), "batch_size": cfg.get("batch_size"), "steps": cfg.get("steps"), "seed": cfg.get("seed"), "dataset": (cfg.get("dataset") or {}).get("repo_id"),
              "warm_start": (p.get("pretrained_path") or "scratch").replace(HOME + "/so101/outputs/train/", ""), "device": p.get("device")}
    return {k: v for k, v in params.items() if v is not None}
def ckpt_step(d):
    """checkpoints/<step>/pretrained_model -> step (follows `last` symlinks); None for a flat dir."""
    parent = os.path.dirname(d)
    if os.path.basename(d) == "pretrained_model" and os.path.basename(os.path.dirname(parent)) == "checkpoints":
        s = os.path.basename(os.path.realpath(parent)); return s if s.isdigit() else None
    return None
def hf_snapshot(repo, local_dir, step=None):
    """snapshot_download of a Hub model repo into local_dir. Offline (HF_HUB_OFFLINE=1) means cache only, no network."""
    from huggingface_hub import snapshot_download   # why: lazy, the registry side must work without huggingface_hub installed
    offline = os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false")
    patterns = [f"checkpoints/{step}/*", "*.json", "*.safetensors", "*.md"] if step else None   # why: a job repo holds every checkpoint, ~200 MB each
    return snapshot_download(repo_id=repo, local_dir=local_dir, allow_patterns=patterns, local_files_only=offline)
def pick_checkpoint(root, step=None):
    """A flat pretrained_model at the root, or checkpoints/<step>/pretrained_model (the newest unless --step)."""
    if all(os.path.isfile(os.path.join(root, f)) for f in REQUIRED): return root
    cands = sorted(glob.glob(os.path.join(root, "checkpoints", "*", "pretrained_model")))
    cands = [c for c in cands if os.path.basename(os.path.dirname(c)) != "last"]
    if step: cands = [c for c in cands if os.path.basename(os.path.dirname(c)) == step]
    if not cands: raise Fail(f"{root}: no pretrained_model found" + (f" for step {step}" if step else ""))
    return cands[-1]
# ---------- registry (works on mlflow 2.x and 3.x; aliases fall back to version tags before 2.3) ----------
def client():
    import logging, mlflow
    logging.getLogger("mlflow").setLevel(logging.WARNING)   # why: "Creating initial MLflow database tables" and friends would bury the one-line answers
    mlflow.set_tracking_uri(tracking_uri()); return mlflow, mlflow.MlflowClient()
def ensure_experiment(mlflow):
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp: return exp.experiment_id
    loc = None if tracking_uri().startswith("http") else os.path.join(artifact_root(), EXPERIMENT)   # why: a server picks its own artifact root; a sqlite client must be told
    return mlflow.create_experiment(EXPERIMENT, artifact_location=loc)
def aliases_of(c, name):
    try: return {k: str(v) for k, v in (c.get_registered_model(name).aliases or {}).items()}
    except AttributeError: return {t[6:]: str(m.version) for m in c.search_model_versions(f"name='{name}'") for t in m.tags if t.startswith("alias.")}
def set_alias(c, name, alias, version):
    try: c.set_registered_model_alias(name, alias, str(version))
    except AttributeError:   # why: mlflow < 2.3 has no aliases; a version tag alias.<name> stands in and is moved by hand
        for v in c.search_model_versions(f"name='{name}'"):
            if f"alias.{alias}" in v.tags: c.delete_model_version_tag(name, v.version, f"alias.{alias}")
        c.set_model_version_tag(name, str(version), f"alias.{alias}", "1")
def alias_version(c, name, alias):
    try: return c.get_model_version_by_alias(name, alias).version
    except AttributeError:
        return next((v.version for v in c.search_model_versions(f"name='{name}'") if f"alias.{alias}" in v.tags), None)
    except Exception: return None   # why: MlflowException "alias not found"; the exact class moved between versions
def versions_of(c, name):
    try: vs = list(c.search_model_versions(f"name='{name}'"))
    except Exception as e: raise Fail(f"registered model {name!r}: {e}") from e
    return sorted(vs, key=lambda m: int(m.version))
def resolve(c, spec):
    """so101-x@3 | so101-x@candidate | so101-x (latest) -> (name, ModelVersion)"""
    name, _, ref = spec.partition("@"); vs = versions_of(c, name)
    if not vs: raise Fail(f"{name}: no versions registered")
    if not ref: return name, vs[-1]
    if ref.isdigit():
        mv = next((m for m in vs if str(m.version) == ref), None)
        if not mv: raise Fail(f"{name}: no version {ref} (have {', '.join(str(m.version) for m in vs)})")
        return name, mv
    v = alias_version(c, name, ref)
    if v is None: raise Fail(f"{name}: no alias {ref!r} (aliases: {', '.join(f'{a}->v{v}' for a, v in aliases_of(c, name).items()) or 'none'})")
    return name, next(m for m in vs if str(m.version) == str(v))
def version_dir(mlflow, c, name, mv):
    """The version's checkpoint directory on this machine: the run's local artifact dir when the store is local, else a download into policies_hub."""
    run = c.get_run(mv.run_id); uri = run.info.artifact_uri
    local = uri[7:] if uri.startswith("file://") else uri if uri.startswith("/") else None
    if local and os.path.isdir(os.path.join(local, "model")): return os.path.join(local, "model")
    dst = os.path.join(hub_root(), "registry", name, f"v{mv.version}")
    if all(os.path.isfile(os.path.join(dst, "model", f)) for f in REQUIRED): return os.path.join(dst, "model")
    import mlflow.artifacts
    return mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=dst)
# ---------- the Jetson (or a local stand-in) ----------
class Target:
    """ssh <host> bash -s / rsync to <host>:, or, with SO101_DEPLOY_LOCAL_ROOT, bash -s with that directory as $HOME and a local rsync."""
    def __init__(self, host):
        self.host = host; self.local = os.environ.get("SO101_DEPLOY_LOCAL_ROOT") or None
        if self.local: self.local = os.path.abspath(self.local); os.makedirs(self.local, exist_ok=True)
    def __str__(self): return f"local stand-in {self.local}" if self.local else f"ssh {self.host}"
    def sh(self, script, timeout=120):
        if self.local: r = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout, env={**os.environ, "HOME": self.local})
        else: r = subprocess.run(["ssh", *SSH_OPTS, self.host, "bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout)   # why: via stdin, ssh would re-split a -c argument
        if r.returncode != 0: raise Fail(f"{self}: {(r.stderr or r.stdout).strip()[-400:] or f'exit {r.returncode}'}")
        return r.stdout
    def rsync(self, src_dir, remote_dir):
        """remote_dir is a $HOME-relative path like so101/policies/<name>/v3 (an absolute remote path is passed through)."""
        dest = os.path.join(self.local, remote_dir) if self.local else f"{self.host}:{remote_dir}"
        if self.local: os.makedirs(dest, exist_ok=True)
        r = subprocess.run(["rsync", "-a", "--delete", src_dir.rstrip("/") + "/", dest.rstrip("/") + "/"], capture_output=True, text=True, timeout=3600)
        if r.returncode != 0: raise Fail(f"rsync: {r.stderr.strip()[-400:]}")
def home_rel(p):
    """~/so101/policies/x | $HOME/so101/policies/x -> so101/policies/x (what rsync and the scripts use); absolute paths stay absolute."""
    for pre in ("~/", "$HOME/"):
        if p.startswith(pre): return p[len(pre):]
    return p
def shq(p): return '"$HOME/' + home_rel(p) + '"' if not home_rel(p).startswith("/") else shlex.quote(p)
PROBE = f'''cat {DEPLOYED} 2>/dev/null; echo "###current"; for d in "$HOME"/so101/policies/*/; do n=$(basename "$d"); c="$d/current"; [ -L "$c" ] && echo "$n $(readlink "$c") $([ -f "$c/model.safetensors" ] && echo ok || echo missing) $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("type","?"))' "$c/config.json" 2>/dev/null || echo ?)"; done; echo "###versions"; for d in "$HOME"/so101/policies/*/v*/; do [ -d "$d" ] && echo "$(basename "$(dirname "$d")") $(basename "$d")"; done; true'''
def read_target(t):
    """One round trip: DEPLOYED.json (or {}), name -> (current version dir, weights present, policy type), name -> [version dirs]."""
    out = t.sh(PROBE); head, _, rest = out.partition("###current"); cur_txt, _, ver_txt = rest.partition("###versions")
    try: dep = json.loads(head) if head.strip() else {}
    except json.JSONDecodeError: dep = {"corrupt": head.strip()[:200]}
    cur = {}; vers = {}
    for l in cur_txt.strip().splitlines():
        p = l.split()
        if len(p) >= 3: cur[p[0]] = {"link": p[1], "weights": p[2] == "ok", "type": p[3] if len(p) > 3 else "?"}
    for l in ver_txt.strip().splitlines():
        p = l.split()
        if len(p) == 2: vers.setdefault(p[0], []).append(p[1])
    return dep, cur, vers
def write_deploy_script(name, version, dest, entry, dep, keep):
    """Flip `current`, write DEPLOYED.json (top level = this deploy, history appended), prune versions not in `keep`, print the remote sha."""
    hist = [h for h in (dep.get("history") or []) if isinstance(h, dict)][-49:] + [entry]
    doc = {**entry, "path": f"~/so101/policies/{name}/current" if home_rel(dest) == f"so101/policies/{name}" else dest + "/current", "history": hist}
    js = json.dumps(doc, indent=1)
    assert "\nEOF\n" not in js
    prune = " ".join(shlex.quote(f"v{v}") for v in keep)
    return f'''set -e; D={shq(dest)}; cd "$D"; ln -sfn v{version} current
mkdir -p "$HOME/so101/policies"; cat > {DEPLOYED}.tmp <<'EOF'
{js}
EOF
mv -f {DEPLOYED}.tmp {DEPLOYED}
for d in v*/; do d=${{d%/}}; case " {prune} " in *" $d "*) ;; *) rm -rf "$d"; echo "pruned $d";; esac; done
echo "sha $( (sha256sum current/model.safetensors 2>/dev/null || shasum -a 256 current/model.safetensors) | cut -d' ' -f1)"'''
# ---------- commands ----------
def cmd_register(a):
    mlflow, c = client()
    if a.from_hf:
        if a.checkpoint: raise Fail("give either a checkpoint dir or --from-hf, not both")
        local = os.path.join(hub_root(), a.from_hf.replace("/", "__"))
        print(f"hf: {a.from_hf} -> {local}" + (" (HF_HUB_OFFLINE: cache only)" if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false") else ""))
        try: root = hf_snapshot(a.from_hf, local, a.step)
        except Exception as e: raise Fail(f"snapshot_download {a.from_hf}: {type(e).__name__}: {str(e)[:300]}") from e
        ckpt = check_checkpoint(pick_checkpoint(root, a.step)); a.hf_repo = a.hf_repo or a.from_hf
    elif a.checkpoint: ckpt = check_checkpoint(a.checkpoint)
    else: raise Fail("register needs a checkpoint dir or --from-hf")
    cfg = json.load(open(os.path.join(ckpt, "train_config.json"))); params = ckpt_params(cfg); step = a.step or ckpt_step(ckpt)
    sha = sha256_of(os.path.join(ckpt, "model.safetensors"))
    tags = {k: v for k, v in {"hf_repo": a.hf_repo, "hf_job": a.hf_job, "dataset": a.dataset or params.get("dataset"), "host": platform.node().split(".")[0], "git_commit": git_commit(),
                              "note": a.note, "source_dir": ckpt, "checkpoint_step": step, "sha256": sha, "registered_by": getpass.getuser()}.items() if v}
    exp_id = ensure_experiment(mlflow)
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{a.name}:register", tags=tags) as r:
        if params: mlflow.log_params(params)
        mlflow.log_artifacts(ckpt, artifact_path="model")
    try: c.get_registered_model(a.name)
    except Exception: c.create_registered_model(a.name, description=f"SO-101 policy {a.name}")
    mv = c.create_model_version(a.name, source=f"runs:/{r.info.run_id}/model", run_id=r.info.run_id, tags={k: tags[k] for k in ("hf_repo", "hf_job", "dataset", "source_dir", "checkpoint_step", "sha256", "host", "git_commit") if k in tags}, description=a.note or None)
    set_alias(c, a.name, "candidate", mv.version)
    print(f"registered {a.name} v{mv.version}  run {r.info.run_id}  alias candidate  sha256 {sha[:12]}  {len(params)} params  from {ckpt}")
    return 0
def cmd_list(a):
    mlflow, c = client(); names = [a.name] if a.name else sorted(m.name for m in c.search_registered_models())
    if not names: print("no registered models"); return 0
    for name in names:
        vs = versions_of(c, name); al = aliases_of(c, name)
        print(f"{name}  ({len(vs)} version{'s' if len(vs) != 1 else ''}; " + (", ".join(f"{k} -> v{v}" for k, v in sorted(al.items())) or "no aliases") + ")")
        for m in vs:
            t = m.tags or {}; origin = t.get("hf_repo") or t.get("source_dir") or "?"
            marks = [k for k, v in al.items() if str(v) == str(m.version)]
            parts = [f"v{m.version}".ljust(3), ('[' + ','.join(sorted(marks)) + ']' if marks else '').ljust(20), time.strftime("%Y-%m-%d %H:%M", time.localtime((m.creation_timestamp or 0) / 1000)), f"run {str(m.run_id or '')[:8]}",
                     f"from {origin}" + (f" (job {t['hf_job']})" if t.get("hf_job") else "") + (f" step {t['checkpoint_step']}" if t.get("checkpoint_step") else "")]
            if "jetson" in marks: parts.append(f"DEPLOYED {t.get('deployed_at') or ''} -> {t.get('deployed_to') or 'jetson'}")
            print("  " + "  ".join(parts))
    return 0
def do_deploy(a, spec, action):
    mlflow, c = client(); name, mv = resolve(c, spec); t = Target(a.host); v = str(mv.version)
    dest = a.dest or f"~/so101/policies/{name}"; dep, cur, vers = read_target(t)
    if dep.get("name") == name and str(dep.get("version")) == v and cur.get(name, {}).get("link") == f"v{v}" and cur[name]["weights"]:
        print(f"{name} v{v} is already deployed on {t} (current -> v{v}); nothing to do"); return 0
    src = version_dir(mlflow, c, name, mv); sha = sha256_of(os.path.join(src, "model.safetensors"))
    hist = [h for h in (dep.get("history") or []) if isinstance(h, dict) and h.get("name") == name]
    keep = []
    for h in [{"version": v}] + hist[::-1]:
        if str(h.get("version")) not in keep: keep.append(str(h.get("version")))
    keep = keep[:KEEP_VERSIONS]
    entry = {"name": name, "version": int(v), "run_id": mv.run_id, "sha256": sha, "when": now_iso(), "by": getpass.getuser(), "from_host": platform.node().split(".")[0], "action": action}
    prune = [x for x in vers.get(name, []) if x[1:] not in keep]
    plan = [f"{action} {name} v{v} -> {t}: rsync {src} -> {home_rel(dest)}/v{v}/", f"current -> v{v}; DEPLOYED.json; keep v{', v'.join(keep)}" + (f"; prune {', '.join(prune)}" if prune else ""),
            f"alias jetson -> v{v} (was v{alias_version(c, name, 'jetson')})" if alias_version(c, name, "jetson") else f"alias jetson -> v{v}"]
    if a.dry_run: print("dry run:"); [print("  " + p) for p in plan]; return 0
    print(plan[0]); t.sh(f'mkdir -p {shq(dest)}/v{v}'); t.rsync(src, home_rel(dest) + f"/v{v}")
    out = t.sh(write_deploy_script(name, v, dest, entry, dep, keep)); remote_sha = next((l.split()[1] for l in out.splitlines() if l.startswith("sha ")), "")
    for l in out.splitlines():
        if l.startswith("pruned"): print("  " + l)
    if remote_sha != sha: raise Fail(f"sha256 mismatch after copy: local {sha[:12]} remote {remote_sha[:12] or '?'}")
    set_alias(c, name, "jetson", v); c.set_model_version_tag(name, v, "deployed_at", entry["when"]); c.set_model_version_tag(name, v, "deployed_to", a.host); c.set_model_version_tag(name, v, "deployed_by", entry["by"])
    print(f"  {plan[1]}\n  {plan[2]}\n  sha256 {sha[:12]} verified; {name} v{v} is live ({action})")
    return 0
def cmd_deploy(a): return do_deploy(a, a.spec, "deploy")
def cmd_rollback(a):
    t = Target(a.host); dep, cur, vers = read_target(t)
    hist = [h for h in (dep.get("history") or []) if isinstance(h, dict) and h.get("name") == a.name]
    if not hist: raise Fail(f"{a.name}: never deployed on {t}, nothing to roll back to")
    now = str(hist[-1].get("version")); prev = next((str(h.get("version")) for h in hist[::-1] if str(h.get("version")) != now), None)
    if prev is None: raise Fail(f"{a.name}: only v{now} was ever deployed, nothing to roll back to")
    print(f"rollback {a.name}: v{now} -> v{prev}")
    return do_deploy(a, f"{a.name}@{prev}", "rollback")
def cmd_status(a):
    mlflow, c = client(); t = Target(a.host); dep, cur, vers = read_target(t)
    names = [a.name] if a.name else sorted(set(n.name for n in c.search_registered_models()) | set(cur))
    print(f"target: {t}")
    if dep.get("name"): print(f"DEPLOYED.json: {dep['name']} v{dep.get('version')} {dep.get('when', '')} by {dep.get('by', '?')} ({dep.get('action', 'deploy')}); history {len(dep.get('history') or [])} entries")
    elif "corrupt" in dep: print("DEPLOYED.json: unreadable")
    else: print("DEPLOYED.json: none (nothing deployed through the registry yet)")
    ok = True
    for name in names:
        reg = alias_version(c, name, "jetson"); on = cur.get(name); have = ", ".join(vers.get(name, [])) or "-"
        if on:
            agree = reg is not None and f"v{reg}" == on["link"]; ok &= agree and on["weights"]
            print(f"{name}: jetson current -> {on['link']} ({on['type']}, weights {'ok' if on['weights'] else 'MISSING'}; on disk: {have}); registry jetson alias -> {'v' + str(reg) if reg else 'none'}  {'OK' if agree and on['weights'] else 'MISMATCH'}")
        else:
            if reg: ok = False
            print(f"{name}: not on the jetson (registry jetson alias -> {'v' + str(reg) if reg else 'none'})  {'MISMATCH' if reg else ''}".rstrip())
    name = dep.get("name"); on = cur.get(name) if name else None
    if on and on["weights"]:
        runner = {"act": "so101-run", "smolvla": "so101-run-vla"}.get(on["type"], "so101-run")
        print(f"{runner} default -> ~/so101/policies/{name}/current ({name} {on['link']})")
    else: print("so101-run default -> fallback (~/so101/policies/act_lighter); no deployed `current` with weights")
    return 0 if ok else 1
def main(argv=None):
    ap = argparse.ArgumentParser(prog="so101-policy", description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="checkpoint -> MLflow run + model version, alias candidate"); r.add_argument("checkpoint", nargs="?"); r.add_argument("--name", required=True)
    r.add_argument("--from-hf", metavar="REPO"); r.add_argument("--step"); r.add_argument("--hf-repo"); r.add_argument("--hf-job"); r.add_argument("--dataset"); r.add_argument("--note"); r.set_defaults(fn=cmd_register)
    l = sub.add_parser("list", help="versions, aliases, origins"); l.add_argument("name", nargs="?"); l.set_defaults(fn=cmd_list)
    for n, fn in (("deploy", cmd_deploy), ("rollback", cmd_rollback), ("status", cmd_status)):
        p = sub.add_parser(n); p.add_argument("spec" if n == "deploy" else "name", nargs=None if n != "status" else "?", help="name@version|alias" if n == "deploy" else "registered model name")
        p.add_argument("--host", default=os.environ.get("SO101_JETSON", "jetson")); p.set_defaults(fn=fn)
        if n != "status": p.add_argument("--dry-run", action="store_true")
        if n == "deploy": p.add_argument("--dest", help="remote directory (default ~/so101/policies/<name>)")
        else: p.set_defaults(dest=None)
    a = ap.parse_args(argv)
    try: return a.fn(a)
    except Fail as e: print(f"so101-policy: {e}", file=sys.stderr); return 2
if __name__ == "__main__": sys.exit(main())
