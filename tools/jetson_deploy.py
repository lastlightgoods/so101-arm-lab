#!/usr/bin/env python3
"""jetson_deploy.py - one way to put the repo's jetson/ code on the Jetson, with a record of what is there (docs/jetson-deploy.md).

  tools/jetson_deploy.py status   [--ref REF]                      compare ~/so101 on the Jetson with the files at git REF (default HEAD)
  tools/jetson_deploy.py deploy   [--ref REF] [--files a b ..] [--yes] [--restart recorder|camd] [--allow-system]
  tools/jetson_deploy.py rollback [--to BACKUP] [--yes]

Files come from `git show REF:path`, never from the working tree, so uncommitted edits cannot deploy. `deploy` is a dry run unless --yes.
Nothing is deleted on the Jetson by a deploy; nothing is restarted unless --restart names one of two idle services. The Jetson's calibration
(~/so101/calib) is written THERE by cam_calib.py / cam_lock.py and the Mac copy in jetson/calib is the synced mirror, so calib is never deployed by default.
udev / systemd / NetworkManager / firewall files are never installed by this tool: with --files NAME --allow-system it stages the file in ~/so101
and prints the sudo lines.

Overnight preflight (one line; does not touch the arm):
    from tools.jetson_deploy import deployed_matches; ok, why = deployed_matches("<commit the run was tested on>")
"""
import argparse, datetime, hashlib, json, os, re, shlex, subprocess, sys

HOST = "jetson"                       # the ssh alias; the only name this file knows
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = "DEPLOYED.json"
BAK_ROOT = "deploy_bak"
BUSY_FLAGS = (".arms_busy", ".cams_busy")

# --- what is deployable
PLUGIN_DIRS = ("plugins/lerobot-robot-busheal/", "plugins/lerobot-policy-actasync/")   # editable installs in the Jetson venv; lerobot-camera-avf is the Mac's AVFoundation camera and stays off
SYSTEM_FILES = {   # never copied by a normal deploy: need root or change the network. value = the sudo lines to print
    "hotspot.sh": ["# not installed: it is run by hand (docs/hotspot.md); deploy only stages the file"],
    "90-so101-hotspot.dispatcher": ["sudo install -m 755 -o root -g root ~/so101/90-so101-hotspot.dispatcher /etc/NetworkManager/dispatcher.d/90-so101-hotspot", "# the interface and profile names are filled in by `so101-hotspot install` (docs/hotspot.md); read that before replacing a working dispatcher"],
    "so101-hotspot.nft": ["sudo install -m 644 ~/so101/so101-hotspot.nft /etc/so101-hotspot.nft", "sudo nft -f /etc/so101-hotspot.nft   # only with the hotspot up (docs/hotspot.md)"],
    "99-so101-cams.rules": ["sudo install -m 644 ~/so101/99-so101-cams.rules /etc/udev/rules.d/99-so101-cams.rules", "sudo udevadm control --reload && sudo udevadm trigger   # briefly re-enumerates the cameras: only with no arm or camera job running"],
    "so101-cam-lock.service": ["sudo install -m 644 ~/so101/so101-cam-lock.service /etc/systemd/system/so101-cam-lock.service", "sudo systemctl daemon-reload   # then restart the unit only when .cams_busy is absent"],
}
SETUP_ONLY = {"jetson_setup.sh"}      # deployed as a file (it is in the repo), never run by this tool
ROOTS = {   # long-running (or per-session) entry points: file -> (what it is, how a new version gets picked up, --restart name or None)
    "arm_state.py": ("arm_state.py (the Mac dashboard starts it over ssh)", "nothing to do: the next poller start loads it; to force it, restart the dashboard (so101-dash) on the Mac", None),
    "camd.py": ("camd.py serve (camera daemon)", "idle only: `deploy --restart camd` (refuses while a recorder is configured); the dashboard camera tiles reconnect", "camd"),
    "demo_recorderd.py": ("demo_recorderd.py serve (demo recorder)", "idle only: `deploy --restart recorder` (refuses with an episode open; the Mac re-sends `configure` when autoloop --record-demos starts)", "recorder"),
    "pick_seq.py": ("pick_seq.py --serve (one per Mac session)", "nothing to do: the next session loads it; a running session keeps the old code until it exits", None),
    "cam_lock.py": ("cam_lock.py guard (resident re-locker, if running)", "manual, with .cams_busy absent: pkill -f 'cam_lock.py guard'; so101-cams-guard", None),
}
ARM_RE = re.compile(r"lerobot-(teleoperate|record|rollout|calibrate|replay)|(^|[\s/])(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|scanpose_teach|free_arm|calib_gripper|servo_probe|mark_pose|cam_calib|cam_health|cam_drift)\.py")
RECORDER_RE = re.compile(r"demo_recorderd\.py serve")
CAMD_RE = re.compile(r"camd\.py serve")
MAC_RE = re.compile(r"tools/(tidy|autoloop|pick_loop)\.py")
LERO_RE = re.compile(r"lerobot-(teleoperate|record|rollout|calibrate|replay)")


def md5(b): return hashlib.md5(b).hexdigest()
def utcnow(): return datetime.datetime.now(datetime.timezone.utc)


# ---------------------------------------------------------------- git (injected in tests)
class Git:
    def __init__(self, root=REPO): self.root = root

    def _run(self, *a):
        r = subprocess.run(["git", "-C", self.root, *a], capture_output=True)
        if r.returncode: raise RuntimeError(f"git {' '.join(a)}: {r.stderr.decode()[:200].strip()}")
        return r.stdout

    def rev(self, ref): return self._run("rev-parse", "--verify", ref + "^{commit}").decode().strip()
    def ls(self, ref):
        """[(mode, path)] of every file under jetson/ and plugins/ at ref."""
        out = []
        for line in self._run("ls-tree", "-r", "-z", ref, "jetson", "plugins").decode().split("\0"):
            if line: meta, path = line.split("\t", 1); out.append((meta.split()[0], path))
        return out
    def show(self, ref, path): return self._run("show", f"{ref}:{path}")
    def dirty(self):
        """Tracked, uncommitted changes under jetson/ and plugins/ (untracked files cannot deploy, so they do not count)."""
        return [l for l in self._run("status", "--porcelain", "--untracked-files=no", "--", "jetson", "plugins").decode().splitlines() if l.strip()]


# ---------------------------------------------------------------- the manifest
class Item:
    def __init__(self, path, remote, kind, mode, git_ref):
        self.path, self.remote, self.kind, self.mode, self.ref = path, remote, kind, mode, git_ref   # kind: deploy | calib | system
        self._blob = None
    def blob(self, git):
        if self._blob is None: self._blob = git.show(self.ref, self.path)
        return self._blob
    def want(self, git): return md5(self.blob(git))


def classify(mode, path):
    """(remote path, kind) for one tracked path, or None when it is not something this tool knows (docs, pngs...)."""
    exe = "755" if mode == "100755" else "644"
    if path.startswith("jetson/calib/"): return path[len("jetson/"):], "calib", exe
    if path.startswith(PLUGIN_DIRS):
        return path, ("deploy" if "/__pycache__/" not in path and not path.endswith(".pyc") else None), exe
    if path.startswith("jetson/") and path.count("/") == 1:
        name = path[len("jetson/"):]
        if name in SYSTEM_FILES: return name, "system", exe
        if name.endswith((".py", ".sh")): return name, "deploy", exe
        if name.endswith((".rules", ".service", ".nft", ".dispatcher")): return name, "system", exe
    return None


def manifest(git, ref):
    items = {}
    for mode, path in git.ls(ref):
        c = classify(mode, path)
        if c and c[1]: items[c[0]] = Item(path, c[0], c[1], c[2], ref)
    return items


def users(git, ref, items, remote):
    """[(what, pickup step, restart name)] for a deployed file: which long-running processes load it (their transitive local imports, read from the files at ref)."""
    if remote == "env.sh": return [("env.sh (sourced by ~/.bashrc)", "new shells only; an open shell keeps the old functions until `source ~/so101/env.sh`", None)]
    if remote.startswith("plugins/"): return [("lerobot jobs and pick_seq (editable plugin in the venv)", "nothing to do: loaded when the next job starts", None)]
    names = {r[:-3] for r in items if "/" not in r and r.endswith(".py")}
    graph = {}
    def imports(mod):
        if mod in graph: return graph[mod]
        graph[mod] = set()
        it = items.get(mod + ".py")
        if not it: return graph[mod]
        try: src = it.blob(git).decode("utf-8", "replace")
        except Exception: return graph[mod]
        found = set()
        for line in src.splitlines():
            m = re.match(r"\s*from\s+(\w+)\s+import\b", line)
            if m: found.add(m.group(1))
            m = re.match(r"\s*import\s+([\w\s,]+?)(?:\s+as\s+\w+)?\s*(?:#.*)?$", line)
            if m: found |= {x.strip() for x in m.group(1).split(",")}
        graph[mod] = found & names
        return graph[mod]
    def closure(root):
        seen, todo = set(), [root]
        while todo:
            m = todo.pop()
            if m in seen: continue
            seen.add(m); todo.extend(imports(m))
        return seen
    mod = remote[:-3] if remote.endswith(".py") else None
    out = [(w, step, rs) for root, (w, step, rs) in ROOTS.items() if mod and (mod in closure(root[:-3]) or root == remote)]
    if out: return out
    if remote in SETUP_ONLY: return [(remote + " (install script)", "nothing: this tool never runs it; run it by hand per JETSON_RUNBOOK.md only when asked", None)]
    return [(remote + " (run on demand)", "nothing to do: the next run uses it", None)]


# ---------------------------------------------------------------- the Jetson (injected in tests)
def _safe(rel):
    if rel.startswith("/") or ".." in rel.split("/") or not rel or "\n" in rel: raise ValueError(f"unsafe remote path {rel!r}")
    return rel


class SshTransport:
    """Everything is relative to ~/so101 on the Jetson. `run(argv, input) -> (rc, stdout bytes)` is injectable."""
    def __init__(self, host=HOST, run=None, timeout=20):
        self.host, self.timeout = host, timeout
        self._run = run or self._ssh

    def _ssh(self, argv, data):
        r = subprocess.run(argv, input=data, capture_output=True, timeout=self.timeout + 60)
        return r.returncode, r.stdout

    def sh(self, script, data=None, check=True):
        rc, out = self._run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.timeout}", self.host, script], data)
        if check and rc: raise RuntimeError(f"ssh {self.host}: exit {rc}")
        return rc, out

    def listing(self):
        _, out = self.sh('cd "$HOME/so101" && for f in *.py *.sh; do [ -f "$f" ] && md5sum "$f"; done; true')
        return {l.split()[1]: l.split()[0] for l in out.decode().splitlines() if len(l.split()) == 2}

    def md5s(self, rels):
        if not rels: return {}
        _, out = self.sh('cd "$HOME/so101" && md5sum -- ' + " ".join(shlex.quote(_safe(r)) for r in rels) + " 2>/dev/null; true")
        got = {l.split(None, 1)[1].strip(): l.split()[0] for l in out.decode().splitlines() if len(l.split()) >= 2}
        return {r: got.get(r) for r in rels}

    def read(self, rel):
        rc, out = self.sh(f'cat "$HOME/so101/"{shlex.quote(_safe(rel))}', check=False)
        return out if rc == 0 else None

    def flags(self): return [f for f in BUSY_FLAGS if self.sh(f'test -e "$HOME/so101/{f}"', check=False)[0] == 0]

    def procs(self):
        _, out = self.sh("ps -eo pid,args --no-headers")
        return [l.strip() for l in out.decode().splitlines() if l.strip() and "ps -eo pid,args" not in l]

    def ctl_status(self):
        rc, out = self.sh('cd "$HOME/so101" && .venv/bin/python demo_recorderd.py ctl status', check=False)
        try: return json.loads(out.decode().strip().splitlines()[-1]) if rc == 0 else None
        except (ValueError, IndexError): return None

    def backup(self, bakdir, rels):
        d = shlex.quote(_safe(bakdir))
        self.sh(f'cd "$HOME/so101" && mkdir -p {d} && for f in ' + " ".join(shlex.quote(_safe(r)) for r in rels) + f'; do [ -f "$f" ] && cp -p --parents -- "$f" {d}/; done; true')

    def put(self, rel, data, mode="644"):
        rel = _safe(rel); tmp = shlex.quote(rel + ".deploy_tmp"); dst = shlex.quote(rel)
        self.sh(f'set -e; cd "$HOME/so101"; mkdir -p "$(dirname {dst})"; cat > {tmp}; chmod {int(mode, 8):o} {tmp}; mv -f {tmp} {dst}', data)

    def restore(self, bakdir, rel):
        rel = _safe(rel); src = shlex.quote(_safe(bakdir) + "/" + rel); tmp = shlex.quote(rel + ".deploy_tmp"); dst = shlex.quote(rel)
        self.sh(f'set -e; cd "$HOME/so101"; cp -p -- {src} {tmp}; mv -f {tmp} {dst}')

    def remove(self, rel): self.sh(f'rm -f -- "$HOME/so101/"{shlex.quote(_safe(rel))}')
    def backups(self):
        _, out = self.sh(f'ls -1 "$HOME/so101/{BAK_ROOT}" 2>/dev/null; true')
        return sorted(l for l in out.decode().split() if l)

    def restart(self, name):
        if name == "recorder":
            script = ('cd "$HOME/so101"; pkill -TERM -f "demo_recorderd.py serve"; for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f "demo_recorderd.py serve" >/dev/null || break; sleep 1; done; '
                      'pgrep -f "demo_recorderd.py serve" >/dev/null && { echo "old recorder still running"; exit 1; }; mkdir -p outputs/run; '
                      'nohup nice -n 5 "$HOME/so101/.venv/bin/python" "$HOME/so101/demo_recorderd.py" serve > "$HOME/so101/outputs/run/demo_recorderd.log" 2>&1 < /dev/null & sleep 3; pgrep -f "demo_recorderd.py serve" >/dev/null && echo restarted')
        elif name == "camd":
            script = ('cd "$HOME/so101"; pkill -TERM -f "camd.py serve"; for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f "camd.py serve" >/dev/null || break; sleep 1; done; '
                      'pgrep -f "camd.py serve" >/dev/null && { echo "old camd still running"; exit 1; }; mkdir -p outputs/run; '
                      'nohup python3 "$HOME/so101/camd.py" serve > "$HOME/so101/outputs/run/camd.log" 2>&1 < /dev/null & sleep 3; pgrep -f "camd.py serve" >/dev/null && echo restarted')
        else: raise ValueError(name)
        rc, out = self.sh(script, check=False)
        return rc == 0 and b"restarted" in out, out.decode().strip()[-200:]


def local_procs():
    r = subprocess.run(["ps", "-axo", "args"], capture_output=True, text=True)
    return [l.strip() for l in r.stdout.splitlines()[1:] if l.strip()]


# ---------------------------------------------------------------- status
def compare(git, tr, ref):
    """Full comparison -> dict. Reads only."""
    items = manifest(git, ref)
    dep = {r: i for r, i in items.items() if i.kind == "deploy"}
    listing = tr.listing()
    remote = {r: listing[r] for r in dep if r in listing}
    rest = [r for r in dep if r not in listing]
    remote.update({r: m for r, m in tr.md5s(rest).items() if m})
    cats = {"match": [], "differs": [], "missing": []}
    for r, it in sorted(dep.items()):
        have = remote.get(r)
        cats["missing" if have is None else "match" if have == it.want(git) else "differs"].append(r)
    sysnames = {r for r, i in items.items() if i.kind == "system"}
    only = sorted(n for n in listing if n not in dep and n not in sysnames)
    sysinfo = {}
    sys_md5 = tr.md5s(sorted(sysnames))
    for r in sorted(sysnames): sysinfo[r] = "missing" if not sys_md5.get(r) else "match" if sys_md5[r] == items[r].want(git) else "differs"
    rec = None
    raw = tr.read(RECORD)
    if raw:
        try: rec = json.loads(raw.decode())
        except ValueError: rec = {"corrupt": True}
    edited = []
    if rec and isinstance(rec.get("files"), dict):
        now = tr.md5s(sorted(rec["files"]))
        edited = sorted(p for p, m in rec["files"].items() if now.get(p) != m)
    at = not cats["differs"] and not cats["missing"]
    state = "dirty" if edited else ("at" if at else "behind")
    return {"ref": ref, "items": items, "dep": dep, "cats": cats, "only": only, "system": sysinfo, "record": rec, "edited": edited, "state": state, "remote": remote}


def report(git, c, sha):
    L = [f"ref {c['ref']} = {sha[:10]}"]
    for k in ("differs", "missing"): L += [f"  {k.upper():8} {p}" for p in c["cats"][k]]
    L += [f"  ONLY-ON-JETSON {p}" for p in c["only"]]
    L.append(f"  {len(c['cats']['match'])} match, {len(c['cats']['differs'])} differ, {len(c['cats']['missing'])} missing, {len(c['only'])} only on the Jetson")
    sysd = {k: v for k, v in c["system"].items() if v != "match"}
    if sysd: L.append("  system files (never deployed by default): " + ", ".join(f"{k} {v}" for k, v in sysd.items()))
    rec = c["record"]
    if rec is None: L.append(f"{RECORD}: none (no deploy recorded)")
    else:
        L.append(f"{RECORD}: commit {str(rec.get('commit'))[:10]} at {rec.get('time')}" + (" (partial)" if rec.get("partial") else ""))
        if c["edited"]: L.append("  edited on the Jetson since that deploy: " + ", ".join(c["edited"]))
    word = {"at": f"AT {sha[:10]}", "behind": f"BEHIND {sha[:10]}", "dirty": "DIRTY (edited on the Jetson since the last deploy)"}[c["state"]]
    if c["state"] == "dirty" and (c["cats"]["differs"] or c["cats"]["missing"]): word += f"; also not at {sha[:10]}"
    L.append("Jetson is " + word)
    return "\n".join(L)


def deployed_matches(ref="HEAD", transport=None, git=None):
    """(ok, report) for an unattended run's preflight: ok only when every deployable file on the Jetson equals the file at `ref`
    and nothing was edited there since the last deploy. Read only; an unreachable Jetson is (False, reason)."""
    git, tr = git or Git(), transport or SshTransport()
    try:
        sha = git.rev(ref); c = compare(git, tr, sha)
    except Exception as e: return False, f"cannot compare the Jetson with {ref}: {type(e).__name__}: {str(e)[:150]}"
    return c["state"] == "at", report(git, c, sha)


# ---------------------------------------------------------------- busy
def busy_reasons(tr, mac_procs=local_procs):
    """Why a write must not happen right now ([] = idle). Anything that cannot be determined counts as busy."""
    why = []
    try:
        why += [f"{f} is set on the Jetson" for f in tr.flags()]
        procs = tr.procs()
        why += [f"arm job running: {p[:90]}" for p in procs if ARM_RE.search(p) and not re.match(r"\d+ (bash|sh) -c", p)]
        if any(RECORDER_RE.search(p) and not re.match(r"\d+ (bash|sh) -c", p) for p in procs):
            st = tr.ctl_status()
            if st is None: why.append("recorder is running but did not answer `ctl status`")
            elif st.get("episode"): why.append(f"recorder has an open episode: {st['episode']}")
    except Exception as e: why.append(f"cannot read the Jetson state ({type(e).__name__}); refusing")
    try: why += [f"on the Mac: {p[:90]}" for p in mac_procs() if MAC_RE.search(p) and "jetson_deploy" not in p]
    except Exception as e: why.append(f"cannot list Mac processes ({type(e).__name__}); refusing")
    return why


# ---------------------------------------------------------------- deploy / rollback
def now_record(git, tr, items, sha, ref, **extra):
    paths = sorted(r for r, i in items.items() if i.kind == "deploy")
    cur = {r: m for r, m in tr.md5s(paths).items() if m}
    return {"commit": sha, "ref": ref, "time": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "files": cur, **extra}


def put_record(tr, rec): tr.put(RECORD, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode(), "644")


def plan_deploy(git, tr, ref, files=None, allow_system=False):
    """-> (sha, compare dict, [(Item, 'differs'|'missing'|'staged')], [errors])."""
    sha = git.rev(ref); c = compare(git, tr, sha); items, err, todo = c["items"], [], []
    if files:
        for f in files:
            it = items.get(f) or items.get(f.removeprefix("jetson/"))
            if not it: err.append(f"{f}: not in the manifest at {ref}"); continue
            if it.kind != "deploy" and not allow_system: err.append(f"{f}: a {it.kind} file; needs --files {f} --allow-system"); continue
            todo.append(it)
    else: todo = [items[r] for r in c["cats"]["differs"] + c["cats"]["missing"]]
    have = c["remote"]; out = []
    for it in todo:
        if it.kind == "deploy": st = "missing" if it.remote not in have else ("differs" if have[it.remote] != it.want(git) else None)
        else:
            m = tr.md5s([it.remote])[it.remote]; st = None if m == it.want(git) else ("missing" if m is None else "differs")
        if st: out.append((it, st))
    return sha, c, out, err


def sudo_steps(it): return SYSTEM_FILES.get(it.remote, [])


def cmd_deploy(git, tr, ref=None, files=None, yes=False, restart=None, allow_system=False, mac_procs=local_procs, out=print):
    """Returns the exit code: 0 ok / dry run, 2 refused or failed."""
    if restart not in (None, "recorder", "camd"): out(f"--restart {restart}: only recorder and camd may be restarted by this tool"); return 2
    if yes and ref is None:
        d = git.dirty()
        if d: out("refusing: the working tree has uncommitted changes under jetson/ or plugins/ (deploy ships committed files only):\n  " + "\n  ".join(d) + "\ncommit them, or name a commit with --ref"); return 2
    ref = ref or "HEAD"
    sha, c, todo, err = plan_deploy(git, tr, ref, files, allow_system)
    if err: out("\n".join(err)); return 2
    was = (c["record"] or {}).get("commit")
    out(f"plan: bring the Jetson from {str(was)[:10] if was else 'an unrecorded state'} to {sha[:10]} ({ref}); {len(todo)} file(s)")
    changed_deploy = [it for it, _ in todo if it.kind == "deploy"]
    for it, st in todo:
        tag = {"deploy": "", "calib": "  [CALIB: the Jetson is the source of truth]", "system": "  [SYSTEM: staged only]"}[it.kind]
        out(f"  {st:8} {it.remote}{tag}")
        for what, step, rs in users(git, sha, c["items"], it.remote): out(f"           used by {what}\n           -> {step}")
    if c["state"] == "dirty": out(f"warning: edited on the Jetson since the last deploy (the backup keeps those versions): {', '.join(c['edited'])}")
    why = busy_reasons(tr, mac_procs)
    if why: out(("REFUSED now: " if yes else "would be refused now: ") + "; ".join(why))
    if not yes:
        out("dry run: nothing changed. Add --yes to deploy."); return 0
    if why: return 2
    if not todo: out("nothing to do")
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ"); bak = f"{BAK_ROOT}/{stamp}_{sha[:7]}"
    if todo:
        prev = tr.read(RECORD)
        replaced = [it.remote for it, st in todo if st == "differs"]
        tr.backup(bak, replaced + ([RECORD] if prev else []))
        meta = {"commit": sha, "time": stamp, "replaced": replaced, "added": {it.remote: it.want(git) for it, st in todo if st == "missing"}, "prev_deployed": json.loads(prev.decode()) if prev else None}
        tr.put(f"{bak}/BACKUP.json", (json.dumps(meta, indent=1, sort_keys=True) + "\n").encode(), "644")
        for it, st in todo:
            tr.put(it.remote, it.blob(git), it.mode)
        bad = [it.remote for it, st in todo if tr.md5s([it.remote])[it.remote] != it.want(git)]
        if bad: out(f"VERIFY FAILED (md5 after copy) for {bad}; DEPLOYED.json not updated. Run: tools/jetson_deploy.py rollback --yes   (backup {bak})"); return 2
        put_record(tr, now_record(git, tr, c["items"], sha, ref, partial=any(r not in {i.remote for i, _ in todo} for r in c["cats"]["differs"] + c["cats"]["missing"]), backup=bak))
        out(f"deployed {len(todo)} file(s), md5 verified, backup in ~/{'so101/' + bak}, {RECORD} written")
        for it, st in todo:
            if it.kind == "system":
                out(f"{it.remote} is staged in ~/so101 only. To install, run on the Jetson (needs sudo; this tool will not):"); [out("  " + s) for s in sudo_steps(it)]
    needed = sorted({(rs, what, step) for it in changed_deploy for what, step, rs in users(git, sha, c["items"], it.remote)}, key=str)
    rc = 0
    if restart:
        if not any(rs == restart for rs, _, _ in needed): out(f"--restart {restart}: none of the deployed files is used by it; not restarting")
        else: rc = do_restart(tr, restart, out)
    for rs, what, step in needed:
        if rs and rs == restart and rc == 0: continue
        out(f"no restart: {what}: {step}" if step.startswith("nothing") else f"restart needed: {what}\n  {step}")
    return rc


def do_restart(tr, name, out):
    """The only restarts this tool performs. recorder: running, and unconfigured or no episode open. camd: running, and no recorder configured. Both re-check the busy flags."""
    procs = tr.procs(); live = lambda rx: any(rx.search(p) and not re.match(r"\d+ (bash|sh) -c", p) for p in procs)
    flags = tr.flags()
    if flags: out(f"--restart {name} refused: {', '.join(flags)} set"); return 2
    rec_up = live(RECORDER_RE); st = tr.ctl_status() if rec_up else None
    if rec_up and st is None: out(f"--restart {name} refused: recorder did not answer `ctl status`"); return 2
    if name == "recorder":
        if not rec_up: out("--restart recorder refused: not running (start it per docs/scripted-demos.md)"); return 2
        if st.get("episode"): out("--restart recorder refused: an episode is open"); return 2
    else:
        if not live(CAMD_RE): out("--restart camd refused: camd is not running (start it per docs/scripted-demos.md)"); return 2
        if rec_up and st.get("configured"): out("--restart camd refused: the recorder is configured and reads from camd; restart the recorder first or wait until it is idle and unconfigured"); return 2
    ok, msg = tr.restart(name)
    out(f"restart {name}: {'ok' if ok else 'FAILED: ' + msg}")
    return 0 if ok else 2


def cmd_rollback(git, tr, to=None, yes=False, mac_procs=local_procs, out=print):
    bks = tr.backups()
    pick = [b for b in bks if b == to] or [b for b in bks if to and to in b] if to else bks[-1:]
    if len(pick) != 1: out(f"no unique backup matches {to!r}; have: {', '.join(bks) or 'none'}"); return 2
    bak = f"{BAK_ROOT}/{pick[0]}"
    raw = tr.read(f"{bak}/BACKUP.json")
    if not raw: out(f"{bak} has no BACKUP.json; cannot roll back safely"); return 2
    meta = json.loads(raw.decode())
    added = [r for r, m in meta["added"].items() if tr.md5s([r])[r] == m]
    out(f"rollback to before {meta['commit'][:10]} ({pick[0]}): restore {len(meta['replaced'])} file(s)" + (f", remove {len(added)} file(s) that deploy added" if added else ""))
    for r in meta["replaced"]: out(f"  restore {r}")
    for r in added: out(f"  remove  {r}")
    why = busy_reasons(tr, mac_procs)
    if why: out(("REFUSED now: " if yes else "would be refused now: ") + "; ".join(why))
    if not yes: out("dry run: nothing changed. Add --yes."); return 0
    if why: return 2
    for r in meta["replaced"]: tr.restore(bak, r)
    for r in added: tr.remove(r)
    bad = [r for r in meta["replaced"] if tr.md5s([r])[r] is None]
    if bad: out(f"restore failed for {bad}"); return 2
    prev = meta.get("prev_deployed")
    items = {r: Item(r, r, "deploy", "644", "") for r in (prev or {}).get("files", {})} or {r: Item(r, r, "deploy", "644", "") for r in meta["replaced"]}
    cur = {r: m for r, m in tr.md5s(sorted(items)).items() if m}
    rec = {"commit": (prev or {}).get("commit"), "ref": (prev or {}).get("ref"), "time": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "files": cur, "rolled_back_from": meta["commit"], "partial": True}
    put_record(tr, rec)
    out(f"rolled back; {RECORD} now says commit {str(rec['commit'])[:10]} (rolled back from {meta['commit'][:10]}). Restart any process that loads the restored files (see `deploy` output for the mapping).")
    return 0


# ---------------------------------------------------------------- cli
def main(argv=None, git=None, tr=None, mac_procs=local_procs, out=print):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("--ref", default="HEAD")
    d = sub.add_parser("deploy"); d.add_argument("--ref"); d.add_argument("--files", nargs="+"); d.add_argument("--yes", action="store_true"); d.add_argument("--restart", choices=["recorder", "camd"]); d.add_argument("--allow-system", action="store_true")
    r = sub.add_parser("rollback"); r.add_argument("--to"); r.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv); git, tr = git or Git(), tr or SshTransport()
    try:
        if a.cmd == "status":
            sha = git.rev(a.ref); c = compare(git, tr, sha); out(report(git, c, sha)); return 0 if c["state"] == "at" else 1
        if a.cmd == "deploy": return cmd_deploy(git, tr, a.ref, a.files, a.yes, a.restart, a.allow_system, mac_procs, out)
        return cmd_rollback(git, tr, a.to, a.yes, mac_procs, out)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as e:
        out(f"error: {type(e).__name__}: {e}"); return 2


if __name__ == "__main__": sys.exit(main())
