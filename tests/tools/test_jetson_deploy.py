"""tools/jetson_deploy.py against a fake git and a fake Jetson (an in-memory filesystem plus processes). Nothing here can reach a real machine."""
import hashlib, importlib.util, json, os
import pytest

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "jetson_deploy.py")
_s = importlib.util.spec_from_file_location("jetson_deploy", _p); jd = importlib.util.module_from_spec(_s); _s.loader.exec_module(jd)
M = lambda b: hashlib.md5(b).hexdigest()

REPO_FILES = {   # path at the ref -> bytes
    "jetson/camd.py": b"import os\n# camd v2\n", "jetson/demo_cams.py": b"import camd\n", "jetson/demo_recorder.py": b"x = 2\n",
    "jetson/demo_recorderd.py": b"import demo_recorder as dr\nimport lab_config as lc\ndef f():\n    import demo_cams\n", "jetson/lab_config.py": b"y = 1\n",
    "jetson/pick_seq.py": b"import servo_telemetry\n", "jetson/servo_telemetry.py": b"t = 1\n", "jetson/arm_state.py": b"import servo_telemetry\n", "jetson/goto.py": b"g = 1\n",
    "jetson/env.sh": b"# env v2\n", "jetson/hotspot.sh": b"#!/bin/sh\n", "jetson/99-so101-cams.rules": b"rule\n", "jetson/JETSON_RUNBOOK.md": b"doc\n",
    "jetson/calib/top_table.json": b"{}\n", "plugins/lerobot-robot-busheal/lerobot_robot_busheal/__init__.py": b"p = 2\n", "plugins/lerobot-camera-avf/x.py": b"mac only\n",
    "plugins/lerobot-robot-busheal/lerobot_robot_busheal/__pycache__/a.pyc": b"\0",
}
MODES = {"jetson/hotspot.sh": "100755", "jetson/env.sh": "100644"}


class FakeGit:
    def __init__(self, files=None, dirty=()): self.files = dict(files or REPO_FILES); self.dirty_list = list(dirty); self.shown = []
    def rev(self, ref): return {"HEAD": "a" * 40, "main": "a" * 40, "old": "b" * 40}.get(ref, ref.ljust(40, "0"))
    def ls(self, ref): return [(MODES.get(p, "100644"), p) for p in sorted(self.files)]
    def show(self, ref, path): self.shown.append((ref, path)); return self.files[path]
    def dirty(self): return list(self.dirty_list)


class FakeJetson:
    """Implements the transport interface in memory. `ps` is the process list; ctl is what `demo_recorderd.py ctl status` answers (None = no answer)."""
    def __init__(self, fs=None, ps=(), flags=(), ctl=None):
        self.fs = dict(fs or {}); self.ps = list(ps); self.flag_set = list(flags); self.ctl = ctl; self.calls = []; self.restarted = []; self.corrupt = set(); self.restart_ok = True
    def listing(self): return {k: M(v) for k, v in self.fs.items() if "/" not in k and k.endswith((".py", ".sh"))}
    def md5s(self, rels): return {r: (M(self.fs[r]) if r in self.fs else None) for r in rels}
    def read(self, rel): return self.fs.get(rel)
    def flags(self): return list(self.flag_set)
    def procs(self): return list(self.ps)
    def ctl_status(self): return self.ctl
    def backup(self, bakdir, rels):
        self.calls.append(("backup", bakdir, tuple(rels)))
        for r in rels:
            if r in self.fs: self.fs[f"{bakdir}/{r}"] = self.fs[r]
    def put(self, rel, data, mode="644"):
        self.calls.append(("put", rel, mode))
        self.fs[rel + ".deploy_tmp"] = data   # atomic: written beside, then moved
        self.fs[rel] = b"CORRUPT" if rel in self.corrupt else self.fs.pop(rel + ".deploy_tmp")
    def restore(self, bakdir, rel): self.calls.append(("restore", rel)); self.fs[rel] = self.fs[f"{bakdir}/{rel}"]
    def remove(self, rel): self.calls.append(("remove", rel)); self.fs.pop(rel, None)
    def backups(self): return sorted({k.split("/")[1] for k in self.fs if k.startswith("deploy_bak/")})
    def restart(self, name): self.restarted.append(name); return self.restart_ok, "restarted"


def jetson_at(git, **kw):
    """A Jetson holding exactly the deployable files of the fake repo."""
    fs = {}
    for mode, p in git.ls("HEAD"):
        c = jd.classify(mode, p)
        if c and c[1] == "deploy": fs[c[0]] = git.files[p]
    return FakeJetson(fs, **kw)


def run(argv, git, tr, mac=(), out=None):
    lines = [] if out is None else out
    rc = jd.main(argv, git=git, tr=tr, mac_procs=lambda: list(mac), out=lines.append)
    return rc, "\n".join(lines)


def test_manifest_scope():
    m = jd.manifest(FakeGit(), "HEAD")
    kinds = {r: i.kind for r, i in m.items()}
    assert kinds["camd.py"] == kinds["env.sh"] == "deploy"
    assert kinds["hotspot.sh"] == kinds["99-so101-cams.rules"] == "system" and kinds["calib/top_table.json"] == "calib"
    assert "JETSON_RUNBOOK.md" not in m and not any("avf" in r or r.endswith(".pyc") for r in m)
    assert kinds["plugins/lerobot-robot-busheal/lerobot_robot_busheal/__init__.py"] == "deploy" and m["hotspot.sh"].mode == "755"


def test_status_categories_and_exit_code():
    git = FakeGit(); tr = jetson_at(git)
    tr.fs["camd.py"] = b"old"; del tr.fs["goto.py"]; tr.fs["ping_arm.py"] = b"only here"
    rc, out = run(["status"], git, tr)
    assert rc == 1
    assert "DIFFERS  camd.py" in out and "MISSING  goto.py" in out and "ONLY-ON-JETSON ping_arm.py" in out and "BEHIND" in out
    assert "no deploy recorded" in out


def test_status_at_behind_dirty():
    git = FakeGit(); tr = jetson_at(git)
    rc, out = run(["status"], git, tr); assert rc == 0 and "AT aaaaaaaaaa" in out   # all match, no record: at
    run(["deploy", "--yes"], git, tr); assert run(["status"], git, tr)[0] == 0   # still at after a deploy (nothing to do)
    tr.fs["camd.py"] = b"old"
    assert "BEHIND" in run(["status"], git, tr)[1]
    # a recorded deploy, then an edit on the Jetson -> dirty even though the file may still equal REF's other files
    tr = jetson_at(git); jd.put_record(tr, jd.now_record(git, tr, jd.manifest(git, "a" * 40), "a" * 40, "HEAD"))
    assert run(["status"], git, tr)[0] == 0
    tr.fs["goto.py"] = b"hand edit"
    rc, out = run(["status"], git, tr); assert rc == 1 and "DIRTY" in out and "goto.py" in out


def test_deploy_is_dry_run_by_default():
    git = FakeGit(); tr = jetson_at(git); tr.fs["camd.py"] = b"old"
    before = dict(tr.fs)
    rc, out = run(["deploy"], git, tr)
    assert rc == 0 and "dry run" in out and "differs  camd.py" in out and tr.fs == before and not tr.calls


def test_deploy_backup_atomic_verify_record():
    git = FakeGit(); tr = jetson_at(git); tr.fs["camd.py"] = b"old"; del tr.fs["goto.py"]
    rc, out = run(["deploy", "--yes"], git, tr)
    assert rc == 0 and "md5 verified" in out
    assert tr.fs["camd.py"] == REPO_FILES["jetson/camd.py"] and tr.fs["goto.py"] == REPO_FILES["jetson/goto.py"]
    bak = [k for k in tr.fs if k.startswith("deploy_bak/") and k.endswith("/camd.py")][0]
    assert tr.fs[bak] == b"old" and bak.split("/")[1].endswith("_aaaaaaa")
    assert not any(k.endswith(".deploy_tmp") for k in tr.fs)   # tmp moved into place
    rec = json.loads(tr.fs["DEPLOYED.json"]); assert rec["commit"] == "a" * 40 and rec["files"]["camd.py"] == M(REPO_FILES["jetson/camd.py"]) and not rec["partial"]
    assert ("backup", bak.rsplit("/", 1)[0], ("camd.py",)) == tr.calls[0]   # only replaced files are backed up; goto.py was new
    assert run(["status"], git, tr)[0] == 0


def test_deploy_md5_mismatch_is_reported_and_record_not_written():
    git = FakeGit(); tr = jetson_at(git); tr.fs["camd.py"] = b"old"; tr.corrupt.add("camd.py")
    rc, out = run(["deploy", "--yes"], git, tr)
    assert rc == 2 and "VERIFY FAILED" in out and "DEPLOYED.json" not in tr.fs and "rollback" in out


def test_deploy_uses_committed_blob_not_working_tree():
    git = FakeGit(); tr = jetson_at(git); tr.fs["camd.py"] = b"old"
    run(["deploy", "--yes", "--ref", "old"], git, tr)
    assert ("b" * 40, "jetson/camd.py") in git.shown and all(r == "b" * 40 for r, _ in git.shown)   # always `git show <sha>:path`


def test_refuses_dirty_tree_unless_ref_given():
    git = FakeGit(dirty=[" M jetson/camd.py"]); tr = jetson_at(git); tr.fs["camd.py"] = b"old"
    rc, out = run(["deploy", "--yes"], git, tr)
    assert rc == 2 and "uncommitted" in out and tr.fs["camd.py"] == b"old"
    assert run(["deploy", "--yes", "--ref", "main"], git, tr)[0] == 0 and tr.fs["camd.py"] != b"old"
    assert run(["deploy"], FakeGit(dirty=["x"]), jetson_at(git))[0] == 0   # a dry run only looks, so it may run on a dirty tree


@pytest.mark.parametrize("kw,needle", [
    (dict(flags=[".arms_busy"]), ".arms_busy"), (dict(flags=[".cams_busy"]), ".cams_busy"),
    (dict(ps=["123 python pick_seq.py --serve --idle-s 60"]), "pick_seq"),
    (dict(ps=["124 /home/x/.venv/bin/python /usr/bin/lerobot-record --robot.type=so101"]), "lerobot"),
    (dict(ps=["125 /home/x/.venv/bin/python /home/x/so101/demo_recorderd.py serve"], ctl={"ok": True, "configured": True, "episode": {"n": 3}}), "open episode"),
    (dict(ps=["125 /home/x/.venv/bin/python /home/x/so101/demo_recorderd.py serve"], ctl=None), "did not answer"),
])
def test_refuses_when_jetson_busy(kw, needle):
    git = FakeGit(); tr = jetson_at(git, **kw); tr.fs["camd.py"] = b"old"
    rc, out = run(["deploy", "--yes"], git, tr)
    assert rc == 2 and "REFUSED" in out and needle in out and tr.fs["camd.py"] == b"old" and not tr.calls
    assert "would be refused" in run(["deploy"], git, tr)[1]   # the dry run tells you


def test_refuses_when_tidy_or_autoloop_run_on_the_mac():
    git = FakeGit()
    for p in ("python tools/tidy.py run", "/x/python tools/autoloop.py --cycles 10"):
        tr = jetson_at(git); tr.fs["camd.py"] = b"old"
        rc, out = run(["deploy", "--yes"], git, tr, mac=[p]); assert rc == 2 and "on the Mac" in out and tr.fs["camd.py"] == b"old"


def test_idle_recorder_and_cam_taps_do_not_block():
    git = FakeGit(); ps = ["9 /v/python /h/demo_recorderd.py serve", "8 python3 /h/camd.py serve", "7 python3 /h/camd.py tap top", "6 bash -c while [ -f x ]; do :; done; camd.py tap top; pick_seq.py", "5 python /h/arm_state.py"]
    tr = jetson_at(git, ps=ps, ctl={"ok": True, "configured": False, "episode": None}); tr.fs["camd.py"] = b"old"
    assert run(["deploy", "--yes"], git, tr)[0] == 0


def test_unreachable_jetson_is_a_refusal_not_a_crash():
    git = FakeGit(); tr = jetson_at(git)
    tr.flags = lambda: (_ for _ in ()).throw(RuntimeError("ssh exit 255"))
    tr.fs["camd.py"] = b"old"; rc, out = run(["deploy", "--yes"], git, tr)
    assert rc == 2 and "cannot read the Jetson state" in out and tr.fs["camd.py"] == b"old"


def test_system_and_calib_need_both_flags():
    git = FakeGit(); tr = jetson_at(git)
    for f in ("99-so101-cams.rules", "hotspot.sh", "calib/top_table.json"):
        rc, out = run(["deploy", "--yes", "--files", f], git, tr); assert rc == 2 and "--allow-system" in out
    assert "99-so101-cams.rules" not in tr.fs and "calib/top_table.json" not in tr.fs
    assert "99-so101-cams.rules" not in run(["deploy", "--yes"], git, tr)[1]   # a bare deploy never plans them
    assert "99-so101-cams.rules" not in tr.fs
    rc, out = run(["deploy", "--yes", "--allow-system", "--files", "99-so101-cams.rules"], git, tr)
    assert rc == 0 and "sudo install" in out and "udevadm" in out and "will not" in out
    assert tr.fs["99-so101-cams.rules"] == b"rule\n" and not any(c[0] == "put" and c[1].startswith("/etc") for c in tr.calls)   # staged in ~/so101 only; nothing ran with sudo
    rc, out = run(["deploy", "--yes", "--allow-system", "--files", "calib/top_table.json"], git, tr)
    assert rc == 0 and "source of truth" in out and tr.fs["calib/top_table.json"] == b"{}\n"


def test_files_subset_marks_record_partial_and_unknown_file_errors():
    git = FakeGit(); tr = jetson_at(git); tr.fs["camd.py"] = b"old"; tr.fs["goto.py"] = b"old"
    rc, out = run(["deploy", "--yes", "--files", "camd.py"], git, tr)
    assert rc == 0 and tr.fs["goto.py"] == b"old" and json.loads(tr.fs["DEPLOYED.json"])["partial"] is True
    assert run(["deploy", "--yes", "--files", "nope.py"], git, tr)[0] == 2


def test_per_file_restart_guidance():
    git = FakeGit(); tr = jetson_at(git)
    for f in ("camd.py", "demo_recorder.py", "servo_telemetry.py", "env.sh", "goto.py"): tr.fs[f] = b"old"
    rc, out = run(["deploy", "--yes"], git, tr)
    assert "restart needed: camd.py serve" in out and "restart needed: demo_recorderd.py serve" in out   # camd.py reaches the recorder through demo_cams
    assert "pick_seq.py --serve" in out and "arm_state.py" in out and "source ~/so101/env.sh" in out and "goto.py (run on demand)" in out
    assert tr.restarted == []   # never automatic


def test_restart_recorder_only_idle():
    git = FakeGit(); rps = ["9 /v/python /h/demo_recorderd.py serve"]
    tr = jetson_at(git, ps=rps, ctl={"ok": True, "configured": True, "episode": None}); tr.fs["demo_recorder.py"] = b"old"
    rc, out = run(["deploy", "--yes", "--restart", "recorder"], git, tr); assert rc == 0 and tr.restarted == ["recorder"]   # configured but idle is fine
    tr = jetson_at(git, ps=rps, ctl={"ok": True, "configured": True, "episode": {"n": 1}}); tr.fs["demo_recorder.py"] = b"old"
    assert run(["deploy", "--yes", "--restart", "recorder"], git, tr)[0] == 2 and tr.restarted == [] and tr.fs["demo_recorder.py"] == b"old"   # busy check refuses first
    tr = jetson_at(git, ps=[], ctl=None); tr.fs["demo_recorder.py"] = b"old"
    rc, out = run(["deploy", "--yes", "--restart", "recorder"], git, tr); assert rc == 2 and "not running" in out and tr.restarted == []
    tr = jetson_at(git, ps=rps, ctl={"ok": True, "configured": False, "episode": None}); tr.fs["goto.py"] = b"old"
    rc, out = run(["deploy", "--yes", "--restart", "recorder"], git, tr); assert "none of the deployed files" in out and tr.restarted == []


def test_restart_camd_only_without_configured_recorder():
    git = FakeGit(); cps = ["8 python3 /h/camd.py serve"]
    tr = jetson_at(git, ps=cps + ["9 /v/python /h/demo_recorderd.py serve"], ctl={"ok": True, "configured": True, "episode": None}); tr.fs["camd.py"] = b"old"
    rc, out = run(["deploy", "--yes", "--restart", "camd"], git, tr); assert rc == 2 and "recorder is configured" in out and tr.restarted == []
    tr = jetson_at(git, ps=cps + ["9 /v/python /h/demo_recorderd.py serve"], ctl={"ok": True, "configured": False, "episode": None}); tr.fs["camd.py"] = b"old"
    assert run(["deploy", "--yes", "--restart", "camd"], git, tr)[0] == 0 and tr.restarted == ["camd"]
    tr = jetson_at(git, ps=cps); tr.fs["camd.py"] = b"old"
    assert run(["deploy", "--yes", "--restart", "camd"], git, tr)[0] == 0 and tr.restarted == ["camd"]
    with pytest.raises(SystemExit): jd.main(["deploy", "--restart", "arm_state"], git=git, tr=tr)   # nothing else is on the whitelist


def test_rollback_restores_newest_and_updates_record():
    git = FakeGit(); tr = jetson_at(git); tr.fs["camd.py"] = b"old"; del tr.fs["goto.py"]
    run(["deploy", "--yes"], git, tr)
    rc, out = run(["rollback"], git, tr); assert rc == 0 and "dry run" in out and tr.fs["camd.py"] != b"old"
    rc, out = run(["rollback", "--yes"], git, tr)
    assert rc == 0 and tr.fs["camd.py"] == b"old" and "goto.py" not in tr.fs   # added file removed because it still holds what deploy wrote
    rec = json.loads(tr.fs["DEPLOYED.json"]); assert rec["rolled_back_from"] == "a" * 40 and rec["files"]["camd.py"] == M(b"old")


def test_rollback_keeps_added_file_edited_since_and_checks_busy():
    git = FakeGit(); tr = jetson_at(git); del tr.fs["goto.py"]; tr.fs["camd.py"] = b"old"
    run(["deploy", "--yes"], git, tr); tr.fs["goto.py"] = b"edited after"
    tr.flag_set = [".cams_busy"]
    rc, out = run(["rollback", "--yes"], git, tr); assert rc == 2 and tr.fs["camd.py"] != b"old"
    tr.flag_set = []
    assert run(["rollback", "--yes"], git, tr)[0] == 0 and tr.fs["goto.py"] == b"edited after"
    assert run(["rollback", "--to", "no-such", "--yes"], git, tr)[0] == 2


def test_deployed_matches_for_preflight():
    git = FakeGit(); tr = jetson_at(git)
    ok, why = jd.deployed_matches("main", transport=tr, git=git); assert ok and "AT aaaa" in why
    tr.fs["camd.py"] = b"old"
    ok, why = jd.deployed_matches("main", transport=tr, git=git); assert not ok and "camd.py" in why
    class Down(FakeJetson):
        def listing(self): raise RuntimeError("ssh exit 255")
    ok, why = jd.deployed_matches("main", transport=Down(), git=git); assert not ok and "cannot compare" in why


def test_ssh_transport_builds_safe_commands():
    seen = []
    t = jd.SshTransport(run=lambda argv, data: (seen.append((argv, data)) or (0, b"")))
    t.put("a/b.py", b"data", "755")
    argv, data = seen[-1]; assert argv[:2] == ["ssh", "-o"] and "jetson" in argv and data == b"data" and "mv -f" in argv[-1] and "chmod 755" in argv[-1]
    for bad in ("../x", "/etc/passwd", ""):
        with pytest.raises(ValueError): t.put(bad, b"")
    t.md5s(["a b.py; rm -rf x"]); assert "'a b.py; rm -rf x'" in seen[-1][0][-1]


def test_no_lan_addresses_or_hosts_in_the_tool():
    import re
    src = open(_p).read()
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src) and "/Users/" not in src and "@" not in src.replace("\"@\"", "")


def test_restart_scripts_never_match_their_own_shell():
    """2026-11-11: `bash -c '... pkill -f "demo_recorderd.py serve" ...'` matches (and kills) its own shell; the patterns must be bracketed."""
    import re
    scripts = {}
    for name in ("recorder", "camd"):
        seen = []; tr = jd.SshTransport(run=lambda argv, data, _s=seen: (_s.append(argv), (0, b"restarted"))[1]); tr.restart(name); scripts[name] = seen[-1][-1]
    for name, daemon in (("recorder", "/home/x/.venv/bin/python /home/x/so101/demo_recorderd.py serve"), ("camd", "python3 /home/x/so101/camd.py serve")):
        sc = scripts[name]; pats = re.findall(r'(?:pkill -TERM|pgrep) -f "([^"]+)"', sc); assert pats, sc
        for pat in pats: assert re.search(pat, daemon) and not re.search(pat, "bash -c " + sc), (pat, name)
