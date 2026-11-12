#!/usr/bin/env python3
"""learn_loop_train.py - where `learn_loop.py train` runs (docs/learn-loop.md). TC, 11 Nov 2026: training is on Hugging Face or on the Jetson, never on the Mac.

  hf      a Hugging Face Job on an A100: push the dataset to a PRIVATE Hub dataset repo, `hf jobs run --flavor a100-large ... lerobot-train ... --policy.push_to_hub=true`, poll
          `hf jobs inspect`, `hf download` the model repo, jetson/prep_ckpt.py on the result. Costs money: nothing is uploaded or launched without --yes.
  jetson  lerobot-train on the Orin Nano over ssh (CUDA, batch 4, ~1.9 step/s), detached with nohup and polled; refuses while the arm, the cameras, a recorder episode or another
          job is in use; reads the dataset where the recorder wrote it; resumes from checkpoints/last; the checkpoint is rsynced back for the registry.
  local   lerobot-train on this machine: --dry-run only (the fakes), the Mac is not a training target.

Every external call goes through an injectable seam (the Loop's `sh`/`capture` for hf and local commands, `jetson.run(script)` for ssh, `fetch(src, dst)` for rsync), so tests use fakes.
"""
import json, math, os, re, shlex, subprocess, sys

# why: measured. HF: the sim cloud rounds (sim/CLOUD_ROUNDS.md, 21 Oct 2026) trained ACT at ~3.1 step/s, batch 32 on a100-large at $2.50/h; the SmolVLA job of 16 Oct took 61 min for 10k steps (~$2.55).
# Jetson: the so101-finetune header in jetson/env.sh, ~1.9 step/s at batch 4, which is its ceiling on the shared 8 GB (NEXT_STEPS.md, 15 Oct 2026).
HF_RATE = {"flavor": "a100-large", "steps_per_s": 3.1, "batch": 32, "usd_per_h": 2.50, "startup_s": 300.0}   # startup (image pull, dataset download) is an estimate, never measured
JETSON_RATE = {"steps_per_s": 1.9, "batch": 4}
IMAGE = "huggingface/lerobot-gpu"   # the LeRobot GPU image; the repo does not record the image of the 16-21 Oct jobs, so this is the one thing to confirm with `hf jobs run --dry-run`
DEFAULTS = {"hf": {"steps": 10000, "batch": 32, "workers": 4, "save_freq": 2500}, "jetson": {"steps": 20000, "batch": 4, "workers": 1, "save_freq": 2000}, "local": {"steps": 1000, "batch": 8, "workers": 2, "save_freq": 500}}
TERMINAL_BAD = ("ERROR", "CANCELED", "CANCELLED", "DELETED", "TIMEOUT", "FAILED")


def pick(a, key): return getattr(a, key) if getattr(a, key, None) is not None else DEFAULTS[a.where][key]


def plan_numbers(a):
    """{wall_s, cost_usd, timeout, ...} for the chosen backend from the measured rates above."""
    steps = pick(a, "steps")
    if a.where == "hf":
        r = HF_RATE; sps = r["steps_per_s"]; wall = steps / sps + r["startup_s"]; hours = max(1, math.ceil(1.5 * wall / 3600))
        return {"where": "hf", "flavor": r["flavor"], "steps": steps, "batch": pick(a, "batch"), "steps_per_s": sps, "wall_s": wall, "usd_per_h": r["usd_per_h"], "cost_usd": round(wall / 3600 * r["usd_per_h"], 2), "timeout": f"{hours}h", "max_cost_usd": round(hours * r["usd_per_h"], 2)}
    if a.where == "jetson":
        sps = JETSON_RATE["steps_per_s"] * JETSON_RATE["batch"] / pick(a, "batch"); return {"where": "jetson", "steps": steps, "batch": pick(a, "batch"), "steps_per_s": round(sps, 2), "wall_s": steps / sps, "cost_usd": 0.0}
    return {"where": "local", "steps": steps, "batch": pick(a, "batch"), "steps_per_s": None, "wall_s": 0.0, "cost_usd": 0.0}


def plan_text(p):
    h = lambda s: f"{s / 3600:.1f} h" if s >= 5400 else f"{s / 60:.0f} min"
    if p["where"] == "hf": return f"HF Job {p['flavor']}: {p['steps']} steps at batch {p['batch']}, {p['steps_per_s']} step/s measured (sim/CLOUD_ROUNDS.md) + 5 min startup (estimate) = {h(p['wall_s'])}, about ${p['cost_usd']:.2f} at ${p['usd_per_h']:.2f}/h; timeout {p['timeout']} caps it at ${p['max_cost_usd']:.2f}"
    if p["where"] == "jetson": return f"Jetson CUDA: {p['steps']} steps at batch {p['batch']}, {p['steps_per_s']} step/s measured (so101-finetune, jetson/env.sh) = {h(p['wall_s'])}, no cost; holds the arm's computer for that long"
    return "local (dry-run fakes only)"


# ---------------------------------------------------------------- ssh
class SshShell:
    """run(script, timeout) -> (rc, stdout+stderr): `ssh <host> bash -s` with the script on stdin. The real seam; tests inject FakeJetson (tools/learn_loop_fakes.py)."""
    def __init__(self, host, run=subprocess.run): self.host, self._run = host, run
    def run(self, script, timeout=120):
        r = self._run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.host, "bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout); return r.returncode, (r.stdout or "") + (r.stderr or "")


def rsync_fetch(src, dst, run=subprocess.run):
    """rsync src/ -> dst/ (either side may be host:path); returns (rc, stderr tail)."""
    os.makedirs(dst, exist_ok=True); r = run(["rsync", "-a", "--delete", src.rstrip("/") + "/", dst.rstrip("/") + "/"], capture_output=True, text=True, timeout=3600); return r.returncode, (r.stderr or "").strip()[-300:]


BUSY_SCRIPT = r'''# ll:busy
for f in "$HOME/so101/.arms_busy" "$HOME/so101/.cams_busy"; do [ -e "$f" ] && echo "BUSY $(basename "$f") exists"; done
pgrep -af 'lerobot-(teleoperate|record|rollout|calibrate|replay|train)' | grep -v pgrep | sed 's/^/BUSY lerobot job: /' | cut -c1-160
pgrep -af 'pick_seq.py.*--serve' | grep -v pgrep | sed 's/^/BUSY pick_seq session: /' | cut -c1-160
pgrep -af 'so101/(dance|demo_wave|nudge|goto|pick_seq|scripted_record|scan_pose|free_arm)\.py' | grep -v pgrep | sed 's/^/BUSY arm script: /' | cut -c1-160
if [ -S /dev/shm/so101_demorec.sock ] && [ -f "$HOME/so101/demo_recorderd.py" ]; then
  s=$("$HOME/so101/.venv/bin/python" "$HOME/so101/demo_recorderd.py" ctl ping 2>/dev/null) && echo "$s" | grep -q '"configured": true' && ! echo "$s" | grep -q '"episode": null' && echo "BUSY recorder has an open episode"
fi
true
'''


def q(x):
    """shell-quote one argv token; a token holding $HOME stays double-quoted so the Jetson expands it."""
    x = str(x)
    return '"' + x.replace('"', '\\"') + '"' if "$HOME" in x and not re.search(r"[`'\\]", x) else shlex.quote(x)


def train_script(argv, job):
    """Detached lerobot-train on the Jetson: a run script (env.sh, .cams_busy up to free the 8 GB shared memory as the NEXT_STEPS gotcha says, removed again by the trap) started with nohup;
    the log sits beside the other train logs so the dashboard's loss chart follows it; prints PID."""
    cmd = " ".join(q(x) for x in argv); assert "LLEOF" not in cmd
    return f'''# ll:train-launch
mkdir -p "$HOME/so101/outputs/train"; LOG="$HOME/so101/outputs/train/{job}.log"; RUN="$HOME/so101/outputs/train/{job}.run.sh"
cat > "$RUN" <<'LLEOF'
source "$HOME/so101/env.sh" >/dev/null 2>&1
touch "$HOME/so101/.cams_busy"; trap 'rm -f "$HOME/so101/.cams_busy"' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{cmd}
LLEOF
nohup bash "$RUN" >> "$LOG" 2>&1 &
echo "PID $!"
'''


def status_script(job):
    return f'''# ll:train-status
D="$HOME/so101/outputs/train/{job}"; L="$D/checkpoints/last"
if pgrep -f 'lerobot-train.*--job_name={job}|lerobot-train.*{job}/checkpoints' >/dev/null; then echo RUNNING 1; else echo RUNNING 0; fi
[ -L "$L" ] && echo "LAST $(basename "$(readlink -f "$L")")" || echo "LAST none"
echo ---LOG; tr '\\r' '\\n' < "$HOME/so101/outputs/train/{job}.log" 2>/dev/null | grep -a -E "step:|step/s" | tail -n 400
'''


def parse_status(out):
    running = bool(re.search(r"^RUNNING 1", out, re.M)); m = re.search(r"^LAST (\S+)", out, re.M); last = int(m.group(1)) if m and m.group(1).isdigit() else 0
    return running, last, out.split("---LOG", 1)[1] if "---LOG" in out else ""


# ---------------------------------------------------------------- backends
class Backend:
    def __init__(self, L): self.L, self.a = L, L.a
    def fail(self, msg, code=1): raise self.L.StageError(msg, code)


class HfBackend(Backend):
    """Dataset to the Hub, a Job, a model repo back. State kept in state.json ids: hf_job (so a repeated command polls instead of paying twice)."""
    def names(self, rec, job):
        user = self.a.hf_user; return f"{user}/{rec['repo_id'].split('/')[-1]}-{rec['dataset_hash'][:8]}", f"{user}/{job}"

    def job_cmd(self, cfg, ds_repo, model_repo, job):
        a = self.a; steps = pick(a, "steps")
        inner = ["lerobot-train", f"--dataset.repo_id={ds_repo}", "--policy.type=act", f"--policy.chunk_size={cfg['chunk_size']}", f"--policy.n_action_steps={cfg['chunk_size']}", f"--policy.input_features={json.dumps(cfg['input_features'], separators=(',', ':'))}",
                 "--policy.device=cuda", f"--policy.repo_id={model_repo}", "--policy.push_to_hub=true", "--output_dir=/tmp/out", f"--job_name={job}", f"--steps={steps}", f"--batch_size={pick(a, 'batch')}", f"--num_workers={pick(a, 'workers')}",
                 f"--save_freq={pick(a, 'save_freq')}", "--log_freq=100", "--wandb.enable=false"]
        if cfg["policy"] == "smolvla":
            inner = [x for x in inner if not x.startswith(("--policy.type", "--policy.chunk_size", "--policy.n_action_steps", "--policy.input_features"))] + ["--policy.path=lerobot/smolvla_base", "--policy.empty_cameras=1",
                     '--rename_map={"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}']   # the recipe of NEXT_STEPS.md, 16 Oct 2026
        return ["hf", "jobs", "run", "--flavor", HF_RATE["flavor"], "--timeout", plan_numbers(a)["timeout"], "--detach", "--secrets", "HF_TOKEN", "--name", job, "--label", f"learn_loop={a.name}", a.hf_image, "bash", "-c", " ".join(shlex.quote(x) for x in inner)]

    def gate(self, rec, job):
        """Refuse to spend without --yes (a job already launched may be polled without it). Called before anything is created."""
        a = self.a
        if self.L.state["ids"].get("hf_job") or a.yes or a.dry_run: return
        ds_repo, model_repo = self.names(rec, job); self.fail(f"not launching: this spends money on the Hub account. {plan_text(plan_numbers(a))}. Repeat with --yes to upload the dataset to the private repo {ds_repo} and start the job (the model comes back from {model_repo}).")

    def run(self, cfg, rec, job, run_dir, log):
        L, a = self.L, self.a; ids = L.state["ids"]; ds_repo, model_repo = self.names(rec, job); p = plan_numbers(a); jid = ids.get("hf_job")
        if jid is None:
            self.gate(rec, job)
            root = L.local_dataset(rec)
            for argv in (["hf", "upload", ds_repo, root, ".", "--repo-type", "dataset", "--private", "--commit-message", f"learn-loop {a.name} dataset {rec['dataset_hash'][:12]}"], ["hf", "repos", "tag", "create", ds_repo, "v3.0", "--repo-type", "dataset"]):
                rc, out = L.capture(argv)
                if rc and not (argv[2:4] == ["tag", "create"] and "exist" in out.lower()): self.fail(f"{' '.join(argv[:4])} failed ({rc}): {out.strip()[-300:]}")
            rc, out = L.capture(self.job_cmd(cfg, ds_repo, model_repo, job))
            m = re.search(r"\b[0-9a-f]{24}\b", out)
            if rc or not m: self.fail(f"hf jobs run failed ({rc}) or printed no job id: {out.strip()[-300:]}")
            jid = ids["hf_job"] = m.group(0); L.save(); L.out(f"[train] HF job {jid} started; dataset {ds_repo} (private), model repo {model_repo}")
        else: L.out(f"[train] HF job {jid} already launched; polling")
        stage = None
        while True:
            rc, out = L.capture(["hf", "jobs", "inspect", jid, "--format", "json"])
            try: d = json.loads(out); d = d[0] if isinstance(d, list) else d; stage = str(d["status"]["stage"]).upper()
            except (ValueError, KeyError, TypeError, IndexError): stage = None
            if stage == "COMPLETED": break
            if stage in TERMINAL_BAD: self.fail(f"HF job {jid} ended {stage} (hf jobs logs {jid}); the model repo {model_repo} keeps the checkpoints it pushed. Start over with `train --force --yes`.")
            L.sleep(a.poll_s)
        rc, text = L.capture(["hf", "jobs", "logs", jid])
        dst = os.path.join(run_dir, "hf_model")
        rc, out = L.capture(["hf", "download", model_repo, "--local-dir", dst])
        if rc: self.fail(f"hf download {model_repo} failed ({rc}): {out.strip()[-300:]}")
        tool = L.tool(); ck = tool.pick_checkpoint(dst); rc = L.sh([sys.executable, os.path.join(L.root, "jetson", "prep_ckpt.py"), ck], log)   # why: cloud checkpoints carry config keys our lerobot 0.6.1 rejects (`dtype`; sim/CLOUD_ROUNDS.md)
        if rc: self.fail(f"jetson/prep_ckpt.py on {ck} failed ({rc})")
        return {"checkpoint": ck, "log_text": text, "steps_done": int(tool.ckpt_step(ck) or pick(a, "steps")), "extra": {"hf_job": jid, "hf_repo": model_repo, "hf_dataset_repo": ds_repo, "flavor": HF_RATE["flavor"], "est_cost_usd": p["cost_usd"]}}


class JetsonBackend(Backend):
    def preflight(self):
        L = self.L; rc, out = L.jetson.run(BUSY_SCRIPT, 60)
        if rc: self.fail(f"could not check the Jetson ({rc}): {out.strip()[-200:]}")
        busy = [l[5:] for l in out.splitlines() if l.startswith("BUSY ")]
        try: busy += [f"on the Mac: {x}" for x in L.local_procs()]
        except Exception as e: self.fail(f"could not look for a running autoloop/tidy on this machine: {e}")
        if busy: self.fail("refusing to train on the Jetson, it is in use: " + "; ".join(busy[:6]) + ". Stop that first (so101-stop, the stop file), then repeat the command.", code=1)

    def run(self, cfg, rec, job, run_dir, log):
        L, a = self.L, self.a
        if not rec.get("remote_root"): self.fail("this dataset is not on the Jetson (record adopted a Mac folder). Use `train --where hf`, or record with the Jetson recorder, or give --jetson-dataset PATH.")
        out_dir = f"$HOME/so101/outputs/train/{job}"; remote_ds = rec["remote_root"].replace("~/", "$HOME/", 1); ids = L.state["ids"]
        rc, st = L.jetson.run(status_script(job), 60); running, last, _ = parse_status(st)
        if rc: self.fail(f"Jetson status failed ({rc}): {st.strip()[-200:]}")
        steps = pick(a, "steps")
        if running: L.out(f"[train] lerobot-train {job} is already running on the Jetson (step {last}); polling")
        else:
            self.preflight()
            if last >= steps: L.out(f"[train] {job} already reached step {last}")
            else:
                if last: argv = ["lerobot-train", "--resume=true", f"--config_path={out_dir}/checkpoints/last/pretrained_model/train_config.json"]
                else:
                    tc = {"dataset.repo_id": rec["repo_id"], "dataset.root": remote_ds}
                    argv = ["lerobot-train", f"--dataset.repo_id={tc['dataset.repo_id']}", f"--dataset.root={tc['dataset.root']}", "--policy.type=act", f"--policy.chunk_size={cfg['chunk_size']}", f"--policy.n_action_steps={cfg['chunk_size']}",
                            f"--policy.input_features={json.dumps(cfg['input_features'], separators=(',', ':'))}", "--policy.device=cuda", "--policy.push_to_hub=false", f"--output_dir={out_dir}", f"--job_name={job}", f"--steps={steps}",
                            f"--batch_size={pick(a, 'batch')}", f"--num_workers={pick(a, 'workers')}", f"--save_freq={pick(a, 'save_freq')}", "--log_freq=100", "--wandb.enable=false"]
                rc, out = L.jetson.run(train_script(argv, job), 60)
                if rc or "PID" not in out: self.fail(f"could not start lerobot-train on the Jetson ({rc}): {out.strip()[-300:]}")
                ids["jetson_train_pid"] = out.split("PID", 1)[1].split()[0]; L.save(); L.out(f"[train] started on the Jetson, {'resuming from step ' + str(last) if last else 'fresh'}; poll every {a.poll_s:g} s")
        text = ""
        while True:
            rc, st = L.jetson.run(status_script(job), 60)
            if rc: self.fail(f"Jetson status failed ({rc}): {st.strip()[-200:]}; the job keeps running there, repeat the command to poll again")
            running, last, text = parse_status(st)
            if not running: break
            L.sleep(a.poll_s)
        if last < steps: self.fail(f"lerobot-train on the Jetson stopped at step {last} of {steps} (log ~/so101/outputs/train/{job}.log on the Jetson); repeat the command to resume from checkpoints/last")
        local = os.path.join(run_dir, "jetson_model"); rc, err = L.fetch(f"{a.host}:so101/outputs/train/{job}/checkpoints/last/pretrained_model", local)
        if rc: self.fail(f"fetching the checkpoint from the Jetson failed ({rc}): {err}")
        return {"checkpoint": local, "log_text": text, "steps_done": last, "extra": {"jetson_host": a.host, "remote_dataset": remote_ds}}
