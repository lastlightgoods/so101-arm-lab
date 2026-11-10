"""Download each new checkpoint of the two cloud jobs and score it in sim (50 unseen layouts). Appends to CLOUD_ROUNDS.md."""
import os, re, subprocess, time, json
from huggingface_hub import HfApi, snapshot_download
HOME = os.path.expanduser("~"); SIM = f"{HOME}/so101/sim"; HUB = f"{HOME}/so101/policies_hub"; PY = f"{HOME}/so101/.venv/bin/python"
JOBS = {"act_sim_pick_long": ("6ad877680a35baab53a1b3ca", "tclastlight/act_sim_pick_long", 5000),
        "act_sim_pick_v3":   ("6ad877870a35baab53a1b3d6", "tclastlight/act_sim_pick_v3_2026-10-21_08-27-49", 2500)}
api = HfApi(); done = set(); log = open(f"{SIM}/cloud_eval_driver.log", "a")
def say(*a): print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)
def job_stage(jid):
    try: d = json.loads(subprocess.run([f"{HOME}/so101/.venv/bin/hf", "jobs", "inspect", jid, "--format", "json"], capture_output=True, text=True, timeout=60).stdout); d = d[0] if isinstance(d, list) else d; return d["status"]["stage"]
    except Exception as e: return f"?{e!r:.40}"
def evals_running():  # real python eval processes only (bash -c wrappers also contain the string)
    return len([l for l in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines() if re.search(r"(^|/)(python[\d.]*|Python) eval_sim\.py", l)])
def score(job, repo, step):
    out = f"{HUB}/{job}_{step}"
    snapshot_download(repo, revision=step, allow_patterns=[f"checkpoints/{step}/pretrained_model/**"], local_dir=out)
    pm = f"{out}/checkpoints/{step}/pretrained_model"
    subprocess.run([PY, f"{HOME}/so101/jetson/prep_ckpt.py", pm], capture_output=True)  # cloud image writes keys our 0.6.1 rejects (dtype)
    while evals_running() >= 2: time.sleep(30)
    lg = f"{SIM}/eval_{job}_{step}.log"
    if os.path.exists(lg) and re.search(r"SUCCESS \d+/\d+", open(lg, errors="replace").read()): say("already scored", job, step); return
    say("eval", job, step)
    with open(lg, "w") as f: subprocess.run([PY, "eval_sim.py", "--policy", pm, "--episodes", "50", "--seed", "9100", "--goal-thresh", "0.04", "--overhead-margin", "0.04"], cwd=f"{SIM}/so101-lerobot", stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "MUJOCO_GL": "glfw"})
    m = re.search(r"SUCCESS (\d+)/(\d+) = (\d+)%", open(lg, errors="replace").read()); res = f"{m.group(1)}/{m.group(2)} = {m.group(3)}%" if m else "eval failed"
    say("result", job, step, res)
    with open(f"{SIM}/CLOUD_ROUNDS.md", "a") as f: f.write(f"| {job} | ckpt {step} | {res} | {pm} |\n")
    return res
pass  # table header already written
while True:
    stages = {j: job_stage(v[0]) for j, v in JOBS.items()}; pending = False
    for job, (jid, repo, every) in JOBS.items():
        try: tags = sorted(t.name for t in api.list_repo_refs(repo).tags if t.name.isdigit())
        except Exception as e: say("refs", job, repr(e)[:80]); tags = []
        for t in tags:
            if int(t) % every == 0 and (job, t) not in done:
                done.add((job, t)); score(job, repo, t)
    if all(s in ("COMPLETED", "ERROR", "CANCELED", "DELETED") for s in stages.values()):
        # final sweep for tags that appeared after the last loop
        for job, (jid, repo, every) in JOBS.items():
            for t in sorted(t.name for t in api.list_repo_refs(repo).tags if t.name.isdigit()):
                if int(t) % every == 0 and (job, t) not in done: done.add((job, t)); score(job, repo, t)
        say("all done", stages); break
    time.sleep(120)
