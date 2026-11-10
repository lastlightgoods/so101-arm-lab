"""Make a SmolVLA checkpoint trained with a newer LeRobot loadable on this machine's LeRobot 0.6.1 and the 8 GB Jetson:
drop config keys the local SmolVLAConfig doesn't know (e.g. dtype), load weights on CPU first (device=cpu; the rollout
moves the model to cuda afterwards) and skip the base VLM download (load_vlm_weights=false; the checkpoint has them).
    python prep_vla_ckpt.py <pretrained_model dir> [...]      (original kept as config.orig.json)"""
import json, sys, os, dataclasses
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
known = {f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}
for d in sys.argv[1:]:
    p = os.path.join(d, "config.json"); c = json.load(open(p))
    if not os.path.exists(os.path.join(d, "config.orig.json")): json.dump(c, open(os.path.join(d, "config.orig.json"), "w"), indent=2)
    dropped = [k for k in c if k not in known]
    for k in dropped: c.pop(k)
    c["device"] = "cpu"; c["load_vlm_weights"] = False
    json.dump(c, open(p, "w"), indent=2); print(f"{d}: dropped {dropped}, device=cpu, load_vlm_weights=false")
