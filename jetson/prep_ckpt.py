"""Make a LeRobot checkpoint written by a newer LeRobot loadable by this machine's LeRobot (0.6.1):
drop config.json keys the local policy config dataclass does not know (e.g. `dtype`), keeping config.orig.json.
For SmolVLA also stage weights on CPU and skip the base-VLM download (8 GB Jetson). Idempotent.
    python prep_ckpt.py <pretrained_model dir> [...]"""
import json, sys, os, dataclasses, importlib
CLASSES = {"act": ("lerobot.policies.act.configuration_act", "ACTConfig"), "smolvla": ("lerobot.policies.smolvla.configuration_smolvla", "SmolVLAConfig"),
           "diffusion": ("lerobot.policies.diffusion.configuration_diffusion", "DiffusionConfig"), "pi0": ("lerobot.policies.pi0.configuration_pi0", "PI0Config")}
for d in sys.argv[1:]:
    p = os.path.join(d, "config.json"); c = json.load(open(p)); kind = c.get("type", "act")
    mod, name = CLASSES.get(kind, CLASSES["act"]); cls = getattr(importlib.import_module(mod), name)
    known = {f.name for f in dataclasses.fields(cls)} | {"type"}
    if not os.path.exists(os.path.join(d, "config.orig.json")): json.dump(c, open(os.path.join(d, "config.orig.json"), "w"), indent=2)
    dropped = [k for k in c if k not in known]
    for k in dropped: c.pop(k)
    if kind == "smolvla": c["device"] = "cpu"; c["load_vlm_weights"] = False
    json.dump(c, open(p, "w"), indent=2); print(f"{d}: {kind}, dropped {dropped}" + (", device=cpu, load_vlm_weights=false" if kind == "smolvla" else ""))
