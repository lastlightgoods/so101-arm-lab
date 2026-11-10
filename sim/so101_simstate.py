"""Publishes the MuJoCo SO-101 simulator's state for the lab dashboard (auto-loaded via so101_simstate.pth).
After so101_nexus.mujoco.base_env is imported, wraps the env's step()/reset() to write /tmp/so101_sim_state.json
(<= 30 Hz): joints in LeRobot dataset units (deg, gripper 0-100), object/target/tcp poses, success, episode, mode."""
import sys, os, json, time, importlib.abc, importlib.machinery
TARGET = "so101_nexus.mujoco.base_env"; PATH = "/tmp/so101_sim_state.json"
def _patch(mod):
    Env = getattr(mod, "SO101NexusMuJoCoBaseEnv", None)
    if Env is None or getattr(Env, "_so101_state_patched", False): return
    import numpy as np
    try: from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row
    except Exception: sim_qpos_to_dataset_row = None
    mode = os.path.basename(sys.argv[0] or "sim").replace(".py", ""); state = {"ep": 0, "t": 0.0, "last": 0.0}
    def dump(self, obs, info, step):
        now = time.time()
        if now - state["last"] < 1 / 30: return
        state["last"] = now
        try:
            q = obs.get("state") if isinstance(obs, dict) else None
            if q is None or sim_qpos_to_dataset_row is None: return
            lim = (float(self.action_space.low[-1]), float(self.action_space.high[-1]))
            row = sim_qpos_to_dataset_row(np.asarray(q, dtype=np.float64), gripper_limits_rad=lim)
            names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            d = {"t": now, "mode": mode, "pid": os.getpid(), "ep": state["ep"], "step": step, "pos": {n: float(v) for n, v in zip(names, row)},
                 "success": bool(info.get("success", False)) if isinstance(info, dict) else False, "grasped": bool(info.get("is_grasped", 0)) if isinstance(info, dict) else False}
            for key, fn in (("obj", "_get_object_pose"), ("target", "_get_target_pos"), ("tcp", "_get_tcp_pose")):
                f = getattr(self, fn, None)
                if f:
                    try: d[key] = [float(x) for x in np.asarray(f()).ravel()[:3]]
                    except Exception: pass
            tmp = PATH + ".tmp"; open(tmp, "w").write(json.dumps(d)); os.replace(tmp, PATH)
        except Exception: pass
    _step, _reset = Env.step, Env.reset
    def step(self, action, *a, **k):
        out = _step(self, action, *a, **k); state["t"] += 1
        try: dump(self, out[0], out[4] if len(out) > 4 else {}, int(state["t"]))
        except Exception: pass
        return out
    def reset(self, *a, **k):
        out = _reset(self, *a, **k); state["ep"] += 1; state["t"] = 0
        try: dump(self, out[0], out[1] if len(out) > 1 else {}, 0)
        except Exception: pass
        return out
    Env.step, Env.reset, Env._so101_state_patched = step, reset, True
class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET: return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec and spec.loader:
            ex = spec.loader.exec_module
            def exec_module(mod): ex(mod); _patch(mod)
            spec.loader.exec_module = exec_module
        return spec
import importlib.util
sys.meta_path.insert(0, _Finder())
