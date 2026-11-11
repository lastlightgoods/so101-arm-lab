#!/usr/bin/env python3
"""lab_config.py - the configuration labels that ride on every demo episode (docs/scripted-demos.md, plan doc 51B).

One dataset grows over hardware changes (stock gripper now, a parallel gripper or a rebuilt arm later), so every episode says what it was recorded
with. The labels live in a LabConfig; `env_vector()` turns the part a policy can use into the observation.environment_state feature (constant
within an episode) and `sidecar_labels()` is the part written to meta/lab/episodes.jsonl. Pure Python, no lerobot import, no hardware.

    cfg = LabConfig.from_file("lab_config.json")        # or LabConfig() for the stock-v1 defaults
    cfg.env_vector()                                    # [gripper one-hot..., arm-version one-hot..., jaw_max_mm / 100]
    calib_hashes(calib_dir)                             # {"top_table.json": {"sha12": ..., "mtime": ...}, ...}
    cfg.check_hashes(calib_hashes(calib_dir))           # -> list of mismatches against the hashes the config pins (empty = ok)
"""
import dataclasses, hashlib, json, os

GRIPPERS = ("stock", "robonine_parallel")
ARM_VERSIONS = ("stock", "special_edition_1")
ENV_NAMES = [f"gripper_{g}" for g in GRIPPERS] + [f"arm_{a}" for a in ARM_VERSIONS] + ["jaw_max_mm_over_100"]
CALIB_FILES = ("top_table.json", "top_armfit.json", "wrist_campose.json", "front_campose.json", "cam_settings.json")   # in jetson/calib


def file_hash(path):
    """First 12 hex digits of the file's sha256, or None when the file is missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
        return h.hexdigest()[:12]
    except OSError: return None


def calib_hashes(calib_dir, extra=None):
    """{name: {"sha12", "mtime"}} for the calibration files that exist in `calib_dir`; `extra` = {name: path} adds more (the follower calibration json lives outside the repo)."""
    out = {}
    for name, path in [(n, os.path.join(calib_dir, n)) for n in CALIB_FILES] + list((extra or {}).items()):
        sha = file_hash(path)
        if sha: out[name] = {"sha12": sha, "mtime": int(os.path.getmtime(path))}
    return out


@dataclasses.dataclass
class LabConfig:
    config_id: str = "stock-v1"
    gripper_type: str = "stock"                    # one of GRIPPERS
    gripper_variant: str = "jaw_open_35_close_20_then_0"   # how the scripted pick drives the jaw (pick_loop JAW_OPEN, close steps)
    arm_version: str = "stock"                     # one of ARM_VERSIONS
    robot_id: str = "ll_follower"
    servo_set: str = "STS3215 x6"                  # model per joint; add the supply voltage here when it changes
    camera_mount: str = "wrist-mount-r1"           # wrist mount revision; the hash of calib/ref_*.jpg belongs here once it is recorded
    lighting_tag: str = "untagged"                 # varied BETWEEN sessions, never within one (docs/camera-settings.md)
    jaw_max_mm: float = 0.0                        # caliper-measured jaw gap at full open; 0.0 = not measured yet (plan 8.2)
    pinned_hashes: dict = dataclasses.field(default_factory=dict)   # {file: sha12} the live calibration must match, or {} to pin nothing

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}; extra = set(d) - known
        if extra: raise ValueError(f"unknown lab_config fields: {sorted(extra)}")
        c = cls(**d); c.validate(); return c

    @classmethod
    def from_file(cls, path):
        with open(path) as f: return cls.from_dict(json.load(f))

    def validate(self):
        if self.gripper_type not in GRIPPERS: raise ValueError(f"gripper_type {self.gripper_type!r} not in {GRIPPERS}")
        if self.arm_version not in ARM_VERSIONS: raise ValueError(f"arm_version {self.arm_version!r} not in {ARM_VERSIONS}")
        if not self.config_id or not self.robot_id: raise ValueError("config_id and robot_id must be set")

    def to_dict(self): return dataclasses.asdict(self)

    def env_vector(self):
        """observation.environment_state: gripper one-hot, arm-version one-hot, jaw_max_mm / 100 (len(ENV_NAMES) floats)."""
        return [float(self.gripper_type == g) for g in GRIPPERS] + [float(self.arm_version == a) for a in ARM_VERSIONS] + [float(self.jaw_max_mm) / 100.0]

    def sidecar_labels(self, live_hashes=None):
        """The configuration group of a sidecar row. `live_hashes` = calib_hashes() at the start of the session."""
        d = {k: v for k, v in self.to_dict().items() if k != "pinned_hashes"}
        d["calibration"] = {k: dict(v) for k, v in (live_hashes or {}).items()}; return d

    def check_hashes(self, live):
        """Mismatches between the pinned hashes and `live` (calib_hashes output): ["top_table.json: pinned abc, live def", ...]; [] when all match or nothing is pinned."""
        bad = []
        for name, want in self.pinned_hashes.items():
            got = (live.get(name) or {}).get("sha12")
            if got != want: bad.append(f"{name}: pinned {want}, live {got}")
        return bad
