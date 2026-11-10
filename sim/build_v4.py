"""v4 = all of v2 (clean) + the 73 v3 episodes that contain a re-grasp (sim/v3_recovery_episodes.json)."""
import json, shutil, os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
D = os.path.expanduser("~/so101/sim/data"); keep = set(json.load(open(os.path.expanduser("~/so101/sim/v3_recovery_episodes.json"))))
v3 = LeRobotDataset("tclastlight/so101_sim_pick_v3", root=f"{D}/so101_sim_pick_v3")
drop = [e for e in range(v3.meta.total_episodes) if e not in keep]
for d in ("so101_sim_pick_v3_recov", "so101_sim_pick_v4"): shutil.rmtree(f"{D}/{d}", ignore_errors=True)
v3r = delete_episodes(v3, drop, output_dir=f"{D}/so101_sim_pick_v3_recov", repo_id="tclastlight/so101_sim_pick_v3_recov")
print("v3_recov episodes", v3r.meta.total_episodes, "frames", v3r.meta.total_frames, flush=True)
v2 = LeRobotDataset("tclastlight/so101_sim_pick_v2", root=f"{D}/so101_sim_pick_v2")
v4 = merge_datasets([v2, v3r], output_repo_id="tclastlight/so101_sim_pick_v4", output_dir=f"{D}/so101_sim_pick_v4")
print("v4 episodes", v4.meta.total_episodes, "frames", v4.meta.total_frames, flush=True)
