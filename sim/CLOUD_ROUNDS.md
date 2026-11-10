# Sim cloud rounds (HF Jobs, a100-large $2.50/h) — dataset tclastlight/so101_sim_pick_v2 (600 eps, 92,828 frames)

Baseline on the Mac: act_sim_pick_v2/checkpoints/010000 = 76% (50 unseen layouts, seed 9100, goal 0.04).

| job | id | steps | warm start | checkpoint | success (50 unseen) | est. cost |
| --- | --- | --- | --- | --- | --- | --- |
| act_sim_pick_long | 6ad877680a35baab53a1b3ca | 60000 | none (scratch) | pending | pending | ~$7 (3 h cap) |
| act_sim_pick_v3 | 6ad877870a35baab53a1b3d6 | 15000 | act_sim_pick_v2_10k | pending | pending | ~$2 |

Model repos: tclastlight/act_sim_pick_long (job 6ad87768), tclastlight/act_sim_pick_v3_2026-10-21_08-27-49 (job 6ad87787). First v3 submission failed client-side: ACT has no `scheduler_decay_steps` (SmolVLA-only flag); resubmitted without it.

## Checkpoint scores (as they land)

| job | checkpoint | success (50 unseen) | path |
| --- | --- | --- | --- |

Both jobs train at ~3.1 step/s on the A100 (batch 32). The 60k "long" run needs ~5.3 h, so its 3 h cap will stop it near step 33k;
checkpoints every 2500 are on the Hub regardless. Projected cost: long ~$7.50 + v3 ~$3.40 = ~$11 (cap $12).
Scores: long every 5000 steps, v3 every 2500, by sim/cloud_eval_driver.py (log sim/cloud_eval_driver.log).
Note: every cloud checkpoint needs `python jetson/prep_ckpt.py <pretrained_model>` before our LeRobot 0.6.1 loads it (drops `dtype`); the driver now does this. Mac round 3 (recovery-expert data, warm start) scored 66/64/46/58/62% at 5k..15k, below the 76% baseline.
| act_sim_pick_long | ckpt 005000 | 22/50 = 44% | ~/so101/policies_hub/act_sim_pick_long_005000/checkpoints/005000/pretrained_model |
| act_sim_pick_long | ckpt 010000 | 28/50 = 56% | ~/so101/policies_hub/act_sim_pick_long_010000/checkpoints/010000/pretrained_model |
| act_sim_pick_long | ckpt 015000 | 23/50 = 46% | ~/so101/policies_hub/act_sim_pick_long_015000/checkpoints/015000/pretrained_model |
| act_sim_pick_long | ckpt 020000 | 26/50 = 52% | ~/so101/policies_hub/act_sim_pick_long_020000/checkpoints/020000/pretrained_model |
| act_sim_pick_long | ckpt 025000 | 27/50 = 54% | ~/so101/policies_hub/act_sim_pick_long_025000/checkpoints/025000/pretrained_model |
| act_sim_pick_long | ckpt 030000 | 31/50 = 62% | ~/so101/policies_hub/act_sim_pick_long_030000/checkpoints/030000/pretrained_model |
| act_sim_pick_v3 | ckpt 002500 | 36/50 = 72% | ~/so101/policies_hub/act_sim_pick_v3_002500/checkpoints/002500/pretrained_model |
| act_sim_pick_v3 | ckpt 005000 | 33/50 = 66% | ~/so101/policies_hub/act_sim_pick_v3_005000/checkpoints/005000/pretrained_model |
| act_sim_pick_v3 | ckpt 007500 | 29/50 = 58% | ~/so101/policies_hub/act_sim_pick_v3_007500/checkpoints/007500/pretrained_model |
| act_sim_pick_v3 | ckpt 010000 | 29/50 = 58% | ~/so101/policies_hub/act_sim_pick_v3_010000/checkpoints/010000/pretrained_model |
| act_sim_pick_v3 | ckpt 012500 | 34/50 = 68% | ~/so101/policies_hub/act_sim_pick_v3_012500/checkpoints/012500/pretrained_model |
| act_sim_pick_v3 | ckpt 015000 | 33/50 = 66% | ~/so101/policies_hub/act_sim_pick_v3_015000/checkpoints/015000/pretrained_model |

## Outcome (21 Oct 2026)
- No cloud checkpoint beat the Mac baseline (act_sim_pick_v2/010000 = 76%). Best cloud: v3 warm start at 2,500 steps = 72%;
  long from-scratch run reached 62% at 30k when its 3 h cap killed it (trend still rising: 44/56/46/52/54/62 at 5k..30k).
- v3 (warm start, batch 32, lr default) degraded after 2.5k steps (72 -> 66 -> 58 -> 58 -> 68 -> 66): fine-tuning the
  76% checkpoint at batch 32 mostly hurts; if repeated, use a lower LR / fewer steps, or evaluate every 1k steps.
- Cost: long 10,801 s ($7.50) + v3 4,911 s ($3.41) = ~$10.9 of the $12 cap. Jobs: 6ad87768 (ERROR: timeout), 6ad87787 (COMPLETED).
- Scoring: 50 unseen layouts, seed 9100, goal 0.04 — one 50-episode eval has ~±7% noise, so 72 vs 76 is within noise.
