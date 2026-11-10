# Inference tuning · act_sim_pick_v2/checkpoints/010000 (chunk_size 100)

50 unseen layouts, seed 9100, goal 0.04 m, overhead margin 0.04, Mac MPS. Each run writes `sim/eval_inf_<tag>.log`.

| setting | n_action_steps | temporal ensemble | success | wall |
| --- | --- | --- | --- | --- |
| default (control) | 100 | off | **38/50 = 76%** | 465 s |
| --n-action-steps 50 | 50 | off | 29/50 = 58% | 455 s |
| --n-action-steps 25 | 25 | off | 29/50 = 58% | 464 s |
| --n-action-steps 10 | 10 | off | 8/50 = 16% | 478 s |
| --temporal-ensemble 0.01 | 1 | 0.01 | 28/50 = 56% | 770 s |

**Result:** nothing beats the checkpoint's default of executing the full 100-step chunk open loop. Every setting that
re-plans more often is worse, and the more often it re-plans, the worse it gets (16% at every 10 steps). No
confirmation run was needed because no setting came within 6 points of the default. The policy's chunks are
self-consistent; re-observing mid-chunk makes it re-aim and dither at the grasp.

**Jetson translation (so101-run-async, rtc engine):** the queue threshold sets how many actions remain when the next
chunk is computed. Our current `--inference.queue_threshold=90` re-plans every ~10 steps, which is the 16% regime here.
For this sim-trained ACT use `--inference.queue_threshold=10` (re-plan when ~10 actions are left, i.e. execute ~90 of
each 100-step chunk) with `--inference.rtc.execution_horizon=10`. Keep the rtc engine for timing; just lower the
threshold. Re-check once with 20 (every ~80 steps) if 10 feels laggy on the real arm. This does not necessarily
transfer to the lighter ACT model (trained on human teleop); tune that one the same way in sim-free A/B on the arm.
