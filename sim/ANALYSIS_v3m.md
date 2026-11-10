# Why round 3 (act_sim_pick_v3m) regressed: 66% vs 76%

Same 50 seeds. Baseline v2/10k 38/50; v3m/5k 33/50. Logs `sim/eval_v3m_*.log`, `sim/eval_v3m_005000_gifs.log`, GIFs `sim/rollouts/v3m_005000_all/`.

## 1. Failure taxonomy, v3m/5k vs baseline
| Mode | v2/10k | v3m/5k | Notes |
| --- | --- | --- | --- |
| Never grasped | 11 | 10 | v3m now *retries* (eps 3, 4, 38 re-hover 2-3 times) but misses by ~1 cm each time |
| Grasped, then re-opened, never lifted | 0 | **3** (eps 21, 29, 45) | new: imitates the retry's `open_r` on a good grasp |
| Lifted, not placed (drop or 1 cm outside goal) | 1 | **4** (eps 12, 25, 35, 39) | new: placement scatter; final 0.046/0.051 m just outside 0.04 |
| Total | 12 | 17 | |

Episode-level: v3m **rescued 7** baseline failures (9, 18, 20, 28, 32, 46, 49) and **added 12 new ones**. Successful episodes are not slower (median first success 116 vs 115 steps). So the recovery skill transferred; the cost came from two side effects: opening after a real grasp, and sloppier grasp/place precision.

## 2. The v3 dataset
- Only **73 of 600 episodes (12%) contain a recovery** (two gripper closes), although 30% had a deliberate 1.2-3 cm first-descent offset: most offset descents still caught the 25 mm cube. Those ~18% of episodes teach "an off-centre descent is fine", and the grasp-offset compensation then shifts the place waypoint, which matches the new placement scatter.
- Lengths: median 158 vs 152 (v2), p90 231 vs 167, max 415; 111 of 711 attempts were discarded (v2: 7 of 307), so the kept set is survivor-biased.
- Recovery episodes look sane in the overhead frames (miss, open, re-hover, grasp, place); the retry itself is not pathological. The retry is triggered after `lift` instead of right after `close`, so some demos open the jaws while holding the cube.

## 3. Round 4, ranked
1. **Recovery minority mixed with clean data, no regeneration:** v4 = all 600 v2 episodes + the 73 v3 recovery episodes (11%); fine-tune from v2/10k, 10k steps, score every 2.5k. Keeps the 7 rescues, drops the sloppy-grasp majority.
2. If regenerating: keep only episodes whose first grasp truly failed (check `is_grasped` right after `close`), retry from there, never after a successful lift; DART noise 0.012 on `hover_obj` only, not `above_tgt`.
3. DART-only without misses: second choice; the placement regression suggests noise at the place hover hurts.
4. 1200 clean demos: low value per hour; the data sweep plateaued at 200-300.
5. Longer training on v2: no; v2 peaked at 10k and fell to 52% by 15k.
