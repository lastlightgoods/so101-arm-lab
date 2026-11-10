# Why act_sim_pick_v2/010000 fails 12 of 50 (sim, 21 Oct 2026)

Policy: ACT, 600 scripted demos, warm start, 10k steps. Eval: 50 unseen seeds (9100), goal 0.04 m, overhead margin 0.04.
Logs: `sim/analysis_v2_10k.log` (400 steps), `sim/analysis_v2_10k_600steps.log` (600 steps), GIFs `sim/rollouts/analysis/`, geometry `sim/analysis_geometry.json`.

## Taxonomy (12 failures)
| Mode | Count | Episodes |
| --- | --- | --- |
| Never grasped: descends 1-3 cm off the cube, pushes or misses it, then **carries nothing to the disc and parks** ("phantom carry") or hovers | 11 | 2, 9, 14, 17, 18, 21, 28, 30, 32, 46, 49 |
| Grasped and lifted (8.6 cm), dropped in transit, landed 9 cm from the disc | 1 | 20 |
| Rescued by a 600-step budget instead of 400 | 0 | identical 38/50 at 600 steps |

Successes close the gripper 45-58 steps in and finish in ~100-130 steps; the three slow successes (ep 4, 15, 37) are re-grasps after a first miss, so the policy *can* recover but rarely does. Failures are terminal: once the gripper closes on air the policy follows the demo script to the disc.

## Correlations
- **Reach:** cube radius from base, failures mean 0.256 m vs successes 0.228 m. At r >= 0.29 m, 3 of 8 fail (38%); elsewhere 9 of 42 (21%). Spawn radius 0.17 around (0.15, 0) puts cubes out to 0.32 m, close to the 0.33 m top-down reach limit where the approach is steepest.
- **Angle:** none (mean |angle| 17 deg both groups).
- **Cube-to-disc distance:** slight (fail 0.139 m vs ok 0.115 m); the one drop (ep 20) is the longest carry, 0.217 m.
- The wrist view shows the cube in frame in every failure; this is lateral precision at the moment of descent, not perception of where the cube is.

## Recommendations (cheap first)
1. **Put recovery in the demos.** In `collect_sim.py` `Expert.act`, after the `close` stage check `info["is_grasped"]`; if false, splice `hover_obj -> descend -> close` again (max 2 retries) instead of proceeding. Today every demo is a perfect first grasp, so the policy has no re-grasp behaviour; 11 of 12 failures would be rescued by one retry.
2. **Demonstrate corrections (DART-style).** Add 1-2 cm noise to `above_obj` only (`WAYPOINT_NOISE_M` 0.004 -> 0.012 for the hover waypoint, keep `at_obj` noise at 0.002) so the descend stage shows the arm re-centring over the cube. This targets the dominant failure directly.
3. **Oversample the far band or trim it.** Either `--spawn-max 0.15` (r <= 0.30 m) for both collect and eval, or collect an extra 150 demos with `--spawn-min 0.13 --spawn-max 0.17` and merge. The far band costs ~17 points of success on its own.
4. **Score every checkpoint.** Success peaked at 10k and fell to 52% by 15k; also evaluate 5k, 7.5k and 12.5k (`--save_freq=2500` already writes them) before picking.
5. **Keep 400 steps for eval.** 600 changes nothing; the time goes to training and data instead.
Not covered here: inference settings (n_action_steps, temporal ensemble) are another agent's job.
