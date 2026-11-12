# tidy: every toy on the mat into the tape ring, one recorded pick each

`tools/tidy.py` puts every toy on the mat into the 2 in tall blue tape ring, one pick per toy, and each pick can be recorded as a LeRobot training episode (the task "Pick up the toy and put it in the tape ring", dataset `lastlight/so101-toys-to-ring`, rejects beside it). It also resets the mat by itself (lift the ring off the pile, spread the toys), so a recording session can run round after round with nobody there.

It wraps `tools/pick_loop.py`'s closed loop (`closed_loop_pick`: readback-corrected hover, grasp, lift) and edits nothing in it: the WP table, `build_plan`, `jaw_check` and `run_plan` are patched for the length of one pick and put back.

## Live numbers (11 Nov 2026, the scratch version of this tool)

7 toys into the ring in 8 tries; median final grasp error about 2 mm. What it added to the plain closed loop, all in `tools/tidy.py`:

- the ring found by its blue colour in the top frame (least-squares circle, `fit_ring`);
- an obstacle check of every planned waypoint and of the joint path between them against the ring as a solid cylinder (`Rig.hits`): points along wrist, gripper, gripper_frame and the tip; radius 8 mm at the tip, 20 mm on the body; the arm's sag subtracted from every height (4 mm at 150 mm reach, 24 mm at 290 mm); the 51 mm rim plus 5 mm;
- a release over the ring at 110 mm with a tipped gripper (`cam_calib.ik_tip(down=False)`), because the vertical IK only reaches 76-84 mm there;
- an adaptive route: rest straight to the toy when that path is clean, else up in front of the base and over at 110 mm; the carry swapped for HIGH (closed) -> open -> straight home when that path is clean;
- hover and lift heights per toy from the highest reachable one;
- the "fluid" trims: no `place` / `lift2` that repeat a pose, stage 2's hover merged into its descent, 0.1 s settles on the back-offs, the carry at 80 deg/s;
- the wrist roll for an elongated toy from the scene map's angle, resolved with the arm at rest (the map cannot see a toy under the arm);
- aim 4 mm (nut, wing nut) / 5 mm (spacer); the toy kind comes from the scene map's size and aspect (`toy_kind`: round ~10-12 mm nut; ratio 1.3-1.8 wing nut; standing 6 mm or ratio >= 1.8 spacer; bigger than 22 mm unknown);
- a miss retries once, then goes home without the carry; a per-pick timeline JSON (`outputs/run/tidy_<session>/pick_N.json`).

## Commands

    tools/tidy.py plan                       # no motion: ring, toys, reachable / blocked per toy, the order
    tools/tidy.py run [--record] [--max N]   # tidy the toys one by one, re-reading the scene map between picks
    tools/tidy.py reset [--record-spread]    # ring off the pile, set down elsewhere, toys spread
    tools/tidy.py loop --rounds K [--record] # tidy all -> reset -> tidy all ... (K tidy rounds, K-1 resets)

`plan` prints JSON: the ring centre, every toy with kind, `done` (inside the ring: skipped) / `ok` / `blocked` (the check found a waypoint or path through the rim) / `unreachable`, the route (direct or over at 110 mm), and the order. Order `--order clean` (default): toys whose straight route from rest is clean first, then nearest to the arm's base; `--order near`: nearest to the ring first. Blocked and unreachable toys are never driven: `run` skips them and stops with `refused: no toy has a clean route` when nothing else is left. `plan` needs the ring in the top frame (the arm parked, not over it).

Toys inside the ring's footprint (centre within 48 mm of the ring's centre) are done and skipped. The scene map is re-read, and the ring re-fitted from the top frame, before every pick.

Guardrails (tools/guardrails.py; the stop file `outputs/run/autoloop.stop` is the same one as autoloop's): stop file, panel and Jetson liveness, `arm_busy`, overload, marker / calibration health, the workspace box on every planned waypoint (ceiling 4.6 in so the 110 mm release passes), the pick_seq motion error, servo health (heat `--temp-limit` 60 C, current, supply, error bits, stall: see docs/autoloop.md, Overnight guards; `/api/state` publishes no temperature, the checks read the plan rows and `/api/tele`), `--max-minutes` / `--max-tries` caps, `--unattended`, and a park at rest on Ctrl-C, SIGTERM or a crash, the demo recorder when recording, and 3 consecutive failed picks. A trip mid-pick sends the arm home if the dashboard still reads back.

MLflow: experiment `so101-tidy`, one run per pick (params: kind, aim, roll, route, ring and toy position; metrics: held, placed, spot_empty, ok, seconds, residual, `gripper_after_lift`, `camera_score`, `interior_diff_px`, `interior_need_px`; `timeline.json`), plus the dataset link (`dataset_repo`, `kept`, `discard_reason`, `episode_index`) when recording.

## The keep rule (strict, after the 11 Nov 21:32 pilot)

The first recorded pilot missed the grasp. On the retry the gripper read 0.54 after the lift (empty) while the wrist camera's ring check said in_jaws True (score 0.388: a false ring on the mat pattern between the empty jaws); the spacer had been pushed off its spot, so "spot empty" was true, and the episode was kept although the toy never reached the ring. So:

1. **held** = the gripper readback after the lift > 2.0. The camera ring check counts only when the scene map calls the toy a wing nut (a thin wing reads ~1.5, like empty jaws). The camera alone never passes a nut or a spacer (`grip_decision`; truth table in the tests).
2. **placed** = the toy really reached the ring: the grey top frame before the pick is compared with the one after the release over a disc of 0.8 x the fitted rim radius around the fitted centre (the rim and its shadow stay out); pixels with |diff| > 30 are counted and must reach 40 % of the toy's scene-map area in pixels (at least 6) (`placed_verdict`). The numbers (`diff_px`, `need_px`) are in the verdict. "Spot empty" stays as an extra condition, never the proof.
3. **keep** = held AND placed AND spot empty AND healthy recording AND no trip. Everything else goes to the rejects dataset with the reasons (`not_held`, `not_in_ring`, `toy_still_at_spot`, ...); a miss with no carry is rejected, not dropped; a trip mid-pick aborts the episode (nothing partial is saved).

Logged for tuning the thresholds from data: per try the gripper reading after the lift, the camera result and score, and the interior diff counts, in the episode meta (`teacher.grip`, `teacher.interior`) and in MLflow. The thresholds (2.0, 0.8, 30, 40 %) are constants at the top of the checks, not tuned yet. Known limit: the release is always over the ring's centre, so a toy landing on top of an earlier one changes fewer pixels and can be rejected (a false reject, the safe direction); vary the release point per toy if that costs too many episodes.

`--grip-check own` (default) is this tool's check; `--grip-check loop` leaves closed_loop_pick's own (switch to it when the `final-correct` branch's readback and grip check are on main, then drop `own`).

## Recording

Each ring pick is one episode through the phase-2 recorder (`RemoteRecorder` from `tools/demo_session.py`, branch `demos-phase2`; this tool only needs `open / end / commit / abort / alive`):

- open with meta `task`, `scene` (toy kind, toy position, ring position, grasp roll, aim, lighting tag) and `teacher` (session, pick number); `jetson/demo_recorder.py` takes the task from `meta["task"]` (a two-line hook; the default stays the nut task);
- the episode starts before the first motion (the route from rest is part of it) and ends AFTER the arm is back at rest: the carry stage is one pick_seq plan (hover over the ring, open, home), so the end state equals the start state, a policy learns to park, and no stage boundary is added for the recorder. The scene-map judging and everything after is not recorded;
- a recorder that refuses to open never stops the arm: that pick runs unrecorded and the result says `recorder_open_failed`;
- pick_seq writes ticks only when recording: `--record` sets `pick_loop.PICK_SEQ_ARGS = "--tick-sink PATH"` before the session starts (the phase-2 hook in `PickSession`; on a checkout without it nothing is written);
- `--record-spread` records the reset's spread picks into their own dataset (`lastlight/so101-toys-spread`, task "Pick up the toy and put it down on the mat"), off by default. The resident recorder holds one dataset configuration, so switching between the two reconfigures it (`Episode.activate`; not tried live).

### How to run a recording session

1. Start camd and the demo recorder on the Jetson (docs/scripted-demos.md, phase 2); the dashboard and MLflow up on the Mac.
2. `tools/tidy.py plan`: every toy `ok` or `done`, the ring found. Fix what is `blocked` by hand.
3. `tools/tidy.py run --record --max 1 --lighting-tag evening-lamps`: one pick, then look at the episode and at `outputs/run/tidy_<session>/pick_1.json`.
4. `tools/tidy.py loop --rounds 5 --record --lighting-tag evening-lamps`. `touch outputs/run/autoloop.stop` ends it after the current pick.
5. Vary the lighting tag between sessions, never within one.

## Reset

`tidy.py reset` (and `loop`) do, with the toys left on the mat as a pile where the ring stood:

(a) Lift the ring off the pile (`RingMover`). Grip the wall on the side facing the base, jaws radial (wrist turned to make it so), jaw 50, the thumb 12 mm behind the wall midline, one readback correction at the hover. Improvements on the 11 Nov first try (the arm sagged ~24 mm under the ring, set it ~0.5 in short, and it hung tilted and dragged toys): the tips grip 15 mm up the wall instead of 25, so the ring's bottom is 10 mm higher for the same tip height; the lift is a tipped-IK pose as high as reachable (cap 110 mm) and the carry stays at that height; the predicted ring bottom is `tip - sag(reach) - 15 mm` with the sag 24 mm at 230 mm reach scaled linearly with reach (an assumption, not measured), aiming for >= 40 mm; under 25 mm the plan is refused, and after the real lift the readback tip height is checked again (a ring that hangs too low is put back where it was). A ring that is not gripped (readback <= 2.0 after the lift) is not carried. Set down at a random spot (`sample_ring_spot`): inside `--ring-box` (default 5.5-12.0 x 3.3-8.7 in, wide on purpose: the ring has to hop at least 3.6 in to leave 1.5 in of floor between its wall and the pile), 1.5 in clear of the pile and every other toy, wall >= 30 mm from the mat edges and 35 mm from a marker centre, >= 60 mm from where it stood, and only where the grip, the tipped lift and the set-down are reachable. Lowered until the tip is ~1 mm above the grip height, open, lift straight up, home, then the ring is re-fitted from the top frame: the next round uses the measured centre, not the commanded one.

(b) Spread the pile onto random clear spots, one closed-loop pick each with the normal place waypoint. Spots (`sample_spread_spot`): autoloop's rules (40 mm from every other object, 50 mm from a marker, 60 mm from where the toy is) plus 40 mm clear of the ring's wall and outside the pile's radius, inside `--spread-box` (default 5.5-12.5 x 3.0-9.0 in: comfortable reach, off the far edges where 4-6 mm grasp errors were seen). A spot whose carry would pass the ring is not used. The scene map merges touching toys into one blob: `pile_candidates` picks at the most compact toy-sized piece of the blob's mask (`split_mask`: growing erosion of a foreground mask cut from the top frame; NOT validated on the real mat, `--no-blob-mask` skips it) and otherwise at the ends of the blob's rotated rectangle, a toy radius in from the edge. The grip check tells whether a pick got one; the scene map is re-read after every pick. A blob that fails twice is nudged once (a low sweep, 7 mm up, 30 mm each side of its centre, open jaws, 30 deg/s, away from the ring), and never when it is within 90 mm of the ring's wall. Is the nudge safe? Mostly, not entirely: it is slow and low, and light toys do not fly, but a sweep can tip a standing spacer into a wing nut's wing and jam the jaws, push a toy toward the ring or the mat edge, and the tip is within ~6 mm of the mat. `--no-nudge` turns it off; it needs one live look before it runs unattended.

(c) `loop --rounds K`: tidy all, reset, tidy all, ... with a reset only between rounds; it stops on the stop file, on 3 consecutive failures, on a refused route, on a reset that finds no spot, or on any trip.

## Not verified live

Everything is tested on fakes (`tests/tools/test_tidy.py`: ShortArm and a small world that moves toys). Never run on the real arm from this branch: the ring lift's new numbers (15 mm grip, the sag model, the 40 mm target), the interior-diff thresholds, the blob mask, the nudge, the dataset switch, and the ring / spread boxes are defaults to confirm with `plan` and one supervised `reset`.
