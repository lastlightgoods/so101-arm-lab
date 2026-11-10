#!/bin/bash
cd ~/so101/sim/so101-lerobot
CK=~/so101/outputs/train/act_sim_pick_v2/checkpoints/010000/pretrained_model
run() { tag=$1; shift
  until [ "$(pgrep -f eval_sim.py | wc -l)" -lt 2 ]; do sleep 30; done
  s=$(date +%s); MUJOCO_GL=glfw ../../.venv/bin/python eval_sim.py --policy $CK --episodes 50 --seed ${SEED:-9100} --goal-thresh 0.04 --overhead-margin 0.04 "$@" > ~/so101/sim/eval_inf_${tag}.log 2>&1
  echo "$tag wall=$(( $(date +%s) - s ))s $(grep -E '^inference:' ~/so101/sim/eval_inf_${tag}.log | head -1) | $(grep -E '^SUCCESS' ~/so101/sim/eval_inf_${tag}.log)" >> ~/so101/sim/inf_tuning_summary.txt; }
: > ~/so101/sim/inf_tuning_summary.txt
run default
run nas50 --n-action-steps 50
run nas25 --n-action-steps 25
run nas10 --n-action-steps 10
run te001 --temporal-ensemble 0.01
echo ALLDONE >> ~/so101/sim/inf_tuning_summary.txt
