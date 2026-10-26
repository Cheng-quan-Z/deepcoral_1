#!/usr/bin/env bash
# 阶段一的两组对照（可选，需先跑完 run_stage1.sh coral）
#
# 用法：bash stage1_reproduction/run_stage1_controls.sh [shared|inv|all]
#   shared  1.3 第 4 步：所有任务共用 A→W 标定出的 λ*，与各任务自己的 λ* 对比
#   inv     1.1：学习率改用 lr0 / (1 + 10p)^0.75 衰减调度，重跑基线并重新标定 λ*
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${1:-all}
DATA_DIR=${DATA_DIR:-data_cache}
OUT=stage1_reproduction/results
TASKS=${TASKS:-"AW AD DA DW WA WD"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
INV_SEEDS=${INV_SEEDS:-"1 2 3"}
CALIB_SEED=${CALIB_SEED:-1}
SAVE=${SAVE:-""}
COMMON="--data_dir $DATA_DIR --out_dir $OUT $SAVE"

lam_of() {
  python -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['lam_star'])" "$1" "$2"
}

run_shared() {
  lam_aw=$(lam_of "$OUT/lambda_star.json" AW)
  for task in $TASKS; do
    [ "$task" = AW ] && continue  # A→W 上两种做法相同
    for seed in $SEEDS; do
      python train.py --task "$task" --seed "$seed" --lam "$lam_aw" --name "s1-sharedlam-$task-lamAW-seed$seed" $COMMON
    done
  done
}

run_inv() {
  for task in $TASKS; do
    for seed in $INV_SEEDS; do
      python train.py --task "$task" --seed "$seed" --lam 0 --lr_schedule inv --name "s1-baseinv-$task-lam0-seed$seed" $COMMON
    done
  done
  python stage1_reproduction/calibrate_lambda.py --tasks $TASKS --seed "$CALIB_SEED" \
    --exp coralinv --calib_exp calibinv --lambda_file "$OUT/lambda_star_inv.json" --lr_schedule inv $COMMON
  for task in $TASKS; do
    lam=$(lam_of "$OUT/lambda_star_inv.json" "$task")
    for seed in $INV_SEEDS; do
      python train.py --task "$task" --seed "$seed" --lam "$lam" --lr_schedule inv --name "s1-coralinv-$task-lamstar-seed$seed" $COMMON
    done
  done
}

case "$PHASE" in
  shared) run_shared ;;
  inv) run_inv ;;
  all) run_shared; run_inv ;;
  *) echo "未知阶段：$PHASE（可选 shared|inv|all）"; exit 1 ;;
esac
python stage1_reproduction/analyze_stage1.py --out_dir "$OUT"
