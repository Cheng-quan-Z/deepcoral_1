#!/usr/bin/env bash
# 阶段一：忠实复现。6 任务 × 5 种子 × {无适应（λ = 0），Deep CORAL（λ*）}
#
# 用法：bash stage1_reproduction/run_stage1.sh [baseline|coral|analyze|all]
#   baseline  只跑无适应基线；跑完先看 results/stage1_summary.md 里的“基线验收”，不达标先别跑 coral
#   coral     按 1.3 的规则标定各任务的 λ*，再跑 Deep CORAL
#   analyze   只做汇总
# 已完成的运行（日志 JSON 已存在）会被跳过，中断后重跑同一命令即可续跑。
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${1:-all}
DATA_DIR=${DATA_DIR:-data_cache}
OUT=stage1_reproduction/results
TASKS=${TASKS:-"AW AD DA DW WA WD"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
CALIB_SEED=${CALIB_SEED:-1}
# 最后一轮的特征供阶段三、四离线分析；权重每个约 230 MB，磁盘紧张可设 SAVE="--save_features"
SAVE=${SAVE:-"--save_features --save_ckpt"}
COMMON="--data_dir $DATA_DIR --out_dir $OUT $SAVE"

lam_of() {
  python -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['lam_star'])" "$1" "$2"
}

run_baseline() {
  for task in $TASKS; do
    for seed in $SEEDS; do
      python train.py --task "$task" --seed "$seed" --lam 0 --name "s1-base-$task-lam0-seed$seed" $COMMON
    done
  done
}

run_coral() {
  python stage1_reproduction/calibrate_lambda.py --tasks $TASKS --seed "$CALIB_SEED" $COMMON
  for task in $TASKS; do
    lam=$(lam_of "$OUT/lambda_star.json" "$task")
    for seed in $SEEDS; do
      python train.py --task "$task" --seed "$seed" --lam "$lam" --name "s1-coral-$task-lamstar-seed$seed" $COMMON
    done
  done
}

case "$PHASE" in
  baseline) run_baseline ;;
  coral) run_coral ;;
  analyze) ;;
  all) run_baseline; run_coral ;;
  *) echo "未知阶段：$PHASE（可选 baseline|coral|analyze|all）"; exit 1 ;;
esac
python stage1_reproduction/analyze_stage1.py --out_dir "$OUT"
