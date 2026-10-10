#!/bin/bash
# 重训缺失 fold（第二轮 TODO 第 1 步）。完成守卫认 split_{k}_results.pkl（坑#13/§26 教训）。
# 用法: bash scripts/_retrain_missing.sh <exp_code> <split_dir> <folds...>
# 例: bash scripts/_retrain_missing.sh luad_lusc_CLAM_sb_strict task_2_strict150 1 3
set -u
PY="/d/Software/Miniconda3/envs/clam_latest/python.exe"
cd /e/Projects/DP/CLAM || exit 1
cd /e/Projects/DP/CLAM || exit 1
EXP="$1"; SPLIT="$2"; shift 2
for i in "$@"; do
  DONE="results/${EXP}_s1/split_${i}_results.pkl"
  CKPT="results/${EXP}_s1/s_${i}_checkpoint.pt"
  if [ -f "$DONE" ]; then echo "[fold $i] 已完成（results.pkl 存在），跳过"; continue; fi
  if [ -f "$CKPT" ]; then echo "[fold $i] 有中途 checkpoint 但无 results.pkl，视为半成品，删除重来"; rm -f "$CKPT"; rm -rf "results/${EXP}_s1/${i}"; fi
  echo "[fold $i] 开始训练 $(date +%H:%M:%S)"
  # GPU 偶发 illegal instruction（上下文损坏后重试常能过）：每折最多试 3 次
  OK=0
  for try in 1 2 3; do
    "$PY" -X utf8 -u main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 5 \
      --exp_code "$EXP" --split_dir "$SPLIT" --weighted_sample --bag_loss ce \
      --inst_loss svm --task task_2_tumor_subtyping --model_type clam_sb --subtyping \
      --embed_dim 1024 --data_root_dir E:/Projects/DP/CLAM-tutorial/results/tcga/features \
      --k_start $i --k_end $((i+1)) > "results/${EXP}_s1_fold${i}.log" 2>&1
    RC=$?
    if [ -f "$DONE" ]; then OK=1; break; fi
    echo "[fold $i] 第 $try 次失败（退出码 $RC）"
    sleep 20   # 让 GPU 驱动缓一下
  done
  if [ "$OK" = 0 ]; then
    echo "[fold $i] 连试 3 次都失败，停止。日志尾部："
    tail -40 "results/${EXP}_s1_fold${i}.log"
    exit 1
  fi
  echo "[fold $i] 完成 $(date +%H:%M:%S)"
done
echo "本实验缺失 fold 全部完成: $EXP"
