#!/bin/bash
# 按医院分组的 5 折训练（审计 TODO 第 5 步）。每折独立进程 + checkpoint 守卫（坑#13 教训）。
set -u
PY="/d/Software/Miniconda3/envs/clam_latest/python.exe"
cd /e/Projects/DP/CLAM || exit 1
EXP=luad_lusc_CLAM_sb_site
for i in 0 1 2 3 4; do
  CKPT="results/${EXP}_s1/s_${i}_checkpoint.pt"
  DONE="results/${EXP}_s1/split_${i}_results.pkl"
  # checkpoint 训练中途就会存（坑#13），真正的完成标志是 split_i_results.pkl
  if [ -f "$DONE" ]; then echo "[fold $i] 已完成（results.pkl 存在），跳过"; continue; fi
  if [ -f "$CKPT" ]; then echo "[fold $i] 有中途 checkpoint 但无 results.pkl，视为半成品，删除重来"; rm -f "$CKPT"; rm -rf "results/${EXP}_s1/${i}"; fi
  echo "[fold $i] 开始训练 $(date +%H:%M:%S)"
  "$PY" -X utf8 -u main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 5 \
    --exp_code $EXP --split_dir task_2_site150 --weighted_sample --bag_loss ce \
    --inst_loss svm --task task_2_tumor_subtyping --model_type clam_sb --subtyping \
    --embed_dim 1024 --data_root_dir E:/Projects/DP/CLAM-tutorial/results/tcga/features \
    --k_start $i --k_end $((i+1)) > "results/${EXP}_s1_fold${i}.log" 2>&1
  RC=$?
  if [ ! -f "$DONE" ]; then
    echo "[fold $i] 失败！进程退出码 $RC 且无 results.pkl，停止循环。日志尾部："
    tail -40 "results/${EXP}_s1_fold${i}.log"
    exit 1
  fi
  echo "[fold $i] 完成 $(date +%H:%M:%S)"
done
echo "全部 5 折完成"
