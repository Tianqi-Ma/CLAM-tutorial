#!/bin/bash
# site 训练完成后的评估链：eval.py → 复制结果 → eval_cv_summary（TODO 第 5 步后半）
set -e
PY="/d/Software/Miniconda3/envs/clam_latest/python.exe"
cd /e/Projects/DP/CLAM
"$PY" -X utf8 eval.py --k 5 --models_exp_code luad_lusc_CLAM_sb_site_s1 \
  --save_exp_code luad_lusc_CLAM_sb_site_s1_cv \
  --splits_dir E:/Projects/DP/CLAM/splits/task_2_site150 \
  --task task_2_tumor_subtyping --model_type clam_sb --results_dir results \
  --embed_dim 1024 --data_root_dir E:/Projects/DP/CLAM-tutorial/results/tcga/features 2>&1 | tail -15
SRC="/e/Projects/DP/CLAM/eval_results/EVAL_luad_lusc_CLAM_sb_site_s1_cv"
DST="/e/Projects/DP/CLAM-tutorial/results/eval_site150"
mkdir -p "$DST"
cp "$SRC"/fold_*.csv "$SRC"/summary.csv "$DST/"
cd /e/Projects/DP/CLAM-tutorial
python -X utf8 scripts/eval_cv_summary.py --eval-dir results/eval_site150 | tee results/eval_site150/cv_summary.txt
