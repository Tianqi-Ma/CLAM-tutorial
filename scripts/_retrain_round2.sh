#!/bin/bash
# 第二轮 TODO 第 1 步全链：重训缺失折 → 重评估 → 汇总（GPU 串行）
set -u
TUT=/e/Projects/DP/CLAM-tutorial
cd "$TUT"

bash scripts/_retrain_missing.sh luad_lusc_CLAM_sb_strict task_2_strict150 1 3 || exit 1
bash scripts/_retrain_missing.sh luad_lusc_CLAM_sb task_2_tumor_subtyping_100 2 3 4 || exit 1

PY="/d/Software/Miniconda3/envs/clam_latest/python.exe"
cd /e/Projects/DP/CLAM

echo "=== eval strict150 ==="
"$PY" -X utf8 -u eval.py --k 5 --models_exp_code luad_lusc_CLAM_sb_strict_s1 \
  --save_exp_code luad_lusc_CLAM_sb_strict_s1_cv \
  --splits_dir E:/Projects/DP/CLAM/splits/task_2_strict150 \
  --task task_2_tumor_subtyping --model_type clam_sb --results_dir results \
  --embed_dim 1024 --data_root_dir E:/Projects/DP/CLAM-tutorial/results/tcga/features 2>&1 | tail -8

echo "=== eval MC150 ==="
"$PY" -X utf8 -u eval.py --k 5 --models_exp_code luad_lusc_CLAM_sb_s1 \
  --save_exp_code luad_lusc_CLAM_sb_s1_cv \
  --splits_dir E:/Projects/DP/CLAM/splits/task_2_tumor_subtyping_100 \
  --task task_2_tumor_subtyping --model_type clam_sb --results_dir results \
  --embed_dim 1024 --data_root_dir E:/Projects/DP/CLAM-tutorial/results/tcga/features 2>&1 | tail -8

cd "$TUT"
for pair in "EVAL_luad_lusc_CLAM_sb_strict_s1_cv:results/eval_strict150" "EVAL_luad_lusc_CLAM_sb_s1_cv:results/eval"; do
  SRC="/e/Projects/DP/CLAM/eval_results/${pair%%:*}"; DST="${pair##*:}"
  cp "$SRC"/fold_*.csv "$SRC"/summary.csv "$DST/"
done

echo "=== cv_summary strict150 ==="
python -X utf8 scripts/eval_cv_summary.py --eval-dir results/eval_strict150 | tee results/eval_strict150/cv_summary.txt | head -8
echo "=== cv_summary MC150 ==="
python -X utf8 scripts/eval_cv_summary.py --eval-dir results/eval | tee results/eval/cv_summary.txt | head -8
echo "=== 配对比较 strict150 vs site150 ==="
python -X utf8 scripts/eval_cv_summary.py --eval-dir results/eval_strict150 --compare results/eval_site150 | tee results/compare_strict_vs_site.txt | tail -12
echo "ALL DONE"
