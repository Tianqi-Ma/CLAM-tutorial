# 待办：在数据所在的电脑上完成（2026-10 审计后）

这份清单在存数据的 Windows 电脑上（`E:\Projects\DP\`）用。按顺序做，每一步都写了怎么判断有没有跑对。
审计的来龙去脉见 [README](README.md) 和 [docs/clam-setup-log.md](docs/clam-setup-log.md) 第 24 节。

下面的命令默认已经 `conda activate clam_latest`，当前目录是 `E:\Projects\DP\CLAM-tutorial`。

---

## 第 0 步：准备（几分钟）

- [x] **拉取更新**：先 `git status` 确认本地没有未提交的改动，再 `git pull`。
  - 拉完后，`results/multiomics/` 和 `results/survival/` 里的旧结果会挪到 `results/archive_v1/`，这是正常的。
- [x] **补装可选依赖**：`uv pip install gseapy`。装了才会跑预排序 GSEA；不装也能跑，只是跳过这一项。
- [x] **提交划分文件**：把 `E:\Projects\DP\CLAM\splits\task_2_strict150\` 和 `task_2_tumor_subtyping_100\` 复制到仓库里的 `splits\` 下，然后提交。
  - 原因：换个 sklearn 版本，同样的种子会分出不同的组，这两套划分以后没法再生成。

## 第 1 步：补取数据信息（几分钟，需要联网）

- [x] **刷新元数据**：`python scripts/download_tcga.py --metadata-only`
  - 只向 GDC 查询信息，不下载切片和表达文件。网速慢的话先设 `GDC_PROXY`。
  - 日志里应看到 `clinical.json: 144 cases (… with follow_ups)`，括号里的数字应该是大多数病人。
  - 日志里应看到 `rna_files.csv: … files {'Primary Tumor': …, 'Solid Tissue Normal': …}`。
- [x] **（可选，推荐）准备 TCGA-CDR 生存表**：下载 Liu et al. 2018 *Cell* 的补充表 S1（TCGA-CDR），把 `TCGA-CDR` 这个工作表另存为 `data/metadata/TCGA-CDR.csv`。
  - 有这个文件时，生存分析会自动优先用它。它是官方整理过的生存数据，比 GDC 原始记录可靠。

## 第 2 步：检查放大倍数（几分钟）

- [x] **跑检查脚本**：`python scripts/check_slide_mpp.py`
  - 输出 `results/tcga/slide_mpp.csv`，并打印"倍率 × 亚型"的交叉表。
  - 如果只有一种倍率，这个问题就不存在。
  - 如果 20× 和 40× 混在一起，先记下来，第 6 步再处理。

## 第 3 步：生成新的形态特征（可能要几十分钟，要读 69GB 特征）

- [x] **跑嵌入脚本**：`python scripts/embed_slides.py`
  - 产物在 `results/embeddings/`：`meanpool_*.csv`，以及 `clam_fold0~4_*.csv`。

## 第 4 步：重跑多组学和生存（每个几分钟到十几分钟）

- [x] `python scripts/analysis_multiomics.py`
  - 日志第一行应写"按样本类型保留原发肿瘤 144 个"。
  - **如果写的是"无元数据 N 个"且 N > 0**，说明 GDC 的文件名和本地的对不上，先停下排查。
- [x] `python scripts/analysis_de.py`
  - 需要联网下载 Enrichr 基因集；Python 连不上时会自动改用 `curl.exe`。
- [x] `python scripts/analysis_morph_predict.py`
- [x] `python scripts/analysis_survival.py`
  - **如果打印"缺失不平衡，生存分析不可用"**，说明第 1 步没取到随访记录。先停下排查，不要加 `--force`。
- [x] `python scripts/analysis_cox.py`

⚠️ **暂时不要跑带 `--embedding clam` 的版本**：它的输出文件名和默认版本一样，会把结果覆盖掉。需要时先给脚本的输出文件名加后缀。

这一步也可以直接在 notebook 里做：主 notebook §8 末尾那一格（"8d. v2 重跑"）就是第 1、3、4 步的全部命令。

## 第 5 步：按医院分组重新训练（GPU，几个小时；回答"模型是不是在认医院"）

- [x] **生成按医院分组的划分**：
  ```
  python scripts/make_strict_splits.py data/dataset_csv/tcga_luad_lusc.csv E:/Projects/DP/CLAM/splits/task_2_site150 --group-by site
  ```
- [x] **训练**：在 `E:\Projects\DP\CLAM` 目录下逐折训练，参数和严格版一样，只换实验名和划分目录。每折单独开一个进程，即加 `--k_start i --k_end i+1`，i 从 0 到 4。
  ```
  python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 5 --exp_code luad_lusc_CLAM_sb_site --split_dir task_2_site150 --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping --model_type clam_sb --subtyping --embed_dim 1024 --data_root_dir E:/Projects/DP/CLAM-tutorial/results/tcga/features --k_start 0 --k_end 1
  ```
- [x] **评估**：注意坑 #12，`--splits_dir` 要给完整路径，并且不加 `--subtyping`。
  ```
  python eval.py --k 5 --models_exp_code luad_lusc_CLAM_sb_site_s1 --save_exp_code luad_lusc_CLAM_sb_site_s1_cv --splits_dir E:/Projects/DP/CLAM/splits/task_2_site150 --task task_2_tumor_subtyping --model_type clam_sb --results_dir results --embed_dim 1024 --data_root_dir E:/Projects/DP/CLAM-tutorial/results/tcga/features
  ```
- [x] **整理结果**：把 `CLAM/eval_results/EVAL_luad_lusc_CLAM_sb_site_s1_cv/` 里的 `fold_*.csv` 和 `summary.csv` 复制到仓库的 `results/eval_site150/`，然后运行：
  ```
  python scripts/eval_cv_summary.py --eval-dir results/eval_site150
  ```
  - 拿这里的结果和严格 5 折比：池化 AUC 0.783，患者级 0.775。分数明显下降，说明之前确实有一部分分数是靠"认医院"拿的。

## 第 6 步：按需做

- [x] **重画热图**：用修好的 `scripts/make_heatmap.py` 重画。
  - 示例切片建议从 `results/eval_strict150/fold_*.csv` 里挑模型很确定的（p 接近 0 或 1），并用这张切片所在测试折的模型。
- [ ] **统一倍率**（决定不做，留作已知限制）：第 2 步发现 142 张 40× + 8 张 20×（7 LUSC / 1 LUAD）。只涉及 8 张，重切+重提特征+重训成本远超收益，记录在 README 审计表。只有第 2 步发现倍率混杂时才做。把 40× 切片改成切 512 像素的方块（提特征时会统一缩到 224），再重新提特征、重新训练。工作量大，可以排在最后。

## 第 7 步：提交结果

- [x] **提交并推送新产物**（用个人账号），包括：
  - `data/metadata/clinical.json`、`data/metadata/rna_files.csv`；
  - `results/embeddings/`、`results/multiomics/`、`results/survival/`；
  - `results/eval_site150/`、`results/tcga/slide_mpp.csv`；
  - `splits/`。
- [x] **更新文档**：根据新结果更新这些地方：
  - 主 notebook §6（加按医院分组的成绩）；
  - §8、§9（把 v1 的说明换成 v2 结果；在这台电脑上执行 §8 的 8d 格，让新输出嵌进 notebook）；
  - README 的状态说明和 `docs/clam-setup-log.md`。
- [x] 全部做完后删掉这个文件，或者把已完成的项打勾留作记录。（2026-10-10 全部完成，打勾留档）
