# CLAM-tutorial

在 Windows 11 上从零复现 CLAM（mahmoodlab/CLAM）WSI 弱监督分析流水线的完整工程。
配套踩坑实录见 [docs/clam-setup-log.md](docs/clam-setup-log.md)。

> **当前状态（2026-10-09 审计后）**
> - 数据：150 张 TCGA 诊断切片（LUAD/LUSC 各 75，144 名患者，138GB）→ 842 万 patch → ResNet50 特征（69GB）。
> - 分类（CLAM-SB，LUAD vs LUSC）：严格 5 折（每个患者恰好测一次）每折 AUC **0.834 ± 0.090**；把 5 折预测拼起来的池化 AUC **0.783**；
>   患者级 AUC **0.775（bootstrap 95% CI 0.70–0.85）**。CLAM 默认的蒙特卡洛划分每折均值 0.788 ± 0.181，且只覆盖 52/150 张。
> - ⚠️ **中心混杂**：43 个组织来源中心每个只贡献一种亚型，150 张测试切片里 138 张的中心在训练集出现过，分数可能部分来自"认医院"。按中心分组的重训还没做。
> - ⚠️ **多组学和生存分析的 v1 结果有会改变结论的错误**（见下文），v1 产物归档在 `results/archive_v1/`。v2 脚本已重写，并在合成数据上把整条流程跑通，但还需要在原机器上用真实数据重跑。
> - 📋 在数据所在电脑上要做的事，按顺序列在 [TODO.md](TODO.md)。

## 2026-10 审计：问题与改动

### 前期流程（特征 / 5 折训练 / 评估 / 热图）

| 问题 | 影响 | 改动 |
|---|---|---|
| 组织来源中心与标签完全重合 | 分数可能含"认医院"的成分；12 张来自训练集没出现过的中心的切片 AUC 只有 0.69（样本太少，不能下结论） | `eval_cv_summary.py` 量化；`make_strict_splits.py --group-by site` 支持按中心分组 |
| 只报每折 AUC 的均值 | 0.834 偏乐观，也没有不确定性 | 补池化 AUC、患者级 AUC + bootstrap 区间、校准与阈值（`results/eval*/cv_summary.txt`） |
| 倍率没统一（level 0 切 256 像素） | 40× 和 20× 切片的 patch 物理尺度差 4 倍 | `check_slide_mpp.py` 查倍率分布；建议统一到 20× 等效 |
| 严格折的划分依赖 sklearn 版本 | 用 sklearn 1.9.1 重新生成的划分和原来的对不上 | 脚本打印版本；划分文件应随结果入库 |
| 热图标题概率被做了两次 softmax | 显示的概率被压向 0.5；4 张示例热图里 3 张本身就接近 0.5，其中 1 张判错 | `make_heatmap.py` 修正；notebook 补充读图注意事项 |
| 特征 PCA 预览只用了 1 张 LUAD + 1 张 LUSC | "两种癌天然分成两坨"无法和切片/中心差异区分 | notebook 说明 |

### 多组学（§8）

- **表达矩阵可能混入癌旁正常样本**：167 个 RNA-seq 文件对应 144 个患者，v1 按患者覆盖写入。`download_tcga.py` 现在保存文件 ↔ 样本类型（`rna_files.csv`），`common.load_expression` 只取原发肿瘤；没有元数据时会停下并提示。
- **形态嵌入来自 5 个不一致的模型空间**：每张切片用自己测试折的模型提嵌入，PC2–PC5 的方差有 81–97% 可由折号解释（`audit_v1_embedding_folds.py`）。`embed_slides.py` 改为 meanpool（ResNet50 特征直接平均）或每折模型给全部切片提嵌入、只在折内使用。
- **marker 相关**：v1 对每个 marker 挑 |r| 最大的主成分报 p 值，散点图还画错了一列 → 全部 marker × PC 一起做 BH，另报亚型内相关。
- **"Top200 ∩ DE = 141（OR 23.5）说明形态≅分子"是循环论证**：嵌入为分亚型而训练，自然能预测亚型差异基因 → v2 加只用亚型的基线、ΔR²（形态在亚型之外的部分）和亚型内置换零分布。
- **DE 与富集**：log2FC 改为 log2(TPM+1) 均值差（v1 的算术均值版会被极端样本拉大）；ORA 用全部显著基因、以参与检验的基因为背景；可选预排序 GSEA；Keratinization / Surfactant metabolism 作阳性对照。
- v1 的"零泄漏把 R² 从 +0.11 打回 −0.09"不成立：100 张时代剔除含泄漏患者后 R² 反而从 0.11 升到 0.17。

### 生存（§9）

- **v1 请求 GDC 临床数据时漏了 `follow_ups`**：70 个 LUAD 患者里 44 个活着的全部"没有随访"，留下的 26 个 LUAD 全是死亡病例。v1 的 LUAD vs LUSC 曲线、分期 KM、Cox（包括"LUSC HR=0.39"和各个 C-index）都建立在这张有偏的表上，作废。
- v2：TCGA-CDR 优先，否则用带 follow_ups 的 GDC 数据；缺失检查写成程序，不通过就停（仓库现有 clinical.json 的检查结果见 `results/survival/run_log.txt`）；分期取原发诊断、合并为 I / II / III-IV；性别改读 `sex_at_birth`；Cox 以亚型分层，重复交叉验证、bootstrap ΔC，风险分组阈值来自训练折。

### 原理玩具版 notebook（已在本地 CPU 重新执行）

- **TITAN**：迷你 Transformer 没有位置信息，被遮住的 patch 不可能根据邻居"完形填空"，重建误差停在 1.07（相当于只猜整张切片的平均）。加了 1D ALiBi 距离偏置（真实 TITAN 用 2D ALiBi）后降到 0.29，并保留"拿掉位置信息"的对照；提取指纹前补上 `model.eval()`；参数量、训练目标按论文更正（切片编码器是 6 层 ViT，约 4850 万参数；第一阶段是 iBOT 老师-学生自蒸馏，不是像素/特征重建）。
- **CLAM**：加了独立考试卷。肿瘤占比 1% 时注意力的"86.5%"其实是训练准确率，考试卷上接近瞎猜；补充注意力热图的读图限制。
- **UNI2**：ResNet50 的预训练数据是 ImageNet-1k（约 128 万张），不是 1400 万；MLP 参数是 259；iBOT 预测的是老师网络的输出分布而不是像素；"扫描仪痕迹被抹掉"改为"明显变淡但没有消失"；补充基础模型仍编码医院/扫描仪信息的研究。
- **CONCH**：Linear(4→2) 是 10 个参数；解释了对比损失只能降到 log(600) 的原因（同类图注互为假负样本）；文本编码器预训练细节按论文更正。

## 目录结构

```
CLAM-tutorial/
├── README.md
├── data/                        ←（原始数据不入库，用 scripts/download_tcga.py 自行下载）
│   ├── slides/tcga/*.svs        ← TCGA-LUAD/LUSC 诊断切片 ×150（138GB）
│   ├── dataset_csv/             ← CLAM 格式数据集 csv（case_id, slide_id, label）
│   ├── metadata/
│   │   ├── clinical.json        ← GDC 临床数据（144 例；仓库里这份没有 follow_ups，需 --metadata-only 重取）
│   │   ├── rna_files.csv        ← RNA-seq 文件 ↔ 样本类型（--metadata-only 生成）
│   │   ├── TCGA-CDR.csv         ← 可选：TCGA-CDR 终点表（Liu et al. 2018 Cell，自行下载另存）
│   │   └── expression/          ← RNA-seq STAR counts（167 个文件，含癌旁正常/复发样本）
│   └── download_manifest.json   ← 下载断点续传清单
├── scripts/
│   ├── common.py                ← 共用：路径、只取原发肿瘤的表达读取、带 follow_ups/CDR 的生存终点、分期/性别
│   ├── download_tcga.py         ← GDC 下载（断点续传；--metadata-only 只刷新元数据）
│   ├── make_strict_splits.py    ← 严格 5 折（StratifiedGroupKFold；--group-by case|site）
│   ├── eval_cv_summary.py       ← 池化/患者级 AUC、bootstrap 区间、校准、中心混杂
│   ├── check_slide_mpp.py       ← 每张切片的倍率（mpp）
│   ├── make_heatmap.py          ← 轻量注意力热图（复用已有特征）
│   ├── embed_slides.py          ← 形态嵌入：meanpool / 每折 CLAM 模型
│   ├── analysis_multiomics.py   ← marker 体检 + 形态 × marker（BH、亚型基线、按中心分组 CV）
│   ├── analysis_de.py           ← 全基因组 DE + ORA / 预排序 GSEA
│   ├── analysis_morph_predict.py← 全基因组 形态→表达（嵌套 CV、ΔR²、置换零分布）
│   ├── analysis_survival.py     ← 缺失检查 + KM
│   ├── analysis_cox.py          ← Cox：临床 vs 临床+形态（亚型分层、重复 CV、ΔC）
│   └── audit_v1_embedding_folds.py ← 审计 v1 嵌入的折号伪结构
├── notebooks/
│   ├── CLAM_workflow.ipynb      ← 全流程 notebook（主文档）
│   ├── CLAM_原理_玩具版.ipynb    ← MIL + 注意力原理（合成数据，建议先看）
│   ├── UNI2_原理_玩具版.ipynb    ← 自监督特征学习（对比学习版，讲批次效应）
│   ├── CONCH_原理_玩具版.ipynb   ← 图文对齐与零样本分类
│   └── TITAN_原理_玩具版.ipynb   ← 切片级基础模型（遮住重建 + ALiBi + 切片指纹）
├── results/
│   ├── tcga/                    ← patching + features + QC 图（大文件不入库）
│   ├── eval/                    ← MC 版 5 折成绩 + 逐切片预测 + cv_summary.txt
│   ├── eval_strict150/          ← 严格 5 折成绩 + 逐切片预测 + cv_summary.txt
│   ├── heatmaps/                ← 注意力热图（MC150 fold-4 模型，2 LUAD + 2 LUSC）
│   ├── survival/                ← v2 缺失检查在现有 clinical.json 上的输出（按设计停止）
│   ├── archive_v1/              ← v1 多组学 / 生存产物（有已知错误，仅供对照）
│   ├── eval_mc100/              ← 归档：100 张时代 MC 评估
│   ├── heatmaps_mc100/          ← 归档：100 张时代热图
│   ├── multiomics_strict100/    ← 归档：100 张时代严格折多组学
│   └── survival_100/            ← 归档：100 张时代生存分析
└── docs/
    └── clam-setup-log.md        ← 环境搭建 + debug 全记录（末尾有 2026-10 审计勘误）
```

CLAM 本体在同级目录 `../CLAM`（官方仓库 clone + 若干 Windows 兼容补丁，见踩坑实录）。脚本默认用本仓库根目录和同级的 `CLAM` 目录，
可用环境变量 `CLAM_TUTORIAL_ROOT` / `CLAM_DIR` 覆盖。

## 环境

- conda 环境：`clam_latest`（Python 3.10）；torch 2.6.0+cu124 / torchvision 0.21.0+cu124
- 硬件：Windows 11，16GB RAM，RTX 2060 6GB
- v2 分析脚本另需：pandas、scipy、scikit-learn、lifelines；可选 gseapy（预排序 GSEA）、openslide（check_slide_mpp.py）
- 玩具版 notebook 只需 numpy / torch / scikit-learn / matplotlib，CPU 几分钟跑完

## 复现入口

- 从头复现：按 `notebooks/CLAM_workflow.ipynb` 逐节执行（每节有幂等守卫，中断可续跑）。
- v2 多组学 / 生存（需要原始数据在本地）：
  ```
  python scripts/download_tcga.py --metadata-only   # 只查 GDC 元数据：rna_files.csv + 带 follow_ups 的 clinical.json
  python scripts/embed_slides.py
  python scripts/analysis_multiomics.py
  python scripts/analysis_de.py
  python scripts/analysis_morph_predict.py
  python scripts/analysis_survival.py
  python scripts/analysis_cox.py
  ```
- 只看结果：`results/eval_strict150/cv_summary.txt`、`results/eval*/summary.csv`、`results/heatmaps/*.png`。

> 原始切片（138GB）与特征文件（69GB）不入库——`python scripts/download_tcga.py` 可重新下载，后续步骤全部幂等续跑。
