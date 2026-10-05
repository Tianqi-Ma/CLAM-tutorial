# CLAM-tutorial

在 Windows 11 上从零复现 CLAM（mahmoodlab/CLAM）WSI 弱监督分析流水线的完整工程。
配套踩坑实录见 [docs/clam-setup-log.md](docs/clam-setup-log.md)。

> ✅ **150 张时代全流程已完成（2026-10-05）**：150 张 TCGA（LUAD/LUSC 各 75，144 名患者，138GB）→
> 842 万 patch → ResNet50 特征（69GB）→ CLAM-SB **双协议 5 折交叉验证**：
> CLAM 默认蒙特卡洛 CV **AUC 0.788 ± 0.181** ｜ 严格 5 折（每患者恰考一次）**AUC 0.834 ± 0.090**。
> 加分项（全部零泄漏嵌入）：全基因组 DE（1,526 个显著基因）→ 形态预测表达（Top200∩DE = 141，Fisher OR=23.5）
> → Cox 预后建模（OOF C-index 0.50，诚实阴性结果）→ 生存分析（分期 KM + 缺失数据陷阱演示）。
> 逐切片预测明细见 `results/eval*/`，注意力热图见 `results/heatmaps/`。

## 目录结构

```
CLAM-tutorial/
├── README.md                    ← 本文件
├── data/                        ←（原始数据不入库，用 scripts/download_tcga.py 自行下载）
│   ├── slides/
│   │   ├── cmu/CMU-1.svs        ← OpenSlide 官方测试切片（冒烟测试用）
│   │   └── tcga/*.svs           ← TCGA-LUAD/LUSC 诊断切片 ×150（138GB）
│   ├── dataset_csv/             ← CLAM 格式数据集 csv（case_id, slide_id, label）
│   ├── metadata/
│   │   ├── clinical.json        ← GDC 临床数据（144 例）
│   │   └── expression/          ← RNA-seq STAR counts（167 个文件）
│   └── download_manifest.json   ← 下载断点续传清单
├── scripts/
│   ├── download_tcga.py         ← GDC 下载脚本（断点续传；扩容只增不换；GDC_PROXY 走代理）
│   ├── make_strict_splits.py    ← 教科书式严格 5 折（StratifiedGroupKFold，患者零跨组）
│   ├── make_heatmap.py          ← 轻量注意力热图（复用已有特征，秒级/张）
│   ├── analysis_multiomics.py   ← 多组学：形态嵌入 × marker 基因（严格折零泄漏）
│   ├── analysis_de.py           ← 全基因组差异表达 + Enrichr 富集
│   ├── analysis_morph_predict.py← 形态→表达全基因组留一法预测 + Fisher 重叠检验
│   ├── analysis_survival.py     ← 生存分析：KM 曲线 + 缺失数据陷阱演示
│   └── analysis_cox.py          ← Cox 预后建模：单因素 + OOF C-index + 风险分层 KM
├── notebooks/
│   ├── CLAM_workflow.ipynb      ← 全流程 notebook（逐步复现 + 双协议成绩，GitHub 主文档）
│   ├── CLAM_原理_玩具版.ipynb    ← MIL+注意力原理（合成数据，建议先看；含手算节）
│   ├── UNI2_原理_玩具版.ipynb    ← 自监督特征学习（DINOv2 思想的对比学习版）
│   ├── CONCH_原理_玩具版.ipynb   ← 图文对齐与零样本分类
│   └── TITAN_原理_玩具版.ipynb   ← 切片级基础模型（遮住重建自监督）
├── results/
│   ├── cmu_test/                ← 冒烟测试产物
│   ├── tcga/                    ← patching（842 万 patch 坐标）+ features（69GB）+ QC图 ★ 大文件不入库
│   ├── eval/                    ← MC 版 5 折成绩单 + 逐切片预测明细 ★
│   ├── eval_strict150/          ← 严格 5 折成绩单 ★
│   ├── heatmaps/                ← 注意力热图（MC150 fold-4 模型，2 LUAD + 2 LUSC）★
│   ├── multiomics/              ← 箱线图/散点图/火山图/富集图/预测直方图 + de_table.csv ★
│   ├── survival/                ← 分期 KM + 缺失陷阱演示 + Cox 汇总 + 风险分层 KM ★
│   ├── eval_mc100/              ← 归档：100 张时代 MC 评估（AUC 0.919±0.065）
│   ├── heatmaps_mc100/          ← 归档：100 张时代热图 ×11
│   ├── multiomics_strict100/    ← 归档：100 张严格折多组学
│   └── survival_100/            ← 归档：100 张时代生存分析
├── wheels/                      ← 离线下载的大 wheel（torch CUDA 版等，不入库）
└── docs/
    └── clam-setup-log.md        ← 环境搭建 + debug 全记录（文章素材，20+ 个坑）
```

CLAM 本体在同级目录 `../CLAM`（官方仓库 clone + 若干 Windows 兼容补丁，全部补丁见踩坑实录）。

## 环境

- conda 环境：`clam_latest`（Python 3.10）
- torch 2.6.0+cu124 / torchvision 0.21.0+cu124（本地 wheel 安装在 `wheels/`）
- 硬件：Windows 11，16GB RAM，RTX 2060 6GB
- 详细环境清单见 docs/clam-setup-log.md

## 当前进度

- [x] 环境搭建（含 8 个坑的完整 debug 记录）
- [x] CMU-1 冒烟测试：分割+切块 8.4s（4679 patches）、特征提取 35.8s（RTX 2060）
- [x] TCGA-LUAD/LUSC 数据下载（75+75 张 138GB，含临床 + 表达谱；GDC 晚高峰限速 → 代理 6MB/s）
- [x] TCGA 批量 patching（842 万 patch）+ 特征提取（ResNet50，69GB；双进程并行约 2.8 分钟/张）
- [x] LUAD vs LUSC 亚型分类训练（CLAM-SB，**双协议 5 折**：MC + 严格，按患者分层）
- [x] 评估：**MC 0.788 ± 0.181**（0.918/0.968/0.551/0.643/0.857）｜**严格 0.834 ± 0.090**（0.855/0.788/0.936/0.705/0.889）
- [x] 注意力热图（自研轻量版，MC150 fold-4 测试集出图）
- [x] 预处理可视化：mask-vs-stitch 质检对比图 + 特征数据预览（小白向）
- [x] 多组学加深：marker 体检（p 最低 3e-20）→ 形态主成分×marker（|r|=0.32~0.35）→ 全基因组 DE（1,526 显著基因）→ 形态预测表达（Fisher OR=23.5, p=5e-95）→ 反泄漏实证（零泄漏 R²=-0.09 戳破 100 张时代 +0.11 假阳性）
- [x] 生存分析加深：分期 KM（I vs III p=0.26）+ 缺失数据陷阱演示（26 例 Alive LUAD 无随访 → 假显著）+ Cox 预后建模（OOF C-index 0.50，诚实阴性）

## 复现入口

想从头复现：按 `notebooks/CLAM_workflow.ipynb` 逐节执行（每节有幂等守卫，中断可续跑）。
想直接看结果：`results/eval/summary.csv` + `results/eval_strict150/summary.csv` + `results/heatmaps/*.png`。

> 原始切片（138GB）与特征文件（69GB）不入库——`python scripts/download_tcga.py` 可一键重新下载，
> 后续步骤全部幂等续跑。
