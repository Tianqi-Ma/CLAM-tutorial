# -*- coding: utf-8 -*-
"""切片级形态嵌入（2026-10 审计后新增，替代 analysis_multiomics.py 里"每张切片用它当 test 那一折模型"的做法）。

为什么要换：严格 5 折的 5 个 CLAM 模型各自训练，参数不同，fc1 + 注意力池化后得到的 512 维空间也不同。
旧做法让每张切片用自己测试折的模型提嵌入，再把 5 个空间的向量拼成一张表做 PCA——主成分里混进了"来自哪一折"。
用 v1 的 morph_embeddings.csv 实测（scripts/audit_v1_embedding_folds.py）：PC2–PC5 的方差有 81~97% 可由折号解释。

两种口径（都不需要重新训练）：
  meanpool   每张切片所有 patch 的 ResNet50 特征（1024 维）直接取平均。不经过任何用标签训练的模型，
             所有切片在同一个空间里，适合做"形态 × 表达 / 生存"的关联分析（默认）。
  clam       每一折的模型给全部 150 张切片提 fc1 + 注意力池化嵌入（512 维），每折一个文件。
             下游分析在折内使用：用第 k 折模型的嵌入，在第 k 折的训练患者上拟合、测试患者上评估，空间始终一致。
             注意它是为了分 LUAD/LUSC 训练出来的，天然带着亚型信息。
产物: results/embeddings/{meanpool,clam_fold0..4}_{slide,patient}.csv
用法: python scripts/embed_slides.py [--mode meanpool clam] [--ckpt-exp luad_lusc_CLAM_sb_strict_s1]
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch

from common import CLAM_DIR, EMB_DIR, FEAT_DIR, load_dataset

sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('--mode', nargs='+', default=['meanpool', 'clam'], choices=['meanpool', 'clam'])
ap.add_argument('--ckpt-exp', default='luad_lusc_CLAM_sb_strict_s1', help='CLAM/results/ 下的实验目录名')
ap.add_argument('--k', type=int, default=5)
args = ap.parse_args()

EMB_DIR.mkdir(parents=True, exist_ok=True)
ds = load_dataset()
slides = list(ds['slide_id'])
sl2case = ds.set_index('slide_id')['case_id']


def save(name, rows):
    X = pd.DataFrame(rows).T
    X.index.name = 'slide_id'
    X.round(6).to_csv(EMB_DIR / f'{name}_slide.csv')
    P = X.groupby(sl2case.loc[X.index].values).mean()   # 同一患者多张切片取平均
    P.index.name = 'case_id'
    P.round(6).to_csv(EMB_DIR / f'{name}_patient.csv')
    dead = int((X.std(axis=0) < 1e-8).sum())
    print(f'  {name}: {X.shape[0]} 张切片 / {P.shape[0]} 个患者 × {X.shape[1]} 维（常数维 {dead} 个，下游会丢弃）')


def feats(s):
    return torch.load(FEAT_DIR / f'{s}.pt', map_location='cpu', mmap=True)


if 'meanpool' in args.mode:
    print('== meanpool：ResNet50 patch 特征直接平均 ==')
    rows = {}
    for i, s in enumerate(slides):
        rows[s] = feats(s).float().mean(0).numpy()
        if i % 25 == 24:
            print(f'  {i + 1}/{len(slides)}')
    save('meanpool', rows)

if 'clam' in args.mode:
    sys.path.insert(0, str(CLAM_DIR))
    from models.model_clam import CLAM_SB
    from topk.svm import SmoothTop1SVM
    print('== clam：每折模型给全部切片提嵌入（fc1 + 门控注意力池化，512 维）==')
    for k in range(args.k):
        m = CLAM_SB(dropout=0., n_classes=2, subtyping=True, embed_dim=1024, instance_loss_fn=SmoothTop1SVM(2))
        m.load_state_dict(torch.load(CLAM_DIR / 'results' / args.ckpt_exp / f's_{k}_checkpoint.pt', map_location='cpu'))
        m.eval()
        rows = {}
        with torch.no_grad():
            for s in slides:
                h = feats(s)
                hh = m.attention_net[2](m.attention_net[1](m.attention_net[0](h)))   # fc1 → ReLU → Dropout(eval 时恒等)
                A, _ = m.attention_net[3](hh)                                       # 门控打分 (N, 1)
                A = torch.softmax(A, dim=0)
                rows[s] = (A * hh).sum(0).numpy()                                   # 注意力池化 → 512 维
        save(f'clam_fold{k}', rows)
print('✅ 嵌入已写入', EMB_DIR)
