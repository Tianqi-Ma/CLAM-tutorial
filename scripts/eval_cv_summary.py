# -*- coding: utf-8 -*-
"""交叉验证成绩的补充评估（2026-10 审计后新增；只读 eval.py 已经写好的 fold_k.csv，几秒跑完）。

eval.py 的 summary.csv 只有"每折 AUC 再取平均"。这里补上：
  1. 池化 OOF AUC：严格 5 折里每张切片恰好测一次，把 5 折的预测拼起来算一个 AUC。
     各折模型的输出概率尺度不同，池化 AUC 通常低于每折 AUC 的平均，两者都该报。
  2. 患者级 AUC：同一患者的多张切片取 p_1 平均，再以患者为单位算 AUC + 患者 bootstrap 95% 区间。
  3. 校准与阈值：两类的平均预测概率、Brier 分数、0.5 阈值下的灵敏度/特异度（AUC 高而准确率低 = 阈值没校准）。
  4. 组织来源中心（TSS）混杂：本数据集每个中心只贡献一种亚型。统计测试切片所在中心是否在训练折出现过，
     并分别计算"见过的中心 / 没见过的中心"上的 AUC。模型可能部分在认"哪家医院的染色"，
     严格评估应该用按中心分组的划分（make_strict_splits.py --group-by site，Howard et al. 2021 Nat Commun）。
用法: python scripts/eval_cv_summary.py [--eval-dir results/eval_strict150] [--k 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from common import ROOT, load_dataset

sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('--eval-dir', default=str(ROOT / 'results' / 'eval_strict150'))
ap.add_argument('--k', type=int, default=5)
ap.add_argument('--n-boot', type=int, default=2000)
args = ap.parse_args()

ds = load_dataset().set_index('slide_id')
rows = []
for k in range(args.k):
    f = pd.read_csv(Path(args.eval_dir) / f'fold_{k}.csv')
    f['fold'] = k
    rows.append(f)
P = pd.concat(rows, ignore_index=True)
P['case_id'] = P['slide_id'].map(ds['case_id'])
P['tss'] = P['slide_id'].map(ds['tss'])
P['Y'] = P['Y'].astype(int)
n_unique = P['slide_id'].nunique()
print(f'{args.eval_dir}: {len(P)} 条预测 / {n_unique} 张不同切片 / {P["case_id"].nunique()} 个患者')
if n_unique < len(P):
    print('  ⚠️ 有切片在多个测试折出现（蒙特卡洛 CV），下面的"池化"只是把各折预测堆在一起，解读要谨慎')
if n_unique < len(ds):
    print(f'  ⚠️ 只覆盖 {n_unique}/{len(ds)} 张切片，其余从没进过测试集')

per_fold = pd.Series({k: roc_auc_score(g['Y'], g['p_1']) for k, g in P.groupby('fold')})
print(f"\n每折 AUC: {' / '.join(f'{v:.3f}' for v in per_fold)} → 均值 {per_fold.mean():.3f} ± {per_fold.std(ddof=1):.3f}")
print(f"池化切片级 AUC: {roc_auc_score(P['Y'], P['p_1']):.3f}（n={len(P)}）")

pat = P.groupby('case_id').agg(Y=('Y', 'first'), p=('p_1', 'mean'))
auc_pat = roc_auc_score(pat['Y'], pat['p'])
rng = np.random.default_rng(0)
boot = []
for _ in range(args.n_boot):
    s = pat.iloc[rng.integers(0, len(pat), len(pat))]
    if s['Y'].nunique() == 2:
        boot.append(roc_auc_score(s['Y'], s['p']))
print(f'患者级 AUC: {auc_pat:.3f}，bootstrap 95% 区间 [{np.quantile(boot, 0.025):.3f}, {np.quantile(boot, 0.975):.3f}]'
      f'（n={len(pat)}）')

pred = (P['p_1'] >= 0.5).astype(int)
sens = ((pred == 1) & (P['Y'] == 1)).sum() / (P['Y'] == 1).sum()
spec = ((pred == 0) & (P['Y'] == 0)).sum() / (P['Y'] == 0).sum()
print(f"\n校准: 平均 p(LUSC) | 真 LUAD {P.loc[P['Y'] == 0, 'p_1'].mean():.3f} / 真 LUSC {P.loc[P['Y'] == 1, 'p_1'].mean():.3f}；"
      f"Brier {brier_score_loss(P['Y'], P['p_1']):.3f}")
print(f'阈值 0.5: 准确率 {(pred == P["Y"]).mean():.3f}，LUSC 灵敏度 {sens:.3f}，LUAD 特异度 {spec:.3f}，'
      f'预测为 LUSC 的比例 {pred.mean():.2f}（真实 {P["Y"].mean():.2f}）')

# 组织来源中心
lab_by_tss = ds.groupby('tss')['label'].nunique()
print(f'\n== 组织来源中心（TSS）==\n{ds["tss"].nunique()} 个中心，其中只有一种亚型的 {int((lab_by_tss == 1).sum())} 个，'
      f'覆盖 {int(ds["tss"].map(lab_by_tss).eq(1).sum())}/{len(ds)} 张切片')
if P['slide_id'].nunique() == len(P):
    seen = P.apply(lambda r: (P.loc[P['fold'] != r['fold'], 'tss'] == r['tss']).any(), axis=1)
    print(f'测试切片所在中心在其他折（训练侧）出现过: {int(seen.sum())}/{len(P)}')
    for flag, name in ((True, '见过的中心'), (False, '没见过的中心')):
        sub = P[seen == flag]
        auc = roc_auc_score(sub['Y'], sub['p_1']) if sub['Y'].nunique() == 2 else float('nan')
        print(f"  {name}: n={len(sub)}（LUAD {int((sub['Y'] == 0).sum())} / LUSC {int((sub['Y'] == 1).sum())}），AUC {auc:.3f}")
    print('  没见过的中心样本太少，区间很宽，不能据此下结论；要回答"模型是否在认中心"，需按中心分组重训。')
