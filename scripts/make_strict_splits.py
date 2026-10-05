# -*- coding: utf-8 -*-
"""严格 k 折划分（教科书版）：替代 CLAM 官方 create_splits_seq 的蒙特卡洛 CV。
用 sklearn StratifiedGroupKFold（group=case_id）：
  - 5 个测试集互不重叠、合起来恰好覆盖全部切片（每张恰好考一次）
  - 同一患者的所有切片永不跨 train/val/test（防"认病人"泄漏）
  - val 从该折训练组里再切 ~12.5%（≈总体 10%），同样按患者分组+类别分层
输出 CLAM 兼容格式 splits_{0..k-1}.csv（train/val/test 三列，短列末尾留空）
用法: python make_strict_splits.py <dataset.csv> <out_dir> [--k 5] [--seed 42]
"""
import argparse, os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument('csv', help='CLAM 格式数据集 csv（case_id, slide_id, label）')
ap.add_argument('out_dir', help='输出目录（splits_0.csv ... 写在这里）')
ap.add_argument('--k', type=int, default=5)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

df = pd.read_csv(args.csv)
groups = df['case_id'].values
y = df['label'].values
X = np.arange(len(df))
n_pat = df['case_id'].nunique()
print(f'数据集: {len(df)} 张切片 / {n_pat} 个患者, 标签分布: {dict(df["label"].value_counts())}')
assert n_pat >= args.k * 2, '患者太少，撑不起这个 k'

os.makedirs(args.out_dir, exist_ok=True)
sgkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
lab = df.set_index('slide_id')['label']
test_of = {}
for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
    sub = df.iloc[tr]
    sg2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=args.seed + k)
    tr2, va2 = next(sg2.split(np.arange(len(sub)), sub['label'].values, sub['case_id'].values))
    train_slides = sub.iloc[tr2]['slide_id'].values
    val_slides = sub.iloc[va2]['slide_id'].values
    test_slides = df.iloc[te]['slide_id'].values
    for s in test_slides:
        assert s not in test_of, f'{s} 进了两次测试集！'
        test_of[s] = k
    gtr = set(sub.iloc[tr2]['case_id']); gva = set(sub.iloc[va2]['case_id']); gte = set(df.iloc[te]['case_id'])
    assert not (gtr & gva) and not (gtr & gte) and not (gva & gte), f'fold{k} 患者跨组泄漏!'
    pd.DataFrame({'train': pd.Series(train_slides),
                  'val': pd.Series(val_slides),
                  'test': pd.Series(test_slides)}).to_csv(f'{args.out_dir}/splits_{k}.csv', index=False)
    print(f'fold {k}: train {len(train_slides)} / val {len(val_slides)} / test {len(test_slides)} | '
          f'test 分布: {dict(lab[test_slides].value_counts())}')
ok = len(test_of) == len(df)
print(f'\n覆盖检查: {len(test_of)}/{len(df)} 张切片各恰好进一次测试集 ->', 'PASS ✅' if ok else 'FAIL ❌')
assert ok
print('已写入:', args.out_dir)
