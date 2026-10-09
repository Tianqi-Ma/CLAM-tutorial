# -*- coding: utf-8 -*-
"""审计 v1 形态嵌入：PCA 主成分有多少方差可以由"来自哪一折的模型"解释（2026-10 审计）。

v1 的 morph_embeddings.csv 是每张切片用它自己测试折的 CLAM 模型提的 512 维嵌入（再按患者取平均）。
如果 5 个模型的嵌入空间一致，主成分和折号应该无关；实际上 PC2/PC4/PC5 几乎完全由折号决定。
只读仓库里已有的文件：results/archive_v1/multiomics/morph_embeddings.csv + results/eval_strict150/fold_*.csv。
用法: python scripts/audit_v1_embedding_folds.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import RESULTS, load_dataset, strict_fold_of_case

sys.stdout.reconfigure(encoding='utf-8')
X = pd.read_csv(RESULTS / 'archive_v1' / 'multiomics' / 'morph_embeddings.csv', index_col=0)
dead = X.std(axis=0) < 1e-8
X = X.loc[:, ~dead]
fold = pd.Series(strict_fold_of_case()).loc[X.index].values
label = load_dataset().drop_duplicates('case_id').set_index('case_id')['label'].loc[X.index].values
print(f'v1 嵌入: {X.shape[0]} 患者 × {X.shape[1]} 维（另有 {int(dead.sum())} 个常数维已去掉）')

pca = PCA(10, random_state=0).fit(StandardScaler().fit_transform(X))
Z = pca.transform(StandardScaler().fit_transform(X))


def r2_by(z, g):
    fit = pd.Series(z).groupby(g).transform('mean').values
    return 1 - np.sum((z - fit) ** 2) / np.sum((z - z.mean()) ** 2)


print('\n主成分 | 方差占比 | 折号解释的 R² | 亚型解释的 R²')
for j in range(10):
    print(f'  PC{j + 1:<2d} | {pca.explained_variance_ratio_[j]:6.1%} | {r2_by(Z[:, j], fold):6.1%} | {r2_by(Z[:, j], label):6.1%}')
print('\n折号只是"哪个模型提的嵌入"，和患者的生物学无关；它能解释的方差就是 5 个模型空间不一致造成的假结构。')
