# -*- coding: utf-8 -*-
"""多组学③：形态在亚型之外还能预测哪些基因？（v2，2026-10 审计后重写）

v1 做法与问题：形态嵌入 → PC1-10 → 岭回归 → 留一法 R²，再看"R² 前 200 名"和"LUAD/LUSC 差异基因"的重叠
（141/200，Fisher OR=23.5），解读为"形态与分子同构"。但嵌入来自为分 LUAD/LUSC 训练的 CLAM 模型，
天然编码亚型；任何带亚型信息的特征都能预测亚型差异基因，这个重叠几乎是必然的，不是独立证据。
另外 PCA 用了 5 个不同折模型的嵌入（空间不一致），α 在报告 R² 的同一批 LOO 结果上挑选（略乐观），α 网格最大值 100 正好被选中（网格不够）。

v2 做法：
  - 嵌入默认 meanpool（同一空间、不用标签）；--embedding clam 时按严格 5 折在各折模型空间内拟合与评估。
  - 每个基因三个模型：只用亚型标签 / 只用形态 PC / 亚型 + 形态。交叉验证外层 5 折；
    PCA、标准化、α（每个基因各自用训练集上的闭式 LOO 选）都只在训练折内拟合。
  - ΔR² = R²(亚型+形态) − R²(只用亚型)：形态在亚型之外的预测力。
    零分布：在亚型内部打乱形态嵌入（保留亚型与形态的关系、切断形态与表达的逐例对应），重复 --n-perm 次。
  - 重叠检验两份都报：top-200 R²(形态) × DE（预期很高，因为二者都由亚型驱动），top-200 ΔR² × DE。
  - 两个基因程序分数：细胞溶解活性 CYT（GZMA、PRF1 的 log 均值，Rooney et al. 2015 Cell）与增殖分数。
产物: results/multiomics/morph_predict_r2.csv, morph_predict_programs.csv, morph_predict_hist.png
用法: python scripts/analysis_morph_predict.py [--embedding meanpool|clam] [--n-pc 10] [--n-perm 5]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import (RESULTS, log_to, describe_expression_report, load_dataset, load_expression, load_patient_embedding,
                    setup_matplotlib, strict_fold_of_case)

sys.stdout.reconfigure(encoding='utf-8')
plt = setup_matplotlib()
ap = argparse.ArgumentParser()
ap.add_argument('--embedding', default='meanpool', choices=['meanpool', 'clam'])
ap.add_argument('--n-pc', type=int, default=10)
ap.add_argument('--n-perm', type=int, default=5)
ap.add_argument('--min-frac', type=float, default=0.2)
ap.add_argument('--allow-ambiguous', action='store_true')
args = ap.parse_args()
OUT = RESULTS / 'multiomics'
OUT.mkdir(parents=True, exist_ok=True)
log_to(OUT / 'morph_log.txt')
ALPHAS = np.logspace(-2, 5, 22)
PROGRAMS = {'CYT（GZMA, PRF1）': ['GZMA', 'PRF1'],
            '增殖（MKI67, TOP2A, BIRC5, CCNB1, CDK1, MCM2）': ['MKI67', 'TOP2A', 'BIRC5', 'CCNB1', 'CDK1', 'MCM2']}

# ---------- 数据 ----------
ds = load_dataset()
label_of = ds.drop_duplicates('case_id').set_index('case_id')['label']
E, rep = load_expression(cases=label_of.index, strict=not args.allow_ambiguous)
print('表达矩阵:', describe_expression_report(rep))
if args.embedding == 'meanpool':
    Xdf = load_patient_embedding('meanpool')
    cases = [c for c in Xdf.index if c in E.columns]
    Xk = [Xdf.loc[cases].values] * 5
    splits = list(KFold(5, shuffle=True, random_state=0).split(cases))
else:
    fold_of = strict_fold_of_case()
    Xs = [load_patient_embedding(f'clam_fold{k}') for k in range(5)]
    cases = [c for c in Xs[0].index if c in E.columns and c in fold_of]
    Xk = [X.loc[cases].values for X in Xs]
    folds = np.array([fold_of[c] for c in cases])
    splits = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(5)]
E = E[cases]
genes = E.index[(E >= 1).mean(axis=1) >= args.min_frac]
Y = np.log2(E.loc[genes].values.T + 1)                       # (n, g)
lab = np.array([label_of[c] == 'LUSC' for c in cases], dtype=float)
print(f'{len(cases)} 患者 × {len(genes)} 基因（≥{args.min_frac:.0%} 患者 TPM≥1）；嵌入 {args.embedding}')


# ---------- 闭式岭回归：每个基因各自用训练集 LOO 选 α ----------
def ridge_fit(F, Yt):
    Fm, Ym = F.mean(0), Yt.mean(0)
    U, s, Vt = np.linalg.svd(F - Fm, full_matrices=False)
    UtY = U.T @ (Yt - Ym)
    resid0 = (Yt - Ym) - U @ UtY                              # 不在 F 列空间里的部分，任何 α 都去不掉
    n = len(F)
    best_sse = np.full(Yt.shape[1], np.inf)
    best_a = np.zeros(Yt.shape[1])
    for a in ALPHAS:
        d = s ** 2 / (s ** 2 + a)
        fit_res = resid0 + U @ ((1 - d)[:, None] * UtY)       # 训练残差
        h = (U ** 2) @ d + 1.0 / n                            # 帽子矩阵对角线（含截距）
        sse = ((fit_res / (1 - h)[:, None]) ** 2).sum(0)      # 闭式 LOO 残差平方和
        better = sse < best_sse
        best_sse[better] = sse[better]
        best_a[better] = a
    B = np.empty((F.shape[1], Yt.shape[1]))
    for a in np.unique(best_a):
        idx = best_a == a
        B[:, idx] = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY[:, idx])
    return Fm, Ym, B


def build(mode, X, tr, te):
    parts_tr, parts_te = [], []
    if mode in ('label', 'both'):
        parts_tr.append(lab[tr, None])
        parts_te.append(lab[te, None])
    if mode in ('morph', 'both'):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(args.n_pc, random_state=0).fit(sc.transform(X[tr]))
        parts_tr.append(pca.transform(sc.transform(X[tr])))
        parts_te.append(pca.transform(sc.transform(X[te])))
    F_tr, F_te = np.hstack(parts_tr), np.hstack(parts_te)
    sc = StandardScaler().fit(F_tr)
    return sc.transform(F_tr), sc.transform(F_te)


def cv_r2(mode, Yt, Xfold):
    pred = np.empty_like(Yt)
    for k, (tr, te) in enumerate(splits):
        F_tr, F_te = build(mode, Xfold[k], tr, te)
        Fm, Ym, B = ridge_fit(F_tr, Yt[tr])
        pred[te] = (F_te - Fm) @ B + Ym
    sst = ((Yt - Yt.mean(0)) ** 2).sum(0)
    return 1 - ((Yt - pred) ** 2).sum(0) / sst


print('\n交叉验证中（3 个模型 × 5 折）...')
r2 = {m: cv_r2(m, Y, Xk) for m in ('label', 'morph', 'both')}
res = pd.DataFrame({'R2_label': r2['label'], 'R2_morph': r2['morph'], 'R2_label_morph': r2['both']}, index=genes)
res['dR2_morph_beyond_label'] = res['R2_label_morph'] - res['R2_label']
res.index.name = 'gene'

# 零分布：亚型内打乱形态嵌入
rng = np.random.default_rng(0)
null = []
for i in range(args.n_perm):
    perm = np.arange(len(cases))
    for v in (0.0, 1.0):
        idx = np.where(lab == v)[0]
        perm[idx] = rng.permutation(idx)
    null.append(cv_r2('both', Y, [X[perm] for X in Xk]) - res['R2_label'].values)
null = np.concatenate(null) if null else np.array([0.0])
thr = np.quantile(null, 0.99)
res['dR2_above_null99'] = res['dR2_morph_beyond_label'] > thr
res = res.sort_values('dR2_morph_beyond_label', ascending=False)
res.round(4).to_csv(OUT / 'morph_predict_r2.csv')

print(f"\n== 全基因组结果（{len(res)} 个基因）==")
print(f"  R²(只用形态) > 0.1 的基因: {(res['R2_morph'] > 0.1).sum()}；R²(只用亚型) > 0.1: {(res['R2_label'] > 0.1).sum()}")
print(f"  ΔR²（形态在亚型之外）中位数 {res['dR2_morph_beyond_label'].median():+.4f}；"
      f"零分布 99% 分位 {thr:+.4f}；超过的基因 {int(res['dR2_above_null99'].sum())} 个"
      f"（纯随机预期约 {0.01 * len(res):.0f} 个）")
print('  ΔR² 最大的 15 个基因:')
print(res.head(15)[['R2_label', 'R2_morph', 'R2_label_morph', 'dR2_morph_beyond_label']].round(3).to_string())

# ---------- 与 DE 的重叠：两种排序都报 ----------
de_path = OUT / 'de_table.csv'
if de_path.exists():
    de = pd.read_csv(de_path, index_col=0)
    universe = set(res.index) & set(de.index)
    de_sig = set(de[(de['q_BH'] < 0.05) & (de['log2FC_LUAD_vs_LUSC'].abs() > 1)].index) & universe
    print(f'\n== top-200 × DE 显著基因（共同基因宇宙 {len(universe)}，DE 显著 {len(de_sig)}）==')
    for col, note in [('R2_morph', '重叠高也不是独立证据：形态带亚型、DE 本身就是亚型差异'),
                      ('dR2_morph_beyond_label', '看形态在亚型之外能预测的基因是否仍偏向亚型差异基因')]:
        ranked = [g for g in res.sort_values(col, ascending=False).index if g in universe]
        top = set(ranked[:200])
        a = len(top & de_sig)
        table = [[a, 200 - a], [len(de_sig) - a, len(universe) - 200 - len(de_sig) + a]]
        orr, pf = fisher_exact(table)
        print(f'  按 {col:24s} 排序: 重叠 {a}/200，OR={orr:.1f}，p={pf:.1e} —— {note}')

# ---------- 基因程序分数 ----------
L = np.log2(E + 1)
prog_rows = []
print('\n== 基因程序分数 ==')
for name, gl in PROGRAMS.items():
    gl = [g for g in gl if g in L.index]
    if len(gl) < 2:
        continue
    z = L.loc[gl].sub(L.loc[gl].mean(axis=1), axis=0).div(L.loc[gl].std(axis=1), axis=0)
    score = z.mean(axis=0).values[:, None]
    rr = {m: cv_r2(m, score, Xk)[0] for m in ('label', 'morph', 'both')}
    prog_rows.append({'program': name, 'genes': ','.join(gl), 'R2_label': rr['label'], 'R2_morph': rr['morph'],
                      'R2_label_morph': rr['both'], 'dR2_morph_beyond_label': rr['both'] - rr['label']})
    print(f"  {name}: 亚型 {rr['label']:+.3f} | 形态 {rr['morph']:+.3f} | 亚型+形态 {rr['both']:+.3f}")
pd.DataFrame(prog_rows).round(4).to_csv(OUT / 'morph_predict_programs.csv', index=False)

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(res['R2_label'].clip(-0.5, 1), res['R2_morph'].clip(-0.5, 1), s=3, alpha=0.3, c='#7a5195')
ax.plot([-0.5, 1], [-0.5, 1], 'k--', lw=0.8)
ax.set_xlabel('CV R²：只用亚型标签')
ax.set_ylabel('CV R²：只用形态 PC')
ax.set_title('每个点一个基因：只用形态 vs 只用亚型（落在对角线下方 = 亚型标签就够了）')
ax = axes[1]
ax.hist(np.clip(null, -0.1, 0.2), bins=80, density=True, alpha=0.5, color='gray', label='零分布（亚型内打乱形态）')
ax.hist(res['dR2_morph_beyond_label'].clip(-0.1, 0.2), bins=80, density=True, alpha=0.6, color='#7a5195',
        label='实际 ΔR²')
ax.axvline(thr, c='k', ls='--', lw=0.8, label='零分布 99% 分位')
ax.set_xlabel('ΔR² = R²(亚型+形态) − R²(只用亚型)')
ax.set_title('形态在亚型之外的预测力')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'morph_predict_hist.png', dpi=110, bbox_inches='tight')
plt.close(fig)
print('\n✅ 产物已写入', OUT)
