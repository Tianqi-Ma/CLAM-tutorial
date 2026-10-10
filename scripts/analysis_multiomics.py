# -*- coding: utf-8 -*-
"""多组学①：marker 体检 + 形态嵌入 × marker 表达（v2，2026-10 审计后重写）

v1 的问题与 v2 的改法：
  - 表达：v1 按患者覆盖写入，多文件患者可能用上癌旁正常样本 → 只取原发肿瘤（common.load_expression）。
  - 嵌入：v1 每张切片用自己测试折的 CLAM 模型提嵌入，5 个模型的空间不同，PCA 主成分混进了折号
    （PC2–PC5 有 81~97% 方差由折号解释，见 audit_v1_embedding_folds.py）→ 默认改用 meanpool（ResNet50 特征直接平均，所有切片同一空间，
    且不经过用 LUAD/LUSC 标签训练的模型）；--embedding clam 时每折用自己的模型空间，在折内拟合与评估。
  - 相关：v1 对每个 marker 在 PC1-5 里挑 |r| 最大的一个报告 p 值（挑选后 p 值偏小），散点图还画错了一列
    （Z[:, pc_show] 用 1 起的编号去索引 0 起的数组）→ 报告全部 marker × PC 的相关并做 BH 校正。
  - 预测：v1 只看"形态 → marker"的 R²。可 marker 本身就是亚型标志，形态嵌入又带着亚型，R² 高低主要反映亚型。
    v2 同时给出 只用亚型标签 / 只用形态 / 亚型+形态 三个模型，ΔR² = 亚型+形态 − 只用亚型 才是"形态在亚型之外"的信息；
    交叉验证有随机 5 折和按组织来源中心（TSS）分组两种，后者检验结论是不是靠中心特有的染色/批次撑起来的。
产物: results/multiomics/markers_boxplot.png, morph_vs_expr.png, morph_marker_assoc.csv, morph_marker_predict.csv, run_log.txt
      （--embedding clam 时文件名加 _clam 后缀，且不做只在同一空间里才有意义的 PC 关联）
用法: python scripts/analysis_multiomics.py [--embedding meanpool|clam] [--n-pc 10] [--allow-ambiguous]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from common import (RESULTS, log_to, bh_fdr, describe_expression_report, load_dataset, load_expression,
                    load_patient_embedding, setup_matplotlib, strict_fold_of_case)

sys.stdout.reconfigure(encoding='utf-8')
plt = setup_matplotlib()
ap = argparse.ArgumentParser()
ap.add_argument('--embedding', default='meanpool', choices=['meanpool', 'clam'])
ap.add_argument('--n-pc', type=int, default=10)
ap.add_argument('--allow-ambiguous', action='store_true', help='没有 rna_files.csv 时剔除多文件患者继续')
args = ap.parse_args()
OUT = RESULTS / 'multiomics'
OUT.mkdir(parents=True, exist_ok=True)
TAG = '' if args.embedding == 'meanpool' else '_clam'      # clam 模式的产物加后缀，不覆盖默认结果
log_to(OUT / f'run_log{TAG}.txt')
MARKERS = {'NKX2-1': 'LUAD（TTF-1）', 'NAPSA': 'LUAD（Napsin A）', 'TP63': 'LUSC（p40/p63）',
           'KRT5': 'LUSC（CK5/6）', 'KRT6A': 'LUSC（CK5/6）', 'SOX2': 'LUSC'}
ALPHAS = np.logspace(-2, 4, 25)

# ---------- 1. 表达 + marker 体检 ----------
ds = load_dataset()
label_of = ds.drop_duplicates('case_id').set_index('case_id')['label']
tss_of = ds.drop_duplicates('case_id').set_index('case_id')['tss']
E, rep = load_expression(cases=label_of.index, strict=not args.allow_ambiguous)
print('表达矩阵:', describe_expression_report(rep))
print(f'  → {E.shape[0]} 个蛋白编码基因 × {E.shape[1]} 个患者')
logE = np.log2(E + 1)
markers = [g for g in MARKERS if g in logE.index]

print('\n== marker 体检（log2(TPM+1) 中位数，Mann-Whitney）==')
fig, axes = plt.subplots(1, len(markers), figsize=(3.2 * len(markers), 3.8))
for ax, g in zip(np.atleast_1d(axes), markers):
    a = logE.loc[g, [c for c in E.columns if label_of[c] == 'LUAD']]
    b = logE.loc[g, [c for c in E.columns if label_of[c] == 'LUSC']]
    p = mannwhitneyu(a, b).pvalue
    ax.boxplot([a, b], showfliers=False)
    ax.set_xticks([1, 2], ['LUAD', 'LUSC'])
    ax.set_title(f'{g}  {MARKERS[g]}\np={p:.1e}', fontsize=9)
    ax.set_ylabel('log2(TPM+1)')
    print(f'  {g:7s} {MARKERS[g]:14s}: LUAD {a.median():.2f} vs LUSC {b.median():.2f}, p={p:.1e}')
fig.suptitle('已知 LUAD / LUSC marker 在标签分组下的表达（方向对 = 标签和表达数据对得上）', fontsize=10)
fig.tight_layout()
fig.savefig(OUT / f'markers_boxplot{TAG}.png', dpi=110, bbox_inches='tight')
plt.close(fig)


# ---------- 2. 交叉验证工具 ----------
def fit_predict(F_tr, y_tr, F_te):
    sc = StandardScaler().fit(F_tr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(F_tr), y_tr)     # α 在训练集内部用 LOO 选（嵌套）
    return m.predict(sc.transform(F_te))


def features(mode, X_tr, X_te, lab_tr, lab_te):
    """mode: label / morph / both。PCA 只在训练集上拟合。"""
    out_tr, out_te = [], []
    if mode in ('label', 'both'):
        out_tr.append(lab_tr[:, None])
        out_te.append(lab_te[:, None])
    if mode in ('morph', 'both'):
        sc = StandardScaler().fit(X_tr)
        pca = PCA(min(args.n_pc, X_tr.shape[0] - 1), random_state=0).fit(sc.transform(X_tr))
        out_tr.append(pca.transform(sc.transform(X_tr)))
        out_te.append(pca.transform(sc.transform(X_te)))
    return np.hstack(out_tr), np.hstack(out_te)


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def cv_r2(X_of_fold, lab, y, splits, mode):
    """X_of_fold(k) 返回第 k 折使用的嵌入矩阵（meanpool 各折相同；clam 每折一个空间）。"""
    pred = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(splits):
        X = X_of_fold(k)
        F_tr, F_te = features(mode, X[tr], X[te], lab[tr], lab[te])
        pred[te] = fit_predict(F_tr, y[tr], F_te)
    return r2(y, pred)


# ---------- 3. 嵌入 ----------
if args.embedding == 'meanpool':
    Xdf = load_patient_embedding('meanpool')
    cases = [c for c in Xdf.index if c in E.columns]
    Xall = Xdf.loc[cases].values
    X_of_fold = lambda k: Xall
    print(f'\n形态嵌入: meanpool（ResNet50 特征平均）{len(cases)} 患者 × {Xall.shape[1]} 维（已去常数列）')
else:
    fold_of = strict_fold_of_case()
    Xs = [load_patient_embedding(f'clam_fold{k}') for k in range(5)]
    cases = [c for c in Xs[0].index if c in E.columns and c in fold_of]
    Xk = [X.loc[cases].values for X in Xs]
    X_of_fold = lambda k: Xk[k]
    print(f'\n形态嵌入: CLAM 每折模型空间（折内拟合/评估）{len(cases)} 患者')
lab = np.array([label_of[c] == 'LUSC' for c in cases], dtype=float)
tss = np.array([tss_of[c] for c in cases])
Y = logE.loc[markers, cases].T.values

# ---------- 4. 关联：全部 marker × PC，BH 校正（只在 meanpool 下做；clam 的空间每折不同） ----------
assoc = None
if args.embedding == 'meanpool':
    Xs_all = StandardScaler().fit_transform(Xall)
    pca_all = PCA(args.n_pc, random_state=0).fit(Xs_all)
    evr = pca_all.explained_variance_ratio_
    Z = pca_all.transform(Xs_all)
    rows = []
    for gi, g in enumerate(markers):
        for j in range(args.n_pc):
            r_all, p_all = spearmanr(Z[:, j], Y[:, gi])
            r_ad = spearmanr(Z[lab == 0, j], Y[lab == 0, gi])[0]
            r_sc = spearmanr(Z[lab == 1, j], Y[lab == 1, gi])[0]
            rows.append({'gene': g, 'pc': j + 1, 'rho': r_all, 'p': p_all, 'rho_within_LUAD': r_ad,
                         'rho_within_LUSC': r_sc})
    assoc = pd.DataFrame(rows)
    assoc['q_BH'] = bh_fdr(assoc['p'])
    assoc.round({'rho': 4, 'rho_within_LUAD': 4, 'rho_within_LUSC': 4}).to_csv(OUT / 'morph_marker_assoc.csv', index=False)
    print(f'\n== 形态 PC1-{args.n_pc} × marker（Spearman，{len(assoc)} 次检验统一做 BH）==')
    for g in markers:
        sub = assoc[assoc['gene'] == g].sort_values('p').iloc[0]
        print(f"  {g:7s}: 最强 PC{int(sub['pc'])} ρ={sub['rho']:+.2f}, q={sub['q_BH']:.1e}"
              f"（亚型内: LUAD ρ={sub['rho_within_LUAD']:+.2f} / LUSC ρ={sub['rho_within_LUSC']:+.2f}）")
    print('  读法: 如果亚型内 ρ 明显变小，相关主要来自"两个亚型形态不同、表达也不同"，而不是形态逐例跟着表达变。')

    # 主成分由什么解释：亚型 vs 组织来源中心（TSS 嵌套在亚型里，中心的 R² 已包含亚型）。
    # 43 个中心的自由度很多，普通 R² 纯靠运气也有 ~30%，所以报调整 R²（≤0 = 没有超出随机的解释力）。
    def adj_r2(z, g):
        n, k = len(z), len(set(g))
        fit = pd.Series(z).groupby(g).transform('mean').values
        r2 = 1 - np.sum((z - fit) ** 2) / np.sum((z - z.mean()) ** 2)
        return 1 - (1 - r2) * (n - 1) / (n - k)
    print('\n== 各主成分的方差由谁解释（单因素方差分析的调整 R²）==')
    for j in range(min(5, args.n_pc)):
        print(f'  PC{j + 1}（方差 {evr[j]:.1%}）: 亚型 {adj_r2(Z[:, j], lab):+.1%} | '
              f'组织来源中心 {adj_r2(Z[:, j], tss):+.1%}（含亚型；中心明显高于亚型 = 有亚型之外的中心效应）')

# ---------- 5. 预测：亚型 / 形态 / 亚型+形态，两种 CV ----------
schemes = {}
if args.embedding == 'meanpool':
    schemes['随机5折×5次'] = [list(KFold(5, shuffle=True, random_state=s).split(Y)) for s in range(5)]
    schemes['按中心分组5折'] = [list(GroupKFold(5).split(Y, groups=tss))]
else:
    folds = np.array([fold_of[c] for c in cases])
    schemes['严格5折（与 CLAM 训练同划分）'] = [[(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(5)]]
rows = []
print('\n== 交叉验证 R²（嵌套：PCA 和 α 都只在训练折内拟合）==')
for gi, g in enumerate(markers):
    y = Y[:, gi]
    for sname, reps in schemes.items():
        res = {m: np.mean([cv_r2(X_of_fold, lab, y, sp, m) for sp in reps]) for m in ('label', 'morph', 'both')}
        rows.append({'gene': g, 'cv': sname, 'R2_label': res['label'], 'R2_morph': res['morph'],
                     'R2_label_morph': res['both'], 'dR2_morph_beyond_label': res['both'] - res['label']})
        print(f"  {g:7s} [{sname}] 亚型 {res['label']:+.2f} | 形态 {res['morph']:+.2f} | 亚型+形态 {res['both']:+.2f}"
              f" → 形态在亚型之外 ΔR²={res['both'] - res['label']:+.3f}")
pd.DataFrame(rows).round(4).to_csv(OUT / f'morph_marker_predict{TAG}.csv', index=False)
print('  读法: "形态" 一栏接近 "亚型" 一栏，说明形态能预测 marker 主要是因为它认得亚型；ΔR² 才是额外信息。')

# ---------- 6. 图 ----------
if assoc is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for v, name, c in [(0, 'LUAD', '#d62728'), (1, 'LUSC', '#1f77b4')]:
        ax.scatter(Z[lab == v, 0], Z[lab == v, 1], s=25, alpha=0.7, c=c, label=name)
    ax.set_xlabel(f'形态 PC1（{evr[0]:.1%}）')
    ax.set_ylabel(f'形态 PC2（{evr[1]:.1%}）')
    ax.set_title('meanpool 形态嵌入 PCA（每点一个患者）')
    ax.legend()
    ax = axes[1]
    M = assoc.pivot(index='gene', columns='pc', values='rho').loc[markers]
    Q = assoc.pivot(index='gene', columns='pc', values='q_BH').loc[markers]
    im = ax.imshow(M.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if Q.values[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center', fontsize=12)
    ax.set_xticks(range(M.shape[1]), [f'PC{j}' for j in M.columns])
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.set_title('Spearman ρ：形态 PC × marker（* = BH q<0.05，全部检验一起校正）')
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(OUT / 'morph_vs_expr.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
print('\n✅ 产物已写入', OUT)
