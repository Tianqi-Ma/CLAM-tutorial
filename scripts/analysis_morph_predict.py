# -*- coding: utf-8 -*-
"""多组学深入版②：形态到底能预测哪些基因？（全基因组岭回归 LOO-CV）
做法：形态嵌入(512) → PC1-10 → 多输出岭回归预测每个基因的 log1p(TPM)。
留一交叉验证用帽子矩阵闭式解：e_i^LOO = r_i / (1 - h_ii)，19,938 个基因一次矩阵运算搞定。
产出：
  - R² 分布直方图 + top 20 可预测基因（morph_predict_hist.png）
  - NKX2-1 观测 vs LOO 预测散点（morph_predict_nkx2.png）
  - 与差异表达的交叉验证：top-200 R² 基因是否富集于 DE 显著基因（Fisher 检验）
    → 如果富集，说明"形态可预测的基因"正是"两型间差异的基因"：
      形态空间与分子空间同构的又一独立证据（不依赖 marker 先验）
依赖: morph_embeddings.csv（analysis_multiomics.py）, de_table.csv（analysis_de.py）
"""
import sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import fisher_exact

PROJ = r'E:/Projects/DP/CLAM-tutorial'
OUT  = f'{PROJ}/results/multiomics'

# ---------- 数据 ----------
emb_path = f'{OUT}/morph_embeddings.csv'
if not os.path.exists(emb_path):
    raise SystemExit('缺少 morph_embeddings.csv —— 先跑 analysis_multiomics.py')
X_case = pd.read_csv(emb_path, index_col=0)
csv = pd.read_csv(f'{PROJ}/data/dataset_csv/tcga_luad_lusc.csv')
label_of_case = dict(zip(csv['case_id'], csv['label']))

expr = {}
for f in glob.glob(f'{PROJ}/data/metadata/expression/*.tsv'):
    case = os.path.basename(f).split('__')[0]
    df = pd.read_csv(f, sep='\t', comment='#', usecols=['gene_name', 'gene_type', 'tpm_unstranded'])
    df = df[df['gene_type'] == 'protein_coding'].drop_duplicates('gene_name').set_index('gene_name')
    expr[case] = df['tpm_unstranded']
E = pd.DataFrame(expr)
cases = [c for c in X_case.index if c in E.columns]
X_case, E = X_case.loc[cases], E[cases]
mean_tpm = E.mean(1)
genes = mean_tpm[mean_tpm >= 1].index          # 只评有足够表达的基因
Y = np.log1p(E.loc[genes].values).T            # (n, g)
print(f'{len(cases)} 患者 × {len(genes)} 基因（TPM≥1 过滤）')

# ---------- 特征: 形态 PC1-10 ----------
Z = PCA(10).fit_transform(StandardScaler().fit_transform(X_case.values))
Z = np.hstack([np.ones((len(Z), 1)), Z])       # 截距项
n, p = Z.shape

# ---------- 多输出岭回归 + 帽子矩阵 LOO ----------
def loo_r2(alpha):
    G = Z.T @ Z + alpha * np.eye(p)
    G[0, 0] -= alpha                            # 截距不正则
    B = np.linalg.solve(G, Z.T @ Y)             # (p, g) 一次解出全部基因
    R = Y - Z @ B                               # 残差 (n, g)
    H = np.sum(Z @ np.linalg.inv(G) * Z, axis=1)  # 帽子矩阵对角线 h_ii
    E_loo = R / (1 - H)[:, None]
    sse = np.sum(E_loo ** 2, axis=0)
    sst = np.sum((Y - Y.mean(0)) ** 2, axis=0)
    return 1 - sse / sst

print('\nalpha 网格（按全基因组平均 R² 选）:')
best_a, best_m, r2s = None, -9, {}
for a in [0.1, 1, 10, 100]:
    r2 = loo_r2(a)
    r2s[a] = r2
    print(f'  alpha={a:>6}: 平均R²={np.nanmean(r2):.3f}, 中位R²={np.nanmedian(r2):.3f}, R²>0.1 基因数={(r2 > 0.1).sum()}')
    if np.nanmean(r2) > best_m: best_m, best_a = np.nanmean(r2), a
r2 = r2s[best_a]
res = pd.DataFrame({'gene': genes, 'R2_loo': r2}).set_index('gene').sort_values('R2_loo', ascending=False)
res.round(4).to_csv(f'{OUT}/morph_predict_r2.csv')
print(f'\n选用 alpha={best_a}。top 15 可预测基因:')
print(res.head(15).to_string())
for g in ['NKX2-1', 'KRT5', 'TP63']:
    if g in res.index:
        rk = res.index.get_loc(g)
        print(f'  [marker] {g}: R²={res.loc[g, "R2_loo"]:.3f}（全基因组第 {rk+1} 名 / {len(res)}）')

# ---------- 图1: R² 分布 ----------
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.hist(np.clip(r2, -1, 1), bins=80, color='#7a5195', alpha=0.85)
ax.axvline(0, c='gray', ls='--', lw=0.8)
ax.set_xlabel('留一交叉验证 R²（>0 = 形态里含有该基因表达的信息）')
ax.set_ylabel('基因数')
ax.set_title(f'形态能预测哪些基因？全基因组扫描（{len(cases)} 患者 × {len(res)} 基因）\n'
             f'R²>0 的基因 {(r2 > 0).sum()} 个 / R²>0.1 的 {(r2 > 0.1).sum()} 个——形态远不止能分两类')
fig.tight_layout(); fig.savefig(f'{OUT}/morph_predict_hist.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---------- 图2: NKX2-1 观测 vs 预测 ----------
G_ = Z.T @ Z + best_a * np.eye(p); G_[0, 0] -= best_a
B = np.linalg.solve(G_, Z.T @ Y)
gi = list(genes).index('NKX2-1') if 'NKX2-1' in set(genes) else None
if gi is not None:
    H = np.sum(Z @ np.linalg.inv(G_) * Z, axis=1)
    pred = (Z @ B)[:, gi]; resid = Y[:, gi] - pred; loo_pred = Y[:, gi] - resid / (1 - H)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for lab, c in [('LUAD', '#d62728'), ('LUSC', '#1f77b4')]:
        m = [i for i, cc in enumerate(cases) if label_of_case[cc] == lab]
        ax.scatter(Y[m, gi], loo_pred[m], s=22, alpha=0.7, c=c, label=lab)
    lim = [Y[:, gi].min() - 0.3, Y[:, gi].max() + 0.3]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlabel('NKX2-1 真实表达 log1p(TPM)'); ax.set_ylabel('LOO 预测表达')
    ax.set_title(f'从形态预测 NKX2-1 表达（留一预测，无泄漏）\nR²={res.loc["NKX2-1", "R2_loo"]:.2f}')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{OUT}/morph_predict_nkx2.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---------- 与 DE 交叉验证（Fisher 富集） ----------
de_path = f'{OUT}/de_table.csv'
if os.path.exists(de_path):
    de = pd.read_csv(de_path, index_col=0)
    de_sig = set(de[(de['q_BH'] < 0.05) & (de['log2FC_LUAD_vs_LUSC'].abs() > 1)].index)
    de_sig &= set(res.index)                    # 统一基因宇宙（两边 TPM 过滤口径略不同，取交集）
    top200 = set(res.head(200).index)
    both = len(top200 & de_sig)
    table = [[both, 200 - both], [len(de_sig) - both, len(res) - 200 - (len(de_sig) - both)]]
    oratio, pf = fisher_exact(table)
    print(f'\n== 交叉验证：top-200 形态可预测基因 × DE 显著基因 ==')
    print(f'  重叠 {both}/200，Fisher OR={oratio:.1f}, p={pf:.1e}')
    print('  → 富集即说明：形态可预测的基因正是两型间差异表达的基因（形态≅分子，独立证据）')
else:
    print('\n[提示] 未找到 de_table.csv，跳过 DE 交叉验证（先跑 analysis_de.py）')
print('\n✅ 图已保存: morph_predict_hist.png' + (', morph_predict_nkx2.png' if gi is not None else ''))
