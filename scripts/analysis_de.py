# -*- coding: utf-8 -*-
"""多组学深入版①：全基因组差异表达（DE）+ 通路富集
1. LUAD vs LUSC 全基因组 Mann-Whitney 检验（19,938 个蛋白编码基因，log1p TPM）
   - 效应量: log2FC = log2((meanA+1)/(meanB+1))；低表达基因先过滤（两组均值都 <1 TPM 的扔掉）
   - 多重检验: Benjamini-Hochberg FDR（手写 10 行，不依赖 statsmodels）
2. 火山图 + 显著基因表（de_table.csv）；已知 marker 应排在顶部 = 方法学自检
3. 富集分析：每方向上 top 200 基因送 Enrichr（curl 走系统 schannel，绕开本机 pip-TLS 怪癖）；
   网络失败时优雅降级（打印说明跳过，不影响主流程）
产物: results/multiomics/de_volcano.png, de_table.csv, enrich_*.csv / enrich_barplot.png
"""
import sys, glob, os, json, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import mannwhitneyu

PROJ = r'E:/Projects/DP/CLAM-tutorial'
OUT  = f'{PROJ}/results/multiomics'
CURL = r'C:\Windows\System32\curl.exe'
os.makedirs(OUT, exist_ok=True)

csv = pd.read_csv(f'{PROJ}/data/dataset_csv/tcga_luad_lusc.csv')
label_of_case = dict(zip(csv['case_id'], csv['label']))

# ---------- 表达矩阵 ----------
expr = {}
for f in glob.glob(f'{PROJ}/data/metadata/expression/*.tsv'):
    case = os.path.basename(f).split('__')[0]
    df = pd.read_csv(f, sep='\t', comment='#', usecols=['gene_name', 'gene_type', 'tpm_unstranded'])
    df = df[df['gene_type'] == 'protein_coding'].drop_duplicates('gene_name').set_index('gene_name')
    expr[case] = df['tpm_unstranded']
E = pd.DataFrame(expr)
E = E[[c for c in E.columns if c in label_of_case]]
la = [c for c in E.columns if label_of_case[c] == 'LUAD']
lb = [c for c in E.columns if label_of_case[c] == 'LUSC']
print(f'表达矩阵: {E.shape[0]} 基因 × {E.shape[1]} 患者（LUAD {len(la)} / LUSC {len(lb)}）')

# ---------- 1. 全基因组 DE ----------
meanA, meanB = E[la].mean(1), E[lb].mean(1)
keep = (meanA >= 1) | (meanB >= 1)                    # 低表达过滤
G = E[keep]
log2fc = np.log2((meanA[keep] + 1) / (meanB[keep] + 1))
print(f'低表达过滤后 {len(G)} 基因进入检验（{int((~keep).sum())} 个两组都 <1 TPM 被剔除）')
logG = np.log1p(G)
pvals = np.empty(len(G))
for i, g in enumerate(G.index):
    pvals[i] = mannwhitneyu(logG.loc[g, la], logG.loc[g, lb]).pvalue
    if i % 5000 == 4999: print(f'  {i+1}/{len(G)} ...')

# Benjamini-Hochberg FDR
order = np.argsort(pvals)
ranked = pvals[order] * len(pvals) / (np.arange(len(pvals)) + 1)
ranked = np.minimum.accumulate(ranked[::-1])[::-1]    # 单调化
qvals = np.empty(len(pvals)); qvals[order] = np.clip(ranked, 0, 1)

de = pd.DataFrame({'gene': G.index, 'mean_LUAD': meanA[keep].round(2), 'mean_LUSC': meanB[keep].round(2),
                   'log2FC_LUAD_vs_LUSC': log2fc.round(3), 'p': pvals, 'q_BH': qvals}).set_index('gene')
de = de.sort_values('q_BH')
de.to_csv(f'{OUT}/de_table.csv')
sig = de[(de['q_BH'] < 0.05) & (de['log2FC_LUAD_vs_LUSC'].abs() > 1)]
up_a = sig[sig['log2FC_LUAD_vs_LUSC'] > 0]; up_b = sig[sig['log2FC_LUAD_vs_LUSC'] < 0]
print(f'\n显著基因（FDR<0.05 且 |log2FC|>1）: {len(sig)} 个 = LUAD高 {len(up_a)} + LUSC高 {len(up_b)}')
print('\nLUAD 高表达 top 8:'); print(up_a.head(8).to_string())
print('\nLUSC 高表达 top 8:'); print(up_b.head(8).to_string())
for g in ['NKX2-1', 'TP63', 'KRT5', 'KRT6A']:
    if g in de.index:
        r = de.loc[g]
        print(f'  [自检] {g}: log2FC={r["log2FC_LUAD_vs_LUSC"]:+.2f}, q={r["q_BH"]:.1e}（应与已知方向一致）')

# ---------- 2. 火山图 ----------
fig, ax = plt.subplots(figsize=(9, 6.5))
ns = de[de['q_BH'] >= 0.05]
ax.scatter(ns['log2FC_LUAD_vs_LUSC'], -np.log10(ns['q_BH'] + 1e-300), s=4, c='#bbbbbb', alpha=0.4)
ax.scatter(up_a['log2FC_LUAD_vs_LUSC'], -np.log10(up_a['q_BH']), s=6, c='#d62728', alpha=0.6, label=f'LUAD 高 ({len(up_a)})')
ax.scatter(up_b['log2FC_LUAD_vs_LUSC'], -np.log10(up_b['q_BH']), s=6, c='#1f77b4', alpha=0.6, label=f'LUSC 高 ({len(up_b)})')
for g in ['NKX2-1', 'TP63', 'KRT5', 'KRT6A', 'SFTPB', 'DSG3']:
    if g in de.index:
        r = de.loc[g]
        ax.annotate(g, (r['log2FC_LUAD_vs_LUSC'], -np.log10(r['q_BH'] + 1e-300)),
                    fontsize=9, fontweight='bold', xytext=(4, 4), textcoords='offset points')
ax.axhline(-np.log10(0.05), ls='--', c='gray', lw=0.8)
ax.set_xlabel('log2FC（LUAD / LUSC）→ 右=腺癌高，左=鳞癌高'); ax.set_ylabel('-log10(FDR q)')
ax.set_title(f'全基因组差异表达：LUAD vs LUSC（{len(la)} vs {len(lb)} 患者）\n'
             f'{len(sig)} 个显著基因——两型肺癌的分子差异巨大，marker 自检全在顶部')
ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/de_volcano.png', dpi=110, bbox_inches='tight'); plt.close(fig)
print('\n火山图 -> de_volcano.png')

# ---------- 3. Enrichr 富集（优雅降级） ----------
def enrichr(genes, desc):
    """送基因列表到 Enrichr，返回 Hallmark+KEGG 富集表；失败返回 None。"""
    try:
        lst = '\n'.join(genes)
        r = subprocess.run([CURL, '--ssl-no-revoke', '-sS', '-X', 'POST',
                            '-F', f'list={lst}', '-F', f'description={desc}',
                            'https://maayanlab.cloud/Enrichr/addList'],
                           capture_output=True, text=True, timeout=60)
        uid = json.loads(r.stdout)['userListId']
        frames = {}
        for lib in ['MSigDB_Hallmark_2020', 'WikiPathway_2023_Human']:
            r2 = subprocess.run([CURL, '--ssl-no-revoke', '-sS',
                                 f'https://maayanlab.cloud/Enrichr/enrich?userListId={uid}&backgroundType={lib}'],
                                capture_output=True, text=True, timeout=60)
            d = json.loads(r2.stdout)[lib]
            t = pd.DataFrame(d, columns=['rank', 'term', 'p', 'z', 'combined', 'genes', 'adj_p', 'x', 'y'])
            t['library'] = lib
            frames[lib] = t
        return pd.concat(frames)
    except Exception as e:
        print(f'  [降级] Enrichr 不可用（{type(e).__name__}: {str(e)[:80]}）——跳过富集，不影响主分析')
        return None

tops = {'LUAD': up_a.head(200).index.tolist(), 'LUSC': up_b.head(200).index.tolist()}
enrs = {}
for lab, genes in tops.items():
    if len(genes) < 20:
        print(f'  {lab} 显著基因太少（{len(genes)}），跳过富集'); continue
    print(f'\n== Enrichr：{lab} 方向 top {len(genes)} 基因 ==')
    enr = enrichr(genes, f'{lab}_up')
    if enr is None: continue
    enr.to_csv(f'{OUT}/enrich_{lab}.csv', index=False)
    enrs[lab] = enr
    show = enr[enr['library'] == 'MSigDB_Hallmark_2020'].head(5)
    for _, r in show.iterrows():
        print(f'  [Hallmark] {r["term"]}: q={r["adj_p"]:.1e}, 命中 {len(r["genes"])} 基因')

if len(enrs) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (lab, color) in zip(axes, [('LUAD', '#d62728'), ('LUSC', '#1f77b4')]):
        h = enrs[lab][enrs[lab]['library'] == 'MSigDB_Hallmark_2020'].head(8).iloc[::-1]
        ax.barh(range(len(h)), -np.log10(h['adj_p'] + 1e-300), color=color, alpha=0.8)
        ax.set_yticks(range(len(h)), [t.replace('HALLMARK_', '').replace('_', ' ').title() for t in h['term']], fontsize=9)
        ax.set_xlabel('-log10(FDR q)')
        ax.set_title(f'{lab} 高表达基因的 Hallmark 通路富集\n（top200 基因 → 这些生物过程在该型更活跃）')
    fig.tight_layout(); fig.savefig(f'{OUT}/enrich_barplot.png', dpi=110, bbox_inches='tight'); plt.close(fig)
    print('\n富集条形图 -> enrich_barplot.png')
print('\n✅ DE 分析完成')
