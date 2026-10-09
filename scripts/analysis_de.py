# -*- coding: utf-8 -*-
"""多组学②：全基因组差异表达（DE）+ 通路富集（v2，2026-10 审计后重写）

方法：
  1. 每个患者一个原发肿瘤样本（common.load_expression；v1 可能混入癌旁正常样本）。
  2. 过滤：至少 20% 的患者 TPM ≥ 1。
  3. 检验：log2(TPM+1) 上逐基因 Wilcoxon 秩和检验（Mann-Whitney），BH 校正。
     每组 70 例以上的人群样本，秩检验比 DESeq2/edgeR 的参数模型更能控制假阳性
     （Li et al. 2022, Genome Biology 23:79）。
  4. 效应量：两组 log2(TPM+1) 均值之差（≈ 几何均值的 log2 倍数变化）。
     v1 用 log2((算术均值A+1)/(算术均值B+1))，会被少数极端样本拉大：SST 的"log2FC=5.8"其实 q=0.29。
  5. 富集：
     a. 过表达分析（ORA）：用全部显著基因，背景 = 通过过滤、真正参与检验的基因，超几何检验 + BH。
        v1 每侧只送 top 200 给 Enrichr，背景是全基因组，没参与检验的低表达基因也被当成"背景"。
     b. 预排序 GSEA（装了 gseapy 才跑）：全部基因按 sign(Δ)×−log10(p) 排序，不需要先定显著性阈值。
     基因集：MSigDB Hallmark 2020 + Reactome 2022（Reactome 里有 Keratinization、Surfactant metabolism，
     分别是 LUSC、LUAD 的阳性对照——富集方法靠不靠谱，先看阳性对照出没出来）。
  ⚠️ 本数据集 43 个组织来源中心每个只贡献一种亚型，中心批次效应和亚型完全混在一起，DE 无法把两者分开；
     marker 和阳性对照方向正确只能说明亚型信号占主导，不能说明没有批次成分。
产物: results/multiomics/de_table.csv, de_volcano.png, enrich_ora.csv, enrich_gsea.csv（可选）, enrich_barplot.png
用法: python scripts/analysis_de.py [--allow-ambiguous]
"""
import argparse
import os
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu

from common import (RESULTS, bh_fdr, describe_expression_report, load_dataset, load_expression, log_to,
                    setup_matplotlib)

sys.stdout.reconfigure(encoding='utf-8')
plt = setup_matplotlib()
ap = argparse.ArgumentParser()
ap.add_argument('--allow-ambiguous', action='store_true')
ap.add_argument('--min-frac', type=float, default=0.2, help='至少这么多比例的患者 TPM≥1 才检验')
args = ap.parse_args()
OUT = RESULTS / 'multiomics'
GS_DIR = OUT / 'genesets'
OUT.mkdir(parents=True, exist_ok=True)
log_to(OUT / 'de_log.txt')
LIBS = ['MSigDB_Hallmark_2020', 'Reactome_2022']
CONTROLS = {'LUSC': ['Keratinization', 'Formation Of Cornified Envelope'], 'LUAD': ['Surfactant Metabolism']}

ds = load_dataset()
label_of = ds.drop_duplicates('case_id').set_index('case_id')['label']
E, rep = load_expression(cases=label_of.index, strict=not args.allow_ambiguous)
print('表达矩阵:', describe_expression_report(rep))
la = [c for c in E.columns if label_of[c] == 'LUAD']
lb = [c for c in E.columns if label_of[c] == 'LUSC']

# ---------- 1. DE ----------
keep = (E >= 1).mean(axis=1) >= args.min_frac
G = np.log2(E[keep] + 1)
print(f'{E.shape[0]} 个蛋白编码基因 × {E.shape[1]} 患者（LUAD {len(la)} / LUSC {len(lb)}）；'
      f'过滤后 {len(G)} 个基因参与检验')
p = mannwhitneyu(G[la].values, G[lb].values, axis=1).pvalue
de = pd.DataFrame({'mean_log2tpm_LUAD': G[la].mean(axis=1), 'mean_log2tpm_LUSC': G[lb].mean(axis=1),
                   'median_TPM_LUAD': E.loc[G.index, la].median(axis=1),
                   'median_TPM_LUSC': E.loc[G.index, lb].median(axis=1)})
de['log2FC_LUAD_vs_LUSC'] = de['mean_log2tpm_LUAD'] - de['mean_log2tpm_LUSC']
de['p'] = p
de['q_BH'] = bh_fdr(p)
de.index.name = 'gene'
de = de.sort_values('p')
de.round({c: 4 for c in de.columns if c not in ('p', 'q_BH')}).to_csv(OUT / 'de_table.csv')   # p/q 不能四舍五入，否则小 p 全变 0
sig = de[(de['q_BH'] < 0.05) & (de['log2FC_LUAD_vs_LUSC'].abs() > 1)]
up = {'LUAD': sig[sig['log2FC_LUAD_vs_LUSC'] > 0], 'LUSC': sig[sig['log2FC_LUAD_vs_LUSC'] < 0]}
print(f"显著（q<0.05 且 |log2FC|>1）: {len(sig)} 个 = LUAD 高 {len(up['LUAD'])} + LUSC 高 {len(up['LUSC'])}")
for g in ['NKX2-1', 'NAPSA', 'SFTPB', 'TP63', 'KRT5', 'KRT6A', 'SOX2', 'DSG3']:
    if g in de.index:
        r = de.loc[g]
        print(f"  [marker] {g:7s}: log2FC={r['log2FC_LUAD_vs_LUSC']:+.2f}, q={r['q_BH']:.1e}")

fig, ax = plt.subplots(figsize=(9, 6.5))
y = -np.log10(de['q_BH'].clip(lower=1e-300))
ns = ~de.index.isin(sig.index)
ax.scatter(de.loc[ns, 'log2FC_LUAD_vs_LUSC'], y[ns], s=4, c='#bbbbbb', alpha=0.4)
for lab, c in [('LUAD', '#d62728'), ('LUSC', '#1f77b4')]:
    ax.scatter(up[lab]['log2FC_LUAD_vs_LUSC'], y[up[lab].index], s=6, c=c, alpha=0.6,
               label=f'{lab} 高 ({len(up[lab])})')
for g in ['NKX2-1', 'NAPSA', 'TP63', 'KRT5', 'KRT6A', 'DSG3']:
    if g in de.index:
        ax.annotate(g, (de.loc[g, 'log2FC_LUAD_vs_LUSC'], y[g]), fontsize=9, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points')
ax.axhline(-np.log10(0.05), ls='--', c='gray', lw=0.8)
ax.set_xlabel('log2FC（两组 log2(TPM+1) 均值之差；右 = LUAD 高）')
ax.set_ylabel('-log10(BH q)')
ax.set_title(f'LUAD vs LUSC 差异表达（Wilcoxon，{len(la)} vs {len(lb)} 例原发肿瘤）\n'
             f'注意：组织来源中心与亚型完全重合，差异里可能含中心批次成分')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'de_volcano.png', dpi=110, bbox_inches='tight')
plt.close(fig)


# ---------- 2. 富集 ----------
def fetch_library(lib):
    """Enrichr 的基因集库（文本格式，一行一个基因集），缓存到 results/multiomics/genesets/。"""
    GS_DIR.mkdir(exist_ok=True)
    path = GS_DIR / f'{lib}.txt'
    if not path.exists():
        url = f'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib}'
        try:
            text = urllib.request.urlopen(url, timeout=60).read().decode('utf-8')
        except Exception:
            curl = r'C:\Windows\System32\curl.exe'           # 作者机器上 Python 的 TLS 会被中间设备掐断
            if not os.path.exists(curl):
                raise
            text = subprocess.run([curl, '--ssl-no-revoke', '-sS', url], capture_output=True, text=True,
                                  encoding='utf-8', timeout=120, check=True).stdout
        path.write_text(text, encoding='utf-8')
    sets = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = [x.split(',')[0] for x in line.split('\t') if x]
        if len(parts) > 2:
            sets[parts[0]] = set(parts[1:])
    return sets


def ora(genes, background, sets, lib):
    genes = set(genes) & background
    rows = []
    for term, members in sets.items():
        m = members & background
        if len(m) < 10 or len(m) > 500:
            continue
        k = len(genes & m)
        pval = hypergeom.sf(k - 1, len(background), len(m), len(genes)) if k else 1.0
        rows.append({'library': lib, 'term': term, 'set_size': len(m), 'overlap': k,
                     'expected': len(genes) * len(m) / len(background), 'p': pval,
                     'genes': ','.join(sorted(genes & m)[:30])})
    t = pd.DataFrame(rows)
    t['q_BH'] = bh_fdr(t['p'])
    return t.sort_values('p')


background = set(de.index)
libraries = {}
for lib in LIBS:
    try:
        libraries[lib] = fetch_library(lib)
    except Exception as e:
        print(f'  [跳过] 取不到基因集库 {lib}（{type(e).__name__}: {str(e)[:80]}）')
ora_tabs = []
for direction, genes in up.items():
    for lib, sets in libraries.items():
        t = ora(genes.index, background, sets, lib)
        t.insert(0, 'direction', direction)
        ora_tabs.append(t)
if ora_tabs:
    ora_all = pd.concat(ora_tabs)
    ora_all.round({'expected': 2}).to_csv(OUT / 'enrich_ora.csv', index=False)
    print('\n== ORA（显著基因 vs 参与检验的背景，超几何 + BH）==')
    for direction in up:
        top = ora_all[(ora_all['direction'] == direction)].sort_values('p').head(6)
        for _, r in top.iterrows():
            print(f"  {direction} [{r['library'].split('_')[0]}] {r['term'][:60]}: "
                  f"{r['overlap']}/{r['set_size']}（期望 {r['expected']:.1f}），q={r['q_BH']:.1e}")
    print('  阳性对照:')
    for direction, terms in CONTROLS.items():
        for term in terms:
            hit = ora_all[(ora_all['direction'] == direction) & ora_all['term'].str.startswith(term)]
            if len(hit):
                r = hit.iloc[0]
                print(f"    {direction} 侧 {r['term'][:60]}: q={r['q_BH']:.1e}")
            else:
                print(f'    {direction} 侧 {term}: 基因集库里没找到')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (direction, color) in zip(axes, [('LUAD', '#d62728'), ('LUSC', '#1f77b4')]):
        h = ora_all[ora_all['direction'] == direction].sort_values('p').head(10).iloc[::-1]
        ax.barh(range(len(h)), -np.log10(h['q_BH'].clip(lower=1e-300)), color=color, alpha=0.8)
        ax.set_yticks(range(len(h)), [t[:55] for t in h['term']], fontsize=8)
        ax.axvline(-np.log10(0.05), ls='--', c='gray', lw=0.8)
        ax.set_xlabel('-log10(BH q)')
        ax.set_title(f'{direction} 高表达基因的 ORA（Hallmark + Reactome，背景 = 参与检验的基因）', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'enrich_barplot.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

try:
    import gseapy
    # 秩检验的 p 值经常并列（U 统计量是整数），加一个按效应量的微小扰动打破并列
    rnk = (np.sign(de['log2FC_LUAD_vs_LUSC']) * -np.log10(de['p'].clip(lower=1e-300))
           + 1e-6 * de['log2FC_LUAD_vs_LUSC']).sort_values(ascending=False)
    gsea_tabs = []
    for lib, sets in libraries.items():
        res = gseapy.prerank(rnk=rnk, gene_sets={k: list(v) for k, v in sets.items()}, min_size=10, max_size=500,
                             permutation_num=1000, seed=0, outdir=None, verbose=False)
        t = res.res2d.copy()
        t.insert(0, 'library', lib)
        gsea_tabs.append(t)
    gsea = pd.concat(gsea_tabs)
    gsea.to_csv(OUT / 'enrich_gsea.csv', index=False)
    print('\n== 预排序 GSEA（NES>0 = LUAD 侧富集）==')
    for _, r in gsea.sort_values('FDR q-val').head(10).iterrows():
        print(f"  [{r['library'].split('_')[0]}] {r['Term'][:60]}: NES={float(r['NES']):+.2f}, FDR={float(r['FDR q-val']):.1e}")
except ImportError:
    print('\n[提示] 没装 gseapy，跳过预排序 GSEA（pip install gseapy 后重跑即可）')
print('\n✅ DE 分析完成 →', OUT)
