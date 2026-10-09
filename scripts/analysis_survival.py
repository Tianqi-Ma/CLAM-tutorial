# -*- coding: utf-8 -*-
"""临床生存分析：缺失检查 + Kaplan-Meier（v2，2026-10 审计后重写）

v1 的结论需要更正：v1 报告"活着的 LUAD 患者在 GDC 没有随访时间"，并把 LUAD vs LUSC 曲线当作"非随机缺失"的反面教材。
真正的原因在我们自己的提取代码：
  - download_tcga.py 请求临床数据时只 expand 了 demographic / diagnoses / exposures，没有 follow_ups；
    TCGA-LUAD 的随访天数记在 follow_ups 实体里，于是 70 个 LUAD 患者里 44 个活着的全部"没有随访"，
    留下来的 26 个 LUAD 全是死亡病例（事件率 100%）。
  - 随访天数用 `a or b` 取值，0 天会被当成缺失。
这是"提取方式制造的缺失"，正确做法是把数据取全（或用 TCGA-CDR），而不是删病例后再解释曲线。
v1 的分期 KM、Cox（包括"LUSC HR=0.39"）都建立在这张有偏的表上，同样作废。

v2：
  1. 终点：有 data/metadata/TCGA-CDR.csv 就用 TCGA-CDR（Liu et al. 2018 Cell，TCGA 官方整理的 OS/PFI/DSS），
     否则用 clinical.json（重新下载时会带 follow_ups：python scripts/download_tcga.py --metadata-only）。
  2. 缺失检查写成程序：按亚型统计缺失比例；任一组缺失 >20% 或组间相差 >15 个百分点就判为不可用，
     不可用时只输出缺失表，不画会误导人的曲线（--force 可强制出图，图上标注"无效"）。
  3. 分期：取原发诊断的 AJCC 病理分期，合并为 I / II / III-IV（IV 期只有个位数）；
     多组 log-rank + 趋势检验（Cox 模型里分期按 1-4 序数的 Wald 检验，另报以亚型分层的版本）。
产物: results/survival/survival_table.csv, missingness.csv, km_by_stage.png, km_luad_vs_lusc.png, run_log.txt
用法: python scripts/analysis_survival.py [--source auto|gdc|cdr] [--force]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from common import RESULTS, clinical_table, log_to, setup_matplotlib, survival_usable

sys.stdout.reconfigure(encoding='utf-8')
plt = setup_matplotlib()
ap = argparse.ArgumentParser()
ap.add_argument('--source', default='auto', choices=['auto', 'gdc', 'cdr'])
ap.add_argument('--force', action='store_true', help='缺失检查不通过也出图（图上标注无效）')
args = ap.parse_args()
OUT = RESULTS / 'survival'
log_to(OUT / 'run_log.txt')

df = clinical_table(args.source)
df.to_csv(OUT / 'survival_table.csv')
print(f"患者 {len(df)} 例；生存终点来源: {df['os_source'].value_counts().to_dict()}")
print(f"性别: {df['sex'].value_counts(dropna=False).to_dict()}；分期组: {df['stage_group'].value_counts(dropna=False).to_dict()}")

# ---------- 1. 缺失检查 ----------
usable, miss = survival_usable(df, 'label')
miss.to_csv(OUT / 'missingness.csv', index=False)
print('\n== 第一步：按亚型检查生存终点缺失 ==')
print(miss.to_string(index=False))
for lab, sub in df.dropna(subset=['os_days']).groupby('label'):
    print(f"  {lab} 有终点的 {len(sub)} 例中事件 {int(sub['os_event'].sum())} 例（事件率 {sub['os_event'].mean():.0%}）")
if not usable:
    print('\n⚠️ 缺失不平衡，生存分析不可用。常见原因是临床数据没有取全：')
    print('   1) python scripts/download_tcga.py --metadata-only   重新取 clinical.json（带 follow_ups）')
    print('   2) 或把 TCGA-CDR 表（Liu et al. 2018 Cell 补充表 S1）另存为 data/metadata/TCGA-CDR.csv')
    if not args.force:
        print('   本次只输出缺失表。确认要看（无效的）曲线请加 --force。')
        sys.exit(0)

d = df.dropna(subset=['os_days']).copy()
d['t'] = d['os_years']
d['event'] = d['os_event'].astype(int)
invalid = '' if usable else '【缺失不平衡，结果无效】'


def km(ax, sub, name, color):
    kmf = KaplanMeierFitter().fit(sub['t'], sub['event'], label=f'{name} (n={len(sub)}, 事件 {int(sub["event"].sum())})')
    kmf.plot_survival_function(ax=ax, color=color, ci_alpha=0.15)
    return kmf


# ---------- 2. 分期 KM ----------
ds = d.dropna(subset=['stage_group'])
groups = ['I', 'II', 'III-IV']
print(f'\n== 分期 KM（n={len(ds)}）==')
print('各组例数 / 事件:', {g: (int((ds['stage_group'] == g).sum()), int(ds.loc[ds['stage_group'] == g, 'event'].sum()))
                       for g in groups})
mlr = multivariate_logrank_test(ds['t'], ds['stage_group'], ds['event'])
trend = CoxPHFitter().fit(ds[['t', 'event', 'stage_ord']], 't', 'event')
trend_s = CoxPHFitter().fit(ds[['t', 'event', 'stage_ord', 'label']], 't', 'event', strata=['label'])
print(f'  三组 log-rank p={mlr.p_value:.3g}')
print(f"  趋势（每升一期 HR={np.exp(trend.params_['stage_ord']):.2f}，p={trend.summary.loc['stage_ord', 'p']:.3g}）；"
      f"以亚型分层后 HR={np.exp(trend_s.params_['stage_ord']):.2f}，p={trend_s.summary.loc['stage_ord', 'p']:.3g}")
fig, ax = plt.subplots(figsize=(8, 5.5))
for g, c in zip(groups, ['#2ca02c', '#ff7f0e', '#9467bd']):
    sub = ds[ds['stage_group'] == g]
    if len(sub) >= 3:
        km(ax, sub, f'Stage {g}', c)
ax.set_title(f'{invalid}KM：按 AJCC 病理分期（原发诊断）\n三组 log-rank p={mlr.p_value:.3g}；趋势 p={trend.summary.loc["stage_ord", "p"]:.3g}')
ax.set_xlabel('年')
ax.set_ylabel('总生存概率')
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(OUT / 'km_by_stage.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# ---------- 3. LUAD vs LUSC ----------
a, b = d[d['label'] == 'LUAD'], d[d['label'] == 'LUSC']
lr = logrank_test(a['t'], b['t'], a['event'], b['event'])
fig, ax = plt.subplots(figsize=(8, 5.5))
kma = km(ax, a, 'LUAD', '#d62728')
kmb = km(ax, b, 'LUSC', '#1f77b4')
ax.set_title(f'{invalid}KM：LUAD vs LUSC（log-rank p={lr.p_value:.3g}）\n'
             f'TCGA 是方便样本、两个项目入组来源不同，这条对比只作描述，不作预后结论')
ax.set_xlabel('年')
ax.set_ylabel('总生存概率')
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(OUT / 'km_luad_vs_lusc.png', dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'\n== LUAD vs LUSC ==\n  中位生存 LUAD {kma.median_survival_time_:.1f} 年 vs LUSC {kmb.median_survival_time_:.1f} 年，'
      f'log-rank p={lr.p_value:.3g}{"（无效：缺失不平衡）" if invalid else ""}')
print('\n✅ 产物已写入', OUT)
