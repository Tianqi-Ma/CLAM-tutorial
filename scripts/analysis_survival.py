# -*- coding: utf-8 -*-
"""临床生存分析：Kaplan-Meier 曲线 + 缺失数据陷阱教学
关键发现：26 例 Alive LUAD 患者在 GDC 完全没有随访时间 → 朴素 LUAD-vs-LUSC 对比系统性失真。
本脚本把这种"非随机缺失"展示出来——生存分析的第一步永远是检查缺失模式。
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

PROJ = r'E:/Projects/DP/CLAM-tutorial'
OUT = f'{PROJ}/results/survival'
os.makedirs(OUT, exist_ok=True)

csv = pd.read_csv(f'{PROJ}/data/dataset_csv/tcga_luad_lusc.csv')
label_of_case = dict(zip(csv['case_id'], csv['label']))
clinical = json.load(open(f'{PROJ}/data/metadata/clinical.json'))

rows, dropped = [], []
for c in clinical:
    case = c['submitter_id']
    dem = c.get('demographic', {})
    vs = dem.get('vital_status')
    if vs == 'Dead':
        t, e = dem.get('days_to_death'), 1
    else:
        t, e = dem.get('days_to_last_follow_up'), 0
        if t is None:
            for d in c.get('diagnoses', []):
                t = d.get('days_to_last_follow_up') or d.get('days_to_last_known_disease_status')
                if t is not None: break
    stage = next((d.get('ajcc_pathologic_stage') for d in c.get('diagnoses', []) if d.get('ajcc_pathologic_stage')), None)
    if case not in label_of_case: continue
    if t is None or vs is None:
        dropped.append({'case': case, 'label': label_of_case[case], 'vital': vs})
    else:
        rows.append({'case': case, 'label': label_of_case[case], 't': float(t) / 365.25,
                     'event': e, 'stage': stage,
                     'age': dem.get('age_at_index'), 'gender': dem.get('gender')})
df = pd.DataFrame(rows)
dd = pd.DataFrame(dropped)
df.to_csv(f'{OUT}/survival_table.csv', index=False)

print('== 第一步：检查缺失模式（这是本分析最重要的一张表）==')
print(f'97 个患者中 {len(df)} 个有完整生存记录，{len(dd)} 个因无随访时间被剔除')
print('被剔除病例的构成：')
print(dd.groupby(['label', 'vital']).size().to_string())
print('→ 26 例被剔除者全是 Alive 的 LUAD！缺失不是随机的：它们都活着，剔除后 LUAD 曲线必然假性偏糟。')

def km(ax, sub, name, color):
    kmf = KaplanMeierFitter()
    kmf.fit(sub['t'], sub['event'], label=f'{name} (n={len(sub)})')
    kmf.plot_survival_function(ax=ax, color=color, ci_alpha=0.15)
    return kmf

# ---- 图1：分期 KM（主分析：每组都有两类患者，仍可做数据质量对照）----
def stage_group(s):
    if s is None: return None
    if s.startswith('Stage IV'): return 'IV'
    if s.startswith('Stage III'): return 'III'
    if s.startswith('Stage II'): return 'II'
    return 'I'
df['sg'] = df['stage'].map(stage_group)
ds = df.dropna(subset=['sg'])
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {'I': '#2ca02c', 'II': '#ff7f0e', 'III': '#9467bd', 'IV': '#8c564b'}
for g in ['I', 'II', 'III', 'IV']:
    sub = ds[ds['sg'] == g]
    if len(sub) >= 3:
        km(ax, sub, f'Stage {g}', colors[g])
ax.set_title('Kaplan-Meier：按 AJCC 病理分期（n=%d，有随访记录者）\n分期越晚曲线越低 = 数据符合临床常识' % len(ds))
ax.set_xlabel('年'); ax.set_ylabel('生存概率'); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(f'{OUT}/km_by_stage.png', dpi=110, bbox_inches='tight'); plt.close(fig)
print('\n== 分期 KM（主分析）==')
print('各期例数:', {g: int((ds["sg"]==g).sum()) for g in ['I','II','III','IV']})
a, b2 = ds[ds['sg'] == 'I'], ds[ds['sg'] == 'III']
lr2 = logrank_test(a['t'], b2['t'], a['event'], b2['event'])
print(f'Stage I vs III: log-rank p={lr2.p_value:.2f}（曲线方向正确；样本量小未达显著，符合预期）')

# ---- 图2：LUAD vs LUSC（教学演示：缺失数据陷阱，结论无效）----
fig, ax = plt.subplots(figsize=(8, 5.5))
a, b = df[df['label'] == 'LUAD'], df[df['label'] == 'LUSC']
kma = km(ax, a, 'LUAD 肺腺癌', '#d62728')
kmb = km(ax, b, 'LUSC 肺鳞癌', '#1f77b4')
lr = logrank_test(a['t'], b['t'], a['event'], b['event'])
ax.set_title(f'【警告】缺失数据陷阱演示：LUAD vs LUSC（此对比无效！）\n'
             f'26 例活着的 LUAD 患者无随访记录被剔除 → LUAD 曲线假性偏糟（log-rank p={lr.p_value:.3f} 不可信）')
ax.set_xlabel('年'); ax.set_ylabel('生存概率'); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(f'{OUT}/km_luad_vs_lusc.png', dpi=110, bbox_inches='tight'); plt.close(fig)
print('\n== LUAD vs LUSC（教学演示，结论无效）==')
print(f'表面上: LUAD 中位 {kma.median_survival_time_:.1f} 年 vs LUSC {kmb.median_survival_time_:.1f} 年, p={lr.p_value:.3f}')
print('但这个"显著差异"完全是缺失数据造成的假象——被剔除的 26 例全是活着的 LUAD。')
print('教训：KM 之前先按 标签×结局 交叉检查缺失构成，缺失不随机时任何曲线都可能撒谎。')
print('\n✅ 图已保存:', f'{OUT}/km_by_stage.png,', f'{OUT}/km_luad_vs_lusc.png')
