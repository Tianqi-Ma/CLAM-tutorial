# -*- coding: utf-8 -*-
"""Cox 预后建模：临床 vs 临床 + 形态（v2，2026-10 审计后重写）

v1 的问题：
  - 生存表本身有偏（见 analysis_survival.py：LUAD 只剩死亡病例），所有 HR 和 C-index 作废；
  - 形态特征来自 5 个不同折模型拼起来的嵌入（空间不一致）；
  - 亚型当普通协变量，HR=0.39 其实是缺失造成的；分期当 1-4 的连续变量；没用性别（字段已改名 sex_at_birth）；
  - 只有一个池化 C-index，没有区间，也没有"加了形态比不加好多少"的统计；
  - 风险高低组用全体 OOF 风险分的中位数切分，阈值用到了测试患者的信息。
v2：
  - 先过 survival_usable 缺失检查，不通过就退出（--force 可强跑，结果标注无效）；
  - 临床模型：分期组（I 为参照，II、III-IV 两个哑变量）+ 年龄 + 性别，亚型作为分层变量（各亚型有自己的基线风险）；
  - 形态：meanpool PC（每个训练折内拟合标准化 + PCA），或 --embedding clam（每折用自己的模型空间）；
  - 重复分层 5 折（按事件分层，默认 20 次）：报告每折 C-index 的均值±SD、池化 OOF C-index，
    以及 ΔC = C(临床+形态) − C(临床) 在各次重复上的分布；另对第一次重复做患者 bootstrap 给 95% 区间；
  - 风险分组阈值取训练折风险分的中位数；
  - 对全数据临床模型做比例风险假设检验（Schoenfeld 残差，rank 时间变换）。
产物: results/survival/cox_summary.txt, km_risk_groups.png, cox_log.txt
用法: python scripts/analysis_cox.py [--embedding meanpool|clam] [--n-pc 5] [--n-rep 20] [--force]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (RESULTS, clinical_table, load_patient_embedding, log_to, setup_matplotlib, strict_fold_of_case,
                    survival_usable)

sys.stdout.reconfigure(encoding='utf-8')
plt = setup_matplotlib()
ap = argparse.ArgumentParser()
ap.add_argument('--embedding', default='meanpool', choices=['meanpool', 'clam'])
ap.add_argument('--source', default='auto', choices=['auto', 'gdc', 'cdr'])
ap.add_argument('--n-pc', type=int, default=5)
ap.add_argument('--n-rep', type=int, default=20)
ap.add_argument('--n-boot', type=int, default=1000)
ap.add_argument('--penalizer', type=float, default=0.1)
ap.add_argument('--force', action='store_true')
args = ap.parse_args()
OUT = RESULTS / 'survival'
log_to(OUT / 'cox_log.txt')

# ---------- 数据 ----------
tab = clinical_table(args.source)
usable, miss = survival_usable(tab, 'label')
print('生存终点缺失（按亚型）:')
print(miss.to_string(index=False))
if not usable and not args.force:
    raise SystemExit('缺失不平衡，Cox 结果会有偏。先按 analysis_survival.py 的提示补全临床数据，或加 --force。')
flag = '' if usable else '【缺失不平衡，结果无效】'
d = tab.dropna(subset=['os_days', 'stage_group']).copy()
d['t'] = d['os_years']
d['event'] = d['os_event'].astype(int)
d['stage_II'] = (d['stage_group'] == 'II').astype(float)
d['stage_III_IV'] = (d['stage_group'] == 'III-IV').astype(float)
d['male'] = (d['sex'] == 'male').astype(float)
d['age'] = d['age'].fillna(d['age'].median())
CLIN = ['stage_II', 'stage_III_IV', 'age', 'male']

if args.embedding == 'meanpool':
    emb = load_patient_embedding('meanpool')
    d = d[d.index.isin(emb.index)]
    Xk = [emb.loc[d.index].values] * 5
else:
    fold_of = strict_fold_of_case()
    embs = [load_patient_embedding(f'clam_fold{k}') for k in range(5)]
    d = d[d.index.isin(embs[0].index) & d.index.isin(list(fold_of))]
    Xk = [e.loc[d.index].values for e in embs]
n_ev = int(d['event'].sum())
print(f"\n分析集: {len(d)} 例（有终点 + 有分期 + 有嵌入），事件 {n_ev} 例；亚型 {d['label'].value_counts().to_dict()}")
print(f"临床模型 {len(CLIN)} 个参数 → 每参数事件数 {n_ev / len(CLIN):.0f}；加形态 PC 后 {n_ev / (len(CLIN) + args.n_pc):.0f}"
      f"（经验下限约 10）")

# ---------- 全数据临床模型 + 比例风险检验 ----------
full = CoxPHFitter(penalizer=0.01).fit(d[CLIN + ['t', 'event', 'label']], 't', 'event', strata=['label'])
print('\n== 全数据临床 Cox（以亚型分层）==')
print(full.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(3).to_string())
ph = proportional_hazard_test(full, d[CLIN + ['t', 'event', 'label']], time_transform='rank')
print('比例风险检验（p<0.05 提示该变量的 HR 随时间变化）:')
print(ph.summary[['test_statistic', 'p']].round(3).to_string())


# ---------- 交叉验证 ----------
def design(tr, te, k, use_morph):
    sc = StandardScaler().fit(d.iloc[tr][CLIN])
    F_tr, F_te = [sc.transform(d.iloc[tr][CLIN])], [sc.transform(d.iloc[te][CLIN])]
    if use_morph:
        X = Xk[k]
        s2 = StandardScaler().fit(X[tr])
        pca = PCA(args.n_pc, random_state=0).fit(s2.transform(X[tr]))
        F_tr.append(pca.transform(s2.transform(X[tr])))
        F_te.append(pca.transform(s2.transform(X[te])))
    cols = CLIN + ([f'pc{i + 1}' for i in range(args.n_pc)] if use_morph else [])
    return (pd.DataFrame(np.hstack(F_tr), columns=cols, index=d.index[tr]),
            pd.DataFrame(np.hstack(F_te), columns=cols, index=d.index[te]))


def fit_predict(tr, te, k, use_morph):
    F_tr, F_te = design(tr, te, k, use_morph)
    F_tr[['t', 'event', 'label']] = d.iloc[tr][['t', 'event', 'label']].values
    F_tr['t'] = F_tr['t'].astype(float)
    F_tr['event'] = F_tr['event'].astype(int)
    cph = CoxPHFitter(penalizer=args.penalizer).fit(F_tr, 't', 'event', strata=['label'])
    F_te['label'] = d.iloc[te]['label'].values
    r_te = cph.predict_log_partial_hazard(F_te).values.ravel()
    r_tr = cph.predict_log_partial_hazard(F_tr.drop(columns=['t', 'event'])).values.ravel()
    return r_tr, r_te


def cindex(idx, risk):
    return concordance_index(d['t'].values[idx], -risk, d['event'].values[idx])


if args.embedding == 'meanpool':
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.n_rep, random_state=0)
    all_splits = list(rskf.split(np.zeros(len(d)), d['event']))
    reps = [all_splits[i:i + 5] for i in range(0, len(all_splits), 5)]
else:
    folds = np.array([fold_of[c] for c in d.index])
    reps = [[(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(5)]]

fold_c = {'clin': [], 'both': []}
pooled = {'clin': [], 'both': []}
first = None
for r, splits in enumerate(reps):
    oof = {'clin': np.full(len(d), np.nan), 'both': np.full(len(d), np.nan)}
    group = np.empty(len(d), dtype=object)
    for k, (tr, te) in enumerate(splits):
        for name, use in (('clin', False), ('both', True)):
            r_tr, r_te = fit_predict(tr, te, k, use)
            oof[name][te] = r_te
            if d['event'].values[te].sum() > 0:
                fold_c[name].append(cindex(te, r_te))
            if name == 'both':
                group[te] = np.where(r_te >= np.median(r_tr), '高', '低')   # 阈值来自训练折
    for name in oof:
        pooled[name].append(cindex(np.arange(len(d)), oof[name]))
    if r == 0:
        first = (oof, group)

dC = np.array(pooled['both']) - np.array(pooled['clin'])
print(f'\n== 交叉验证 C-index（{len(reps)} 次重复 × 5 折）==')
for name, lab in (('clin', '临床（分期+年龄+性别，亚型分层）'), ('both', f'临床 + 形态 PC1-{args.n_pc}')):
    print(f"  {lab}: 每折 {np.mean(fold_c[name]):.3f} ± {np.std(fold_c[name], ddof=1):.3f} | 池化 OOF {np.mean(pooled[name]):.3f}")
rng_txt = (f'，各次重复的 2.5–97.5% 范围 [{np.quantile(dC, 0.025):+.3f}, {np.quantile(dC, 0.975):+.3f}]'
           if len(dC) > 1 else '（只有一次划分，看下面的 bootstrap 区间）')
print(f'  ΔC（加形态 − 不加）: 均值 {dC.mean():+.3f}{rng_txt}')

oof, group = first
rng = np.random.default_rng(0)
boot = []
for _ in range(args.n_boot):
    idx = rng.integers(0, len(d), len(d))
    if d['event'].values[idx].sum() == 0:
        continue
    boot.append((cindex(idx, oof['clin'][idx]), cindex(idx, oof['both'][idx])))
boot = np.array(boot)
ci = lambda x: f'[{np.quantile(x, 0.025):.3f}, {np.quantile(x, 0.975):.3f}]'
print(f"  第一次重复的患者 bootstrap 95% 区间: 临床 {ci(boot[:, 0])}，临床+形态 {ci(boot[:, 1])}，"
      f"ΔC {ci(boot[:, 1] - boot[:, 0])}")

# ---------- 风险分组 KM ----------
hi, lo = d[group == '高'], d[group == '低']
lr = logrank_test(hi['t'], lo['t'], hi['event'], lo['event'])
fig, ax = plt.subplots(figsize=(8, 5.5))
for sub, name, c in [(hi, '高风险', '#d62728'), (lo, '低风险', '#2ca02c')]:
    KaplanMeierFitter().fit(sub['t'], sub['event'], label=f'{name} (n={len(sub)})').plot_survival_function(
        ax=ax, color=c, ci_alpha=0.15)
ax.set_title(f'{flag}临床+形态 Cox 的 OOF 风险分组（阈值 = 训练折中位数）\nlog-rank p={lr.p_value:.3g}')
ax.set_xlabel('年')
ax.set_ylabel('总生存概率')
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(OUT / 'km_risk_groups.png', dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'\n风险分组: 高 {len(hi)} / 低 {len(lo)}，log-rank p={lr.p_value:.3g}')

with open(OUT / 'cox_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f'{flag}分析集 n={len(d)}，事件 {n_ev}；嵌入 {args.embedding}；{len(reps)} 次重复 × 5 折\n\n')
    f.write('== 全数据临床 Cox（以亚型分层）==\n')
    f.write(full.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(3).to_string() + '\n\n')
    f.write('== 交叉验证 C-index ==\n')
    for name in ('clin', 'both'):
        f.write(f"{name}: 每折 {np.mean(fold_c[name]):.3f} ± {np.std(fold_c[name], ddof=1):.3f}，池化 {np.mean(pooled[name]):.3f}\n")
    f.write(f'ΔC 均值 {dC.mean():+.3f}{rng_txt}；bootstrap 95% {ci(boot[:, 1] - boot[:, 0])}\n')
    f.write(f'风险分组 KM log-rank p={lr.p_value:.3g}\n')
print('✅ 汇总 → cox_summary.txt')
