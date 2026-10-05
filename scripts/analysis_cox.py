# -*- coding: utf-8 -*-
"""生存分析深入版：Cox 比例风险模型 + 形态学预后评分（方法学演示）
1. 单因素 Cox 基线表：分期(序数 I~IV→1~4)、年龄、亚型标签
2. 形态嵌入(512 维, 严格折零泄漏) → PCA → 惩罚 CoxPH：
   - 交叉验证 C-index：严格 k 折分组，测试折患者由其余 4 折训练的 Cox 预测风险
     → 池化 out-of-fold 风险评分算 C-index（无泄漏的模型比较）
   - 三组对比: 纯临床(stage+age) / 纯形态(PC1-5) / 临床+形态
3. OOF 风险评分中位分组 → KM 高/低风险曲线 + log-rank（honest 版，分数全是 OOF）
⚠️ 数据声明：26 例 Alive LUAD 无随访时间被剔除（见 analysis_survival.py 的缺失陷阱分析），
   本分析展示的是"WSI 预后建模的正确姿势"，n=71/事件 45 的规模不足以发现真实生物标志物。
依赖: morph_embeddings.csv（analysis_multiomics.py）+ survival_table.csv（analysis_survival.py）
用法: python analysis_cox.py [--splits-tag task_2_strict150]
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

PROJ = r'E:/Projects/DP/CLAM-tutorial'
CLAM = r'E:/Projects/DP/CLAM'
OUT  = f'{PROJ}/results/survival'
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--splits-tag', default='task_2_strict150')
ap.add_argument('--n-pc', type=int, default=5)
args = ap.parse_args()

# ---------- 数据 ----------
sv = pd.read_csv(f'{OUT}/survival_table.csv')
emb_path = f'{PROJ}/results/multiomics/morph_embeddings.csv'
if not os.path.exists(emb_path):
    raise SystemExit('缺少 morph_embeddings.csv —— 先跑 analysis_multiomics.py 生成形态嵌入')
emb = pd.read_csv(emb_path, index_col=0)

def stage_ord(s):
    if not isinstance(s, str): return np.nan
    if s.startswith('Stage IV'): return 4
    if s.startswith('Stage III'): return 3
    if s.startswith('Stage II'): return 2
    return 1
sv['stage_n'] = sv['stage'].map(stage_ord)
df = sv.merge(emb, left_on='case', right_index=True, how='inner')
PCOLS = [c for c in emb.columns]
print(f'合并后: {len(df)} 患者（有嵌入+随访），事件 {int(df["event"].sum())} 例')
print('⚠️ 声明：26 例 Alive LUAD 因无随访被剔除 → 以下是方法学演示，不是标志物发现\n')

# 患者 → 严格折号（其切片所在测试折；分组保证同一患者只在一折）
fold_of = {}
K = 5
for k in range(K):
    sp = pd.read_csv(f'{CLAM}/splits/{args.splits_tag}/splits_{k}.csv')
    sl2case = pd.read_csv(f'{PROJ}/data/dataset_csv/tcga_luad_lusc.csv').set_index('slide_id')['case_id']
    for s in sp['test'].dropna():
        if s in sl2case.index: fold_of[sl2case[s]] = k
df['fold'] = df['case'].map(fold_of)
assert df['fold'].notna().all(), '有患者找不到所属折'

# ---------- 1. 单因素 Cox 基线表 ----------
print('== 单因素 Cox（风险比 HR>1 = 更危险）==')
uni_rows = []
for name, colds in [('分期(序数)', ['stage_n']), ('年龄', ['age']), ('亚型(LUSC=1)', None)]:
    d = df.copy()
    if name.startswith('亚型'): d['is_lusc'] = (d['label'] == 'LUSC').astype(int); colds = ['is_lusc']
    d = d.dropna(subset=colds)
    cph = CoxPHFitter(penalizer=0.01).fit(d[colds + ['t', 'event']], 't', 'event')
    hr = np.exp(cph.params_[colds[0]]); p = cph.summary['p'][colds[0]]
    uni_rows.append((name, hr, p, len(d)))
    print(f'  {name:12s}: HR={hr:.2f}, p={p:.3f} (n={len(d)})')

# ---------- 2. 交叉验证 C-index 三模型对比 ----------
def oof_risk(feats_fn, tag):
    """逐折：训练折拟合(含scaler/PCA)，测试折预测 log partial hazard。返回全部患者的 OOF 风险分。"""
    risk = pd.Series(np.nan, index=df['case'], dtype=float)
    for k in range(K):
        tr = df[df['fold'] != k]; te = df[df['fold'] == k]
        Xtr, Xte, base = feats_fn(tr, te)
        dtr = pd.DataFrame(Xtr, columns=[f'f{i}' for i in range(Xtr.shape[1])])
        dtr['t'] = tr['t'].values; dtr['event'] = tr['event'].values
        cph = CoxPHFitter(penalizer=0.1).fit(dtr, 't', 'event')
        dte = pd.DataFrame(Xte, columns=dtr.columns[:-2])
        risk.loc[te['case'].values] = cph.predict_log_partial_hazard(dte).values.flatten()
    assert np.isfinite(risk.values).all(), f'{tag}: 有患者没拿到风险分'
    ci = concordance_index(df['t'], -risk.values, df['event'])
    print(f'  {tag:18s}: C-index(OOF) = {ci:.3f}')
    return risk, ci

def feats_clinical(tr, te):
    sc = StandardScaler().fit(tr[['stage_n', 'age']])
    return sc.transform(tr[['stage_n', 'age']]), sc.transform(te[['stage_n', 'age']]), None

def feats_morph(tr, te):
    sc = StandardScaler().fit(tr[PCOLS]); pc = PCA(args.n_pc).fit(sc.transform(tr[PCOLS]))
    return pc.transform(sc.transform(tr[PCOLS])), pc.transform(sc.transform(te[PCOLS])), None

def feats_both(tr, te):
    Xtr_c, Xte_c, _ = feats_clinical(tr, te)
    Xtr_m, Xte_m, _ = feats_morph(tr, te)
    return np.hstack([Xtr_c, Xtr_m]), np.hstack([Xte_c, Xte_m]), None

# 临床模型对缺失分期敏感：仅用分期非缺失患者做三组公平对比
dfc = df.dropna(subset=['stage_n']).copy()
print(f'\n== 交叉验证 C-index（分期完整 n={len(dfc)}，事件 {int(dfc["event"].sum())}）==')
df_save = df; df = dfc                      # 临时切换到分期完整子集
risk_c, ci_c = oof_risk(feats_clinical, '临床(stage+age)')
risk_m, ci_m = oof_risk(feats_morph,    f'形态(PC1-{args.n_pc})')
risk_b, ci_b = oof_risk(feats_both,     '临床+形态')

# ---------- 3. OOF 风险分组 KM ----------
dfc['risk'] = risk_b.values                 # log partial hazard（实数，无下溢问题）
med = dfc['risk'].median()
hi = dfc[dfc['risk'] >= med]; lo = dfc[dfc['risk'] < med]
print(f'\n风险分组: 高 {len(hi)} / 低 {len(lo)}（中位阈值 {med:.2f}）')
assert len(hi) > 0 and len(lo) > 0
lr = logrank_test(hi['t'], lo['t'], hi['event'], lo['event'])
fig, ax = plt.subplots(figsize=(8, 5.5))
for sub, name, c in [(hi, f'高风险组 (n={len(hi)})', '#d62728'), (lo, f'低风险组 (n={len(lo)})', '#2ca02c')]:
    kmf = KaplanMeierFitter()
    kmf.fit(sub['t'], sub['event'], label=name)
    kmf.plot_survival_function(ax=ax, color=c, ci_alpha=0.15)
ax.set_title(f'形态+临床 Cox 模型的 OOF 风险评分分组 KM\n'
             f'（分数全部来自"没见过该患者"的模型，无泄漏）log-rank p={lr.p_value:.3f}')
ax.set_xlabel('年'); ax.set_ylabel('生存概率'); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(f'{OUT}/km_risk_groups.png', dpi=110, bbox_inches='tight'); plt.close(fig)
print(f'\n高/低风险组 KM: p={lr.p_value:.3f} -> km_risk_groups.png')
df = df_save                                # 还原

# ---------- 汇总 ----------
with open(f'{OUT}/cox_summary.txt', 'w', encoding='utf-8') as f:
    f.write('== 单因素 Cox ==\n')
    for name, hr, p, n in uni_rows: f.write(f'{name}: HR={hr:.2f}, p={p:.3f} (n={n})\n')
    f.write(f'\n== OOF C-index（n={len(dfc)}）==\n临床: {ci_c:.3f}\n形态: {ci_m:.3f}\n临床+形态: {ci_b:.3f}\n')
    f.write(f'\n风险分组 KM log-rank p={lr.p_value:.3f}\n')
    f.write('\n声明: 26 例 Alive LUAD 无随访被剔除，结果为方法学演示。\n')
print('\n✅ 汇总 -> cox_summary.txt')
