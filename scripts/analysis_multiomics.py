# -*- coding: utf-8 -*-
"""多组学分析：组织学形态特征 × 基因表达
1. 已知 marker 基因 sanity check（NKX2-1=LUAD, TP63/KRT5/KRT6A=LUSC）
2. CLAM 注意力池化得到"形态嵌入"（512 维）。划分已升级为严格 k 折（make_strict_splits.py），
   每张切片恰好是一折的测试集 → 全部零泄漏嵌入（权重用严格折实验 luad_lusc_CLAM_sb_strict_s1）。
   背景：CLAM 官方 create_splits_seq 是蒙特卡洛 CV（每折随机 80/10/10），不是严格分区，
   旧版（100 张时代）实测 45 张进过 test、26 张仅 val、29 张全在 train —— 已作为踩坑 #11 修复。
3. 形态 PCA 主成分 vs marker 表达量相关性 + 岭回归交叉验证
"""
import sys, glob, os, argparse
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'E:/Projects/DP/CLAM')
import numpy as np, pandas as pd, torch, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import mannwhitneyu, spearmanr
from models.model_clam import CLAM_SB
from topk.svm import SmoothTop1SVM

PROJ = r'E:/Projects/DP/CLAM-tutorial'
CLAM = r'E:/Projects/DP/CLAM'
OUT  = f'{PROJ}/results/multiomics'
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--splits-tag', default='task_2_strict150', help='CLAM/splits/ 下的划分目录名')
ap.add_argument('--ckpt-exp', default='luad_lusc_CLAM_sb_strict_s1', help='CLAM/results/ 下的实验目录名')
args = ap.parse_args()

# ---------- 1. 标签 + 表达矩阵 ----------
csv = pd.read_csv(f'{PROJ}/data/dataset_csv/tcga_luad_lusc.csv')
label_of_case = dict(zip(csv['case_id'], csv['label']))
MARKERS = {'NKX2-1': 'LUAD（TTF-1，肺腺癌标志）', 'TP63': 'LUSC（鳞癌标志）',
           'KRT5': 'LUSC（鳞癌角化）', 'KRT6A': 'LUSC（鳞癌角化）'}

expr = {}
for f in glob.glob(f'{PROJ}/data/metadata/expression/*.tsv'):
    case = os.path.basename(f).split('__')[0]
    df = pd.read_csv(f, sep='\t', comment='#', usecols=['gene_name', 'gene_type', 'tpm_unstranded'])
    df = df[df['gene_type'] == 'protein_coding'].drop_duplicates('gene_name').set_index('gene_name')
    expr[case] = df['tpm_unstranded']
E = pd.DataFrame(expr)                      # 行=基因, 列=case
E = E[[c for c in E.columns if c in label_of_case]]
print(f'表达矩阵: {E.shape[0]} 个蛋白编码基因 × {E.shape[1]} 个患者（114 个文件中匹配到标签的）')

logE = np.log1p(E)
print('\n== marker 基因 sanity check（log1p TPM 中位数, Mann-Whitney）==')
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
marker_p = {}
for ax, (g, desc) in zip(axes, MARKERS.items()):
    a = logE.loc[g, [c for c in E.columns if label_of_case[c] == 'LUAD']]
    b = logE.loc[g, [c for c in E.columns if label_of_case[c] == 'LUSC']]
    p = mannwhitneyu(a, b).pvalue
    marker_p[g] = p
    ax.boxplot([a, b], labels=['LUAD', 'LUSC'], showfliers=False)
    ax.set_title(f'{g}\np={p:.1e}', fontsize=10)
    ax.set_ylabel('log1p(TPM)')
    print(f'  {g:7s} ({desc}): LUAD中位 {a.median():.2f} vs LUSC中位 {b.median():.2f}, p={p:.2e}')
fig.suptitle('已知的肺腺癌/鳞癌 marker 基因，在我们的标签分组下表达差异是否符合预期（方向对=标签和数据都健康）')
fig.tight_layout(); fig.savefig(f'{OUT}/markers_boxplot.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---------- 2. 形态嵌入（注意力池化，512 维） ----------
# 划分已升级为严格 k 折（scripts/make_strict_splits.py，见 notebook §4 与踩坑 #11）：
# 每张切片恰好属于某一折的测试集 → 全部用它当 test 那一折的模型提嵌入，零泄漏、无借用。
# （防御性 fallback 保留：万一换划分文件，退到 val 折 → fold-0，并如实打印来源统计。）
print('\n== 计算形态嵌入（严格 k 折：每张切片用它当 test 那一折的模型，零泄漏）==')
SPLITS_DIR = f'{CLAM}/splits/{args.splits_tag}'
CKPT_DIR   = f'{CLAM}/results/{args.ckpt_exp}'   # 严格折实验的权重（与划分一一对应）
fold_sets = []
for k in range(5):
    sp = pd.read_csv(f'{SPLITS_DIR}/splits_{k}.csv')
    fold_sets.append({c: set(sp[c].dropna()) for c in ['train', 'val', 'test']})

def pick_fold(s):
    for k in range(5):
        if s in fold_sets[k]['test']: return k, 'test'
    for k in range(5):
        if s in fold_sets[k]['val']:  return k, 'val'
    return 0, 'train-leak'

src_count = {'test': 0, 'val': 0, 'train-leak': 0}
emb, emb_src = {}, {}
D = f'{PROJ}/results/tcga/features/pt_files'
models = {}
for i, s in enumerate(csv['slide_id']):
    k, src = pick_fold(s)
    src_count[src] += 1
    if k not in models:
        m = CLAM_SB(dropout=0., n_classes=2, subtyping=True, embed_dim=1024,
                    instance_loss_fn=SmoothTop1SVM(2))
        m.load_state_dict(torch.load(f'{CKPT_DIR}/s_{k}_checkpoint.pt', map_location='cpu'))
        m.eval(); models[k] = m
    m = models[k]
    h = torch.load(f'{D}/{s}.pt', map_location='cpu', mmap=True)
    with torch.no_grad():
        hh = m.attention_net[1](m.attention_net[0](h))       # fc1 + ReLU → 512 维
        A, _ = m.attention_net[3](hh)                        # 门控打分 → (N,1)
        A = torch.softmax(A, dim=0)
        M = (A * hh).sum(0)                                  # 注意力池化 → 512 维形态嵌入
    emb[s] = M.numpy()
    emb_src[s] = src
    del h
    if i % 25 == 24: print(f'  {i+1}/100')
print(f'  选折来源统计: test折 {src_count["test"]} 张（零泄漏）/ val折 {src_count["val"]} 张（近似无泄漏）'
      f' / 借用fold-0 {src_count["train-leak"]} 张（泄漏，见下）')
X = pd.DataFrame(emb).T                                     # 行=slide, 列=512
sl2case = csv.set_index('slide_id')['case_id']
X_case = X.groupby(sl2case).mean()                          # 患者级（多切片取平均）
# 患者级"干净"标记：该患者所有切片都是 test/val 来源才算干净
clean_slide = pd.Series({s: (emb_src[s] != 'train-leak') for s in emb_src})
clean_case = clean_slide.groupby(sl2case).all()
X_case = X_case.loc[[c for c in X_case.index if c in E.columns]]
print(f'形态嵌入: {X_case.shape[0]} 患者 × 512 维（干净患者 {int(clean_case[X_case.index].sum())}，'
      f'含泄漏切片的患者 {int((~clean_case[X_case.index]).sum())}）')
# 保存嵌入供下游（Cox 预后模型等）复用：morph_embeddings.csv + 逐患者来源 meta
X_case.round(6).to_csv(f'{OUT}/morph_embeddings.csv')
pd.DataFrame({'case': X_case.index,
              'label': [label_of_case[c] for c in X_case.index],
              'clean': [bool(clean_case[c]) for c in X_case.index],
              'n_slides': [int((sl2case == c).sum()) for c in X_case.index]}
             ).to_csv(f'{OUT}/morph_embeddings_meta.csv', index=False)
print('嵌入已保存: morph_embeddings.csv (+_meta.csv)')

# ---------- 3. 形态 PCA vs marker 表达 ----------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
Z = PCA(10).fit_transform(StandardScaler().fit_transform(X_case))
cases = list(X_case.index)
labs = [label_of_case[c] for c in cases]

print('\n== 形态主成分 × marker 表达（Spearman 相关）==')
rows = []
for g in MARKERS:
    y = logE.loc[g, cases].values
    r_best, pc_best, p_best = 0, -1, 1
    for j in range(5):
        r, p = spearmanr(Z[:, j], y)
        if abs(r) > abs(r_best): r_best, pc_best, p_best = r, j, p
    rows.append((g, pc_best + 1, r_best, p_best))
    print(f'  {g:7s}: 最强相关 PC{pc_best+1}, r={r_best:+.2f}, p={p_best:.2e}')

# 散点图：最强那对
g_show, pc_show, r_show, p_show = max(rows, key=lambda x: abs(x[2]))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for lab, c in [('LUAD', '#d62728'), ('LUSC', '#1f77b4')]:
    m = [i for i, l in enumerate(labs) if l == lab]
    ax.scatter(Z[m, 0], Z[m, 1], s=25, alpha=0.7, c=c, label=lab)
ax.set_xlabel('形态 PC1'); ax.set_ylabel('形态 PC2')
ax.set_title('形态嵌入 PCA（每个点=一个患者）\n→ 两类癌在"纯看片子的形态空间"里也分得开'); ax.legend()
ax = axes[1]
y = logE.loc[g_show, cases].values
for lab, c in [('LUAD', '#d62728'), ('LUSC', '#1f77b4')]:
    m = [i for i, l in enumerate(labs) if l == lab]
    ax.scatter(Z[m, pc_show], y[m], s=25, alpha=0.7, c=c, label=lab)
ax.set_xlabel(f'形态 PC{pc_show}'); ax.set_ylabel(f'{g_show} 表达 log1p(TPM)')
ax.set_title(f'跨模态关联：形态 PC{pc_show} × {g_show} 表达\nSpearman r={r_show:+.2f}, p={p_show:.1e}'); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/morph_vs_expr.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# 岭回归：形态 → 预测 NKX2-1 表达（5 折 CV）
y = logE.loc['NKX2-1', cases].values
scores = cross_val_score(RidgeCV(alphas=np.logspace(-2, 3, 20)), Z[:, :5], y, cv=5, scoring='r2')
print(f'\n岭回归（形态 PC1-5 → NKX2-1 表达）5折CV R²: {scores.mean():.2f} ± {scores.std():.2f}')
print('（>0 说明形态里确实含有该基因表达的信息；数值温和属正常——形态只是表达的一个模糊投影）')

# ---- 稳健性复核：只在"干净"患者（test/val 来源）上重算图中那对 + 岭回归 ----
ci = [i for i, c in enumerate(cases) if clean_case[c]]
print(f'\n== 稳健性复核：剔除含泄漏切片的 {len(cases)-len(ci)} 个患者，仅用干净 n={len(ci)} ==')
Zc = Z[ci]
yc_fig = logE.loc[g_show, [cases[i] for i in ci]].values
r_c, p_c = spearmanr(Zc[:, pc_show - 1], yc_fig)
print(f'  图右那对（PC{pc_show} × {g_show}）：干净子集 r={r_c:+.2f}, p={p_c:.1e}'
      f'（全体 r={r_show:+.2f} → 方向/量级一致即说明结论不是泄漏造成的）')
yc = logE.loc['NKX2-1', [cases[i] for i in ci]].values
scores_c = cross_val_score(RidgeCV(alphas=np.logspace(-2, 3, 20)), Zc[:, :5], yc, cv=5, scoring='r2')
print(f'  岭回归干净子集 5折CV R²: {scores_c.mean():.2f} ± {scores_c.std():.2f}（全体 {scores.mean():.2f}）')
print('\n✅ 图已保存:', f'{OUT}/markers_boxplot.png,', f'{OUT}/morph_vs_expr.png')
