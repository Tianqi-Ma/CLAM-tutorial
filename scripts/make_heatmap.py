# -*- coding: utf-8 -*-
"""轻量注意力热图：复用已有特征，无需重跑特征提取。
用法: python make_heatmap.py <slide_id前23字符> <checkpoint> <输出目录>
原理: A = model(h, attention_only=True) 拿到每个 patch 的注意力分数（softmax 之前的原始分）,
      按 patching 阶段的 coords 画回切片缩略图, jet 色标 + alpha 叠加。
读图注意（2026-10 审计补充）:
  - checkpoint 要用"这张切片在其测试集里"的那一折，否则画的是模型在训练数据上的注意力。
  - CLAM_SB 只有一个注意力分支，LUAD 和 LUSC 的证据都会得高分；高注意力 = 对切片表示贡献大，
    不等于"像癌"，也可能落在出血、炭末、褶皱等伪影上。标题里的概率接近 0.5 时热图尤其不可读。
  - 2026-10 修正：v1 把 Y_prob（模型输出已经是 softmax 概率）又做了一次 softmax，标题概率被压向 0.5。
"""
import sys, os
import numpy as np
import torch
import h5py
import openslide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from PIL import Image
from scipy.ndimage import gaussian_filter

from common import CLAM_DIR, ROOT, LABELS

sys.path.insert(0, str(CLAM_DIR))
from models.model_clam import CLAM_SB
from topk.svm import SmoothTop1SVM

D_PT   = f'{ROOT}/results/tcga/features/pt_files'
D_H5   = f'{ROOT}/results/tcga/patching/patches'
D_SVS  = f'{ROOT}/data/slides/tcga'
PATCH  = 256          # patching 时的 patch_size (level 0)
VIS_DS = 32           # 可视化降采样倍数
ALPHA  = 0.45

def main(slide_prefix, ckpt, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 找全名
    sid  = next(f[:-3] for f in os.listdir(D_PT) if f.startswith(slide_prefix))
    svs  = next(f for f in os.listdir(D_SVS) if f.startswith(slide_prefix))

    # 1) 注意力分数
    model = CLAM_SB(dropout=0., n_classes=2, subtyping=True, embed_dim=1024,
                    instance_loss_fn=SmoothTop1SVM(2))
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    h = torch.load(f'{D_PT}/{sid}.pt', map_location='cpu', mmap=True)
    with torch.no_grad():
        A = model(h, attention_only=True).squeeze().cpu().numpy()   # (N,)
        logits, Y_prob, *_ = model(h)
    prob = Y_prob[0].numpy()                     # Y_prob 已经是 softmax 概率，不要再 softmax
    pred = LABELS[int(prob.argmax())]

    # 2) 坐标 + 缩略图
    with h5py.File(f'{D_H5}/{sid}.h5', 'r') as f:
        coords = f['coords'][:]
    slide = openslide.open_slide(f'{D_SVS}/{svs}')
    W, H = slide.dimensions
    tw, th = W // VIS_DS, H // VIS_DS
    thumb = np.array(slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])
                     .convert('RGB').resize((tw, th), Image.BILINEAR))

    # 3) 画热度：每个 patch 在缩略图上占 PATCH/VIS_DS 像素
    cell = PATCH // VIS_DS                       # 8 px
    heat = np.zeros((th, tw), dtype=np.float32)
    cnt  = np.zeros((th, tw), dtype=np.float32)
    # 分数按百分位归一（抗离群）
    lo, hi = np.percentile(A, 1), np.percentile(A, 99)
    An = np.clip((A - lo) / (hi - lo + 1e-9), 0, 1)
    for (x, y), a in zip(coords, An):
        xi, yi = x // VIS_DS, y // VIS_DS
        heat[yi:yi+cell, xi:xi+cell] += a
        cnt[yi:yi+cell, xi:xi+cell] += 1
    mask = cnt > 0
    heat[mask] /= cnt[mask]
    heat = gaussian_filter(heat, sigma=1.2)      # 轻微平滑

    # 4) 叠加展示
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(thumb); axes[0].set_title(f'{sid[:23]} (原图)', fontsize=10); axes[0].axis('off')
    axes[1].imshow(thumb)
    hm = axes[1].imshow(np.ma.masked_where(~mask, heat), cmap='jet', alpha=ALPHA, vmin=0, vmax=1)
    axes[1].set_title(f'注意力热图 | 预测: {pred} ({prob.max():.2f})', fontsize=10); axes[1].axis('off')
    fig.colorbar(hm, ax=axes[1], fraction=0.02, label='attention (percentile-norm)')
    out = f'{out_dir}/{sid[:23]}_heatmap.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print('saved:', out)
    # 顺带打印 top-10 高注意力 patch 坐标（供 ROI 放大核对）
    top10 = A.argsort()[-10:][::-1]
    print('top-10 attention patches (x, y, score):')
    for i in top10:
        print(f'  {tuple(coords[i])}  {A[i]:.5f}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
