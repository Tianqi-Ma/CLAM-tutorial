# -*- coding: utf-8 -*-
"""统计每张切片的扫描倍率（2026-10 审计后新增）。

create_patches_fp.py 默认在 level 0（扫描原始分辨率）切 256×256。TCGA 的诊断切片有 40×（约 0.25 µm/像素）
也有 20×（约 0.5 µm/像素）：同样 256 像素，40× 切片的 patch 只覆盖 20× 切片四分之一的组织面积。
混着用时，模型看到的"一个 patch"不是同一个物理尺度，倍率还可能和来源中心（进而和标签）相关。
本脚本只读 svs 文件头，几秒一张，输出 results/tcga/slide_mpp.csv 和倍率分布。
统一的办法：40× 切片用 512×512 切再缩到 256（CLAM 旧版 extract_features 的 custom_downsample=2），
或用 --patch_level 选一个约 0.5 µm/像素的金字塔层。
用法: python scripts/check_slide_mpp.py
"""
import sys

import openslide
import pandas as pd

from common import DATA, RESULTS, load_dataset

sys.stdout.reconfigure(encoding='utf-8')
ds = load_dataset()
rows = []
for _, r in ds.iterrows():
    path = DATA / 'slides' / 'tcga' / f"{r['slide_id']}.svs"
    if not path.exists():
        rows.append({'slide_id': r['slide_id'], 'label': r['label'], 'tss': r['tss'], 'mpp': None, 'objective': None})
        continue
    s = openslide.OpenSlide(str(path))
    p = s.properties
    rows.append({'slide_id': r['slide_id'], 'label': r['label'], 'tss': r['tss'],
                 'mpp': float(p.get('openslide.mpp-x', 'nan')),
                 'objective': p.get('openslide.objective-power') or p.get('aperio.AppMag'),
                 'level0_size': f'{s.dimensions[0]}x{s.dimensions[1]}', 'levels': s.level_count})
    s.close()
df = pd.DataFrame(rows)
out = RESULTS / 'tcga' / 'slide_mpp.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
df['mag'] = pd.cut(df['mpp'], [0, 0.35, 0.75, 10], labels=['~40×', '~20×', '更低'])
print(f'{len(df)} 张切片，缺文件 {int(df["mpp"].isna().sum())} 张')
print('\n倍率 × 亚型:')
print(pd.crosstab(df['mag'], df['label'], margins=True).to_string())
print(f'\n256 像素 patch 的边长: 40× ≈ {256 * 0.25:.0f} µm，20× ≈ {256 * 0.5:.0f} µm')
if df['mag'].nunique() > 1:
    print('⚠️ 倍率混杂：建议统一到 20× 等效后重新切块、提特征；如果倍率和亚型相关，它也是模型可以利用的捷径。')
print('已写入', out)
