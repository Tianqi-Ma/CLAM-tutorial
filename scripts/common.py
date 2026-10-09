# -*- coding: utf-8 -*-
"""分析脚本共用的路径、数据读取和临床变量整理（2026-10 审计后新增）。

之前每个 analysis_*.py 各自复制一份读表达、读临床的代码，并且有三个共同的错误：
  1. 表达矩阵按患者覆盖写入：一个患者有多个 RNA-seq 文件（原发肿瘤 01 + 癌旁正常 11 + 复发 02）时，
     最后读到的那个赢——可能把癌旁正常组织当成肿瘤用。现在只取原发肿瘤（sample type 01）。
  2. 随访时间只看 demographic 和 diagnoses，没请求 GDC 的 follow_ups 实体，且用 `or` 把 0 天当缺失。
     TCGA-LUAD 的随访记录在 follow_ups 里，于是活着的 LUAD 患者全部"没有随访时间"。
  3. 性别读 demographic.gender（GDC 已改名 sex_at_birth），分期取"第一个有分期的诊断"而不是原发诊断。
路径不再写死 E:/：默认取本仓库根目录及其同级的 CLAM 目录，可用环境变量 CLAM_TUTORIAL_ROOT / CLAM_DIR 覆盖。
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('CLAM_TUTORIAL_ROOT', Path(__file__).resolve().parents[1]))
CLAM_DIR = Path(os.environ.get('CLAM_DIR', ROOT.parent / 'CLAM'))
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
DATASET_CSV = DATA / 'dataset_csv' / 'tcga_luad_lusc.csv'
CLINICAL_JSON = DATA / 'metadata' / 'clinical.json'
EXPR_DIR = DATA / 'metadata' / 'expression'
RNA_MANIFEST = DATA / 'metadata' / 'rna_files.csv'      # download_tcga.py --metadata-only 生成
CDR_TABLE = DATA / 'metadata' / 'TCGA-CDR.csv'           # 可选：Liu et al. 2018 Cell 的 TCGA-CDR 表另存为 csv
FEAT_DIR = RESULTS / 'tcga' / 'features' / 'pt_files'
EMB_DIR = RESULTS / 'embeddings'
STRICT_EVAL_DIR = RESULTS / 'eval_strict150'
LABELS = ['LUAD', 'LUSC']                                # CLAM label_dict: LUAD=0, LUSC=1


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def log_to(path):
    """之后的 print 同时写进 path（notebook 直接读这个日志，日志和脚本输出保持一致）。"""
    import sys
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, f)
    return f


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def tss_of(case_id):
    """TCGA 条形码第二段 = 组织来源中心（tissue source site）。TCGA-49-4514 → '49'。"""
    return case_id.split('-')[1]


def load_dataset():
    """CLAM 数据集表：case_id, slide_id, label，外加 tss 列。"""
    df = pd.read_csv(DATASET_CSV)
    df['tss'] = df['case_id'].map(tss_of)
    return df


def bh_fdr(p):
    """Benjamini-Hochberg 校正，NaN 原样保留。"""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(pv))
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# ---------------------------------------------------------------- 表达矩阵
def _expr_files():
    rows = []
    for f in sorted(glob.glob(str(EXPR_DIR / '*.tsv'))):
        base = os.path.basename(f)
        case, file_name = base.split('__', 1)
        rows.append({'path': f, 'case_id': case, 'file_name': file_name})
    return pd.DataFrame(rows, columns=['path', 'case_id', 'file_name'])


def select_primary_tumor_files(files, manifest=None, strict=True):
    """每个患者挑一个原发肿瘤（sample type 01）RNA-seq 文件。

    manifest（rna_files.csv：file_name, sample_submitter_id, sample_type）存在时按样本类型挑：
    只留 'Primary Tumor'，同一患者多个时取样本编号最小的（01A 先于 01B）。
    没有 manifest 时无法区分肿瘤和癌旁正常：只有一个文件的患者保留，多个文件的患者剔除并报告；
    strict=True 时直接报错，提示先跑 `python scripts/download_tcga.py --metadata-only` 补元数据。
    返回 (选中的文件表, 报告字典)。
    """
    report = {'files': len(files), 'cases': files['case_id'].nunique()}
    if manifest is not None:
        m = files.merge(manifest[['file_name', 'sample_submitter_id', 'sample_type']], on='file_name', how='left')
        report['no_metadata'] = int(m['sample_type'].isna().sum())
        report['not_primary'] = int((m['sample_type'].notna() & (m['sample_type'] != 'Primary Tumor')).sum())
        m = m[m['sample_type'] == 'Primary Tumor'].sort_values(['case_id', 'sample_submitter_id', 'file_name'])
        report['cases_multi_primary'] = int(m['case_id'].duplicated().sum())
        sel = m.drop_duplicates('case_id', keep='first')
        report['method'] = 'manifest'
    else:
        n = files['case_id'].value_counts()
        multi = sorted(n[n > 1].index)
        report['cases_multi_file'] = len(multi)
        if multi and strict:
            raise SystemExit(
                f'{len(multi)} 个患者有多个 RNA-seq 文件（原发肿瘤 / 癌旁正常 / 复发），没有 {RNA_MANIFEST.name} 无法区分。\n'
                f'先跑 `python scripts/download_tcga.py --metadata-only` 生成样本元数据（只查 GDC 元数据，不下载数据），\n'
                f'或加 --allow-ambiguous 剔除这些患者继续。例: {multi[:5]}')
        sel = files[~files['case_id'].isin(multi)]
        report['method'] = 'single-file-only'
    report['selected'] = len(sel)
    return sel, report


def load_expression(value='tpm_unstranded', cases=None, protein_coding=True, strict=True):
    """基因 × 患者矩阵（每个患者一个原发肿瘤样本）。value='unstranded' 返回原始 counts。"""
    files = _expr_files()
    if cases is not None:
        files = files[files['case_id'].isin(set(cases))]
    if files.empty:
        raise SystemExit(f'{EXPR_DIR} 下没有表达文件')
    manifest = pd.read_csv(RNA_MANIFEST) if RNA_MANIFEST.exists() else None
    sel, report = select_primary_tumor_files(files, manifest, strict=strict)
    cols = {}
    for _, r in sel.iterrows():
        df = pd.read_csv(r['path'], sep='\t', comment='#', usecols=['gene_name', 'gene_type', value])
        if protein_coding:
            df = df[df['gene_type'] == 'protein_coding']
        df = df.dropna(subset=['gene_name']).drop_duplicates('gene_name').set_index('gene_name')
        cols[r['case_id']] = df[value]
    E = pd.DataFrame(cols)
    return E, report


def describe_expression_report(rep):
    if rep['method'] == 'manifest':
        return (f"{rep['files']} 个文件 / {rep['cases']} 个患者 → 按样本类型保留原发肿瘤 {rep['selected']} 个"
                f"（剔除非原发 {rep['not_primary']} 个、无元数据 {rep['no_metadata']} 个；"
                f"{rep['cases_multi_primary']} 个患者有多份原发样本，取编号最小的）")
    return (f"{rep['files']} 个文件 / {rep['cases']} 个患者 → 无样本元数据，只保留单文件患者 {rep['selected']} 个"
            f"（{rep['cases_multi_file']} 个多文件患者被剔除）")


# ---------------------------------------------------------------- 临床
def load_clinical():
    return json.load(open(CLINICAL_JSON, encoding='utf-8'))


def primary_diagnosis(case):
    """原发诊断：diagnosis_is_primary_disease=True，其次 classification_of_tumor='primary'，再次第一个。"""
    dx = case.get('diagnoses') or []
    for d in dx:
        if d.get('diagnosis_is_primary_disease') is True:
            return d
    for d in dx:
        if str(d.get('classification_of_tumor', '')).lower() == 'primary':
            return d
    return dx[0] if dx else {}


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v >= 0 else None


def followup_days(case):
    """最长随访天数：follow_ups.days_to_follow_up、各诊断的 days_to_last_follow_up、
    旧版 demographic.days_to_last_follow_up 里取最大值。0 天是有效值，不当缺失。"""
    vals = [_num((case.get('demographic') or {}).get('days_to_last_follow_up'))]
    for d in case.get('diagnoses') or []:
        vals.append(_num(d.get('days_to_last_follow_up')))
    for f in case.get('follow_ups') or []:
        vals.append(_num(f.get('days_to_follow_up')))
        vals.append(_num(f.get('days_to_last_follow_up')))
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def gdc_overall_survival(case):
    """GDC 记录里的总生存：(天数, 事件, 缺失原因)。"""
    dem = case.get('demographic') or {}
    vs = dem.get('vital_status')
    if vs == 'Dead':
        t = _num(dem.get('days_to_death'))
        return (t, 1, None) if t is not None else (None, None, 'Dead 但无 days_to_death')
    if vs == 'Alive':
        t = followup_days(case)
        return (t, 0, None) if t is not None else (None, None, 'Alive 但无任何随访天数')
    return None, None, 'vital_status 缺失'


def stage_roman(s):
    """'Stage IIIA' → 'III'；无法识别返回 None。"""
    if not isinstance(s, str) or not s.startswith('Stage '):
        return None
    r = s[6:].rstrip('ABC')
    return r if r in ('0', 'I', 'II', 'III', 'IV') else None


STAGE_ORD = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}


def stage_group(roman):
    """I / II / III-IV（本队列 IV 期只有个位数，单独成组无法估计）。"""
    if roman in ('I', 'II'):
        return roman
    if roman in ('III', 'IV'):
        return 'III-IV'
    return None


def load_cdr():
    """可选的 TCGA-CDR 表（bcr_patient_barcode, OS, OS.time, PFI, PFI.time, DSS, DSS.time）。"""
    if not CDR_TABLE.exists():
        return None
    cdr = pd.read_csv(CDR_TABLE)
    cdr = cdr.rename(columns={'bcr_patient_barcode': 'case_id'}).set_index('case_id')
    return cdr


def clinical_table(source='auto'):
    """每个患者一行：label, tss, age, sex, stage, stage_group, stage_ord, os_days, os_event, os_source, os_missing。

    source='gdc'  只用 clinical.json；'cdr' 只用 TCGA-CDR；'auto' 有 CDR 表就用 CDR，否则 GDC。
    TCGA-CDR 是 TCGA 官方整理过的终点表，生存分析首选（还带 PFI/DSS）。
    """
    ds = load_dataset().drop_duplicates('case_id').set_index('case_id')
    cdr = load_cdr() if source in ('auto', 'cdr') else None
    if source == 'cdr' and cdr is None:
        raise SystemExit(f'找不到 {CDR_TABLE}')
    rows = []
    for c in load_clinical():
        cid = c['submitter_id']
        if cid not in ds.index:
            continue
        dem = c.get('demographic') or {}
        dx = primary_diagnosis(c)
        roman = stage_roman(dx.get('ajcc_pathologic_stage'))
        row = {'case_id': cid, 'label': ds.loc[cid, 'label'], 'tss': tss_of(cid),
               'age': _num(dem.get('age_at_index')),
               'sex': dem.get('sex_at_birth') or dem.get('gender'),
               'stage': dx.get('ajcc_pathologic_stage'), 'stage_roman': roman,
               'stage_group': stage_group(roman), 'stage_ord': STAGE_ORD.get(roman)}
        if cdr is not None and cid in cdr.index:
            r = cdr.loc[cid]
            ok = pd.notna(r.get('OS.time')) and pd.notna(r.get('OS'))
            row.update(os_days=float(r['OS.time']) if ok else None, os_event=int(r['OS']) if ok else None,
                       os_source='TCGA-CDR', os_missing=None if ok else 'CDR 无 OS')
            for ep in ('PFI', 'DSS'):
                if ep in r and pd.notna(r.get(ep)) and pd.notna(r.get(f'{ep}.time')):
                    row[f'{ep.lower()}_days'] = float(r[f'{ep}.time'])
                    row[f'{ep.lower()}_event'] = int(r[ep])
        else:
            t, e, why = gdc_overall_survival(c)
            row.update(os_days=t, os_event=e, os_source='GDC', os_missing=why)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('case_id')
    df['os_years'] = df['os_days'] / 365.25
    return df


def missingness_by_group(df, group='label'):
    """按组统计生存终点缺失：组内人数、缺失人数、缺失比例、缺失原因。"""
    out = []
    for g, sub in df.groupby(group):
        miss = sub['os_days'].isna()
        out.append({group: g, 'n': len(sub), 'missing': int(miss.sum()),
                    'missing_frac': round(miss.mean(), 3),
                    'reasons': '; '.join(f'{k}={v}' for k, v in sub.loc[miss, 'os_missing'].value_counts().items())})
    return pd.DataFrame(out)


def survival_usable(df, group='label', max_frac=0.2, max_gap=0.15):
    """缺失是否可以接受：任一组缺失超过 max_frac，或组间缺失比例相差超过 max_gap，就判为不可用。"""
    m = missingness_by_group(df, group)
    fr = m['missing_frac']
    ok = bool((fr <= max_frac).all() and (fr.max() - fr.min()) <= max_gap)
    return ok, m


# ---------------------------------------------------------------- 折与嵌入
def strict_fold_of_slide(eval_dir=STRICT_EVAL_DIR, k=5):
    """严格 5 折的测试折编号（从 eval 输出的 fold_k.csv 读，不依赖 CLAM 目录）。"""
    fold = {}
    for i in range(k):
        for s in pd.read_csv(Path(eval_dir) / f'fold_{i}.csv')['slide_id']:
            fold[s] = i
    return fold


def strict_fold_of_case(eval_dir=STRICT_EVAL_DIR, k=5):
    ds = load_dataset()
    fs = strict_fold_of_slide(eval_dir, k)
    out = {}
    for _, r in ds.iterrows():
        if r['slide_id'] in fs:
            f = fs[r['slide_id']]
            if out.get(r['case_id'], f) != f:
                raise ValueError(f"{r['case_id']} 的切片落在不同测试折")
            out[r['case_id']] = f
    return out


def load_patient_embedding(name='meanpool'):
    """患者级形态嵌入（embed_slides.py 的产物）。name='meanpool' 或 'clam_fold{k}'。"""
    path = EMB_DIR / f'{name}_patient.csv'
    if not path.exists():
        raise SystemExit(f'缺少 {path}：先跑 python scripts/embed_slides.py')
    X = pd.read_csv(path, index_col=0)
    keep = X.std(axis=0) > 1e-8                  # 去掉常数列（ReLU 后全零的"死维度"）
    return X.loc[:, keep]
