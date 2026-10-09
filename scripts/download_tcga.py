#!/usr/bin/env python3
"""Download TCGA-LUAD/LUSC diagnostic slides (+ clinical & RNA-seq metadata) from GDC.

Why the weird curl usage: this machine's OpenSSL-based git-bash curl gets its TLS
handshake killed by a middlebox, while Windows' schannel curl works with
--ssl-no-revoke. So all HTTP goes through C:/Windows/System32/curl.exe.

Usage:
  python download_tcga.py --query-only     # show counts/sizes, no download
  python download_tcga.py                  # full run (resumable via manifest)
  python download_tcga.py --n 30           # 30 slides per class instead of 50
  python download_tcga.py --metadata-only  # only refresh GDC metadata for the existing dataset csv:
                                           #   clinical.json (now with follow_ups) + rna_files.csv; no file downloads

2026-10 audit fixes:
  - clinical query now expands follow_ups. TCGA-LUAD keeps days_to_follow_up there; without it every
    living LUAD patient looked like "no follow-up" and the survival analysis kept only LUAD deaths.
  - RNA-seq query is restricted to sample_type = Primary Tumor, and file -> sample metadata
    (sample barcode, sample type) is saved to metadata/rna_files.csv. Before, 167 files for 144 cases
    (tumor + solid tissue normal + recurrence) were saved under the case id only and the analysis
    scripts could not tell which one was the tumor.
  - OUT can be overridden with the CLAM_TUTORIAL_DATA environment variable.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CURL = r"C:\Windows\System32\curl.exe"
# GDC 晚高峰对直连限速到 ~20KB/s；设 GDC_PROXY 环境变量走本地代理可满速（实测 6MB/s）
PROXY = os.environ.get("GDC_PROXY", "")
PROXY_ARGS = ["-x", PROXY] if PROXY else []
BASE = "https://api.gdc.cancer.gov"
OUT = os.environ.get("CLAM_TUTORIAL_DATA",
                     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
SLIDE_DIR = os.path.join(OUT, "slides", "tcga")
META_DIR = os.path.join(OUT, "metadata")
EXPR_DIR = os.path.join(META_DIR, "expression")
CSV_DIR = os.path.join(OUT, "dataset_csv")
MANIFEST = os.path.join(OUT, "download_manifest.json")
PROJECTS = ["TCGA-LUAD", "TCGA-LUSC"]
WORKERS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def curl_json(url, payload=None, timeout=300):
    cmd = [CURL, "--ssl-no-revoke", "-sS"] + PROXY_ARGS
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed ({r.returncode}): {r.stderr[:300]}")
    return json.loads(r.stdout)


def query_slides(project):
    payload = {
        "filters": {"op": "and", "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ]},
        "fields": "file_id,file_name,file_size,cases.submitter_id,cases.case_id",
        "format": "JSON", "size": "2000",
    }
    hits = curl_json(f"{BASE}/files", payload)["data"]["hits"]
    rows = []
    for h in hits:
        case = h["cases"][0]
        rows.append({
            "file_id": h["file_id"],
            "file_name": h["file_name"],
            "file_size": h["file_size"],
            "case_id": case["submitter_id"],     # patient id, e.g. TCGA-05-4244
            "case_uuid": case["case_id"],        # GDC internal uuid
            "label": project.replace("TCGA-", ""),
        })
    return rows


def fetch_clinical(uuids):
    """Clinical metadata for the given case uuids, including follow_ups (one bulk request)."""
    clin = curl_json(f"{BASE}/cases", {
        "filters": {"op": "in", "content": {"field": "case_id", "value": uuids}},
        "expand": "demographic,diagnoses,exposures,follow_ups",
        "format": "JSON", "size": "2000",
    })
    with open(os.path.join(META_DIR, "clinical.json"), "w") as f:
        json.dump(clin["data"]["hits"], f, indent=1)
    n_fu = sum(1 for c in clin["data"]["hits"] if c.get("follow_ups"))
    log(f"clinical.json: {len(clin['data']['hits'])} cases ({n_fu} with follow_ups)")


def query_rna_files(uuids, primary_only=True):
    """STAR-Counts files for the cases, with sample barcode and sample type."""
    content = [
        {"op": "in", "content": {"field": "cases.case_id", "value": uuids}},
        {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
    if primary_only:
        content.append({"op": "=", "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"}})
    hits = curl_json(f"{BASE}/files", {
        "filters": {"op": "and", "content": content},
        "fields": "file_id,file_name,file_size,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type",
        "format": "JSON", "size": "5000",
    })["data"]["hits"]
    rows = []
    for h in hits:
        case = h["cases"][0]
        sample = (case.get("samples") or [{}])[0]
        rows.append({"file_id": h["file_id"], "file_name": h["file_name"], "file_size": h["file_size"],
                     "case_id": case["submitter_id"], "sample_submitter_id": sample.get("submitter_id", ""),
                     "sample_type": sample.get("sample_type", "")})
    return rows


def write_rna_manifest(rows):
    path = os.path.join(META_DIR, "rna_files.csv")
    with open(path, "w") as f:
        f.write("file_id,file_name,case_id,sample_submitter_id,sample_type\n")
        for r in rows:
            f.write(f"{r['file_id']},{r['file_name']},{r['case_id']},{r['sample_submitter_id']},{r['sample_type']}\n")
    types = {}
    for r in rows:
        types[r["sample_type"]] = types.get(r["sample_type"], 0) + 1
    log(f"rna_files.csv: {len(rows)} files {types}")


def case_uuids_from_csv():
    """Case uuids for the cases already in the dataset csv (used by --metadata-only)."""
    csv_path = os.path.join(CSV_DIR, "tcga_luad_lusc.csv")
    cases = sorted({line.split(",")[0] for line in open(csv_path).read().splitlines()[1:] if line})
    hits = curl_json(f"{BASE}/cases", {
        "filters": {"op": "in", "content": {"field": "submitter_id", "value": cases}},
        "fields": "case_id,submitter_id", "format": "JSON", "size": "2000",
    })["data"]["hits"]
    log(f"dataset csv: {len(cases)} cases -> {len(hits)} GDC case uuids")
    return sorted(h["case_id"] for h in hits)


def download_file(file_id, dest, expected_size=None):
    """curl with resume; returns True on success."""
    if expected_size and os.path.exists(dest) and os.path.getsize(dest) == expected_size:
        return True
    cmd = [CURL, "--ssl-no-revoke", "-sS", "-L", "-C", "-"] + PROXY_ARGS + [
           "--retry", "15", "--retry-all-errors", "--retry-delay", "3",
           "-o", dest, f"{BASE}/data/{file_id}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        log(f"  FAIL {os.path.basename(dest)}: {r.stderr[:200]}")
        return False
    if expected_size and os.path.getsize(dest) != expected_size:
        log(f"  SIZE MISMATCH {os.path.basename(dest)}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="slides per class")
    ap.add_argument("--min-size", type=float, default=0.3, help="min slide size in GB")
    ap.add_argument("--query-only", action="store_true")
    ap.add_argument("--metadata-only", action="store_true",
                    help="refresh clinical.json (with follow_ups) and rna_files.csv for the existing dataset csv")
    args = ap.parse_args()

    os.makedirs(SLIDE_DIR, exist_ok=True)
    os.makedirs(EXPR_DIR, exist_ok=True)

    if args.metadata_only:
        uuids = case_uuids_from_csv()
        fetch_clinical(uuids)
        # all sample types, so files already on disk (incl. normals) can be labelled
        write_rna_manifest(query_rna_files(uuids, primary_only=False))
        log("metadata refreshed (no files downloaded)")
        return

    # 1. query slides
    all_rows = []
    for proj in PROJECTS:
        rows = query_slides(proj)
        total_gb = sum(r["file_size"] for r in rows) / 1e9
        log(f"{proj}: {len(rows)} open diagnostic slides, total {total_gb:.1f} GB")
        # skip degenerate tiny scans, then sample a representative subset
        rows = [r for r in rows if r["file_size"] >= args.min_size * 1e9]
        log(f"{proj}: {len(rows)} slides >= {args.min_size} GB")
        import random
        # 扩容只增不减：已在本地（已下载完成）的切片永远保留进数据集，
        # 不够的份额再从剩余候选里补抽（直接 Random(42).sample(n=75) 可能换掉老样本）
        on_disk = set(os.listdir(SLIDE_DIR)) if os.path.isdir(SLIDE_DIR) else set()
        kept = [r for r in rows if r["file_name"] in on_disk]
        rest = [r for r in rows if r["file_name"] not in on_disk]
        need = max(0, min(args.n, len(rows)) - len(kept))
        rows = kept + random.Random(42).sample(rest, min(need, len(rest)))
        log(f"{proj}: 保留本地 {len(kept)} 张 + 新抽 {len(rows) - len(kept)} 张 = {len(rows)}")
        all_rows += rows
    sel_gb = sum(r["file_size"] for r in all_rows) / 1e9
    log(f"selected {len(all_rows)} slides ({args.n}/class, random seed=42), {sel_gb:.1f} GB")
    if args.query_only:
        for r in all_rows[:5]:
            log(f"  e.g. {r['file_name']} {r['file_size']/1e9:.2f}GB {r['case_id']} {r['label']}")
        return

    # 2. download slides (3 parallel, resumable)
    done = set()
    if os.path.exists(MANIFEST):
        done = set(json.load(open(MANIFEST)).get("completed", []))
    todo = [r for r in all_rows if r["file_id"] not in done]
    log(f"downloading: {len(todo)} to go, {len(done)} already done")

    t0 = time.time()
    finished = [0]

    def work(row):
        dest = os.path.join(SLIDE_DIR, row["file_name"])
        ok = download_file(row["file_id"], dest, row["file_size"])
        if ok:
            done.add(row["file_id"])
            json.dump({"completed": sorted(done)}, open(MANIFEST, "w"))
            finished[0] += 1
            el = time.time() - t0
            log(f"  [{finished[0]}/{len(todo)}] {row['file_name']} "
                f"({row['file_size']/1e9:.2f}GB) ok | elapsed {el/60:.0f}min")
        return ok

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo))
    log(f"slides done: {len(done)}/{len(all_rows)}")

    # 3. dataset csv (CLAM format: case_id, slide_id, label)
    os.makedirs(CSV_DIR, exist_ok=True)
    csv_path = os.path.join(CSV_DIR, "tcga_luad_lusc.csv")
    with open(csv_path, "w") as f:
        f.write("case_id,slide_id,label\n")
        for r in all_rows:
            if r["file_id"] in done:
                sid = r["file_name"][:-4] if r["file_name"].endswith(".svs") else r["file_name"]
                f.write(f"{r['case_id']},{sid},{r['label']}\n")
    log(f"dataset csv -> {csv_path}")

    # 4. clinical metadata (one bulk request, with follow_ups)
    uuids = sorted({r["case_uuid"] for r in all_rows})
    fetch_clinical(uuids)

    # 5. RNA-seq STAR counts (primary tumor only) for the same cases + file -> sample manifest
    rows = query_rna_files(uuids, primary_only=True)
    write_rna_manifest(rows)
    ok_n = 0
    for r in rows:
        dest = os.path.join(EXPR_DIR, f"{r['case_id']}__{r['file_name']}")
        if download_file(r["file_id"], dest, r["file_size"]):
            ok_n += 1
    log(f"expression downloaded: {ok_n}/{len(rows)}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
