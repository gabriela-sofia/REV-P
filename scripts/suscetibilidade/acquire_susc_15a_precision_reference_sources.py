"""SUSC-15A FASE 4 - acquire precision reference source candidates."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import read_csv, rel, sha256_file, write_csv, write_markdown  # noqa: E402

os.environ.setdefault(wd.NETWORK_ENV, "1")
wd.TIMEOUT = 8
wd.RETRIES = 1

MAN = ROOT / "manifests" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
REGISTRY = MAN / "susc_15a_precision_reference_source_registry_v1.csv"
OUTDIR = DAT / "precision_references_susc15a"
MANIFEST = MAN / "susc_15a_precision_reference_download_manifest_v1.csv"
REPORT = OUT / "SUSC_15A_precision_reference_acquisition_report.md"

FIELDS = [
    "download_id", "source_id", "region", "institution", "url", "format", "download_status",
    "http_status", "content_type", "file_path", "file_size_bytes", "sha256", "blocked_reason",
    "requires_manual_review", "allowed_for_training", "can_be_ground_truth",
    "can_be_used_as_ground_truth", "review_only", "notes",
]
ALLOWED = {".csv", ".tsv", ".xlsx", ".xls", ".geojson", ".json", ".kml", ".kmz", ".zip", ".gpkg", ".gml", ".xml", ".wkt", ".pdf"}
BLOCKED = {".tif", ".tiff", ".jp2", ".img", ".nc", ".hdf", ".grib", ".grib2", ".exe", ".msi"}
MAX_NEW = 60
MAX_BYTES = 250 * 1024 * 1024


def _ext(url: str, fmt: str) -> str:
    low = (url or "").lower().split("?")[0]
    for ext in sorted(ALLOWED | BLOCKED, key=len, reverse=True):
        if low.endswith(ext):
            return ext
    fmt = "." + (fmt or "").lower().lstrip(".")
    return fmt if fmt in ALLOWED | BLOCKED else ".json"


def _safe(name: str, ext: str) -> str:
    import re
    return (re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80].strip("_") or "source") + ext


def _base(idx: int, row: dict, status: str, reason: str) -> dict:
    return {
        "download_id": f"S15ADL_{idx:04d}", "source_id": row.get("source_id", ""),
        "region": row.get("region", ""), "institution": row.get("institution", ""),
        "url": row.get("url", ""), "format": row.get("format", ""),
        "download_status": status, "http_status": "", "content_type": "",
        "file_path": "", "file_size_bytes": "", "sha256": "", "blocked_reason": reason,
        "requires_manual_review": "true", "allowed_for_training": "false",
        "can_be_ground_truth": "false", "can_be_used_as_ground_truth": "false",
        "review_only": "true", "notes": "SUSC-15A precision reference acquisition",
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Precision Reference Acquisition")
    print("=" * 60)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / ".gitignore").write_text("*\n!.gitignore\n!README.md\n", encoding="utf-8")
    (OUTDIR / "README.md").write_text("SUSC-15A raw precision references. Raw downloads are ignored by Git.\n", encoding="utf-8")
    rows, attempted = [], 0
    for idx, src in enumerate(read_csv(REGISTRY) if REGISTRY.exists() else []):
        row = _base(idx, src, "not_attempted", "")
        ext = _ext(src.get("url", ""), src.get("format", ""))
        if src.get("download_candidate") != "true":
            row["download_status"] = "not_direct_data"
            row["blocked_reason"] = "registry marks portal or non-direct source"
            rows.append(row)
            continue
        if ext in BLOCKED:
            row["download_status"] = "blocked_unsafe_or_raster"
            row["blocked_reason"] = f"blocked extension {ext}"
            rows.append(row)
            continue
        if attempted >= MAX_NEW:
            row["download_status"] = "not_attempted_batch_cap"
            row["blocked_reason"] = f"max_{MAX_NEW}_new_resources_reached"
            rows.append(row)
            continue
        path = OUTDIR / _safe(src.get("source_id", f"src{idx}"), ext)
        if path.exists() and path.stat().st_size > 0:
            row["download_status"] = "reused_cached_download"
            row["file_path"] = rel(path)
            row["file_size_bytes"] = str(path.stat().st_size)
            row["sha256"] = sha256_file(path)
            rows.append(row)
            attempted += 1
            continue
        attempted += 1
        rec = wd.http_get(src.get("url", ""), max_bytes=MAX_BYTES)
        row["http_status"] = str(rec.get("http_code") or rec.get("status") or "")
        row["content_type"] = str(rec.get("content_type") or "")
        body = rec.get("bytes") or b""
        if not body:
            row["download_status"] = str(rec.get("status") or "failed_no_body")
            row["blocked_reason"] = str(rec.get("error") or "no_body")
            rows.append(row)
            continue
        path.write_bytes(body)
        row["download_status"] = "downloaded"
        row["file_path"] = rel(path)
        row["file_size_bytes"] = str(path.stat().st_size)
        row["sha256"] = sha256_file(path)
        if path.stat().st_size > 50 * 1024 * 1024:
            row["blocked_reason"] = "raw_gt_50mb_gitignored"
        rows.append(row)
    write_csv(MANIFEST, rows, FIELDS)
    counts = Counter(r["download_status"] for r in rows)
    write_markdown(REPORT, f"""# SUSC-15A - aquisicao de referencias de precisao

Status: review-only. Brutos ficam em `datasets/suscetibilidade/precision_references_susc15a/`
e sao ignorados por Git.

- Linhas no manifesto: **{len(rows)}**
- Tentativas de download: **{attempted}**
- Status: **{dict(counts)}**

Rasters pesados, executaveis, HTML sem dado direto e arquivos acima do limite
permanecem bloqueados ou apenas manifestados.
""")
    print(f"manifest rows: {len(rows)} | status {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
