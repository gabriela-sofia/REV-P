"""SUSC-14B FASE 2 - acquire official spatial reference candidates."""

from __future__ import annotations

import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import read_csv, rel, sha256_file, write_csv, write_markdown  # noqa: E402

os.environ.setdefault(wd.NETWORK_ENV, "1")
wd.TIMEOUT = 8
wd.RETRIES = 1

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_registry_v1.csv"
OUTDIR = ROOT / "datasets" / "suscetibilidade" / "official_spatial_references_susc14b"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_download_manifest_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14B_official_spatial_reference_acquisition_report.md"

FIELDS = [
    "download_id", "reference_id", "region", "institution", "source_url", "source_title",
    "access_method", "download_status", "http_status", "content_type", "file_path",
    "file_size_bytes", "sha256", "content_ext", "blocked_reason", "requires_manual_review",
    "allowed_for_training", "can_be_ground_truth", "can_be_used_as_ground_truth", "review_only", "notes",
]
ALLOWED = {".csv", ".tsv", ".xlsx", ".xls", ".geojson", ".json", ".kml", ".kmz", ".zip", ".gpkg", ".gml", ".xml"}
RASTER = {".tif", ".tiff", ".jp2", ".img", ".nc", ".hdf", ".grib", ".grib2"}
MAX_BYTES = 250 * 1024 * 1024
MAX_DOWNLOAD_ATTEMPTS = 120


def _safe_name(text: str, suffix: str) -> str:
    import re
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80].strip("_") or "reference"
    return stem + suffix


def _ext(url: str, fmt: str) -> str:
    low = (url or "").lower().split("?")[0]
    for ext in sorted(ALLOWED | RASTER, key=len, reverse=True):
        if low.endswith(ext):
            return ext
    if fmt:
        f = "." + fmt.lower().lstrip(".")
        if f in ALLOWED:
            return f
    return ".json" if "query" in url or "f=pjson" in url else ""


def _base_row(idx, ref, status, reason):
    return {
        "download_id": f"S14BDL_{idx:04d}",
        "reference_id": ref.get("reference_id", ""),
        "region": ref.get("region", ""),
        "institution": ref.get("institution", ""),
        "source_url": ref.get("source_url", ""),
        "source_title": ref.get("source_title", ""),
        "access_method": ref.get("access_method", ""),
        "download_status": status,
        "http_status": "",
        "content_type": "",
        "file_path": "",
        "file_size_bytes": "",
        "sha256": "",
        "content_ext": "",
        "blocked_reason": reason,
        "requires_manual_review": "true",
        "allowed_for_training": "false",
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "review_only": "true",
        "notes": "review-only official reference acquisition",
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-14B Official Spatial Reference Acquisition")
    print("=" * 60)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / ".gitignore").write_text("*\n!.gitignore\n!README.md\n", encoding="utf-8")
    (OUTDIR / "README.md").write_text(
        "SUSC-14B local raw official spatial references. Raw downloads are ignored by Git; public commit keeps manifests only.\n",
        encoding="utf-8",
    )
    rows = []
    attempted = 0
    for idx, ref in enumerate(read_csv(REGISTRY) if REGISTRY.exists() else []):
        url = ref.get("source_url", "")
        if ref.get("access_method") == "local_reuse":
            row = _base_row(idx, ref, "reused_local_official", "")
            row["file_path"] = url
            p = ROOT / url
            if p.exists():
                row["file_size_bytes"] = str(p.stat().st_size)
                row["sha256"] = sha256_file(p)
                row["content_ext"] = p.suffix.lower()
            rows.append(row)
            continue
        ext = _ext(url, ref.get("format", ""))
        row = _base_row(idx, ref, "not_attempted", "")
        row["content_ext"] = ext
        if ext in RASTER:
            row["download_status"] = "blocked_raster"
            row["blocked_reason"] = "raster_or_image_blocked"
            rows.append(row)
            continue
        if ref.get("download_candidate") != "true" or not ext:
            row["download_status"] = "not_direct_data"
            row["blocked_reason"] = "html_or_service_root_without_direct_vector"
            rows.append(row)
            continue
        if attempted >= MAX_DOWNLOAD_ATTEMPTS:
            row["download_status"] = "not_attempted_batch_cap"
            row["blocked_reason"] = f"download_attempt_cap_{MAX_DOWNLOAD_ATTEMPTS}_reached"
            rows.append(row)
            continue
        attempted += 1
        fname = _safe_name(ref.get("reference_id", f"ref{idx}"), ext)
        p = OUTDIR / fname
        if p.exists() and p.stat().st_size > 0:
            row["download_status"] = "reused_cached_download"
            row["file_path"] = rel(p)
            row["file_size_bytes"] = str(p.stat().st_size)
            row["sha256"] = sha256_file(p)
            rows.append(row)
            continue
        rec = wd.http_get(url, max_bytes=MAX_BYTES)
        row["http_status"] = str(rec.get("http_code") or rec.get("status") or "")
        row["content_type"] = str(rec.get("content_type") or "")
        body = rec.get("bytes") or b""
        if not body:
            row["download_status"] = str(rec.get("status") or "failed_no_body")
            row["blocked_reason"] = str(rec.get("error") or "no_body")
            rows.append(row)
            continue
        p.write_bytes(body)
        row["download_status"] = "downloaded"
        row["file_path"] = rel(p)
        row["file_size_bytes"] = str(p.stat().st_size)
        row["sha256"] = sha256_file(p)
        if p.stat().st_size > 50 * 1024 * 1024:
            row["blocked_reason"] = "raw_gt_50mb_gitignored"
        rows.append(row)
    write_csv(MANIFEST, rows, FIELDS)
    status_counts = {}
    for row in rows:
        status_counts[row["download_status"]] = status_counts.get(row["download_status"], 0) + 1
    write_markdown(REPORT, f"""# SUSC-14B - aquisicao de referencias espaciais oficiais

Status: review-only. Brutos em `datasets/suscetibilidade/official_spatial_references_susc14b/`
ficam gitignored.

- Tentativas registradas: **{len(rows)}**
- Status: **{status_counts}**
- Downloads bem-sucedidos: **{status_counts.get('downloaded', 0)}**
- Reuso local oficial: **{status_counts.get('reused_local_official', 0)}**

Raster, imagem, Sentinel bruto, HTML sem dado direto e arquivos acima do limite
sao bloqueados ou mantidos apenas como registro de manifesto.
""")
    print(f"manifest -> {MANIFEST.relative_to(ROOT)} ({len(rows)} rows; downloaded {status_counts.get('downloaded', 0)})")
    print("review-only. No ground truth. No training. Raw files gitignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
