"""SUSC-14C FASE 2 - acquire additional official references in small batches."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import read_csv, rel, sha256_file, write_csv, write_markdown  # noqa: E402

os.environ.setdefault(wd.NETWORK_ENV, "1")
wd.TIMEOUT = 8
wd.RETRIES = 1

DOWNLOADS_14B = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_download_manifest_v1.csv"
REGISTRY_14B = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_registry_v1.csv"
OUTDIR = ROOT / "datasets" / "suscetibilidade" / "official_spatial_references_susc14c"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14c_additional_reference_download_manifest_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_additional_reference_acquisition_report.md"

FIELDS = [
    "download_id", "reference_id", "region", "institution", "source_url", "source_title",
    "priority_bucket", "download_status", "http_status", "content_type", "file_path",
    "file_size_bytes", "sha256", "content_ext", "blocked_reason", "requires_manual_review",
    "allowed_for_training", "can_be_ground_truth", "can_be_used_as_ground_truth", "review_only", "notes",
]
ALLOWED = {".csv", ".tsv", ".xlsx", ".xls", ".geojson", ".json", ".kml", ".kmz", ".zip", ".gpkg", ".gml", ".xml"}
RASTER = {".tif", ".tiff", ".jp2", ".img", ".nc", ".hdf", ".grib", ".grib2"}
MAX_BYTES = 250 * 1024 * 1024
MAX_NEW = 80


def _ext(url: str, fmt: str) -> str:
    low = (url or "").lower().split("?")[0]
    for ext in sorted(ALLOWED | RASTER, key=len, reverse=True):
        if low.endswith(ext):
            return ext
    f = "." + (fmt or "").lower().lstrip(".")
    return f if f in ALLOWED else ""


def _safe(text: str, ext: str) -> str:
    import re
    return (re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80].strip("_") or "reference") + ext


def _bucket(ref: dict) -> int:
    text = f"{ref.get('region','')} {ref.get('source_title','')} {ref.get('source_url','')}".lower()
    if "recife" in text and any(k in text for k in ("logradouro", "eixo", "viario", "rua", "endereco")):
        return 1
    if "recife" in text and any(k in text for k in ("alagamento", "inundacao", "ponto critico")):
        return 2
    if "recife" in text and any(k in text for k in ("bairro", "bairros")):
        return 3
    if "curitiba" in text and any(k in text for k in ("alagamento", "logradouro", "rua")):
        return 4
    if "petropolis" in text and any(k in text for k in ("atingid", "logradouro", "setor")):
        return 5
    if any(k in text for k in ("drenagem", "hidrografia", "rio", "canal")):
        return 6
    return 9


def _row(idx: int, ref: dict, status: str, reason: str) -> dict:
    return {
        "download_id": f"S14CDL_{idx:04d}",
        "reference_id": ref.get("reference_id", ""),
        "region": ref.get("region", ""),
        "institution": ref.get("institution", ""),
        "source_url": ref.get("source_url", ""),
        "source_title": ref.get("source_title", ""),
        "priority_bucket": _bucket(ref),
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
        "notes": "SUSC-14C additional official reference batch",
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Additional Official Reference Acquisition")
    print("=" * 60)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / ".gitignore").write_text("*\n!.gitignore\n!README.md\n", encoding="utf-8")
    (OUTDIR / "README.md").write_text("SUSC-14C raw additional official references. Raw files are ignored by Git.\n", encoding="utf-8")
    registry = {r["reference_id"]: r for r in read_csv(REGISTRY_14B)}
    pending = []
    for row in read_csv(DOWNLOADS_14B):
        if row.get("download_status") != "not_attempted_batch_cap":
            continue
        ref = registry.get(row.get("reference_id"), {})
        if ref:
            pending.append(ref)
    pending.sort(key=lambda r: (_bucket(r), r.get("region", ""), r.get("priority", "9"), r.get("source_title", "")))
    rows = []
    attempted = 0
    for idx, ref in enumerate(pending):
        ext = _ext(ref.get("source_url", ""), ref.get("format", ""))
        row = _row(idx, ref, "not_attempted", "")
        row["content_ext"] = ext
        if attempted >= MAX_NEW:
            row["download_status"] = "not_attempted_batch_cap"
            row["blocked_reason"] = f"max_{MAX_NEW}_new_resources_reached"
            rows.append(row)
            continue
        if ext in RASTER:
            row["download_status"] = "blocked_raster"
            row["blocked_reason"] = "raster_or_image_blocked"
            rows.append(row)
            continue
        if not ext:
            row["download_status"] = "not_direct_data"
            row["blocked_reason"] = "no_direct_vector_or_table_extension"
            rows.append(row)
            continue
        p = OUTDIR / _safe(ref.get("reference_id", f"ref{idx}"), ext)
        if p.exists() and p.stat().st_size > 0:
            row["download_status"] = "reused_cached_download"
            row["file_path"] = rel(p)
            row["file_size_bytes"] = str(p.stat().st_size)
            row["sha256"] = sha256_file(p)
            rows.append(row)
            attempted += 1
            continue
        attempted += 1
        rec = wd.http_get(ref.get("source_url", ""), max_bytes=MAX_BYTES)
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
    from collections import Counter
    status = Counter(r["download_status"] for r in rows)
    write_markdown(REPORT, f"""# SUSC-14C - aquisicao adicional por lotes

Status: review-only. Brutos em `datasets/suscetibilidade/official_spatial_references_susc14c/`
ficam gitignored.

- Recursos 14B pendentes por teto: **{len(pending)}**
- Novos recursos tentados nesta sprint: **{min(attempted, MAX_NEW)}**
- Status: **{dict(status)}**

Raster, imagem, Sentinel bruto e arquivos acima do limite permanecem bloqueados
ou apenas manifestados. Nenhum dado bruto pesado foi preparado para commit.
""")
    print(f"manifest rows: {len(rows)} | attempted {min(attempted, MAX_NEW)} | status {dict(status)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
