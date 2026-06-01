"""REV-P v1qi — Local asset availability and preprocessing audit.

For each smoke sample row, resolves the local visual/TIF file using env roots
and records availability + minimal metadata. Pixels are read ONLY when
REVP_DINO_PIXEL_READ_ALLOWED=true. No image or pixel array is ever saved.
Absolute paths are never written.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, env_true, file_sha256_short, normalize_region,
    path_hash, read_csv, require_no_abs_paths, resolve_local_asset,
    write_csv, write_doc, write_schema,
)

IN_SEL = _p("REVP_V1QI_IN_SEL", DATASETS / "dino_smoke_sample_selection_v1qh.csv")
OUT_AUDIT = _p("REVP_V1QI_OUT_AUDIT", DATASETS / "dino_local_asset_preprocessing_audit_v1qi.csv")
OUT_SUM = _p("REVP_V1QI_OUT_SUM", DATASETS / "dino_local_asset_preprocessing_summary_v1qi.csv")
SCH_AUDIT = _p("REVP_V1QI_SCH_AUDIT", SCHEMAS / "dino_local_asset_preprocessing_audit_v1qi_schema.csv")
SCH_SUM = _p("REVP_V1QI_SCH_SUM", SCHEMAS / "dino_local_asset_preprocessing_summary_v1qi_schema.csv")
DOC = _p("REVP_V1QI_DOC", DOCS / "revp_v1qi_local_asset_preprocessing_audit.md")

AUDIT_FIELDS = [
    "asset_check_id", "smoke_id", "patch_id", "region", "relative_path",
    "path_hash", "local_path_hash", "file_exists", "file_sha256_short",
    "file_size_bytes", "ext", "pixel_read_allowed", "pixel_read_performed",
    "image_width", "image_height", "image_channels", "raster_bands", "dtype",
    "preprocessing_ready", "status", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_RASTER_EXT = {".tif", ".tiff"}


def _read_metadata(path: Path, ext: str) -> tuple[dict[str, Any], str]:
    """Read minimal metadata without persisting pixels. Returns (meta, note)."""
    meta: dict[str, Any] = {
        "image_width": "", "image_height": "", "image_channels": "",
        "raster_bands": "", "dtype": "",
    }
    note = ""
    if ext in _RASTER_EXT:
        try:
            import rasterio  # type: ignore
            with rasterio.open(path) as ds:
                meta["image_width"] = str(ds.width)
                meta["image_height"] = str(ds.height)
                meta["raster_bands"] = str(ds.count)
                meta["dtype"] = str(ds.dtypes[0]) if ds.dtypes else ""
            return meta, "rasterio"
        except Exception:
            note = "rasterio_unavailable_or_failed"
    # PIL fallback (PNG/JPG always; TIF when PIL supports it).
    try:
        from PIL import Image
        with Image.open(path) as img:
            meta["image_width"] = str(img.width)
            meta["image_height"] = str(img.height)
            bands = len(img.getbands()) if img.getbands() else ""
            meta["image_channels"] = str(bands)
            meta["dtype"] = str(img.mode)
        return meta, (note + ";pil" if note else "pil")
    except Exception:
        return meta, (note + ";pil_failed" if note else "pil_failed")


def audit(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pixel_allowed = env_true("REVP_DINO_PIXEL_READ_ALLOWED", False)
    out: list[dict[str, Any]] = []
    counts = {"resolved": 0, "missing": 0, "metadata_only": 0, "ready": 0, "unreadable": 0}

    for i, r in enumerate(rows, 1):
        rel = r.get("relative_path", "")
        smoke_id = r.get("smoke_id", "")
        patch = (r.get("patch_id", "") or "UNKNOWN_PATCH").upper()
        region = normalize_region(r.get("region", ""))
        resolved = resolve_local_asset(rel) if rel else None

        base: dict[str, Any] = {
            "asset_check_id": f"V1QI_AS_{i:05d}",
            "smoke_id": smoke_id, "patch_id": patch, "region": region,
            "relative_path": rel,
            "path_hash": r.get("path_hash", "") or (path_hash(rel) if rel else ""),
            "local_path_hash": "", "file_exists": "false", "file_sha256_short": "",
            "file_size_bytes": "", "ext": Path(rel).suffix.lower() if rel else "",
            "pixel_read_allowed": str(pixel_allowed).lower(),
            "pixel_read_performed": "false",
            "image_width": "", "image_height": "", "image_channels": "",
            "raster_bands": "", "dtype": "",
            "preprocessing_ready": "false",
            "status": "", "blocked_reason": "", "notes": "",
        }

        if resolved is None:
            counts["missing"] += 1
            base["status"] = "ASSET_MISSING_FAIL_CLOSED"
            base["blocked_reason"] = "local_file_not_resolved" if rel else "empty_relative_path"
            out.append(base)
            continue

        counts["resolved"] += 1
        ext = resolved.suffix.lower()
        base["ext"] = ext
        base["local_path_hash"] = path_hash(resolved.name)
        base["file_exists"] = "true"
        base["file_sha256_short"] = file_sha256_short(resolved)
        try:
            base["file_size_bytes"] = str(resolved.stat().st_size)
        except Exception:
            base["file_size_bytes"] = ""

        if not pixel_allowed:
            counts["metadata_only"] += 1
            base["status"] = "ASSET_METADATA_ONLY_PIXEL_READ_BLOCKED"
            base["blocked_reason"] = "pixel_read_not_permitted"
            base["notes"] = "REVP_DINO_PIXEL_READ_ALLOWED!=true"
            out.append(base)
            continue

        meta, note = _read_metadata(resolved, ext)
        base.update(meta)
        base["pixel_read_performed"] = "true"
        base["notes"] = note
        readable = bool(meta["image_width"]) and bool(meta["image_height"])
        if readable:
            counts["ready"] += 1
            base["preprocessing_ready"] = "true"
            base["status"] = "ASSET_READY_FOR_DINO_PREPROCESSING"
        else:
            counts["unreadable"] += 1
            base["status"] = "ASSET_UNREADABLE_FAIL_CLOSED"
            base["blocked_reason"] = "pixel_read_failed"
        out.append(base)

    return out, counts


def run() -> None:
    rows = read_csv(IN_SEL)
    audit_rows, counts = audit(rows)
    require_no_abs_paths(audit_rows, "v1qi_audit")
    assert_no_forbidden_true(audit_rows, "v1qi_audit")

    if not rows:
        final = "ASSET_MISSING_FAIL_CLOSED"
    elif counts["ready"] > 0:
        final = "ASSET_READY_FOR_DINO_PREPROCESSING"
    elif counts["resolved"] > 0 and not env_true("REVP_DINO_PIXEL_READ_ALLOWED", False):
        final = "ASSET_METADATA_ONLY_PIXEL_READ_BLOCKED"
    elif counts["resolved"] > 0:
        final = "ASSET_UNREADABLE_FAIL_CLOSED"
    else:
        final = "ASSET_MISSING_FAIL_CLOSED"

    summary = [
        {"stat_key": "smoke_rows", "stat_value": str(len(rows))},
        {"stat_key": "assets_resolved", "stat_value": str(counts["resolved"])},
        {"stat_key": "assets_missing", "stat_value": str(counts["missing"])},
        {"stat_key": "metadata_only", "stat_value": str(counts["metadata_only"])},
        {"stat_key": "preprocessing_ready", "stat_value": str(counts["ready"])},
        {"stat_key": "unreadable", "stat_value": str(counts["unreadable"])},
        {"stat_key": "pixel_read_allowed", "stat_value": str(env_true("REVP_DINO_PIXEL_READ_ALLOWED", False)).lower()},
        {"stat_key": "final_status", "stat_value": final},
    ]
    require_no_abs_paths(summary, "v1qi_summary")
    assert_no_forbidden_true(summary, "v1qi_summary")
    write_csv(OUT_AUDIT, audit_rows, AUDIT_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_AUDIT, AUDIT_FIELDS, "v1qi_local_asset_preprocessing_audit")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qi_local_asset_preprocessing_summary")
    write_doc(DOC, "v1qi — Local Asset Availability & Preprocessing Audit", [
        "## Objetivo",
        "Resolver arquivos visuais/TIF locais da amostra smoke usando env roots e "
        "registrar disponibilidade e metadados mínimos.",
        "## Leitura de pixels",
        "Pixels só são lidos se REVP_DINO_PIXEL_READ_ALLOWED=true. Nenhuma imagem ou "
        "array de pixels é salvo. Caminhos absolutos nunca são escritos.",
        "## Status",
        f"**{final}**. Resolvidos: {counts['resolved']}. Prontos: {counts['ready']}.",
    ])
    print(f"[v1qi] status={final} resolved={counts['resolved']} ready={counts['ready']}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qi local asset preprocessing audit").parse_args()
    run()
