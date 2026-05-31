"""REV-P v1pg — DINO artifact discovery (metadata-only).

Scans repository directories for DINO/embedding-related artifacts using term
matching on file names and CSV headers. Reads metadata only (name, size, header
terms) — never reads embedding pixels or raster data. Emits relative POSIX
paths and a path hash, never absolute/private paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    ARTIFACT_TERMS, DATASETS, DOCS, ROOT, SCHEMAS,
    _p, assert_no_forbidden_true, is_fixture_or_synthetic, normalize_region,
    path_hash, read_csv_header, require_no_abs_paths, sanitized_rel_path,
    write_csv, write_doc, write_schema,
)

OUT_DISCOVERY = _p("REVP_V1PG_OUT_DISCOVERY", DATASETS / "dino_artifact_discovery_v1pg.csv")
OUT_SUMMARY = _p("REVP_V1PG_OUT_SUMMARY", DATASETS / "dino_artifact_discovery_summary_v1pg.csv")
SCHEMA_DISCOVERY = _p("REVP_V1PG_SCHEMA_DISCOVERY", SCHEMAS / "dino_artifact_discovery_v1pg_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PG_SCHEMA_SUMMARY", SCHEMAS / "dino_artifact_discovery_summary_v1pg_schema.csv")
DOC = _p("REVP_V1PG_DOC", DOCS / "revp_v1pg_dino_artifact_discovery.md")

SCAN_DIRS = ("datasets", "docs/metodologia_cientifica", "scripts/dino")
OPTIONAL_LOCAL_DIR = "local_runs"

DISCOVERY_FIELDS = [
    "artifact_id", "relative_path", "path_hash", "artifact_type",
    "file_size_bytes", "detected_terms", "likely_embedding_source",
    "likely_visual_source", "likely_similarity_source", "likely_pca_source",
    "region_hint", "patch_hint", "is_fixture_or_synthetic",
    "allowed_for_dino_registry", "blocked_reason", "notes",
]
SUMMARY_FIELDS = ["stat_key", "stat_value"]

_EMB_TERMS = ("embedding", "768", "vector", "features")
_VIS_TERMS = ("dino", "patch", "visual", "sentinel")
_SIM_TERMS = ("similarity", "neighbor", "neighbors", "cosine")
_PCA_TERMS = ("pca", "cluster", "projection")
_REGION_TERMS = ("recife", "petropolis", "petrópolis", "curitiba", "pet", "cwb")


def _artifact_type(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return {
        "csv": "CSV", "json": "JSON", "md": "DOC", "py": "SCRIPT",
        "npy": "NPY_BLOCKED", "npz": "NPZ_BLOCKED",
    }.get(ext, ext.upper() or "UNKNOWN")


def _scan_file(path: Path, base: Path, source_tag: str, idx: int) -> dict[str, Any] | None:
    name = path.name.lower()
    rel = sanitized_rel_path(path, base)
    if path.suffix.lower() == ".csv":
        header = read_csv_header(path)
        header_lc = " ".join(header).lower()
    else:
        header_lc = ""
    haystack = f"{rel.lower()} {header_lc}"

    detected = sorted({t for t in ARTIFACT_TERMS if t in haystack})
    if not detected:
        return None

    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    fixture = is_fixture_or_synthetic(haystack)
    art_type = _artifact_type(name)
    blocked = ""
    allowed = "true"
    if art_type in ("NPY_BLOCKED", "NPZ_BLOCKED"):
        allowed, blocked = "false", "binary_embedding_blob_metadata_only"
    if fixture:
        allowed, blocked = "false", "fixture_or_synthetic"

    region_hint = ""
    for term in _REGION_TERMS:
        if term in haystack:
            region_hint = normalize_region(term)
            break

    # local_runs/ is private and never committed: mask the path, keep only a hash.
    is_local = source_tag == OPTIONAL_LOCAL_DIR or rel.startswith(OPTIONAL_LOCAL_DIR + "/")
    path_h = path_hash(rel)
    if is_local:
        rel = f"local_only:{path_h}"
        allowed, blocked = "false", "local_only_not_committed"
        region_hint = ""
        source_tag = "local_only"

    return {
        "artifact_id": f"V1PG_ART_{idx:04d}",
        "relative_path": rel,
        "path_hash": path_h,
        "artifact_type": art_type,
        "file_size_bytes": str(size),
        "detected_terms": "|".join(detected),
        "likely_embedding_source": str(any(t in haystack for t in _EMB_TERMS)).lower(),
        "likely_visual_source": str(any(t in haystack for t in _VIS_TERMS)).lower(),
        "likely_similarity_source": str(any(t in haystack for t in _SIM_TERMS)).lower(),
        "likely_pca_source": str(any(t in haystack for t in _PCA_TERMS)).lower(),
        "region_hint": region_hint,
        "patch_hint": str("patch_id" in haystack or "alias" in haystack).lower(),
        "is_fixture_or_synthetic": str(fixture).lower(),
        "allowed_for_dino_registry": allowed,
        "blocked_reason": blocked,
        "notes": f"source={source_tag}",
    }


def discover(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    seen: set[str] = set()
    dirs: list[str] = list(SCAN_DIRS)
    local = root / OPTIONAL_LOCAL_DIR
    if local.exists():
        dirs.append(OPTIONAL_LOCAL_DIR)
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = sanitized_rel_path(path, root)
            if rel in seen:
                continue
            seen.add(rel)
            idx += 1
            row = _scan_file(path, root, d, idx)
            if row is not None:
                rows.append(row)
    return rows


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    def c(field: str) -> int:
        return sum(1 for r in rows if r.get(field) == "true")
    return [
        {"stat_key": "artifacts_scanned", "stat_value": str(len(rows))},
        {"stat_key": "likely_embedding_artifacts", "stat_value": str(c("likely_embedding_source"))},
        {"stat_key": "likely_visual_artifacts", "stat_value": str(c("likely_visual_source"))},
        {"stat_key": "likely_similarity_artifacts", "stat_value": str(c("likely_similarity_source"))},
        {"stat_key": "likely_pca_artifacts", "stat_value": str(c("likely_pca_source"))},
        {"stat_key": "fixture_or_synthetic", "stat_value": str(c("is_fixture_or_synthetic"))},
        {"stat_key": "allowed_for_dino_registry", "stat_value": str(c("allowed_for_dino_registry"))},
    ]


def run() -> None:
    rows = discover(ROOT)
    require_no_abs_paths(rows, "v1pg_discovery")
    assert_no_forbidden_true(rows, "v1pg_discovery")
    summary = build_summary(rows)

    write_csv(OUT_DISCOVERY, rows, DISCOVERY_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema(SCHEMA_DISCOVERY, DISCOVERY_FIELDS, "v1pg_dino_artifact_discovery")
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pg_dino_artifact_discovery_summary")
    write_doc(DOC, "v1pg — DINO Artifact Discovery (metadata-only)", [
        "## Objetivo",
        "Escanear o repositório por artefatos relacionados a DINO/embeddings usando "
        "correspondência de termos em nomes de arquivo e cabeçalhos CSV. Leitura "
        "apenas de metadados — nunca de pixels ou vetores brutos.",
        "## Termos detectados",
        "`" + "`, `".join(ARTIFACT_TERMS) + "`.",
        "## Guardrails",
        "Saídas usam apenas caminhos relativos POSIX e hash de caminho. Blobs binários "
        "(.npy/.npz) são listados como metadados bloqueados. Fixtures/sintéticos são "
        "marcados e bloqueados. Nenhum label, target ou ground truth é criado.",
        f"## Resultado",
        f"Artefatos escaneados: {len(rows)}.",
    ])
    print(f"[v1pg] artifacts={len(rows)} "
          f"emb={sum(1 for r in rows if r['likely_embedding_source']=='true')}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pg dino artifact discovery").parse_args()
    run()
