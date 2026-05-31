"""REV-P v1ph — DINO embedding feature store registry.

Builds a registry of REAL embedding vectors discovered by v1pg. Never invents a
vector: if no real embedding is parsed, the registry CSV is written empty with
its header (fail-closed). DINO label/training/target flags are always false.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, SCHEMAS,
    _f, _p, assert_no_forbidden_true, is_fixture_or_synthetic, normalize_region,
    parse_embedding_from_row, path_hash, read_csv, require_no_abs_paths,
    sha256_short, source_root, validate_vector, vector_stats, write_csv, write_doc, write_schema,
)

IN_DISCOVERY = _p("REVP_V1PH_IN_DISCOVERY", DATASETS / "dino_artifact_discovery_v1pg.csv")
OUT_REGISTRY = _p("REVP_V1PH_OUT_REGISTRY", DATASETS / "dino_embedding_feature_store_registry_v1ph.csv")
OUT_SUMMARY = _p("REVP_V1PH_OUT_SUMMARY", DATASETS / "dino_embedding_feature_store_summary_v1ph.csv")
SCHEMA_REGISTRY = _p("REVP_V1PH_SCHEMA_REGISTRY", SCHEMAS / "dino_embedding_feature_store_registry_v1ph_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PH_SCHEMA_SUMMARY", SCHEMAS / "dino_embedding_feature_store_summary_v1ph_schema.csv")
DOC = _p("REVP_V1PH_DOC", DOCS / "revp_v1ph_dino_embedding_feature_store_registry.md")

REGISTRY_FIELDS = [
    "embedding_id", "patch_id", "alias", "region", "source_artifact_id",
    "source_path_hash", "vector_dim", "vector_sha256_16", "vector_norm_l2",
    "vector_mean", "vector_std", "has_nan", "has_inf", "is_zero_vector",
    "is_duplicate_vector", "embedding_status", "dino_allowed_use",
    "dino_can_create_label", "dino_can_train_model", "dino_target_field_created",
    "blocked_reason", "notes",
]
SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _patch_fields(row: dict[str, str]) -> tuple[str, str, str]:
    patch = (row.get("patch_id") or row.get("patch") or row.get("reference_patch_id") or "").strip()
    alias = (row.get("alias") or row.get("patch_alias") or "").strip()
    region = normalize_region(row.get("region") or row.get("region_hint") or "")
    return (patch or "UNKNOWN_PATCH", alias, region)


def _allowed_use(status: str, fixture: bool) -> str:
    if fixture:
        return "BLOCKED_FIXTURE_OR_SYNTHETIC"
    if status == "VALID_REVIEW_ONLY":
        return "REVIEW_ONLY_REPRESENTATION"
    if status == "BLOCKED_NO_EMBEDDING":
        return "BLOCKED_NO_EMBEDDING"
    return "BLOCKED_INVALID_VECTOR"


def build_registry(root: Path) -> list[dict[str, Any]]:
    discovery = read_csv(IN_DISCOVERY)
    rows: list[dict[str, Any]] = []
    seen_vectors: dict[str, str] = {}
    idx = 0

    # Only artifacts that look like embedding sources and are CSV/JSON we can parse.
    for art in discovery:
        if art.get("likely_embedding_source") != "true":
            continue
        if art.get("allowed_for_dino_registry") != "true":
            continue
        rel = art.get("relative_path", "")
        path = root / rel
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        source_rows = read_csv(path)
        for srow in source_rows:
            vec = parse_embedding_from_row(srow)
            if vec is None:
                # No real vector on this row — do not invent one; skip silently.
                continue
            idx += 1
            patch, alias, region = _patch_fields(srow)
            fixture = is_fixture_or_synthetic(rel + " " + " ".join(srow.values()))
            status, blocked = validate_vector(vec)
            st = vector_stats(vec)
            vsha = sha256_short(",".join(f"{x:.6g}" for x in vec))
            dup = vsha in seen_vectors
            if status == "VALID_REVIEW_ONLY":
                seen_vectors.setdefault(vsha, f"V1PH_EMB_{idx:05d}")
            if dup and not blocked:
                blocked = "duplicate_vector_review_flag"
            rows.append({
                "embedding_id": f"V1PH_EMB_{idx:05d}",
                "patch_id": patch, "alias": alias, "region": region,
                "source_artifact_id": art.get("artifact_id", ""),
                "source_path_hash": art.get("path_hash") or path_hash(rel),
                "vector_dim": str(st["dim"]),
                "vector_sha256_16": vsha,
                "vector_norm_l2": _f(st["norm"]),
                "vector_mean": _f(st["mean"]),
                "vector_std": _f(st["std"]),
                "has_nan": str(st["has_nan"]).lower(),
                "has_inf": str(st["has_inf"]).lower(),
                "is_zero_vector": str(st["is_zero"]).lower(),
                "is_duplicate_vector": str(dup).lower(),
                "embedding_status": status,
                "dino_allowed_use": _allowed_use(status, fixture),
                "dino_can_create_label": "false",
                "dino_can_train_model": "false",
                "dino_target_field_created": "false",
                "blocked_reason": "fixture_or_synthetic" if fixture else blocked,
                "notes": "",
            })
    return rows


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    valid = [r for r in rows if r["embedding_status"] == "VALID_REVIEW_ONLY"]
    return [
        {"stat_key": "embeddings_parsed", "stat_value": str(len(rows))},
        {"stat_key": "valid_768d_embeddings", "stat_value": str(len(valid))},
        {"stat_key": "invalid_embeddings", "stat_value": str(len(rows) - len(valid))},
        {"stat_key": "duplicate_vectors", "stat_value": str(sum(1 for r in rows if r["is_duplicate_vector"] == "true"))},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "training_targets_created", "stat_value": "0"},
        {"stat_key": "feature_store_status",
         "stat_value": "FEATURE_STORE_READY_REVIEW_ONLY" if valid else "NO_REAL_EMBEDDING_FAIL_CLOSED"},
    ]


def run() -> None:
    rows = build_registry(source_root())
    require_no_abs_paths(rows, "v1ph_registry")
    assert_no_forbidden_true(rows, "v1ph_registry")
    summary = build_summary(rows)

    write_csv(OUT_REGISTRY, rows, REGISTRY_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema(SCHEMA_REGISTRY, REGISTRY_FIELDS, "v1ph_dino_embedding_feature_store")
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ph_dino_embedding_feature_store_summary")
    valid = sum(1 for r in rows if r["embedding_status"] == "VALID_REVIEW_ONLY")
    write_doc(DOC, "v1ph — DINO Embedding Feature Store Registry", [
        "## Objetivo",
        "Registrar vetores de embedding REAIS descobertos em v1pg. Nunca inventa "
        "vetor: se nenhum embedding real for parseado, o registry é gravado vazio "
        "com header (fail-closed).",
        "## Regras de validação",
        "Dimensão 768 → `VALID_REVIEW_ONLY`. Dimensão diferente → "
        "`BLOCKED_INVALID_DIMENSION`. NaN/inf/zero → bloqueado. Duplicata → flag de "
        "revisão, nunca label.",
        "## Guardrails DINO",
        "`dino_can_create_label`, `dino_can_train_model` e `dino_target_field_created` "
        "são sempre false. Embeddings são representação visual auto-supervisionada, "
        "não rótulo.",
        f"## Resultado",
        f"Embeddings parseados: {len(rows)}. Válidos 768D: {valid}.",
    ])
    print(f"[v1ph] parsed={len(rows)} valid768={valid}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ph dino embedding feature store").parse_args()
    run()
