"""REV-P v1qk — Import smoke embeddings into the representation layer.

Reads the v1qj feature store, validates all 768D vectors, deduplicates by
(patch_id, path_hash, model_path_hash), and writes a consolidated review-only
representation feature store. Never mixes with labels. Never calls C3/C4.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, SCHEMAS,
    _p, assert_no_forbidden_true, embedding_columns, normalize_region,
    parse_embedding_from_row, read_csv, require_no_abs_paths,
    validate_vector, vector_to_columns, write_csv, write_doc, write_schema,
)

IN_STORE = _p("REVP_V1QK_IN_STORE", DATASETS / "dino_smoke_embeddings_feature_store_v1qj.csv")
OUT_STORE = _p("REVP_V1QK_OUT_STORE", DATASETS / "dino_representation_feature_store_with_smoke_v1qk.csv")
OUT_SUM = _p("REVP_V1QK_OUT_SUM", DATASETS / "dino_representation_feature_store_with_smoke_summary_v1qk.csv")
SCH_STORE = _p("REVP_V1QK_SCH_STORE", SCHEMAS / "dino_representation_feature_store_with_smoke_v1qk_schema.csv")
SCH_SUM = _p("REVP_V1QK_SCH_SUM", SCHEMAS / "dino_representation_feature_store_with_smoke_summary_v1qk_schema.csv")
DOC = _p("REVP_V1QK_DOC", DOCS / "revp_v1qk_import_smoke_embeddings_to_representation_layer.md")

META_FIELDS = [
    "representation_id", "embedding_id", "patch_id", "alias", "region",
    "visual_asset_id", "source_stage", "model_name", "embedding_dim",
    "vector_valid", "duplicate_group_id", "dino_allowed_use", "review_only",
    "cluster_is_label", "can_create_label", "can_train_model", "target_created",
    "blocked_reason", "notes",
]
STORE_FIELDS = META_FIELDS + embedding_columns(EXPECTED_DINO_DIM)
SUM_FIELDS = ["stat_key", "stat_value"]


def consolidate(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    dup_groups: dict[str, str] = {}
    counts = {"input": len(rows), "valid": 0, "invalid": 0, "duplicates": 0}
    idx = 0
    for r in rows:
        vec = parse_embedding_from_row(r)
        vstatus, blocked = validate_vector(vec, EXPECTED_DINO_DIM)
        valid = vstatus == "VALID_REVIEW_ONLY"
        patch = (r.get("patch_id", "") or "UNKNOWN").upper()
        key = "|".join([patch, r.get("path_hash", ""), r.get("model_path_hash", "")])
        if key not in dup_groups:
            dup_groups[key] = f"V1QK_DG_{len(dup_groups)+1:05d}"
        dgid = dup_groups[key]
        is_dup = False
        if valid:
            # First occurrence of this key keeps; later ones flagged duplicate.
            already = any(o["duplicate_group_id"] == dgid and o["vector_valid"] == "true" for o in out)
            is_dup = already
        idx += 1
        meta = {
            "representation_id": f"V1QK_REP_{idx:05d}",
            "embedding_id": r.get("embedding_id", ""), "patch_id": patch,
            "alias": r.get("alias", "") or patch, "region": normalize_region(r.get("region", "")),
            "visual_asset_id": r.get("visual_asset_id", ""), "source_stage": "v1qj_smoke",
            "model_name": r.get("model_name", ""), "embedding_dim": str(len(vec) if vec else 0),
            "vector_valid": str(valid).lower(), "duplicate_group_id": dgid,
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION", "review_only": "true",
            "cluster_is_label": "false", "can_create_label": "false",
            "can_train_model": "false", "target_created": "false",
            "blocked_reason": "" if valid else blocked,
            "notes": "duplicate" if is_dup else "",
        }
        if valid:
            assert vec is not None  # validate_vector guarantees non-None here
            counts["valid"] += 1
            if is_dup:
                counts["duplicates"] += 1
            meta_row = dict(meta)
            meta_row.update(vector_to_columns(vec, EXPECTED_DINO_DIM))
            out.append(meta_row)
        else:
            counts["invalid"] += 1
            out.append(meta)
    return out, counts


def run() -> None:
    rows = read_csv(IN_STORE)
    out, counts = consolidate(rows)
    require_no_abs_paths(out, "v1qk_store")
    assert_no_forbidden_true(out, "v1qk_store")

    unique_valid = counts["valid"] - counts["duplicates"]
    if unique_valid > 0:
        final = "DINO_REPRESENTATION_WITH_SMOKE_READY_REVIEW_ONLY"
    else:
        final = "DINO_REPRESENTATION_WITH_SMOKE_EMPTY_FAIL_CLOSED"
    summary = [
        {"stat_key": "input_rows", "stat_value": str(counts["input"])},
        {"stat_key": "valid_vectors", "stat_value": str(counts["valid"])},
        {"stat_key": "invalid_vectors", "stat_value": str(counts["invalid"])},
        {"stat_key": "duplicate_vectors", "stat_value": str(counts["duplicates"])},
        {"stat_key": "unique_valid_vectors", "stat_value": str(unique_valid)},
        {"stat_key": "embedding_dim", "stat_value": str(EXPECTED_DINO_DIM)},
        {"stat_key": "cluster_is_label", "stat_value": "false"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "c3_c4_called", "stat_value": "false"},
        {"stat_key": "final_status", "stat_value": final},
    ]
    require_no_abs_paths(summary, "v1qk_summary")
    assert_no_forbidden_true(summary, "v1qk_summary")
    write_csv(OUT_STORE, out, STORE_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_STORE, STORE_FIELDS, "v1qk_representation_feature_store_with_smoke")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qk_representation_feature_store_with_smoke_summary")
    write_doc(DOC, "v1qk — Import Smoke Embeddings to Representation Layer", [
        "## Objetivo",
        "Importar embeddings smoke (v1qj) para um feature store de representação "
        "review-only consolidado, validando 768D e deduplicando por "
        "(patch_id, path_hash, model_path_hash).",
        "## Fronteira",
        "Não mistura com rótulos. Não chama C3/C4. cluster_is_label=false.",
        "## Status",
        f"**{final}**. Vetores válidos únicos: {unique_valid}.",
    ])
    print(f"[v1qk] status={final} unique_valid={unique_valid}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qk import smoke embeddings to representation layer").parse_args()
    run()
