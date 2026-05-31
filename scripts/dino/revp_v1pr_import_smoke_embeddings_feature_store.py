"""REV-P v1pr — Import smoke embeddings into feature store.

Reads v1pq results, validates vectors, builds a feature store compatible with
v1ph/v1pm. Does NOT overwrite v1ph. Output is a separate file.
Only real 768D VALID_REVIEW_ONLY embeddings are imported.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from revp_v1pn_v1pt_dino_execution_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, build_vector_row_fields, make_vector_row,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import (
    normalize_region, parse_embedding_from_text, read_csv,
)

OUT_STORE = _p("REVP_V1PR_OUT_STORE", DATASETS / "dino_smoke_embedding_feature_store_v1pr.csv")
OUT_SUM = _p("REVP_V1PR_OUT_SUM", DATASETS / "dino_smoke_embedding_feature_store_summary_v1pr.csv")
SCH_STORE = _p("REVP_V1PR_SCH_STORE", SCHEMAS / "dino_smoke_embedding_feature_store_v1pr_schema.csv")
SCH_SUM = _p("REVP_V1PR_SCH_SUM", SCHEMAS / "dino_smoke_embedding_feature_store_summary_v1pr_schema.csv")
DOC = _p("REVP_V1PR_DOC", DOCS / "revp_v1pr_import_smoke_embeddings_feature_store.md")
IN_RESULTS = _p("REVP_V1PR_IN_RESULTS", DATASETS / "dino_controlled_smoke_embedding_results_v1pq.csv")

STORE_FIELDS = build_vector_row_fields()
SUM_FIELDS = ["stat_key", "stat_value"]


def build_store() -> list[dict[str, Any]]:
    results = read_csv(IN_RESULTS)
    store: list[dict[str, Any]] = []
    for r in results:
        if r.get("status") != "EMBEDDING_EXECUTED_REVIEW_ONLY":
            continue
        emb_str = r.get("embedding", "")
        vec = parse_embedding_from_text(emb_str)
        if vec is None:
            continue
        idx = len(store) + 1
        row = make_vector_row(
            idx=idx,
            patch_id=r.get("patch_id", "UNKNOWN"),
            alias=r.get("alias", ""),
            region=normalize_region(r.get("region", "")),
            run_id=r.get("embedding_run_id", ""),
            asset_id=r.get("visual_asset_id", ""),
            vec=vec,
        )
        store.append(row)
    return store


def run() -> None:
    store = build_store()
    require_no_abs_paths(store, "v1pr_store")
    assert_no_forbidden_true(store, "v1pr_store")
    valid = sum(1 for r in store if r["embedding_status"] == "VALID_REVIEW_ONLY")
    summary = [
        {"stat_key": "embeddings_imported", "stat_value": str(len(store))},
        {"stat_key": "valid_768d", "stat_value": str(valid)},
        {"stat_key": "invalid_blocked", "stat_value": str(len(store) - valid)},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "import_status",
         "stat_value": "IMPORT_READY_REVIEW_ONLY" if valid else "IMPORT_EMPTY_NO_VALID_EMBEDDINGS"},
    ]
    write_csv(OUT_STORE, store, STORE_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_STORE, STORE_FIELDS, "v1pr_smoke_embedding_feature_store")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pr_smoke_embedding_feature_store_summary")
    write_doc(DOC, "v1pr — Smoke Embedding Feature Store", [
        "## Objetivo",
        "Importar embeddings válidos de v1pq para feature store incremental "
        "compatível com v1ph. Não sobrescreve v1ph.",
        "## Guardrails",
        "dino_can_create_label, dino_can_train_model e dino_target_field_created "
        "sempre false. Apenas vetores 768D válidos são importados.",
        f"## Resultado",
        f"Embeddings importados: {len(store)}. Válidos 768D: {valid}.",
    ])
    print(f"[v1pr] imported={len(store)} valid={valid}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pr import smoke embeddings").parse_args()
    run()
