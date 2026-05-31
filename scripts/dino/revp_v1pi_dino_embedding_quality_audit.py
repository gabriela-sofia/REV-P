"""REV-P v1pi — DINO embedding quality audit.

Audits the v1ph feature store registry. One audit row per embedding with explicit
boolean checks (dim 768, no NaN/inf, norm > 0, patch_id present, region present,
not fixture, not duplicate) plus the always-false label/target/training guardrails.
Empty registry ⇒ empty audit with header (fail-closed).
"""

from __future__ import annotations

import argparse
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, SCHEMAS,
    _p, assert_no_forbidden_true, read_csv, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

IN_REGISTRY = _p("REVP_V1PI_IN_REGISTRY", DATASETS / "dino_embedding_feature_store_registry_v1ph.csv")
OUT_AUDIT = _p("REVP_V1PI_OUT_AUDIT", DATASETS / "dino_embedding_quality_audit_v1pi.csv")
OUT_SUMMARY = _p("REVP_V1PI_OUT_SUMMARY", DATASETS / "dino_embedding_quality_summary_v1pi.csv")
SCHEMA_AUDIT = _p("REVP_V1PI_SCHEMA_AUDIT", SCHEMAS / "dino_embedding_quality_audit_v1pi_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PI_SCHEMA_SUMMARY", SCHEMAS / "dino_embedding_quality_summary_v1pi_schema.csv")
DOC = _p("REVP_V1PI_DOC", DOCS / "revp_v1pi_dino_embedding_quality_audit.md")

AUDIT_FIELDS = [
    "audit_id", "embedding_id", "patch_id", "region",
    "check_dim_768", "check_no_nan", "check_no_inf", "check_norm_positive",
    "check_patch_id_present", "check_region_present", "check_not_fixture",
    "check_not_duplicate", "check_no_label", "check_no_target", "check_no_training",
    "overall_quality", "embedding_status", "blocked_reason", "notes",
]
SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _ok(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


def build_audit(registry: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(registry, 1):
        status = r.get("embedding_status", "")
        dim_ok = r.get("vector_dim") == str(EXPECTED_DINO_DIM)
        no_nan = r.get("has_nan", "false") != "true"
        no_inf = r.get("has_inf", "false") != "true"
        try:
            norm_pos = float(r.get("vector_norm_l2") or "0") > 0
        except ValueError:
            norm_pos = False
        patch_ok = bool(r.get("patch_id", "").strip()) and r.get("patch_id") != "UNKNOWN_PATCH"
        region_ok = bool(r.get("region", "").strip()) and r.get("region") != "UNKNOWN"
        not_fixture = r.get("dino_allowed_use") != "BLOCKED_FIXTURE_OR_SYNTHETIC"
        not_dup = r.get("is_duplicate_vector", "false") != "true"
        no_label = r.get("dino_can_create_label", "false") != "true"
        no_target = r.get("dino_target_field_created", "false") != "true"
        no_train = r.get("dino_can_train_model", "false") != "true"
        checks = [dim_ok, no_nan, no_inf, norm_pos, no_label, no_target, no_train]
        overall = "VALID_REVIEW_ONLY" if all(checks) and not_fixture else "BLOCKED"
        rows.append({
            "audit_id": f"V1PI_QA_{i:05d}",
            "embedding_id": r.get("embedding_id", ""),
            "patch_id": r.get("patch_id", ""),
            "region": r.get("region", ""),
            "check_dim_768": _ok(dim_ok),
            "check_no_nan": _ok(no_nan),
            "check_no_inf": _ok(no_inf),
            "check_norm_positive": _ok(norm_pos),
            "check_patch_id_present": _ok(patch_ok),
            "check_region_present": _ok(region_ok),
            "check_not_fixture": _ok(not_fixture),
            "check_not_duplicate": _ok(not_dup),
            "check_no_label": _ok(no_label),
            "check_no_target": _ok(no_target),
            "check_no_training": _ok(no_train),
            "overall_quality": overall,
            "embedding_status": status,
            "blocked_reason": r.get("blocked_reason", ""),
            "notes": "",
        })
    return rows


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    valid = sum(1 for r in rows if r["overall_quality"] == "VALID_REVIEW_ONLY")
    return [
        {"stat_key": "embeddings_audited", "stat_value": str(len(rows))},
        {"stat_key": "valid_review_only", "stat_value": str(valid)},
        {"stat_key": "blocked", "stat_value": str(len(rows) - valid)},
        {"stat_key": "label_violations", "stat_value": str(sum(1 for r in rows if r["check_no_label"] == "FAIL"))},
        {"stat_key": "training_violations", "stat_value": str(sum(1 for r in rows if r["check_no_training"] == "FAIL"))},
        {"stat_key": "audit_status",
         "stat_value": "QUALITY_AUDIT_PASS_REVIEW_ONLY" if valid else "NO_VALID_EMBEDDING_FAIL_CLOSED"},
    ]


def run() -> None:
    registry = read_csv(IN_REGISTRY)
    rows = build_audit(registry)
    require_no_abs_paths(rows, "v1pi_audit")
    assert_no_forbidden_true(rows, "v1pi_audit")
    summary = build_summary(rows)

    write_csv(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema(SCHEMA_AUDIT, AUDIT_FIELDS, "v1pi_dino_embedding_quality_audit")
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pi_dino_embedding_quality_summary")
    valid = sum(1 for r in rows if r["overall_quality"] == "VALID_REVIEW_ONLY")
    write_doc(DOC, "v1pi — DINO Embedding Quality Audit", [
        "## Objetivo",
        "Auditar o feature store v1ph com checks booleanos explícitos por embedding: "
        "dimensão 768, ausência de NaN/inf, norma positiva, patch_id e região "
        "presentes, não-fixture, não-duplicata, e ausência de label/target/treino.",
        "## Guardrails",
        "Os checks `check_no_label`, `check_no_target` e `check_no_training` confirmam "
        "que nenhum campo de label, target ou treino foi criado. Embeddings são "
        "representação visual review-only.",
        f"## Resultado",
        f"Embeddings auditados: {len(rows)}. Válidos review-only: {valid}.",
    ])
    print(f"[v1pi] audited={len(rows)} valid={valid}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pi dino embedding quality audit").parse_args()
    run()
