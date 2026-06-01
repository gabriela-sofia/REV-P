"""REV-P v1qo — Smoke asset local reconciliation.

Tries to match each smoke sample row (v1qh) to a real local file using
configured env roots. Produces ranked candidates. Never reads pixels.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qn_v1qt_local_readiness_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, candidate_roots, normalize_patch,
    ranked_candidates, read_smoke_sample, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

IN_SMOKE = _p("REVP_V1QO_IN_SMOKE", DATASETS / "dino_smoke_sample_selection_v1qh.csv")
OUT_REC  = _p("REVP_V1QO_OUT_REC",  DATASETS / "dino_smoke_asset_local_reconciliation_v1qo.csv")
OUT_CAND = _p("REVP_V1QO_OUT_CAND", DATASETS / "dino_smoke_asset_local_reconciliation_candidates_v1qo.csv")
OUT_SUM  = _p("REVP_V1QO_OUT_SUM",  DATASETS / "dino_smoke_asset_local_reconciliation_summary_v1qo.csv")
SCH_REC  = _p("REVP_V1QO_SCH_REC",  SCHEMAS / "dino_smoke_asset_local_reconciliation_v1qo_schema.csv")
SCH_CAND = _p("REVP_V1QO_SCH_CAND", SCHEMAS / "dino_smoke_asset_local_reconciliation_candidates_v1qo_schema.csv")
SCH_SUM  = _p("REVP_V1QO_SCH_SUM",  SCHEMAS / "dino_smoke_asset_local_reconciliation_summary_v1qo_schema.csv")
DOC      = _p("REVP_V1QO_DOC",       DOCS / "revp_v1qo_smoke_asset_local_reconciliation.md")

REC_FIELDS = [
    "reconciliation_id", "smoke_id", "patch_id", "alias", "region",
    "expected_relative_path", "expected_filename", "root_env_name",
    "local_candidate_filename", "local_candidate_relative", "local_path_hash",
    "match_type", "match_confidence", "file_exists", "file_size_bytes",
    "file_sha256_short", "ready_for_embedding", "review_only",
    "can_create_label", "can_train_model", "target_created",
    "blocked_reason", "notes",
]
CAND_FIELDS = [
    "candidate_id", "smoke_id", "patch_id", "rank",
    "root_env_name", "local_candidate_filename", "local_candidate_relative",
    "local_path_hash", "match_type", "match_confidence",
    "file_size_bytes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def reconcile(smoke: list[dict[str, str]],
              roots: list[tuple[str, Any]]) -> tuple[list[dict], list[dict], dict]:
    all_cands = ranked_candidates(smoke, roots)
    rec_rows: list[dict] = []
    cand_rows: list[dict] = []
    counts = {"total": len(smoke), "exact": 0, "partial": 0, "unresolved": 0}

    for i, r in enumerate(smoke, 1):
        pid, alias, region = normalize_patch(r)
        rel = r.get("relative_path", "") or ""
        smoke_id = r.get("smoke_id", "")
        candidates = all_cands.get(smoke_id, [])
        best = candidates[0] if candidates else None

        if best and best["match_confidence"] >= 0.9:
            mt = best["match_type"]; conf = best["match_confidence"]
            ready = "true"; blocked = ""
            if mt == "exact_relative":
                counts["exact"] += 1
            else:
                counts["partial"] += 1
        elif best and best["match_confidence"] >= 0.6:
            mt = best["match_type"]; conf = best["match_confidence"]
            ready = "false"; blocked = "low_confidence_match"
            counts["partial"] += 1
        else:
            mt = "unresolved"; conf = 0.0
            ready = "false"; blocked = "no_local_file_found"
            counts["unresolved"] += 1
            best = None

        rec_rows.append({
            "reconciliation_id": f"V1QO_REC_{i:05d}",
            "smoke_id": smoke_id, "patch_id": pid, "alias": alias, "region": region,
            "expected_relative_path": rel,
            "expected_filename": r.get("relative_path", "").split("/")[-1] if rel else "",
            "root_env_name": best["env_name"] if best else "",
            "local_candidate_filename": best["filename"] if best else "",
            "local_candidate_relative": best["relative_candidate"] if best else "",
            "local_path_hash": best["local_path_hash"] if best else "",
            "match_type": mt, "match_confidence": f"{conf:.3f}",
            "file_exists": str(best is not None and best["file_exists"]).lower(),
            "file_size_bytes": str(best["file_size_bytes"]) if best else "",
            "file_sha256_short": "",
            "ready_for_embedding": ready, "review_only": "true",
            "can_create_label": "false", "can_train_model": "false",
            "target_created": "false", "blocked_reason": blocked, "notes": "",
        })
        for j, cand in enumerate(candidates[:5], 1):
            cand_rows.append({
                "candidate_id": f"V1QO_CAND_{i:05d}_{j:02d}",
                "smoke_id": smoke_id, "patch_id": pid, "rank": str(j),
                "root_env_name": cand["env_name"],
                "local_candidate_filename": cand["filename"],
                "local_candidate_relative": cand["relative_candidate"],
                "local_path_hash": cand["local_path_hash"],
                "match_type": cand["match_type"],
                "match_confidence": f"{cand['match_confidence']:.3f}",
                "file_size_bytes": str(cand["file_size_bytes"]),
            })
    return rec_rows, cand_rows, counts


def run() -> None:
    smoke = read_smoke_sample(IN_SMOKE)
    roots = candidate_roots()
    rec, cand, counts = reconcile(smoke, roots)
    for rows, label in ((rec, "v1qo_rec"), (cand, "v1qo_cand")):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)

    reconciled = counts["exact"] + counts["partial"]
    if reconciled == counts["total"] and counts["total"] > 0:
        final = "SMOKE_ASSETS_RECONCILED_READY"
    elif reconciled > 0:
        final = "SMOKE_ASSETS_PARTIAL_RECONCILIATION"
    else:
        final = "SMOKE_ASSETS_UNRESOLVED_FAIL_CLOSED"

    summary = [
        {"stat_key": "smoke_rows",           "stat_value": str(counts["total"])},
        {"stat_key": "exact_matches",        "stat_value": str(counts["exact"])},
        {"stat_key": "partial_matches",      "stat_value": str(counts["partial"])},
        {"stat_key": "unresolved",           "stat_value": str(counts["unresolved"])},
        {"stat_key": "roots_available",      "stat_value": str(len(roots))},
        {"stat_key": "candidate_rows",       "stat_value": str(len(cand))},
        {"stat_key": "labels_created",       "stat_value": "0"},
        {"stat_key": "targets_created",      "stat_value": "0"},
        {"stat_key": "final_status",         "stat_value": final},
    ]
    require_no_abs_paths(summary, "v1qo_summary")
    assert_no_forbidden_true(summary, "v1qo_summary")
    write_csv(OUT_REC,  rec,     REC_FIELDS)
    write_csv(OUT_CAND, cand,    CAND_FIELDS)
    write_csv(OUT_SUM,  summary, SUM_FIELDS)
    write_schema(SCH_REC,  REC_FIELDS,  "v1qo_smoke_asset_local_reconciliation")
    write_schema(SCH_CAND, CAND_FIELDS, "v1qo_smoke_asset_local_reconciliation_candidates")
    write_schema(SCH_SUM,  SUM_FIELDS,  "v1qo_smoke_asset_local_reconciliation_summary")
    write_doc(DOC, "v1qo — Smoke Asset Local Reconciliation", [
        "## Objetivo",
        "Reconciliar cada item da amostra smoke (v1qh) com um arquivo local real "
        "usando env roots. Sem leitura de pixel.",
        "## Status",
        f"**{final}**. Resolvidos: {reconciled}/{counts['total']}. "
        f"Exatos: {counts['exact']}. Parciais: {counts['partial']}. "
        f"Não-resolvidos: {counts['unresolved']}.",
    ])
    print(f"[v1qo] {final} resolved={reconciled}/{counts['total']}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qo smoke asset local reconciliation").parse_args()
    run()
