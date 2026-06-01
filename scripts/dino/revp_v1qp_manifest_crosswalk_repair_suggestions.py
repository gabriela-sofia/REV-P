"""REV-P v1qp — Manifest crosswalk repair suggestions.

Compares v1fu, v1fm, v1qa, v1qh and v1qo to identify desalignments and
suggest repairs. Does NOT edit existing manifests. Read-only analysis.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qn_v1qt_local_readiness_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, normalize_patch, read_csv,
    read_smoke_sample, read_v1fm_designation, read_v1fu_manifest,
    read_v1qa_queue, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

IN_SMOKE = _p("REVP_V1QP_IN_SMOKE", DATASETS / "dino_smoke_sample_selection_v1qh.csv")
IN_REC   = _p("REVP_V1QP_IN_REC",   DATASETS / "dino_smoke_asset_local_reconciliation_v1qo.csv")
OUT_SUGG = _p("REVP_V1QP_OUT_SUGG", DATASETS / "dino_manifest_crosswalk_repair_suggestions_v1qp.csv")
OUT_SUM  = _p("REVP_V1QP_OUT_SUM",  DATASETS / "dino_manifest_crosswalk_repair_summary_v1qp.csv")
SCH_SUGG = _p("REVP_V1QP_SCH_SUGG", SCHEMAS / "dino_manifest_crosswalk_repair_suggestions_v1qp_schema.csv")
SCH_SUM  = _p("REVP_V1QP_SCH_SUM",  SCHEMAS / "dino_manifest_crosswalk_repair_summary_v1qp_schema.csv")
DOC      = _p("REVP_V1QP_DOC",       DOCS / "revp_v1qp_manifest_crosswalk_repair_suggestions.md")

SUGG_FIELDS = [
    "suggestion_id", "source_stage", "patch_id", "alias", "region",
    "issue_type", "description", "suggested_action", "priority",
    "review_only", "can_create_label", "can_train_model", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

ISSUE_TYPES = {
    "ADD_LOCAL_ROOT", "FIX_RELATIVE_PATH", "REVIEW_PATCH_ALIAS",
    "REVIEW_REGION_MISMATCH", "MISSING_LOCAL_FILE", "NO_ACTION_READY",
}


def build_suggestions(
    smoke: list[dict], qa: list[dict], fu: list[dict],
    fm: list[dict], rec: list[dict],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    idx = [0]

    def add(source: str, pid: str, alias: str, region: str,
            issue: str, desc: str, action: str, priority: str = "MEDIUM") -> None:
        idx[0] += 1
        suggestions.append({
            "suggestion_id": f"V1QP_SG_{idx[0]:05d}",
            "source_stage": source, "patch_id": pid, "alias": alias, "region": region,
            "issue_type": issue, "description": desc, "suggested_action": action,
            "priority": priority, "review_only": "true",
            "can_create_label": "false", "can_train_model": "false", "notes": "",
        })

    # Build lookup sets
    fm_patches = {(r.get("canonical_patch_id", "") or "").upper() for r in fm}
    fm_by_patch = {(r.get("canonical_patch_id", "") or "").upper(): r for r in fm}
    qa_patches = {(r.get("patch_id", "") or "").upper() for r in qa}
    rec_by_smoke = {r.get("smoke_id", ""): r for r in rec}

    # Check each smoke row
    for r in smoke:
        pid, alias, region = normalize_patch(r)
        smoke_id = r.get("smoke_id", "")
        rel = r.get("relative_path", "") or ""
        rec_row = rec_by_smoke.get(smoke_id, {})
        match_type = rec_row.get("match_type", "unresolved")
        ready = rec_row.get("ready_for_embedding", "false")

        # No local roots ⇒ cannot resolve
        if not rec_row:
            add("v1qh+v1qo", pid, alias, region, "MISSING_LOCAL_FILE",
                "smoke item not in reconciliation output", "run v1qo after configuring roots",
                "HIGH")
        elif match_type == "unresolved":
            if not rel:
                add("v1qh", pid, alias, region, "FIX_RELATIVE_PATH",
                    "relative_path empty in smoke queue",
                    "set REVP_SENTINEL_LOCAL_ROOT or add TIF path to v1fm/v1qa", "HIGH")
            else:
                add("v1qo", pid, alias, region, "MISSING_LOCAL_FILE",
                    f"no local file matched for {pid}",
                    "configure REVP_SENTINEL_LOCAL_ROOT with TIF storage location", "HIGH")
        elif ready == "false":
            add("v1qo", pid, alias, region, "ADD_LOCAL_ROOT",
                f"candidate found but low confidence ({match_type})",
                "verify root path, check filename convention", "MEDIUM")
        else:
            add("v1qo", pid, alias, region, "NO_ACTION_READY",
                f"asset resolved ({match_type})", "no repair needed", "LOW")

        # v1fm crosswalk
        if pid in fm_patches:
            fm_r = fm_by_patch[pid]
            fm_region = (fm_r.get("region", "") or "").strip().upper()
            if fm_region and region != "UNKNOWN" and not fm_region.startswith(region[:3]):
                add("v1fm", pid, alias, region, "REVIEW_REGION_MISMATCH",
                    f"region in smoke={region} vs v1fm={fm_region}",
                    "check canonical region mapping", "MEDIUM")
            tif_fn = fm_r.get("tif_filename", "") or ""
            if not tif_fn:
                add("v1fm", pid, alias, region, "FIX_RELATIVE_PATH",
                    "no tif_filename in v1fm designation table",
                    "update v1fm with actual TIF filename", "HIGH")
        elif pid not in fm_patches and fm:
            add("v1fm", pid, alias, region, "REVIEW_PATCH_ALIAS",
                f"{pid} not found in v1fm designation table",
                "add canonical_patch_id to v1fm or verify alias mapping", "MEDIUM")

    return suggestions


def run() -> None:
    smoke = read_smoke_sample(IN_SMOKE)
    qa    = read_v1qa_queue()
    fu    = read_v1fu_manifest()
    fm    = read_v1fm_designation()
    rec   = read_csv(IN_REC)

    sugg = build_suggestions(smoke, qa, fu, fm, rec)
    require_no_abs_paths(sugg, "v1qp_suggestions")
    assert_no_forbidden_true(sugg, "v1qp_suggestions")

    by_type: dict[str, int] = {}
    for s in sugg:
        by_type[s["issue_type"]] = by_type.get(s["issue_type"], 0) + 1

    summary = [{"stat_key": "total_suggestions", "stat_value": str(len(sugg))}]
    for it in sorted(ISSUE_TYPES):
        summary.append({"stat_key": f"type_{it.lower()}", "stat_value": str(by_type.get(it, 0))})
    summary += [
        {"stat_key": "labels_created",  "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "final_status",    "stat_value": "CROSSWALK_REPAIR_SUGGESTIONS_GENERATED"},
    ]
    require_no_abs_paths(summary, "v1qp_summary")
    assert_no_forbidden_true(summary, "v1qp_summary")

    write_csv(OUT_SUGG, sugg,    SUGG_FIELDS)
    write_csv(OUT_SUM,  summary, SUM_FIELDS)
    write_schema(SCH_SUGG, SUGG_FIELDS, "v1qp_manifest_crosswalk_repair_suggestions")
    write_schema(SCH_SUM,  SUM_FIELDS,  "v1qp_manifest_crosswalk_repair_summary")
    write_doc(DOC, "v1qp — Manifest Crosswalk Repair Suggestions", [
        "## Objetivo",
        "Comparar v1fu, v1fm, v1qa, v1qh e v1qo para identificar desalinhamentos "
        "e gerar sugestões de reparo. NÃO edita manifests antigos automaticamente.",
        "## Tipos de sugestão",
        "ADD_LOCAL_ROOT, FIX_RELATIVE_PATH, REVIEW_PATCH_ALIAS, "
        "REVIEW_REGION_MISMATCH, MISSING_LOCAL_FILE, NO_ACTION_READY.",
        "## Status",
        f"Total de sugestões: {len(sugg)}. Ready: {by_type.get('NO_ACTION_READY', 0)}. "
        f"Missing local file: {by_type.get('MISSING_LOCAL_FILE', 0)}.",
    ])
    print(f"[v1qp] suggestions={len(sugg)} ready={by_type.get('NO_ACTION_READY',0)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qp manifest crosswalk repair suggestions").parse_args()
    run()
