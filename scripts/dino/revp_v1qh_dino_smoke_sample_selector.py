"""REV-P v1qh — DINO smoke sample selector.

Selects a small stratified smoke sample (16-32 patches) from the v1qa expanded
visual queue for review-only DINO embeddings.

Does NOT require scene_date. Does NOT require temporal unlock. Blocks
fixture/test/synthetic rows. Never creates labels/targets/ground truth.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, env_int, env_str, guardrail_ok,
    is_fixture_or_synthetic, is_fixture_patch, normalize_identity, path_hash,
    read_queue_v1qa, require_no_abs_paths, write_csv, write_doc, write_schema,
)

OUT_SEL = _p("REVP_V1QH_OUT_SEL", DATASETS / "dino_smoke_sample_selection_v1qh.csv")
OUT_SUM = _p("REVP_V1QH_OUT_SUM", DATASETS / "dino_smoke_sample_summary_v1qh.csv")
SCH_SEL = _p("REVP_V1QH_SCH_SEL", SCHEMAS / "dino_smoke_sample_selection_v1qh_schema.csv")
SCH_SUM = _p("REVP_V1QH_SCH_SUM", SCHEMAS / "dino_smoke_sample_summary_v1qh_schema.csv")
DOC = _p("REVP_V1QH_DOC", DOCS / "revp_v1qh_dino_smoke_sample_selector.md")

SEL_FIELDS = [
    "smoke_id", "execution_queue_id", "visual_asset_id", "patch_id", "alias",
    "region", "relative_path", "path_hash", "visual_type", "sample_rank",
    "selection_reason", "linkage_confidence", "dino_allowed_use",
    "can_create_label", "can_train_model", "target_created", "selected_for_smoke",
    "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_CONF_RANK = {"HIGH": 0, "MEDIUM": 1, "MED": 1, "LOW": 2, "": 3}


def _region_token(region: str) -> str:
    """Short token used to match REVP_DINO_SMOKE_REGIONS values."""
    r = (region or "").upper()
    if r.startswith("REC"):
        return "REC"
    if r.startswith("PET"):
        return "PET"
    if r.startswith("CUR") or r == "CWB":
        return "CUR"
    return r[:3]


def select(queue: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    smoke_n = env_int("REVP_DINO_SMOKE_N", 32)
    min_per_region = env_int("REVP_DINO_SMOKE_MIN_PER_REGION", 4)
    regions_filter_raw = env_str("REVP_DINO_SMOKE_REGIONS", "")
    regions_filter = {r.strip().upper() for r in regions_filter_raw.split(",") if r.strip()}

    blocked_count = 0
    eligible: list[dict[str, Any]] = []
    for r in queue:
        patch, alias, region = normalize_identity(r)
        rel = r.get("relative_path", "")
        text = " ".join([patch, alias, rel, r.get("visual_type", ""), r.get("notes", "")])
        ok_guard, _ = guardrail_ok(r)
        if not ok_guard:
            blocked_count += 1
            continue
        if is_fixture_or_synthetic(text) or is_fixture_patch(patch):
            blocked_count += 1
            continue
        tok = _region_token(region)
        if regions_filter and tok not in regions_filter:
            continue
        conf = (r.get("linkage_confidence", "") or "").upper()
        try:
            prio = int(r.get("queue_priority", "9") or "9")
        except (TypeError, ValueError):
            prio = 9
        eligible.append({
            "row": r, "patch": patch, "alias": alias, "region": region,
            "region_tok": tok, "conf": conf,
            "conf_rank": _CONF_RANK.get(conf, 3), "prio": prio,
        })

    # Sort: higher confidence (lower rank), lower priority, region, patch_id.
    eligible.sort(key=lambda e: (e["conf_rank"], e["prio"], e["region_tok"], e["patch"]))

    # Round-robin across regions, one asset per patch_id before repeating.
    by_region: dict[str, list[dict[str, Any]]] = {}
    seen_patch: set[str] = set()
    for e in eligible:
        by_region.setdefault(e["region_tok"], []).append(e)

    selected: list[dict[str, Any]] = []
    # First pass: guarantee min_per_region where possible.
    for tok, items in by_region.items():
        taken = 0
        for e in items:
            if taken >= min_per_region or len(selected) >= smoke_n:
                break
            if e["patch"] in seen_patch:
                continue
            seen_patch.add(e["patch"])
            e["reason"] = "min_per_region_quota"
            selected.append(e)
            taken += 1
    # Second pass: fill remaining by global ranking, new patches first.
    for e in eligible:
        if len(selected) >= smoke_n:
            break
        if e in selected or e["patch"] in seen_patch:
            continue
        seen_patch.add(e["patch"])
        e["reason"] = "global_rank_diversity"
        selected.append(e)
    # Third pass: if still short, allow repeats of patch_id.
    for e in eligible:
        if len(selected) >= smoke_n:
            break
        if e in selected:
            continue
        e["reason"] = "fill_repeat_patch"
        selected.append(e)

    selected = selected[:smoke_n]

    out_rows: list[dict[str, Any]] = []
    region_counts: dict[str, int] = {}
    for i, e in enumerate(selected, 1):
        r = e["row"]
        region_counts[e["region_tok"]] = region_counts.get(e["region_tok"], 0) + 1
        rel = r.get("relative_path", "")
        out_rows.append({
            "smoke_id": f"V1QH_SMK_{i:05d}",
            "execution_queue_id": r.get("execution_queue_id", "") or r.get("source_queue_id", ""),
            "visual_asset_id": r.get("visual_asset_id", ""),
            "patch_id": e["patch"], "alias": e["alias"], "region": e["region"],
            "relative_path": rel,
            "path_hash": r.get("path_hash", "") or path_hash(rel),
            "visual_type": r.get("visual_type", ""),
            "sample_rank": str(i),
            "selection_reason": e.get("reason", "global_rank_diversity"),
            "linkage_confidence": e["conf"],
            "dino_allowed_use": r.get("dino_allowed_use", "REVIEW_ONLY_REPRESENTATION"),
            "can_create_label": "false", "can_train_model": "false", "target_created": "false",
            "selected_for_smoke": "true",
            "blocked_reason": "", "notes": "",
        })
    stats = {"eligible": len(eligible), "blocked": blocked_count, "selected": len(out_rows)}
    stats_regions = region_counts
    return out_rows, {**stats, **{f"region_{k}": v for k, v in stats_regions.items()}}


def run() -> None:
    queue = read_queue_v1qa()
    rows, stats = select(queue)
    require_no_abs_paths(rows, "v1qh_selection")
    assert_no_forbidden_true(rows, "v1qh_selection")

    status = "DINO_SMOKE_SAMPLE_READY" if rows else "DINO_SMOKE_SAMPLE_EMPTY_FAIL_CLOSED"
    summary = [
        {"stat_key": "queue_rows_read", "stat_value": str(len(queue))},
        {"stat_key": "eligible_rows", "stat_value": str(stats.get("eligible", 0))},
        {"stat_key": "blocked_fixture_or_guardrail", "stat_value": str(stats.get("blocked", 0))},
        {"stat_key": "selected_smoke_rows", "stat_value": str(stats.get("selected", 0))},
        {"stat_key": "smoke_n_target", "stat_value": str(env_int("REVP_DINO_SMOKE_N", 32))},
        {"stat_key": "min_per_region", "stat_value": str(env_int("REVP_DINO_SMOKE_MIN_PER_REGION", 4))},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "final_status", "stat_value": status},
    ]
    for k, v in stats.items():
        if k.startswith("region_"):
            summary.insert(-3, {"stat_key": f"selected_{k}", "stat_value": str(v)})
    require_no_abs_paths(summary, "v1qh_summary")
    assert_no_forbidden_true(summary, "v1qh_summary")
    write_csv(OUT_SEL, rows, SEL_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_SEL, SEL_FIELDS, "v1qh_smoke_sample_selection")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qh_smoke_sample_summary")
    write_doc(DOC, "v1qh — DINO Smoke Sample Selector", [
        "## Objetivo",
        "Selecionar amostra estratificada de 16-32 patches da fila v1qa para "
        "embeddings DINO review-only, priorizando confiança de linkage, prioridade "
        "menor e diversidade regional.",
        "## Não-requisitos",
        "Não exige scene_date. Não exige desbloqueio temporal. Bloqueia "
        "fixture/test/synthetic. Não cria rótulo, target ou ground truth.",
        "## Status",
        f"**{status}**. Selecionados: {len(rows)}.",
    ])
    print(f"[v1qh] status={status} selected={len(rows)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qh dino smoke sample selector").parse_args()
    run()
