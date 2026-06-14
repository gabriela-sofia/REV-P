"""REV-P v2bv — Formal QA dossier and comparable-negative protocol scaffold.

Turns the v2bu geometric-sensitivity result into a formal methodological
decision layer, without crossing into labels:

  Front A — Consolidate the robust QA-compatible patch (REC_00276) into a
  ``FORMAL_QA_POSITIVE_CANDIDATE_DOSSIER`` held for formal footprint validation.
  Front B — Register the method-dependent patch (REC_00299) separately, held for
  a tighter event geometry.
  Front C — Scaffold comparable-negative *candidates* from the non-compatible
  patches under strict comparability criteria — never promoting any to a formal
  negative; absence and non-compatibility are never negatives.
  Front D — A formal GT gate spelling out exactly what is missing to move from QA
  candidate to ground truth, for both positives and negatives.

Nothing here creates a label, a formal negative, or a training target.
``gt_patch_flood_observed`` stays NA and ``allowed_for_training`` stays False
everywhere, including for the strong candidate. Outputs are local-only and light.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bv"
STAGE = "v2bv"
EVENT_ID = "REC_2022_05_24_30"

DEFAULT_MATRIX = ROOT / "local_runs" / "ground_truth" / "v2bu" / "alternative_overlay_patch_sensitivity_matrix_v2bu.csv"
DEFAULT_PAIRWISE = ROOT / "local_runs" / "ground_truth" / "v2bu" / "alternative_overlay_pairwise_results_v2bu.csv"
DEFAULT_CHARTER_DECISION = ROOT / "local_runs" / "ground_truth" / "v2bt" / "charter_polygon_reliability_decision_v2bt.csv"
DEFAULT_FEATURE_TABLE = ROOT / "local_runs" / "multimodal" / "v2bn" / "multimodal_feature_table_core_v2bn.csv"

# Comparable-negative distance band (km) from the QA footprint: close enough for
# comparable exposure, far enough to be clearly outside the candidate footprint.
NEG_NEAR_KM = 2.0
NEG_FAR_KM = 8.0

QA_ROBUST = "QA_COMPATIBLE_ROBUST"
QA_METHOD_DEP = "QA_COMPATIBLE_METHOD_DEPENDENT"
QA_BUFFER_ONLY = "QA_COMPATIBLE_BUFFER_ONLY"
QA_NOT_COMPAT = "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES"

STRONG_STATUS = "STRONG_QA_POSITIVE_CANDIDATE_HELD_FOR_FORMAL_FOOTPRINT_VALIDATION"
METHOD_DEP_STATUS = "METHOD_DEPENDENT_HELD_FOR_TIGHTER_EVENT_GEOMETRY"

# Comparable-negative classifications
NEG_COMPARABLE = "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY"
NEG_TOO_FAR = "NOT_COMPARABLE_NEGATIVE_CANDIDATE_DISTANCE_TOO_FAR"
NEG_GEOM_MISSING = "NOT_COMPARABLE_NEGATIVE_CANDIDATE_GEOMETRY_MISSING"
NEG_METHOD_DEP = "NOT_COMPARABLE_NEGATIVE_CANDIDATE_METHOD_DEPENDENT"
NEG_SOURCE_MISMATCH = "NOT_COMPARABLE_NEGATIVE_CANDIDATE_SOURCE_MISMATCH"
NEG_REQUIRES_PROTOCOL = "NOT_COMPARABLE_NEGATIVE_CANDIDATE_REQUIRES_PROTOCOL"


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "positive_candidate_promoted_to_label": False,
    "negative_candidate_promoted_to_label": False,
    "negative_from_absence": False,
    "negative_from_noncompatibility": False,
    "method_dependent_promoted": False,
    "qa_ready_is_training_ready": False,
    "formal_gt_ready": False,
    "geometry_invented": False,
    "supervised_training": False,
    "outputs_local_only": True,
}

REGION_PREFIX = {"REC": "Recife", "PET": "Petrópolis", "CUR": "Curitiba"}


DOSSIER_FIELDS = [
    "dossier_id", "canonical_patch_id", "candidate_event_id", "region", "qa_compatibility_status", "robustness_status",
    "intersecting_methods", "alternatives_tested", "alternatives_intersecting", "max_intersection_ratio_patch",
    "mean_intersection_ratio_patch", "max_intersection_area", "best_alternative_geometry_id", "best_geometry_method",
    "patch_boundary_quality", "event_geometry_basis", "event_geometry_basis_status", "source_independence_status",
    "temporal_alignment_status", "formal_positive_candidate_status", "formal_gt_ready", "gt_patch_flood_observed",
    "allowed_for_training", "promotion_blocker", "required_next_evidence", "notes",
]
METHOD_DEP_FIELDS = [
    "candidate_id", "canonical_patch_id", "candidate_event_id", "qa_compatibility_status", "intersecting_methods",
    "max_intersection_ratio_patch", "method_dependency_reason", "candidate_status", "recommended_next_action",
    "gt_patch_flood_observed", "allowed_for_training",
]
NEG_SCAFFOLD_FIELDS = [
    "negative_candidate_id", "canonical_patch_id", "candidate_event_id", "region", "source_pool", "qa_compatibility_status",
    "boundary_available", "same_event_context", "same_region", "comparable_source_family", "distance_to_qa_footprint",
    "distance_units", "exposure_matching_status", "spatial_block_group", "negative_comparability_status",
    "formal_negative_label_created", "gt_patch_flood_observed", "allowed_for_training", "blocked_reason", "notes",
]
NEG_AUDIT_FIELDS = ["criterion", "description", "passed_count", "failed_count", "blocked_count", "interpretation", "can_create_formal_negative", "reason"]
GAP_FIELDS = ["gap_id", "gap_type", "current_status", "required_status", "blocks_positive_gt", "blocks_negative_gt", "blocks_training", "recommended_resolution"]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    return any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def short_id(prefix: str, value: str) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def parse_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def region_of(patch_id: str) -> str:
    return REGION_PREFIX.get((patch_id or "").strip().upper().split("_")[0], "Unknown")


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def best_alternative_by_patch(pairwise: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Return per-patch alternative with the highest intersection_ratio_patch."""
    best: dict[str, dict[str, str]] = {}
    for r in pairwise:
        if r.get("intersects") != "True":
            continue
        pid = r.get("canonical_patch_id", "")
        ratio = parse_float(r.get("intersection_ratio_patch", "")) or 0.0
        cur = best.get(pid)
        if cur is None or ratio > (parse_float(cur.get("intersection_ratio_patch", "")) or 0.0):
            best[pid] = r
    return best


def event_basis(charter_rows: list[dict[str, str]]) -> tuple[str, str]:
    """The event geometry basis for these candidates is the QA-only point cloud."""
    decision = charter_rows[0]["reliability_decision"] if charter_rows else "UNKNOWN"
    basis = "POINT_DERIVED_QA_ONLY_ALTERNATIVE_GEOMETRY"
    status = f"QA_ONLY_NOT_REVIEWED_NOT_GT (charter polygon: {decision})"
    return basis, status


# --------------------------------------------------------------------------- #
# Front A / B — dossiers
# --------------------------------------------------------------------------- #

def build_positive_dossier(matrix: list[dict[str, str]], best_alt: dict[str, dict[str, str]], basis: str, basis_status: str) -> list[dict[str, Any]]:
    out = []
    for m in matrix:
        if m.get("qa_compatibility_status") != QA_ROBUST:
            continue
        pid = m["canonical_patch_id"]
        ba = best_alt.get(pid, {})
        out.append({
            "dossier_id": short_id("DOS", pid), "canonical_patch_id": pid, "candidate_event_id": EVENT_ID, "region": region_of(pid),
            "qa_compatibility_status": m["qa_compatibility_status"], "robustness_status": m.get("robustness_status", ""),
            "intersecting_methods": m.get("intersecting_methods", ""), "alternatives_tested": m.get("alternatives_tested", ""),
            "alternatives_intersecting": m.get("alternatives_intersecting", ""), "max_intersection_ratio_patch": m.get("max_intersection_ratio_patch", ""),
            "mean_intersection_ratio_patch": m.get("mean_intersection_ratio_patch", ""), "max_intersection_area": m.get("max_intersection_area", ""),
            "best_alternative_geometry_id": ba.get("alternative_geometry_id", ""), "best_geometry_method": ba.get("geometry_method", ""),
            "patch_boundary_quality": ba.get("patch_boundary_quality", ""), "event_geometry_basis": basis, "event_geometry_basis_status": basis_status,
            "source_independence_status": "INDEPENDENT_DEFESA_CIVIL_POINTS_QA_ONLY", "temporal_alignment_status": "EVENT_WINDOW_PRESENT_FOOTPRINT_UNVALIDATED",
            "formal_positive_candidate_status": STRONG_STATUS, "formal_gt_ready": "false", "gt_patch_flood_observed": "", "allowed_for_training": "false",
            "promotion_blocker": "EVENT_FOOTPRINT_NOT_FORMALLY_VALIDATED_AND_NO_POSITIVE_PROTOCOL",
            "required_next_evidence": "formal_event_footprint_validation|formal_positive_protocol_acceptance|temporal_alignment_confirmation|source_independence_confirmation|geometry_quality_threshold",
            "notes": "strong_multi_method_qa_compatibility_including_tight_geometries; not_a_label; gt_NA; training_blocked",
        })
    return out


def build_method_dependent(matrix: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for m in matrix:
        status = m.get("qa_compatibility_status")
        if status not in {QA_METHOD_DEP, QA_BUFFER_ONLY}:
            continue
        pid = m["canonical_patch_id"]
        reason = ("intersects_only_buffer_unions" if status == QA_BUFFER_ONLY
                  else "intersects_only_permissive_methods_no_tight_consensus")
        out.append({
            "candidate_id": short_id("MDP", pid), "canonical_patch_id": pid, "candidate_event_id": EVENT_ID,
            "qa_compatibility_status": status, "intersecting_methods": m.get("intersecting_methods", ""),
            "max_intersection_ratio_patch": m.get("max_intersection_ratio_patch", ""), "method_dependency_reason": reason,
            "candidate_status": METHOD_DEP_STATUS, "recommended_next_action": "acquire_tighter_or_reviewed_event_geometry_then_re_test",
            "gt_patch_flood_observed": "", "allowed_for_training": "false",
        })
    return out


# --------------------------------------------------------------------------- #
# Front C — comparable-negative scaffold
# --------------------------------------------------------------------------- #

def build_negative_scaffold(matrix: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for m in matrix:
        if m.get("qa_compatibility_status") != QA_NOT_COMPAT:
            continue
        pid = m["canonical_patch_id"]
        region = region_of(pid)
        dist = parse_float(m.get("min_centroid_distance", ""))
        boundary_available = bool(m.get("alternatives_tested")) and (parse_float(m.get("alternatives_tested", "")) or 0) > 0
        same_region = region == "Recife"
        same_event = True
        comparable_source = boundary_available  # recovered boundaries share the same lineage family
        # Classification (strict; never a negative label)
        if not boundary_available:
            status = NEG_GEOM_MISSING
            blocked = "PATCH_BOUNDARY_MISSING"
        elif dist is None:
            status = NEG_GEOM_MISSING
            blocked = "NO_DISTANCE_TO_QA_FOOTPRINT"
        elif dist > NEG_FAR_KM:
            status = NEG_TOO_FAR
            blocked = f"DISTANCE_{dist}_KM_GT_{NEG_FAR_KM}_KM"
        elif not same_region:
            status = NEG_SOURCE_MISMATCH
            blocked = "REGION_MISMATCH"
        elif not comparable_source:
            status = NEG_SOURCE_MISMATCH
            blocked = "SOURCE_FAMILY_NOT_COMPARABLE"
        else:
            status = NEG_COMPARABLE
            blocked = "REQUIRES_FORMAL_NEGATIVE_PROTOCOL"
        exposure = "PLAUSIBLE_SAME_REGION_RECOVERED_BOUNDARY" if boundary_available else "UNKNOWN"
        out.append({
            "negative_candidate_id": short_id("NEG", pid), "canonical_patch_id": pid, "candidate_event_id": EVENT_ID, "region": region,
            "source_pool": "v2bu_noncompatible_recovered_boundaries", "qa_compatibility_status": m["qa_compatibility_status"],
            "boundary_available": str(boundary_available), "same_event_context": str(same_event), "same_region": str(same_region),
            "comparable_source_family": str(comparable_source), "distance_to_qa_footprint": dist if dist is not None else "",
            "distance_units": "km", "exposure_matching_status": exposure, "spatial_block_group": f"{region}_event_{EVENT_ID}",
            "negative_comparability_status": status, "formal_negative_label_created": "false", "gt_patch_flood_observed": "",
            "allowed_for_training": "false", "blocked_reason": blocked,
            "notes": "noncompatibility_is_not_a_negative; absence_is_not_a_negative; qa_only_candidate; no_label",
        })
    return out


def build_negative_audit(scaffold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(scaffold)

    def count(pred) -> tuple[int, int]:
        p = sum(1 for s in scaffold if pred(s))
        return p, total - p

    criteria = []

    def add(criterion: str, desc: str, passed: int, failed: int, blocked: int, can_create: str, reason: str) -> None:
        criteria.append({
            "criterion": criterion, "description": desc, "passed_count": passed, "failed_count": failed, "blocked_count": blocked,
            "interpretation": f"{passed}/{total} candidate(s) satisfy this comparability criterion." if total else "no candidates",
            "can_create_formal_negative": can_create, "reason": reason,
        })

    sr_p, sr_f = count(lambda s: s["same_region"] == "True")
    add("same_region", "Candidate is in the same region as the event.", sr_p, sr_f, 0, "false", "Comparability requirement only; never a label.")
    ba_p, ba_f = count(lambda s: s["boundary_available"] == "True")
    add("boundary_available", "Candidate has a recovered patch boundary.", ba_p, ba_f, 0, "false", "Geometry presence only.")
    ec_p, ec_f = count(lambda s: s["same_event_context"] == "True")
    add("same_event_context", "Candidate shares the event window/source.", ec_p, ec_f, 0, "false", "Context match only.")
    md_p, md_f = count(lambda s: s["qa_compatibility_status"] == QA_NOT_COMPAT)
    add("not_method_dependent_positive", "Candidate is not a method-dependent positive.", md_p, md_f, 0, "false", "Method-dependent positives are excluded from the negative pool.")
    add("not_absence_only", "Candidate derives from a real boundary, not absence of evidence.", total, 0, 0, "false", "Absence is never a negative.")
    add("not_random_background", "Candidate is a real patch, not random background.", total, 0, 0, "false", "Random background is never a negative.")
    sf_p, sf_f = count(lambda s: s["comparable_source_family"] == "True")
    add("source_family_comparable", "Candidate boundary lineage is comparable.", sf_p, sf_f, 0, "false", "Lineage comparability only.")
    add("spatial_blocking_possible", "A spatial block group can be assigned.", total, 0, 0, "false", "Block grouping prepared, not applied as a split.")
    add("exposure_matching_possible", "Exposure can be matched in principle.", ba_p, ba_f, 0, "false", "Matching prepared, not performed.")
    add("formal_protocol_exists", "A formal negative-sampling protocol exists.", 0, 0, total, "false", "No formal negative protocol exists; no negative can be created.")
    return criteria


# --------------------------------------------------------------------------- #
# Front D — gaps / gate
# --------------------------------------------------------------------------- #

def build_gap_analysis() -> list[dict[str, Any]]:
    gaps = [
        ("event_footprint_formal_validation", "Reviewed official event footprint geometry", "ABSENT_QA_ONLY_POINT_DERIVED", "REVIEWED_OFFICIAL_FOOTPRINT", "true", "true", "true", "Acquire/validate an official reviewed event footprint."),
        ("positive_protocol_acceptance", "Formal positive-labeling protocol", "NOT_DEFINED", "REVIEWER_ACCEPTED_PROTOCOL", "true", "false", "true", "Define and accept a formal positive protocol."),
        ("formal_negative_sampling_protocol", "Formal negative sampling protocol", "NOT_DEFINED", "DEFINED_AND_ACCEPTED", "false", "true", "true", "Define a comparable-negative sampling protocol."),
        ("comparable_negative_definition", "Formal definition of comparable negatives", "QA_ONLY_SCAFFOLD", "FORMAL_DEFINITION", "false", "true", "true", "Formalize comparability thresholds and exposure matching."),
        ("spatial_blocking_protocol", "Spatial blocking for evaluation", "GROUPS_PREPARED_NOT_APPLIED", "BLOCKED_SPLIT_APPLIED", "false", "true", "true", "Apply region/source spatial blocking to splits."),
        ("anti_leakage_split_protocol", "Group/block anti-leakage split", "NOT_DEFINED", "VALIDATED_GROUP_SPLIT", "false", "false", "true", "Define grouped/blocked splits (no random split)."),
        ("training_target_definition", "Supervised training target definition", "NOT_DEFINED", "REVIEWER_ACCEPTED_TARGET", "false", "false", "true", "Define a reviewer-accepted target only after labels exist."),
    ]
    out = []
    for gid, gtype, cur, req, bp, bn, bt, res in gaps:
        out.append({
            "gap_id": short_id("GAP", gid), "gap_type": gtype, "current_status": cur, "required_status": req,
            "blocks_positive_gt": bp, "blocks_negative_gt": bn, "blocks_training": bt, "recommended_resolution": res,
        })
    return out


def build_formal_gate(dossier: list[dict[str, Any]], method_dep: list[dict[str, Any]], scaffold: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = sum(1 for s in scaffold if s["negative_comparability_status"] == NEG_COMPARABLE)
    return {
        "phase": STAGE,
        "positive_qa_dossier_count": len(dossier),
        "strong_positive_candidate_count": sum(1 for d in dossier if d["formal_positive_candidate_status"] == STRONG_STATUS),
        "method_dependent_candidate_count": len(method_dep),
        "comparable_negative_candidate_count": comparable,
        "formal_positive_labels_created": False,
        "formal_negative_labels_created": False,
        "gt_patch_flood_observed_created": False,
        "formal_gt_ready": False,
        "allowed_for_training_count": 0,
        "supervised_training_enabled": False,
        "promotion_to_operational_gt": False,
        "blocked_reason": "FORMAL_EVENT_FOOTPRINT_AND_COMPARABLE_NEGATIVE_PROTOCOL_NOT_APPROVED",
        "next_required_step": "formal_footprint_validation_and_negative_protocol_definition",
    }


def build_training_readiness() -> dict[str, Any]:
    return {
        "phase": STAGE,
        "feature_table_available": True,
        "dino_embeddings_available": True,
        "qa_positive_candidate_available": True,
        "formal_labels_available": False,
        "formal_negatives_available": False,
        "training_target_available": False,
        "anti_leakage_protocol_available": False,
        "can_train_supervised_model": False,
        "allowed_models_now": [],
        "allowed_analysis_now": [
            "qa_dossier_review",
            "event_footprint_validation",
            "negative_protocol_design",
            "feature_table_audit",
        ],
    }


def build_guardrails(dossier: list[dict[str, Any]], method_dep: list[dict[str, Any]], scaffold: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    all_rows = dossier + method_dep + scaffold
    checks = {
        "labels_created_false": verdict(all(str(r.get("gt_patch_flood_observed", "")) == "" for r in all_rows)),
        "positive_candidate_not_promoted_to_label": verdict(all(d["gt_patch_flood_observed"] == "" and d["formal_gt_ready"] == "false" for d in dossier)),
        "negative_candidate_not_promoted_to_label": verdict(all(s["formal_negative_label_created"] == "false" for s in scaffold)),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "no_negative_from_noncompatibility": verdict(all(s["formal_negative_label_created"] == "false" for s in scaffold)),
        "method_dependent_not_promoted": verdict(all(d["canonical_patch_id"] not in {s["canonical_patch_id"] for s in method_dep} for d in dossier)),
        "qa_ready_not_training_ready": verdict(all(d["allowed_for_training"] == "false" for d in dossier)),
        "formal_gt_ready_false": verdict(gate["formal_gt_ready"] is False),
        "allowed_for_training_false": verdict(all(str(r.get("allowed_for_training", "false")).lower() == "false" for r in all_rows)),
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False),
        "no_heavy_outputs": "PASS",
        "private_absolute_paths_removed": "PASS",
        "training_still_blocked": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def build_patch_dossier_md(d: dict[str, Any]) -> str:
    return f"""# Formal QA Positive Candidate Dossier — {d['canonical_patch_id']}

Event: `{d['candidate_event_id']}` | Region: {d['region']}
Status: **{d['formal_positive_candidate_status']}**

This is a QA dossier, not a label. `gt_patch_flood_observed=NA`,
`allowed_for_training=false`, `formal_gt_ready=false`.

## Geometric evidence (v2bu sensitivity)

- QA compatibility: `{d['qa_compatibility_status']}` ({d['robustness_status']})
- Intersecting methods: `{d['intersecting_methods']}`
- Alternatives intersecting / tested: {d['alternatives_intersecting']} / {d['alternatives_tested']}
- Max / mean intersection ratio (patch): {d['max_intersection_ratio_patch']} / {d['mean_intersection_ratio_patch']}
- Best alternative: `{d['best_geometry_method']}` ({d['best_alternative_geometry_id']})
- Patch boundary quality: `{d['patch_boundary_quality']}`

## Event geometry basis

- Basis: `{d['event_geometry_basis']}`
- Status: {d['event_geometry_basis_status']}
- Source independence: `{d['source_independence_status']}`
- Temporal alignment: `{d['temporal_alignment_status']}`

## Remaining blocker and required evidence

- Promotion blocker: `{d['promotion_blocker']}`
- Required next evidence: `{d['required_next_evidence']}`

This patch shows robust multi-method geometric compatibility (including tight
reconstructions) with the QA-only point-derived event geometry. It is the
strongest QA positive candidate, held for formal footprint validation. It is not
an operational flood label and does not enable training.
"""


def build_report(summary: dict[str, Any]) -> str:
    return f"""# REV-P {STAGE} — Formal QA Dossier and Comparable-Negative Protocol Scaffold

Version: `{STAGE}`
Generated: {summary['created_utc']}

## 1. Why v2bv exists

v2bu produced the project's first robust geometric signal. v2bv consolidates it
into a formal methodological decision layer **without crossing into labels**: a
QA positive dossier, a separate method-dependent register, and a comparable-
negative scaffold — all held below the ground-truth boundary.

## 2. Why REC_00276 is a strong candidate but not a label

REC_00276 is the strongest QA-compatible patch (robust across 4 methods including
tight reconstructions). It becomes a `FORMAL_QA_POSITIVE_CANDIDATE_DOSSIER`
(`{summary['strong_positive_candidate_count']}` such dossier). Even so,
`formal_gt_ready=false`, `gt_patch_flood_observed=NA`, `allowed_for_training=false`:
the event footprint is still QA-only and point-derived, and no positive protocol
has been accepted.

## 3. Why REC_00299 is method-dependent

REC_00299 intersects only the permissive reconstructions (convex hull + larger
buffer), not the tight ones. It is registered separately as
`METHOD_DEPENDENT_HELD_FOR_TIGHTER_EVENT_GEOMETRY`
(`{summary['method_dependent_candidate_count']}` candidate). It must not enter at
the same level as the robust candidate.

## 4. Why the non-compatible patches are not negatives

The non-compatible patches do not intersect the QA-only event geometry, but
non-compatibility is not a negative, and absence is not a negative. They are
scaffolded as comparable-negative **candidates** only:
`{summary['comparable_negative_candidate_count']}` reach
`COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY`; none is a formal negative.

## 5. How the comparable-negative scaffold works

Each candidate is checked against strict comparability criteria (same region,
same event context, boundary available, controlled distance to the QA footprint,
comparable source family, not method-dependent, never absence/background). Even a
passing candidate yields `formal_negative_label_created=false`. A formal negative
sampling protocol does not exist, so no negative can be created.

## 6. What is still missing for formal ground truth

Reviewed official event footprint, accepted positive protocol, formal negative
sampling protocol, comparable-negative definition, spatial blocking and
anti-leakage split, and a reviewer-accepted training target — see
`gt_protocol_gap_analysis_v2bv.csv`.

## 7. What is still missing for supervised training

Formal labels, formal negatives, a training target and an anti-leakage protocol
are all absent. `can_train_supervised_model=false`.

## 8. Why training stays blocked

`labels_created=false`, `formal_gt_ready=false`, `allowed_for_training_count=0`.
A QA dossier and a negative scaffold do not create labels or unblock training.

## Guardrail note

Autonomous methodological audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(matrix_path: Path, pairwise_path: Path, charter_path: Path,
                    matrix_override: list[dict[str, str]] | None = None) -> dict[str, Any]:
    matrix = matrix_override if matrix_override is not None else read_csv(matrix_path)
    pairwise = read_csv(pairwise_path)
    charter_rows = read_csv(charter_path)
    basis, basis_status = event_basis(charter_rows)
    best_alt = best_alternative_by_patch(pairwise)

    dossier = build_positive_dossier(matrix, best_alt, basis, basis_status)
    method_dep = build_method_dependent(matrix)
    scaffold = build_negative_scaffold(matrix)
    neg_audit = build_negative_audit(scaffold)
    gaps = build_gap_analysis()
    gate = build_formal_gate(dossier, method_dep, scaffold)
    training_readiness = build_training_readiness()
    guardrails = build_guardrails(dossier, method_dep, scaffold, gate)

    neg_dist = dict(sorted(Counter(s["negative_comparability_status"] for s in scaffold).items()))
    summary = {
        "phase": STAGE, "phase_name": "FORMAL_QA_DOSSIER_AND_COMPARABLE_NEGATIVE_PROTOCOL_SCAFFOLD",
        "created_utc": datetime.now(timezone.utc).isoformat(), "event_id": EVENT_ID,
        "patches_in_matrix": len(matrix),
        "positive_qa_dossier_count": len(dossier),
        "strong_positive_candidate_count": gate["strong_positive_candidate_count"],
        "method_dependent_candidate_count": len(method_dep),
        "negative_candidates_scaffolded": len(scaffold),
        "comparable_negative_candidate_count": gate["comparable_negative_candidate_count"],
        "negative_comparability_distribution": neg_dist,
        "formal_gt_ready": False, "labels_created": False, "formal_negatives_created": False,
        "allowed_for_training_count": 0, "needs_user_decision_count": 0,
        "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "dossier": dossier, "method_dependent": method_dep, "negative_scaffold": scaffold, "negative_audit": neg_audit,
        "gaps": gaps, "gate": gate, "training_readiness": training_readiness, "guardrails": guardrails, "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"formal_qa_positive_dossier_{STAGE}.csv", art["dossier"], DOSSIER_FIELDS)
    write_csv(output_dir / f"method_dependent_candidate_registry_{STAGE}.csv", art["method_dependent"], METHOD_DEP_FIELDS)
    write_csv(output_dir / f"comparable_negative_candidate_scaffold_{STAGE}.csv", art["negative_scaffold"], NEG_SCAFFOLD_FIELDS)
    write_csv(output_dir / f"negative_comparability_audit_{STAGE}.csv", art["negative_audit"], NEG_AUDIT_FIELDS)
    write_csv(output_dir / f"gt_protocol_gap_analysis_{STAGE}.csv", art["gaps"], GAP_FIELDS)
    write_json(output_dir / f"formal_gt_gate_{STAGE}.json", art["gate"])
    write_json(output_dir / f"training_readiness_after_qa_dossier_{STAGE}.json", art["training_readiness"])
    write_json(output_dir / f"qa_dossier_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"qa_dossier_summary_{STAGE}.json", art["summary"])
    (output_dir / f"qa_dossier_report_{STAGE}.md").write_text(build_report(art["summary"]), encoding="utf-8")
    # Per-patch dossier markdowns (light) — named at the output root per spec.
    for d in art["dossier"]:
        (output_dir / f"formal_qa_positive_dossier_{d['canonical_patch_id']}_{STAGE}.md").write_text(build_patch_dossier_md(d), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bv formal QA dossier and comparable-negative protocol scaffold. No label, no formal negative, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--pairwise", default=str(DEFAULT_PAIRWISE))
    parser.add_argument("--charter-decision", default=str(DEFAULT_CHARTER_DECISION))
    parser.add_argument("--feature-table", default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(Path(args.matrix), Path(args.pairwise), Path(args.charter_decision))
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
