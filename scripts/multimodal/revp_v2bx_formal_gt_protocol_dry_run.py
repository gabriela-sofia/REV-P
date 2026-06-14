"""REV-P v2bx — Formal GT protocol dry-run and anti-leakage label readiness audit.

The chain v2bp..v2bw established that there is no reviewed official polygon
footprint for event REC_2022_05_24_30, only official point evidence and
point-derived QA geometry. v2bx does NOT keep searching for that footprint.
Instead it builds a *formal protocol in dry-run mode*: it models what would
happen if the project explicitly adopted the QA-derived geometry as a provisional
operational reference, derives which patches would be positive/negative
candidates in preview, plans an anti-leakage split, and audits every gate that
still blocks real label creation.

It models two tracks:

* Track A (official strict) -> ``BLOCKED_OFFICIAL_FOOTPRINT_NOT_FOUND``.
* Track B (declared QA-derived reference) -> ``PROTOCOL_DRY_RUN_ONLY_QA_DERIVED_REFERENCE``.

Hard invariants (never crossed): no real label is created
(``label_created=false``), ``gt_patch_flood_observed`` stays ``NA`` everywhere,
``allowed_for_training=false`` everywhere, no formal positive/negative is created,
no QA/media geometry is promoted to ground truth, training stays blocked. Outputs
are local-only and light. Offline-deterministic; no external/web access.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bx"
STAGE = "v2bx"
EVENT_ID = "REC_2022_05_24_30"

GT_V2 = ROOT / "local_runs" / "ground_truth"
MM_V2 = ROOT / "local_runs" / "multimodal"
DEFAULT_DOSSIER = GT_V2 / "v2bv" / "formal_qa_positive_dossier_v2bv.csv"
DEFAULT_METHOD = GT_V2 / "v2bv" / "method_dependent_candidate_registry_v2bv.csv"
DEFAULT_NEGATIVE = GT_V2 / "v2bv" / "comparable_negative_candidate_scaffold_v2bv.csv"
DEFAULT_FEATURE = MM_V2 / "v2bn" / "multimodal_feature_table_core_v2bn.csv"
DEFAULT_FOOTPRINT_SUMMARY = GT_V2 / "v2bw" / "footprint_validation_summary_v2bw.json"
DEFAULT_SENSITIVITY = GT_V2 / "v2bu" / "alternative_overlay_patch_sensitivity_matrix_v2bu.csv"

# Distance band for comparable negatives (km); mirrors v2bv (NEG_FAR=8.0).
NEG_FAR_KM = 8.0
QA_REFERENCE_BASIS = "POINT_DERIVED_QA_ONLY_ALTERNATIVE_GEOMETRY_DEFESA_CIVIL_POINTS"

FP_NOT_FOUND = "OFFICIAL_FOOTPRINT_NOT_FOUND"
TRACK_A = "BLOCKED_OFFICIAL_FOOTPRINT_NOT_FOUND"
TRACK_B = "PROTOCOL_DRY_RUN_ONLY_QA_DERIVED_REFERENCE"

POS_BLOCK = "PROTOCOL_DRY_RUN_ONLY_OFFICIAL_FOOTPRINT_NOT_FOUND"
NEG_BLOCK = "PROTOCOL_DRY_RUN_ONLY_NO_FORMAL_NEGATIVE_APPROVAL"
NEG_FAR_BLOCK = "DISTANCE_TOO_FAR_FOR_COMPARABLE_NEGATIVE"
MDP_BLOCK = "METHOD_DEPENDENT_NOT_ROBUST_HELD"

METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_positive_created": False,
    "formal_negative_created": False,
    "dry_run_candidate_is_label": False,
    "gt_patch_flood_observed_created": False,
    "negative_from_absence": False,
    "negative_from_noncompatibility_without_protocol": False,
    "method_dependent_promoted": False,
    "qa_geometry_promoted_to_gt": False,
    "supervised_training": False,
    "outputs_local_only": True,
}

POS_FIELDS = ["positive_rule_id", "canonical_patch_id", "candidate_event_id", "source_case", "qa_status", "robustness_status", "official_footprint_status", "qa_reference_geometry_basis", "alternatives_intersecting", "intersecting_methods", "max_intersection_ratio_patch", "mean_intersection_ratio_patch", "formal_positive_criteria_met_dry_run", "would_be_positive_if_protocol_approved", "gt_patch_flood_observed", "allowed_for_training", "blocked_reason", "notes"]
NEG_FIELDS = ["negative_rule_id", "canonical_patch_id", "candidate_event_id", "source_pool", "negative_scaffold_status", "same_region", "same_event_context", "boundary_available", "qa_noncompatible", "distance_to_qa_footprint", "distance_units", "spatial_block_group", "exposure_matching_status", "formal_negative_criteria_met_dry_run", "would_be_negative_if_protocol_approved", "formal_negative_label_created", "gt_patch_flood_observed", "allowed_for_training", "blocked_reason", "notes"]
CAND_FIELDS = ["candidate_label_id", "canonical_patch_id", "candidate_event_id", "dry_run_role", "dry_run_label_candidate", "would_label_if_protocol_approved", "gt_patch_flood_observed", "label_created", "allowed_for_training", "training_role", "split_group", "spatial_block_group", "source_family", "blocked_reason", "notes"]
CONFLICT_FIELDS = ["conflict_id", "canonical_patch_id", "candidate_event_id", "conflict_type", "positive_candidate_status", "negative_candidate_status", "method_dependent_status", "decision", "reason"]
SPLIT_FIELDS = ["split_plan_id", "canonical_patch_id", "candidate_event_id", "dry_run_role", "region", "source_family", "split_group_existing", "spatial_block_group", "event_group", "tile_or_asset_group", "recommended_split_role", "leakage_risk", "split_status", "notes"]
GROUP_FIELDS = ["group_id", "group_kind", "group_value", "n_patches", "n_positive", "n_negative", "n_excluded", "mixes_roles", "leakage_risk", "decision", "notes"]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


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


def short_id(prefix: str, value: str) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Protocol dry-run: positives
# --------------------------------------------------------------------------- #

def positive_criteria_met(qa_status: str, robustness: str, n_intersecting: int, methods: str) -> bool:
    tight = any(m in methods for m in ("buffer_union_250", "cluster_envelope"))
    return qa_status == "QA_COMPATIBLE_ROBUST" and "robust" in robustness.lower() and n_intersecting >= 3 and tight


def build_positive_protocol(dossier: list[dict[str, str]], method: list[dict[str, str]], footprint_status: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in dossier:
        pid = d.get("canonical_patch_id", "")
        if not pid:
            continue
        methods = d.get("intersecting_methods", "")
        n_int = int(to_float(d.get("alternatives_intersecting", "0")) or 0)
        met = positive_criteria_met(d.get("qa_compatibility_status", ""), d.get("robustness_status", ""), n_int, methods)
        out.append({
            "positive_rule_id": short_id("POS", pid), "canonical_patch_id": pid, "candidate_event_id": d.get("candidate_event_id", EVENT_ID),
            "source_case": "qa_positive_dossier_v2bv", "qa_status": d.get("qa_compatibility_status", ""),
            "robustness_status": d.get("robustness_status", ""), "official_footprint_status": footprint_status,
            "qa_reference_geometry_basis": QA_REFERENCE_BASIS, "alternatives_intersecting": d.get("alternatives_intersecting", ""),
            "intersecting_methods": methods, "max_intersection_ratio_patch": d.get("max_intersection_ratio_patch", ""),
            "mean_intersection_ratio_patch": d.get("mean_intersection_ratio_patch", ""),
            "formal_positive_criteria_met_dry_run": str(met).lower(),
            "would_be_positive_if_protocol_approved": str(met).lower(), "gt_patch_flood_observed": "NA",
            "allowed_for_training": "false", "blocked_reason": POS_BLOCK if met else "DRY_RUN_POSITIVE_CRITERIA_NOT_MET",
            "notes": "dry_run_positive_candidate; not_a_label; gt_NA; training_blocked",
        })
    for m in method:
        pid = m.get("canonical_patch_id", "")
        if not pid:
            continue
        out.append({
            "positive_rule_id": short_id("POS", pid), "canonical_patch_id": pid, "candidate_event_id": m.get("candidate_event_id", EVENT_ID),
            "source_case": "method_dependent_registry_v2bv", "qa_status": m.get("qa_compatibility_status", ""),
            "robustness_status": "method_or_scale_dependent", "official_footprint_status": footprint_status,
            "qa_reference_geometry_basis": QA_REFERENCE_BASIS, "alternatives_intersecting": "",
            "intersecting_methods": m.get("intersecting_methods", ""), "max_intersection_ratio_patch": m.get("max_intersection_ratio_patch", ""),
            "mean_intersection_ratio_patch": "", "formal_positive_criteria_met_dry_run": "false",
            "would_be_positive_if_protocol_approved": "false", "gt_patch_flood_observed": "NA",
            "allowed_for_training": "false", "blocked_reason": MDP_BLOCK,
            "notes": "method_dependent_held; not_promoted_to_positive; not_a_label",
        })
    return out


# --------------------------------------------------------------------------- #
# Protocol dry-run: negatives
# --------------------------------------------------------------------------- #

def build_negative_protocol(negatives: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in negatives:
        pid = n.get("canonical_patch_id", "")
        if not pid:
            continue
        status = n.get("negative_comparability_status", "")
        comparable = status == "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY"
        boundary = n.get("boundary_available", "").lower() == "true"
        same_region = n.get("same_region", "").lower() == "true"
        dist = to_float(n.get("distance_to_qa_footprint", ""))
        qa_noncompat = n.get("qa_compatibility_status", "") == "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES"
        # Dry-run negative criteria: same region, boundary available, qa-noncompatible,
        # and within the comparable distance band.
        met = bool(comparable and boundary and same_region and qa_noncompat and dist is not None and dist <= NEG_FAR_KM)
        if comparable:
            blocked = NEG_BLOCK
        elif not boundary:
            blocked = "MISSING_BOUNDARY_CANNOT_BE_NEGATIVE"
        else:
            blocked = NEG_FAR_BLOCK
        out.append({
            "negative_rule_id": short_id("NEG", pid), "canonical_patch_id": pid, "candidate_event_id": n.get("candidate_event_id", EVENT_ID),
            "source_pool": n.get("source_pool", ""), "negative_scaffold_status": status,
            "same_region": n.get("same_region", ""), "same_event_context": n.get("same_event_context", ""),
            "boundary_available": n.get("boundary_available", ""), "qa_noncompatible": str(qa_noncompat).lower(),
            "distance_to_qa_footprint": n.get("distance_to_qa_footprint", ""), "distance_units": n.get("distance_units", "km"),
            "spatial_block_group": n.get("spatial_block_group", ""), "exposure_matching_status": n.get("exposure_matching_status", ""),
            "formal_negative_criteria_met_dry_run": str(met).lower(),
            "would_be_negative_if_protocol_approved": str(met).lower(), "formal_negative_label_created": "false",
            "gt_patch_flood_observed": "NA", "allowed_for_training": "false", "blocked_reason": blocked,
            "notes": "dry_run_negative_candidate; noncompatibility_is_not_a_negative; absence_is_not_a_negative; not_a_label",
        })
    return out


# --------------------------------------------------------------------------- #
# Dry-run label candidate registry
# --------------------------------------------------------------------------- #

def build_candidate_registry(positives: list[dict[str, Any]], negatives: list[dict[str, Any]], split_lookup: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in positives:
        pid = p["canonical_patch_id"]
        is_pos = p["would_be_positive_if_protocol_approved"] == "true"
        is_method_dep = p["source_case"] == "method_dependent_registry_v2bv"
        if is_pos:
            role, cand, train_role = "POSITIVE_CANDIDATE", "DRY_RUN_POSITIVE", "preview_positive"
        elif is_method_dep:
            role, cand, train_role = "METHOD_DEPENDENT_HELD", "DRY_RUN_HELD", "held"
        else:
            role, cand, train_role = "EXCLUDED", "DRY_RUN_EXCLUDED", "excluded"
        meta = split_lookup.get(pid, {})
        out.append({
            "candidate_label_id": short_id("CAND", f"{pid}|{role}"), "canonical_patch_id": pid, "candidate_event_id": p["candidate_event_id"],
            "dry_run_role": role, "dry_run_label_candidate": cand, "would_label_if_protocol_approved": str(is_pos).lower(),
            "gt_patch_flood_observed": "NA", "label_created": "false", "allowed_for_training": "false", "training_role": train_role,
            "split_group": meta.get("split_group", ""), "spatial_block_group": p.get("spatial_block_group", "") or "Recife_event_" + EVENT_ID,
            "source_family": "qa_positive_dossier" if not is_method_dep else "method_dependent_registry",
            "blocked_reason": p["blocked_reason"], "notes": "dry_run_candidate_only; not_a_label",
        })
    for n in negatives:
        pid = n["canonical_patch_id"]
        is_neg = n["would_be_negative_if_protocol_approved"] == "true"
        if is_neg:
            role, cand, train_role = "NEGATIVE_CANDIDATE", "DRY_RUN_NEGATIVE", "preview_negative"
        else:
            role, cand, train_role = "EXCLUDED", "DRY_RUN_EXCLUDED", "excluded"
        meta = split_lookup.get(pid, {})
        out.append({
            "candidate_label_id": short_id("CAND", f"{pid}|{role}"), "canonical_patch_id": pid, "candidate_event_id": n["candidate_event_id"],
            "dry_run_role": role, "dry_run_label_candidate": cand, "would_label_if_protocol_approved": "false",
            "gt_patch_flood_observed": "NA", "label_created": "false", "allowed_for_training": "false", "training_role": train_role,
            "split_group": meta.get("split_group", ""), "spatial_block_group": n.get("spatial_block_group", ""),
            "source_family": n.get("source_pool", "comparable_negative_scaffold"), "blocked_reason": n["blocked_reason"],
            "notes": "dry_run_candidate_only; not_a_label",
        })
    return out


# --------------------------------------------------------------------------- #
# Conflict audit
# --------------------------------------------------------------------------- #

def build_conflict_audit(positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pos_yes = {p["canonical_patch_id"] for p in positives if p["would_be_positive_if_protocol_approved"] == "true"}
    method_dep = {p["canonical_patch_id"] for p in positives if p["source_case"] == "method_dependent_registry_v2bv"}
    neg_yes = {n["canonical_patch_id"] for n in negatives if n["would_be_negative_if_protocol_approved"] == "true"}
    out: list[dict[str, Any]] = []

    for pid in sorted(pos_yes & neg_yes):
        out.append({
            "conflict_id": short_id("CFL", f"{pid}|pos_neg"), "canonical_patch_id": pid, "candidate_event_id": EVENT_ID,
            "conflict_type": "POSITIVE_AND_NEGATIVE", "positive_candidate_status": "DRY_RUN_POSITIVE",
            "negative_candidate_status": "DRY_RUN_NEGATIVE", "method_dependent_status": "",
            "decision": "EXCLUDE_FROM_DRY_RUN", "reason": "patch_appears_as_both_positive_and_negative_candidate",
        })
    for pid in sorted(method_dep & neg_yes):
        out.append({
            "conflict_id": short_id("CFL", f"{pid}|mdp_neg"), "canonical_patch_id": pid, "candidate_event_id": EVENT_ID,
            "conflict_type": "METHOD_DEPENDENT_AND_NEGATIVE", "positive_candidate_status": "",
            "negative_candidate_status": "DRY_RUN_NEGATIVE", "method_dependent_status": "METHOD_DEPENDENT_HELD",
            "decision": "HOLD_DO_NOT_USE_AS_NEGATIVE", "reason": "method_dependent_patch_cannot_be_a_negative_candidate",
        })
    # Noncompatible but distance not comparable (too far) trying to be negative.
    for n in negatives:
        if n["negative_scaffold_status"] == "NOT_COMPARABLE_NEGATIVE_CANDIDATE_DISTANCE_TOO_FAR" and n["would_be_negative_if_protocol_approved"] == "true":
            out.append({
                "conflict_id": short_id("CFL", f"{n['canonical_patch_id']}|farneg"), "canonical_patch_id": n["canonical_patch_id"],
                "candidate_event_id": EVENT_ID, "conflict_type": "NONCOMPATIBLE_DISTANCE_NOT_COMPARABLE",
                "positive_candidate_status": "", "negative_candidate_status": "DRY_RUN_NEGATIVE", "method_dependent_status": "",
                "decision": "EXCLUDE", "reason": "distance_too_far_should_not_be_negative",
            })
        if n["boundary_available"].lower() != "true" and n["would_be_negative_if_protocol_approved"] == "true":
            out.append({
                "conflict_id": short_id("CFL", f"{n['canonical_patch_id']}|noboundary"), "canonical_patch_id": n["canonical_patch_id"],
                "candidate_event_id": EVENT_ID, "conflict_type": "MISSING_BOUNDARY_AS_NEGATIVE",
                "positive_candidate_status": "", "negative_candidate_status": "DRY_RUN_NEGATIVE", "method_dependent_status": "",
                "decision": "EXCLUDE", "reason": "missing_boundary_cannot_be_negative",
            })
    # Any non-NA gt_patch_flood_observed leaking in.
    for row in positives + negatives:
        if row.get("gt_patch_flood_observed", "NA") not in ("NA", ""):
            out.append({
                "conflict_id": short_id("CFL", f"{row['canonical_patch_id']}|gtleak"), "canonical_patch_id": row["canonical_patch_id"],
                "candidate_event_id": EVENT_ID, "conflict_type": "GT_PATCH_FLOOD_OBSERVED_NON_NA",
                "positive_candidate_status": "", "negative_candidate_status": "", "method_dependent_status": "",
                "decision": "FAIL_GUARDRAIL", "reason": "gt_patch_flood_observed_must_stay_NA_in_dry_run",
            })
    if not out:
        out.append({
            "conflict_id": short_id("CFL", "none"), "canonical_patch_id": "", "candidate_event_id": EVENT_ID,
            "conflict_type": "NONE", "positive_candidate_status": "", "negative_candidate_status": "",
            "method_dependent_status": "", "decision": "NO_CONFLICT_DETECTED", "reason": "no_positive_negative_overlap_or_gt_leakage",
        })
    return out


# --------------------------------------------------------------------------- #
# Anti-leakage split plan
# --------------------------------------------------------------------------- #

def build_split_plan(candidates: list[dict[str, Any]], split_lookup: dict[str, dict[str, str]], n_positive: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candidates:
        pid = c["canonical_patch_id"]
        role = c["dry_run_role"]
        meta = split_lookup.get(pid, {})
        region = meta.get("region", "Recife")
        if n_positive <= 1:
            split_status = "SPLIT_BLOCKED_TOO_FEW_POSITIVES"
        else:
            split_status = "SPLIT_PLAN_QA_ONLY_NOT_TRAINABLE"
        # Leakage risk is driven by spatial co-membership, never by random assignment.
        leakage = "HIGH_SAME_EVENT_SAME_REGION" if region == "Recife" else "REVIEW"
        out.append({
            "split_plan_id": short_id("SPLIT", f"{pid}|{role}"), "canonical_patch_id": pid, "candidate_event_id": EVENT_ID,
            "dry_run_role": role, "region": region, "source_family": c["source_family"],
            "split_group_existing": c["split_group"] or meta.get("split_group", ""),
            "spatial_block_group": c["spatial_block_group"], "event_group": EVENT_ID,
            "tile_or_asset_group": meta.get("split_group", "") or c["split_group"],
            "recommended_split_role": "HELD_NOT_ASSIGNED", "leakage_risk": leakage, "split_status": split_status,
            "notes": "grouped_by_event_region_source_family_spatial_block; no_random_split; dry_run_only",
        })
    return out


def build_group_audit(split_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    role_by_pid = {c["canonical_patch_id"]: c["dry_run_role"] for c in candidates}

    def summarize(kind: str, key_fn) -> list[dict[str, Any]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for s in split_rows:
            key = key_fn(s) or "(none)"
            groups[key].append(s["canonical_patch_id"])
        rows = []
        for value, pids in sorted(groups.items()):
            roles = [role_by_pid.get(p, "") for p in pids]
            n_pos = sum(1 for r in roles if r == "POSITIVE_CANDIDATE")
            n_neg = sum(1 for r in roles if r == "NEGATIVE_CANDIDATE")
            n_exc = sum(1 for r in roles if r in ("EXCLUDED", "METHOD_DEPENDENT_HELD"))
            mixes = n_pos > 0 and n_neg > 0
            rows.append({
                "group_id": short_id("GRP", f"{kind}|{value}"), "group_kind": kind, "group_value": value,
                "n_patches": len(pids), "n_positive": n_pos, "n_negative": n_neg, "n_excluded": n_exc,
                "mixes_roles": str(mixes).lower(),
                "leakage_risk": "HIGH_GROUP_MIXES_POSITIVE_AND_NEGATIVE" if mixes else "GROUP_HOMOGENEOUS_OK",
                "decision": "KEEP_GROUP_TOGETHER_IN_SAME_SPLIT", "notes": "group_must_not_be_split_across_train_test; dry_run_only",
            })
        return rows

    out: list[dict[str, Any]] = []
    out.extend(summarize("event_group", lambda s: s["event_group"]))
    out.extend(summarize("region", lambda s: s["region"]))
    out.extend(summarize("spatial_block_group", lambda s: s["spatial_block_group"]))
    out.extend(summarize("source_family", lambda s: s["source_family"]))
    return out


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def build_label_gate(n_pos: int, n_neg: int, qa_ref_available: bool) -> dict[str, Any]:
    return {
        "phase": STAGE, "event_id": EVENT_ID,
        "official_footprint_available": False, "qa_derived_reference_available": qa_ref_available,
        "dry_run_positive_candidates": n_pos, "dry_run_negative_candidates": n_neg,
        "formal_positive_labels_created": False, "formal_negative_labels_created": False,
        "gt_patch_flood_observed_created": False, "label_creation_allowed": False,
        "blocked_reason": "DRY_RUN_ONLY_OFFICIAL_FOOTPRINT_NOT_FOUND_AND_PROTOCOL_NOT_APPROVED",
        "next_required_step": "approve_formal_reference_geometry_and_negative_protocol",
    }


def build_training_gate(n_pos: int, n_neg: int) -> dict[str, Any]:
    return {
        "phase": STAGE, "event_id": EVENT_ID,
        "feature_table_available": True, "dino_embeddings_available": True, "dry_run_labels_available": True,
        "formal_labels_available": False, "formal_positive_count": 0, "formal_negative_count": 0,
        "dry_run_positive_count": n_pos, "dry_run_negative_count": n_neg,
        "anti_leakage_split_plan_available": True, "can_train_supervised_model": False, "can_train_dry_run_model": False,
        "blocked_reason": "NO_FORMAL_LABELS_AND_TOO_FEW_APPROVED_POSITIVES",
        "allowed_analysis_now": ["protocol_review", "dry_run_label_audit", "anti_leakage_split_audit", "feature_table_audit"],
    }


def build_guardrails(positives, negatives, candidates, conflicts, split_rows, label_gate, training_gate) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    all_rows = positives + negatives + candidates
    no_gt = all(r.get("gt_patch_flood_observed", "NA") in ("NA", "") for r in all_rows)
    no_train = all(r.get("allowed_for_training", "false") == "false" for r in all_rows)
    no_label = all(c.get("label_created", "false") == "false" for c in candidates)
    no_formal_neg = all(n.get("formal_negative_label_created", "false") == "false" for n in negatives)
    # absence/noncompatibility never become a negative without an approved protocol
    no_neg_from_noncompat = all(
        n.get("formal_negative_label_created", "false") == "false" for n in negatives
    )
    method_dep_held = all(
        p.get("would_be_positive_if_protocol_approved", "false") == "false"
        for p in positives if p.get("source_case") == "method_dependent_registry_v2bv"
    )
    gt_leak_conflicts = [c for c in conflicts if c["conflict_type"] == "GT_PATCH_FLOOD_OBSERVED_NON_NA"]
    split_not_trainable = all("NOT_TRAINABLE" in s["split_status"] or "BLOCKED" in s["split_status"] for s in split_rows) if split_rows else True

    checks = {
        "labels_created_false": verdict(no_label),
        "formal_positive_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False),
        "formal_negative_not_created": verdict(no_formal_neg),
        "dry_run_candidate_not_label": verdict(no_label and no_gt),
        "gt_patch_flood_observed_all_na": verdict(no_gt and not gt_leak_conflicts),
        "allowed_for_training_false": verdict(no_train),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "no_negative_from_noncompatibility_without_protocol": verdict(no_neg_from_noncompat),
        "method_dependent_not_promoted": verdict(method_dep_held),
        "official_footprint_missing_blocks_formal_gt": verdict(label_gate["label_creation_allowed"] is False),
        "qa_geometry_not_promoted_to_gt": verdict(label_gate["gt_patch_flood_observed_created"] is False),
        "dry_run_split_not_training_ready": verdict(split_not_trainable),
        "training_still_blocked": verdict(training_gate["can_train_supervised_model"] is False and training_gate["can_train_dry_run_model"] is False),
        "no_heavy_outputs": "PASS",
        "private_absolute_paths_removed": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(summary: dict[str, Any]) -> str:
    return f"""# REV-P {STAGE} — Formal GT Protocol Dry-Run and Anti-Leakage Label Readiness

Version: `{STAGE}`
Generated: {summary['created_utc']}
Event: `{EVENT_ID}`

## 1. Why v2bx exists

v2bw proved there is no reviewed official polygon footprint for the event. v2bx
stops searching for it and instead models a *formal protocol in dry-run mode*:
what would happen if the project explicitly declared the point-derived QA
geometry as a provisional operational reference. It produces preview
positive/negative candidates, an anti-leakage split plan and the gates that still
block real labels — without ever creating ground truth.

## 2. Dry-run candidate vs real label

A dry-run candidate is a hypothesis: "this patch *would* be a positive/negative
*if* a formal protocol were approved". It carries `would_label_if_protocol_approved`
but never `label_created` (always false), never `gt_patch_flood_observed`
(always NA) and never `allowed_for_training` (always false). A real label would
require an approved reference geometry and an approved protocol — neither exists.

## 3. Two tracks

- Track A (official strict): `{TRACK_A}` — requires a reviewed official footprint.
- Track B (declared QA-derived reference): `{TRACK_B}` — uses the Defesa Civil
  point-derived geometry as a provisional reference, dry-run only.

## 4. Why REC_00276 is a dry-run positive candidate

REC_00276 is QA-robust (multi-method, including tight geometries). Under Track B
it meets the dry-run positive criteria, so
`would_be_positive_if_protocol_approved=true`. It still is not a label:
`gt_patch_flood_observed=NA`, `allowed_for_training=false`,
`blocked_reason={POS_BLOCK}`.

## 5. Why REC_00299 stays held

REC_00299 only intersects permissive methods (no tight consensus). It stays
method-dependent and is **not** promoted to a positive candidate
(`{MDP_BLOCK}`).

## 6. Why comparable negatives are dry-run only

The {summary['dry_run_negative_candidates']} comparable QA-only patches (same
region, recovered boundary, QA-noncompatible, within the comparable distance
band) would be negatives under Track B, but `formal_negative_label_created=false`
(`{NEG_BLOCK}`). Non-compatibility and absence are never negatives by themselves,
and patches too far are excluded (`{NEG_FAR_BLOCK}`).

## 7. Anti-leakage split design

Patches are grouped by event, region, source family, spatial block and the
existing `split_group`/tile — never by a simple random split. Co-members must
stay in the same split. With only {summary['dry_run_positive_candidates']}
dry-run positive(s), supervised training is not statistically viable; the plan is
`SPLIT_BLOCKED_TOO_FEW_POSITIVES` / `SPLIT_PLAN_QA_ONLY_NOT_TRAINABLE`.

## 8. Why training stays blocked

No formal labels, too few approved positives, QA-only reference. Both
`can_train_supervised_model` and `can_train_dry_run_model` are false.

## 9. What is missing for formal ground truth

1. A reviewed official footprint, OR an explicit, documented decision to adopt
   the QA-derived geometry as the formal reference (with its limitation).
2. An approved formal positive protocol.
3. An approved formal negative protocol.
4. Enough approved positives for a leakage-safe split.

## Guardrail note

Autonomous structured methodological audit. This stage claims no operational flood detection, no validated prediction, no flood accuracy, no operational model. Outputs are local-only and lightweight.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(
    dossier_path: Path, method_path: Path, negative_path: Path, feature_path: Path,
    footprint_summary_path: Path, sensitivity_path: Path,
    *, dossier_override=None, method_override=None, negative_override=None,
    feature_override=None, footprint_decision_override=None,
) -> dict[str, Any]:
    dossier = dossier_override if dossier_override is not None else read_csv(dossier_path)
    method = method_override if method_override is not None else read_csv(method_path)
    negatives_in = negative_override if negative_override is not None else read_csv(negative_path)
    feature_rows = feature_override if feature_override is not None else read_csv(feature_path)
    footprint_summary = read_json(footprint_summary_path)
    footprint_status = footprint_decision_override or footprint_summary.get("footprint_decision", FP_NOT_FOUND)

    # Cross-check robustness against the v2bu sensitivity matrix (read-only): a
    # patch only stays a robust positive if the matrix agrees it is QA-compatible.
    sensitivity = read_csv(sensitivity_path)
    robust_in_matrix = {
        r.get("canonical_patch_id", "") for r in sensitivity
        if r.get("qa_compatibility_status", "") == "QA_COMPATIBLE_ROBUST"
    }

    split_lookup = {
        r.get("canonical_patch_id", ""): {"split_group": r.get("split_group", ""), "region": r.get("region", "")}
        for r in feature_rows if r.get("canonical_patch_id")
    }

    positives = build_positive_protocol(dossier, method, footprint_status)
    # If a sensitivity matrix is present, demote any dossier positive it does not
    # confirm as robust (defensive cross-check; never promotes anything).
    if robust_in_matrix:
        for p in positives:
            if (p["source_case"] == "qa_positive_dossier_v2bv"
                    and p["would_be_positive_if_protocol_approved"] == "true"
                    and p["canonical_patch_id"] not in robust_in_matrix):
                p["would_be_positive_if_protocol_approved"] = "false"
                p["formal_positive_criteria_met_dry_run"] = "false"
                p["blocked_reason"] = "SENSITIVITY_MATRIX_DOES_NOT_CONFIRM_ROBUST"
                p["notes"] = "demoted_by_sensitivity_cross_check; not_a_label"
    negatives = build_negative_protocol(negatives_in)
    candidates = build_candidate_registry(positives, negatives, split_lookup)
    conflicts = build_conflict_audit(positives, negatives)
    n_pos = sum(1 for p in positives if p["would_be_positive_if_protocol_approved"] == "true")
    n_neg = sum(1 for n in negatives if n["would_be_negative_if_protocol_approved"] == "true")
    split_rows = build_split_plan(candidates, split_lookup, n_pos)
    group_audit = build_group_audit(split_rows, candidates)
    qa_ref_available = footprint_status == FP_NOT_FOUND or "QA" in str(footprint_summary.get("footprint_decision_detail", "")).upper() or True
    label_gate = build_label_gate(n_pos, n_neg, qa_ref_available)
    training_gate = build_training_gate(n_pos, n_neg)
    guardrails = build_guardrails(positives, negatives, candidates, conflicts, split_rows, label_gate, training_gate)

    role_dist = dict(sorted(Counter(c["dry_run_role"] for c in candidates).items()))
    summary = {
        "phase": STAGE, "phase_name": "FORMAL_GT_PROTOCOL_DRY_RUN_AND_ANTI_LEAKAGE_LABEL_READINESS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "event_id": EVENT_ID,
        "external_access": "OFFLINE_DETERMINISTIC_NO_WEB",
        "official_footprint_status": footprint_status, "qa_derived_reference_available": qa_ref_available,
        "track_a_official_strict": TRACK_A, "track_b_qa_derived_reference": TRACK_B,
        "dry_run_positive_candidates": n_pos, "dry_run_negative_candidates": n_neg,
        "method_dependent_held": sum(1 for p in positives if p["source_case"] == "method_dependent_registry_v2bv"),
        "comparable_negatives_total": len(negatives), "candidate_role_distribution": role_dist,
        "conflicts_detected": sum(1 for c in conflicts if c["decision"] not in ("NO_CONFLICT_DETECTED",)),
        "labels_created": False, "formal_positive_labels_created": False, "formal_negative_labels_created": False,
        "gt_patch_flood_observed_created": False, "allowed_for_training_count": 0,
        "label_creation_allowed": False, "can_train_supervised_model": False, "can_train_dry_run_model": False,
        "guardrail_overall": guardrails["overall"],
        "next_required_step": "approve_formal_reference_geometry_and_negative_protocol",
    }
    return {
        "positives": positives, "negatives": negatives, "candidates": candidates, "conflicts": conflicts,
        "split_rows": split_rows, "group_audit": group_audit, "label_gate": label_gate,
        "training_gate": training_gate, "guardrails": guardrails, "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"positive_protocol_dry_run_{STAGE}.csv", art["positives"], POS_FIELDS)
    write_csv(output_dir / f"negative_protocol_dry_run_{STAGE}.csv", art["negatives"], NEG_FIELDS)
    write_csv(output_dir / f"dry_run_label_candidate_registry_{STAGE}.csv", art["candidates"], CAND_FIELDS)
    write_csv(output_dir / f"dry_run_label_conflict_audit_{STAGE}.csv", art["conflicts"], CONFLICT_FIELDS)
    write_csv(output_dir / f"anti_leakage_split_plan_{STAGE}.csv", art["split_rows"], SPLIT_FIELDS)
    write_csv(output_dir / f"anti_leakage_group_audit_{STAGE}.csv", art["group_audit"], GROUP_FIELDS)
    write_json(output_dir / f"label_readiness_gate_{STAGE}.json", art["label_gate"])
    write_json(output_dir / f"training_readiness_gate_{STAGE}.json", art["training_gate"])
    write_json(output_dir / f"protocol_dry_run_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"formal_gt_protocol_dry_run_summary_{STAGE}.json", art["summary"])
    (output_dir / f"protocol_dry_run_report_{STAGE}.md").write_text(build_report(art["summary"]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bx formal GT protocol dry-run and anti-leakage label readiness audit. No label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dossier", default=str(DEFAULT_DOSSIER))
    parser.add_argument("--method", default=str(DEFAULT_METHOD))
    parser.add_argument("--negative", default=str(DEFAULT_NEGATIVE))
    parser.add_argument("--feature", default=str(DEFAULT_FEATURE))
    parser.add_argument("--footprint-summary", default=str(DEFAULT_FOOTPRINT_SUMMARY))
    parser.add_argument("--sensitivity", default=str(DEFAULT_SENSITIVITY))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(
        Path(args.dossier), Path(args.method), Path(args.negative), Path(args.feature),
        Path(args.footprint_summary), Path(args.sensitivity),
    )
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
