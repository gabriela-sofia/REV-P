"""REV-P v2bp — Autonomous evidence adjudication and patch-event consistency audit.

Reinterprets every "human review required" flag left by upstream stages as a
*structured autonomous audit*: this stage compares the existing artifacts and
decides, per patch/event/candidate, whether the evidence is internally
consistent, contradictory, circular, insufficient or genuinely ambiguous —
without asking a human to redo work the data already settles.

It adjudicates the v2at event-patch package registry (and the source catalog,
overlay reports, v2bn feature table and v2bo scaffold when present) into an
auditable decision matrix. It can promote a case to
``AUTO_VALIDATED_CANDIDATE_POSITIVE`` when the verifiable evidence agrees, but
it never creates an operational label, never derives a negative from absence,
never enables training. ``gt_patch_flood_observed`` stays NA and
``allowed_for_training`` stays False until a formal ground-truth protocol with
formal negatives is satisfied. ``NEEDS_USER_DECISION`` is reserved for
methodological ambiguity that the repository data cannot resolve.

Outputs are local-only and lightweight.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bp"

STAGE = "v2bp"

# Primary substrate produced by v2at; discovered fail-closed if moved.
DEFAULT_PACKAGE_REGISTRY = ROOT / "datasets" / "v2at_event_patch_package_registry.csv"
DEFAULT_SOURCE_CATALOG = ROOT / "datasets" / "v2at_external_evidence_source_catalog.csv"
DEFAULT_FEATURE_TABLE = ROOT / "local_runs" / "multimodal" / "v2bn" / "multimodal_feature_table_core_v2bn.csv"
DEFAULT_GT_SCAFFOLD = ROOT / "local_runs" / "ground_truth" / "v2bo" / "gt_patch_registry_scaffold_v2bo.csv"

REGION_PREFIX = {"REC": "recife", "PET": "petropolis", "CUR": "curitiba"}

# Source classes that carry independent observational weight vs context-only.
INDEPENDENT_SOURCE_CLASSES = {
    "official_hydromet",
    "official_disaster_record",
    "official_geoinfo",
    "official_geological",
    "operational_mapping",
    "vhr_optical",
}
CONTEXT_ONLY_SOURCE_CLASSES = {"context_low", "methodological_benchmark"}
# Source classes that overlap the visual/structural feature space (DINO/Sentinel
# or GIS proxy) and therefore carry circularity risk if reused as a label.
FEATURE_OVERLAP_SOURCE_CLASSES = {"vhr_optical", "operational_mapping"}


METHODOLOGICAL_GUARDRAILS = {
    "autonomous_adjudication_enabled": True,
    "human_review_interpreted_as_autonomous_audit": True,
    "review_only": True,
    "labels_created": False,
    "targets_created": False,
    "formal_negative_created": False,
    "negative_from_absence": False,
    "supervised_training": False,
    "multimodal_execution_enabled": False,
    "multimodal_training_enabled": False,
    "predictive_claims": False,
    "candidate_positive_is_not_label": True,
    "coordinates_or_events_invented": False,
    "outputs_local_only": True,
}


ADJUDICATION_FIELDS = [
    "adjudication_id",
    "canonical_patch_id",
    "dino_input_id",
    "region",
    "candidate_event_id",
    "source_family",
    "source_path",
    "evidence_type",
    "evidence_status",
    "geometry_status",
    "temporal_alignment_status",
    "spatial_alignment_status",
    "region_consistency_status",
    "patch_consistency_status",
    "source_independence_status",
    "circularity_risk",
    "contradiction_status",
    "missingness_status",
    "auto_decision",
    "auto_decision_confidence",
    "auto_decision_reason",
    "candidate_positive_status",
    "gt_patch_flood_observed",
    "allowed_for_training",
    "needs_user_decision",
    "user_decision_reason",
    "guardrail_status",
    "notes",
]

CONSISTENCY_FIELDS = [
    "package_id",
    "canonical_patch_id",
    "dino_input_id",
    "region",
    "candidate_event_id",
    "region_consistency_status",
    "patch_consistency_status",
    "temporal_alignment_status",
    "spatial_alignment_status",
    "geometry_status",
    "overlay_status",
    "contradiction_status",
    "auto_decision",
    "candidate_positive_status",
]

CANDIDATE_POSITIVE_FIELDS = [
    "candidate_id",
    "canonical_patch_id",
    "region",
    "candidate_event_id",
    "evidence_sources_count",
    "independent_sources_count",
    "geometry_support_status",
    "temporal_support_status",
    "spatial_support_status",
    "auto_validated_candidate_positive",
    "gt_label_created",
    "allowed_for_training",
    "promotion_blocker",
    "reason",
]

REJECTION_FIELDS = [
    "rejection_id",
    "canonical_patch_id",
    "candidate_event_id",
    "rejection_type",
    "reason",
    "conflicting_sources",
    "guardrail_reference",
]

AMBIGUITY_FIELDS = [
    "ambiguity_id",
    "canonical_patch_id",
    "candidate_event_id",
    "ambiguity_type",
    "what_is_missing",
    "why_auto_adjudication_cannot_resolve",
    "recommended_user_decision",
]

SOURCE_INDEPENDENCE_FIELDS = [
    "source_id",
    "source_family",
    "used_as_feature",
    "used_as_label_candidate",
    "used_as_context",
    "independent_from_dino_asset",
    "independent_from_gis_proxy",
    "circularity_risk",
    "decision",
    "reason",
]

TEMPORAL_SPATIAL_FIELDS = [
    "canonical_patch_id",
    "candidate_event_id",
    "region",
    "event_date",
    "asset_date",
    "temporal_window_status",
    "geometry_overlap_status",
    "crs_status",
    "distance_status",
    "alignment_decision",
    "alignment_reason",
]

# Decision vocabulary
AUTO_ACCEPT = "AUTO_ACCEPT_EVIDENCE_CONSISTENT"
AUTO_REJECT_CONTRA = "AUTO_REJECT_EVIDENCE_CONTRADICTORY"
AUTO_REJECT_CIRC = "AUTO_REJECT_SOURCE_CIRCULARITY"
AUTO_REJECT_GEOM = "AUTO_REJECT_GEOMETRY_MISSING"
AUTO_REJECT_TEMPORAL = "AUTO_REJECT_TEMPORAL_MISMATCH"
AUTO_REJECT_REGION = "AUTO_REJECT_REGION_MISMATCH"
AUTO_REJECT_PATCH = "AUTO_REJECT_PATCH_ID_MISMATCH"
AUTO_REJECT_DUP = "AUTO_REJECT_DUPLICATE_OR_STALE"
AUTO_REVIEW_INSUFFICIENT = "AUTO_REVIEW_INSUFFICIENT_EVIDENCE"
AUTO_REVIEW_AMBIGUOUS = "AUTO_REVIEW_AMBIGUOUS"
NEEDS_USER = "NEEDS_USER_DECISION"
BLOCKED_NO_GEOMETRY = "BLOCKED_NO_GEOMETRY"
BLOCKED_NO_BINDING = "BLOCKED_NO_EVENT_BINDING"
BLOCKED_NO_NEGATIVE = "BLOCKED_NO_FORMAL_NEGATIVE"
READY_FOR_GT = "READY_FOR_GT_PROTOCOL_REVIEW"

REJECT_DECISIONS = {
    AUTO_REJECT_CONTRA, AUTO_REJECT_CIRC, AUTO_REJECT_GEOM, AUTO_REJECT_TEMPORAL,
    AUTO_REJECT_REGION, AUTO_REJECT_PATCH, AUTO_REJECT_DUP,
}
BLOCKED_DECISIONS = {BLOCKED_NO_GEOMETRY, BLOCKED_NO_BINDING, BLOCKED_NO_NEGATIVE}


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


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def short_id(prefix: str, value: str) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


# --------------------------------------------------------------------------- #
# Normalization / consistency primitives
# --------------------------------------------------------------------------- #

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_region(text: str) -> str:
    return strip_accents((text or "").strip()).lower()


def region_from_prefix(identifier: str) -> str:
    """Return the normalized region implied by an ID prefix, or '' if unknown."""
    token = (identifier or "").strip().upper().split("_")[0]
    return REGION_PREFIX.get(token, "")


def is_unknown(value: str) -> bool:
    v = (value or "").strip().upper()
    return v in {"", "UNKNOWN", "NA", "N/A", "NONE"} or "MISSING" in v


def truthy(value: str) -> bool:
    return (value or "").strip().lower() == "true"


def parse_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Core per-package adjudication
# --------------------------------------------------------------------------- #

def adjudicate_package(pkg: dict[str, str], feature_index: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Apply explicit consistency rules to one event-patch package.

    Order matters: objective rejections fire first, then circularity, then
    binding/geometry classification, then candidate-positive promotion. Nothing
    here creates a label or enables training.
    """
    patch_id = (pkg.get("patch_id") or "").strip()
    event_id = (pkg.get("candidate_event_id") or pkg.get("event_id") or "").strip()
    region = (pkg.get("region") or "").strip()
    declared_region = normalize_region(region)
    patch_region = region_from_prefix(patch_id)
    event_region = region_from_prefix(event_id)

    has_temporal = truthy(pkg.get("has_temporal_anchor", ""))
    has_spatial = truthy(pkg.get("has_spatial_support", ""))
    has_official = truthy(pkg.get("has_official_source", ""))
    has_geometry = truthy(pkg.get("has_geometry", ""))
    has_overlay = truthy(pkg.get("has_patch_overlay", ""))
    only_context = truthy(pkg.get("has_only_contextual_sources", ""))
    conflict_count = parse_int(pkg.get("conflict_count", "")) or 0
    strong_ev = parse_int(pkg.get("strong_evidence_count", "")) or 0
    evidence_count = parse_int(pkg.get("evidence_count", "")) or 0
    intersection = parse_float(pkg.get("intersection_ratio", ""))
    allowed_use = (pkg.get("allowed_use") or "").strip()
    circular_label_class = (pkg.get("primary_label_source_class") or "").strip().lower()
    feature_for_label = truthy(pkg.get("source_used_as_feature_and_label", ""))
    # Explicit, data-carried marker of irreducible methodological ambiguity
    # (e.g. several equally-weighted typed events plausibly assigned to one
    # patch). Absent from the current real registry, so this stays at zero.
    ambiguous_assignment = truthy(pkg.get("ambiguous_event_assignment", ""))

    feat = feature_index.get(patch_id, {})
    dino_input_id = feat.get("dino_input_id", "")

    # --- consistency sub-statuses (always reported) ---
    if is_unknown(patch_id):
        patch_consistency = "PATCH_UNKNOWN"
    elif patch_region and patch_region != declared_region:
        patch_consistency = "PATCH_REGION_MISMATCH"
    else:
        patch_consistency = "PATCH_CONSISTENT"

    if is_unknown(event_id):
        region_consistency = "EVENT_UNKNOWN"
    elif event_region and declared_region and event_region != declared_region:
        region_consistency = "REGION_MISMATCH"
    elif patch_region and event_region and patch_region != event_region:
        region_consistency = "PATCH_EVENT_REGION_MISMATCH"
    else:
        region_consistency = "REGION_CONSISTENT"

    temporal_status = "TEMPORAL_ANCHOR_PRESENT" if has_temporal else "TEMPORAL_MISSING"
    spatial_status = "SPATIAL_SUPPORT_PRESENT" if has_spatial else "SPATIAL_MISSING"
    if has_overlay:
        geometry_status = "EVENT_GEOMETRY_AND_PATCH_OVERLAY_PRESENT"
    elif has_geometry:
        geometry_status = "EVENT_GEOMETRY_PRESENT_NO_PATCH_OVERLAY"
    else:
        geometry_status = "NO_EVENT_GEOMETRY"
    contradiction_status = "CONFLICT_PRESENT" if conflict_count > 0 else "NO_CONFLICT"

    if has_official and not only_context:
        independence_status = "INDEPENDENT_OFFICIAL_EVIDENCE"
    elif only_context:
        independence_status = "CONTEXT_ONLY"
    else:
        independence_status = "INDEPENDENCE_UNVERIFIED"
    circularity = "HIGH" if (feature_for_label or circular_label_class in FEATURE_OVERLAP_SOURCE_CLASSES) else ("NONE" if independence_status == "INDEPENDENT_OFFICIAL_EVIDENCE" else "LOW")

    missing = [name for name, ok in (("temporal", has_temporal), ("spatial", has_spatial), ("official_source", has_official)) if not ok]
    missingness_status = "COMPLETE_CORE_EVIDENCE" if not missing else "MISSING_" + "_".join(m.upper() for m in missing)

    # --- decision state machine (first match wins) ---
    candidate_positive = "NONE"
    promotion_blocker = ""
    confidence = "HIGH"

    if is_unknown(event_id) or is_unknown(patch_id):
        decision = AUTO_REJECT_CONTRA
        reason = "Event id or patch id is missing/untyped; cannot bind an event to a patch."
    elif patch_consistency == "PATCH_REGION_MISMATCH":
        decision = AUTO_REJECT_REGION
        reason = f"Patch prefix region '{patch_region}' disagrees with declared region '{declared_region}'."
    elif region_consistency in {"REGION_MISMATCH", "PATCH_EVENT_REGION_MISMATCH"}:
        decision = AUTO_REJECT_PATCH
        reason = f"Region tokens disagree across patch/event/declared ({patch_region}/{event_region}/{declared_region})."
    elif conflict_count > 0:
        decision = AUTO_REJECT_CONTRA
        reason = f"{conflict_count} conflicting evidence item(s) recorded for this package."
    elif feature_for_label or circular_label_class in FEATURE_OVERLAP_SOURCE_CLASSES:
        decision = AUTO_REJECT_CIRC
        reason = "A feature-space source would also serve as the label source; circular validation."
    elif ambiguous_assignment and has_official and has_temporal:
        decision = NEEDS_USER
        reason = "Multiple equally-supported typed events plausibly map to this patch; assignment is a methodological choice not settled by the data."
        confidence = "NOT_APPLICABLE"
    elif only_context or (not has_official):
        decision = AUTO_REVIEW_INSUFFICIENT
        reason = "Only contextual sources present; context cannot promote a candidate alone."
        confidence = "MEDIUM"
    elif not has_temporal:
        decision = AUTO_REVIEW_INSUFFICIENT
        reason = "No temporal anchor; cannot align the event to a Sentinel observation."
        confidence = "MEDIUM"
    elif has_overlay and (intersection is None or intersection > 0):
        # Full binding: event geometry overlaps the patch and everything agrees.
        decision = AUTO_ACCEPT
        reason = "Patch-event overlay present and region/patch/event/temporal evidence agree."
        candidate_positive = "AUTO_VALIDATED_CANDIDATE_POSITIVE"
    else:
        # Consistent strong evidence but no patch overlay yet → technical binding blocker.
        promotion_blocker = "NO_PATCH_EVENT_OVERLAY_GEOMETRY"
        strong_grade = (
            allowed_use == "candidate_reference"
            or (has_official and has_spatial and strong_ev >= 1 and allowed_use != "secondary_evaluation_candidate")
        )
        if strong_grade:
            decision = READY_FOR_GT
            candidate_positive = "AUTO_VALIDATED_CANDIDATE_POSITIVE"
            reason = "Strong, independent, temporally-anchored evidence is consistent; only patch overlay geometry is pending for operational GT."
        else:
            decision = BLOCKED_NO_BINDING
            candidate_positive = "SECONDARY_EVALUATION_CANDIDATE_HELD"
            reason = "Official, temporally-anchored but secondary-strength evidence; held pending patch overlay geometry."
            confidence = "MEDIUM"

    # NEEDS_USER_DECISION only for irreducible methodological ambiguity. With the
    # current artifacts every case above resolves objectively, so this stays rare.
    needs_user = decision == NEEDS_USER
    user_reason = reason if needs_user else ""

    return {
        "adjudication_id": short_id("ADJ", pkg.get("package_id", "") or f"{patch_id}|{event_id}"),
        "canonical_patch_id": patch_id,
        "dino_input_id": dino_input_id or "NOT_LINKED",
        "region": region,
        "candidate_event_id": event_id,
        "source_family": "v2at_event_patch_package",
        "source_path": "datasets/v2at_event_patch_package_registry.csv",
        "evidence_type": pkg.get("hazard_type", "") or "unknown_hazard",
        "evidence_status": "INDEPENDENT_OFFICIAL" if independence_status == "INDEPENDENT_OFFICIAL_EVIDENCE" else independence_status,
        "geometry_status": geometry_status,
        "temporal_alignment_status": temporal_status,
        "spatial_alignment_status": spatial_status,
        "region_consistency_status": region_consistency,
        "patch_consistency_status": patch_consistency,
        "source_independence_status": independence_status,
        "circularity_risk": circularity,
        "contradiction_status": contradiction_status,
        "missingness_status": missingness_status,
        "auto_decision": decision,
        "auto_decision_confidence": confidence,
        "auto_decision_reason": reason,
        "candidate_positive_status": candidate_positive,
        "gt_patch_flood_observed": "",  # NA — no operational label
        "allowed_for_training": "False",  # hard gate
        "needs_user_decision": "True" if needs_user else "False",
        "user_decision_reason": user_reason,
        "guardrail_status": "PASS",
        "notes": f"evidence_count={evidence_count}; promotion_blocker={promotion_blocker or 'NONE'}; candidate_positive_is_not_label",
        # carried for downstream views
        "_promotion_blocker": promotion_blocker,
        "_evidence_count": evidence_count,
        "_strong_ev": strong_ev,
        "_has_geometry": has_geometry,
        "_has_overlay": has_overlay,
        "_has_temporal": has_temporal,
        "_has_spatial": has_spatial,
        "_package_id": pkg.get("package_id", ""),
        "_time_delta": pkg.get("time_delta_days", ""),
        "_event_window_start": pkg.get("event_window_start", ""),
        "_asset_date": pkg.get("sentinel_observation_date", ""),
        "_intersection": pkg.get("intersection_ratio", ""),
    }


# --------------------------------------------------------------------------- #
# Derived registries
# --------------------------------------------------------------------------- #

def build_consistency_matrix(adj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in adj_rows:
        out.append(
            {
                "package_id": r["_package_id"],
                "canonical_patch_id": r["canonical_patch_id"],
                "dino_input_id": r["dino_input_id"],
                "region": r["region"],
                "candidate_event_id": r["candidate_event_id"],
                "region_consistency_status": r["region_consistency_status"],
                "patch_consistency_status": r["patch_consistency_status"],
                "temporal_alignment_status": r["temporal_alignment_status"],
                "spatial_alignment_status": r["spatial_alignment_status"],
                "geometry_status": r["geometry_status"],
                "overlay_status": "PRESENT" if r["_has_overlay"] else "ABSENT",
                "contradiction_status": r["contradiction_status"],
                "auto_decision": r["auto_decision"],
                "candidate_positive_status": r["candidate_positive_status"],
            }
        )
    return out


def build_candidate_positive_registry(adj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in adj_rows:
        if r["candidate_positive_status"] != "AUTO_VALIDATED_CANDIDATE_POSITIVE":
            continue
        out.append(
            {
                "candidate_id": short_id("CPOS", r["_package_id"] or r["adjudication_id"]),
                "canonical_patch_id": r["canonical_patch_id"],
                "region": r["region"],
                "candidate_event_id": r["candidate_event_id"],
                "evidence_sources_count": r["_evidence_count"],
                "independent_sources_count": r["_strong_ev"],
                "geometry_support_status": r["geometry_status"],
                "temporal_support_status": r["temporal_alignment_status"],
                "spatial_support_status": r["spatial_alignment_status"],
                "auto_validated_candidate_positive": "True",
                "gt_label_created": "False",
                "allowed_for_training": "False",
                "promotion_blocker": r["_promotion_blocker"] or "NO_PATCH_EVENT_OVERLAY_GEOMETRY",
                "reason": r["auto_decision_reason"],
            }
        )
    return out


def build_rejection_registry(adj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in adj_rows:
        if r["auto_decision"] not in REJECT_DECISIONS:
            continue
        out.append(
            {
                "rejection_id": short_id("REJ", r["_package_id"] or r["adjudication_id"]),
                "canonical_patch_id": r["canonical_patch_id"],
                "candidate_event_id": r["candidate_event_id"],
                "rejection_type": r["auto_decision"],
                "reason": r["auto_decision_reason"],
                "conflicting_sources": r["contradiction_status"],
                "guardrail_reference": "v2bp:autonomous_audit:objective_consistency",
            }
        )
    return out


def build_ambiguity_registry(adj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in adj_rows:
        if r["needs_user_decision"] != "True":
            continue
        out.append(
            {
                "ambiguity_id": short_id("AMB", r["_package_id"] or r["adjudication_id"]),
                "canonical_patch_id": r["canonical_patch_id"],
                "candidate_event_id": r["candidate_event_id"],
                "ambiguity_type": r["auto_decision"],
                "what_is_missing": r["missingness_status"],
                "why_auto_adjudication_cannot_resolve": r["user_decision_reason"],
                "recommended_user_decision": "Methodological decision required; not resolvable from current artifacts.",
            }
        )
    return out


def build_source_independence_audit(source_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Audit each evidence source for circularity (feature vs label reuse)."""
    out = []
    for s in source_rows:
        sclass = (s.get("source_class") or "").strip().lower()
        feature_overlap = sclass in FEATURE_OVERLAP_SOURCE_CLASSES
        context_only = sclass in CONTEXT_ONLY_SOURCE_CLASSES
        independent = sclass in INDEPENDENT_SOURCE_CLASSES
        # No source may currently act as a label (no labels exist); we assess role.
        used_as_label = "candidate" if (independent and not context_only) else "no"
        used_as_context = "yes" if context_only or not independent else "secondary"
        circularity = "LOW" if feature_overlap else ("NONE" if independent else "LOW")
        if context_only:
            decision = "CONTEXT_ONLY_CANNOT_PROMOTE_ALONE"
            reason = "Contextual/benchmark source; supports review but never promotes a candidate alone."
        elif feature_overlap:
            decision = "INDEPENDENT_BUT_FLAG_FEATURE_OVERLAP"
            reason = "Independent optical/mapping source; flag potential overlap with the visual feature space if ever used as a label."
        elif independent:
            decision = "INDEPENDENT_LABEL_CANDIDATE_SOURCE"
            reason = "Independent official source; may open/reinforce a candidate under a formal protocol, never alone."
        else:
            decision = "ROLE_UNVERIFIED"
            reason = "Source role not classified; treat as context until verified."
        out.append(
            {
                "source_id": s.get("source_id", ""),
                "source_family": sclass or "unknown",
                "used_as_feature": "yes" if feature_overlap else "no",
                "used_as_label_candidate": used_as_label,
                "used_as_context": used_as_context,
                "independent_from_dino_asset": "True",
                "independent_from_gis_proxy": "False" if sclass == "operational_mapping" else "True",
                "circularity_risk": circularity,
                "decision": decision,
                "reason": reason,
            }
        )
    return out


def build_temporal_spatial_audit(adj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in adj_rows:
        event_date = r["_event_window_start"] if not is_unknown(r["_event_window_start"]) else "MISSING"
        asset_date = r["_asset_date"] if not is_unknown(r["_asset_date"]) else "MISSING"
        temporal_window = "WITHIN_ANCHOR" if r["_has_temporal"] else "MISSING"
        overlap = "PATCH_OVERLAY_PRESENT" if r["_has_overlay"] else ("EVENT_GEOMETRY_NO_PATCH_OVERLAY" if r["_has_geometry"] else "MISSING")
        crs_status = "NOT_ESTABLISHED"
        distance_status = "OVERLAY_PENDING" if not r["_has_overlay"] else "WITHIN_PATCH"
        if temporal_window == "MISSING" or overlap == "MISSING":
            decision = "ALIGNMENT_INCOMPLETE"
            reason = "Temporal or geometric alignment data missing; not inferred."
        elif r["_has_overlay"]:
            decision = "ALIGNED"
            reason = "Temporal anchor and patch overlay both present."
        else:
            decision = "TEMPORAL_OK_OVERLAY_PENDING"
            reason = "Temporal anchor present; patch overlay geometry pending."
        out.append(
            {
                "canonical_patch_id": r["canonical_patch_id"],
                "candidate_event_id": r["candidate_event_id"],
                "region": r["region"],
                "event_date": event_date,
                "asset_date": asset_date,
                "temporal_window_status": temporal_window,
                "geometry_overlap_status": overlap,
                "crs_status": crs_status,
                "distance_status": distance_status,
                "alignment_decision": decision,
                "alignment_reason": reason,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Gate / guardrails / report
# --------------------------------------------------------------------------- #

def build_promotion_gate(adj_rows: list[dict[str, Any]], candidate_positives: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_training = sum(1 for r in adj_rows if str(r.get("allowed_for_training")) == "True")
    return {
        "phase": STAGE,
        "autonomous_adjudication_enabled": True,
        "human_review_interpreted_as_autonomous_audit": True,
        "labels_created": False,
        "candidate_positives_created": bool(candidate_positives),
        "candidate_positive_count": len(candidate_positives),
        "formal_negatives_created": False,
        "allowed_for_training_count": allowed_training,
        "supervised_training_enabled": False,
        "promotion_to_operational_gt": False,
        "blocked_reason": "NO_FORMAL_NEGATIVES_AND_NO_FINAL_GT_PROTOCOL_APPROVAL",
        "next_required_step": "formal_positive_negative_gt_protocol_resolution",
    }


def build_guardrails(adj_rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, str] = {}

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    checks["no_label_created"] = verdict(all(str(r.get("gt_patch_flood_observed", "")) == "" for r in adj_rows))
    checks["no_training_allowed"] = verdict(all(str(r.get("allowed_for_training")) == "False" for r in adj_rows))
    checks["labels_created_false"] = verdict(gate["labels_created"] is False)
    checks["formal_negatives_false"] = verdict(gate["formal_negatives_created"] is False)
    checks["no_promotion_to_operational_gt"] = verdict(gate["promotion_to_operational_gt"] is False)
    checks["candidate_positive_is_not_label"] = verdict(
        all(r["candidate_positive_status"] != "AUTO_VALIDATED_CANDIDATE_POSITIVE" or str(r["gt_patch_flood_observed"]) == "" for r in adj_rows)
    )
    checks["multimodal_disabled"] = verdict(METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False)
    checks["local_runs_ignored"] = verdict(local_runs_ignored())
    overall = "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary: dict[str, Any]) -> str:
    dd = summary["decision_distribution"]
    dist_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(dd.items())) or "- (none)"
    return f"""# REV-P {STAGE} — Autonomous Evidence Adjudication and Patch-Event Consistency Audit

Version: `{STAGE}`
Generated: {summary['created_utc']}
Primary substrate: `{summary['package_source']}` ({summary['package_count']} packages)

## 1. What this stage did

It read the existing event-patch evidence and adjudicated each package by
explicit, traceable rules — region/patch/event consistency, temporal anchor,
spatial support, geometry/overlay, source independence and contradiction —
producing an auditable decision per case. It created no operational label and
enabled no training.

## 2. "Human review" reinterpreted

Every upstream "human review required" flag was treated as a request for a
*structured autonomous audit*, not a request to stop and ask the user. Cases
that the existing artifacts settle objectively were accepted, rejected or
blocked automatically. Only irreducible methodological ambiguity is left to the
user.

## 3. Sources compared

Event-patch package registry, external evidence source catalog, patch-event
overlay status, the v2bn feature table and the v2bo ground-truth scaffold
(whichever were present). Region was compared accent-normalized, so
`Petropolis` and `Petrópolis` are treated as the same region.

## 4-7. Adjudication outcome

- Auto-accepted (full binding, consistent): **{summary['auto_accepted']}**
- Auto-validated candidate positives (held for overlay, not labels): **{summary['candidate_positive_count']}**
- Auto-rejected (objective inconsistency/contradiction/circularity): **{summary['auto_rejected']}**
- Blocked (technical binding/geometry pending): **{summary['blocked']}**
- Insufficient evidence (auto-review, not user): **{summary['auto_review_insufficient']}**
- Genuinely needs user decision: **{summary['needs_user_decision']}**

Full decision distribution:

{dist_lines}

## 8. Why there is still no operational label

`labels_created=false`, `formal_negatives_created=false`,
`allowed_for_training_count=0`. A candidate positive means the verifiable
evidence agrees — not that an operational flood label exists. The patch-event
overlay geometry is still pending for the strong candidates, and no formal
negative protocol exists. Absence of evidence was never turned into a negative.

## 9. What is missing for formal ground truth

- Patch-event overlay geometry for the candidate positives (digitization).
- A formal positive/negative ground-truth protocol with comparable negatives.
- Reviewer-recorded adjudication for the candidate positives before any label.

## 10. Recommended next step

Acquire/compute the patch-event overlay geometry for the auto-validated
candidate positives, then run the formal positive/negative GT protocol. Training
stays blocked until then.

## Guardrail note

Autonomous methodological audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def discover_input(default: Path, patterns: list[str], search_dirs: list[Path]) -> Path:
    """Return the default if present, else the first rglob match, else default."""
    if default.exists():
        return default
    for base in search_dirs:
        if not base.exists():
            continue
        for pattern in patterns:
            matches = sorted(base.rglob(pattern))
            if matches:
                return matches[0]
    return default


def build_feature_index(feature_table: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in read_csv(feature_table):
        cid = (row.get("canonical_patch_id") or "").strip()
        if cid:
            index[cid] = {"dino_input_id": (row.get("dino_input_id") or "").strip()}
    return index


def build_artifacts(
    package_registry: Path,
    source_catalog: Path,
    feature_table: Path,
) -> dict[str, Any]:
    packages = read_csv(package_registry)
    source_rows = read_csv(source_catalog)
    feature_index = build_feature_index(feature_table)

    adj_rows = [adjudicate_package(p, feature_index) for p in packages]

    consistency = build_consistency_matrix(adj_rows)
    candidate_positives = build_candidate_positive_registry(adj_rows)
    rejections = build_rejection_registry(adj_rows)
    ambiguities = build_ambiguity_registry(adj_rows)
    source_independence = build_source_independence_audit(source_rows)
    temporal_spatial = build_temporal_spatial_audit(adj_rows)
    gate = build_promotion_gate(adj_rows, candidate_positives)
    guardrails = build_guardrails(adj_rows, gate)

    decision_dist = dict(sorted(Counter(r["auto_decision"] for r in adj_rows).items()))
    summary = {
        "phase": STAGE,
        "phase_name": "AUTONOMOUS_EVIDENCE_ADJUDICATION_AND_PATCH_EVENT_CONSISTENCY_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_source": rel_to_root(package_registry),
        "package_count": len(packages),
        "decision_distribution": decision_dist,
        "auto_accepted": sum(1 for r in adj_rows if r["auto_decision"] == AUTO_ACCEPT),
        "auto_rejected": sum(1 for r in adj_rows if r["auto_decision"] in REJECT_DECISIONS),
        "blocked": sum(1 for r in adj_rows if r["auto_decision"] in BLOCKED_DECISIONS),
        "ready_for_gt_protocol": sum(1 for r in adj_rows if r["auto_decision"] == READY_FOR_GT),
        "auto_review_insufficient": sum(1 for r in adj_rows if r["auto_decision"] == AUTO_REVIEW_INSUFFICIENT),
        "needs_user_decision": sum(1 for r in adj_rows if r["needs_user_decision"] == "True"),
        "candidate_positive_count": len(candidate_positives),
        "region_counts": dict(sorted(Counter(r["region"] for r in adj_rows if r["region"]).items())),
        "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "adjudication": adj_rows,
        "consistency": consistency,
        "candidate_positives": candidate_positives,
        "rejections": rejections,
        "ambiguities": ambiguities,
        "source_independence": source_independence,
        "temporal_spatial": temporal_spatial,
        "gate": gate,
        "guardrails": guardrails,
        "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"autonomous_evidence_adjudication_{STAGE}.csv", art["adjudication"], ADJUDICATION_FIELDS)
    write_csv(output_dir / f"patch_event_consistency_matrix_{STAGE}.csv", art["consistency"], CONSISTENCY_FIELDS)
    write_csv(output_dir / f"autonomous_candidate_positive_registry_{STAGE}.csv", art["candidate_positives"], CANDIDATE_POSITIVE_FIELDS)
    write_csv(output_dir / f"autonomous_rejection_registry_{STAGE}.csv", art["rejections"], REJECTION_FIELDS)
    write_csv(output_dir / f"autonomous_ambiguity_registry_{STAGE}.csv", art["ambiguities"], AMBIGUITY_FIELDS)
    write_csv(output_dir / f"source_independence_audit_{STAGE}.csv", art["source_independence"], SOURCE_INDEPENDENCE_FIELDS)
    write_csv(output_dir / f"temporal_spatial_alignment_audit_{STAGE}.csv", art["temporal_spatial"], TEMPORAL_SPATIAL_FIELDS)
    write_json(output_dir / f"gt_promotion_gate_{STAGE}.json", art["gate"])
    write_json(output_dir / f"autonomous_adjudication_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"autonomous_adjudication_summary_{STAGE}.json", art["summary"])
    (output_dir / f"autonomous_adjudication_report_{STAGE}.md").write_text(build_report(art["summary"]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bp autonomous evidence adjudication. Audits existing artifacts; creates no label and enables no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--package-registry", default="")
    parser.add_argument("--source-catalog", default="")
    parser.add_argument("--feature-table", default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    package_registry = (
        Path(args.package_registry)
        if args.package_registry
        else discover_input(DEFAULT_PACKAGE_REGISTRY, ["*event_patch*package*registry*.csv", "*event_patch*registry*.csv"], [ROOT / "datasets", ROOT / "local_runs"])
    )
    source_catalog = (
        Path(args.source_catalog)
        if args.source_catalog
        else discover_input(DEFAULT_SOURCE_CATALOG, ["*evidence_source_catalog*.csv", "*source_catalog*.csv"], [ROOT / "datasets", ROOT / "local_runs"])
    )
    art = build_artifacts(package_registry, source_catalog, Path(args.feature_table))
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
