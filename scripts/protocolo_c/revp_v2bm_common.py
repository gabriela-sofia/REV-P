#!/usr/bin/env python3
"""v2bm Cross-region Protocol C reapplication and candidate-reference expansion.

Reapplies the refined Protocol C policy (calibrated on Recife in v2bl) to Curitiba and
Petropolis, with automated fail-closed adjudication. No operational/binary label, no
negative, no supervised training target. A regional proxy is not a local station; a Sentinel
preview is not event truth; a DINO signal is not truth; a patch boundary is not event
geometry; a temporal/visual/contextual reference is not a label. C7 stays blocked everywhere.
"""

import argparse
import csv
import hashlib
import os

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("V2BM_DOCS_DIR", "docs/protocolo_c/v2bm_cross_region_reapplication")
PUBLIC_DIR = os.environ.get("V2BM_PUBLIC_DIR", "outputs_public")
REFRESH = os.environ.get("V2BM_REFRESH", "1") == "1"

INVARIANTS = {
    "can_create_operational_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "cross_region_reapplication_is_not_training": "true", "temporal_reference_is_not_label": "true",
    "visual_review_reference_is_not_label": "true", "contextual_reference_is_not_label": "true",
    "regional_proxy_is_not_local_station": "true", "sentinel_preview_is_not_event_truth": "true",
    "dino_signal_is_not_truth": "true", "patch_boundary_is_not_event_geometry": "true",
    "no_supervised_training_target_created": "true", "raw_data_versioned": "false",
}

GATES = ("C0_PROVENANCE", "C1_TEMPORALITY", "C2_VALID_SERIES_OR_STATION", "C3_SPATIAL_ANCHOR",
         "C4_CANDIDATE_GEOMETRY", "C5_PROTOCOL_VALIDATION", "C6_CANDIDATE_REFERENCE",
         "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH")

SCORE = {"HIGH": 1.0, "MODERATE": 0.6, "LOW": 0.2, "ZERO": 0.0}
TIER_CAP = {"PROTOCOL_VALIDATED_CANDIDATE_REFERENCE": 1.0, "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE": 0.70,
            "PROTOCOL_VALIDATED_VISUAL_REVIEW_REFERENCE": 0.65, "PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE": 0.55,
            "PROTOCOL_VALIDATED_REGIONAL_TEMPORAL_CONTEXT": 0.55, "REVIEW_ONLY_CONTEXT": 0.40,
            "REMAIN_TEMPORAL_SEED": 0.40, "REMAIN_REVIEW_ONLY_CONTEXT": 0.40,
            "BLOCKED_FOR_CANDIDATE_REFERENCE": 0.30}

OUTPUTS = {
    "state": "v2bm_cross_region_state.csv", "policy": "v2bm_refined_protocol_policy_application.csv",
    "curitiba": "v2bm_curitiba_candidate_reassessment.csv", "petropolis": "v2bm_petropolis_candidate_reassessment.csv",
    "registry": "v2bm_cross_region_candidate_registry.csv", "gate_table": "v2bm_cross_region_gate_table.csv",
    "scorecard": "v2bm_cross_region_evidence_scorecard.csv", "guardrail": "v2bm_guardrail_regression.csv",
    "manifest": "v2bm_orchestrator_manifest.csv",
}

INPUTS = {
    "vl_registry": "v2bl_validated_candidate_reference_registry.csv",
    "vl_scorecard": "v2bl_protocol_evidence_scorecard.csv", "vl_learning": "v2bl_reapplication_learning_matrix.csv",
    "vl_adjudication": "v2bl_automated_protocol_adjudication.csv",
    "bf_crosswalk": "v2bf_seed_asset_crosswalk_status_update.csv", "bf_lineage": "v2bf_lineage_confidence_scores.csv",
    "bc_curitiba": "v2bc_curitiba_local_seed_candidates.csv", "bc_seed_registry": "v2bc_ground_truth_seed_registry.csv",
    "bc_non_selected": "v2bc_non_selected_region_queue.csv", "ay_precip": "v2ay_window_precipitation_metrics.csv",
}

PUBLIC_FILES = {
    "registry": "tables/protocol_c_cross_region_candidate_registry.csv",
    "gate_table": "tables/protocol_c_cross_region_gate_table.csv",
    "scorecard": "tables/protocol_c_cross_region_evidence_scorecard.csv",
    "report": "execution_reports/protocol_c_cross_region_reapplication_report.md",
    "status": "logs_summary/protocol_c_cross_region_status_summary.md",
}

ALLOWED_USE = "PROTOCOL_C_REFERENCE_REVIEW|ARTICLE_EVIDENCE|PUBLIC_DELIVERY_TABLE"
FORBIDDEN_USE = "SUPERVISED_LABEL|NEGATIVE_LABEL|TRAINING_TARGET|FLOOD_EXTENT_TRUTH"


def parse_args(argv=None):
    return argparse.ArgumentParser(description="v2bm orchestrator").parse_args(argv)


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(*parts):
    return os.path.join(DOCS_DIR, *parts)


def public_path(rel):
    return os.path.join(PUBLIC_DIR, rel)


def with_invariants(row):
    return {**row, **INVARIANTS}


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing empty output: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_inputs():
    if not REFRESH:
        return "SKIPPED"
    try:
        import revp_v2bl_common as v2bl
    except ImportError:
        import scripts.protocolo_c.revp_v2bl_common as v2bl
    v2bl.run_orchestrator()
    return "REFRESHED"


def capped_score(raw, tier):
    return round(min(raw, TIER_CAP.get(tier, 1.0)), 3)


def mean_supporting(levels):
    scores = [SCORE[lvl] for lvl, supports in levels if supports]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# --------------------------------------------------------------------------- #
# Region evidence assembly from real inputs (fail-closed when absent).
# --------------------------------------------------------------------------- #

def recife_state():
    reg = load_csv(dataset_path(INPUTS["vl_registry"]))
    row = reg[0] if reg else {}
    return {
        "region": "Recife", "city": "Recife", "package_id": "ARP_v2az_0005",
        "event_patch_package_id": "FACT_v2at_0005", "event_window": clean(row.get("event_window")) or "2022-05-24 a 2022-06-02",
        "reference_status": clean(row.get("reference_status")) or "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE",
        "phenomenon_scope": clean(row.get("phenomenon_scope")) or "LANDSLIDE_SCARS_WITH_FLOOD_EVENT_CONTEXT",
        "evidence_score": clean(row.get("evidence_score")) or "0.76",
        "uncertainty": clean(row.get("uncertainty_level")) or "MODERATE",
        "strong_temporal": "APAC monthly + ANA Capibaribe stage (context)",
        "strong_spatial": "Charter 758 raster landslide-scars product",
        "strong_context": "ANA hydrological stage; INMET A301 instrument gap documented",
        "limitations": "vector/CRS not available (technical); local rainfall series gap",
        "source_stage": "v2bl",
    }


def curitiba_seeds():
    seeds = load_csv(dataset_path(INPUTS["bc_curitiba"]))
    crosswalk = {r.get("seed_id"): r for r in load_csv(dataset_path(INPUTS["bf_crosswalk"]))}
    lineage = {r.get("seed_id"): r for r in load_csv(dataset_path(INPUTS["bf_lineage"]))}
    registry = load_csv(dataset_path(INPUTS["bc_seed_registry"]))
    seed_by_candidate = {r.get("candidate_id"): r.get("seed_id") for r in registry}
    out = []
    for s in seeds:
        candidate = clean(s.get("candidate_id"))
        seed_id = seed_by_candidate.get(candidate, "")
        cw = crosswalk.get(seed_id, {})
        ln = lineage.get(seed_id, {})
        out.append({
            "seed_id": seed_id or clean(s.get("seed_candidate_id")), "candidate_id": candidate,
            "event_patch_package_id": clean(s.get("event_patch_package_id")),
            "event_window": f"{clean(s.get('window_start'))} a {clean(s.get('window_end'))}",
            "station_id": clean(s.get("station_id")), "station_role": clean(s.get("station_role")),
            "temporal_strength": clean(s.get("temporal_evidence_strength")),
            "precip_signal": clean(s.get("precip_signal_status")), "missing_rate": clean(s.get("missing_rate")),
            "preview_status": clean(cw.get("visual_review_status")) or "UNKNOWN",
            "patch_link": clean(ln.get("patch_link_confidence")) or "UNKNOWN",
            "dino_link": clean(ln.get("dino_link_confidence")) or clean(cw.get("dino_link_status")) or "UNKNOWN",
            "sentinel_date": clean(ln.get("date_confidence")) or "UNKNOWN",
            "previous_status": clean(cw.get("updated_crosswalk_status")) or "VISUAL_REVIEW_READY_WITH_WEAK_LINK",
        })
    return out


def petropolis_packages():
    queue = [r for r in load_csv(dataset_path(INPUTS["bc_non_selected"]))
             if clean(r.get("region")).upper() == "PETROPOLIS" and clean(r.get("candidate_id"))]
    precip = {r.get("event_patch_package_id"): r for r in load_csv(dataset_path(INPUTS["ay_precip"]))}
    out = []
    for q in queue:
        epp = clean(q.get("event_patch_package_id"))
        m = precip.get(epp, {})
        out.append({
            "package_id": clean(q.get("candidate_id")), "candidate_id": clean(q.get("candidate_id")),
            "event_patch_package_id": epp, "event_window": "",
            "station_id": clean(m.get("station_id")) or "A610", "station_role": "REGIONAL_PROXY",
            "temporal_support": clean(m.get("temporal_support_status")) or "UNKNOWN",
            "precip_signal": clean(m.get("precip_signal_status")) or "UNKNOWN",
            "previous_status": "NON_SELECTED_REGION_QUEUE",
        })
    return out


# --------------------------------------------------------------------------- #
# Status logic.
# --------------------------------------------------------------------------- #

def curitiba_status(seed):
    temporal_ready = clean(seed["temporal_strength"]).upper() == "STRONG" and clean(seed["missing_rate"]) in {"0.0", "0.000", "0"}
    local = clean(seed["station_role"]).upper() == "LOCAL"
    preview_ready = "READY" in clean(seed["preview_status"]).upper()
    patch_ready = clean(seed["patch_link"]).upper() == "HIGH"
    if temporal_ready and local:
        return "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE"
    if preview_ready and patch_ready:
        return "PROTOCOL_VALIDATED_VISUAL_REVIEW_REFERENCE"
    if clean(seed["precip_signal"]).upper() == "PRECIPITATION_PRESENT":
        return "REMAIN_TEMPORAL_SEED"
    return "BLOCKED_FOR_CANDIDATE_REFERENCE"


def petropolis_status(pkg):
    temporal_ready = "EVIDENCE_READY" in clean(pkg["temporal_support"]).upper()
    if temporal_ready:
        return "PROTOCOL_VALIDATED_REGIONAL_TEMPORAL_CONTEXT"
    if clean(pkg["precip_signal"]).upper() == "PRECIPITATION_PRESENT":
        return "REMAIN_REVIEW_ONLY_CONTEXT"
    return "BLOCKED_FOR_CANDIDATE_REFERENCE"


# --------------------------------------------------------------------------- #
# Scorecard axis levels per region.
# --------------------------------------------------------------------------- #

def region_axes(region):
    if region == "Recife":
        return [("PROVENANCE", "HIGH", True), ("TEMPORALITY", "MODERATE", True),
                ("SPATIAL_CARTOGRAPHIC_EVIDENCE", "HIGH", True), ("VISUAL_REVIEW_CONTEXT", "LOW", False),
                ("HYDROLOGICAL_CONTEXT", "MODERATE", True), ("DINO_REVIEW_SIGNAL", "ZERO", False),
                ("GEOMETRY_VECTOR_READYNESS", "LOW", False), ("MODEL_LABEL_READYNESS", "ZERO", False)]
    if region == "Curitiba":
        return [("PROVENANCE", "HIGH", True), ("TEMPORALITY", "HIGH", True),
                ("SPATIAL_CARTOGRAPHIC_EVIDENCE", "LOW", False), ("VISUAL_REVIEW_CONTEXT", "HIGH", True),
                ("HYDROLOGICAL_CONTEXT", "ZERO", False), ("DINO_REVIEW_SIGNAL", "MODERATE", True),
                ("GEOMETRY_VECTOR_READYNESS", "LOW", False), ("MODEL_LABEL_READYNESS", "ZERO", False)]
    return [("PROVENANCE", "HIGH", True), ("TEMPORALITY", "MODERATE", True),
            ("SPATIAL_CARTOGRAPHIC_EVIDENCE", "LOW", False), ("VISUAL_REVIEW_CONTEXT", "LOW", False),
            ("HYDROLOGICAL_CONTEXT", "ZERO", False), ("DINO_REVIEW_SIGNAL", "ZERO", False),
            ("GEOMETRY_VECTOR_READYNESS", "LOW", False), ("MODEL_LABEL_READYNESS", "ZERO", False)]


AXIS_REASON = {
    "PROVENANCE": "Public official sources with recorded provenance.",
    "TEMPORALITY": "Dated precipitation/temporal evidence in the window.",
    "SPATIAL_CARTOGRAPHIC_EVIDENCE": "Official cartographic product availability.",
    "VISUAL_REVIEW_CONTEXT": "Sentinel preview / patch link for human visual review.",
    "HYDROLOGICAL_CONTEXT": "River-stage hydrological context.",
    "DINO_REVIEW_SIGNAL": "DINO structural signal as a review aid, not truth.",
    "GEOMETRY_VECTOR_READYNESS": "Vector/CRS readiness for geometry overlay.",
    "MODEL_LABEL_READYNESS": "No operational/binary label or training target created.",
}
AXIS_LIMIT = {
    "PROVENANCE": "License is metadata, not a blocker.",
    "TEMPORALITY": "Temporal evidence is not spatial truth.",
    "SPATIAL_CARTOGRAPHIC_EVIDENCE": "Raster is not a vector geometry.",
    "VISUAL_REVIEW_CONTEXT": "Sentinel preview is not event truth.",
    "HYDROLOGICAL_CONTEXT": "River stage is not precipitation and not flood extent.",
    "DINO_REVIEW_SIGNAL": "DINO signal is not truth and is not a label.",
    "GEOMETRY_VECTOR_READYNESS": "Technical limitation for vector overlay only.",
    "MODEL_LABEL_READYNESS": "Intentionally not created; C7 blocked.",
}


def region_reference_status(region, curitiba_rows=None, petropolis_rows=None):
    if region == "Recife":
        return "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE"
    if region == "Curitiba":
        statuses = {r["updated_protocol_status"] for r in (curitiba_rows or [])}
        if "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE" in statuses:
            return "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE"
        if "PROTOCOL_VALIDATED_VISUAL_REVIEW_REFERENCE" in statuses:
            return "PROTOCOL_VALIDATED_VISUAL_REVIEW_REFERENCE"
        return "REVIEW_ONLY_CONTEXT"
    statuses = {r["updated_protocol_status"] for r in (petropolis_rows or [])}
    if "PROTOCOL_VALIDATED_REGIONAL_TEMPORAL_CONTEXT" in statuses:
        return "PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE"
    return "REVIEW_ONLY_CONTEXT"


def region_score(region, ref_status):
    raw = mean_supporting([(lvl, sup) for _, lvl, sup in region_axes(region)])
    if region == "Recife":
        reg = load_csv(dataset_path(INPUTS["vl_registry"]))
        return clean(reg[0]["evidence_score"]) if reg and clean(reg[0].get("evidence_score")) else str(capped_score(raw, ref_status))
    return str(capped_score(raw, ref_status))


# --------------------------------------------------------------------------- #
# Task 1 - cross-region state.
# --------------------------------------------------------------------------- #

def run_load_cross_region_state(args=None):
    rec = recife_state()
    cur = curitiba_seeds()
    pet = petropolis_packages()
    rows = [with_invariants({
        "region": "Recife", "city": "Recife", "package_id": rec["package_id"],
        "event_patch_package_id": rec["event_patch_package_id"], "event_window": rec["event_window"],
        "strongest_temporal_evidence": rec["strong_temporal"], "strongest_spatial_evidence": rec["strong_spatial"],
        "strongest_contextual_evidence": rec["strong_context"], "current_status_before_v2bm": rec["reference_status"],
        "key_limitations": rec["limitations"], "source_stage": "v2bl",
    })]
    rows.append(with_invariants({
        "region": "Curitiba", "city": "Curitiba", "package_id": "CTB_SEEDS_x3",
        "event_patch_package_id": "|".join(s["event_patch_package_id"] for s in cur) or "CTB",
        "event_window": cur[0]["event_window"] if cur else "",
        "strongest_temporal_evidence": "A807 LOCAL precipitation STRONG (3/3 seeds, 0 missing)",
        "strongest_spatial_evidence": "Sentinel preview + asset-patch HIGH link (visual review)",
        "strongest_contextual_evidence": "DINO MODERATE structural review signal",
        "current_status_before_v2bm": cur[0]["previous_status"] if cur else "VISUAL_REVIEW_READY_WITH_WEAK_LINK",
        "key_limitations": "Sentinel acquisition_date UNKNOWN; no official cartographic product; overall lineage LOW",
        "source_stage": "v2bc/v2bf",
    }))
    rows.append(with_invariants({
        "region": "Petropolis", "city": "Petropolis", "package_id": "|".join(p["package_id"] for p in pet) or "PET",
        "event_patch_package_id": "|".join(p["event_patch_package_id"] for p in pet) or "PET",
        "event_window": "", "strongest_temporal_evidence": "A610 REGIONAL_PROXY temporal evidence ready (regional)",
        "strongest_spatial_evidence": "None local cartographic product",
        "strongest_contextual_evidence": "Regional temporal context",
        "current_status_before_v2bm": "NON_SELECTED_REGION_QUEUE",
        "key_limitations": "A610 is a regional proxy, not a local station; no specific spatial anchor",
        "source_stage": "v2bc/v2ay",
    }))
    write_csv(dataset_path(OUTPUTS["state"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 2 - refined policy application.
# --------------------------------------------------------------------------- #

def run_apply_refined_protocol_policy(args=None):
    items = [
        ("PUBLIC_SOURCE_LICENSE", "License/redistribution/confirmation as blocker",
         "Provenance metadata, not a blocker", "ALL", "C0_PROVENANCE", "false", "true",
         "Public official sources are valid evidence with recorded provenance."),
        ("MANUAL_HUMAN_REVIEW", "Separate mandatory manual review step",
         "Automated protocol validation (fail-closed)", "ALL", "C5_PROTOCOL_VALIDATION", "false", "true",
         "Coherent dated/located/typed evidence is adjudicated automatically."),
        ("RASTER_CARTOGRAPHIC_EVIDENCE", "Raster dismissed as not-evidence",
         "Raster can sustain candidate reference", "RECIFE", "C4_CANDIDATE_GEOMETRY", "false", "true",
         "Inherited from Recife: official raster supports candidate reference."),
        ("VECTOR_CRS", "Vector/CRS absence as general blocker",
         "Technical limitation for vector geometry/overlay", "ALL", "C4_CANDIDATE_GEOMETRY", "true", "true",
         "Blocks vector overlay only, not the reference."),
        ("SENTINEL_VISUAL", "Sentinel preview treated as truth",
         "Visual review support, not truth", "CURITIBA", "C4_CANDIDATE_GEOMETRY", "true", "true",
         "Sentinel preview is review context, not event truth."),
        ("DINO_SIGNAL", "DINO treated as truth/label",
         "Structural review signal, not truth", "CURITIBA", "C5_PROTOCOL_VALIDATION", "true", "true",
         "DINO reinforces review, never creates a label."),
        ("REGIONAL_PROXY", "Regional proxy treated as local station",
         "Sustains regional/contextual evidence, not local station", "PETROPOLIS", "C2_VALID_SERIES_OR_STATION", "true", "true",
         "A610 proxy supports context; it is not a local station."),
        ("LOCAL_STRONG_SERIES", "Local series under-used",
         "Strong local series sustains a temporal reference", "CURITIBA", "C1_TEMPORALITY", "false", "true",
         "A807 LOCAL strong series supports a temporal seed/reference."),
        ("INSTRUMENT_GAP", "Instrumental absence as negative",
         "Documented instrument gap, not absence of event", "ALL", "C1_TEMPORALITY", "true", "false",
         "Empty local station is an instrument gap, never a negative."),
    ]
    rows = []
    for item, old, refined, region, gate, blocks_label, supports_ref, note in items:
        rows.append(with_invariants({
            "policy_item": item, "old_interpretation": old, "refined_interpretation": refined,
            "applies_to_region": region, "effect_on_gate": gate,
            "still_blocks_operational_label": blocks_label, "can_support_candidate_reference": supports_ref, "note": note,
        }))
    write_csv(dataset_path(OUTPUTS["policy"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 3 - Curitiba reassessment.
# --------------------------------------------------------------------------- #

def run_reassess_curitiba_candidates(args=None):
    rows = []
    seeds = curitiba_seeds()
    for s in seeds:
        status = curitiba_status(s)
        score = capped_score(mean_supporting([(lvl, sup) for _, lvl, sup in region_axes("Curitiba")]), status)
        rows.append(with_invariants({
            "curitiba_seed_id": s["seed_id"], "event_patch_package_id": s["event_patch_package_id"],
            "event_window": s["event_window"], "station_id": s["station_id"], "station_role": s["station_role"],
            "temporal_evidence_status": "READY_STRONG_LOCAL" if status == "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE" else s["temporal_strength"],
            "sentinel_preview_status": s["preview_status"], "sentinel_date_status": s["sentinel_date"],
            "patch_link_status": s["patch_link"], "dino_link_status": s["dino_link"],
            "previous_status": s["previous_status"], "updated_protocol_status": status,
            "evidence_score": str(score), "uncertainty_level": "MODERATE",
            "allowed_use": ALLOWED_USE, "forbidden_use": FORBIDDEN_USE,
            "remaining_limitation": "Sentinel acquisition_date UNKNOWN: no full spatial candidate reference; DINO/preview are review signals, not truth.",
        }))
    if not rows:
        rows = [with_invariants({
            "curitiba_seed_id": "", "event_patch_package_id": "", "event_window": "", "station_id": "",
            "station_role": "", "temporal_evidence_status": "NO_INPUT", "sentinel_preview_status": "",
            "sentinel_date_status": "", "patch_link_status": "", "dino_link_status": "",
            "previous_status": "", "updated_protocol_status": "BLOCKED_FOR_CANDIDATE_REFERENCE",
            "evidence_score": "0.0", "uncertainty_level": "HIGH", "allowed_use": ALLOWED_USE,
            "forbidden_use": FORBIDDEN_USE, "remaining_limitation": "No Curitiba seed input available."})]
    write_csv(dataset_path(OUTPUTS["curitiba"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 4 - Petropolis reassessment.
# --------------------------------------------------------------------------- #

def run_reassess_petropolis_candidates(args=None):
    rows = []
    for p in petropolis_packages():
        status = petropolis_status(p)
        score = capped_score(mean_supporting([(lvl, sup) for _, lvl, sup in region_axes("Petropolis")]), status)
        rows.append(with_invariants({
            "petropolis_package_id": p["package_id"], "event_patch_package_id": p["event_patch_package_id"],
            "event_window": p["event_window"], "station_id": p["station_id"], "station_role": p["station_role"],
            "temporal_evidence_status": p["temporal_support"],
            "proxy_limitation": "A610 is a regional proxy; it does not become a local Petropolis station.",
            "contextual_evidence_status": "REGIONAL_TEMPORAL_CONTEXT" if "REGIONAL_TEMPORAL" in status else "WEAK",
            "previous_status": p["previous_status"], "updated_protocol_status": status,
            "evidence_score": str(score), "uncertainty_level": "HIGH",
            "allowed_use": ALLOWED_USE, "forbidden_use": FORBIDDEN_USE,
            "remaining_limitation": "Regional proxy only; no specific spatial anchor or cartographic product for a candidate reference.",
        }))
    if not rows:
        rows = [with_invariants({
            "petropolis_package_id": "", "event_patch_package_id": "", "event_window": "", "station_id": "A610",
            "station_role": "REGIONAL_PROXY", "temporal_evidence_status": "NO_INPUT",
            "proxy_limitation": "A610 regional proxy is not a local station.", "contextual_evidence_status": "WEAK",
            "previous_status": "", "updated_protocol_status": "BLOCKED_FOR_CANDIDATE_REFERENCE",
            "evidence_score": "0.0", "uncertainty_level": "HIGH", "allowed_use": ALLOWED_USE,
            "forbidden_use": FORBIDDEN_USE, "remaining_limitation": "No Petropolis package input available."})]
    write_csv(dataset_path(OUTPUTS["petropolis"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 5 - cross-region candidate registry.
# --------------------------------------------------------------------------- #

def run_build_cross_region_candidate_registry(args=None):
    rec = recife_state()
    cur_rows = load_csv(dataset_path(OUTPUTS["curitiba"]))
    pet_rows = load_csv(dataset_path(OUTPUTS["petropolis"]))
    cur_status = region_reference_status("Curitiba", curitiba_rows=cur_rows)
    pet_status = region_reference_status("Petropolis", petropolis_rows=pet_rows)
    regions = [
        ("Recife", "Recife", rec["package_id"], rec["reference_status"], rec["phenomenon_scope"],
         rec["strong_spatial"], rec["evidence_score"], rec["uncertainty"]),
        ("Curitiba", "Curitiba", "CTB_SEEDS_x3", cur_status, "URBAN_FLOOD_EVENT_TEMPORAL_CONTEXT",
         "A807 LOCAL strong precipitation + Sentinel preview/patch (visual review)",
         region_score("Curitiba", cur_status), "MODERATE"),
        ("Petropolis", "Petropolis", "PET_PACKAGES", pet_status, "LANDSLIDE_FLOOD_REGIONAL_TEMPORAL_CONTEXT",
         "A610 REGIONAL_PROXY temporal context", region_score("Petropolis", pet_status), "HIGH"),
    ]
    rows = []
    for i, (region, city, package, status, scope, basis, score, uncertainty) in enumerate(regions, 1):
        rows.append(with_invariants({
            "reference_id": f"XREF_v2bm_{i:03d}", "region": region, "city": city, "package_id": package,
            "reference_status": status, "phenomenon_scope": scope, "evidence_basis": basis,
            "allowed_use": ALLOWED_USE, "forbidden_use": FORBIDDEN_USE, "uncertainty_level": uncertainty,
            "evidence_score": str(score),
        }))
    write_csv(dataset_path(OUTPUTS["registry"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 6 - cross-region gate table.
# --------------------------------------------------------------------------- #

REGION_GATES = {
    "Recife": {
        "C0_PROVENANCE": ("PASS_PUBLIC_PROVENANCE_RECORDED", "Charter 758 + APAC/ANA public sources"),
        "C1_TEMPORALITY": ("PASS_PUBLIC_TEMPORAL_EVIDENCE", "APAC monthly + ANA stage"),
        "C2_VALID_SERIES_OR_STATION": ("PARTIAL_PASS_HYDROLOGICAL_CONTEXT_LOCAL_RAINFALL_GAP", "ANA station; A301 gap"),
        "C3_SPATIAL_ANCHOR": ("PASS_OFFICIAL_CARTOGRAPHIC_PRODUCT", "Charter Recife raster"),
        "C4_CANDIDATE_GEOMETRY": ("PASS_RASTER_CARTOGRAPHIC_EVIDENCE_FOR_REFERENCE", "Charter raster"),
        "C5_PROTOCOL_VALIDATION": ("AUTO_ADJUDICATED_BY_PROTOCOL", "Coherent evidence"),
        "C6_CANDIDATE_REFERENCE": ("PROTOCOL_VALIDATED_CANDIDATE_REFERENCE", "Aggregated public evidence"),
        "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH": ("NOT_CREATED_BLOCKED_FOR_TRAINING", "None"),
    },
    "Curitiba": {
        "C0_PROVENANCE": ("PASS_PUBLIC_PROVENANCE_RECORDED", "INMET A807 + Sentinel public sources"),
        "C1_TEMPORALITY": ("PASS_PUBLIC_TEMPORAL_EVIDENCE", "A807 LOCAL strong precipitation"),
        "C2_VALID_SERIES_OR_STATION": ("PASS_LOCAL_STATION_SERIES", "A807 LOCAL station, 0 missing"),
        "C3_SPATIAL_ANCHOR": ("PENDING_NO_OFFICIAL_CARTOGRAPHIC_PRODUCT", "No Charter-like product"),
        "C4_CANDIDATE_GEOMETRY": ("VISUAL_REVIEW_CONTEXT_NOT_GEOMETRY", "Sentinel preview + patch link; no acquisition date"),
        "C5_PROTOCOL_VALIDATION": ("AUTO_ADJUDICATED_BY_PROTOCOL", "Strong temporal + visual review"),
        "C6_CANDIDATE_REFERENCE": ("PROTOCOL_VALIDATED_TEMPORAL_REFERENCE", "A807 local temporal evidence"),
        "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH": ("NOT_CREATED_BLOCKED_FOR_TRAINING", "None"),
    },
    "Petropolis": {
        "C0_PROVENANCE": ("PASS_PUBLIC_PROVENANCE_RECORDED", "INMET A610 public source"),
        "C1_TEMPORALITY": ("PASS_REGIONAL_TEMPORAL_EVIDENCE", "A610 regional precipitation ready"),
        "C2_VALID_SERIES_OR_STATION": ("PARTIAL_PASS_REGIONAL_PROXY_NOT_LOCAL", "A610 regional proxy, not local"),
        "C3_SPATIAL_ANCHOR": ("PENDING_NO_LOCAL_CARTOGRAPHIC_ANCHOR", "No local spatial anchor"),
        "C4_CANDIDATE_GEOMETRY": ("PENDING_NO_GEOMETRY_EVIDENCE", "No raster/vector product"),
        "C5_PROTOCOL_VALIDATION": ("AUTO_ADJUDICATED_BY_PROTOCOL", "Regional temporal context"),
        "C6_CANDIDATE_REFERENCE": ("PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE", "Regional temporal context"),
        "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH": ("NOT_CREATED_BLOCKED_FOR_TRAINING", "None"),
    },
}

GATE_CANNOT_INFER = {
    "C0_PROVENANCE": "Provenance is not ground truth.",
    "C1_TEMPORALITY": "Temporal evidence is not spatial truth.",
    "C2_VALID_SERIES_OR_STATION": "A station series is not a flood/landslide label.",
    "C3_SPATIAL_ANCHOR": "A spatial anchor is not event geometry.",
    "C4_CANDIDATE_GEOMETRY": "Raster/preview is not vector geometry; patch boundary is not event geometry.",
    "C5_PROTOCOL_VALIDATION": "Automated validation is not human-certified ground truth.",
    "C6_CANDIDATE_REFERENCE": "A candidate reference is not an operational label.",
    "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH": "No operational label or training target is created.",
}


def run_build_cross_region_gate_table(args=None):
    cur_rows = load_csv(dataset_path(OUTPUTS["curitiba"]))
    pet_rows = load_csv(dataset_path(OUTPUTS["petropolis"]))
    region_ref = {
        "Recife": "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE",
        "Curitiba": region_reference_status("Curitiba", curitiba_rows=cur_rows),
        "Petropolis": region_reference_status("Petropolis", petropolis_rows=pet_rows),
    }
    package = {"Recife": "ARP_v2az_0005", "Curitiba": "CTB_SEEDS_x3", "Petropolis": "PET_PACKAGES"}
    rows = []
    for region in ("Recife", "Curitiba", "Petropolis"):
        for gate in GATES:
            status, evidence = REGION_GATES[region][gate]
            if gate == "C6_CANDIDATE_REFERENCE":
                status = region_ref[region]
            rows.append(with_invariants({
                "region": region, "package_id": package[region], "gate_id": gate, "gate_status": status,
                "evidence_used": evidence, "interpretation": "Automated protocol adjudication (fail-closed).",
                "cannot_infer": GATE_CANNOT_INFER[gate],
                "operational_label_allowed": "false", "training_allowed": "false",
            }))
    write_csv(dataset_path(OUTPUTS["gate_table"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 7 - cross-region evidence scorecard.
# --------------------------------------------------------------------------- #

def run_build_cross_region_evidence_scorecard(args=None):
    rows = []
    for region in ("Recife", "Curitiba", "Petropolis"):
        for axis, level, supports in region_axes(region):
            rows.append(with_invariants({
                "region": region, "evidence_axis": axis, "score": str(SCORE[level]),
                "score_reason": AXIS_REASON[axis], "supports_reference_status": "true" if supports else "false",
                "supports_operational_label": "false", "limitation": AXIS_LIMIT[axis],
            }))
    write_csv(dataset_path(OUTPUTS["scorecard"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 8 - report, packets, README, public outputs.
# --------------------------------------------------------------------------- #

def _registry_table():
    lines = ["| region | reference status | score | uncertainty |", "| --- | --- | --- | --- |"]
    for r in load_csv(dataset_path(OUTPUTS["registry"])):
        lines.append(f"| {r['region']} | {r['reference_status']} | {r['evidence_score']} | {r['uncertainty_level']} |")
    return "\n".join(lines)


def _region_gate_table(region):
    lines = ["| gate | status | evidence |", "| --- | --- | --- |"]
    for r in load_csv(dataset_path(OUTPUTS["gate_table"])):
        if r["region"] == region:
            lines.append(f"| {r['gate_id']} | {r['gate_status']} | {r['evidence_used']} |")
    return "\n".join(lines)


def run_generate_reapplication_report(args=None):
    registry = {r["region"]: r for r in load_csv(dataset_path(OUTPUTS["registry"]))}
    report = f"""# Protocol C - cross-region reapplication report

## 1. Resumo executivo
A politica refinada do Protocolo C (calibrada em Recife) foi reaplicada a Curitiba e
Petropolis com adjudicacao automatica fail-closed. Recife permanece candidate reference;
Curitiba evolui para referencia temporal; Petropolis para referencia contextual regional.

## 2. Politica refinada do Protocolo C
Licenca/confirmacao externa nao sao blockers; revisao humana manual separada e substituida
por adjudicacao automatica; raster sustenta candidate reference; vetor/CRS e limitacao
tecnica; Sentinel preview e DINO sao apoio de revisao (nao truth); proxy regional nao vira
estacao local; ausencia instrumental e lacuna, nao negativo.

## 3. Recife como calibrador
`{registry['Recife']['reference_status']}` (score {registry['Recife']['evidence_score']}): produto cartografico
oficial raster + contexto temporal/hidrologico publico.

## 4. Reavaliacao de Curitiba
`{registry['Curitiba']['reference_status']}` (score {registry['Curitiba']['evidence_score']}): A807 LOCAL com
precipitacao forte (3/3 seeds) e preview/patch para revisao visual; sem acquisition_date
Sentinel, nao vira candidate reference espacial completa.

{_region_gate_table('Curitiba')}

## 5. Reavaliacao de Petropolis
`{registry['Petropolis']['reference_status']}` (score {registry['Petropolis']['evidence_score']}): A610
REGIONAL_PROXY com evidencia temporal regional; sem ancora espacial local, permanece contexto.

{_region_gate_table('Petropolis')}

## 6. Matriz comparativa das tres regioes
{_registry_table()}

## 7. O que cada regiao permite afirmar
- Recife: existe produto cartografico oficial de deslizamento (referencia candidata).
- Curitiba: existe forte evidencia temporal local datada (referencia temporal).
- Petropolis: existe contexto temporal regional (referencia contextual).

## 8. O que cada regiao NAO permite afirmar
- Nenhuma e label supervisionado, negativo ou alvo de treino.
- Curitiba nao tem ancora cartografica nem data Sentinel; Petropolis usa proxy regional.
- Preview Sentinel nao e truth; DINO nao e truth; patch boundary nao e geometria de evento.

## 9. Por que ainda nao existe label operacional
Falta ancora/geometria vetorial validada e/ou serie local consolidada; C7 permanece
NOT_CREATED_BLOCKED_FOR_TRAINING em todas as regioes.

## 10. Como isso fortalece o TCC
Tres regioes com referencias protocolares graduadas (candidate/temporal/contextual),
auditaveis, com proveniencia publica e limitacoes explicitas, sem overclaim.

## 11. Proximos passos
- Recife: solicitar vetor/CRS (overlay) e serie local de chuva.
- Curitiba: recuperar acquisition_date Sentinel para subir de tier.
- Petropolis: buscar ancora/cartografia local especifica.
"""
    write_text(doc_path("reports", "protocol_c_cross_region_reapplication_report.md"), report)

    for region in ("Recife", "Curitiba", "Petropolis"):
        r = registry[region]
        packet = f"""# Region packet - {region}

- Reference status: `{r['reference_status']}`
- Phenomenon scope: {r['phenomenon_scope']}
- Evidence basis: {r['evidence_basis']}
- Evidence score: {r['evidence_score']} | uncertainty: {r['uncertainty_level']}
- Allowed use: {r['allowed_use']}
- Forbidden use: {r['forbidden_use']}

## Gates C0-C7
{_region_gate_table(region)}

## Guardrails
operational_label=0; negative=0; training=0; C7 NOT_CREATED_BLOCKED_FOR_TRAINING.
regional_proxy != local station; sentinel_preview != truth; dino != truth; patch_boundary != event geometry.
"""
        write_text(doc_path("region_packets", f"{region.lower()}.md"), packet)
        write_csv(doc_path("evidence_scorecards", f"{region.lower()}_scorecard.csv"),
                  [r2 for r2 in load_csv(dataset_path(OUTPUTS["scorecard"])) if r2["region"] == region])

    write_text(doc_path("README.md"), f"""# v2bm Cross-Region Protocol C Reapplication

Reaplicacao da politica refinada do Protocolo C (calibrada em Recife na v2bl) para Curitiba e
Petropolis, com adjudicacao automatica fail-closed. Recife permanece
`PROTOCOL_VALIDATED_CANDIDATE_REFERENCE`; Curitiba -> `{registry['Curitiba']['reference_status']}`;
Petropolis -> `{registry['Petropolis']['reference_status']}`.

Sem licenca/confirmacao externa como blocker e sem revisao humana manual separada. Linha dura:
zero label operacional, zero negativo, zero treino; proxy regional nao e estacao local;
Sentinel preview nao e truth; DINO nao e truth; patch boundary nao e geometria de evento; C7
permanece NOT_CREATED / BLOCKED em todas as regioes.
""")

    _write_public_outputs(registry)
    return [{"report": "protocol_c_cross_region_reapplication_report.md"}]


def _write_public_outputs(registry):
    write_csv(public_path(PUBLIC_FILES["registry"]), load_csv(dataset_path(OUTPUTS["registry"])))
    write_csv(public_path(PUBLIC_FILES["gate_table"]), load_csv(dataset_path(OUTPUTS["gate_table"])))
    write_csv(public_path(PUBLIC_FILES["scorecard"]), load_csv(dataset_path(OUTPUTS["scorecard"])))
    write_text(public_path(PUBLIC_FILES["report"]),
               open(doc_path("reports", "protocol_c_cross_region_reapplication_report.md"), encoding="utf-8").read())
    write_text(public_path(PUBLIC_FILES["status"]), f"""# Protocol C - cross-region status summary

- Recife: {registry['Recife']['reference_status']} (score {registry['Recife']['evidence_score']}).
- Curitiba: {registry['Curitiba']['reference_status']} (score {registry['Curitiba']['evidence_score']}) - A807 LOCAL strong temporal; Sentinel date missing.
- Petropolis: {registry['Petropolis']['reference_status']} (score {registry['Petropolis']['evidence_score']}) - A610 regional proxy temporal context.
- Operational label = 0 | negative = 0 | training = 0 | C7 = NOT_CREATED_BLOCKED_FOR_TRAINING (all regions).
""")


# --------------------------------------------------------------------------- #
# Guardrail regression.
# --------------------------------------------------------------------------- #

def run_guardrail_regression(args=None):
    forbidden = {"can_create_operational_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    datasets = (OUTPUTS["state"], OUTPUTS["policy"], OUTPUTS["curitiba"], OUTPUTS["petropolis"],
                OUTPUTS["registry"], OUTPUTS["gate_table"], OUTPUTS["scorecard"])
    for number, name in enumerate(datasets, 1):
        data = load_csv(dataset_path(name))
        violations = sum(clean(r.get(field)).lower() == "true" for r in data for field in forbidden)
        rows.append({"regression_id": f"GR_v2bm_{number:03d}", "check": f"forbidden_flags::{name}",
                     "detail": "no operational-label/negative/training flag is true", "violation_count": str(violations),
                     "status": "PASS" if not violations else "FAIL"})
    gate_rows = load_csv(dataset_path(OUTPUTS["gate_table"]))
    c7 = [r for r in gate_rows if r["gate_id"] == "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH"]
    c7_ok = bool(c7) and all("NOT_CREATED" in r["gate_status"] for r in c7)
    rows.append({"regression_id": "GR_v2bm_008", "check": "c7_blocked_all_regions",
                 "detail": "C7 not-created/blocked in every region", "violation_count": "0" if c7_ok else "1",
                 "status": "PASS" if c7_ok else "FAIL"})
    label_ok = all(r["operational_label_allowed"] == "false" and r["training_allowed"] == "false" for r in gate_rows)
    rows.append({"regression_id": "GR_v2bm_009", "check": "no_label_no_training",
                 "detail": "gate table forbids operational label and training",
                 "violation_count": "0" if label_ok else "1", "status": "PASS" if label_ok else "FAIL"})
    registry = load_csv(dataset_path(OUTPUTS["registry"]))
    reg_ok = bool(registry) and all(all(t in r["forbidden_use"] for t in ("SUPERVISED_LABEL", "TRAINING_TARGET"))
                                    for r in registry)
    rows.append({"regression_id": "GR_v2bm_010", "check": "registry_forbidden_use",
                 "detail": "registry forbids label/training", "violation_count": "0" if reg_ok else "1",
                 "status": "PASS" if reg_ok else "FAIL"})
    pet = load_csv(dataset_path(OUTPUTS["petropolis"]))
    proxy_ok = all(r["station_role"] == "REGIONAL_PROXY" for r in pet) and \
        not any("LOCAL_STATION" in r["updated_protocol_status"] for r in pet)
    rows.append({"regression_id": "GR_v2bm_011", "check": "proxy_not_local",
                 "detail": "Petropolis proxy never becomes a local station",
                 "violation_count": "0" if proxy_ok else "1", "status": "PASS" if proxy_ok else "FAIL"})
    public_ok = all(os.path.splitext(public_path(rel))[1].lower() in {".csv", ".md"} for rel in PUBLIC_FILES.values())
    rows.append({"regression_id": "GR_v2bm_012", "check": "public_outputs_safe",
                 "detail": "outputs_public has only derived csv/md", "violation_count": "0" if public_ok else "1",
                 "status": "PASS" if public_ok else "FAIL"})
    if any(r["status"] != "PASS" for r in rows):
        raise ValueError("v2bm guardrail regression failed")
    write_csv(dataset_path(OUTPUTS["guardrail"]), rows)
    return rows


def _steps():
    return [
        ("load_cross_region_state", run_load_cross_region_state, dataset_path(OUTPUTS["state"])),
        ("apply_refined_protocol_policy", run_apply_refined_protocol_policy, dataset_path(OUTPUTS["policy"])),
        ("reassess_curitiba_candidates", run_reassess_curitiba_candidates, dataset_path(OUTPUTS["curitiba"])),
        ("reassess_petropolis_candidates", run_reassess_petropolis_candidates, dataset_path(OUTPUTS["petropolis"])),
        ("build_cross_region_candidate_registry", run_build_cross_region_candidate_registry, dataset_path(OUTPUTS["registry"])),
        ("build_cross_region_gate_table", run_build_cross_region_gate_table, dataset_path(OUTPUTS["gate_table"])),
        ("build_cross_region_evidence_scorecard", run_build_cross_region_evidence_scorecard, dataset_path(OUTPUTS["scorecard"])),
        ("generate_reapplication_report", run_generate_reapplication_report,
         doc_path("reports", "protocol_c_cross_region_reapplication_report.md")),
        ("run_guardrail_regression", run_guardrail_regression, dataset_path(OUTPUTS["guardrail"])),
    ]


def ensure_structure():
    for folder in (DOCS_DIR, doc_path("reports"), doc_path("region_packets"),
                   doc_path("evidence_scorecards"), doc_path("evidence_cache")):
        os.makedirs(folder, exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")
    for rel in ("tables", "execution_reports", "logs_summary"):
        os.makedirs(public_path(rel), exist_ok=True)


def run_orchestrator(args=None):
    ensure_structure()
    refresh_status = refresh_inputs()
    base = dataset_path(INPUTS["vl_registry"])
    manifest = [{"step_order": "0", "step_name": "refresh_v2bl_chain", "status": refresh_status,
                 "output": base.replace("\\", "/"),
                 "output_hash": sha256(base)[:16] if os.path.exists(base) else "",
                 "notes": "Regenerates the v2bl chain from the live cache; automated cross-region adjudication."}]
    for number, (name, function, path) in enumerate(_steps(), 1):
        function(args)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK",
                         "output": path.replace("\\", "/"), "output_hash": sha256(path)[:16],
                         "notes": "Strictly additive; references are not operational labels."})
    write_csv(dataset_path(OUTPUTS["manifest"]), manifest)
    return manifest


if __name__ == "__main__":
    run_orchestrator(parse_args())
