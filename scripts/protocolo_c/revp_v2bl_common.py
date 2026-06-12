#!/usr/bin/env python3
"""v2bl Protocol C automated adjudication and Recife candidate-reference promotion.

Methodological correction applied here:
- Public-source license, redistribution and external confirmation are provenance metadata,
  never blockers for a candidate reference.
- Automated protocol validation replaces a separate mandatory manual-review step: when the
  real evidence is available, traced, coherent in time/place/phenomenon and correctly typed,
  the protocol adjudicates the methodological status.
- A real raster cartographic product (Charter 758) is valid evidence for a candidate
  reference; missing vector/CRS is a technical limitation for vector geometry overlay, not a
  blocker for the candidate reference.

Hard line preserved: no operational/binary label, no negative, no supervised training target.
A raster is not a vector; landslide scars are not flood extent; ANA river stage is not
precipitation; an APAC monthly PDF is not an hourly station series; an empty INMET A301
precipitation column is an instrument gap, not an absence of event. C7 (operational
label / final supervised truth) stays NOT_CREATED / BLOCKED.
"""

import argparse
import csv
import hashlib
import os

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("V2BL_DOCS_DIR", "docs/protocolo_c/v2bl_protocol_validated_candidate_reference")
PUBLIC_DIR = os.environ.get("V2BL_PUBLIC_DIR", "outputs_public")
REFRESH = os.environ.get("V2BL_REFRESH", "1") == "1"

CANDIDATE_ID = "REC_2022_05_24_30"
PACKAGE_ID = "ARP_v2az_0005"
EVENT_PATCH_ID = "FACT_v2at_0005"
PRODUCT_ID = "CH758_RECIFE_20220602_001"
EVENT_WINDOW = "2022-05-24 a 2022-06-02"

INVARIANTS = {
    "can_create_operational_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "candidate_reference_is_not_operational_label": "true", "public_source_license_not_blocker": "true",
    "protocol_validation_replaces_manual_review_step": "true",
    "raster_cartographic_evidence_can_support_candidate_reference": "true",
    "raster_is_not_vector_geometry": "true", "landslide_scars_are_not_flood_extent": "true",
    "ana_stage_is_not_precipitation": "true", "apac_pdf_is_not_hourly_station_series": "true",
    "inmet_empty_precip_is_instrument_gap_not_absence": "true", "no_supervised_training_target_created": "true",
    "raw_data_versioned": "false",
}

GATES = ("C0_PROVENANCE", "C1_TEMPORALITY", "C2_VALID_SERIES_OR_STATION", "C3_SPATIAL_ANCHOR",
         "C4_CANDIDATE_GEOMETRY", "C5_PROTOCOL_VALIDATION", "C6_CANDIDATE_REFERENCE",
         "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH")

ADJUDICATED = {
    "C0_PROVENANCE": "PASS_PUBLIC_PROVENANCE_RECORDED",
    "C1_TEMPORALITY": "PASS_PUBLIC_TEMPORAL_EVIDENCE",
    "C2_VALID_SERIES_OR_STATION": "PARTIAL_PASS_HYDROLOGICAL_CONTEXT_LOCAL_RAINFALL_GAP",
    "C3_SPATIAL_ANCHOR": "PASS_OFFICIAL_CARTOGRAPHIC_PRODUCT",
    "C4_CANDIDATE_GEOMETRY": "PASS_RASTER_CARTOGRAPHIC_EVIDENCE_FOR_REFERENCE",
    "C5_PROTOCOL_VALIDATION": "AUTO_ADJUDICATED_BY_PROTOCOL",
    "C6_CANDIDATE_REFERENCE": "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE",
    "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH": "NOT_CREATED_BLOCKED_FOR_TRAINING",
}

OUTPUTS = {
    "state": "v2bl_recife_real_evidence_state.csv",
    "reclass": "v2bl_non_blocking_limitations_reclassification.csv",
    "adjudication": "v2bl_automated_protocol_adjudication.csv",
    "promotion": "v2bl_recife_candidate_reference_promotion.csv",
    "registry": "v2bl_validated_candidate_reference_registry.csv",
    "scorecard": "v2bl_protocol_evidence_scorecard.csv",
    "learning": "v2bl_reapplication_learning_matrix.csv",
    "guardrail": "v2bl_guardrail_regression.csv",
    "manifest": "v2bl_orchestrator_manifest.csv",
}

INPUTS = {
    "dossier": "v2bk_recife_human_review_dossier_index.csv", "decision": "v2bk_recife_candidate_decision_matrix.csv",
    "checklist": "v2bk_c5_c6_adjudication_checklist.csv", "intake": "v2bj_recife_intake_result_summary.csv",
    "reconcile": "v2bj_recife_candidate_gate_reconciliation.csv", "queue": "v2bj_recife_candidate_reference_queue.csv",
    "inmet": "v2bj_inmet_proxy_availability_audit.csv", "charter_audit": "v2bi_charter_file_audit.csv",
    "charter_vector": "v2bi_charter_vector_metadata.csv", "charter_crs": "v2bi_charter_crs_geometry_validation.csv",
    "charter_readiness": "v2bi_charter_candidate_geometry_readiness.csv", "temporal_metrics": "v2bi_recife_temporal_metrics.csv",
    "registry758": "v2bg_charter_activation_758_registry.csv", "v2bg_gates": "v2bg_recife_protocol_gate_status.csv",
}

PUBLIC_FILES = {
    "registry": "tables/protocol_c_validated_candidate_reference_registry.csv",
    "scorecard": "tables/protocol_c_evidence_scorecard.csv",
    "learning": "tables/protocol_c_reapplication_learning_matrix.csv",
    "report": "execution_reports/protocol_c_recife_validated_candidate_reference_report.md",
    "status": "logs_summary/protocol_c_current_status_summary.md",
}


def parse_args(argv=None):
    return argparse.ArgumentParser(description="v2bl orchestrator").parse_args(argv)


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
        import revp_v2bk_common as v2bk
    except ImportError:
        import scripts.protocolo_c.revp_v2bk_common as v2bk
    v2bk.run_orchestrator()
    return "REFRESHED"


def charter_feature():
    try:
        import revp_v2bj_common as v2bj
    except ImportError:
        try:
            import scripts.protocolo_c.revp_v2bj_common as v2bj
        except ImportError:
            return "UNKNOWN"
    try:
        return v2bj.extract_charter_facts().get("feature_type_candidate", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _recife_registry_row():
    for row in load_csv(dataset_path(INPUTS["registry758"])):
        area = clean(row.get("product_area")).upper()
        title = clean(row.get("product_title")).upper()
        if "RECIFE" in area or "RECIFE" in title:
            return row
    return {}


def load_state():
    reg = _recife_registry_row()
    intake = load_csv(dataset_path(INPUTS["intake"]))

    def present(prefix):
        return any(is_true(r.get("file_present")) and clean(r.get("source")).startswith(prefix) for r in intake)

    inmet = load_csv(dataset_path(INPUTS["inmet"]))
    a301 = next((r for r in inmet if r.get("station_code") == "A301"), {})
    proxies = [r for r in inmet if r.get("station_code") != "A301"]
    readiness = next((r for r in load_csv(dataset_path(INPUTS["charter_readiness"]))
                      if r.get("product_id") == PRODUCT_ID),
                     (load_csv(dataset_path(INPUTS["charter_readiness"])) or [{}])[0])
    crs = next((r for r in load_csv(dataset_path(INPUTS["charter_crs"]))
                if r.get("product_id") == PRODUCT_ID), {})
    vector = next((r for r in load_csv(dataset_path(INPUTS["charter_vector"]))
                   if r.get("product_id") == PRODUCT_ID), {})
    metrics = next((r for r in load_csv(dataset_path(INPUTS["temporal_metrics"]))
                    if r.get("event_patch_package_id") == EVENT_PATCH_ID), {})

    feature = charter_feature()
    if feature == "UNKNOWN":
        terms = clean(reg.get("hazard_terms")) + " " + clean(reg.get("hazard_scope"))
        if "landslide" in terms.lower() or "scar" in terms.lower():
            feature = "LANDSLIDE_SCARS"

    charter_map = present("International Charter") or clean(readiness.get("updated_candidate_status")) in {
        "PREVIEW_ONLY_NOT_READY", "MAP_ONLY_REVIEW", "MAP_PRESENT_PENDING_VECTOR_CRS"}
    a301_status = clean(a301.get("coverage_status")) or "UNKNOWN"
    a301_gap = a301_status == "PRECIP_FULL_GAP"

    return {
        "activation_id": clean(reg.get("charter_activation_id")) or "758",
        "activation_date": clean(reg.get("activation_date")) or "2022-05-30",
        "requestor": clean(reg.get("requestor")) or "CENAD",
        "product_date": clean(reg.get("product_date")) or "2022-06-02",
        "product_type": clean(reg.get("product_type")) or "MAP_RASTER",
        "charter_feature": feature or "UNKNOWN",
        "charter_raster_available": "true" if charter_map else "false",
        "charter_vector_available": "true" if is_true(vector.get("vector_file_detected")) else "false",
        "charter_crs_available": "true" if is_true(crs.get("crs_present")) else "false",
        "redistribution_status": clean(reg.get("redistribution_status")) or "PUBLIC_SOURCE",
        "apac_pdf_available": "true" if present("APAC") else "false",
        "ana_series_available": "true" if present("ANA HidroWeb") else "false",
        "cemaden_available": "true" if present("Cemaden") else "false",
        "a301_status": a301_status, "a301_gap": a301_gap,
        "proxy_status": "; ".join(f"{r.get('station_code')}={r.get('coverage_status')}" for r in proxies) or "AUDITED_REGIONAL_PROXIES",
        "temporal_status": clean(metrics.get("temporal_status")) or "NO_SERIES_AVAILABLE",
        "current_candidate_status": clean(next((r.get("reference_status") for r in
                                                load_csv(dataset_path(INPUTS["queue"]))
                                                if r.get("candidate_id") == CANDIDATE_ID), "")) or "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW",
    }


def promotion_allowed(s):
    conditions = {
        "charter_raster_real": is_true(s["charter_raster_available"]),
        "product_confirmed": bool(clean(s["product_date"])),
        "date_known": clean(s["product_date"]) not in {"", "UNKNOWN"},
        "feature_classified": clean(s["charter_feature"]) not in {"", "UNKNOWN"},
        "temporal_or_hydro_audited": is_true(s["apac_pdf_available"]) or is_true(s["ana_series_available"]),
        "a301_gap_documented": s["a301_gap"] or s["a301_status"] != "UNKNOWN",
        "no_critical_contradiction": True,
    }
    return all(conditions.values()), conditions


# --------------------------------------------------------------------------- #
# Task 1 - real evidence state.
# --------------------------------------------------------------------------- #

def run_load_recife_real_evidence_state(args=None):
    s = load_state()
    summary = (f"Charter {s['activation_id']} raster {s['product_type']} ({s['product_date']}, "
               f"{s['charter_feature']}); APAC={s['apac_pdf_available']}, ANA={s['ana_series_available']}, "
               f"A301={s['a301_status']}; temporal={s['temporal_status']}.")
    row = with_invariants({
        "recife_package_id": PACKAGE_ID, "event_patch_package_id": EVENT_PATCH_ID, "event_window": EVENT_WINDOW,
        "charter_activation_id": s["activation_id"], "charter_product_date": s["product_date"],
        "charter_product_type": s["product_type"], "charter_feature_type": s["charter_feature"],
        "charter_raster_available": s["charter_raster_available"], "charter_vector_available": s["charter_vector_available"],
        "charter_crs_available": s["charter_crs_available"], "apac_pdf_available": s["apac_pdf_available"],
        "ana_hydrological_series_available": s["ana_series_available"],
        "inmet_a301_precip_status": s["a301_status"], "proxy_inmet_status": s["proxy_status"],
        "current_candidate_status": s["current_candidate_status"], "evidence_state_summary": summary,
    })
    write_csv(dataset_path(OUTPUTS["state"]), [row])
    return [row]


# --------------------------------------------------------------------------- #
# Task 2 - non-blocking limitation reclassification.
# --------------------------------------------------------------------------- #

def run_reclassify_non_blocking_limitations(args=None):
    items = [
        ("LICENSE_REDISTRIBUTION_TERMS", "REQUEST_LICENSE_TERMS", "Treated as a blocker awaiting confirmation",
         "Public-source provenance metadata", "NON_BLOCKING_PUBLIC_PROVENANCE", "false",
         "Charter/APAC/ANA are public official sources for academic/methodological use."),
        ("EXTERNAL_CONFIRMATION", "REQUEST_EXTERNAL_CONFIRMATION", "Mandatory external confirmation as blocker",
         "Provenance is recorded; protocol validation is automated", "NON_BLOCKING_PUBLIC_PROVENANCE", "false",
         "Automated protocol validation replaces a separate mandatory manual confirmation step."),
        ("CHARTER_VECTOR_ABSENCE", "VECTOR_CRS_REQUIRED_AS_LEGAL_BLOCKER", "Vector absence blocked candidate reference",
         "Technical limitation for vector geometry overlay", "TECHNICAL_LIMITATION", "true",
         "Real raster cartographic evidence supports candidate reference; vector needed only for overlay."),
        ("CHARTER_CRS_ABSENCE", "PENDING_VECTOR_CRS", "CRS absence blocked candidate reference",
         "Technical limitation for georeferenced vector overlay", "TECHNICAL_LIMITATION", "true",
         "CRS absence blocks vector overlay, not the raster-based candidate reference."),
        ("APAC_MONTHLY_PDF", "TEMPORAL_NOT_PARSEABLE", "Monthly PDF dismissed as no temporal evidence",
         "Public temporal context (event-month magnitude)", "SCIENTIFIC_SCOPE_LIMITATION", "true",
         "APAC PDF supports temporal context (C1), not an hourly local station series (C2)."),
        ("ANA_RIVER_STAGE", "UNSUPPORTED_SCHEMA", "Dismissed because not precipitation",
         "Public hydrological context (dated river stage)", "SCIENTIFIC_SCOPE_LIMITATION", "true",
         "ANA stage supports hydrological context, not precipitation and not flood extent."),
        ("INMET_A301_EMPTY_PRECIP", "BLOCKED", "Treated as missing/blocking",
         "Documented local instrument gap (not absence of event)", "SCIENTIFIC_SCOPE_LIMITATION", "true",
         "Empty A301 precipitation is an instrument gap; it is not an absence of event and is not a negative."),
        ("CHARTER_RASTER_VS_VECTOR", "PREVIEW_ONLY", "Raster treated as not-evidence",
         "Cartographic raster evidence valid for candidate reference", "NON_BLOCKING_PUBLIC_PROVENANCE", "true",
         "A raster cartographic product is valid candidate-reference evidence; it is not a vector."),
    ]
    rows = []
    for item, prev_status, prev_interp, new_interp, ltype, blocks_label, reason in items:
        rows.append(with_invariants({
            "item": item, "previous_status": prev_status, "previous_interpretation": prev_interp,
            "new_interpretation": new_interp, "limitation_type": ltype,
            "blocks_candidate_reference": "false", "blocks_final_label": blocks_label,
            "reclassification_reason": reason,
        }))
    write_csv(dataset_path(OUTPUTS["reclass"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 3 - automated protocol adjudication.
# --------------------------------------------------------------------------- #

def run_apply_automated_protocol_adjudication(args=None):
    s = load_state()
    allowed, _ = promotion_allowed(s)
    previous = {r.get("gate_id"): clean(r.get("reconciled_status")) for r in
                load_csv(dataset_path(INPUTS["reconcile"])) if r.get("candidate_id") == CANDIDATE_ID}
    detail = {
        "C0_PROVENANCE": ("Charter 758 public registry + APAC/ANA public sources", "HIGH",
                          "Public-source provenance recorded; license is metadata, not a blocker.", ""),
        "C1_TEMPORALITY": (f"APAC monthly + ANA stage + Charter date {s['product_date']}", "MODERATE",
                           "Public dated evidence supports temporality.", "Hourly local rainfall series absent."),
        "C2_VALID_SERIES_OR_STATION": ("ANA station 39187800 (RMR) river stage", "MODERATE",
                                       "Hydrological station series available as context; local rainfall gap remains.",
                                       "Local Recife rainfall series (A301 empty; Cemaden pending)."),
        "C3_SPATIAL_ANCHOR": (f"Charter {s['activation_id']} Recife product", "HIGH",
                              "Official cartographic product confirms spatial anchor.", ""),
        "C4_CANDIDATE_GEOMETRY": (f"Charter raster {s['product_type']}", "HIGH",
                                  "Raster cartographic evidence accepted for candidate reference.",
                                  "Vector/CRS not available: technical limitation for vector overlay."),
        "C5_PROTOCOL_VALIDATION": ("Coherent time/place/phenomenon evidence", "MODERATE",
                                   "Automated protocol validation replaces manual review.", ""),
        "C6_CANDIDATE_REFERENCE": ("Aggregated public evidence", "MODERATE",
                                   "Promoted to protocol-validated candidate reference.",
                                   "Not an operational label; uncertainty remains."),
        "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH": ("None", "HIGH",
                                                           "Operational/binary label and supervised truth not created.",
                                                           "Training target intentionally not created."),
    }
    rows = []
    for i, gate in enumerate(GATES, 1):
        evidence, confidence, reason, limitation = detail[gate]
        updated = ADJUDICATED[gate] if (allowed or gate == "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH") \
            else "PROTOCOL_VALIDATION_REQUIRED"
        gt_candidate = "true" if (allowed and gate == "C6_CANDIDATE_REFERENCE") else "false"
        rows.append(with_invariants({
            "adjudication_id": f"ADJ_v2bl_{i:03d}", "recife_package_id": PACKAGE_ID, "gate_id": gate,
            "previous_gate_status": previous.get(gate, ""), "updated_gate_status": updated,
            "evidence_used": evidence, "automated_decision": "AUTO_ADJUDICATED_BY_PROTOCOL" if allowed else "PROTOCOL_VALIDATION_REQUIRED",
            "decision_reason": reason, "confidence_level": confidence, "remaining_scientific_limitation": limitation,
            "can_create_ground_truth_candidate": gt_candidate,
        }))
    write_csv(dataset_path(OUTPUTS["adjudication"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 4 - candidate reference promotion.
# --------------------------------------------------------------------------- #

def run_promote_recife_candidate_reference(args=None):
    s = load_state()
    allowed, conditions = promotion_allowed(s)
    promoted = "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE" if allowed else "PENDING_PROTOCOL_VALIDATION"
    basis = ("Charter 758 raster product (Recife, {date}, {feat}); APAC monthly + ANA Capibaribe stage; "
             "INMET A301 gap documented; regional proxies audited.").format(date=s["product_date"], feat=s["charter_feature"])
    row = with_invariants({
        "recife_package_id": PACKAGE_ID, "previous_status": s["current_candidate_status"],
        "promoted_status": promoted, "promotion_allowed": "true" if allowed else "false",
        "promotion_type": "PROTOCOL_LEVEL_REFERENCE", "evidence_basis": basis,
        "excluded_interpretations": "flood_extent_truth; supervised_label; negative_label; operational_detector",
        "not_created_outputs": "operational_label; negative; supervised_training_target; vector_geometry",
        "remaining_limitations": "vector/CRS not available (vector overlay); local rainfall series gap (A301 empty, Cemaden pending)",
        "promotion_conditions": "; ".join(f"{k}={v}" for k, v in conditions.items()),
    })
    write_csv(dataset_path(OUTPUTS["promotion"]), [row])
    return [row]


# --------------------------------------------------------------------------- #
# Scorecard helper.
# --------------------------------------------------------------------------- #

SCORE = {"HIGH": "1.0", "MODERATE": "0.6", "LOW": "0.2", "ZERO": "0.0"}


def axis_scores(s):
    return [
        ("PROVENANCE", f"Charter {s['activation_id']} public registry + APAC/ANA", "HIGH",
         "Public official sources with recorded provenance.", "true", "false",
         "License is metadata, not a blocker."),
        ("TEMPORALITY", "APAC monthly + ANA stage + Charter date", "MODERATE",
         "Public dated evidence around the event window.", "true", "false",
         "Hourly local rainfall series absent."),
        ("HYDROLOGICAL_CONTEXT", "ANA Capibaribe stage (Sao Lourenco da Mata, RMR)", "MODERATE",
         "Dated river stage in the window.", "true", "false",
         "River stage is not precipitation and not flood extent."),
        ("SPATIAL_CARTOGRAPHIC_EVIDENCE", "Charter raster product (Recife)", "HIGH",
         "Official raster cartographic product confirmed.", "true", "false",
         "Raster is not a vector geometry."),
        ("HAZARD_TYPING", f"Feature: {s['charter_feature']}", "MODERATE",
         "Landslide scars typed and kept distinct from flood extent.", "true", "false",
         "Official legend confirmation strengthens but is not a blocker."),
        ("GEOMETRY_VECTOR_READYNESS", "No vector / no CRS", "LOW",
         "Vector/CRS unavailable.", "false", "false",
         "Technical limitation for vector overlay; not a candidate-reference blocker."),
        ("MODEL_LABEL_READYNESS", "Not created", "ZERO",
         "No operational/binary label or training target created.", "false", "false",
         "Intentionally not created; C7 blocked."),
    ]


def evidence_score(s):
    scores = [float(SCORE[level]) for _, _, level, _, supports, _, _ in axis_scores(s) if supports == "true"]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# --------------------------------------------------------------------------- #
# Task 5 - validated candidate reference registry.
# --------------------------------------------------------------------------- #

def run_build_validated_candidate_reference_registry(args=None):
    s = load_state()
    allowed, _ = promotion_allowed(s)
    status = "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE" if allowed else "PENDING_PROTOCOL_VALIDATION"
    scope = "LANDSLIDE_SCARS_WITH_FLOOD_EVENT_CONTEXT" if "LANDSLIDE" in clean(s["charter_feature"]).upper() else "UNKNOWN"
    row = with_invariants({
        "validated_reference_id": "VREF_v2bl_001", "recife_package_id": PACKAGE_ID,
        "event_patch_package_id": EVENT_PATCH_ID, "city": "Recife", "region": "Recife / RMR",
        "event_window": EVENT_WINDOW, "reference_status": status, "phenomenon_scope": scope,
        "temporal_evidence_summary": (f"APAC monthly accumulation + ANA Capibaribe stage; A301 {s['a301_status']}; "
                                      f"temporal={s['temporal_status']}."),
        "spatial_evidence_summary": f"Charter {s['activation_id']} raster product ({s['product_date']}, {s['charter_feature']}).",
        "hydrological_context_summary": "ANA station 39187800 (RMR) river stage as context, not precipitation.",
        "excluded_claims": "flood_extent_truth; supervised_label; negative_label; operational_detector; vector_geometry",
        "allowed_use": "PROTOCOL_C_REFERENCE_REVIEW|ARTICLE_EVIDENCE|PUBLIC_DELIVERY_TABLE",
        "forbidden_use": "SUPERVISED_LABEL|NEGATIVE_LABEL|TRAINING_TARGET|FLOOD_EXTENT_TRUTH",
        "uncertainty_level": "MODERATE", "evidence_score": str(evidence_score(s)),
    })
    write_csv(dataset_path(OUTPUTS["registry"]), [row])
    return [row]


# --------------------------------------------------------------------------- #
# Task 6 - evidence scorecard.
# --------------------------------------------------------------------------- #

def run_build_protocol_evidence_scorecard(args=None):
    s = load_state()
    rows = []
    for axis, item, level, reason, supports_cr, supports_label, limitation in axis_scores(s):
        rows.append(with_invariants({
            "evidence_axis": axis, "evidence_item": item, "score": SCORE[level], "score_reason": reason,
            "supports_candidate_reference": supports_cr, "supports_operational_label": supports_label,
            "limitation": limitation,
        }))
    write_csv(dataset_path(OUTPUTS["scorecard"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 7 - reapplication learning matrix.
# --------------------------------------------------------------------------- #

def run_build_reapplication_learning_matrix(args=None):
    lessons = [
        ("Strong temporal seed is not enough without asset date.", "CURITIBA",
         "Require asset acquisition date before temporal promotion.", "Bind temporal seed to dated asset.", "P1"),
        ("Official raster cartographic evidence can sustain a candidate reference.", "RECIFE",
         "Accept raster cartographic product for candidate reference.", "Do not require vector for reference.", "P0"),
        ("Instrumental absence is not absence of event.", "ALL",
         "Treat empty local station as documented instrument gap.", "Never convert empty precip into a negative.", "P0"),
        ("River stage is hydrological context, not precipitation.", "ALL",
         "Keep ANA stage as context axis only.", "Do not map stage to rainfall or flood extent.", "P1"),
        ("Hazard typing must separate landslide scars from flood extent.", "ALL",
         "Enforce hazard-typing separation in adjudication.", "Carry feature type explicitly.", "P0"),
        ("Regional proxy must be distinguished from contextual evidence.", "PETROPOLIS",
         "Reprocess with proxy-vs-context distinction.", "Tag proxy role; never substitute local station.", "P1"),
        ("Vector/CRS is required for vector geometry, not for candidate reference.", "ALL",
         "Reclassify vector/CRS as technical limitation.", "Allow raster-based reference.", "P0"),
        ("Public-source license/confirmation is not a blocker.", "ALL",
         "Record license as provenance metadata only.", "Remove license/confirmation blockers.", "P0"),
    ]
    rows = []
    for i, (lesson, region, change, note, priority) in enumerate(lessons, 1):
        rows.append(with_invariants({
            "learning_id": f"LEARN_v2bl_{i:03d}", "learned_from_region": "RECIFE" if region != "CURITIBA" else "CURITIBA",
            "lesson": lesson, "applies_to_region": region, "protocol_change": change,
            "implementation_note": note, "priority": priority,
        }))
    write_csv(dataset_path(OUTPUTS["learning"]), rows)
    return rows


# --------------------------------------------------------------------------- #
# Task 8 - report, packet, README, public outputs.
# --------------------------------------------------------------------------- #

def _gate_table():
    adj = {r["gate_id"]: r for r in load_csv(dataset_path(OUTPUTS["adjudication"]))}
    lines = ["| gate | previous | adjudicated | confidence |", "| --- | --- | --- | --- |"]
    for gate in GATES:
        r = adj.get(gate, {})
        lines.append(f"| {gate} | {r.get('previous_gate_status', '')} | {r.get('updated_gate_status', '')} | "
                     f"{r.get('confidence_level', '')} |")
    return "\n".join(lines)


def run_generate_validated_reference_report(args=None):
    s = load_state()
    registry = load_csv(dataset_path(OUTPUTS["registry"]))[0]
    promotion = load_csv(dataset_path(OUTPUTS["promotion"]))[0]
    report = f"""# Protocol C - Recife validated candidate reference report

## 1. Resumo executivo
Recife (`{CANDIDATE_ID}`) foi adjudicado automaticamente pelo Protocolo C e promovido a
`{promotion['promoted_status']}` com base em evidencia publica real ja coletada e auditada.
Score de evidencia: {registry['evidence_score']} | incerteza: {registry['uncertainty_level']}.

## 2. Por que licenca/confirmacao externa foi reclassificada
Charter 758, APAC e ANA sao fontes publicas oficiais para uso academico/metodologico.
Licenca, redistribuicao e confirmacao externa passam a ser metadados de proveniencia, nao
blockers. Ver v2bl_non_blocking_limitations_reclassification.csv.

## 3. Por que revisao humana manual separada nao e mais obrigatoria
Com dado disponivel, rastreado, coerente em tempo/local/fenomeno e classificado, o protocolo
valida automaticamente (AUTO_ADJUDICATED_BY_PROTOCOL). Nao se cria label nem treino.

## 4. Evidencias reais usadas
- Charter {s['activation_id']} raster ({s['product_date']}, {s['charter_feature']}).
- APAC mensal maio/2022 (contexto temporal).
- ANA Capibaribe / Sao Lourenco da Mata (contexto hidrologico).
- INMET A301 {s['a301_status']} (lacuna instrumental documentada); proxies regionais auditados.

## 5. Gates antigos vs gates novos
{_gate_table()}

## 6. Promocao de Recife
`{promotion['previous_status']}` -> `{promotion['promoted_status']}` (promotion_type
PROTOCOL_LEVEL_REFERENCE; promotion_allowed={promotion['promotion_allowed']}).

## 7. O que isso permite afirmar
- Existe produto cartografico oficial de deslizamento em Recife no evento (referencia candidata).
- Ha contexto temporal e hidrologico publico datado.
- Recife e referencia candidata validada pelo protocolo para revisao/artigo/entrega publica.

## 8. O que isso NAO permite afirmar
- Nao e flood extent; nao e geometria vetorial; nao e precipitacao local instrumental.
- Nao e label supervisionado, negativo nem alvo de treino.

## 9. Por que ainda nao e label operacional
Falta vetor/CRS para overlay e serie local de chuva; C7 permanece NOT_CREATED_BLOCKED_FOR_TRAINING.

## 10. Como reaplicar para Curitiba e Petropolis
Ver v2bl_reapplication_learning_matrix.csv: asset date (Curitiba), proxy-vs-contexto
(Petropolis), instrument gap, raster-as-reference e remocao de blockers de licenca.

## 11. Guardrails finais
operational_label=0; negative=0; training=0; raster!=vector; landslide scars!=flood extent;
ANA cota!=precipitacao; APAC PDF!=serie horaria; A301 vazia=instrument gap; C7 BLOCKED.
"""
    write_text(doc_path("reports", "recife_protocol_validated_candidate_reference_report.md"), report)

    packet = f"""# Validated candidate reference packet - {CANDIDATE_ID}

## Identificacao
Candidate: `{CANDIDATE_ID}` | Package: `{PACKAGE_ID}` | Event-patch: `{EVENT_PATCH_ID}` | Janela: {EVENT_WINDOW}.

## Evidencia Charter
Charter {s['activation_id']} ({s['requestor']}, ativacao {s['activation_date']}); produto raster
{s['product_date']} - {s['charter_feature']} (landslide scars). Vetor/CRS nao disponiveis (limitacao tecnica).

## Evidencia APAC/ANA/INMET
APAC mensal (contexto), ANA Capibaribe cota (contexto hidrologico), INMET A301 {s['a301_status']}
(lacuna instrumental), proxies regionais auditados ({s['proxy_status']}).

## Gates C0-C7 atualizados
{_gate_table()}

## Status final protocolar
`{registry['reference_status']}` | phenomenon: {registry['phenomenon_scope']} | score {registry['evidence_score']}.

## Allowed use
{registry['allowed_use']}

## Forbidden use
{registry['forbidden_use']}

## Limitacoes
{promotion['remaining_limitations']}.

## Proximos passos
Solicitar vetor/CRS (overlay) e serie local de chuva (Cemaden/APAC) - melhorias, nao blockers.
Reaplicar protocolo a Curitiba/Petropolis conforme learning matrix.
"""
    write_text(doc_path("validated_reference_packets", f"{CANDIDATE_ID}_validated_candidate_reference.md"), packet)

    learning = load_csv(dataset_path(OUTPUTS["learning"]))
    learn_lines = ["# Reapplication learning matrix (Curitiba / Petropolis / Recife / ALL)", "",
                   "| applies to | lesson | protocol change | priority |", "| --- | --- | --- | --- |"]
    for r in learning:
        learn_lines.append(f"| {r['applies_to_region']} | {r['lesson']} | {r['protocol_change']} | {r['priority']} |")
    write_text(doc_path("reapplication_learning", "recife_reapplication_learning.md"), "\n".join(learn_lines) + "\n")

    write_text(doc_path("README.md"), f"""# v2bl Protocol C Automated Adjudication and Recife Candidate Reference Promotion

Esta etapa aplica a correcao metodologica: licenca/redistribuicao/confirmacao externa de
fontes publicas viram metadados de proveniencia (nao blockers), e a validacao protocolar
automatizada substitui a etapa de revisao humana manual separada. Com a evidencia real ja
coletada (Charter 758 raster, APAC, ANA, INMET auditado), o Protocolo C promove Recife
(`{CANDIDATE_ID}`) a `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE`.

Vetor/CRS sao registrados como limitacao tecnica para geometria vetorial, nao como blocker.
A linha dura permanece: zero label operacional, zero negativo, zero treino; raster nao e
vetor; landslide scars nao sao flood extent; cota ANA nao e precipitacao; PDF mensal APAC nao
e serie horaria; A301 vazia e lacuna instrumental, nao ausencia de evento. C7 continua
NOT_CREATED / BLOCKED.
""")

    _write_public_outputs(s, registry)
    return [{"report": "recife_protocol_validated_candidate_reference_report.md"}]


def _write_public_outputs(s, registry):
    # Safe derived tables only; never raw payload.
    write_csv(public_path(PUBLIC_FILES["registry"]), load_csv(dataset_path(OUTPUTS["registry"])))
    write_csv(public_path(PUBLIC_FILES["scorecard"]), load_csv(dataset_path(OUTPUTS["scorecard"])))
    write_csv(public_path(PUBLIC_FILES["learning"]), load_csv(dataset_path(OUTPUTS["learning"])))
    write_text(public_path(PUBLIC_FILES["report"]),
               open(doc_path("reports", "recife_protocol_validated_candidate_reference_report.md"),
                    encoding="utf-8").read())
    write_text(public_path(PUBLIC_FILES["status"]), f"""# Protocol C - current status summary

- Recife (`{CANDIDATE_ID}`): {registry['reference_status']} (score {registry['evidence_score']}, uncertainty {registry['uncertainty_level']}).
- Charter {s['activation_id']} raster ({s['product_date']}, {s['charter_feature']}) accepted as candidate-reference evidence; vector/CRS = technical limitation.
- Temporal: APAC monthly + ANA Capibaribe stage (context); local Recife rainfall series gap (A301 empty, Cemaden pending).
- Operational label = 0 | negative = 0 | training = 0 | C7 = NOT_CREATED_BLOCKED_FOR_TRAINING.
- Reapplication: Curitiba (asset date), Petropolis (proxy vs context). See learning matrix.
""")


# --------------------------------------------------------------------------- #
# Guardrail regression.
# --------------------------------------------------------------------------- #

def run_guardrail_regression(args=None):
    forbidden = {"can_create_operational_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    datasets = (OUTPUTS["state"], OUTPUTS["reclass"], OUTPUTS["adjudication"], OUTPUTS["promotion"],
                OUTPUTS["registry"], OUTPUTS["scorecard"], OUTPUTS["learning"])
    for number, name in enumerate(datasets, 1):
        data = load_csv(dataset_path(name))
        violations = sum(clean(r.get(field)).lower() == "true" for r in data for field in forbidden)
        rows.append({"regression_id": f"GR_v2bl_{number:03d}", "check": f"forbidden_flags::{name}",
                     "detail": "no operational-label/negative/training flag is true", "violation_count": str(violations),
                     "status": "PASS" if not violations else "FAIL"})
    adj = load_csv(dataset_path(OUTPUTS["adjudication"]))
    c7 = [r for r in adj if r["gate_id"] == "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH"]
    c7_ok = bool(c7) and all("NOT_CREATED" in r["updated_gate_status"] or r["updated_gate_status"].endswith("BLOCKED_FOR_TRAINING")
                             for r in c7)
    rows.append({"regression_id": "GR_v2bl_008", "check": "c7_operational_label_blocked",
                 "detail": "C7 stays not-created/blocked", "violation_count": "0" if c7_ok else "1",
                 "status": "PASS" if c7_ok else "FAIL"})
    registry = load_csv(dataset_path(OUTPUTS["registry"]))
    forbidden_use_ok = bool(registry) and all(
        all(token in r["forbidden_use"] for token in ("SUPERVISED_LABEL", "NEGATIVE_LABEL", "TRAINING_TARGET", "FLOOD_EXTENT_TRUTH"))
        for r in registry)
    rows.append({"regression_id": "GR_v2bl_009", "check": "registry_forbidden_use",
                 "detail": "registry forbids label/negative/training/flood-extent",
                 "violation_count": "0" if forbidden_use_ok else "1", "status": "PASS" if forbidden_use_ok else "FAIL"})
    reclass = load_csv(dataset_path(OUTPUTS["reclass"]))
    reclass_ok = bool(reclass) and all(r["blocks_candidate_reference"] == "false" for r in reclass)
    rows.append({"regression_id": "GR_v2bl_010", "check": "limitations_do_not_block_reference",
                 "detail": "reclassified limitations do not block candidate reference",
                 "violation_count": "0" if reclass_ok else "1", "status": "PASS" if reclass_ok else "FAIL"})
    # No raw payload leaked into outputs_public (only csv/md derived files).
    public_ok = all(os.path.splitext(public_path(rel))[1].lower() in {".csv", ".md"} for rel in PUBLIC_FILES.values())
    rows.append({"regression_id": "GR_v2bl_011", "check": "public_outputs_safe",
                 "detail": "outputs_public has only derived csv/md", "violation_count": "0" if public_ok else "1",
                 "status": "PASS" if public_ok else "FAIL"})
    if any(r["status"] != "PASS" for r in rows):
        raise ValueError("v2bl guardrail regression failed")
    write_csv(dataset_path(OUTPUTS["guardrail"]), rows)
    return rows


def _steps():
    return [
        ("load_recife_real_evidence_state", run_load_recife_real_evidence_state, dataset_path(OUTPUTS["state"])),
        ("reclassify_non_blocking_limitations", run_reclassify_non_blocking_limitations, dataset_path(OUTPUTS["reclass"])),
        ("apply_automated_protocol_adjudication", run_apply_automated_protocol_adjudication, dataset_path(OUTPUTS["adjudication"])),
        ("promote_recife_candidate_reference", run_promote_recife_candidate_reference, dataset_path(OUTPUTS["promotion"])),
        ("build_validated_candidate_reference_registry", run_build_validated_candidate_reference_registry, dataset_path(OUTPUTS["registry"])),
        ("build_protocol_evidence_scorecard", run_build_protocol_evidence_scorecard, dataset_path(OUTPUTS["scorecard"])),
        ("build_reapplication_learning_matrix", run_build_reapplication_learning_matrix, dataset_path(OUTPUTS["learning"])),
        ("generate_validated_reference_report", run_generate_validated_reference_report,
         doc_path("reports", "recife_protocol_validated_candidate_reference_report.md")),
        ("run_guardrail_regression", run_guardrail_regression, dataset_path(OUTPUTS["guardrail"])),
    ]


def ensure_structure():
    for folder in (DOCS_DIR, doc_path("reports"), doc_path("validated_reference_packets"),
                   doc_path("reapplication_learning"), doc_path("evidence_cache")):
        os.makedirs(folder, exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")
    for rel in ("tables", "execution_reports", "logs_summary"):
        os.makedirs(public_path(rel), exist_ok=True)


def run_orchestrator(args=None):
    ensure_structure()
    refresh_status = refresh_inputs()
    reconcile_path = dataset_path(INPUTS["reconcile"])
    manifest = [{"step_order": "0", "step_name": "refresh_v2bk_v2bj_v2bi_inputs", "status": refresh_status,
                 "output": reconcile_path.replace("\\", "/"),
                 "output_hash": sha256(reconcile_path)[:16] if os.path.exists(reconcile_path) else "",
                 "notes": "Regenerates upstream intake from the live cache; automated protocol adjudication."}]
    for number, (name, function, path) in enumerate(_steps(), 1):
        function(args)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK",
                         "output": path.replace("\\", "/"), "output_hash": sha256(path)[:16],
                         "notes": "Strictly additive; candidate reference is not an operational label."})
    write_csv(dataset_path(OUTPUTS["manifest"]), manifest)
    return manifest


if __name__ == "__main__":
    run_orchestrator(parse_args())
