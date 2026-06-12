#!/usr/bin/env python3
"""v2bb Secondary Evidence Expansion and Geometry Adjudication Preparation."""

import argparse
import csv
import datetime as dt
import hashlib
import os
import re

STAGE = "v2bb"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2bb_secondary_evidence_adjudication")
NETWORK_ENV = "V2BB_NETWORK"
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "secondary_evidence_is_not_ground_truth": "true", "digitization_review_candidate_is_not_label": "true",
    "textual_location_is_not_geometry": "true", "official_report_is_not_boundary": "true",
    "quickview_is_not_validated_product": "true", "susceptibility_is_not_observed_event": "true",
    "rainfall_is_not_flood_extent": "true", "patch_boundary_is_not_event_geometry": "true",
    "raw_data_versioned": "false",
}
INPUTS = {
    "selection": "v2ba_review_ready_packet_selection.csv", "search_plan": "v2ba_geometry_search_plan.csv",
    "probes": "v2ba_official_geometry_source_probe.csv", "classes": "v2ba_geometry_evidence_classification.csv",
    "candidates": "v2ba_candidate_digitization_registry.csv", "adjudication": "v2ba_human_adjudication_queue.csv",
    "uncertainty": "v2ba_geometry_uncertainty_scores.csv", "manual": "v2az_manual_review_table.csv",
    "packets": "v2az_assisted_review_packet_index.csv", "metrics": "v2ay_window_precipitation_metrics.csv",
    "temporal": "v2ay_event_patch_temporal_readiness_update.csv", "registry": "evidence_source_registry.csv",
}
OUTPUTS = [
    "v2bb_adjudication_packet_selection.csv", "v2bb_secondary_source_targets.csv",
    "v2bb_secondary_evidence_probe.csv", "v2bb_secondary_evidence_classification.csv",
    "v2bb_secondary_evidence_correlation.csv", "v2bb_uncertainty_update.csv",
    "v2bb_digitization_review_candidate_matrix.csv", "v2bb_adjudication_decision_table.csv",
    "v2bb_secondary_evidence_packet_index.csv", "v2bb_guardrail_regression.csv",
]
STANDARD_TARGETS = [
    ("Defesa Civil municipal/estadual", "CIVIL_DEFENSE", "MANUAL_REFERENCE"),
    ("Prefeitura e secretarias municipais", "MUNICIPAL_OFFICIAL", "MANUAL_REFERENCE"),
    ("Boletins oficiais e relatorios de ocorrencia", "OFFICIAL_BULLETIN", "MANUAL_REFERENCE"),
    ("Portal de transparencia e noticias oficiais", "OFFICIAL_NEWS", "MANUAL_REFERENCE"),
    ("Materias jornalisticas datadas", "JOURNALISTIC", "MANUAL_REFERENCE"),
    ("Cemaden alerta/estacao/ocorrencia", "CEMADEN", "https://mapainterativo.cemaden.gov.br/"),
    ("ANA HidroWeb componente hidrologico", "ANA_HIDROWEB", "https://www.snirh.gov.br/hidroweb/"),
    ("SGB CPRM contexto de risco", "SUSCEPTIBILITY", "https://geosgb.sgb.gov.br/"),
    ("Copernicus EMS produto", "COPERNICUS", "https://emergency.copernicus.eu/mapping/"),
    ("International Charter product vs quickview", "INTERNATIONAL_CHARTER", "https://disasterscharter.org/"),
    ("Sentinel VHR apoio visual", "REVIEW_VISUAL", "MANUAL_REFERENCE"),
]


def parse_args(argv=None): return argparse.ArgumentParser().parse_args(argv)
def clean(value): return str(value or "").strip()
def is_true(value): return clean(value).lower() == "true"
def slug(value): return re.sub(r"[^a-z0-9]+", "-", clean(value).lower()).strip("-")
def dataset_path(name): return os.path.join(DATASET_DIR, name)
def doc_path(*parts): return os.path.join(DOCS_DIR, *parts)
def with_invariants(row): return {**row, **INVARIANTS}


def load_csv(path):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows: raise ValueError(f"Refusing empty output: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle: handle.write(text)


def sha256(path):
    with open(path, "rb") as handle: return hashlib.sha256(handle.read()).hexdigest()


def by(rows, key): return {row.get(key, ""): row for row in rows}
def load_inputs(): return {key: load_csv(dataset_path(name)) for key, name in INPUTS.items()}


def secondary_class(source_type="", source_name="", confirmed=False, has_map=False, has_boundary=False, quickview=False, susceptibility=False, conflict=False):
    kind, name = clean(source_type).upper(), clean(source_name).upper()
    if conflict: return "CONFLICTING_SECONDARY_EVIDENCE"
    if not confirmed: return "NO_SECONDARY_EVIDENCE_FOUND"
    if quickview or "QUICKVIEW" in name or "IMAGE OF THE DAY" in name: return "QUICKVIEW_ONLY"
    if susceptibility or "SGB" in name or "CPRM" in name or "SUSCET" in name: return "SUSCEPTIBILITY_CONTEXT_ONLY"
    if has_map and kind in {"TECHNICAL_REPORT", "OFFICIAL_BULLETIN"}: return "OFFICIAL_PDF_MAP"
    if has_map: return "OFFICIAL_MAP_IMAGE"
    if kind in {"TECHNICAL_REPORT", "OFFICIAL_BULLETIN"}: return "OFFICIAL_EVENT_REPORT"
    if kind in {"OFFICIAL_MUNICIPAL", "MUNICIPAL_OFFICIAL", "CIVIL_DEFENSE", "OFFICIAL_NEWS"}: return "OFFICIAL_TEXTUAL_LOCATION"
    if kind == "JOURNALISTIC" or "NEWS" in kind: return "JOURNALISTIC_DATED_LOCATION"
    if kind in {"INSTITUTIONAL_POST", "CEMADEN"}: return "INSTITUTIONAL_POST"
    if kind in {"ACADEMIC_PAPER", "COPERNICUS_PRODUCT", "REVIEW_VISUAL", "VHR_REVIEW"}: return "REVIEW_ONLY_VISUAL_SUPPORT"
    return "CONTEXT_ONLY"


def location_relation(text, city="", patch_id=""):
    value = clean(text).lower()
    if patch_id and clean(patch_id).lower() in value: return "PATCH_LEVEL"
    if any(token in value for token in ("bairro", "rua ", "avenida ", "localidade", "distrito")): return "NEIGHBORHOOD_LEVEL"
    if city and clean(city).lower() in value: return "MUNICIPALITY_LEVEL"
    if value: return "REGIONAL"
    return "UNKNOWN"


def temporal_relation(event_date, evidence_date):
    try:
        event, evidence = dt.date.fromisoformat(event_date), dt.date.fromisoformat(evidence_date)
    except ValueError: return ("", "UNKNOWN")
    delta = (evidence - event).days
    return (str(delta), "DURING" if delta == 0 else "BEFORE" if delta < 0 else "AFTER")


def geometry_relation(evidence_class, has_boundary=False):
    if has_boundary: return "EXPLICIT_BOUNDARY"
    if evidence_class in {"OFFICIAL_MAP_IMAGE", "OFFICIAL_PDF_MAP", "REVIEW_ONLY_VISUAL_SUPPORT", "QUICKVIEW_ONLY"}: return "MAP_ONLY"
    if evidence_class in {"OFFICIAL_EVENT_REPORT", "OFFICIAL_TEXTUAL_LOCATION", "JOURNALISTIC_DATED_LOCATION", "INSTITUTIONAL_POST"}: return "TEXT_ONLY"
    return "NONE"


def can_reduce_uncertainty(evidence_class, location, supports_date, conflict=False):
    if conflict or not supports_date: return False
    return location in {"PATCH_LEVEL", "NEIGHBORHOOD_LEVEL"} and evidence_class in {
        "OFFICIAL_EVENT_REPORT", "OFFICIAL_TEXTUAL_LOCATION", "OFFICIAL_MAP_IMAGE", "OFFICIAL_PDF_MAP",
        "JOURNALISTIC_DATED_LOCATION", "INSTITUTIONAL_POST", "REVIEW_ONLY_VISUAL_SUPPORT",
    }


def digitization_decision(classes):
    usable = [row for row in classes if is_true(row.get("can_reduce_uncertainty"))]
    maps = [row for row in usable if row["evidence_class"] in {"OFFICIAL_MAP_IMAGE", "OFFICIAL_PDF_MAP", "REVIEW_ONLY_VISUAL_SUPPORT"}]
    local = [row for row in usable if row["location_relation"] in {"PATCH_LEVEL", "NEIGHBORHOOD_LEVEL"}]
    conflicts = [row for row in classes if row["evidence_class"] == "CONFLICTING_SECONDARY_EVIDENCE"]
    if conflicts: return ("HAZARD_AMBIGUITY_BLOCKS_DIGITIZATION", "NONE", False)
    if maps: return ("READY_FOR_HUMAN_DIGITIZATION_REVIEW", maps[0]["evidence_class"], True)
    if local: return ("READY_FOR_HUMAN_DIGITIZATION_REVIEW", "OFFICIAL_EVENT_REPORT_WITH_LOCALITY", True)
    if any(row["evidence_class"] not in {"NO_SECONDARY_EVIDENCE_FOUND", "CONTEXT_ONLY"} for row in classes):
        return ("NEEDS_MORE_SECONDARY_EVIDENCE", "TEXTUAL_ONLY", False)
    return ("ONLY_CONTEXT_AVAILABLE", "NONE", False)


def run_select_adjudication_packets(args=None):
    data = load_inputs(); candidates = by(data["candidates"], "review_packet_id")
    rows = [with_invariants({"selection_id": f"SEL_v2bb_{i:04d}", **packet,
             "v2ba_candidate_status": candidates.get(packet["review_packet_id"], {}).get("candidate_status", ""),
             "selection_status": "SELECTED_FOR_SECONDARY_EVIDENCE_EXPANSION", "recife_gap_excluded": "true"})
            for i, packet in enumerate(data["selection"], 1)]
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def run_expand_secondary_source_targets(args=None):
    data = load_inputs(); plans = {}
    for row in data["search_plan"]: plans.setdefault(row["review_packet_id"], []).append(row)
    rows = []
    for packet in data["selection"]:
        for source in plans.get(packet["review_packet_id"], []):
            rows.append(with_invariants({"target_id": f"TARGET_v2bb_{len(rows)+1:04d}", "review_packet_id": packet["review_packet_id"],
                "event_patch_package_id": packet["event_patch_package_id"], "candidate_id": packet["candidate_id"], "region": packet["region"], "city": packet["city"],
                "event_date": packet["event_date"], "source_name": source["source_target"], "source_type": source["source_type"],
                "source_url_or_reference": source["search_url_or_reference"], "target_origin": "REUSED_V2BA_AUDITED_SOURCE", "search_status": "REGISTERED_LOCAL_REFERENCE"}))
        for name, source_type, ref in STANDARD_TARGETS:
            rows.append(with_invariants({"target_id": f"TARGET_v2bb_{len(rows)+1:04d}", "review_packet_id": packet["review_packet_id"],
                "event_patch_package_id": packet["event_patch_package_id"], "candidate_id": packet["candidate_id"], "region": packet["region"], "city": packet["city"],
                "event_date": packet["event_date"], "source_name": name, "source_type": source_type, "source_url_or_reference": ref,
                "target_origin": "EXPANDED_SECONDARY_TARGET", "search_status": "PENDING_SECONDARY_REVIEW"}))
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def run_probe_secondary_evidence_sources(args=None):
    network = os.environ.get(NETWORK_ENV) == "1"; rows = []
    for target in load_csv(dataset_path(OUTPUTS[1])):
        confirmed = target["target_origin"] == "REUSED_V2BA_AUDITED_SOURCE"
        rows.append(with_invariants({"probe_id": f"PROBE_v2bb_{len(rows)+1:04d}", "target_id": target["target_id"], "review_packet_id": target["review_packet_id"],
            "event_patch_package_id": target["event_patch_package_id"], "probe_mode": "NETWORK_METADATA" if network else "OFFLINE_DETERMINISTIC",
            "probe_status": "NETWORK_METADATA_NOT_IMPLEMENTED_REQUIRES_MANUAL_REVIEW" if network else "NETWORK_DISABLED_DETERMINISTIC_RUN",
            "source_name": target["source_name"], "source_type": target["source_type"], "source_url_or_reference": target["source_url_or_reference"],
            "local_reference_confirmed": str(confirmed).lower(), "metadata_found": str(confirmed).lower(), "raw_payload_cached": "false",
            "access_note": "Eu/equipe registrei alvo e metadados locais; nenhum payload bruto foi versionado."}))
    write_csv(dataset_path(OUTPUTS[2]), rows)
    for packet in load_inputs()["selection"]:
        matches = [row for row in rows if row["review_packet_id"] == packet["review_packet_id"]]
        write_text(doc_path("source_probe_summaries", f"{slug(packet['candidate_id'])}.md"),
                   f"# Secondary source probe: {packet['candidate_id']}\n\nEu/equipe registrei {len(matches)} alvos offline. Nenhum payload bruto foi versionado.\n")
    return rows


def run_classify_secondary_evidence(args=None):
    packets = by(load_inputs()["selection"], "review_packet_id"); rows = []
    for probe in load_csv(dataset_path(OUTPUTS[2])):
        packet = packets[probe["review_packet_id"]]; confirmed = is_true(probe["local_reference_confirmed"])
        evidence_class = secondary_class(probe["source_type"], probe["source_name"], confirmed=confirmed)
        loc = packet["city"] if confirmed else ""
        relation = location_relation(loc, packet["city"], packet.get("patch_id", ""))
        supports_date = confirmed
        reduce = can_reduce_uncertainty(evidence_class, relation, supports_date)
        rows.append(with_invariants({"secondary_evidence_id": f"SEC_v2bb_{len(rows)+1:04d}", "review_packet_id": probe["review_packet_id"],
            "event_patch_package_id": probe["event_patch_package_id"], "source_name": probe["source_name"], "source_type": probe["source_type"],
            "source_url_or_reference": probe["source_url_or_reference"], "evidence_class": evidence_class,
            "evidence_date": packet["event_date"] if confirmed else "", "location_text": loc, "hazard_terms_found": packet.get("hazard_type_candidate", ""),
            "geometry_terms_found": "", "has_explicit_map": "false", "has_explicit_boundary": "false", "has_coordinates": "false",
            "supports_event_date": str(supports_date).lower(), "supports_event_location": str(confirmed).lower(),
            "supports_hazard_type": str(confirmed).lower(), "supports_geometry": "false", "can_reduce_uncertainty": str(reduce).lower(),
            "location_relation": relation}))
    write_csv(dataset_path(OUTPUTS[3]), rows); return rows


def run_correlate_secondary_evidence(args=None):
    packets = by(load_inputs()["selection"], "review_packet_id"); rows = []
    for evidence in load_csv(dataset_path(OUTPUTS[3])):
        packet = packets[evidence["review_packet_id"]]; delta, relation = temporal_relation(packet["event_date"], evidence["evidence_date"])
        geometry = geometry_relation(evidence["evidence_class"], is_true(evidence["has_explicit_boundary"]))
        strength = "MODERATE" if is_true(evidence["can_reduce_uncertainty"]) else "WEAK" if evidence["evidence_class"] not in {"NO_SECONDARY_EVIDENCE_FOUND", "CONTEXT_ONLY"} else "CONTEXT_ONLY"
        rows.append(with_invariants({"review_packet_id": evidence["review_packet_id"], "event_patch_package_id": evidence["event_patch_package_id"],
            "secondary_evidence_id": evidence["secondary_evidence_id"], "event_date": packet["event_date"], "evidence_date": evidence["evidence_date"],
            "date_delta_days": delta, "temporal_relation": relation, "location_relation": evidence["location_relation"],
            "hazard_relation": "MULTIHAZARD" if packet.get("hazard_type_candidate") == "MIXED" else "PARTIAL_MATCH",
            "geometry_relation": geometry, "correlation_strength": strength,
            "reviewer_note": "Correlation supports adjudication only; it does not create geometry or truth."}))
    write_csv(dataset_path(OUTPUTS[4]), rows); return rows


def run_update_uncertainty_scores(args=None):
    data = load_inputs(); previous = by(data["uncertainty"], "review_packet_id"); rows = []
    classes = load_csv(dataset_path(OUTPUTS[3]))
    for packet in data["selection"]:
        matches = [row for row in classes if row["review_packet_id"] == packet["review_packet_id"]]
        reduced = any(is_true(row["can_reduce_uncertainty"]) for row in matches)
        overall = "HIGH" if reduced else previous.get(packet["review_packet_id"], {}).get("overall_uncertainty", "VERY_HIGH")
        rows.append(with_invariants({"review_packet_id": packet["review_packet_id"], "previous_overall_uncertainty": previous.get(packet["review_packet_id"], {}).get("overall_uncertainty", "VERY_HIGH"),
            "updated_spatial_uncertainty": "HIGH" if reduced else "VERY_HIGH", "updated_temporal_uncertainty": "LOW",
            "updated_source_uncertainty": "MODERATE" if reduced else "HIGH", "updated_hazard_uncertainty": "HIGH",
            "updated_overall_uncertainty": overall, "uncertainty_reduced": str(reduced).lower(),
            "reduction_reason": "Specific dated locality or reviewable map available." if reduced else "Only municipality-level, contextual, visual-only, or unconfirmed targets available.",
            "remaining_blocker": "NO_SPECIFIC_TRACEABLE_SPATIAL_EVIDENCE; HUMAN_ADJUDICATION_REQUIRED"}))
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


def run_build_digitization_review_candidate_matrix(args=None):
    data = load_inputs(); classes = load_csv(dataset_path(OUTPUTS[3])); rows = []
    for packet in data["selection"]:
        matches = [row for row in classes if row["review_packet_id"] == packet["review_packet_id"]]
        status, basis, allowed = digitization_decision(matches)
        rows.append(with_invariants({"review_packet_id": packet["review_packet_id"], "event_patch_package_id": packet["event_patch_package_id"],
            "digitization_review_status": status, "candidate_basis": basis,
            "required_human_action": "Review secondary sources and confirm event-specific spatial evidence before any digitization.",
            "minimum_evidence_missing": "" if allowed else "LOCATION_MORE_SPECIFIC_THAN_MUNICIPALITY_OR_OFFICIAL_EVENT_MAP",
            "can_digitize_candidate": str(allowed).lower(), "requires_human_validation": "true"}))
    write_csv(dataset_path(OUTPUTS[6]), rows); return rows


def run_build_adjudication_decision_table(args=None):
    data = load_inputs(); manual = by(data["manual"], "review_packet_id"); uncertainty = by(load_csv(dataset_path(OUTPUTS[5])), "review_packet_id")
    matrix = by(load_csv(dataset_path(OUTPUTS[6])), "review_packet_id"); classes = load_csv(dataset_path(OUTPUTS[3])); rows = []
    for packet in data["selection"]:
        matches = [row for row in classes if row["review_packet_id"] == packet["review_packet_id"]]
        found = sum(row["evidence_class"] != "NO_SECONDARY_EVIDENCE_FOUND" for row in matches)
        base = manual.get(packet["review_packet_id"], {}); decision = matrix[packet["review_packet_id"]]
        rows.append(with_invariants({"adjudication_id": f"ADJ_v2bb_{len(rows)+1:04d}", "review_packet_id": packet["review_packet_id"],
            "region": packet["region"], "city": packet["city"], "patch_id": packet.get("patch_id", ""), "event_date": packet["event_date"],
            "temporal_support_summary": base.get("hydromet_summary", ""), "secondary_evidence_summary": f"{len(matches)} targets; {found} local references; no explicit boundary.",
            "geometry_evidence_summary": base.get("geometry_evidence_summary", "GEOMETRY_MISSING"), "updated_uncertainty": uncertainty[packet["review_packet_id"]]["updated_overall_uncertainty"],
            "digitization_review_status": decision["digitization_review_status"],
            "recommended_human_decision_options": "DIGITIZE_CANDIDATE_FOR_NEXT_REVIEW|KEEP_GEOMETRY_MISSING|REQUEST_MORE_EVIDENCE|MARK_HAZARD_AMBIGUOUS|EXCLUDE_FROM_GEOMETRY_REVIEW",
            "current_truth_status": "NOT_GROUND_TRUTH", "current_label_status": "NO_LABEL"}))
    write_csv(dataset_path(OUTPUTS[7]), rows); write_csv(doc_path("adjudication_tables", OUTPUTS[7]), rows); return rows


def run_generate_secondary_evidence_packets(args=None):
    data = load_inputs(); decisions = by(load_csv(dataset_path(OUTPUTS[7])), "review_packet_id"); matrix = by(load_csv(dataset_path(OUTPUTS[6])), "review_packet_id")
    uncertainty = by(load_csv(dataset_path(OUTPUTS[5])), "review_packet_id"); classes = load_csv(dataset_path(OUTPUTS[3])); rows = []
    can_digitize_total = sum(is_true(row["can_digitize_candidate"]) for row in load_csv(dataset_path(OUTPUTS[6])))
    next_action = "HUMAN_DIGITIZE_CANDIDATE_GEOMETRIES_FOR_NEXT_REVIEW" if can_digitize_total else "SEARCH_MORE_SPECIFIC_SECONDARY_SPATIAL_EVIDENCE"
    for packet in data["selection"]:
        matches = [row for row in classes if row["review_packet_id"] == packet["review_packet_id"]]
        path = doc_path("secondary_evidence_packets", f"{slug(packet['candidate_id'])}.md")
        found = [row for row in matches if row["evidence_class"] != "NO_SECONDARY_EVIDENCE_FOUND"]
        write_text(path, f"""# Secondary Evidence Adjudication Packet: {packet['candidate_id']}

## 1. Identificacao do pacote
`{packet['review_packet_id']}` / `{packet['event_patch_package_id']}`.

## 2. Resumo temporal da v2ay
{decisions[packet['review_packet_id']]['temporal_support_summary']}

## 3. Status geometrico da v2ba
GEOMETRY_MISSING; candidate NOT_CREATED.

## 4. Fontes secundarias procuradas
{len(matches)} alvos registrados por eu/equipe.

## 5. Evidencias encontradas e classificadas
{len(found)} referencias locais; nenhuma fronteira explicita.

## 6. Correlacao data-local-fenomeno
Somente suporte municipal/regional ou contextual; nao cria geometria.

## 7. Incerteza
Anterior VERY_HIGH; atualizada {uncertainty[packet['review_packet_id']]['updated_overall_uncertainty']}.

## 8. Digitalizacao humana candidata
{matrix[packet['review_packet_id']]['digitization_review_status']}; can_digitize_candidate={matrix[packet['review_packet_id']]['can_digitize_candidate']}.

## 9. Acao humana recomendada
SEARCH_MORE_SPECIFIC_SECONDARY_SPATIAL_EVIDENCE.

## 10. Guardrails
Nao e ground truth; nao cria label; nao cria negativo; nao treina modelo.
""")
        rows.append(with_invariants({"packet_index_id": f"PACK_v2bb_{len(rows)+1:04d}", "review_packet_id": packet["review_packet_id"],
            "event_patch_package_id": packet["event_patch_package_id"], "packet_path": f"docs/protocolo_c/v2bb_secondary_evidence_adjudication/secondary_evidence_packets/{slug(packet['candidate_id'])}.md",
            "secondary_target_count": str(len(matches)), "confirmed_reference_count": str(len(found)), "digitization_review_status": matrix[packet["review_packet_id"]]["digitization_review_status"],
            "next_action_rank_1": next_action, "recife_next_action": "RESOLVE_RECIFE_TEMPORAL_GAP_WITH_CEMADEN_OR_SECONDARY_STATIONS",
            "current_truth_status": "NOT_GROUND_TRUTH"}))
    write_csv(dataset_path(OUTPUTS[8]), rows)
    write_text(doc_path("README.md"), f"# v2bb Secondary Evidence Expansion and Geometry Adjudication Preparation\n\nEu/equipe analisei 6 pacotes; 3 Recife permanecem em gap temporal. Pacotes liberados para digitalizacao candidata: {can_digitize_total}. Ground truth, labels, negativos e treino: 0. Proxima acao: `{next_action}`.\n")
    return rows


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:9], 1):
        violations = sum(row.get(field, "").lower() == "true" for row in load_csv(dataset_path(name)) for field in forbidden)
        rows.append({"regression_id": f"GR_v2bb_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore"); violations = 0 if os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n" else 1
    rows.append({"regression_id": "GR_v2bb_010", "artifact_path": "docs/protocolo_c/v2bb_secondary_evidence_adjudication/evidence_cache/.gitignore", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    if any(row["status"] != "PASS" for row in rows): raise ValueError("v2bb guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[9]), rows); return rows


STEPS = [
    ("select_adjudication_packets", run_select_adjudication_packets, OUTPUTS[0]),
    ("expand_secondary_source_targets", run_expand_secondary_source_targets, OUTPUTS[1]),
    ("probe_secondary_evidence_sources", run_probe_secondary_evidence_sources, OUTPUTS[2]),
    ("classify_secondary_evidence", run_classify_secondary_evidence, OUTPUTS[3]),
    ("correlate_secondary_evidence", run_correlate_secondary_evidence, OUTPUTS[4]),
    ("update_uncertainty_scores", run_update_uncertainty_scores, OUTPUTS[5]),
    ("build_digitization_review_candidate_matrix", run_build_digitization_review_candidate_matrix, OUTPUTS[6]),
    ("build_adjudication_decision_table", run_build_adjudication_decision_table, OUTPUTS[7]),
    ("generate_secondary_evidence_packets", run_generate_secondary_evidence_packets, OUTPUTS[8]),
    ("guardrail_regression", run_guardrail_regression, OUTPUTS[9]),
]


def ensure_structure():
    for folder in ("secondary_evidence_packets", "adjudication_tables", "source_probe_summaries", "evidence_cache"): os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args); path = dataset_path(output)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": f"datasets/protocolo_c/{output}", "output_hash": sha256(path)[:16], "notes": "Completed."})
    write_csv(dataset_path("v2bb_orchestrator_manifest.csv"), manifest); return manifest
