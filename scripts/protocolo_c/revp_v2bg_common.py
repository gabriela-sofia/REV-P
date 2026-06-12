#!/usr/bin/env python3
"""v2bg Recife hydro-geomorphic grounding pack, review-only and fail-closed."""

import argparse
import csv
import hashlib
import os

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2bg_recife_hydro_geomorphic_grounding_pack")
CHARTER_URL = "https://disasterscharter.org/activations/landslide-in-brazil-activation-758-"
INMET_CATALOG_URL = "https://portal.inmet.gov.br/paginas/catalogoaut"
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "charter_product_is_not_ground_truth": "true", "published_map_is_not_confirmed_vector": "true",
    "landslide_scar_is_not_flood_extent": "true", "olinda_product_is_not_recife_evidence": "true",
    "copernicus_not_found_is_not_positive_evidence": "true",
    "regional_proxy_does_not_replace_a301": "true", "raw_data_versioned": "false",
}
OUTPUTS = [
    "v2bg_charter_activation_758_registry.csv", "v2bg_recife_source_priority_registry.csv",
    "v2bg_recife_evidence_separation_matrix.csv", "v2bg_recife_protocol_gate_status.csv",
    "v2bg_recife_manual_product_access_tasks.csv", "v2bg_guardrail_regression.csv",
]


def parse_args(argv=None): return argparse.ArgumentParser().parse_args(argv)
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle: handle.write(text)


def sha256(path):
    with open(path, "rb") as handle: return hashlib.sha256(handle.read()).hexdigest()


def charter_row():
    return with_invariants({
        "charter_activation_id": "758", "activation_title": "Landslide in Brazil",
        "event_type": "LANDSLIDE_FLOOD", "activation_date": "2022-05-30", "requestor": "CENAD",
        "product_count_reported": "51", "product_title": "Landslides after effects in Recife/PE - Brazil",
        "product_date": "2022-06-02", "product_area": "Recife/PE",
        "product_type": "OFFICIAL_DISASTER_MAPPING_PRODUCT", "product_url_or_reference": CHARTER_URL,
        "source_class": "OPERATIONAL_MAPPING", "evidence_axis": "OCCURRENCE|SPATIALITY",
        "hazard_scope": "LANDSLIDE_FLOOD", "product_status": "PRODUCT_PAGE_CONFIRMED",
        "hazard_terms": "LANDSLIDE|FLOOD|MULTIHAZARD", "source_strength": "STRONG_OFFICIAL_DISASTER_MAPPING_SOURCE",
        "geometry_status": "PUBLISHED_MAP_GEOMETRY_VISIBLE_OR_PRODUCT_LISTED",
        "vector_status": "VECTOR_NOT_CONFIRMED", "crs_status": "UNKNOWN", "redistribution_status": "UNKNOWN",
        "geometry_visible_or_implied": "PUBLISHED_MAP_GEOMETRY_VISIBLE_OR_PRODUCT_LISTED",
        "vector_download_confirmed": "false", "crs_confirmed": "false", "redistribution_confirmed": "false",
        "use_in_protocol": "CANDIDATE_GEOMETRY_REVIEW_SOURCE",
        "limitation": "Vector, CRS, redistribution terms, and exact mapped feature type are not confirmed; not automatic flood extent.",
    })


def run_build_charter_registry(args=None):
    rows = [charter_row()]
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def source_row(name, source_class, priority, temporal, spatial, geometry, context, status, limitation, link):
    return with_invariants({
        "source_name": name, "source_class": source_class, "priority": priority,
        "source_status": status, "can_resolve_temporal_gap": temporal,
        "can_support_spatial_anchor": spatial, "can_support_geometry_candidate": geometry,
        "context_only": context, "source_url_or_reference": link, "limitation": limitation,
    })


def run_build_source_priority_registry(args=None):
    rows = [
        source_row("International Charter Activation 758", "OPERATIONAL_MAPPING", "P0", "false", "true", "true", "false", "PRODUCT_PAGE_CONFIRMED", "Vector/CRS/license not confirmed; hazard includes landslide/flood, not automatic flood extent.", CHARTER_URL),
        source_row("CEMADEN Recife", "OFFICIAL_HYDROMETEOROLOGICAL_SERIES", "P1", "true", "false", "false", "false", "ACQUISITION_PENDING", "A valid local series for the May 2022 event window is not yet acquired.", "https://www.gov.br/cemaden/"),
        source_row("APAC Pernambuco", "OFFICIAL_HYDROMETEOROLOGICAL_SERIES", "P1", "true", "false", "false", "false", "ACQUISITION_PENDING", "A valid Recife/RMR series for the event window is not yet acquired.", "https://www.apac.pe.gov.br/"),
        source_row("ANA HidroWeb", "OFFICIAL_HYDROLOGICAL_SERIES", "P1", "true", "false", "false", "false", "ACQUISITION_PENDING", "Station relevance and event-window completeness require review.", "https://www.snirh.gov.br/hidroweb/"),
        source_row("PE3D/eSIG/Defesa Civil Recife", "OFFICIAL_SPATIAL_CONTEXT", "P1", "false", "true", "true", "false", "REVIEW_PENDING", "Requires source-specific geometry, CRS, license, and event linkage review.", ""),
        source_row("Copernicus EMS Recife May 2022", "ON_DEMAND_MAPPING_SEARCH", "P3", "false", "false", "false", "true", "NOT_FOUND", "No Recife/Pernambuco May 2022 product found; absence is not positive evidence.", "https://mapping.emergency.copernicus.eu/"),
        source_row("INMET A357 Palmares", "REGIONAL_HYDROMETEOROLOGICAL_PROXY", "P3", "false", "false", "false", "true", "REGIONAL_PROXY", "Regional proxy only; does not replace Recife A301 or resolve the local temporal gap.", INMET_CATALOG_URL),
        source_row("INMET A328 Surubim", "REGIONAL_HYDROMETEOROLOGICAL_PROXY", "P3", "false", "false", "false", "true", "REGIONAL_PROXY", "Regional proxy only; does not replace Recife A301 or resolve the local temporal gap.", INMET_CATALOG_URL),
        source_row("INMET A320 Joao Pessoa", "REGIONAL_HYDROMETEOROLOGICAL_PROXY", "P3", "false", "false", "false", "true", "REGIONAL_PROXY", "Regional proxy only; does not replace Recife A301 or resolve the local temporal gap.", INMET_CATALOG_URL),
    ]
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def evidence_row(item, axis, evidence_class, occurrence, spatial, geometry, temporal, context, blocker, limitation):
    return with_invariants({
        "evidence_item": item, "evidence_axis": axis, "evidence_class": evidence_class,
        "supports_occurrence": occurrence, "supports_spatial_anchor": spatial,
        "supports_geometry_candidate": geometry, "supports_temporal_gap_resolution": temporal,
        "context_only": context, "promotion_blocker": blocker, "limitation": limitation,
    })


def run_build_evidence_separation_matrix(args=None):
    rows = [
        evidence_row("Charter 758 Recife product", "OCCURRENCE", "OFFICIAL_DISASTER_MAPPING_PRODUCT", "true", "true", "true", "false", "false", "VECTOR_CRS_AND_PRODUCT_ACCESS_NOT_CONFIRMED", "Confirms an official mapping product, not final truth."),
        evidence_row("Charter 758 Recife product", "SPATIALITY", "OFFICIAL_DISASTER_MAPPING_PRODUCT", "true", "true", "true", "false", "false", "VECTOR_CRS_AND_PRODUCT_ACCESS_NOT_CONFIRMED", "Published map geometry is review evidence; vector and feature semantics remain unconfirmed."),
        evidence_row("Charter 758 Recife product", "HAZARD_SEMANTICS", "MULTIHAZARD_PRODUCT", "true", "false", "false", "false", "true", "LANDSLIDE_FLOOD_FEATURES_REQUIRE_SEPARATION", "Landslide scar must not be interpreted as flood extent."),
        evidence_row("Charter 758 Olinda products", "SPATIALITY", "OUT_OF_RECIFE_SCOPE_PRODUCT", "false", "false", "false", "false", "true", "OLINDA_NOT_TRANSFERABLE_TO_RECIFE", "Olinda products dated 2022-06-03 are not Recife evidence."),
        evidence_row("Copernicus EMS Recife May 2022 search", "SOURCE_SEARCH", "NOT_FOUND_SEARCH_RESULT", "false", "false", "false", "false", "true", "NOT_FOUND_IS_NOT_POSITIVE_EVIDENCE", "Search result cannot support occurrence or geometry."),
        evidence_row("INMET regional proxy stations", "TEMPORALITY", "REGIONAL_PROXY", "false", "false", "false", "false", "true", "LOCAL_SPATIAL_REPRESENTATIVENESS_NOT_PROVEN", "A357/A328/A320 do not replace A301."),
    ]
    write_csv(dataset_path(OUTPUTS[2]), rows); return rows


def run_build_protocol_gate_status(args=None):
    recife = load_csv(dataset_path("v2az_recife_gap_review_queue.csv"))
    if not recife: raise ValueError("Missing v2az Recife gap queue")
    rows = []
    for packet in recife:
        may_2022 = packet["candidate_id"] == "REC_2022_05_24_30"
        gates = [
            ("C0_PROVENANCE", "PASS" if may_2022 else "PENDING", "CHARTER_758_REGISTERED" if may_2022 else "EVENT_SPECIFIC_OFFICIAL_MAPPING_NOT_REGISTERED"),
            ("C1_TEMPORALITY", "PENDING", "APAC_CEMADEN_ANA_VALID_SERIES_PENDING"),
            ("C2_VALID_SERIES_OR_STATION", "BLOCKED", "A301_NO_USABLE_EVENT_WINDOW_RECORDS_AND_PROXY_NOT_REPLACEMENT"),
            ("C3_SPATIAL_ANCHOR", "PENDING_REVIEW" if may_2022 else "BLOCKED", "CHARTER_758_PRODUCT_REQUIRES_GEOMETRY_REVIEW" if may_2022 else "EVENT_SPECIFIC_SPATIAL_ANCHOR_MISSING"),
            ("C4_CANDIDATE_GEOMETRY", "PENDING" if may_2022 else "BLOCKED", "VECTOR_CRS_PRODUCT_ACCESS_NOT_CONFIRMED" if may_2022 else "GEOMETRY_MISSING"),
            ("C5_HUMAN_REVIEW", "PENDING" if may_2022 else "BLOCKED", "PRODUCT_ACCESS_AND_FEATURE_TYPE_REVIEW_PENDING" if may_2022 else "REVIEW_INPUTS_INCOMPLETE"),
            ("C6_CANDIDATE_REFERENCE", "BLOCKED", "C1_C2_C4_C5_NOT_PASSED"),
            ("C7_FINAL_GROUND_TRUTH", "BLOCKED", "FINAL_GROUND_TRUTH_PROHIBITED_WITH_UNRESOLVED_GATES"),
        ]
        for gate, status, blocker in gates:
            rows.append(with_invariants({
                "review_packet_id": packet["review_packet_id"], "candidate_id": packet["candidate_id"],
                "event_date": packet["event_date"], "gate": gate, "status": status, "blocker_or_basis": blocker,
                "charter_758_applicable": str(may_2022).lower(),
                "next_action_rank_1": "INVENTORY_CHARTER_758_RECIFE_PRODUCTS_AND_ACCESS_VECTOR_CRS" if may_2022 else "ACQUIRE_CEMADEN_APAC_RECIFE_EVENT_TEMPORAL_SERIES",
                "next_action_rank_2": "ACQUIRE_CEMADEN_APAC_RECIFE_MAY_2022_TEMPORAL_SERIES" if may_2022 else "BUILD_EVENT_SPECIFIC_RECIFE_SPATIAL_ANCHOR",
                "next_action_rank_3": "BUILD_RECIFE_SPATIAL_ANCHOR_REGISTRY_FROM_CHARTER_PE3D_ESIG_DEFESA_CIVIL" if may_2022 else "REVIEW_OFFICIAL_EVENT_SOURCES",
            }))
    write_csv(dataset_path(OUTPUTS[3]), rows); return rows


def run_build_manual_product_access_tasks(args=None):
    tasks = [
        ("OPEN_ACTIVATION_PAGE", "Open Activation 758 and verify activation/product metadata.", "LIGHTWEIGHT_ACTIVATION_METADATA_MANIFEST", "ACTIVATION_PAGE_REVIEW_PENDING"),
        ("INVENTORY_RECIFE_PRODUCTS", "Inventory every Recife-specific product without transferring Olinda products.", "RECIFE_PRODUCT_INVENTORY_CSV", "RECIFE_PRODUCT_INVENTORY_MISSING"),
        ("VERIFY_VECTOR_OR_VAP_DOWNLOAD", "Verify whether a VAP or vector download exists.", "VECTOR_ACCESS_STATUS", "VECTOR_NOT_CONFIRMED"),
        ("VERIFY_CRS", "Inspect downloadable product metadata and record CRS only if explicit.", "CRS_STATUS", "CRS_UNKNOWN"),
        ("VERIFY_LICENSE_REDISTRIBUTION", "Record product access, license, and redistribution terms.", "LICENSE_REDISTRIBUTION_STATUS", "REDISTRIBUTION_UNKNOWN"),
        ("CLASSIFY_MAPPED_FEATURE", "Determine whether mapped features are landslide scars, flood extent, or another product type.", "FEATURE_TYPE_REVIEW", "FEATURE_TYPE_NOT_CONFIRMED"),
        ("CACHE_RAW_LOCAL_ONLY", "If access is permitted, save raw/heavy payload only in git-ignored evidence_cache.", "LOCAL_CACHE_MANIFEST", "RAW_PAYLOAD_MUST_NOT_BE_VERSIONED"),
        ("GENERATE_LIGHTWEIGHT_DERIVED_MANIFEST", "Generate a lightweight manifest containing provenance and review status only.", "LIGHTWEIGHT_DERIVED_MANIFEST", "DERIVED_MANIFEST_MISSING"),
    ]
    rows = [with_invariants({
        "task_id": f"TASK_v2bg_{i:03d}", "source_name": "International Charter Activation 758",
        "activation_id": "758", "product_title": "Landslides after effects in Recife/PE - Brazil",
        "product_date": "2022-06-02", "required_action": action, "expected_artifact": artifact,
        "blocker": blocker, "priority": "P0",
    }) for i, (_, action, artifact, blocker) in enumerate(tasks, 1)]
    write_csv(dataset_path(OUTPUTS[4]), rows); return rows


def run_generate_grounding_pack(args=None):
    readme = """# v2bg Recife Hydro-Geomorphic Grounding Pack

## International Charter Activation 758 as Recife P0 Evidence

A Activation 758 entra como a fonte espacial/observacional mais forte identificada para Recife ate o momento. Ela nao resolve sozinha o ground truth, porque ainda preciso confirmar vetor, CRS, licenca e tipo exato de feicao mapeada. Mesmo assim, ela muda Recife de simples temporal gap para pacote com evidencia oficial de mapeamento de desastre candidata a revisao geometrica.

O produto Recife de 2022-06-02 sustenta ocorrencia e uma ancora espacial em revisao. Ele nao e extensao de inundacao automaticamente, nao transfere produtos de Olinda para Recife e nao cria ground truth, label, negativo ou treino.

## Prioridades

1. `INVENTORY_CHARTER_758_RECIFE_PRODUCTS_AND_ACCESS_VECTOR_CRS`
2. `ACQUIRE_CEMADEN_APAC_RECIFE_MAY_2022_TEMPORAL_SERIES`
3. `BUILD_RECIFE_SPATIAL_ANCHOR_REGISTRY_FROM_CHARTER_PE3D_ESIG_DEFESA_CIVIL`
"""
    write_text(doc_path("README.md"), readme)
    write_text(doc_path("source_packets", "charter_758_recife.md"), f"# Charter 758 Recife review packet\n\nOfficial reference: {CHARTER_URL}\n\nReview-only. Vector, CRS, redistribution, and exact feature type remain unconfirmed.\n")
    return [{"status": "OK"}]


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:5], 1):
        violations = sum(row.get(field, "").lower() == "true" for row in load_csv(dataset_path(name)) for field in forbidden)
        rows.append({"regression_id": f"GR_v2bg_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    checks = [
        ("CHARTER_NOT_FLOOD_EXTENT", all(r["landslide_scar_is_not_flood_extent"] == "true" for r in load_csv(dataset_path(OUTPUTS[0])))),
        ("OLINDA_NOT_TRANSFERRED", all(r["supports_spatial_anchor"] == "false" for r in load_csv(dataset_path(OUTPUTS[2])) if "Olinda" in r["evidence_item"])),
        ("COPERNICUS_NOT_POSITIVE", all(r["supports_occurrence"] == "false" for r in load_csv(dataset_path(OUTPUTS[2])) if "Copernicus" in r["evidence_item"])),
        ("PROXIES_NOT_A301_REPLACEMENT", all(r["can_resolve_temporal_gap"] == "false" for r in load_csv(dataset_path(OUTPUTS[1])) if r["source_status"] == "REGIONAL_PROXY")),
    ]
    for offset, (name, passed) in enumerate(checks, 6):
        rows.append({"regression_id": f"GR_v2bg_{offset:03d}", "artifact_path": name, "violation_count": "0" if passed else "1", "status": "PASS" if passed else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore")
    passed = os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n"
    rows.append({"regression_id": "GR_v2bg_010", "artifact_path": "docs/protocolo_c/v2bg_recife_hydro_geomorphic_grounding_pack/evidence_cache/.gitignore", "violation_count": "0" if passed else "1", "status": "PASS" if passed else "FAIL"})
    if any(row["status"] != "PASS" for row in rows): raise ValueError("v2bg guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


STEPS = [
    ("build_charter_activation_758_registry", run_build_charter_registry, OUTPUTS[0]),
    ("build_recife_source_priority_registry", run_build_source_priority_registry, OUTPUTS[1]),
    ("build_recife_evidence_separation_matrix", run_build_evidence_separation_matrix, OUTPUTS[2]),
    ("build_recife_protocol_gate_status", run_build_protocol_gate_status, OUTPUTS[3]),
    ("build_recife_manual_product_access_tasks", run_build_manual_product_access_tasks, OUTPUTS[4]),
    ("generate_recife_grounding_pack", run_generate_grounding_pack, None),
    ("guardrail_regression", run_guardrail_regression, OUTPUTS[5]),
]


def ensure_structure():
    for folder in ("source_packets", "evidence_cache"): os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args)
        path = dataset_path(output) if output else doc_path("README.md")
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": path.replace("\\", "/"), "output_hash": sha256(path)[:16], "notes": "Review-only Recife grounding evidence; no truth promotion."})
    write_csv(dataset_path("v2bg_orchestrator_manifest.csv"), manifest); return manifest
