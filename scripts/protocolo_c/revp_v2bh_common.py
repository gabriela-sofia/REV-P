#!/usr/bin/env python3
"""v2bh Charter 758 product inventory and geometry access audit."""

import argparse
import csv
import hashlib
import os

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2bh_recife_charter_758_product_audit")
NETWORK_ENABLED = os.environ.get("V2BH_NETWORK", "0") == "1"
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "charter_product_is_not_final_ground_truth": "true", "preview_is_not_vector_geometry": "true",
    "landslide_product_is_not_flood_extent": "true", "olinda_product_is_not_recife": "true",
    "vector_without_review_is_not_label": "true", "crs_missing_blocks_geometry_promotion": "true",
    "temporal_gap_still_requires_apac_cemaden_ana": "true", "raw_data_versioned": "false",
}
INPUTS = {
    "charter": "v2bg_charter_activation_758_registry.csv",
    "priority": "v2bg_recife_source_priority_registry.csv",
    "evidence": "v2bg_recife_evidence_separation_matrix.csv",
    "gates": "v2bg_recife_protocol_gate_status.csv",
    "tasks": "v2bg_recife_manual_product_access_tasks.csv",
    "gap_selection": "v2bg_recife_gap_package_selection.csv",
    "recife_queue": "v2az_recife_gap_review_queue.csv",
    "temporal": "v2ay_event_patch_temporal_readiness_update.csv",
}
OUTPUTS = [
    "v2bh_charter_758_product_inventory.csv", "v2bh_recife_olinda_product_classification.csv",
    "v2bh_product_access_vector_crs_license_audit.csv",
    "v2bh_product_hazard_geometry_type_classification.csv",
    "v2bh_candidate_geometry_source_registry.csv", "v2bh_recife_gate_status_update.csv",
    "v2bh_manual_access_request_tasks.csv", "v2bh_charter_product_review_packet_index.csv",
    "v2bh_guardrail_regression.csv",
]


def parse_args(argv=None): return argparse.ArgumentParser().parse_args(argv)
def dataset_path(name): return os.path.join(DATASET_DIR, name)
def doc_path(*parts): return os.path.join(DOCS_DIR, *parts)
def with_invariants(row): return {**row, **INVARIANTS}
def is_true(value): return str(value or "").strip().lower() == "true"


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


def by(rows, key): return {row.get(key, ""): row for row in rows}
def load_inputs(): return {key: load_csv(dataset_path(name)) for key, name in INPUTS.items()}


def run_load_charter_registry(args=None):
    data = load_inputs()
    if len(data["charter"]) != 1 or data["charter"][0].get("charter_activation_id") != "758":
        raise ValueError("Activation 758 registry is missing or ambiguous")
    if not data["recife_queue"]: raise ValueError("Recife gap review queue is missing")
    return data


def run_inventory_charter_products(args=None):
    charter = run_load_charter_registry()["charter"][0]
    rows = [
        with_invariants({
            "charter_activation_id": "758", "activation_product_count_reported": charter["product_count_reported"],
            "product_id": "CH758_RECIFE_20220602_001", "product_title": charter["product_title"],
            "product_date": charter["product_date"], "product_area_text": charter["product_area"],
            "product_type_raw": charter["product_type"], "product_url_or_reference": charter["product_url_or_reference"],
            "product_sequence_order": "UNKNOWN", "mentioned_hazard_terms": charter["hazard_terms"],
            "mentioned_location_terms": "RECIFE|PERNAMBUCO",
            "product_inventory_status": "CONFIRMED_FROM_REGISTRY", "raw_payload_cached": "false",
            "note": "Recife product confirmed by v2bg registry; product-level download inventory remains incomplete.",
        }),
        with_invariants({
            "charter_activation_id": "758", "activation_product_count_reported": charter["product_count_reported"],
            "product_id": "CH758_OLINDA_20220603_UNTITLED", "product_title": "",
            "product_date": "2022-06-03", "product_area_text": "Olinda/PE",
            "product_type_raw": "UNKNOWN", "product_url_or_reference": charter["product_url_or_reference"],
            "product_sequence_order": "UNKNOWN", "mentioned_hazard_terms": "UNKNOWN",
            "mentioned_location_terms": "OLINDA|PERNAMBUCO",
            "product_inventory_status": "MANUAL_REVIEW_REQUIRED", "raw_payload_cached": "false",
            "note": "Olinda product presence/date is known; title and product-level metadata were not invented.",
        }),
        with_invariants({
            "charter_activation_id": "758", "activation_product_count_reported": charter["product_count_reported"],
            "product_id": "CH758_REMAINING_PRODUCTS_NOT_INVENTORIED", "product_title": "",
            "product_date": "", "product_area_text": "UNKNOWN",
            "product_type_raw": "UNKNOWN", "product_url_or_reference": charter["product_url_or_reference"],
            "product_sequence_order": "UNKNOWN", "mentioned_hazard_terms": "UNKNOWN",
            "mentioned_location_terms": "UNKNOWN",
            "product_inventory_status": "MANUAL_REVIEW_REQUIRED", "raw_payload_cached": "false",
            "note": "The page reports 51 products, but the remaining product-level records are not inventoried offline.",
        }),
    ]
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def municipality_classification(row):
    area = row["product_area_text"].upper()
    if "RECIFE" in area and "OLINDA" in area: return "RECIFE_OLINDA", "true", "", "Both municipalities explicitly named."
    if "RECIFE" in area: return "RECIFE", "true", "", "Recife is explicitly named."
    if "OLINDA" in area: return "OLINDA", "false", "OLINDA_PRODUCT_NOT_TRANSFERABLE_TO_RECIFE", "Olinda is explicitly named without Recife."
    if "PERNAMBUCO" in area or area.endswith("/PE"): return "PERNAMBUCO_REGIONAL", "false", "RECIFE_AREA_NOT_EXPLICIT", "Regional source is context-only unless Recife is explicit."
    return "UNKNOWN", "false", "PRODUCT_AREA_UNKNOWN", "No confirmed municipal scope."


def run_classify_recife_olinda_products(args=None):
    rows = []
    for product in load_csv(dataset_path(OUTPUTS[0])):
        classification, apply, blocker, reason = municipality_classification(product)
        rows.append(with_invariants({
            "product_id": product["product_id"], "product_title": product["product_title"],
            "product_area_text": product["product_area_text"], "municipality_classification": classification,
            "can_apply_to_recife": apply, "transfer_blocker": blocker, "classification_reason": reason,
        }))
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def access_status(vector=False, crs=False, raster=False, map_file=False, preview=False):
    if vector and crs: return "VECTOR_ACCESS_CONFIRMED", ""
    if raster or map_file: return "RASTER_OR_MAP_ONLY", "VECTOR_AND_CRS_NOT_CONFIRMED"
    if preview: return "PREVIEW_ONLY", "PREVIEW_IS_NOT_VECTOR_GEOMETRY"
    if vector and not crs: return "ACCESS_NOT_CONFIRMED", "CRS_NOT_CONFIRMED"
    return "ACCESS_NOT_CONFIRMED", "DOWNLOAD_VECTOR_CRS_LICENSE_NOT_CONFIRMED"


def run_audit_product_access(args=None):
    rows = []
    for product in load_csv(dataset_path(OUTPUTS[0])):
        status, blocker = access_status()
        rows.append(with_invariants({
            "product_id": product["product_id"], "product_title": product["product_title"],
            "product_url_or_reference": product["product_url_or_reference"],
            "download_link_found": "false", "vector_file_found": "false", "raster_file_found": "false",
            "pdf_or_image_found": "false", "preview_only": "false", "crs_confirmed": "false", "crs_value": "",
            "license_or_terms_confirmed": "false", "redistribution_allowed_confirmed": "false",
            "access_status": status, "blocker": blocker, "raw_payload_cached": "false",
            "network_mode": "ENABLED_METADATA_ONLY" if NETWORK_ENABLED else "NETWORK_DISABLED_DETERMINISTIC_RUN",
        }))
    write_csv(dataset_path(OUTPUTS[2]), rows); return rows


def hazard_geometry(product):
    title = product["product_title"].lower()
    if "landslide" in title:
        return "LANDSLIDE", "UNKNOWN", "MODERATE", "false", "true", "true", "Title explicitly identifies landslide after-effects; exact mapped feature type is unconfirmed."
    return "UNKNOWN", "UNKNOWN", "UNKNOWN", "false", "false", "false", "Insufficient product-level metadata."


def run_classify_product_hazard_geometry_type(args=None):
    rows = []
    for product in load_csv(dataset_path(OUTPUTS[0])):
        hazard, geometry, confidence, flood, landslide, multi, reason = hazard_geometry(product)
        rows.append(with_invariants({
            "product_id": product["product_id"], "product_title": product["product_title"],
            "hazard_type_candidate": hazard, "geometry_feature_type_candidate": geometry,
            "hazard_geometry_confidence": confidence, "can_support_flood_truth": flood,
            "can_support_landslide_truth": landslide, "can_support_multihazard_reference": multi,
            "classification_reason": reason,
        }))
    write_csv(dataset_path(OUTPUTS[3]), rows); return rows


def candidate_status(apply_to_recife, access, vector=False, crs=False):
    if not apply_to_recife: return "CONTEXT_ONLY"
    if vector and crs: return "CANDIDATE_GEOMETRY_SOURCE_READY_FOR_HUMAN_REVIEW"
    if access == "PREVIEW_ONLY": return "PREVIEW_ONLY_NOT_READY"
    if access in {"ACCESS_NOT_CONFIRMED", "RASTER_OR_MAP_ONLY"}: return "CANDIDATE_GEOMETRY_SOURCE_PENDING_VECTOR_CRS"
    return "BLOCKED"


def may_packet():
    rows = load_csv(dataset_path(INPUTS["recife_queue"]))
    return next(row for row in rows if row["candidate_id"] == "REC_2022_05_24_30")


def run_build_candidate_geometry_source_registry(args=None):
    classes = by(load_csv(dataset_path(OUTPUTS[1])), "product_id")
    access = by(load_csv(dataset_path(OUTPUTS[2])), "product_id")
    hazards = by(load_csv(dataset_path(OUTPUTS[3])), "product_id")
    packet = may_packet(); rows = []
    for product in load_csv(dataset_path(OUTPUTS[0])):
        cls, acc, hazard = classes[product["product_id"]], access[product["product_id"]], hazards[product["product_id"]]
        applies = is_true(cls["can_apply_to_recife"])
        status = candidate_status(applies, acc["access_status"], is_true(acc["vector_file_found"]), is_true(acc["crs_confirmed"]))
        rows.append(with_invariants({
            "candidate_geometry_source_id": f"CGS_v2bh_{len(rows)+1:03d}", "product_id": product["product_id"],
            "recife_package_id": packet["review_packet_id"], "event_patch_package_id": packet["event_patch_package_id"],
            "source_name": "International Charter Activation 758", "source_strength": "STRONG_OFFICIAL_DISASTER_MAPPING_SOURCE",
            "source_area": product["product_area_text"], "source_date": product["product_date"],
            "hazard_type_candidate": hazard["hazard_type_candidate"],
            "geometry_feature_type_candidate": hazard["geometry_feature_type_candidate"],
            "geometry_access_status": acc["access_status"], "crs_status": "CONFIRMED" if is_true(acc["crs_confirmed"]) else "UNKNOWN",
            "license_status": "CONFIRMED" if is_true(acc["license_or_terms_confirmed"]) else "UNKNOWN",
            "candidate_status": status,
            "required_human_action": "HUMAN_REVIEW_CHARTER_RECIFE_CANDIDATE_GEOMETRY" if status.endswith("READY_FOR_HUMAN_REVIEW") else "REQUEST_OR_MANUALLY_ACCESS_CHARTER_PRODUCT_VECTOR_CRS",
        }))
    write_csv(dataset_path(OUTPUTS[4]), rows); return rows


def updated_gate_status(gate, may_2022=True):
    if not may_2022: return None
    return {
        "C0_PROVENANCE": ("PASS", "Activation 758 and Recife product are registered.", ""),
        "C1_TEMPORALITY": ("PENDING", "Charter product does not resolve the hydrometeorological series gap.", "APAC_CEMADEN_ANA_VALID_SERIES_PENDING"),
        "C2_VALID_SERIES_OR_STATION": ("BLOCKED", "A301 remains unusable and regional proxies do not replace it.", "VALID_LOCAL_SERIES_OR_STATION_MISSING"),
        "C3_SPATIAL_ANCHOR": ("PASS", "Recife-specific official disaster mapping product is confirmed.", "PRODUCT_GEOMETRY_REVIEW_PENDING"),
        "C4_CANDIDATE_GEOMETRY": ("PENDING_VECTOR_CRS", "Product exists but vector and CRS are not confirmed.", "VECTOR_CRS_PRODUCT_ACCESS_NOT_CONFIRMED"),
        "C5_HUMAN_REVIEW": ("PENDING", "Product access and feature-type review remain pending.", "HUMAN_PRODUCT_REVIEW_PENDING"),
        "C6_CANDIDATE_REFERENCE": ("BLOCKED", "C1, C2, C4, and C5 are unresolved.", "UPSTREAM_GATES_UNRESOLVED"),
        "C7_FINAL_GROUND_TRUTH": ("BLOCKED", "Final ground truth is prohibited in v2bh.", "FINAL_GROUND_TRUTH_BLOCKED"),
    }.get(gate)


def run_update_recife_gate_status(args=None):
    rows = []
    package_by_candidate = by(load_csv(dataset_path(INPUTS["recife_queue"])), "candidate_id")
    for previous in load_csv(dataset_path(INPUTS["gates"])):
        may_2022 = previous["candidate_id"] == "REC_2022_05_24_30"
        update = updated_gate_status(previous["gate"], may_2022)
        status, reason, blocker = update if update else (previous["status"], "No event-specific Charter product applied.", previous["blocker_or_basis"])
        package = package_by_candidate.get(previous["candidate_id"], {})
        rows.append(with_invariants({
            "recife_package_id": previous["review_packet_id"], "event_patch_package_id": package.get("event_patch_package_id", ""),
            "candidate_id": previous["candidate_id"],
            "previous_gate_id": previous["gate"], "previous_gate_status": previous["status"],
            "updated_gate_status": status, "update_reason": reason,
            "evidence_used": "CH758_RECIFE_20220602_001" if may_2022 else "",
            "blocker_remaining": blocker,
            "next_action_rank_1": "REQUEST_OR_MANUALLY_ACCESS_CHARTER_PRODUCT_VECTOR_CRS" if may_2022 else previous["next_action_rank_1"],
            "parallel_action": "ACQUIRE_CEMADEN_APAC_RECIFE_MAY_2022_TEMPORAL_SERIES" if may_2022 else "ACQUIRE_EVENT_SPECIFIC_RECIFE_TEMPORAL_SERIES",
        }))
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


def run_build_manual_access_request_tasks(args=None):
    actions = [
        ("OPEN_PRODUCT_PAGE", "LIGHTWEIGHT_PRODUCT_PAGE_METADATA", "PRODUCT_PAGE_METADATA_GAP"),
        ("VERIFY_DOWNLOAD_LINK", "DOWNLOAD_LINK_STATUS", "DOWNLOAD_LINK_NOT_CONFIRMED"),
        ("REQUEST_VECTOR_FILE", "VECTOR_FILE_ACCESS_STATUS", "VECTOR_NOT_CONFIRMED"),
        ("VERIFY_CRS", "CRS_STATUS", "CRS_UNKNOWN"),
        ("VERIFY_LICENSE_TERMS", "LICENSE_REDISTRIBUTION_STATUS", "LICENSE_TERMS_UNKNOWN"),
        ("VERIFY_FEATURE_TYPE", "FEATURE_TYPE_REVIEW", "FEATURE_TYPE_UNKNOWN"),
        ("SAVE_DERIVED_METADATA", "LIGHTWEIGHT_DERIVED_MANIFEST", "DERIVED_MANIFEST_PENDING"),
    ]
    product = load_csv(dataset_path(OUTPUTS[0]))[0]
    rows = [with_invariants({
        "task_id": f"TASK_v2bh_{i:03d}", "product_id": product["product_id"], "product_title": product["product_title"],
        "target_institution": "International Charter Space and Major Disasters",
        "required_action": action, "expected_artifact": artifact, "priority": "P0",
        "blocker_resolved_if_done": blocker,
    }) for i, (action, artifact, blocker) in enumerate(actions, 1)]
    write_csv(dataset_path(OUTPUTS[6]), rows); return rows


def run_generate_charter_product_review_packets(args=None):
    products = by(load_csv(dataset_path(OUTPUTS[0])), "product_id")
    classes = by(load_csv(dataset_path(OUTPUTS[1])), "product_id")
    access = by(load_csv(dataset_path(OUTPUTS[2])), "product_id")
    hazards = by(load_csv(dataset_path(OUTPUTS[3])), "product_id")
    candidates = by(load_csv(dataset_path(OUTPUTS[4])), "product_id")
    rows = []
    for product_id, cls in classes.items():
        if cls["municipality_classification"] != "RECIFE": continue
        product, acc, hazard, candidate = products[product_id], access[product_id], hazards[product_id], candidates[product_id]
        path = doc_path("product_review_packets", f"{product_id.lower()}.md")
        write_text(path, f"""# Activation 758 Product Review Packet

## Produto
`{product['product_title']}`; area `{product['product_area_text']}`; data `{product['product_date']}`.

## Hazard e geometria candidata
Hazard `{hazard['hazard_type_candidate']}`; geometria `{hazard['geometry_feature_type_candidate']}`.

## Vetor, CRS e licenca
Access `{acc['access_status']}`; vector `false`; CRS `false`; license/redistribution `false`.

## Uso permitido no Protocolo C
`{candidate['candidate_status']}` para revisao humana. Nao e ground truth final, label, negativo ou treino.

## Bloqueios e proxima acao
`{acc['blocker']}`. `{candidate['required_human_action']}`.

## Guardrails
Produto Charter nao e ground truth final; preview nao e vetor; produto de deslizamento nao e flood extent; produto Olinda nao e Recife.
""")
        rows.append(with_invariants({
            "packet_id": f"PACK_v2bh_{len(rows)+1:03d}", "product_id": product_id,
            "product_title": product["product_title"], "municipality_classification": cls["municipality_classification"],
            "candidate_status": candidate["candidate_status"],
            "packet_path": path.replace("\\", "/"),
            "next_action_rank_1": candidate["required_human_action"],
        }))
    write_csv(dataset_path(OUTPUTS[7]), rows)
    for name in OUTPUTS[:7]: write_csv(doc_path("product_inventory" if name == OUTPUTS[0] else "access_audit", name), load_csv(dataset_path(name)))
    write_text(doc_path("README.md"), """# v2bh Recife Charter 758 Product Audit

Eu/equipe confirmou em nivel de registro a Activation 758, o produto Recife de 2022-06-02 e o total reportado de 51 produtos. O inventario offline nao confirmou os demais titulos nem acessos individuais.

Charter 758 e P0 porque e uma fonte oficial de mapeamento de desastre especifica para Recife. Product e preview publicado demonstram existencia cartografica, mas nao equivalem a vetor confirmado nem a ground truth.

C3 passa para o evento Recife de maio de 2022 pela confirmacao do produto oficial. C4 permanece `PENDING_VECTOR_CRS`, pois vetor, CRS, licenca, redistribuicao e tipo exato de feicao nao foram confirmados. C7 continua bloqueado.
""")
    return rows


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:8], 1):
        violations = sum(r.get(field, "").lower() == "true" for r in load_csv(dataset_path(name)) for field in forbidden)
        rows.append({"regression_id": f"GR_v2bh_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore")
    passed = os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n"
    rows.append({"regression_id": "GR_v2bh_009", "artifact_path": marker.replace("\\", "/"), "violation_count": "0" if passed else "1", "status": "PASS" if passed else "FAIL"})
    c7 = [r for r in load_csv(dataset_path(OUTPUTS[5])) if r["previous_gate_id"] == "C7_FINAL_GROUND_TRUTH"]
    passed = bool(c7) and all(r["updated_gate_status"] == "BLOCKED" for r in c7)
    rows.append({"regression_id": "GR_v2bh_010", "artifact_path": "C7_FINAL_GROUND_TRUTH", "violation_count": "0" if passed else "1", "status": "PASS" if passed else "FAIL"})
    if any(r["status"] != "PASS" for r in rows): raise ValueError("v2bh guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[8]), rows); return rows


STEPS = [
    ("load_charter_758_registry", run_load_charter_registry, None),
    ("inventory_charter_products", run_inventory_charter_products, OUTPUTS[0]),
    ("classify_recife_olinda_products", run_classify_recife_olinda_products, OUTPUTS[1]),
    ("audit_product_access_vector_crs_license", run_audit_product_access, OUTPUTS[2]),
    ("classify_product_hazard_geometry_type", run_classify_product_hazard_geometry_type, OUTPUTS[3]),
    ("build_candidate_geometry_source_registry", run_build_candidate_geometry_source_registry, OUTPUTS[4]),
    ("update_recife_gate_status", run_update_recife_gate_status, OUTPUTS[5]),
    ("build_manual_access_request_tasks", run_build_manual_access_request_tasks, OUTPUTS[6]),
    ("generate_charter_product_review_packets", run_generate_charter_product_review_packets, OUTPUTS[7]),
    ("run_guardrail_regression", run_guardrail_regression, OUTPUTS[8]),
]


def ensure_structure():
    for folder in ("product_review_packets", "product_inventory", "access_audit", "evidence_cache"): os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args)
        path = dataset_path(output) if output else dataset_path(INPUTS["charter"])
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": path.replace("\\", "/"), "output_hash": sha256(path)[:16], "notes": "Offline product audit; no geometry or truth promotion."})
    write_csv(dataset_path("v2bh_orchestrator_manifest.csv"), manifest); return manifest
