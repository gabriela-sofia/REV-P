#!/usr/bin/env python3
"""v2bc Curitiba Local Ground-Truth Seed Construction.

Creates review-only candidate seeds. It never creates final ground truth,
labels, negatives, training data, or geometry.
"""

import argparse
import csv
import hashlib
import os
import re

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
ROOT_DATASET_DIR = os.environ.get("ROOT_DATASET_DIR", "datasets")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2bc_curitiba_ground_truth_seed")
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "seed_is_not_final_ground_truth": "true", "temporal_seed_is_not_spatial_truth": "true",
    "local_station_is_not_patch_geometry": "true", "sentinel_context_is_not_ground_truth": "true",
    "dino_signal_is_not_ground_truth": "true", "no_geometry_no_final_truth": "true",
    "raw_data_versioned": "false",
}
INPUTS = {
    "readiness": "v2ay_event_patch_temporal_readiness_update.csv",
    "metrics": "v2ay_window_precipitation_metrics.csv", "quality": "v2ay_timeseries_quality_report.csv",
    "stations": "v2ay_station_metadata_registry.csv", "manual": "v2az_manual_review_table.csv",
    "packets": "v2az_assisted_review_packet_index.csv", "geometry_classes": "v2ba_geometry_evidence_classification.csv",
    "digitization": "v2ba_candidate_digitization_registry.csv", "adjudication": "v2bb_adjudication_decision_table.csv",
    "digitization_matrix": "v2bb_digitization_review_candidate_matrix.csv", "uncertainty": "v2bb_uncertainty_update.csv",
}
OPTIONAL = {
    "observed": "v2an_observed_candidate_inventory_normalized.csv",
    "sentinel_crosswalk": "v2an_temporal_sentinel_crosswalk_audit.csv",
    "ground_readiness": "v2an_ground_reference_readiness_scores.csv",
}
ROOT_OPTIONAL = {
    "sentinel_windows": "event_sentinel_temporal_window_registry.csv",
    "dino_linkage": "dino_patch_visual_linkage_registry_v1pv.csv",
}
OUTPUTS = [
    "v2bc_strongest_external_validation_diagnosis.csv", "v2bc_curitiba_local_seed_candidates.csv",
    "v2bc_ground_truth_seed_registry.csv", "v2bc_seed_evidence_bundle.csv",
    "v2bc_sentinel_context_crosscheck.csv", "v2bc_seed_strength_scores.csv",
    "v2bc_non_selected_region_queue.csv", "v2bc_seed_review_packet_index.csv",
    "v2bc_guardrail_regression.csv",
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


def load_inputs():
    data = {key: load_csv(dataset_path(name)) for key, name in INPUTS.items()}
    data.update({key: load_csv(dataset_path(name)) for key, name in OPTIONAL.items()})
    data.update({key: load_csv(os.path.join(ROOT_DATASET_DIR, name)) for key, name in ROOT_OPTIONAL.items()})
    return data


def is_curitiba_local_seed(packet, metric, readiness):
    return (packet.get("city") == "Curitiba" or packet.get("region") == "Curitiba") and metric.get("station_id") == "A807" and packet.get("station_role") == "LOCAL" and readiness.get("temporal_readiness_status") == "TEMPORAL_EVIDENCE_READY_FOR_REVIEW" and float(metric.get("missing_rate") or 1) <= .2 and metric.get("precip_signal_status") == "PRECIPITATION_PRESENT"


def phenomenon(value):
    text = clean(value).lower()
    if "multi" in text or "misto" in text: return "MULTIHAZARD"
    if "flash" in text or "enxurr" in text: return "FLASH_FLOOD"
    if "landslide" in text or "desliz" in text: return "LANDSLIDE"
    if "flood" in text or "alag" in text or "inund" in text or "rain" in text: return "URBAN_FLOODING"
    return "UNKNOWN"


def score_seed(sentinel_available=False, geometry_present=False, proxy=False, quality=True):
    temporal, locality, data_quality = 40, 25 if not proxy else 5, 20 if quality else 5
    sentinel = 10 if sentinel_available else 0
    geometry_penalty = 0 if geometry_present else -15
    proxy_penalty = -20 if proxy else 0
    uncertainty_penalty = -10
    final = temporal + locality + data_quality + sentinel + geometry_penalty + proxy_penalty + uncertainty_penalty
    score_class = "STRONG_SEED_FOR_REVIEW" if final >= 75 else "MODERATE_SEED_FOR_REVIEW" if final >= 50 else "WEAK_SEED" if final > 0 else "NOT_A_SEED"
    return temporal, locality, data_quality, sentinel, geometry_penalty, proxy_penalty, uncertainty_penalty, final, score_class


def joined():
    data = load_inputs()
    lookups = {name: by(data[name], key) for name, key in [
        ("metrics", "event_patch_package_id"), ("readiness", "assertion_id"), ("quality", "event_patch_package_id"),
        ("manual", "event_patch_package_id"), ("adjudication", "review_packet_id"), ("uncertainty", "review_packet_id"),
        ("observed", "candidate_id"), ("sentinel_crosswalk", "candidate_id"), ("ground_readiness", "candidate_id"),
    ]}
    rows = []
    for packet in data["packets"]:
        result = {"packet": packet}
        for name, lookup in lookups.items():
            key = packet["event_patch_package_id"] if name in {"metrics", "readiness", "quality", "manual"} else packet["review_packet_id"] if name in {"adjudication", "uncertainty"} else packet["candidate_id"]
            result[name] = lookup.get(key, {})
        rows.append(result)
    return rows


def run_select_strongest_external_validation(args=None):
    data = load_inputs(); stations = by(data["stations"], "station_id")
    metrics = data["metrics"]; readiness = data["readiness"]
    specs = [("A807", "LOCAL", 1, "Local station but not patch geometry."),
             ("A610", "REGIONAL_PROXY", 2, "Regional proxy limits spatial relevance."),
             ("A301", "LOCAL_WITH_TEMPORAL_GAP", 3, "No usable records in event windows.")]
    rows = []
    for station_id, role, rank, limitation in specs:
        station = stations.get(station_id, {})
        count = sum(row.get("station_id") == station_id and row.get("temporal_support_status") == "TEMPORAL_EVIDENCE_READY_FOR_REVIEW" for row in metrics)
        parsed_in_windows = sum(int(row.get("records_available") or 0) for row in metrics if row.get("station_id") == station_id)
        rows.append(with_invariants({"validation_source": "INMET_REAL_ANNUAL_SERIES", "region": station.get("state", ""), "city": station.get("municipality", ""),
            "station_id": station_id, "station_name": station.get("station_name", ""), "station_role": role, "records_parsed": str(parsed_in_windows),
            "temporal_readiness_count": str(count), "spatial_relevance": "LOCAL_CITY_SUPPORT" if station_id != "A610" else "REGIONAL_PROXY_SUPPORT",
            "source_reliability": "OFFICIAL_OBSERVED_SERIES", "limitation": limitation, "validation_strength_rank": str(rank)}))
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def run_select_curitiba_local_seed_candidates(args=None):
    rows = []
    for item in joined():
        packet, metric, ready = item["packet"], item["metrics"], item["readiness"]
        if not is_curitiba_local_seed(packet, metric, ready): continue
        rows.append(with_invariants({"seed_candidate_id": f"SEEDC_v2bc_{len(rows)+1:04d}", "event_patch_package_id": packet["event_patch_package_id"],
            "candidate_id": packet["candidate_id"], "patch_id": packet["patch_id"], "city": packet["city"], "region": packet["region"],
            "event_date": metric["event_date"], "window_start": metric["window_start"], "window_end": metric["window_end"],
            "station_id": metric["station_id"], "station_name": "CURITIBA", "station_role": packet["station_role"], "missing_rate": metric["missing_rate"],
            "precip_total_window": metric["precip_total_window"], "precip_max_1h": metric["precip_max_1h"], "precip_max_24h": metric["precip_max_24h"],
            "precip_signal_status": metric["precip_signal_status"], "temporal_evidence_strength": "STRONG", "selected_as_seed_candidate": "true",
            "selection_reason": "Curitiba A807 local official series with positive temporal readiness and acceptable quality."}))
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def run_build_ground_truth_seed_registry(args=None):
    observed = by(load_inputs()["observed"], "candidate_id"); rows = []
    for candidate in load_csv(dataset_path(OUTPUTS[1])):
        rows.append(with_invariants({"seed_id": candidate["seed_candidate_id"].replace("SEEDC", "SEED"), "seed_candidate_id": candidate["seed_candidate_id"],
            "event_patch_package_id": candidate["event_patch_package_id"], "candidate_id": candidate["candidate_id"], "patch_id": candidate["patch_id"],
            "city": candidate["city"], "region": candidate["region"], "seed_status": "CANDIDATE_GROUND_TRUTH_SEED",
            "phenomenon_candidate": phenomenon(observed.get(candidate["candidate_id"], {}).get("hazard_type", "")), "temporal_support_level": "STRONG",
            "spatial_support_level": "LOCAL_STATION_SUPPORT", "geometry_status": "GEOMETRY_MISSING", "uncertainty_level": "HIGH",
            "required_next_validation": "HUMAN_REVIEW_SENTINEL_AND_CONTEXT|ACQUIRE_GEOMETRY_OR_DIGITIZE_CANDIDATE|ADJUDICATE_SEED"}))
    write_csv(dataset_path(OUTPUTS[2]), rows); return rows


def run_build_seed_evidence_bundle(args=None):
    data = load_inputs(); observed = by(data["observed"], "candidate_id"); candidates = by(load_csv(dataset_path(OUTPUTS[1])), "seed_candidate_id"); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[2])):
        candidate = candidates[seed["seed_candidate_id"]]; obs = observed.get(seed["candidate_id"], {})
        evidence = [
            ("INMET_TEMPORAL_SERIES", "INMET A807", candidate["event_date"], f"Window total {candidate['precip_total_window']} mm; max24h {candidate['precip_max_24h']} mm; missing {candidate['missing_rate']}.", "true", "false", "true", "Rainfall does not prove patch event extent.", "STRONG"),
            ("PATCH_CONTEXT", "v2az assisted review packet", candidate["event_date"], "Patch/event package exists but patch link is not established.", "false", "true", "false", "Patch context is not event geometry.", "WEAK"),
            ("TEXTUAL_CONTEXT", obs.get("primary_source_name", "Prefeitura de Curitiba"), candidate["event_date"], obs.get("notes", "Institutional contextual record."), "true", "true", "true", "Textual context is not geometry.", "MODERATE"),
            ("GEOMETRY_GAP", "v2ba/v2bb geometry audits", "", "No explicit event geometry or candidate geometry.", "false", "false", "false", "No geometry means no final truth.", "STRONG_BLOCKER"),
        ]
        for typ, source, date, summary, temporal, spatial, hazard, limitation, strength in evidence:
            rows.append(with_invariants({"bundle_id": f"BUNDLE_v2bc_{len(rows)+1:04d}", "seed_id": seed["seed_id"], "evidence_type": typ,
                "evidence_source": source, "evidence_date": date, "evidence_summary": summary, "supports_temporal_component": temporal,
                "supports_spatial_component": spatial, "supports_hazard_component": hazard, "limitation": limitation, "evidence_strength": strength}))
    write_csv(dataset_path(OUTPUTS[3]), rows)
    return rows


def run_crosscheck_seed_with_sentinel_context(args=None):
    data = load_inputs(); crosswalk = by(data["sentinel_crosswalk"], "candidate_id"); windows = by(data["sentinel_windows"], "observed_event_id")
    rows = []
    for seed in load_csv(dataset_path(OUTPUTS[2])):
        check = crosswalk.get(seed["candidate_id"], {}); window = windows.get(seed["candidate_id"], {})
        explicit = is_true(check.get("explicit_crosswalk_found"))
        rows.append(with_invariants({"crosscheck_id": f"SC_v2bc_{len(rows)+1:04d}", "seed_id": seed["seed_id"], "patch_id": seed["patch_id"],
            "sentinel_asset_id": check.get("sentinel_asset_id_found", ""), "sentinel_date": check.get("sentinel_asset_date_found", ""), "date_delta_to_event": "",
            "available_visual_context": str(bool(window)).lower(), "available_spectral_context": str(explicit).lower(), "dino_embedding_available": "false",
            "sentinel_support_status": "AVAILABLE_FOR_REVIEW" if explicit else "MISSING", "sentinel_is_ground_truth": "false", "dino_is_ground_truth": "false",
            "crosscheck_note": "Sentinel window metadata may exist, but no explicit event-asset or event-DINO crosswalk was inferred."}))
    write_csv(dataset_path(OUTPUTS[4]), rows); return rows


def run_score_seed_strength(args=None):
    cross = by(load_csv(dataset_path(OUTPUTS[4])), "seed_id"); quality = by(load_inputs()["quality"], "event_patch_package_id"); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[2])):
        sentinel = cross[seed["seed_id"]]["sentinel_support_status"] == "AVAILABLE_FOR_REVIEW"
        values = score_seed(sentinel_available=sentinel, geometry_present=False, proxy=False, quality=quality.get(seed["event_patch_package_id"], {}).get("quality_status") == "QUALITY_ACCEPTABLE_FOR_REVIEW")
        rows.append(with_invariants({"seed_id": seed["seed_id"], "temporal_score": str(values[0]), "station_locality_score": str(values[1]),
            "data_quality_score": str(values[2]), "sentinel_context_score": str(values[3]), "geometry_penalty": str(values[4]), "proxy_penalty": str(values[5]),
            "uncertainty_penalty": str(values[6]), "final_seed_strength_score": str(values[7]), "score_class": values[8],
            "score_note": "Strong temporal/local-station core, penalized by missing geometry and missing explicit Sentinel crosswalk."}))
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


def run_build_non_selected_region_queue(args=None):
    data = load_inputs(); packets = by(data["packets"], "event_patch_package_id"); rows = []
    selected = {row["event_patch_package_id"] for row in load_csv(dataset_path(OUTPUTS[1]))}
    for metric in data["metrics"]:
        if metric["event_patch_package_id"] in selected: continue
        packet = packets.get(metric["event_patch_package_id"], {})
        prefix = metric["event_patch_package_id"]
        if metric["station_id"] == "A610":
            region, city, reason, action = "Petropolis", "Petropolis", "REGIONAL_PROXY_LIMITATION", "RESOLVE_WITH_LOCAL_STATION"
        elif metric["station_id"] == "A301":
            region, city, reason, action = "Recife", "Recife", "TEMPORAL_GAP", "RESOLVE_WITH_CEMADEN"
        else:
            region, city, reason, action = packet.get("region", ""), packet.get("city", ""), "INSUFFICIENT_TEMPORAL_EVIDENCE", "KEEP_FOR_CONTEXT_ONLY"
        rows.append(with_invariants({"queue_id": f"QUEUE_v2bc_{len(rows)+1:04d}", "event_patch_package_id": prefix, "candidate_id": packet.get("candidate_id", ""),
            "region": region, "city": city, "reason_not_selected": reason, "required_action": action}))
    write_csv(dataset_path(OUTPUTS[6]), rows); return rows


def run_generate_seed_review_packets(args=None):
    candidates = by(load_csv(dataset_path(OUTPUTS[1])), "seed_candidate_id"); cross = by(load_csv(dataset_path(OUTPUTS[4])), "seed_id")
    scores = by(load_csv(dataset_path(OUTPUTS[5])), "seed_id"); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[2])):
        candidate, sentinel, score = candidates[seed["seed_candidate_id"]], cross[seed["seed_id"]], scores[seed["seed_id"]]
        path = doc_path("seed_review_packets", f"{slug(seed['candidate_id'])}.md")
        write_text(path, f"""# Curitiba Ground-Truth Seed Candidate: {seed['candidate_id']}

## 1. Identificacao
`{seed['seed_id']}` / `{seed['event_patch_package_id']}`.

## 2. Por que Curitiba/A807
Eu/equipe selecionei Curitiba/A807 por ser estacao INMET local, com serie real e readiness temporal positivo.

## 3. Metricas temporais
Total {candidate['precip_total_window']} mm; max1h {candidate['precip_max_1h']} mm; max24h {candidate['precip_max_24h']} mm; missing {candidate['missing_rate']}.

## 4. Patch e janela
Patch `{candidate['patch_id'] or 'NOT_AVAILABLE'}`; janela {candidate['window_start']} a {candidate['window_end']}.

## 5. Evidencias contextuais
Fonte institucional/documental disponivel para revisao; nao cria geometria.

## 6. Sentinel e DINO
Sentinel: {sentinel['sentinel_support_status']}; DINO disponivel no repositorio: {sentinel['dino_embedding_available']}; ambos review-only.

## 7. Lacuna geometrica
GEOMETRY_MISSING. Estacao local nao e geometria do patch.

## 8. Incerteza
HIGH; score {score['final_seed_strength_score']} ({score['score_class']}).

## 9. Decisao atual
CANDIDATE_GROUND_TRUTH_SEED, nao ground truth final.

## 10. Proxima acao
REVIEW_CURITIBA_SEEDS_WITH_SENTINEL_CONTEXT_AND_GEOMETRY_GAP; adquirir geometria, adjudicar seed e so depois considerar promocao.
""")
        write_text(doc_path("evidence_bundles", f"{slug(seed['candidate_id'])}.md"),
                   f"# Evidence bundle: {seed['candidate_id']}\n\nEu/equipe consolidei INMET A807, contexto textual, Sentinel review-only e lacuna geometrica.\n")
        rows.append(with_invariants({"packet_index_id": f"PACK_v2bc_{len(rows)+1:04d}", "seed_id": seed["seed_id"], "event_patch_package_id": seed["event_patch_package_id"],
            "packet_path": f"docs/protocolo_c/v2bc_curitiba_ground_truth_seed/seed_review_packets/{slug(seed['candidate_id'])}.md",
            "seed_status": seed["seed_status"], "score_class": score["score_class"], "next_action_rank_1": "REVIEW_CURITIBA_SEEDS_WITH_SENTINEL_CONTEXT_AND_GEOMETRY_GAP"}))
    write_csv(dataset_path(OUTPUTS[7]), rows)
    write_text(doc_path("README.md"), "# v2bc Curitiba Local Ground-Truth Seed Construction\n\nEu/equipe selecionei somente Curitiba/A807 como nucleo inicial. As sementes sao candidatas de revisao, nao ground truth final. Ground truth, labels, negativos e treino: 0.\n")
    write_text(doc_path("non_selected_regions", "README.md"), "# Non-selected regions\n\nPetropolis permanece limitada por A610 regional proxy. Recife permanece com gap temporal.\n")
    return rows


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:8], 1):
        violations = sum(row.get(field, "").lower() == "true" for row in load_csv(dataset_path(name)) for field in forbidden)
        rows.append({"regression_id": f"GR_v2bc_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore"); violations = 0 if os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n" else 1
    rows.append({"regression_id": "GR_v2bc_009", "artifact_path": "docs/protocolo_c/v2bc_curitiba_ground_truth_seed/evidence_cache/.gitignore", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    if any(row["status"] != "PASS" for row in rows): raise ValueError("v2bc guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[8]), rows); return rows


STEPS = [
    ("select_strongest_external_validation", run_select_strongest_external_validation, OUTPUTS[0]),
    ("select_curitiba_local_seed_candidates", run_select_curitiba_local_seed_candidates, OUTPUTS[1]),
    ("build_ground_truth_seed_registry", run_build_ground_truth_seed_registry, OUTPUTS[2]),
    ("build_seed_evidence_bundle", run_build_seed_evidence_bundle, OUTPUTS[3]),
    ("crosscheck_seed_with_sentinel_context", run_crosscheck_seed_with_sentinel_context, OUTPUTS[4]),
    ("score_seed_strength", run_score_seed_strength, OUTPUTS[5]),
    ("build_non_selected_region_queue", run_build_non_selected_region_queue, OUTPUTS[6]),
    ("generate_seed_review_packets", run_generate_seed_review_packets, OUTPUTS[7]),
    ("guardrail_regression", run_guardrail_regression, OUTPUTS[8]),
]


def ensure_structure():
    for folder in ("seed_review_packets", "evidence_bundles", "non_selected_regions", "evidence_cache"): os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args); path = dataset_path(output)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": f"datasets/protocolo_c/{output}", "output_hash": sha256(path)[:16], "notes": "Completed."})
    write_csv(dataset_path("v2bc_orchestrator_manifest.csv"), manifest); return manifest
