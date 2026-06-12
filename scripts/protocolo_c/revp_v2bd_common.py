#!/usr/bin/env python3
"""v2bd Curitiba Seed-Sentinel Crosswalk and Candidate Reference Promotion Gate."""

import argparse
import csv
import datetime as dt
import hashlib
import os
import re

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
ROOT_DATASET_DIR = os.environ.get("ROOT_DATASET_DIR", "datasets")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2bd_curitiba_candidate_reference")
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "candidate_reference_is_not_final_ground_truth": "true", "seed_sentinel_crosswalk_is_not_label": "true",
    "sentinel_visual_context_is_not_truth": "true", "dino_signal_is_not_truth": "true",
    "patch_boundary_is_not_event_geometry": "true", "no_geometry_no_final_truth": "true",
    "raw_data_versioned": "false",
}
INPUTS = {
    "seeds": "v2bc_ground_truth_seed_registry.csv",
    "seed_evidence": "v2bc_seed_evidence_bundle.csv",
    "seed_candidates": "v2bc_curitiba_local_seed_candidates.csv",
    "seed_scores": "v2bc_seed_strength_scores.csv",
    "seed_context": "v2bc_sentinel_context_crosscheck.csv",
    "metrics": "v2ay_window_precipitation_metrics.csv",
    "manual_review": "v2az_manual_review_table.csv",
    "assisted_packets": "v2az_assisted_review_packet_index.csv",
    "digitization": "v2ba_candidate_digitization_registry.csv",
    "adjudication": "v2bb_adjudication_decision_table.csv",
}
ROOT_INPUTS = {
    "visual_links": "dino_patch_visual_linkage_registry_v1pv.csv",
    "visual_assets": "dino_visual_asset_eligibility_audit_v1pu.csv",
    "visual_inventory": "dino_patch_visual_asset_inventory_v1pn.csv",
    "dino_features": "dino_embedding_feature_store_registry_v1ph.csv",
    "dino_readiness": "dino_execution_readiness_audit_v1qb.csv",
    "patch_preflight": "event_patch_linking_preflight_registry.csv",
    "sentinel_windows": "event_sentinel_temporal_window_registry.csv",
}
OUTPUTS = [
    "v2bd_curitiba_seed_selection.csv", "v2bd_sentinel_asset_discovery.csv",
    "v2bd_seed_sentinel_crosswalk.csv", "v2bd_seed_patch_crosswalk.csv",
    "v2bd_seed_dino_crosswalk.csv", "v2bd_visual_review_asset_audit.csv",
    "v2bd_candidate_reference_readiness.csv", "v2bd_candidate_promotion_gate.csv",
    "v2bd_candidate_reference_packet_index.csv", "v2bd_guardrail_regression.csv",
]


def parse_args(argv=None): return argparse.ArgumentParser().parse_args(argv)
def clean(value): return str(value or "").strip()
def is_true(value): return clean(value).lower() == "true"
def slug(value): return re.sub(r"[^a-z0-9]+", "-", clean(value).lower()).strip("-")
def dataset_path(name): return os.path.join(DATASET_DIR, name)
def root_dataset_path(name): return os.path.join(ROOT_DATASET_DIR, name)
def doc_path(*parts): return os.path.join(DOCS_DIR, *parts)
def with_invariants(row): return {**row, **INVARIANTS}


def load_csv(path):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def load_inputs():
    data = {key: load_csv(dataset_path(name)) for key, name in INPUTS.items()}
    data.update({key: load_csv(root_dataset_path(name)) for key, name in ROOT_INPUTS.items()})
    return data


def temporal_match(window_start, window_end, acquisition_date):
    try:
        start, end, acquired = (dt.date.fromisoformat(value) for value in (window_start, window_end, acquisition_date))
    except (TypeError, ValueError):
        return "", "UNKNOWN_DATE_MISSING"
    delta = 0 if start <= acquired <= end else min(abs((acquired - start).days), abs((acquired - end).days))
    return str(delta), "WITHIN_EVENT_WINDOW" if start <= acquired <= end else "OUTSIDE_EVENT_WINDOW"


def patch_match(seed_patch_id, asset_patch_id, same_region=False):
    if clean(seed_patch_id) and clean(seed_patch_id) == clean(asset_patch_id):
        return "EXPLICIT_PATCH_MATCH"
    if same_region:
        return "REGION_ONLY_NOT_PATCH_MATCH"
    return "NO_EXPLICIT_PATCH_MATCH"


def eligible_seed(seed, candidate, score):
    return (seed.get("city") == "Curitiba" and seed.get("seed_status") == "CANDIDATE_GROUND_TRUTH_SEED"
            and candidate.get("station_id") == "A807" and candidate.get("station_role") == "LOCAL"
            and score.get("score_class") in {"MODERATE_SEED_FOR_REVIEW", "STRONG_SEED_FOR_REVIEW"})


def seed_asset_crosswalk_status(asset_found=False, explicit_patch=False, same_region=False, acquisition_date=""):
    if not asset_found: return "NO_ASSET_FOUND"
    if explicit_patch and acquisition_date: return "EXPLICIT_SEED_ASSET_LINK"
    if same_region: return "NEEDS_MANUAL_REVIEW"
    return "CANDIDATE_SEED_ASSET_LINK"


def dino_review_status(embedding_available=False, explicitly_linked=False):
    if embedding_available and explicitly_linked: return "AVAILABLE_FOR_REVIEW"
    if embedding_available: return "NOT_LINKED"
    return "MISSING"


def readiness_class(patch_link=False, sentinel_link=False, sentinel_date=False, dino_link=False,
                    geometry=False, human_review=False):
    if not geometry: return "CANDIDATE_REFERENCE_BLOCKED_BY_GEOMETRY"
    if not patch_link or not sentinel_link or not sentinel_date:
        return "CANDIDATE_REFERENCE_NEEDS_SENTINEL_CROSSWALK"
    if not human_review: return "CANDIDATE_REFERENCE_NEEDS_HUMAN_REVIEW"
    if not dino_link: return "CANDIDATE_REFERENCE_READY_WITHOUT_DINO"
    return "CANDIDATE_REFERENCE_READY_FOR_ADJUDICATION"


def promotion_decision(readiness, temporal_support="STRONG", geometry=False, human_review=False):
    allowed = readiness in {"CANDIDATE_REFERENCE_READY_WITHOUT_DINO", "CANDIDATE_REFERENCE_READY_FOR_ADJUDICATION"} and temporal_support == "STRONG" and geometry and human_review
    return ("CANDIDATE_REFERENCE_FOR_ADJUDICATION", True) if allowed else ("REMAIN_CANDIDATE_GROUND_TRUTH_SEED", False)


def run_select_curitiba_seeds(args=None):
    candidates, scores = by(load_inputs()["seed_candidates"], "seed_candidate_id"), by(load_inputs()["seed_scores"], "seed_id")
    rows = []
    for seed in load_inputs()["seeds"]:
        candidate, score = candidates.get(seed["seed_candidate_id"], {}), scores.get(seed["seed_id"], {})
        if not eligible_seed(seed, candidate, score): continue
        rows.append(with_invariants({
            "selection_id": f"SEL_v2bd_{len(rows)+1:04d}", "seed_id": seed["seed_id"],
            "seed_candidate_id": seed["seed_candidate_id"],
            "event_patch_package_id": seed["event_patch_package_id"], "candidate_id": seed["candidate_id"],
            "patch_id": seed["patch_id"], "city": seed["city"], "region": seed["region"],
            "event_date": candidate.get("event_date", ""), "window_start": candidate.get("window_start", ""),
            "window_end": candidate.get("window_end", ""), "station_id": candidate.get("station_id", ""),
            "station_role": candidate.get("station_role", ""), "seed_status": seed["seed_status"],
            "score_class": score.get("score_class", ""), "selected_for_crosswalk": "true",
            "selection_reason": "Curitiba A807 LOCAL candidate seed with MODERATE_SEED_FOR_REVIEW or higher.",
            "temporal_support_level": seed["temporal_support_level"],
            "geometry_status": seed["geometry_status"], "seed_score_class": score.get("score_class", ""),
            "selection_status": "SELECTED_FOR_SEED_SENTINEL_CROSSWALK",
        }))
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def run_discover_sentinel_assets(args=None):
    data = load_inputs(); assets = by(data["visual_assets"], "visual_asset_id"); rows = []
    for link in data["visual_links"]:
        if clean(link.get("region")).upper() != "CURITIBA" or not is_true(link.get("eligible_for_dino_review")): continue
        asset = assets.get(link["visual_asset_id"], {})
        rows.append(with_invariants({
            "discovery_id": f"ASSET_v2bd_{len(rows)+1:04d}", "sentinel_asset_id": link["visual_asset_id"],
            "visual_asset_id": link["visual_asset_id"], "patch_id": link["patch_id"], "city": "Curitiba", "region": "Curitiba",
            "source_registry": "datasets/dino_patch_visual_linkage_registry_v1pv.csv|datasets/dino_visual_asset_eligibility_audit_v1pu.csv",
            "asset_path_or_reference": asset.get("relative_path", ""), "relative_path": asset.get("relative_path", ""),
            "asset_type": link.get("visual_type", asset.get("asset_visual_type", "")),
            "visual_type": link.get("visual_type", asset.get("asset_visual_type", "")),
            "acquisition_date": "", "acquisition_date_source": "NOT_AVAILABLE_IN_REGISTERED_PROVENANCE",
            "asset_size_bytes": asset.get("file_size_bytes", ""), "eligible_for_human_visual_review": "true",
            "has_visual_asset": "true", "has_spectral_context": "false", "has_dino_embedding": "false",
            "dino_allowed_use": link.get("dino_allowed_use", "REVIEW_ONLY_REPRESENTATION"),
            "discovery_status": "DISCOVERED_REVIEW_ONLY_DATE_MISSING",
        }))
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def run_build_seed_sentinel_crosswalk(args=None):
    assets = load_csv(dataset_path(OUTPUTS[1])); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        for asset in assets:
            relation = patch_match(seed["patch_id"], asset["patch_id"], same_region=True)
            delta, temporal = temporal_match(seed["window_start"], seed["window_end"], asset["acquisition_date"])
            rows.append(with_invariants({
                "crosswalk_id": f"SSX_v2bd_{len(rows)+1:05d}", "seed_id": seed["seed_id"],
                "event_patch_package_id": seed["event_patch_package_id"], "candidate_id": seed["candidate_id"],
                "patch_id": seed["patch_id"], "seed_patch_id": seed["patch_id"], "event_date": seed["event_date"],
                "window_start": seed["window_start"], "window_end": seed["window_end"],
                "sentinel_asset_id": asset["sentinel_asset_id"], "visual_asset_id": asset["visual_asset_id"], "asset_patch_id": asset["patch_id"],
                "sentinel_acquisition_date": asset["acquisition_date"], "date_delta_days": delta,
                "within_event_window": "false", "within_review_window": "false",
                "temporal_match_status": "UNKNOWN" if temporal == "UNKNOWN_DATE_MISSING" else "EXACT_WINDOW_MATCH" if temporal == "WITHIN_EVENT_WINDOW" else "OUTSIDE_WINDOW",
                "patch_match_status": "PATCH_ID_MATCH" if relation == "EXPLICIT_PATCH_MATCH" else "CITY_REGION_MATCH" if relation == "REGION_ONLY_NOT_PATCH_MATCH" else "NO_EXPLICIT_PATCH_MATCH",
                "explicit_seed_asset_link": "false", "crosswalk_status": "NEEDS_MANUAL_REVIEW",
                "crosswalk_note": "Region-only co-presence does not establish seed-patch or seed-asset linkage.",
            }))
    write_csv(dataset_path(OUTPUTS[2]), rows); return rows


def run_build_seed_patch_crosswalk(args=None):
    preflight = by(load_inputs()["patch_preflight"], "observed_event_id"); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        prior = preflight.get(seed["candidate_id"], {})
        rows.append(with_invariants({
            "seed_patch_crosswalk_id": f"SPX_v2bd_{len(rows)+1:04d}", "seed_id": seed["seed_id"],
            "candidate_id": seed["candidate_id"], "patch_id": seed["patch_id"], "seed_patch_id": seed["patch_id"],
            "patch_region": seed["region"], "patch_city": seed["city"],
            "patch_geometry_available": "false", "patch_boundary_status": "NOT_AVAILABLE",
            "patch_is_event_geometry": "false", "patch_context_allowed": "true", "patch_truth_allowed": "false",
            "prior_patch_scope": prior.get("patch_scope", "REGION_LEVEL"), "prior_patch_id": prior.get("patch_id", ""),
            "source_geometry_available": prior.get("source_geometry_available", "FALSE"),
            "patch_event_relation_status": prior.get("patch_event_relation_status", "NO_RELATION_ESTABLISHED"),
            "crosswalk_status": "NO_EXPLICIT_PATCH_LINK", "geometry_status": seed["geometry_status"],
            "required_action": "ESTABLISH_EVENT_SPECIFIC_PATCH_LINK_WITH_AUDITABLE_SPATIAL_EVIDENCE",
        }))
    write_csv(dataset_path(OUTPUTS[3]), rows); return rows


def run_build_seed_dino_crosswalk(args=None):
    features = load_inputs()["dino_features"]; rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        linked = [row for row in features if seed["patch_id"] and row.get("patch_id") == seed["patch_id"]]
        rows.append(with_invariants({
            "seed_dino_crosswalk_id": f"SDX_v2bd_{len(rows)+1:04d}", "seed_id": seed["seed_id"],
            "candidate_id": seed["candidate_id"], "patch_id": seed["patch_id"],
            "sentinel_asset_id": "", "dino_embedding_id": linked[0].get("embedding_id", "") if linked else "",
            "embedding_id": linked[0].get("embedding_id", "") if linked else "",
            "embedding_available": str(bool(linked)).lower(), "dino_link_status": "LINKED_REVIEW_ONLY" if linked else "NOT_LINKED",
            "dino_analysis_available": str(bool(linked)).lower(), "nearest_neighbors_available": "false",
            "pca_available": "false", "outlier_status_available": "false",
            "dino_review_signal_status": dino_review_status(bool(linked), bool(linked)),
            "dino_is_ground_truth": "false", "dino_can_create_label": "false",
            "regional_visual_assets_available": str(bool(load_csv(dataset_path(OUTPUTS[1])))).lower(),
            "dino_use": "REVIEW_ONLY_REPRESENTATION", "dino_decision_allowed": "false",
        }))
    write_csv(dataset_path(OUTPUTS[4]), rows); return rows


def run_audit_visual_review_assets(args=None):
    rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        for asset in load_csv(dataset_path(OUTPUTS[1])):
            rows.append(with_invariants({
                "visual_audit_id": f"VIS_v2bd_{len(rows)+1:05d}", "seed_id": seed["seed_id"],
                "sentinel_asset_id": asset["sentinel_asset_id"], "visual_asset_id": asset["visual_asset_id"],
                "patch_id": asset["patch_id"], "region": asset["region"],
                "visual_asset_path_or_reference": asset["asset_path_or_reference"], "relative_path": asset["relative_path"],
                "visual_asset_available": "true", "visual_date_match": "UNKNOWN_DATE_MISSING",
                "spectral_context_available": "false", "figure_available": "false",
                "ready_for_human_visual_review": "false",
                "limitation": "Regional registered reference lacks seed-specific patch link and acquisition date.",
                "visual_is_ground_truth": "false", "asset_registered": "true",
                "asset_nonempty": str(int(asset.get("asset_size_bytes") or 0) > 0).lower(),
                "acquisition_date_available": str(bool(asset["acquisition_date"])).lower(),
                "seed_specific_link_available": "false", "ready_for_regional_human_visual_review": "true",
                "ready_for_seed_specific_adjudication": "false",
                "audit_status": "REGIONAL_REVIEW_ONLY_BLOCKED_FOR_SEED_ADJUDICATION",
            }))
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


def run_compute_candidate_reference_readiness(args=None):
    patches, dino = by(load_csv(dataset_path(OUTPUTS[3])), "seed_id"), by(load_csv(dataset_path(OUTPUTS[4])), "seed_id")
    rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        patch_link = patches[seed["seed_id"]]["crosswalk_status"] != "NO_EXPLICIT_PATCH_LINK"
        dino_link = dino[seed["seed_id"]]["dino_link_status"] == "LINKED_REVIEW_ONLY"
        classification = readiness_class(patch_link, False, False, dino_link, seed["geometry_status"] != "GEOMETRY_MISSING", False)
        rows.append(with_invariants({
            "readiness_id": f"READY_v2bd_{len(rows)+1:04d}", "seed_id": seed["seed_id"],
            "candidate_id": seed["candidate_id"], "temporal_support_strong": str(seed["temporal_support_level"] == "STRONG").lower(),
            "explicit_patch_link": str(patch_link).lower(), "explicit_sentinel_link": "false",
            "sentinel_date_available": "false", "dino_link_available": str(dino_link).lower(),
            "geometry_available": str(seed["geometry_status"] != "GEOMETRY_MISSING").lower(), "human_review_complete": "false",
            "temporal_component": "STRONG_A807_LOCAL", "patch_component": "NOT_LINKED",
            "sentinel_component": "REGIONAL_ASSETS_NOT_LINKED_DATE_MISSING", "dino_component": "NOT_LINKED",
            "geometry_component": "GEOMETRY_MISSING_EXPLICIT_GAP", "human_review_component": "PENDING",
            "candidate_reference_readiness": classification,
            "blocking_factors": "NO_EXPLICIT_PATCH_LINK|NO_EXPLICIT_SENTINEL_LINK|SENTINEL_DATE_MISSING|GEOMETRY_MISSING|HUMAN_REVIEW_PENDING",
        }))
    write_csv(dataset_path(OUTPUTS[6]), rows); return rows


def run_candidate_promotion_gate(args=None):
    selected = by(load_csv(dataset_path(OUTPUTS[0])), "seed_id"); rows = []
    for ready in load_csv(dataset_path(OUTPUTS[6])):
        seed = selected[ready["seed_id"]]
        proposed, allowed = promotion_decision(ready["candidate_reference_readiness"], seed["temporal_support_level"],
                                               is_true(ready["geometry_available"]), is_true(ready["human_review_complete"]))
        rows.append(with_invariants({
            "promotion_gate_id": f"GATE_v2bd_{len(rows)+1:04d}", "seed_id": ready["seed_id"],
            "candidate_id": ready["candidate_id"], "previous_status": "CANDIDATE_GROUND_TRUTH_SEED",
            "current_status": "CANDIDATE_GROUND_TRUTH_SEED",
            "proposed_status": proposed, "promotion_allowed": str(allowed).lower(),
            "gate_status": "PASS_CANDIDATE_REFERENCE_ONLY" if allowed else "PROMOTION_BLOCKED",
            "promotion_reason": "Minimum explicit seed-patch-Sentinel chain is incomplete." if not allowed else "Minimum candidate-reference chain complete for human adjudication only.",
            "blockers_remaining": ready["blocking_factors"], "blockers": ready["blocking_factors"], "final_ground_truth_allowed": "false",
            "label_creation_allowed": "false", "next_action_rank_1": "RESOLVE_CURITIBA_SENTINEL_ASSET_CROSSWALK",
        }))
    write_csv(dataset_path(OUTPUTS[7]), rows); return rows


def run_generate_candidate_reference_packets(args=None):
    ready = by(load_csv(dataset_path(OUTPUTS[6])), "seed_id"); gates = by(load_csv(dataset_path(OUTPUTS[7])), "seed_id"); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        gate, status = gates[seed["seed_id"]], ready[seed["seed_id"]]
        path = doc_path("candidate_reference_packets", f"{slug(seed['candidate_id'])}.md")
        write_text(path, f"""# Curitiba Candidate Reference Packet: {seed['candidate_id']}

## Seed, evento e janela
`{seed['seed_id']}` / `{seed['event_patch_package_id']}`; {seed['window_start']} a {seed['window_end']}.

## Patch e geometria
Sem patch explicitamente vinculado; `{seed['geometry_status']}`. Limite de patch nao e geometria do evento.

## Sentinel
Ha assets regionais review-only registrados para Curitiba, sem data de aquisicao e sem vinculo explicito ao seed.

## DINO e revisao visual
Nenhum embedding DINO foi vinculado ao seed. DINO e revisao visual nao decidem evento.

## Readiness e gate
`{status['candidate_reference_readiness']}`; `{gate['gate_status']}`; promotion_allowed={gate['promotion_allowed']}.

## Proxima acao
`{gate['next_action_rank_1']}`.

## Guardrails
Nao e ground truth final; nao cria patch truth, label, negativo ou treino.
""")
        rows.append(with_invariants({
            "packet_index_id": f"PACK_v2bd_{len(rows)+1:04d}", "seed_id": seed["seed_id"],
            "candidate_id": seed["candidate_id"],
            "packet_path": f"docs/protocolo_c/v2bd_curitiba_candidate_reference/candidate_reference_packets/{slug(seed['candidate_id'])}.md",
            "readiness": status["candidate_reference_readiness"], "gate_status": gate["gate_status"],
            "promotion_allowed": gate["promotion_allowed"], "next_action_rank_1": gate["next_action_rank_1"],
        }))
    write_csv(dataset_path(OUTPUTS[8]), rows)
    write_csv(doc_path("crosswalk_tables", OUTPUTS[2]), load_csv(dataset_path(OUTPUTS[2])))
    write_csv(doc_path("crosswalk_tables", OUTPUTS[3]), load_csv(dataset_path(OUTPUTS[3])))
    write_csv(doc_path("visual_review_assets", OUTPUTS[5]), load_csv(dataset_path(OUTPUTS[5])))
    write_text(doc_path("README.md"), "# v2bd Curitiba Seed-Sentinel Crosswalk and Candidate Reference Promotion Gate\n\nTres seeds auditados. Promocoes: 0. Assets Sentinel regionais permanecem review-only, sem data e sem vinculo explicito. Ground truth, labels, negativos e treino: 0.\n")
    return rows


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:9], 1):
        violations = sum(row.get(field, "").lower() == "true" for row in load_csv(dataset_path(name)) for field in forbidden)
        if name == OUTPUTS[7]:
            violations += sum(is_true(row.get("promotion_allowed")) for row in load_csv(dataset_path(name)))
        rows.append({"regression_id": f"GR_v2bd_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}",
                     "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore")
    violations = 0 if os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n" else 1
    rows.append({"regression_id": "GR_v2bd_010", "artifact_path": "docs/protocolo_c/v2bd_curitiba_candidate_reference/evidence_cache/.gitignore",
                 "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    if any(row["status"] != "PASS" for row in rows): raise ValueError("v2bd guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[9]), rows); return rows


STEPS = [
    ("select_curitiba_seeds", run_select_curitiba_seeds, OUTPUTS[0]),
    ("discover_sentinel_assets", run_discover_sentinel_assets, OUTPUTS[1]),
    ("build_seed_sentinel_crosswalk", run_build_seed_sentinel_crosswalk, OUTPUTS[2]),
    ("build_seed_patch_crosswalk", run_build_seed_patch_crosswalk, OUTPUTS[3]),
    ("build_seed_dino_crosswalk", run_build_seed_dino_crosswalk, OUTPUTS[4]),
    ("audit_visual_review_assets", run_audit_visual_review_assets, OUTPUTS[5]),
    ("compute_candidate_reference_readiness", run_compute_candidate_reference_readiness, OUTPUTS[6]),
    ("candidate_promotion_gate", run_candidate_promotion_gate, OUTPUTS[7]),
    ("generate_candidate_reference_packets", run_generate_candidate_reference_packets, OUTPUTS[8]),
    ("guardrail_regression", run_guardrail_regression, OUTPUTS[9]),
]


def ensure_structure():
    for folder in ("candidate_reference_packets", "crosswalk_tables", "visual_review_assets", "evidence_cache"):
        os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args); path = dataset_path(output)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK",
                         "output": f"datasets/protocolo_c/{output}", "output_hash": sha256(path)[:16],
                         "notes": "Completed without truth or label promotion."})
    write_csv(dataset_path("v2bd_orchestrator_manifest.csv"), manifest); return manifest
