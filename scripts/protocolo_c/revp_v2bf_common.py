#!/usr/bin/env python3
"""v2bf Curitiba patch-asset lineage recovery, review-only and fail-closed."""

import argparse
import csv
import hashlib
import os
import re

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
ROOT_DATASET_DIR = os.environ.get("ROOT_DATASET_DIR", "datasets")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2bf_curitiba_patch_asset_lineage")
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false", "can_create_label": "false",
    "can_create_negative": "false", "can_train_model": "false", "recovered_lineage_is_not_ground_truth": "true",
    "sentinel_review_asset_is_not_truth": "true", "dino_signal_is_not_truth": "true",
    "patch_boundary_is_not_event_geometry": "true", "review_derivative_is_not_label": "true",
    "no_geometry_no_final_truth": "true", "raw_data_versioned": "false",
}
INPUTS = {
    "seeds": "v2be_curitiba_blocked_seed_selection.csv", "scores": "v2be_sentinel_asset_candidate_scores.csv",
    "links": "v2be_candidate_seed_asset_links.csv", "best": "v2be_best_asset_per_seed.csv",
    "visual": "v2be_visual_review_asset_binding.csv", "dino": "v2be_seed_dino_binding.csv",
    "readiness": "v2be_candidate_reference_readiness_update.csv", "gates": "v2be_revised_promotion_gate.csv",
    "assets": "v2bd_sentinel_asset_discovery.csv", "crosswalk": "v2bd_seed_sentinel_crosswalk.csv",
    "v2bc_seeds": "v2bc_ground_truth_seed_registry.csv", "metrics": "v2ay_window_precipitation_metrics.csv",
}
LINEAGE_SOURCE_PATHS = [
    "datasets/dino_patch_visual_linkage_registry_v1pv.csv",
    "datasets/dino_visual_asset_eligibility_audit_v1pu.csv",
    "datasets/dino_execution_readiness_audit_v1qb.csv",
    "datasets/protocolo_c/v1us_patch_registry_resolution.csv",
    "datasets/protocolo_c/v1uw_curitiba_event_patch_prelink_update.csv",
    "datasets/recife_sentinel_product_date_resolution_registry.csv",
    "outputs_public/tables/table_dino_embedding_inventory.csv",
    "outputs_public/tables/table_dino_pca_coordinates.csv",
    "outputs_public/tables/table_dino_nearest_neighbors.csv",
    "outputs_public/metrics/dino_cluster_summary.csv",
    "outputs_public/execution_reports/private_artifact_discovery_inventory.csv",
    "local_runs/dino_embeddings/v1hc/visual_review_preview_manifest_v1hc.csv",
    "local_runs/dino_embeddings/v1hc/visual_review_patch_index_v1hc.csv",
    "local_runs/dino_embeddings/v1ge/dino_expanded_embedding_manifest_v1ge.csv",
    "local_runs/bandas_indices_patch_selection/bandas_indices_manifest.csv",
]
PREVIEW_DIR = "local_runs/dino_embeddings/v1hc/figures"
DINO_PUBLIC = "outputs_public/tables/table_dino_embedding_inventory.csv"
OUTPUTS = [
    "v2bf_internal_sentinel_lineage_sources.csv", "v2bf_asset_date_recovery.csv",
    "v2bf_asset_patch_link_recovery.csv", "v2bf_asset_file_reference_resolution.csv",
    "v2bf_sentinel_review_asset_materialization.csv", "v2bf_dino_link_recovery.csv",
    "v2bf_lineage_confidence_scores.csv", "v2bf_seed_asset_crosswalk_status_update.csv",
    "v2bf_curitiba_lineage_review_packet_index.csv", "v2bf_guardrail_regression.csv",
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


def detect_fields(fieldnames):
    names = {clean(name).lower() for name in fieldnames or []}
    contains = lambda terms: any(any(term in name for term in terms) for name in names)
    return {
        "contains_asset_id": contains(("asset_id", "visual_asset_id", "sentinel_asset_id")),
        "contains_patch_id": contains(("patch_id", "canonical_patch_id")),
        "contains_date": contains(("date", "scene_date", "acquisition")),
        "contains_city_region": contains(("city", "region")),
        "contains_visual_reference": contains(("preview", "figure", "visual", "source_path")),
        "contains_dino_reference": contains(("dino", "embedding", "pca", "neighbor")),
    }


def recover_date(*values):
    pattern = re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")
    for value in values:
        match = pattern.search(clean(value))
        if match:
            candidate = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            return candidate, "FIELD_OR_FILENAME_DATE", "PATTERN_PARSE", "MODERATE"
    return "", "NO_PRODUCT_OR_SCENE_DATE", "NO_ACCEPTABLE_DATE_EVIDENCE", "UNKNOWN"


def recover_patch(explicit="", filename="", path=""):
    if clean(explicit): return clean(explicit), "EXPLICIT_REGISTRY_FIELD", "EXPLICIT_FIELD", "HIGH"
    match = re.search(r"(CUR_\d{5})|patch_curitiba_(\d{5})", f"{filename} {path}", re.I)
    if match:
        patch = match.group(1).upper() if match.group(1) else f"CUR_{match.group(2)}"
        return patch, "FILENAME_OR_PATH", "PATTERN_PARSE", "MODERATE"
    return "", "NO_PATCH_EVIDENCE", "NO_LINK", "UNKNOWN"


def confidence(date="UNKNOWN", patch="UNKNOWN", visual="UNKNOWN", dino="NOT_LINKED"):
    if date == "HIGH" and patch == "HIGH" and visual == "HIGH" and dino in {"HIGH", "MODERATE"}: return "HIGH"
    if date in {"HIGH", "MODERATE"} and patch in {"HIGH", "MODERATE"} and visual in {"HIGH", "MODERATE"}: return "MODERATE"
    if patch in {"HIGH", "MODERATE"} and visual in {"HIGH", "MODERATE"}: return "LOW"
    return "VERY_LOW"


def crosswalk_update(date="UNKNOWN", patch="UNKNOWN", visual="MISSING", overall="VERY_LOW"):
    ready = date in {"HIGH", "MODERATE"} and patch in {"HIGH", "MODERATE"} and visual in {"READY_FOR_HUMAN_REVIEW", "REFERENCE_ONLY"}
    if ready and overall in {"HIGH", "MODERATE"}: return "CANDIDATE_CROSSWALK_RECOVERED", True
    if visual == "READY_FOR_HUMAN_REVIEW" and overall == "LOW": return "VISUAL_REVIEW_READY_WITH_WEAK_LINK", False
    if visual == "NEEDS_ASSET_RENDERING": return "NEEDS_ASSET_RENDERING", False
    return "NEEDS_MANUAL_LINEAGE_RESOLUTION", False


def run_discover_lineage_sources(args=None):
    rows = []
    for source in LINEAGE_SOURCE_PATHS:
        present = os.path.exists(source); fieldnames = []
        if present and source.endswith(".csv"):
            with open(source, encoding="utf-8-sig", newline="") as handle:
                fieldnames = next(csv.reader(handle), [])
        detected = detect_fields(fieldnames)
        usable = present and (detected["contains_patch_id"] or detected["contains_asset_id"]) and (
            detected["contains_date"] or detected["contains_visual_reference"] or detected["contains_dino_reference"])
        rows.append(with_invariants({
            "source_file": source, "source_type": "LOCAL_RUN_REGISTRY" if source.startswith("local_runs/") else "OUTPUT_PUBLIC_TABLE" if source.startswith("outputs_public/") else "VERSIONED_REGISTRY",
            "fields_detected": "|".join(fieldnames), **{key: str(value).lower() for key, value in detected.items()},
            "usable_for_lineage": str(usable).lower(), "note": "Internal metadata source; timestamps are not accepted as Sentinel acquisition dates.",
        }))
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def run_recover_asset_dates(args=None):
    rows = []
    for asset in load_inputs()["assets"]:
        recovered, source, method, conf = recover_date(asset.get("acquisition_date"), asset.get("sentinel_asset_id"), asset.get("asset_path_or_reference"))
        rows.append(with_invariants({
            "sentinel_asset_id": asset["sentinel_asset_id"], "recovered_date": recovered, "date_source": source,
            "date_recovery_method": method, "date_confidence": conf,
            "date_recovery_note": "Execution/modification timestamps were rejected; no Sentinel product or scene date was found." if not recovered else "Date parsed from explicit lineage evidence.",
        }))
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def run_recover_asset_patch_links(args=None):
    rows = []
    for asset in load_inputs()["assets"]:
        patch, source, method, conf = recover_patch(asset.get("patch_id"), asset.get("sentinel_asset_id"), asset.get("asset_path_or_reference"))
        rows.append(with_invariants({
            "sentinel_asset_id": asset["sentinel_asset_id"], "recovered_patch_id": patch, "patch_link_source": source,
            "patch_link_method": method, "patch_link_confidence": conf,
            "patch_link_note": "Asset-to-patch lineage recovered; this does not establish seed-to-patch linkage.",
            "patch_boundary_is_event_geometry": "false",
        }))
    write_csv(dataset_path(OUTPUTS[2]), rows); return rows


def preview_map():
    result = {}
    for row in load_csv("local_runs/dino_embeddings/v1hc/visual_review_preview_manifest_v1hc.csv"):
        path = os.path.join(PREVIEW_DIR, row.get("preview_file", ""))
        if row.get("preview_status") == "GENERATED" and os.path.exists(path): result[row["canonical_patch_id"]] = path.replace("\\", "/")
    return result


def run_resolve_asset_file_references(args=None):
    patches = by(load_csv(dataset_path(OUTPUTS[2])), "sentinel_asset_id"); previews = preview_map(); rows = []
    for asset in load_inputs()["assets"]:
        patch = patches[asset["sentinel_asset_id"]]["recovered_patch_id"]; preview = previews.get(patch, "")
        asset_ref = asset.get("asset_path_or_reference", ""); exists = os.path.exists(asset_ref)
        rows.append(with_invariants({
            "sentinel_asset_id": asset["sentinel_asset_id"], "asset_file_reference": asset_ref,
            "visual_file_reference": preview, "spectral_file_reference": "", "figure_reference": preview,
            "output_public_reference": DINO_PUBLIC if any(r.get("patch_id") == patch for r in load_csv(DINO_PUBLIC)) else "",
            "file_exists": str(exists).lower(), "reference_type": "LOCAL_REVIEW_PREVIEW" if preview else "REGISTERED_REFERENCE_ONLY",
            "usable_for_human_review": str(bool(preview)).lower(),
            "note": "Local lightweight preview may be reviewed; raw Sentinel reference is not present in this checkout." if preview else "No materialized visual payload found.",
        }))
    write_csv(dataset_path(OUTPUTS[3]), rows); return rows


def run_materialize_review_assets(args=None):
    best = load_inputs()["best"]; refs = by(load_csv(dataset_path(OUTPUTS[3])), "sentinel_asset_id"); rows = []
    for selected in best:
        ref = refs[selected["primary_sentinel_asset_id"]]; ready = is_true(ref["usable_for_human_review"])
        rows.append(with_invariants({
            "seed_id": selected["seed_id"], "sentinel_asset_id": selected["primary_sentinel_asset_id"],
            "review_asset_status": "READY_FOR_HUMAN_REVIEW" if ready else "NEEDS_ASSET_RENDERING",
            "review_asset_path_or_reference": ref["visual_file_reference"] or ref["asset_file_reference"],
            "generated_derivative": "false", "derivative_is_review_only": "true", "raw_data_versioned": "false",
            "limitation": "Existing local preview is review-only and has no proven seed/event temporal linkage." if ready else "No preview or payload is available.",
        }))
    write_csv(dataset_path(OUTPUTS[4]), rows)
    write_text(doc_path("sentinel_review_assets", "README.md"), "# Sentinel review assets\n\nEu/equipe referenciou previews leves ja existentes em local_runs. Nenhum raster Sentinel ou embedding bruto foi copiado ou versionado.\n")
    return rows


def run_recover_dino_links(args=None):
    best = load_inputs()["best"]; patches = by(load_csv(dataset_path(OUTPUTS[2])), "sentinel_asset_id"); inventory = by(load_csv(DINO_PUBLIC), "patch_id"); rows = []
    for selected in best:
        patch = patches[selected["primary_sentinel_asset_id"]]["recovered_patch_id"]; dino = inventory.get(patch, {})
        rows.append(with_invariants({
            "seed_id": selected["seed_id"], "sentinel_asset_id": selected["primary_sentinel_asset_id"], "patch_id": patch,
            "dino_embedding_id": dino.get("dino_input_id", ""), "dino_link_method": "ASSET_PATCH_TO_PUBLIC_DINO_INVENTORY" if dino else "NO_LINK",
            "dino_link_confidence": "MODERATE" if dino else "NOT_LINKED", "dino_review_signal_available": str(bool(dino)).lower(),
            "dino_is_ground_truth": "false", "dino_can_create_label": "false",
        }))
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


def run_build_lineage_confidence_scores(args=None):
    dates = by(load_csv(dataset_path(OUTPUTS[1])), "sentinel_asset_id"); patches = by(load_csv(dataset_path(OUTPUTS[2])), "sentinel_asset_id")
    refs = by(load_csv(dataset_path(OUTPUTS[3])), "sentinel_asset_id"); dino = by(load_csv(dataset_path(OUTPUTS[5])), "seed_id"); rows = []
    for selected in load_inputs()["best"]:
        asset = selected["primary_sentinel_asset_id"]; visual = "HIGH" if is_true(refs[asset]["usable_for_human_review"]) else "UNKNOWN"
        overall = confidence(dates[asset]["date_confidence"], patches[asset]["patch_link_confidence"], visual, dino[selected["seed_id"]]["dino_link_confidence"])
        rows.append(with_invariants({
            "seed_id": selected["seed_id"], "sentinel_asset_id": asset, "date_confidence": dates[asset]["date_confidence"],
            "patch_link_confidence": patches[asset]["patch_link_confidence"], "visual_reference_confidence": visual,
            "dino_link_confidence": dino[selected["seed_id"]]["dino_link_confidence"], "overall_lineage_confidence": overall,
            "confidence_reason": "Asset-to-patch, preview, and DINO lineage recovered; Sentinel date and seed-to-patch lineage remain unresolved.",
        }))
    write_csv(dataset_path(OUTPUTS[6]), rows); return rows


def run_update_seed_asset_crosswalk_status(args=None):
    scores = by(load_csv(dataset_path(OUTPUTS[6])), "seed_id"); materials = by(load_csv(dataset_path(OUTPUTS[4])), "seed_id")
    dino = by(load_csv(dataset_path(OUTPUTS[5])), "seed_id"); previous = by(load_inputs()["readiness"], "seed_id"); rows = []
    for selected in load_inputs()["best"]:
        score, material = scores[selected["seed_id"]], materials[selected["seed_id"]]
        status, ready = crosswalk_update(score["date_confidence"], score["patch_link_confidence"], material["review_asset_status"], score["overall_lineage_confidence"])
        rows.append(with_invariants({
            "seed_id": selected["seed_id"], "previous_status": previous[selected["seed_id"]]["updated_readiness_status"],
            "updated_crosswalk_status": status, "best_sentinel_asset_id": selected["primary_sentinel_asset_id"],
            "best_asset_lineage_confidence": score["overall_lineage_confidence"], "visual_review_status": material["review_asset_status"],
            "dino_link_status": dino[selected["seed_id"]]["dino_link_confidence"], "ready_for_candidate_reference_adjudication": str(ready).lower(),
            "remaining_blockers": "SENTINEL_DATE_MISSING|SEED_TO_PATCH_LINK_MISSING|GEOMETRY_MISSING|HUMAN_REVIEW_PENDING",
            "next_action_rank_1": "HUMAN_ADJUDICATE_CURITIBA_CANDIDATE_REFERENCES" if ready else "MANUALLY_REPAIR_SENTINEL_PATCH_ASSET_MANIFEST",
        }))
    write_csv(dataset_path(OUTPUTS[7]), rows); return rows


def run_generate_lineage_review_packets(args=None):
    seeds = by(load_inputs()["seeds"], "seed_id"); dates = by(load_csv(dataset_path(OUTPUTS[1])), "sentinel_asset_id")
    patches = by(load_csv(dataset_path(OUTPUTS[2])), "sentinel_asset_id"); materials = by(load_csv(dataset_path(OUTPUTS[4])), "seed_id")
    dino = by(load_csv(dataset_path(OUTPUTS[5])), "seed_id"); scores = by(load_csv(dataset_path(OUTPUTS[6])), "seed_id"); rows = []
    for status in load_csv(dataset_path(OUTPUTS[7])):
        seed, asset = seeds[status["seed_id"]], status["best_sentinel_asset_id"]
        path = doc_path("lineage_review_packets", f"{slug(status['seed_id'])}.md")
        write_text(path, f"""# Curitiba Patch-Asset Lineage Review: {status['seed_id']}

## Evento, janela e INMET
{seed['event_date']}; {seed['window_start']} a {seed['window_end']}; Curitiba A807 LOCAL com suporte temporal forte.

## Asset principal antes da v2bf
`{asset}`.

## Recuperacao de data e patch
Data: `{dates[asset]['date_confidence']}` / ausente. Asset-patch: `{patches[asset]['recovered_patch_id']}` ({patches[asset]['patch_link_confidence']}).

## Visual e DINO
Visual: `{materials[status['seed_id']]['review_asset_status']}`. DINO: `{dino[status['seed_id']]['dino_link_confidence']}`.

## Confianca e status
Linhagem geral: `{scores[status['seed_id']]['overall_lineage_confidence']}`. Status: `{status['updated_crosswalk_status']}`.

## Bloqueios e proxima acao
{status['remaining_blockers']}. `{status['next_action_rank_1']}`.

## Guardrails
Linhagem recuperada e preview nao sao truth ou label; DINO nao decide evento; nao cria negativo ou treino.
""")
        rows.append(with_invariants({
            "packet_index_id": f"PACK_v2bf_{len(rows)+1:04d}", "seed_id": status["seed_id"], "sentinel_asset_id": asset,
            "packet_path": f"docs/protocolo_c/v2bf_curitiba_patch_asset_lineage/lineage_review_packets/{slug(status['seed_id'])}.md",
            "updated_crosswalk_status": status["updated_crosswalk_status"], "ready_for_candidate_reference_adjudication": status["ready_for_candidate_reference_adjudication"],
            "next_action_rank_1": status["next_action_rank_1"],
        }))
    write_csv(dataset_path(OUTPUTS[8]), rows)
    for name in (OUTPUTS[1], OUTPUTS[2], OUTPUTS[3], OUTPUTS[5], OUTPUTS[7]): write_csv(doc_path("recovered_crosswalk_tables", name), load_csv(dataset_path(name)))
    write_text(doc_path("README.md"), "# v2bf Curitiba Patch-Asset Lineage Recovery\n\nEu/equipe recuperou asset-patch, previews locais e DINO review-only. Datas Sentinel e seed-patch continuam ausentes; adjudicacao candidata permanece bloqueada.\n")
    return rows


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:9], 1):
        violations = sum(row.get(field, "").lower() == "true" for row in load_csv(dataset_path(name)) for field in forbidden)
        violations += sum(is_true(row.get("ready_for_candidate_reference_adjudication")) for row in load_csv(dataset_path(name))) if name == OUTPUTS[7] else 0
        rows.append({"regression_id": f"GR_v2bf_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore"); violations = 0 if os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n" else 1
    rows.append({"regression_id": "GR_v2bf_010", "artifact_path": "docs/protocolo_c/v2bf_curitiba_patch_asset_lineage/evidence_cache/.gitignore", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    if any(row["status"] != "PASS" for row in rows): raise ValueError("v2bf guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[9]), rows); return rows


STEPS = [
    ("discover_internal_sentinel_lineage_sources", run_discover_lineage_sources, OUTPUTS[0]), ("recover_asset_dates", run_recover_asset_dates, OUTPUTS[1]),
    ("recover_asset_patch_links", run_recover_asset_patch_links, OUTPUTS[2]), ("resolve_asset_file_references", run_resolve_asset_file_references, OUTPUTS[3]),
    ("materialize_sentinel_review_assets", run_materialize_review_assets, OUTPUTS[4]), ("recover_dino_links", run_recover_dino_links, OUTPUTS[5]),
    ("build_lineage_confidence_scores", run_build_lineage_confidence_scores, OUTPUTS[6]), ("update_seed_asset_crosswalk_status", run_update_seed_asset_crosswalk_status, OUTPUTS[7]),
    ("generate_curitiba_lineage_review_packets", run_generate_lineage_review_packets, OUTPUTS[8]), ("guardrail_regression", run_guardrail_regression, OUTPUTS[9]),
]


def ensure_structure():
    for folder in ("lineage_review_packets", "recovered_crosswalk_tables", "sentinel_review_assets", "evidence_cache"): os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args); path = dataset_path(output)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": f"datasets/protocolo_c/{output}", "output_hash": sha256(path)[:16], "notes": "Internal lineage recovery completed without truth promotion."})
    write_csv(dataset_path("v2bf_orchestrator_manifest.csv"), manifest); return manifest
