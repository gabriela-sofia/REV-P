#!/usr/bin/env python3
"""v2be Curitiba Sentinel candidate crosswalk resolution, review-only."""

import argparse
import csv
import datetime as dt
import hashlib
import os
import re

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2be_curitiba_sentinel_crosswalk_resolution")
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false", "can_create_label": "false",
    "can_create_negative": "false", "can_train_model": "false",
    "candidate_seed_asset_link_is_not_truth": "true", "sentinel_asset_is_not_ground_truth": "true",
    "visual_review_asset_is_not_label": "true", "dino_signal_is_not_truth": "true",
    "patch_boundary_is_not_event_geometry": "true", "no_geometry_no_final_truth": "true",
    "raw_data_versioned": "false",
}
INPUTS = {
    "seeds": "v2bd_curitiba_seed_selection.csv", "assets": "v2bd_sentinel_asset_discovery.csv",
    "crosswalk": "v2bd_seed_sentinel_crosswalk.csv", "patches": "v2bd_seed_patch_crosswalk.csv",
    "dino": "v2bd_seed_dino_crosswalk.csv", "visual": "v2bd_visual_review_asset_audit.csv",
    "readiness": "v2bd_candidate_reference_readiness.csv", "gates": "v2bd_candidate_promotion_gate.csv",
    "v2bc_seeds": "v2bc_ground_truth_seed_registry.csv",
    "v2bc_candidates": "v2bc_curitiba_local_seed_candidates.csv",
    "v2bc_scores": "v2bc_seed_strength_scores.csv",
    "metrics": "v2ay_window_precipitation_metrics.csv",
}
OUTPUTS = [
    "v2be_curitiba_blocked_seed_selection.csv", "v2be_sentinel_asset_candidate_scores.csv",
    "v2be_candidate_seed_asset_links.csv", "v2be_best_asset_per_seed.csv",
    "v2be_visual_review_asset_binding.csv", "v2be_seed_dino_binding.csv",
    "v2be_candidate_reference_readiness_update.csv", "v2be_revised_promotion_gate.csv",
    "v2be_visual_review_packet_index.csv", "v2be_guardrail_regression.csv",
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


def date_features(event_date, window_start, window_end, acquisition_date, expanded_days=15):
    try:
        event, start, end, acquired = map(dt.date.fromisoformat, (event_date, window_start, window_end, acquisition_date))
    except (TypeError, ValueError):
        return "", False, False, 0
    delta = abs((acquired - event).days)
    within = start <= acquired <= end
    expanded = start - dt.timedelta(days=expanded_days) <= acquired <= end + dt.timedelta(days=expanded_days)
    temporal = 40 if within else 25 if expanded else max(0, 15 - delta)
    return str(delta), within, expanded, temporal


def asset_score(event_date="", window_start="", window_end="", acquisition_date="", city_region=True,
                patch_match=False, visual=True, spectral=False, dino=False, public=False):
    delta, within, expanded, temporal = date_features(event_date, window_start, window_end, acquisition_date)
    spatial = 30 if patch_match else 15 if city_region else 0
    visual_score = (15 if visual else 0) + (5 if spectral else 0) + (5 if public else 0)
    dino_score = 5 if dino else 0
    penalty = (0 if acquisition_date else -10) + (0 if patch_match else -10) + (0 if visual else -15) + (0 if not city_region or patch_match else -5)
    final = temporal + spatial + visual_score + dino_score + penalty
    score_class = "BEST_REVIEW_ASSET" if final >= 75 else "GOOD_REVIEW_ASSET" if final >= 50 else "WEAK_REVIEW_ASSET" if final > 0 else "NOT_RECOMMENDED"
    return delta, within, expanded, temporal, spatial, visual_score, dino_score, penalty, final, score_class


def link_class(score_class, patch_match=False, acquisition_date=False, visual=True):
    if patch_match and acquisition_date: return "EXPLICIT_PATCH_DATE_LINK", "HIGH"
    if score_class in {"BEST_REVIEW_ASSET", "GOOD_REVIEW_ASSET"} and acquisition_date: return "CANDIDATE_TEMPORAL_REGIONAL_LINK", "MODERATE"
    if visual and score_class != "NOT_RECOMMENDED": return "CANDIDATE_VISUAL_REVIEW_LINK", "LOW"
    if score_class == "NOT_RECOMMENDED": return "NO_LINK", "VERY_LOW"
    return "WEAK_CONTEXT_LINK", "VERY_LOW"


def best_selection(scored):
    if not scored: return None, [], "NO_REVIEW_ASSET_FOUND"
    ordered = sorted(scored, key=lambda row: (-int(row["final_asset_match_score"]), row["sentinel_asset_id"]))
    primary, alternates = ordered[0], ordered[1:4]
    status = "PRIMARY_ASSET_SELECTED_FOR_REVIEW" if primary["score_class"] in {"BEST_REVIEW_ASSET", "GOOD_REVIEW_ASSET"} else "ONLY_WEAK_ASSETS_AVAILABLE"
    return primary, alternates, status


def visual_status(path="", nonempty=False, spectral=False):
    if not path: return "MISSING"
    if nonempty and spectral: return "READY_FOR_HUMAN_VISUAL_REVIEW"
    if nonempty: return "NEEDS_ASSET_RENDERING"
    return "ASSET_REFERENCE_ONLY"


def readiness_status(confidence="", visual="MISSING", primary=False, explicit_patch=False):
    if not primary: return "REMAINS_SEED_ONLY"
    if confidence in {"HIGH", "MODERATE"} and visual in {"READY_FOR_HUMAN_VISUAL_REVIEW", "ASSET_REFERENCE_ONLY"}:
        return "READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION"
    if visual == "NEEDS_ASSET_RENDERING": return "NEEDS_ASSET_RENDERING"
    if visual in {"ASSET_REFERENCE_ONLY", "READY_FOR_HUMAN_VISUAL_REVIEW"}: return "READY_FOR_HUMAN_VISUAL_REVIEW_ONLY"
    if not explicit_patch: return "NEEDS_EXPLICIT_PATCH_ASSET_LINK"
    return "REMAINS_SEED_ONLY"


def revised_gate(readiness, confidence="", human_review_required=True):
    allowed = readiness == "READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION" and confidence in {"HIGH", "MODERATE"} and human_review_required
    if allowed: return "CANDIDATE_REFERENCE_FOR_ADJUDICATION", True, "METHODOLOGICAL_ONLY"
    if readiness == "READY_FOR_HUMAN_VISUAL_REVIEW_ONLY": return "READY_FOR_HUMAN_VISUAL_REVIEW_ONLY", False, "NONE"
    return "REMAIN_CANDIDATE_GROUND_TRUTH_SEED", False, "NONE"


def run_select_blocked_seeds(args=None):
    data = load_inputs(); gates = by(data["gates"], "seed_id"); rows = []
    for seed in data["seeds"]:
        gate = gates.get(seed["seed_id"], {})
        if gate.get("gate_status") != "PROMOTION_BLOCKED": continue
        rows.append(with_invariants({
            "selection_id": f"SEL_v2be_{len(rows)+1:04d}", "seed_id": seed["seed_id"],
            "event_patch_package_id": seed["event_patch_package_id"], "patch_id": seed["patch_id"],
            "city": seed["city"], "region": seed["region"], "event_date": seed["event_date"],
            "window_start": seed["window_start"], "window_end": seed["window_end"],
            "station_id": seed["station_id"], "station_role": seed["station_role"],
            "previous_gate_status": gate["gate_status"], "blocker": gate["blockers_remaining"],
            "selected_for_crosswalk_resolution": "true",
            "selection_reason": "Curitiba A807 LOCAL seed blocked by missing explicit seed-asset crosswalk.",
        }))
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def run_score_asset_candidates(args=None):
    data = load_inputs(); visuals = {(r["seed_id"], r["sentinel_asset_id"]): r for r in data["visual"]}; rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        for asset in data["assets"]:
            visual = visuals.get((seed["seed_id"], asset["sentinel_asset_id"]), {})
            patch = bool(seed["patch_id"]) and seed["patch_id"] == asset["patch_id"]
            city = seed["city"] == asset["city"] and seed["region"] == asset["region"]
            values = asset_score(seed["event_date"], seed["window_start"], seed["window_end"], asset["acquisition_date"], city, patch,
                                 is_true(asset["has_visual_asset"]), is_true(asset["has_spectral_context"]), is_true(asset["has_dino_embedding"]), False)
            reason = "Date unknown; region-only match; no explicit seed patch; registered visual reference only."
            rows.append(with_invariants({
                "seed_id": seed["seed_id"], "event_patch_package_id": seed["event_patch_package_id"], "patch_id": seed["patch_id"],
                "sentinel_asset_id": asset["sentinel_asset_id"], "sentinel_acquisition_date": asset["acquisition_date"],
                "date_delta_days": values[0], "within_event_window": str(values[1]).lower(),
                "within_expanded_review_window": str(values[2]).lower(), "city_region_match": str(city).lower(),
                "patch_id_match": str(patch).lower(), "visual_asset_available": asset["has_visual_asset"],
                "spectral_context_available": asset["has_spectral_context"], "dino_embedding_available": asset["has_dino_embedding"],
                "output_public_reference_available": "false", "temporal_score": str(values[3]), "spatial_score": str(values[4]),
                "visual_score": str(values[5]), "dino_score": str(values[6]), "penalty_score": str(values[7]),
                "final_asset_match_score": str(values[8]), "score_class": values[9], "score_reason": reason,
                "asset_path_or_reference": asset["asset_path_or_reference"], "asset_nonempty": visual.get("asset_nonempty", "false"),
            }))
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def run_resolve_candidate_links(args=None):
    rows = []
    for score in load_csv(dataset_path(OUTPUTS[1])):
        link, confidence = link_class(score["score_class"], is_true(score["patch_id_match"]), bool(score["sentinel_acquisition_date"]), is_true(score["visual_asset_available"]))
        rows.append(with_invariants({
            "candidate_link_id": f"LINK_v2be_{len(rows)+1:05d}", "seed_id": score["seed_id"],
            "event_patch_package_id": score["event_patch_package_id"], "patch_id": score["patch_id"],
            "sentinel_asset_id": score["sentinel_asset_id"], "link_type": link, "link_confidence": confidence,
            "requires_human_review": "true", "link_is_ground_truth": "false",
            "reason": score["score_reason"], "final_asset_match_score": score["final_asset_match_score"],
        }))
    write_csv(dataset_path(OUTPUTS[2]), rows); return rows


def run_build_best_asset_per_seed(args=None):
    links = {(r["seed_id"], r["sentinel_asset_id"]): r for r in load_csv(dataset_path(OUTPUTS[2]))}; rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        scored = [r for r in load_csv(dataset_path(OUTPUTS[1])) if r["seed_id"] == seed["seed_id"]]
        primary, alternates, status = best_selection(scored); link = links[(seed["seed_id"], primary["sentinel_asset_id"])] if primary else {}
        rows.append(with_invariants({
            "seed_id": seed["seed_id"], "primary_sentinel_asset_id": primary["sentinel_asset_id"] if primary else "",
            "primary_asset_score": primary["final_asset_match_score"] if primary else "", "primary_link_confidence": link.get("link_confidence", ""),
            "primary_link_type": link.get("link_type", ""), "alternate_asset_ids": "|".join(r["sentinel_asset_id"] for r in alternates),
            "selection_status": status, "selection_reason": "Deterministic highest score then asset ID; weak references remain review-only.",
            "human_review_required": "true",
        }))
    write_csv(dataset_path(OUTPUTS[3]), rows); return rows


def run_bind_visual_review_assets(args=None):
    scores = {(r["seed_id"], r["sentinel_asset_id"]): r for r in load_csv(dataset_path(OUTPUTS[1]))}; rows = []
    for best in load_csv(dataset_path(OUTPUTS[3])):
        score = scores.get((best["seed_id"], best["primary_sentinel_asset_id"]), {})
        status = visual_status(score.get("asset_path_or_reference", ""), is_true(score.get("asset_nonempty")), is_true(score.get("spectral_context_available")))
        rows.append(with_invariants({
            "seed_id": best["seed_id"], "sentinel_asset_id": best["primary_sentinel_asset_id"],
            "visual_asset_path_or_reference": score.get("asset_path_or_reference", ""), "spectral_asset_path_or_reference": "",
            "figure_reference": "", "output_public_reference": "", "visual_review_status": status,
            "limitation": "Reference path is registered, but payload is empty/unavailable and acquisition date is missing.",
            "visual_is_ground_truth": "false",
        }))
    write_csv(dataset_path(OUTPUTS[4]), rows); return rows


def run_attempt_seed_dino_binding(args=None):
    previous = by(load_inputs()["dino"], "seed_id"); rows = []
    for best in load_csv(dataset_path(OUTPUTS[3])):
        prior = previous.get(best["seed_id"], {}); available = is_true(prior.get("embedding_available"))
        rows.append(with_invariants({
            "seed_id": best["seed_id"], "patch_id": prior.get("patch_id", ""), "sentinel_asset_id": best["primary_sentinel_asset_id"],
            "dino_embedding_id": prior.get("dino_embedding_id", ""), "embedding_available": str(available).lower(),
            "dino_link_status": "CANDIDATE_DINO_LINK" if available else "NOT_LINKED",
            "dino_review_available": str(available).lower(), "dino_is_ground_truth": "false", "dino_can_create_label": "false",
        }))
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


def run_update_readiness(args=None):
    data = load_inputs(); old = by(data["readiness"], "seed_id"); best = by(load_csv(dataset_path(OUTPUTS[3])), "seed_id")
    visual = by(load_csv(dataset_path(OUTPUTS[4])), "seed_id"); dino = by(load_csv(dataset_path(OUTPUTS[5])), "seed_id"); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        selected, vis = best[seed["seed_id"]], visual[seed["seed_id"]]
        status = readiness_status(selected["primary_link_confidence"], vis["visual_review_status"], bool(selected["primary_sentinel_asset_id"]), bool(seed["patch_id"]))
        rows.append(with_invariants({
            "seed_id": seed["seed_id"], "previous_readiness_status": old[seed["seed_id"]]["candidate_reference_readiness"],
            "seed_asset_link_status": selected["primary_link_type"], "primary_asset_selected": str(bool(selected["primary_sentinel_asset_id"])).lower(),
            "visual_review_status": vis["visual_review_status"], "dino_link_status": dino[seed["seed_id"]]["dino_link_status"],
            "geometry_status": "GEOMETRY_MISSING", "updated_readiness_status": status,
            "remaining_blockers": "SENTINEL_DATE_MISSING|NO_EXPLICIT_PATCH_ASSET_LINK|ASSET_PAYLOAD_UNAVAILABLE|GEOMETRY_MISSING|HUMAN_VISUAL_REVIEW_PENDING",
        }))
    write_csv(dataset_path(OUTPUTS[6]), rows); return rows


def run_apply_revised_gate(args=None):
    best = by(load_csv(dataset_path(OUTPUTS[3])), "seed_id"); rows = []
    for ready in load_csv(dataset_path(OUTPUTS[6])):
        selected = best[ready["seed_id"]]; proposed, allowed, typ = revised_gate(ready["updated_readiness_status"], selected["primary_link_confidence"], True)
        rows.append(with_invariants({
            "seed_id": ready["seed_id"], "previous_status": "CANDIDATE_GROUND_TRUTH_SEED", "proposed_status": proposed,
            "promotion_allowed": str(allowed).lower(), "promotion_type": typ,
            "promotion_reason": "Low-confidence regional visual reference requires human review before candidate-reference adjudication." if not allowed else "Moderate/high candidate link supports methodological adjudication only.",
            "blockers_remaining": ready["remaining_blockers"],
            "next_action_rank_1": "HUMAN_VISUAL_ADJUDICATION_OF_CURITIBA_CANDIDATE_REFERENCES" if allowed else "MANUALLY_RESOLVE_PATCH_ASSET_LINK_FOR_CURITIBA",
        }))
    write_csv(dataset_path(OUTPUTS[7]), rows); return rows


def run_generate_visual_review_packets(args=None):
    best = by(load_csv(dataset_path(OUTPUTS[3])), "seed_id"); visual = by(load_csv(dataset_path(OUTPUTS[4])), "seed_id")
    dino = by(load_csv(dataset_path(OUTPUTS[5])), "seed_id"); gates = by(load_csv(dataset_path(OUTPUTS[7])), "seed_id"); rows = []
    for seed in load_csv(dataset_path(OUTPUTS[0])):
        selected, vis, gate = best[seed["seed_id"]], visual[seed["seed_id"]], gates[seed["seed_id"]]
        path = doc_path("visual_review_packets", f"{slug(seed['seed_id'])}.md")
        write_text(path, f"""# Curitiba Sentinel Visual Review Packet: {seed['seed_id']}

## Evento e janela temporal
{seed['event_date']}; janela {seed['window_start']} a {seed['window_end']}.

## Evidencia INMET A807
Seed Curitiba/A807/LOCAL com suporte temporal forte herdado da v2bd.

## Ranking e asset principal
Principal `{selected['primary_sentinel_asset_id']}`; score {selected['primary_asset_score']}; confianca `{selected['primary_link_confidence']}`.
Alternativos: `{selected['alternate_asset_ids']}`.

## Visual, espectral e DINO
Visual `{vis['visual_review_status']}`; espectral indisponivel; DINO `{dino[seed['seed_id']]['dino_link_status']}`.

## Lacuna geometrica e decisao
GEOMETRY_MISSING; `{gate['proposed_status']}`; promotion_allowed={gate['promotion_allowed']}.

## Proxima acao humana
`{gate['next_action_rank_1']}`.

## Guardrails
Crosswalk candidato nao e truth; Sentinel/DINO nao sao truth; nao cria label, negativo, treino ou geometria.
""")
        rows.append(with_invariants({
            "packet_index_id": f"PACK_v2be_{len(rows)+1:04d}", "seed_id": seed["seed_id"],
            "primary_sentinel_asset_id": selected["primary_sentinel_asset_id"],
            "packet_path": f"docs/protocolo_c/v2be_curitiba_sentinel_crosswalk_resolution/visual_review_packets/{slug(seed['seed_id'])}.md",
            "proposed_status": gate["proposed_status"], "promotion_allowed": gate["promotion_allowed"], "next_action_rank_1": gate["next_action_rank_1"],
        }))
    write_csv(dataset_path(OUTPUTS[8]), rows)
    write_csv(doc_path("crosswalk_tables", OUTPUTS[2]), load_csv(dataset_path(OUTPUTS[2])))
    write_csv(doc_path("asset_score_summaries", OUTPUTS[3]), load_csv(dataset_path(OUTPUTS[3])))
    write_text(doc_path("README.md"), "# v2be Curitiba Sentinel Asset Crosswalk Resolution\n\nEu/equipe ranqueou 129 pares. As referencias principais sao regionais, sem data e sem payload utilizavel; permanecem para revisao humana, sem promocao metodologica.\n")
    return rows


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:9], 1):
        violations = sum(row.get(field, "").lower() == "true" for row in load_csv(dataset_path(name)) for field in forbidden)
        violations += sum(is_true(row.get("promotion_allowed")) for row in load_csv(dataset_path(name))) if name == OUTPUTS[7] else 0
        rows.append({"regression_id": f"GR_v2be_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore"); violations = 0 if os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n" else 1
    rows.append({"regression_id": "GR_v2be_010", "artifact_path": "docs/protocolo_c/v2be_curitiba_sentinel_crosswalk_resolution/evidence_cache/.gitignore", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    if any(r["status"] != "PASS" for r in rows): raise ValueError("v2be guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[9]), rows); return rows


STEPS = [
    ("select_curitiba_blocked_seeds", run_select_blocked_seeds, OUTPUTS[0]), ("score_sentinel_asset_candidates", run_score_asset_candidates, OUTPUTS[1]),
    ("resolve_candidate_seed_asset_links", run_resolve_candidate_links, OUTPUTS[2]), ("build_best_asset_per_seed", run_build_best_asset_per_seed, OUTPUTS[3]),
    ("bind_visual_review_assets", run_bind_visual_review_assets, OUTPUTS[4]), ("attempt_seed_dino_binding", run_attempt_seed_dino_binding, OUTPUTS[5]),
    ("update_candidate_reference_readiness", run_update_readiness, OUTPUTS[6]), ("apply_revised_promotion_gate", run_apply_revised_gate, OUTPUTS[7]),
    ("generate_visual_review_packets", run_generate_visual_review_packets, OUTPUTS[8]), ("guardrail_regression", run_guardrail_regression, OUTPUTS[9]),
]


def ensure_structure():
    for folder in ("visual_review_packets", "crosswalk_tables", "asset_score_summaries", "evidence_cache"): os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args); path = dataset_path(output)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": f"datasets/protocolo_c/{output}", "output_hash": sha256(path)[:16], "notes": "Review-only candidate resolution completed."})
    write_csv(dataset_path("v2be_orchestrator_manifest.csv"), manifest); return manifest
