#!/usr/bin/env python3
"""
v1ug — Human Review Package Builder

Consolidates all available evidence per event into a review package.
No promotion, no ground reference, no label. Review-only.
"""

import argparse
import csv
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ug"

PACKAGE_COLUMNS = [
    "review_package_id", "event_id", "region", "city", "event_start", "event_end",
    "current_protocol_level", "hydromet_anchor_status", "station_evidence_status",
    "official_sources_count", "local_only_assets_count", "has_event_specific_document",
    "has_official_station_series", "has_observed_geometry", "has_phenomenon_separation",
    "has_patch_overlay", "has_supervisor_review", "review_package_status",
    "reviewer_task", "cannot_promote_reason", "next_required_evidence",
]

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "supervisor_review_completed": False,
    "no_overlay_executed": True,
}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_yaml(path: str) -> dict:
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def determine_package_status(event: dict, hydromet: dict, v1ue_score: dict) -> tuple[str, str, str]:
    hazard = event.get("hazard_scope", "")
    h_level = hydromet.get("hydromet_evidence_level", "NO_STATION_DATA")
    has_series = hydromet.get("has_official_station_series") == "true"
    has_precip = hydromet.get("has_precipitation_during_event") == "true"

    if hazard == "mixed":
        status = "BLOCKED_PHENOMENON_SEPARATION_REQUIRED"
        task = "Separar inundação de deslizamento na evidência. Solicitar SGB/CPRM e DRM-RJ."
        next_ev = "Geodata de campo SGB/CPRM ou DRM-RJ com separação por fenômeno"
    elif h_level == "BLOCKED_INSUFFICIENT_COVERAGE":
        status = "BLOCKED_INSUFFICIENT_COVERAGE"
        task = "Cobertura INMET insuficiente. Solicitar Cemaden e ANA para esta janela."
        next_ev = "Série de pluviômetro Cemaden ou estação ANA para o período do evento"
    elif not has_series:
        status = "READY_FOR_FORMAL_REQUEST"
        task = "Solicitar série INMET ou Cemaden para o evento."
        next_ev = "Série oficial de precipitação/hidrologia do evento"
    elif has_series and has_precip:
        status = "READY_FOR_FORMAL_REQUEST"
        task = "Série hidrometeorológica disponível. Próximo passo: obter geometria observada."
        next_ev = "Geometria observada (ocorrências georreferenciadas) da Defesa Civil / COMPDEC"
    else:
        status = "READY_FOR_FORMAL_REQUEST"
        task = "Solicitar evidência observacional com geometria."
        next_ev = "Ocorrências georreferenciadas com localidade e fenômeno"

    cannot_promote = (
        "has_observed_geometry=false; has_patch_overlay=false; "
        "has_supervisor_review=false; ground_reference_gates_incomplete"
    )
    return status, task, next_ev


def main():
    parser = argparse.ArgumentParser(description="v1ug — Human Review Package Builder")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--hydromet-scorecard", default="datasets/protocolo_c/v1uf_event_hydromet_scorecard.csv")
    parser.add_argument("--v1ue-scorecard", default="datasets/protocolo_c/v1ue_event_evidence_scorecard.csv")
    parser.add_argument("--v1ue-resolution", default="datasets/protocolo_c/v1ue_official_dataset_resolution_registry.csv")
    parser.add_argument("--assets", default="datasets/protocolo_c/v1uf_station_series_asset_registry.csv")
    parser.add_argument("--policy", default="configs/protocolo_c/v1ug_review_package_policy.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_event_review_package_registry.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    hydromet_sc = {r["event_id"]: r for r in load_csv(args.hydromet_scorecard)}
    v1ue_sc = {r["event_id"]: r for r in load_csv(args.v1ue_scorecard)}
    resolutions = load_csv(args.v1ue_resolution)
    assets = load_csv(args.assets)

    res_by_event = {}
    for r in resolutions:
        res_by_event.setdefault(r["event_id"], []).append(r)

    extracted_by_event = {}
    for a in assets:
        if a.get("extraction_status") == "EXTRACTED":
            extracted_by_event.setdefault(a["event_id"], []).append(a)

    rows = []
    for seq, event in enumerate(events):
        event_id = event["event_id"]
        hsc = hydromet_sc.get(event_id, {})
        vsc = v1ue_sc.get(event_id, {})
        ev_res = res_by_event.get(event_id, [])
        ev_assets = extracted_by_event.get(event_id, [])

        official_sources = len([r for r in ev_res if r.get("resolution_status") not in
                                 ("GENERIC_PORTAL", "HTTP_ERROR", "DRY_RUN")])
        has_series = hsc.get("has_official_station_series") == "true"
        has_precip_ev = hsc.get("has_precipitation_during_event") == "true"
        hydromet_level = hsc.get("hydromet_evidence_level", "NO_STATION_DATA")

        status, task, next_ev = determine_package_status(event, hsc, vsc)

        rows.append({
            "review_package_id": f"PKG_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id,
            "region": event.get("region", ""),
            "city": event.get("city", ""),
            "event_start": event.get("start_date", ""),
            "event_end": event.get("end_date", ""),
            "current_protocol_level": event.get("current_level", ""),
            "hydromet_anchor_status": hydromet_level,
            "station_evidence_status": "SERIES_EXTRACTED" if ev_assets else "NO_SERIES",
            "official_sources_count": str(official_sources),
            "local_only_assets_count": str(len(ev_assets)),
            "has_event_specific_document": "false",
            "has_official_station_series": str(has_series).lower(),
            "has_observed_geometry": "false",
            "has_phenomenon_separation": str(
                event.get("hazard_scope") in ("flood", "inundation", "urban_flooding")
            ).lower(),
            "has_patch_overlay": "false",
            "has_supervisor_review": "false",
            "review_package_status": status,
            "reviewer_task": task,
            "cannot_promote_reason": (
                "has_observed_geometry=false; has_patch_overlay=false; "
                "supervisor_review_completed=false"
            ),
            "next_required_evidence": next_ev,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PACKAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Human Review Package Builder v1ug] {len(rows)} packages")
    for r in rows:
        print(f"  {r['event_id']}: {r['review_package_status']}")
    print(f"\n  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
