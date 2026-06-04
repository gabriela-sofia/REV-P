#!/usr/bin/env python3
"""
v1uj — Observed Candidate Promotion Audit

Consolida candidatos a geometria observada a partir do inventory focado
(artefatos baixados) e do registry de layers GeoSGB (metadata de service) e
aplica 13 gates.

Gates:
  G1 public_official_traceable
  G2 event_specific_or_regionally_relevant
  G3 artifact_downloaded_or_service_metadata_available
  G4 geometry_or_coordinate_table_available
  G5 crs_or_coordinate_reference_available
  G6 event_date_available
  G7 event_date_compatible
  G8 hazard_or_phenomenon_available
  G9 phenomenon_not_only_susceptibility
  G10 not_static_map_only
  G11 supervisor_review_required        (sempre FAIL / pendente)
  G12 overlay_not_executed              (sempre FAIL para ground reference)
  G13 label_forbidden                   (sempre FAIL)

Status maximo: OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW.
NUNCA: GROUND_REFERENCE / GROUND_TRUTH / LABEL.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1uj"

GATE_NAMES = [
    "public_official_traceable",
    "event_specific_or_regionally_relevant",
    "artifact_downloaded_or_service_metadata_available",
    "geometry_or_coordinate_table_available",
    "crs_or_coordinate_reference_available",
    "event_date_available",
    "event_date_compatible",
    "hazard_or_phenomenon_available",
    "phenomenon_not_only_susceptibility",
    "not_static_map_only",
    "supervisor_review_required",
    "overlay_not_executed",
    "label_forbidden",
]

AUDIT_COLUMNS = (
    ["promotion_audit_id", "event_id", "source_tag", "candidate_ref"]
    + [f"G{i:02d}_{GATE_NAMES[i-1]}" for i in range(1, 14)]
    + ["gates_passed", "gates_failed", "max_status",
       "can_create_ground_reference", "can_create_training_label", "notes"]
)

EVENT_YEARS = {
    "PET_2022_02_15": "2022",
    "PET_2024_03_21_28": "2024",
    "REC_2022_05_24_30": "2022",
}


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(evidence):
    """Avalia os 13 gates sobre um dict de evidencia normalizada."""
    g = {}
    g["G01"] = "PASS" if evidence["traceable"] else "FAIL"
    g["G02"] = "PASS" if evidence["event_or_regional"] else "FAIL"
    g["G03"] = "PASS" if evidence["downloaded_or_metadata"] else "FAIL"
    g["G04"] = "PASS" if evidence["geometry_or_coords"] else "FAIL"
    g["G05"] = "PASS" if evidence["crs"] else "NEEDS_REVIEW"
    g["G06"] = "PASS" if evidence["date"] else "FAIL"
    g["G07"] = "PASS" if evidence["date_compatible"] else "NEEDS_REVIEW"
    g["G08"] = "PASS" if evidence["hazard"] else "FAIL"
    g["G09"] = "PASS" if evidence["not_only_susceptibility"] else "FAIL"
    g["G10"] = "PASS" if evidence["not_static_only"] else "FAIL"
    g["G11"] = "FAIL"  # supervisor review pendente
    g["G12"] = "FAIL"  # overlay nao executado (bloqueia ground reference)
    g["G13"] = "FAIL"  # label proibido

    passed = sum(1 for v in g.values() if v == "PASS")
    failed = sum(1 for v in g.values() if v == "FAIL")

    g1_g10_ok = all(g[f"G{i:02d}"] == "PASS" for i in range(1, 11))
    if g1_g10_ok:
        max_status = "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"
    elif evidence["geometry_or_coords"]:
        max_status = "CANDIDATE_WITH_BLOCKERS"
    else:
        max_status = "NOT_A_GEOMETRY_CANDIDATE"
    return g, passed, failed, max_status


def evidence_from_inventory(row):
    event_id = row.get("event_id", "")
    has_geom = row.get("has_geometry") == "true"
    classification = row.get("classification", "")
    hazard = row.get("hazard_term_detected") == "true"
    suscept = row.get("susceptibility_term_detected") == "true"
    date = row.get("date_term_detected") == "true"
    crs = bool(row.get("crs"))
    asset_type = row.get("asset_type", "")
    cols_l = (row.get("columns_detected", "") or "").lower()
    is_coord_table = (asset_type == "tabular"
                      and (("latitude" in cols_l and "longitude" in cols_l)
                           or ("lat" in cols_l and ("lon" in cols_l or "lng" in cols_l))
                           or ("|x|" in f"|{cols_l}|" and "|y|" in f"|{cols_l}|")
                           or "geometria" in cols_l or "geometry" in cols_l))
    static_only = classification in ("static_map", "document_only")

    year = EVENT_YEARS.get(event_id, "")
    date_compatible = bool(date and (not year or year in (row.get("internal_path", "") + row.get("columns_detected", ""))))

    return {
        "traceable": bool(row.get("source_tag")),
        "event_or_regional": bool(event_id) or row.get("locality_term_detected") == "true",
        "downloaded_or_metadata": row.get("inventory_status") == "INVENTORIED",
        "geometry_or_coords": has_geom or is_coord_table,
        "crs": crs,
        "date": date,
        "date_compatible": date_compatible,
        "hazard": hazard,
        "not_only_susceptibility": not (suscept and not hazard)
                                   and classification != "CONTEXT_ONLY",
        "not_static_only": not static_only,
    }


def evidence_from_geosgb(row):
    event_id = row.get("event_id", "")
    is_observed = row.get("is_observed_occurrence_candidate") == "true"
    is_contextual = row.get("is_contextual_layer") == "true"
    has_geom = bool(row.get("geometry_type"))
    crs = bool(row.get("spatial_reference"))
    fields_l = (row.get("fields", "") or "").lower()
    has_date_field = any(t in fields_l for t in ("data", "date", "ano"))

    return {
        "traceable": bool(row.get("service_url")),
        "event_or_regional": bool(event_id),
        "downloaded_or_metadata": True,  # metadata de service disponivel
        "geometry_or_coords": has_geom,
        "crs": crs,
        "date": has_date_field,
        "date_compatible": has_date_field,
        "hazard": is_observed,
        "not_only_susceptibility": is_observed and not is_contextual,
        "not_static_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description="v1uj — Observed Candidate Promotion Audit")
    parser.add_argument("--inventory", default="datasets/protocolo_c/v1uj_focused_artifact_inventory.csv")
    parser.add_argument("--geosgb", default="datasets/protocolo_c/v1uj_geosgb_layer_registry.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_observed_candidate_promotion_audit.csv")
    args = parser.parse_args()

    inventory = load_csv(args.inventory)
    geosgb = load_csv(args.geosgb)

    rows = []
    seq = 0

    for inv in inventory:
        if inv.get("asset_type") in ("archive",):
            continue  # o conteudo interno do zip ja gera linhas proprias
        ev = evidence_from_inventory(inv)
        gates, passed, failed, max_status = evaluate(ev)
        rows.append(_row(seq, inv.get("event_id", ""), inv.get("source_tag", ""),
                         inv.get("inventory_id", ""), gates, passed, failed, max_status))
        seq += 1

    for layer in geosgb:
        if not layer.get("geometry_type"):
            continue
        ev = evidence_from_geosgb(layer)
        gates, passed, failed, max_status = evaluate(ev)
        rows.append(_row(seq, layer.get("event_id", ""), "geosgb",
                         layer.get("geosgb_record_id", ""), gates, passed, failed, max_status))
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(1 for r in rows if r["max_status"] == "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW")
    print(f"[Observed Candidate Promotion Audit v1uj] {len(rows)} evaluated | "
          f"observed_for_review={ready}")
    print(f"  max_status=OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW | ground_reference=false")
    print(f"\nRegistry: {args.out}")


def _row(seq, event_id, source_tag, ref, gates, passed, failed, max_status):
    row = {
        "promotion_audit_id": f"PROM_{PROTOCOL_VERSION}_{seq:04d}",
        "event_id": event_id, "source_tag": source_tag, "candidate_ref": ref,
    }
    for i in range(1, 14):
        row[f"G{i:02d}_{GATE_NAMES[i-1]}"] = gates[f"G{i:02d}"]
    row["gates_passed"] = str(passed)
    row["gates_failed"] = str(failed)
    row["max_status"] = max_status
    row["can_create_ground_reference"] = "false"
    row["can_create_training_label"] = "false"
    row["notes"] = ""
    return row


if __name__ == "__main__":
    main()
