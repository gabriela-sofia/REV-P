"""
SUSC-13C consolidated event-to-patch linkage (review-only).

Links the SUSC-13C consolidated catalog (13A+13B+13C live) to SUSC-03 patches
using explicit geometry only (point-in-bbox, bbox intersection, proximity
buffer). Strong/moderate events with sufficient geometry may support
observational evaluation; nothing here is ground truth or training material.
Reuses the SUSC-13B linkage primitives.

Writes:
  - datasets/suscetibilidade/susc_13c_event_patch_linkage_v1.csv
  - outputs_public/suscetibilidade/SUSC_13C_event_patch_linkage_summary.csv
  - outputs_public/suscetibilidade/SUSC_13C_event_patch_linkage_limitations.json
  - outputs_public/suscetibilidade/SUSC_13C_event_patch_linkage_geojson.geojson
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import build_susc_13b_auto_event_patch_linkage as l13b  # noqa: E402
from susc_geometry import bbox_intersect, parse_bbox  # noqa: E402
from susc_io import read_csv, rel, write_csv, write_json  # noqa: E402

CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_13c_consolidated_observed_event_catalog_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13c_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_event_patch_linkage_summary.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_event_patch_linkage_limitations.json"
GEOJSON = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_event_patch_linkage_geojson.geojson"

MANDATORY = (
    "O SUSC-13C-LIVE executa aquisição online real de fontes oficiais/rastreáveis para tentar "
    "materializar eventos observados de alagamento/inundação com data e geometria. Mesmo quando "
    "eventos fortes ou moderados são encontrados, todos os vínculos permanecem review-only, sem "
    "ground truth, sem treino supervisionado, sem score v7 automático e sem uso operacional preditivo."
)

OBSERVED = l13b.OBSERVED_LEVELS
FIELDS = l13b.FIELDS


def main() -> int:
    print("=" * 60)
    print("SUSC-13C Event-Patch Linkage")
    print("=" * 60)
    for path in (l13b.MATRIX, l13b.SCORE, CATALOG):
        if not path.exists():
            print(f"STOP: missing input: {rel(path)}")
            return 1
    patches = l13b._load_patches()
    scores = l13b._load_scores()
    events = read_csv(CATALOG)
    links: list[dict] = []
    for event in events:
        if event.get("preferred_record", "true") == "false":
            continue
        region = (event.get("region") or "").lower()
        level = event.get("evidence_level", "")
        geom = (event.get("geometry_type") or "").lower()
        bbox = parse_bbox(event.get("bbox", ""))
        patches_here = patches.get(region, [])

        if level not in OBSERVED:
            relation = "same_region_period_context" if (event.get("event_date") or event.get("event_period_start")) else "insufficient_for_patch_link"
            links.append(l13b._row(event, "REGION_LEVEL_NO_PATCH_RESOLUTION", relation, None, "", "",
                                   "Contexto fraco/documental mantido sem resolucao por patch.",
                                   "Risco, alerta, administrativo ou documental nao e evento observado."))
            continue
        matched = False
        if geom in {"polygon", "multipolygon", "bbox"} and bbox:
            for pid, pbb, _cen in patches_here:
                if bbox_intersect(bbox, pbb):
                    score, score_class = scores.get(pid, ("", ""))
                    relation = "strong_polygon_intersects_patch" if level == "strong_observed_flood_polygon" else "moderate_bbox_intersects_patch"
                    links.append(l13b._row(event, pid, relation, None, score, score_class,
                                           "Geometria de evento intersecta bbox do patch.",
                                           "Interseccao por bbox e aproximacao review-only; nao e ground truth."))
                    matched = True
        else:
            plinks = l13b._point_links(event, patches_here, scores)
            links.extend(plinks)
            matched = bool(plinks)
        if not matched:
            links.append(l13b._row(event, "REGION_LEVEL_NO_PATCH_RESOLUTION", "same_region_period_context", None, "", "",
                                   "Evento observado/moderado sem sobreposicao/proximidade de patch nesta matriz.",
                                   "Associacao regional/periodo apenas; nao confirma patch."))

    for i, row in enumerate(links):
        row["linkage_id"] = f"S13CLINK_{i:05d}"
    write_csv(LINKAGE, links, FIELDS)

    rel_counts = Counter(r["spatial_relation"] for r in links)
    conf_counts = Counter(r["linkage_confidence"] for r in links)
    eval_links = sum(1 for r in links if r["can_be_used_for_observational_evaluation"] == "true")
    n_pref = sum(1 for e in events if e.get("preferred_record", "true") != "false")
    summary_rows = [
        {"metric": "total_events_preferred", "value": n_pref, "review_only": "true"},
        {"metric": "total_links", "value": len(links), "review_only": "true"},
        {"metric": "links_by_spatial_relation", "value": json.dumps(dict(rel_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "links_by_confidence", "value": json.dumps(dict(conf_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "strong_links", "value": sum(1 for r in links if r["linkage_confidence"] == "strong"), "review_only": "true"},
        {"metric": "moderate_links", "value": sum(1 for r in links if r["linkage_confidence"] == "moderate"), "review_only": "true"},
        {"metric": "observational_evaluation_links", "value": eval_links, "review_only": "true"},
        {"metric": "training_allowed_count", "value": sum(1 for r in links if r["allowed_for_training"] == "true"), "review_only": "true"},
        {"metric": "ground_truth_count", "value": sum(1 for r in links if r["can_be_ground_truth"] == "true"), "review_only": "true"},
    ]
    write_csv(SUMMARY, summary_rows, ["metric", "value", "review_only"])

    features = [f for f in (l13b._geojson_feature(r, patches) for r in links if r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION") if f]
    write_json(GEOJSON, {
        "type": "FeatureCollection", "name": "susc_13c_event_patch_linkage",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "review_only": True, "can_be_ground_truth": False, "allowed_for_training": False,
        "features": features,
    })

    write_json(LIMITS, {
        "artifact": "SUSC-13C consolidated event patch linkage",
        "n_events_total": len(events), "n_events_preferred": n_pref, "n_links": len(links),
        "spatial_relation_counts": dict(rel_counts), "linkage_confidence_counts": dict(conf_counts),
        "observational_evaluation_links": eval_links,
        "can_be_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "score_v7_created": False, "model_persisted": False,
        "key_limitations": [
            "Aquisicao online real foi executada; ainda assim tudo permanece review-only.",
            "Eventos fortes/moderados nao sao footprint validado nem rotulo de treino.",
            "Risco, alerta e registros administrativos nao sao evento observado.",
            "Interseccao por bbox e ponto-dentro-bbox sao aproximacoes para revisao.",
            "Sem ground truth, sem treino supervisionado e sem score v7.",
        ],
        "mandatory_statement": MANDATORY,
    })

    print(f"linkage rows: {len(links)} | eval links: {eval_links} | geojson features: {len(features)}")
    print("relations:", dict(rel_counts))
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
