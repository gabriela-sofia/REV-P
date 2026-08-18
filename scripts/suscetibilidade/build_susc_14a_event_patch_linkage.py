"""
SUSC-14A FASE 8 - rescued event-to-patch linkage (review-only).

Links the SUSC-14A rescued catalog to SUSC-03 patches using explicit geometry /
official geocoded points only. Footprint polygons and official points may yield
strong/moderate patch relations; geocoded points yield moderate relations;
neighborhood-only records stay at same_neighborhood_context and never resolve to
a patch. ``can_be_used_for_observational_evaluation`` is true only for
strong/moderate evidence from an official/traceable source with sufficient
geometry/geocoding and a real (non-regional) spatial relation.

Writes:
  - datasets/suscetibilidade/susc_14a_event_patch_linkage_v1.csv
  - outputs_public/suscetibilidade/SUSC_14A_event_patch_linkage_summary.csv
  - outputs_public/suscetibilidade/SUSC_14A_event_patch_linkage_limitations.json
  - outputs_public/suscetibilidade/SUSC_14A_event_patch_linkage_geojson.geojson
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
from susc_geometry import bbox_intersect, haversine_m, parse_bbox  # noqa: E402
from susc_io import read_csv, rel, write_csv, write_json  # noqa: E402

CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_14a_rescued_observed_event_catalog_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_14a_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_event_patch_linkage_summary.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_event_patch_linkage_limitations.json"
GEOJSON = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_event_patch_linkage_geojson.geojson"

MANDATORY = (
    "O SUSC-14A tenta resgatar evidencia observacional a partir de registros oficiais sem coordenada "
    "explicita, usando apenas referencias espaciais oficiais/rastreaveis e footprints publicos quando "
    "disponiveis. Eventos geocodificados por referencia oficial permanecem review-only, nao criam "
    "ground truth, nao liberam treino supervisionado e nao autorizam score v7 automatico."
)

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {"moderate_official_occurrence_point", "moderate_official_occurrence_geocoded",
            "moderate_official_street_segment_match", "moderate_official_flood_bbox"}
OBSERVED = STRONG | MODERATE
WEAK = {"weak_official_neighborhood_context", "weak_risk_area_context", "weak_alert_context"}
NEAR_BUFFER_DEG = 0.005

FIELDS = [
    "linkage_id", "event_id", "patch_id", "region", "event_date", "event_type",
    "evidence_level", "georeferencing_status", "spatial_relation", "temporal_precision",
    "distance_m", "patch_score_v6", "patch_score_class_v6", "source_confidence",
    "parse_confidence", "geocoding_confidence", "linkage_confidence",
    "observed_event_strength", "can_be_used_for_observational_evaluation",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training",
    "review_only", "requires_manual_review",
    "interpretation", "limitations",
]


def _strength(level: str) -> str:
    if level in STRONG:
        return "strong"
    if level in MODERATE:
        return "moderate"
    if level in WEAK or level.startswith("weak_"):
        return "weak"
    return "documentary_or_rejected"


def _link_conf(relation: str, level: str) -> str:
    if relation in {"strong_polygon_intersects_patch", "strong_point_inside_patch"} and level in STRONG:
        return "strong"
    if relation in {"moderate_point_inside_patch", "moderate_geocoded_point_inside_patch",
                    "moderate_street_segment_intersects_patch", "moderate_bbox_intersects_patch"} and level in OBSERVED:
        return "moderate"
    if relation == "near_patch_buffer_candidate":
        return "weak"
    return "very_weak"


def _can_eval(level: str, relation: str, georef: str) -> str:
    # official/traceable + observed + real spatial relation + not neighborhood-only
    if level not in OBSERVED:
        return "false"
    if relation in {"same_neighborhood_context", "same_region_period_context", "insufficient_for_patch_link"}:
        return "false"
    if georef in {"official_neighborhood_only", "text_only_no_geometry",
                  "blocked_no_official_reference", "blocked_ambiguous_address", "blocked_non_flood"}:
        return "false"
    return "true"


def _row(ev, patch_id, relation, distance, score, score_class, interp, limit):
    level = ev.get("evidence_level", "")
    georef = ev.get("georeferencing_status", "")
    return {
        "linkage_id": "", "event_id": ev.get("event_id", ""), "patch_id": patch_id,
        "region": ev.get("region", ""),
        "event_date": ev.get("event_date") or ev.get("event_period_start", ""),
        "event_type": ev.get("event_type", ""), "evidence_level": level,
        "georeferencing_status": georef, "spatial_relation": relation,
        "temporal_precision": ev.get("temporal_precision", ""),
        "distance_m": "" if distance is None else round(distance, 1),
        "patch_score_v6": score, "patch_score_class_v6": score_class,
        "source_confidence": ev.get("source_confidence", ""),
        "parse_confidence": "medium" if ev.get("temporal_precision") in {"day", "period"} else "low",
        "geocoding_confidence": ev.get("geocoding_confidence", ""),
        "linkage_confidence": _link_conf(relation, level), "observed_event_strength": _strength(level),
        "can_be_used_for_observational_evaluation": _can_eval(level, relation, georef),
        "can_be_ground_truth": "false", "can_be_used_as_ground_truth": "false",
        "allowed_for_training": "false", "review_only": "true",
        "requires_manual_review": "true", "interpretation": interp, "limitations": limit,
    }


def _point_of(ev):
    if ev.get("lat") and ev.get("lon"):
        try:
            return float(ev["lon"]), float(ev["lat"])
        except ValueError:
            return None
    return None


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Event-Patch Linkage")
    print("=" * 60)
    for path in (l13b.MATRIX, l13b.SCORE, CATALOG):
        if not path.exists():
            print(f"STOP: missing input {rel(path)}")
            return 1
    patches = l13b._load_patches()
    scores = l13b._load_scores()
    events = [e for e in read_csv(CATALOG) if e.get("preferred_record", "true") != "false"]
    links: list[dict] = []

    for ev in events:
        region = (ev.get("region") or "").lower()
        level = ev.get("evidence_level", "")
        patches_here = patches.get(region, [])
        bbox = parse_bbox(ev.get("bbox", ""))
        pt = _point_of(ev)
        geom = (ev.get("geometry_type") or "").lower()

        if level in WEAK or level not in OBSERVED:
            has_period = bool(ev.get("event_date") or ev.get("event_period_start"))
            rel_kind = ("same_neighborhood_context" if level == "weak_official_neighborhood_context"
                        else ("same_region_period_context" if has_period else "insufficient_for_patch_link"))
            links.append(_row(ev, "REGION_LEVEL_NO_PATCH_RESOLUTION", rel_kind, None, "", "",
                              "Contexto fraco/bairro/documental mantido sem resolucao por patch.",
                              "Bairro/regiao/risco/alerta nao e evento observado patch-level."))
            continue

        matched = False
        # polygon / bbox geometry
        if geom in {"polygon", "multipolygon", "bbox"} and bbox:
            for pid, pbb, _c in patches_here:
                if bbox_intersect(bbox, pbb):
                    score, sc = scores.get(pid, ("", ""))
                    relation = "strong_polygon_intersects_patch" if level == "strong_observed_flood_polygon" else "moderate_bbox_intersects_patch"
                    links.append(_row(ev, pid, relation, None, score, sc,
                                      "Geometria de evento intersecta bbox do patch.",
                                      "Interseccao por bbox e aproximacao review-only; nao e ground truth."))
                    matched = True
        # point geometry (official latlon or geocoded)
        elif pt is not None:
            for pid, pbb, cen in patches_here:
                inside = pbb[0] <= pt[0] <= pbb[2] and pbb[1] <= pt[1] <= pbb[3]
                near = (pbb[0] - NEAR_BUFFER_DEG <= pt[0] <= pbb[2] + NEAR_BUFFER_DEG and
                        pbb[1] - NEAR_BUFFER_DEG <= pt[1] <= pbb[3] + NEAR_BUFFER_DEG)
                if not (inside or near):
                    continue
                dist = haversine_m(pt[0], pt[1], cen[0], cen[1])
                score, sc = scores.get(pid, ("", ""))
                if inside:
                    if level == "strong_observed_flood_point":
                        relation = "strong_point_inside_patch"
                    elif level == "moderate_official_occurrence_geocoded":
                        relation = "moderate_geocoded_point_inside_patch"
                    elif level == "moderate_official_street_segment_match":
                        relation = "moderate_street_segment_intersects_patch"
                    else:
                        relation = "moderate_point_inside_patch"
                    interp = "Ponto oficial/geocodificado dentro do bbox do patch."
                    limit = "Ponto de ocorrencia/geocodificacao nao e footprint; vinculo moderado review-only."
                else:
                    relation = "near_patch_buffer_candidate"
                    interp = "Ponto oficial/geocodificado no buffer aproximado (~550m) do patch."
                    limit = "Proximidade nao confirma ocorrencia no patch; candidato apenas para revisao."
                links.append(_row(ev, pid, relation, dist, score, sc, interp, limit))
                matched = True

        if not matched:
            links.append(_row(ev, "REGION_LEVEL_NO_PATCH_RESOLUTION", "same_region_period_context", None, "", "",
                              "Evento observado/moderado sem sobreposicao/proximidade de patch nesta matriz.",
                              "Associacao regional/periodo apenas; nao confirma patch."))

    for i, r in enumerate(links):
        r["linkage_id"] = f"S14ALINK_{i:05d}"
    links = [{k: r.get(k, "") for k in FIELDS} for r in links]
    write_csv(LINKAGE, links, FIELDS)

    rel_counts = Counter(r["spatial_relation"] for r in links)
    conf_counts = Counter(r["linkage_confidence"] for r in links)
    eval_links = sum(1 for r in links if r["can_be_used_for_observational_evaluation"] == "true")
    strong_links = sum(1 for r in links if r["linkage_confidence"] == "strong")
    moderate_links = sum(1 for r in links if r["linkage_confidence"] == "moderate")
    summary = [
        {"metric": "total_preferred_events", "value": len(events)},
        {"metric": "total_links", "value": len(links)},
        {"metric": "links_by_spatial_relation", "value": json.dumps(dict(rel_counts), ensure_ascii=False, sort_keys=True)},
        {"metric": "links_by_confidence", "value": json.dumps(dict(conf_counts), ensure_ascii=False, sort_keys=True)},
        {"metric": "strong_links", "value": strong_links},
        {"metric": "moderate_links", "value": moderate_links},
        {"metric": "observational_evaluation_links", "value": eval_links},
        {"metric": "training_allowed_count", "value": sum(1 for r in links if r["allowed_for_training"] == "true")},
        {"metric": "ground_truth_count", "value": sum(1 for r in links if r["can_be_ground_truth"] == "true")},
        {"metric": "ground_truth_use_count", "value": sum(1 for r in links if r["can_be_used_as_ground_truth"] == "true")},
    ]
    for s in summary:
        s["review_only"] = "true"
    write_csv(SUMMARY, summary, ["metric", "value", "review_only"])

    features = []
    for r in links:
        if r["patch_id"] == "REGION_LEVEL_NO_PATCH_RESOLUTION":
            continue
        for pid, _pb, cen in patches.get(r["region"], []):
            if pid == r["patch_id"]:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(cen[0], 6), round(cen[1], 6)]},
                    "properties": {
                        "linkage_id": r["linkage_id"], "event_id": r["event_id"], "patch_id": pid,
                        "region": r["region"], "evidence_level": r["evidence_level"],
                        "georeferencing_status": r["georeferencing_status"],
                        "spatial_relation": r["spatial_relation"], "linkage_confidence": r["linkage_confidence"],
                        "can_be_used_for_observational_evaluation": r["can_be_used_for_observational_evaluation"],
                        "can_be_ground_truth": "false", "can_be_used_as_ground_truth": "false",
                        "allowed_for_training": "false", "review_only": "true",
                    },
                })
                break
    write_json(GEOJSON, {
        "type": "FeatureCollection", "name": "susc_14a_event_patch_linkage",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "review_only": True, "can_be_ground_truth": False, "allowed_for_training": False,
        "features": features,
    })
    write_json(LIMITS, {
        "artifact": "SUSC-14A rescued event patch linkage",
        "n_preferred_events": len(events), "n_links": len(links),
        "spatial_relation_counts": dict(rel_counts), "linkage_confidence_counts": dict(conf_counts),
        "strong_links": strong_links, "moderate_links": moderate_links,
        "observational_evaluation_links": eval_links,
        "can_be_ground_truth": False, "can_be_used_as_ground_truth": False,
        "allowed_for_training": False, "review_only": True, "score_v7_created": False,
        "key_limitations": [
            "Offline e o modo padrao; sem rede a camada de logradouros oficial nao e baixada.",
            "A unica referencia oficial local cobre poucos bairros (risco Sul/Recife), entao o geocoding por endereco resulta majoritariamente em contexto de bairro, nao em ponto patch-level.",
            "Ocorrencia geocodificada por logradouro oficial e no maximo moderada; nunca strong.",
            "Bairro/regiao/risco/alerta nunca viram evento observado patch-level.",
            "Footprint oficial exige geometria explicita + data; sem ele nao ha evento forte.",
            "Sem ground truth, sem treino supervisionado e sem score v7.",
        ],
        "mandatory_statement": MANDATORY,
    })
    print(f"links: {len(links)} | strong {strong_links} moderate {moderate_links} | eval {eval_links} | geojson {len(features)}")
    print("relations:", dict(rel_counts))
    print("review-only. No ground truth. No training. No score v7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
