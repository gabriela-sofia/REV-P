"""
SUSC-13B-AUTO consolidated event-to-patch linkage (review-only).

Links the consolidated observed-event catalog to SUSC-03 patches using explicit
geometry only (point-in-bbox, bbox intersection, polygon intersection when
shapely is available, plus an approximate proximity buffer). Strong/moderate
events with sufficient geometry may support observational evaluation; nothing
here is ground truth or training material.

Writes:
  - datasets/suscetibilidade/susc_13b_auto_event_patch_linkage_v1.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_event_patch_linkage_summary.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_event_patch_linkage_limitations.json
  - outputs_public/suscetibilidade/SUSC_13B_auto_event_patch_linkage_geojson.geojson
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import parse_susc_13a_strong_observed_events as p13a  # noqa: E402
from susc_geometry import bbox_intersect, haversine_m, in_brazil, parse_bbox  # noqa: E402
from susc_io import read_csv, rel, write_csv, write_json  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_consolidated_observed_event_catalog_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_event_patch_linkage_summary.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_event_patch_linkage_limitations.json"
GEOJSON = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_event_patch_linkage_geojson.geojson"

MANDATORY = (
    "O SUSC-13B-AUTO realiza descoberta e aquisição automática de fontes oficiais/rastreáveis "
    "para fortalecer a camada observacional de alagamento/inundação. Mesmo quando encontra "
    "eventos fortes, a etapa mantém todos os vínculos em modo review-only, não cria ground truth, "
    "não treina modelo supervisionado e não cria score v7 automaticamente."
)

BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]
NEAR_BUFFER_DEG = 0.005
MAX_POINTS = 20000
STRONG_LEVELS = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE_LEVELS = {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}
OBSERVED_LEVELS = STRONG_LEVELS | MODERATE_LEVELS

FIELDS = [
    "linkage_id", "event_id", "patch_id", "region", "event_date", "event_type",
    "evidence_level", "spatial_relation", "temporal_precision", "distance_m",
    "patch_score_v6", "patch_score_class_v6", "source_confidence", "parse_confidence",
    "linkage_confidence", "observed_event_strength", "can_be_used_for_observational_evaluation",
    "can_be_ground_truth", "allowed_for_training", "review_only", "requires_manual_review",
    "interpretation", "limitations",
]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_patches():
    by_region = defaultdict(list)
    for row in read_csv(MATRIX):
        vals = [_num(row.get(c)) for c in BBOX_COLS]
        if any(v is None for v in vals):
            continue
        bb = [float(v) for v in vals]  # type: ignore[arg-type]
        centroid = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        by_region[(row.get("regiao") or "").lower()].append((row["patch_id"], bb, centroid))
    return by_region


def _load_scores():
    return {row["patch_id"]: (row.get("susc_score_v6_candidate", ""), row.get("susc_class_v6_candidate", ""))
            for row in read_csv(SCORE)}


def _strength(level: str) -> str:
    if level in STRONG_LEVELS:
        return "strong"
    if level in MODERATE_LEVELS:
        return "moderate"
    if level.startswith("weak_"):
        return "weak"
    return "documentary_or_rejected"


def _link_confidence(relation: str, level: str) -> str:
    if relation in {"strong_polygon_intersects_patch", "strong_point_inside_patch"} and level in STRONG_LEVELS:
        return "strong"
    if relation in {"moderate_point_inside_patch", "moderate_bbox_intersects_patch"} and level in OBSERVED_LEVELS:
        return "moderate"
    if relation == "near_patch_buffer_candidate":
        return "weak"
    return "very_weak"


def _can_eval(level: str, relation: str) -> str:
    return "true" if level in OBSERVED_LEVELS and relation in {
        "strong_polygon_intersects_patch", "strong_point_inside_patch",
        "moderate_point_inside_patch", "moderate_bbox_intersects_patch",
        "near_patch_buffer_candidate"} else "false"


def _parse_conf(event: dict) -> str:
    tp = event.get("temporal_precision", "")
    return "medium" if tp in {"day", "period"} else "low"


def _row(event, patch_id, relation, distance, score, score_class, interp, limit):
    level = event.get("evidence_level", "")
    return {
        "linkage_id": "", "event_id": event.get("event_id", ""), "patch_id": patch_id,
        "region": event.get("region", ""),
        "event_date": event.get("event_date") or event.get("event_period_start", ""),
        "event_type": event.get("event_type", ""), "evidence_level": level,
        "spatial_relation": relation, "temporal_precision": event.get("temporal_precision", ""),
        "distance_m": "" if distance is None else round(distance, 1),
        "patch_score_v6": score, "patch_score_class_v6": score_class,
        "source_confidence": event.get("source_confidence", ""), "parse_confidence": _parse_conf(event),
        "linkage_confidence": _link_confidence(relation, level), "observed_event_strength": _strength(level),
        "can_be_used_for_observational_evaluation": _can_eval(level, relation),
        "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
        "requires_manual_review": "true", "interpretation": interp, "limitations": limit,
    }


def _event_points(event):
    pts = []
    if event.get("lat") and event.get("lon"):
        try:
            pts.append((float(event["lon"]), float(event["lat"])))
        except ValueError:
            pass
    path = (event.get("local_file_path") or "").strip()
    if path:
        p = ROOT / path
        if p.exists():
            ext = p.suffix.lower()
            try:
                if ext in {".geojson", ".json"}:
                    fpts, _ = p13a._parse_geojson(p)
                elif ext in {".csv", ".tsv"}:
                    fpts, _ = p13a._parse_csv_latlon(p)
                elif ext in {".wkt", ".txt"}:
                    fpts, _ = p13a._parse_wkt_text(p)
                elif ext == ".kml":
                    fpts, _ = p13a._parse_kml_text(p.read_text(encoding="utf-8", errors="ignore"))
                else:
                    fpts = []
                pts.extend(fpts[:MAX_POINTS])
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return [(lon, lat) for lon, lat in pts if in_brazil(lon, lat)]


def _point_links(event, patches, scores):
    pts = _event_points(event)
    out = []
    if not pts:
        return out
    agg: dict[str, list] = {}
    for lon, lat in pts:
        for pid, pbb, cen in patches:
            inside = pbb[0] <= lon <= pbb[2] and pbb[1] <= lat <= pbb[3]
            near = (pbb[0] - NEAR_BUFFER_DEG <= lon <= pbb[2] + NEAR_BUFFER_DEG and
                    pbb[1] - NEAR_BUFFER_DEG <= lat <= pbb[3] + NEAR_BUFFER_DEG)
            if not (inside or near):
                continue
            dist = haversine_m(lon, lat, cen[0], cen[1])
            rec = agg.setdefault(pid, [0, 0, dist])
            if inside:
                rec[0] += 1
            elif near:
                rec[1] += 1
            rec[2] = min(rec[2], dist)
    for pid, (inside_n, near_n, dist) in sorted(agg.items()):
        score, score_class = scores.get(pid, ("", ""))
        if inside_n:
            relation = "strong_point_inside_patch" if event["evidence_level"] == "strong_observed_flood_point" else "moderate_point_inside_patch"
            interp = f"{inside_n} ponto(s) rastreavel(is) dentro do bbox do patch."
            limit = "Ponto de ocorrencia nao e footprint de area alagada; vinculo review-only."
        else:
            relation = "near_patch_buffer_candidate"
            interp = f"{near_n} ponto(s) rastreavel(is) no buffer aproximado (~550m) do patch."
            limit = "Proximidade nao confirma ocorrencia no patch; candidato apenas para revisao."
        out.append(_row(event, pid, relation, dist, score, score_class, interp, limit))
    return out


def _geojson_feature(link, patches_by_region):
    region = link["region"]
    for pid, _patch_bb, cen in patches_by_region.get(region, []):
        if pid == link["patch_id"]:
            return {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(cen[0], 6), round(cen[1], 6)]},
                "properties": {
                    "linkage_id": link["linkage_id"], "event_id": link["event_id"],
                    "patch_id": link["patch_id"], "region": region,
                    "evidence_level": link["evidence_level"], "spatial_relation": link["spatial_relation"],
                    "linkage_confidence": link["linkage_confidence"],
                    "can_be_used_for_observational_evaluation": link["can_be_used_for_observational_evaluation"],
                    "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
                },
            }
    return None


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Event-Patch Linkage")
    print("=" * 60)
    for path in (MATRIX, SCORE, CATALOG):
        if not path.exists():
            print(f"STOP: missing input: {rel(path)}")
            return 1

    patches = _load_patches()
    scores = _load_scores()
    events = read_csv(CATALOG)
    links: list[dict] = []
    for event in events:
        # Only the preferred record of each duplicate group is linked to avoid
        # double-counting; non-preferred duplicates stay in the catalog for audit.
        if event.get("preferred_record", "true") == "false":
            continue
        region = (event.get("region") or "").lower()
        level = event.get("evidence_level", "")
        geom = (event.get("geometry_type") or "").lower()
        bbox = parse_bbox(event.get("bbox", ""))
        patches_here = patches.get(region, [])

        if level not in OBSERVED_LEVELS:
            relation = "same_region_period_context" if (event.get("event_date") or event.get("event_period_start")) else "insufficient_for_patch_link"
            links.append(_row(event, "REGION_LEVEL_NO_PATCH_RESOLUTION", relation, None, "", "",
                              "Contexto fraco/documental mantido sem resolucao por patch.",
                              "Risco, alerta, administrativo ou documental nao e evento observado."))
            continue

        matched = False
        if geom in {"polygon", "multipolygon", "bbox"} and bbox:
            for pid, pbb, _cen in patches_here:
                if bbox_intersect(bbox, pbb):
                    score, score_class = scores.get(pid, ("", ""))
                    relation = "strong_polygon_intersects_patch" if level == "strong_observed_flood_polygon" else "moderate_bbox_intersects_patch"
                    links.append(_row(event, pid, relation, None, score, score_class,
                                      "Geometria de evento intersecta bbox do patch.",
                                      "Interseccao por bbox e aproximacao review-only; nao e ground truth."))
                    matched = True
        else:
            plinks = _point_links(event, patches_here, scores)
            links.extend(plinks)
            matched = bool(plinks)

        if not matched:
            links.append(_row(event, "REGION_LEVEL_NO_PATCH_RESOLUTION", "same_region_period_context", None, "", "",
                              "Evento observado/moderado sem sobreposicao/proximidade de patch nesta matriz.",
                              "Associacao regional/periodo apenas; nao confirma patch."))

    for i, row in enumerate(links):
        row["linkage_id"] = f"S13BLINK_{i:05d}"
    write_csv(LINKAGE, links, FIELDS)

    rel_counts = Counter(r["spatial_relation"] for r in links)
    conf_counts = Counter(r["linkage_confidence"] for r in links)
    eval_links = sum(1 for r in links if r["can_be_used_for_observational_evaluation"] == "true")
    summary_rows = [
        {"metric": "total_events_preferred", "value": sum(1 for e in events if e.get("preferred_record", "true") != "false"), "review_only": "true"},
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

    features = [f for f in (_geojson_feature(r, patches) for r in links if r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION") if f]
    write_json(GEOJSON, {
        "type": "FeatureCollection",
        "name": "susc_13b_auto_event_patch_linkage",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "review_only": True, "can_be_ground_truth": False, "allowed_for_training": False,
        "features": features,
    })

    write_json(LIMITS, {
        "artifact": "SUSC-13B-AUTO consolidated event patch linkage",
        "n_events_total": len(events),
        "n_events_preferred": sum(1 for e in events if e.get("preferred_record", "true") != "false"),
        "n_links": len(links), "spatial_relation_counts": dict(rel_counts),
        "linkage_confidence_counts": dict(conf_counts), "observational_evaluation_links": eval_links,
        "can_be_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "score_v7_created": False, "model_persisted": False,
        "key_limitations": [
            "Offline e o modo padrao; sem rede nenhum evento novo e adquirido.",
            "Eventos fortes, se existirem, permanecem review-only.",
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
