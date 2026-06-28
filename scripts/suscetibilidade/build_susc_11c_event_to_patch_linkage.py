"""
SUSC-11C Event-to-Patch Linkage (review-only)

Tests the spatial relation between SUSC-11B observed events and the SUSC patches,
attaching the candidate susceptibility score v6 per patch. Polygon/bbox events use
bbox overlap; point/point_set events reload their real coordinates from the
traceable source file and use point-in-patch (inside) / near-buffer. Events without
usable geometry stay region-level/documentary. Nothing becomes ground truth and
nothing unlocks supervised training.

Reads:
  - datasets/suscetibilidade/susc_features_by_patch_v1.csv
  - datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv
  - datasets/suscetibilidade/susc_11b_observed_event_catalog_v1.csv
Writes:
  - datasets/suscetibilidade/susc_11c_event_patch_linkage_v1.csv
  - outputs_public/suscetibilidade/SUSC_11C_event_patch_linkage_summary.csv
  - outputs_public/suscetibilidade/SUSC_11C_event_patch_linkage_limitations.json
  - outputs_public/suscetibilidade/SUSC_11B_11C_observed_event_linkage_report.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown, rel  # noqa: E402
from susc_geometry import parse_bbox, bbox_intersect, haversine_m, in_brazil  # noqa: E402
import parse_susc_11b_observed_events as parser  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_11b_observed_event_catalog_v1.csv"

LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_11c_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11C_event_patch_linkage_summary.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11C_event_patch_linkage_limitations.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11B_11C_observed_event_linkage_report.md"

BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]
NEAR_BUFFER_DEG = 0.005  # ~550 m expansion of patch bbox for "near"
MAX_POINTS = 20000

OBSERVED_LEVELS = {"observed_flood_polygon_strong", "observed_flood_point_strong",
                   "observed_flood_bbox_moderate", "official_occurrence_point_moderate"}
STRONG_LEVELS = {"observed_flood_polygon_strong", "observed_flood_point_strong"}

LINKAGE_FIELDS = [
    "linkage_id", "event_id", "patch_id", "region", "event_type", "evidence_level",
    "spatial_relation", "temporal_relation", "distance_m", "patch_score_v6",
    "patch_score_class_v6", "source_confidence", "linkage_confidence",
    "can_be_used_as_observed_evidence", "can_be_ground_truth", "allowed_for_training",
    "review_only", "requires_manual_review", "interpretation", "limitations",
]

MANDATORY_STATEMENT = (
    "O SUSC-11B/11C organiza ocorrências reais/documentais de alagamento/inundação e "
    "testa sua ligação espacial com patches. Esses vínculos são evidência observacional "
    "review-only e não constituem ground truth supervisionado, predição operacional ou "
    "confirmação automática de ocorrência por patch."
)


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def load_patches():
    """Return {region: [(patch_id, bbox, centroid)]}."""
    patches = read_csv(MATRIX)
    if not patches or not all(c in patches[0] for c in BBOX_COLS):
        return {}
    by_region = defaultdict(list)
    for r in patches:
        bb = [_num(r.get(c)) for c in BBOX_COLS]
        if all(v is not None for v in bb):
            centroid = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
            by_region[(r.get("regiao") or "").strip().lower()].append((r["patch_id"], bb, centroid))
    return by_region


def load_scores():
    scores = {}
    for r in read_csv(SCORE):
        scores[r["patch_id"]] = (r.get("susc_score_v6_candidate", ""),
                                 r.get("susc_class_v6_candidate", ""))
    return scores


def load_event_points(ev):
    """Reload real coordinates for a point/point_set event from its source file."""
    path = (ev.get("source_url_or_path") or "").strip()
    if not path:
        return []
    p = ROOT / path
    if not p.exists():
        return []
    ext = p.suffix.lower()
    try:
        if ext in {".geojson", ".json"}:
            pts, _ = parser._parse_geojson(p)
        elif ext in {".csv", ".tsv"}:
            pts, _ = parser._parse_csv_latlon(p)
        elif ext in {".wkt", ".txt"}:
            pts, _ = parser._parse_wkt_text(p)
        elif ext == ".kml":
            pts, _ = parser._parse_kml_text(p.read_text(encoding="utf-8", errors="ignore"))
        else:
            return []
    except (OSError, ValueError):
        return []
    return [(lo, la) for lo, la in pts[:MAX_POINTS] if in_brazil(lo, la)]


def temporal_relation(ev) -> str:
    return "dated_event" if (ev.get("event_date") or ev.get("event_period_start")) else "undated"


def linkage_confidence(relation: str, level: str) -> str:
    if relation in ("event_polygon_intersects_patch", "event_point_inside_patch"):
        if level in STRONG_LEVELS:
            return "strong"
        if level in OBSERVED_LEVELS:
            return "moderate"
        return "weak"
    if relation == "event_point_near_patch":
        return "weak"
    return "very_weak"


def can_be_observed_evidence(level: str, relation: str, has_geom: bool, has_date: bool) -> str:
    patch_level_relations = {
        "exact_patch_overlap",
        "event_polygon_intersects_patch",
        "event_point_inside_patch",
        "event_point_near_patch",
    }
    if level in OBSERVED_LEVELS and relation in patch_level_relations and has_geom and has_date:
        return "true"
    return "false"


def _row(event_id, patch_id, region, etype, level, relation, temporal, dist,
         score, sclass, sconf, has_geom, has_date, interp, limit):
    return {
        "linkage_id": "", "event_id": event_id, "patch_id": patch_id, "region": region,
        "event_type": etype, "evidence_level": level, "spatial_relation": relation,
        "temporal_relation": temporal, "distance_m": "" if dist is None else round(dist, 1),
        "patch_score_v6": score, "patch_score_class_v6": sclass, "source_confidence": sconf,
        "linkage_confidence": linkage_confidence(relation, level),
        "can_be_used_as_observed_evidence": can_be_observed_evidence(level, relation, has_geom, has_date),
        "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
        "requires_manual_review": "true", "interpretation": interp, "limitations": limit,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-11C Event-to-Patch Linkage (review-only)")
    print("=" * 60)

    for p in (MATRIX, SCORE, CATALOG):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    region_patches = load_patches()
    if not region_patches:
        print("STOP: patch matrix has no usable bbox columns.")
        return 1
    scores = load_scores()
    catalog = read_csv(CATALOG)

    rows = []
    for ev in catalog:
        eid = ev["event_id"]
        region = (ev.get("region") or "").strip().lower()
        etype = ev.get("event_type", "")
        level = ev.get("evidence_level", "")
        geom = (ev.get("geometry_type") or "").strip().lower()
        bbox = parse_bbox(ev.get("bbox", ""))
        has_geom = bool(ev.get("bbox") or (ev.get("lat") and ev.get("lon")))
        has_date = bool(ev.get("event_date") or ev.get("event_period_start"))
        temporal = temporal_relation(ev)
        sconf = ev.get("source_confidence", "")
        patches_here = region_patches.get(region, [])

        # non-event / no-geometry evidence stays region-level / documentary
        if level == "documentary_context_only" or not has_geom:
            rows.append(_row(eid, "REGION_LEVEL_NO_PATCH_RESOLUTION", region, etype, level,
                             "documentary_context_only", temporal, None, "", "", sconf,
                             has_geom, has_date,
                             "Contexto documental/sem geometria de evento utilizavel para patch.",
                             "Sem footprint de evento; nao e overlap de patch; review-only."))
            continue
        if level == "insufficient":
            rows.append(_row(eid, "REGION_LEVEL_NO_PATCH_RESOLUTION", region, etype, level,
                             "insufficient_for_patch_link", temporal, None, "", "", sconf,
                             has_geom, has_date, "Dado insuficiente para ligacao com patch.",
                             "Sem dado rastreavel suficiente; review-only."))
            continue

        matched = False
        if geom in ("polygon", "multipolygon") and bbox:
            # bbox overlap of event polygon extent with patch bbox
            for pid, pbb, _c in patches_here:
                if bbox_intersect(bbox, pbb):
                    score, sclass = scores.get(pid, ("", ""))
                    rows.append(_row(eid, pid, region, etype, level,
                                     "event_polygon_intersects_patch", temporal, None,
                                     score, sclass, sconf, has_geom, has_date,
                                     "Extensao (bbox) do poligono de evento rastreavel sobrepoe o bbox do patch.",
                                     "bbox e aproximacao; poligono e candidato digitalizado, NAO footprint validado; nao confirma evento no patch."))
                    matched = True
        else:
            # point / point_set / bbox-only -> use real coordinates when available
            pts = load_event_points(ev)
            if not pts and ev.get("lat") and ev.get("lon"):
                pts = [(float(ev["lon"]), float(ev["lat"]))]
            if not pts:
                relk = "same_region_same_period" if has_date else "same_region_only"
                rows.append(_row(eid, "REGION_LEVEL_NO_PATCH_RESOLUTION", region, etype, level,
                                 relk, temporal, None, "", "", sconf, has_geom, has_date,
                                 "Geometria de extensao sem ponto recarregavel; mantida apenas como vinculo regional.",
                                 "Sem poligono de evento explicito utilizavel para intersectar patch; review-only."))
                matched = True
            else:
                agg = {}  # pid -> [n_inside, n_near, min_dist]
                for (lon, lat) in pts:
                    for pid, pbb, cen in patches_here:
                        inside = pbb[0] <= lon <= pbb[2] and pbb[1] <= lat <= pbb[3]
                        near = (pbb[0] - NEAR_BUFFER_DEG <= lon <= pbb[2] + NEAR_BUFFER_DEG and
                                pbb[1] - NEAR_BUFFER_DEG <= lat <= pbb[3] + NEAR_BUFFER_DEG)
                        if not (inside or near):
                            continue
                        d = haversine_m(lon, lat, cen[0], cen[1])
                        rec = agg.setdefault(pid, [0, 0, d])
                        if inside:
                            rec[0] += 1
                        elif near:
                            rec[1] += 1
                        rec[2] = min(rec[2], d)
                for pid, (n_in, n_near, mind) in sorted(agg.items()):
                    score, sclass = scores.get(pid, ("", ""))
                    if n_in > 0:
                        rel_kind = "event_point_inside_patch"
                        interp = f"{n_in} ponto(s) de fonte rastreavel dentro do bbox do patch."
                        limit = ("Ponto de ocorrencia/risco != footprint de area alagada; "
                                 "aderencia espacial review-only; NAO e ground truth.")
                    else:
                        rel_kind = "event_point_near_patch"
                        interp = f"{n_near} ponto(s) proximo(s) (buffer ~550m) do bbox do patch."
                        limit = "Proximidade nao e ocorrencia no patch; review-only."
                    rows.append(_row(eid, pid, region, etype, level, rel_kind, temporal, mind,
                                     score, sclass, sconf, has_geom, has_date, interp, limit))
                    matched = True

        if not matched:
            relk = "same_region_same_period" if has_date else "same_region_only"
            rows.append(_row(eid, "REGION_LEVEL_NO_PATCH_RESOLUTION", region, etype, level,
                             relk, temporal, None, "", "", sconf, has_geom, has_date,
                             "Evento com geometria mas sem sobreposicao/proximidade a patch; vinculo regional.",
                             "Associacao em nivel de regiao; nao e overlap de patch; review-only."))

    # assign stable linkage ids
    for i, r in enumerate(rows):
        r["linkage_id"] = f"LINK_{i:05d}"

    write_csv(LINKAGE, rows, LINKAGE_FIELDS)

    # ---- required summary metrics ----
    rel_counts = Counter(r["spatial_relation"] for r in rows)
    conf_counts = Counter(r["linkage_confidence"] for r in rows)
    by_region = Counter(r["region"] for r in rows)
    by_level = Counter(r["evidence_level"] for r in rows)
    by_region_rel = defaultdict(Counter)
    for r in rows:
        by_region_rel[r["region"]][r["spatial_relation"]] += 1
    strong = conf_counts.get("strong", 0)
    moderate = conf_counts.get("moderate", 0)
    weak = conf_counts.get("weak", 0) + conf_counts.get("very_weak", 0)
    patches_with_event = len({r["patch_id"] for r in rows
                              if r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"
                              and r["can_be_used_as_observed_evidence"] == "true"})
    patches_context_only = len({r["patch_id"] for r in rows
                                if r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"
                                and r["can_be_used_as_observed_evidence"] != "true"})
    n_obs_evidence_rows = sum(1 for r in rows if r["can_be_used_as_observed_evidence"] == "true")
    events_linked = len({r["event_id"] for r in rows if r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"})

    summary_rows = [
        {"metric": "total_events", "value": len({r["event_id"] for r in rows}), "review_only": "true"},
        {"metric": "total_patches", "value": sum(len(v) for v in region_patches.values()), "review_only": "true"},
        {"metric": "total_links", "value": len(rows), "review_only": "true"},
        {"metric": "links_by_region", "value": json.dumps(dict(by_region), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "links_by_spatial_relation", "value": json.dumps(dict(rel_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "links_by_evidence_level", "value": json.dumps(dict(by_level), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "strong_links", "value": strong, "review_only": "true"},
        {"metric": "moderate_links", "value": moderate, "review_only": "true"},
        {"metric": "weak_links", "value": weak, "review_only": "true"},
        {"metric": "events_linked_to_any_patch", "value": events_linked, "review_only": "true"},
        {"metric": "patches_with_observed_event_evidence", "value": patches_with_event, "review_only": "true"},
        {"metric": "patches_with_only_contextual_evidence", "value": patches_context_only, "review_only": "true"},
        {"metric": "review_only_count", "value": sum(1 for r in rows if r["review_only"] == "true"), "review_only": "true"},
        {"metric": "training_allowed_count", "value": sum(1 for r in rows if r["allowed_for_training"] == "true"), "review_only": "true"},
        {"metric": "ground_truth_count", "value": sum(1 for r in rows if r["can_be_ground_truth"] == "true"), "review_only": "true"},
    ]
    write_csv(SUMMARY, summary_rows, ["metric", "value", "review_only"])

    limitations = {
        "artifact": "SUSC-11C event-to-patch linkage",
        "n_events": len({r["event_id"] for r in rows}),
        "n_link_rows": len(rows),
        "spatial_relation_counts": dict(rel_counts),
        "linkage_confidence_counts": dict(conf_counts),
        "links_strong": strong, "links_moderate": moderate, "links_weak": weak,
        "n_rows_usable_as_observed_evidence": n_obs_evidence_rows,
        "n_patches_with_observed_evidence": patches_with_event,
        "n_patches_with_only_contextual_evidence": patches_context_only,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "scientific_status": "observed_event_to_patch_spatial_adherence_review_only_not_ground_truth",
        "key_limitations": [
            "Sem ground truth.",
            "Sem treino supervisionado.",
            "Sem predicao operacional.",
            "Fontes sem URL direta foram registradas para revisao manual.",
            "Fontes nao baixadas permanecem no manifest com motivo explicito.",
            "Limitacoes de geometria: bbox e aproximacao; ponto nao e footprint de area alagada.",
            "Limitacoes de data: registros sem data reduzem confianca e nao viram evidencia observacional utilizavel.",
            "Risco/alerta/contexto nao sao ocorrencia observada.",
            "Link regional nao confirma evento por patch.",
        ],
        "mandatory_statement": MANDATORY_STATEMENT,
    }
    write_json(LIMITS, limitations)

    _write_report(catalog, rows, rel_counts, conf_counts, strong, moderate, weak,
                  n_obs_evidence_rows, patches_with_event)

    print(f"\nlinkage rows: {len(rows)} | strong: {strong} | moderate: {moderate} | weak: {weak}")
    print("spatial relations:", dict(rel_counts))
    print(f"rows usable as observed evidence: {n_obs_evidence_rows} | patches with observed evidence: {patches_with_event}")
    print("review-only. No ground truth. No supervised training.")
    return 0


def _write_report(catalog, rows, rel_counts, conf_counts, strong, moderate, weak,
                  n_obs_evidence_rows, patches_with_event):
    reg = read_csv(ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_source_registry_v1.csv")
    dl = read_csv(ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_download_manifest_v1.csv")
    by_level = Counter(e["evidence_level"] for e in catalog)
    by_region = Counter(e["region"] for e in catalog)
    n_geom = sum(1 for e in catalog if e.get("bbox") or (e.get("lat") and e.get("lon")))
    n_date = sum(1 for e in catalog if e.get("event_date") or e.get("event_period_start"))
    dl_status = Counter(r["download_status"] for r in dl)
    downloaded = dl_status.get("downloaded", 0)

    def tbl(counter):
        return "\n".join(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    md = f"""# SUSC-11B / SUSC-11C — Observed Event Acquisition and Event-to-Patch Linkage

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

{MANDATORY_STATEMENT}

## 1. Objetivo
Buscar, baixar quando possivel, parsear e normalizar dados reais de alagamento/
inundacao/enxurrada; construir um catalogo observacional priorizando fontes
oficiais/tecnicas com coordenada, geometria, data e tipo de evento; e testar a
ligacao espacial desses eventos com os patches do REV-P, classificando a forca do
vinculo. Tudo permanece evidencia observacional review-only.

## 2. Fontes buscadas (registry)
{len(reg)} fontes-alvo registradas por regiao (Recife, Petropolis, Curitiba):
APAC, Defesa Civil, Prefeituras, CEMADEN, CPRM/SGB, ANA/Hidroweb, S2iD, INEA/RJ,
GeoCuritiba/IPPUC, IAT/Aguas Parana.

| regiao | fontes |
|---|---|
{tbl(Counter(r['region'] for r in reg))}

## 3. Fontes baixadas
Downloads bem-sucedidos nesta passada: **{downloaded}**. As fontes-alvo do registry
nao possuem URL direta de arquivo (`url_is_direct_download=false`); a aquisicao
oficial direta permanece manual. O catalogo reusa geometria local ja rastreavel
(SUSC-07B): Charter758 (mancha candidata), Defesa Civil (risco), registros de
ocorrencia e catalogos de estacao.

## 4. Fontes nao baixadas e motivo
| status | n |
|---|---|
{tbl(dl_status)}

`not_attempted_no_direct_url_or_no_network`: fonte oficial documentada, sem URL
direta de arquivo no registry (aquisicao manual pendente). Nenhum raster, nenhum
arquivo >100MB, nenhuma API com chave, nenhum Google Maps.

## 5. Eventos observados extraidos
Total no catalogo: **{len(catalog)}** registros (geometria rastreavel, sem
coordenada inventada).

## 6. Eventos por regiao
| regiao | eventos |
|---|---|
{tbl(by_region)}

## 7. Eventos com geometria
{n_geom} de {len(catalog)} registros tem geometria (bbox/coordenada) rastreavel.

## 8. Eventos com data
{n_date} de {len(catalog)} registros tem data/periodo explicito.

## 9. Niveis de evidencia
| nivel | n |
|---|---|
{tbl(by_level)}

`*_strong` exige footprint validado com geometria explicita (nao disponivel
localmente). O unico poligono de mancha e digitalizacao candidata (Charter),
classificado como `observed_flood_bbox_moderate`. Setor/ponto de risco da Defesa
Civil -> `risk_area_context`; estacoes -> `administrative_record_only`.

## 10. Eventos linkados a patch (relacoes espaciais)
| relacao espacial | n linhas |
|---|---|
{tbl(rel_counts)}

Linhas usaveis como evidencia observacional: **{n_obs_evidence_rows}**;
patches distintos com evidencia observacional: **{patches_with_event}**.

## 11. Links fortes / moderados / fracos
| forca | n |
|---|---|
| strong | {strong} |
| moderate | {moderate} |
| weak | {weak} |

Distribuicao de `linkage_confidence`:
{tbl(conf_counts)}

## 12. Limitacoes
- Nenhum vinculo e ground truth ou rotulo de treino supervisionado.
- Links fortes exigiriam footprint de evento validado; nao ha localmente.
- Poligono de mancha e digitalizacao candidata (Charter) -> no maximo link moderado.
- Ponto de ocorrencia/risco != footprint de area alagada por patch.
- Sobreposicao de bbox e aproximacao; proximidade (~550m) nao confirma ocorrencia.
- `score_v6` do patch e candidato heuristico review-only, nao ocorrencia confirmada.
- Alerta != ocorrencia; setor de risco != evento; registro administrativo != patch-level.

## 13. O que NAO pode ser afirmado
- NAO se pode afirmar que um patch teve alagamento confirmado.
- NAO se pode usar estes vinculos como ground truth ou alvo de treino.
- NAO se pode tratar score_v6 alto como ocorrencia.
- NAO se pode tratar setor de risco/alerta como evento observado.

## 14. O que JA pode ser afirmado
- Existe geometria de evento rastreavel (Charter, mancha candidata) com data 2022-05.
- Ha registros oficiais de ocorrencia com coordenada (review-only) em Recife/Petropolis.
- Esses elementos tem aderencia espacial mensuravel a patches especificos (review-only).
- A cadeia de aquisicao->parsing->catalogo->linkage e reproduzivel e auditavel.

## 15. Proximo marco: SUSC-12A / 12B
Revisao humana das ligacoes moderadas (Charter x patch; ocorrencias x patch),
aquisicao oficial direta de footprint validado (APAC/INEA/Defesa Civil) e desenho
de criterio de referencia sob revisao — ainda sem ground truth automatico.
"""
    write_markdown(REPORT, md)


if __name__ == "__main__":
    raise SystemExit(main())
