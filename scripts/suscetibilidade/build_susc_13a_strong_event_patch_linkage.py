"""
SUSC-13A strong/moderate observed-event to patch linkage (review-only).

Builds patch relations from the SUSC-13A parsed catalog. Strong/moderate event
geometry can be used for observational evaluation only; it is never ground truth
and never training material.

Writes:
  - datasets/suscetibilidade/susc_13a_strong_event_patch_linkage_v1.csv
  - outputs_public/suscetibilidade/SUSC_13A_strong_event_patch_linkage_summary.csv
  - outputs_public/suscetibilidade/SUSC_13A_strong_event_patch_linkage_limitations.json
  - outputs_public/suscetibilidade/SUSC_13A_manual_acquisition_gap_report.md
  - outputs_public/suscetibilidade/SUSC_13A_manual_source_request_table.csv
  - outputs_public/suscetibilidade/SUSC_13A_strong_observed_event_acquisition_report.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_geometry import bbox_intersect, haversine_m, in_brazil, parse_bbox  # noqa: E402
from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402
import parse_susc_11b_observed_events as legacy_parser  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_observed_events_parsed_v1.csv"
REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_source_registry_v1.csv"
DL_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_download_manifest_v1.csv"
ACQ_REPORT_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_acquisition_report.csv"
PARSE_AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_parse_audit.csv"

LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_patch_linkage_summary.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_patch_linkage_limitations.json"
GAP_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_manual_acquisition_gap_report.md"
REQUEST_TABLE = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_manual_source_request_table.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_observed_event_acquisition_report.md"

MANDATORY = (
    "O SUSC-13A busca evidências observacionais mais fortes de alagamento/inundação "
    "para reduzir a fragilidade detectada em SUSC-11/12. Mesmo eventos fortes permanecem "
    "review-only nesta etapa e não criam ground truth, treino supervisionado ou confirmação "
    "operacional automática por patch."
)

BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]
NEAR_BUFFER_DEG = 0.005
MAX_POINTS = 20000
STRONG_LEVELS = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE_LEVELS = {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}
OBSERVED_LEVELS = STRONG_LEVELS | MODERATE_LEVELS

FIELDS = [
    "linkage_id",
    "event_id",
    "patch_id",
    "region",
    "event_date",
    "event_type",
    "evidence_level",
    "spatial_relation",
    "temporal_precision",
    "distance_m",
    "patch_score_v6",
    "patch_score_class_v6",
    "source_confidence",
    "linkage_confidence",
    "observed_event_strength",
    "can_be_used_for_observational_evaluation",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
    "requires_manual_review",
    "interpretation",
    "limitations",
]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_patches():
    rows = read_csv(MATRIX)
    by_region = defaultdict(list)
    for row in rows:
        bb = [_num(row.get(c)) for c in BBOX_COLS]
        if all(v is not None for v in bb):
            centroid = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
            by_region[(row.get("regiao") or "").lower()].append((row["patch_id"], bb, centroid))
    return by_region


def _load_scores():
    return {
        row["patch_id"]: (row.get("susc_score_v6_candidate", ""), row.get("susc_class_v6_candidate", ""))
        for row in read_csv(SCORE)
    }


def _source_confidence(level: str, temporal_precision: str) -> str:
    if level in STRONG_LEVELS:
        return "high"
    if level in MODERATE_LEVELS:
        return "medium" if temporal_precision != "unknown" else "medium_temporal_gap"
    return "low"


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
    if level in OBSERVED_LEVELS and relation in {
        "strong_polygon_intersects_patch",
        "strong_point_inside_patch",
        "moderate_point_inside_patch",
        "moderate_bbox_intersects_patch",
        "near_patch_buffer_candidate",
    }:
        return "true"
    return "false"


def _row(event, patch_id, relation, distance, score, score_class, interpretation, limitations):
    level = event.get("evidence_level", "")
    return {
        "linkage_id": "",
        "event_id": event.get("event_id", ""),
        "patch_id": patch_id,
        "region": event.get("region", ""),
        "event_date": event.get("event_date") or event.get("event_period_start", ""),
        "event_type": event.get("event_type", ""),
        "evidence_level": level,
        "spatial_relation": relation,
        "temporal_precision": event.get("temporal_precision", ""),
        "distance_m": "" if distance is None else round(distance, 1),
        "patch_score_v6": score,
        "patch_score_class_v6": score_class,
        "source_confidence": _source_confidence(level, event.get("temporal_precision", "")),
        "linkage_confidence": _link_confidence(relation, level),
        "observed_event_strength": _strength(level),
        "can_be_used_for_observational_evaluation": _can_eval(level, relation),
        "can_be_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
        "requires_manual_review": "true",
        "interpretation": interpretation,
        "limitations": limitations,
    }


def _load_event_points(event):
    path = (event.get("source_url_or_path") or "").strip()
    if not path:
        return []
    p = ROOT / path
    if not p.exists():
        return []
    ext = p.suffix.lower()
    try:
        if ext in {".geojson", ".json"}:
            pts, _ = legacy_parser._parse_geojson(p)
        elif ext in {".csv", ".tsv"}:
            pts, _ = legacy_parser._parse_csv_latlon(p)
        elif ext in {".wkt", ".txt"}:
            pts, _ = legacy_parser._parse_wkt_text(p)
        elif ext == ".kml":
            pts, _ = legacy_parser._parse_kml_text(p.read_text(encoding="utf-8", errors="ignore"))
        else:
            return []
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return [(lon, lat) for lon, lat in pts[:MAX_POINTS] if in_brazil(lon, lat)]


def _point_links(event, patches, scores):
    pts = _load_event_points(event)
    if not pts and event.get("lat") and event.get("lon"):
        try:
            pts = [(float(event["lon"]), float(event["lat"]))]
        except ValueError:
            pts = []
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
            limit = "Ponto de ocorrencia nao e footprint de area alagada; vinculo observacional review-only."
        else:
            relation = "near_patch_buffer_candidate"
            interp = f"{near_n} ponto(s) rastreavel(is) no buffer aproximado de 550m do patch."
            limit = "Proximidade nao confirma ocorrencia no patch; candidato apenas para revisao."
        out.append(_row(event, pid, relation, dist, score, score_class, interp, limit))
    return out


def _gap_rows():
    return [
        {
            "region": "recife",
            "institution": "Defesa Civil Recife / APAC / Dados Abertos Recife",
            "needed_file_type": "CSV ou GeoJSON oficial com pontos de ocorrencia/alagamento, data e lat/lon; ou shapefile/ZIP vetorial de mancha observada",
            "suggested_query": "Recife ocorrencias alagamento data coordenadas GeoJSON CSV Defesa Civil APAC",
            "why_needed": "Recife tem contexto e mancha candidata, mas precisa evento oficial observado com data e geometria explicita.",
            "priority": "high",
            "manual_action": "baixar manualmente arquivo oficial e colocar em datasets/suscetibilidade/observed_event_sources_susc13a/",
        },
        {
            "region": "petropolis",
            "institution": "CPRM/SGB / Defesa Civil Petropolis / INEA-RJ / Prefeitura de Petropolis",
            "needed_file_type": "footprint, poligono ou ocorrencia 2022 com coordenada/poligono e data",
            "suggested_query": "Petropolis 2022 inundacao area atingida shapefile coordenadas ocorrencias Defesa Civil INEA CPRM",
            "why_needed": "Os registros existentes tem pontos oficiais, mas faltam data/footprint forte para avaliacao observacional robusta.",
            "priority": "high",
            "manual_action": "coletar fonte oficial 2022 rastreavel e preservar URL/instituicao no nome ou metadado.",
        },
        {
            "region": "curitiba",
            "institution": "GeoCuritiba/IPPUC / Defesa Civil Curitiba / Prefeitura de Curitiba / IAT-Aguas Parana",
            "needed_file_type": "ocorrencias oficiais com geometria, CSV lat/lon, GeoJSON, KML/KMZ ou SHP ZIP",
            "suggested_query": "Curitiba pontos de alagamento ocorrencias Defesa Civil GeoCuritiba IPPUC shapefile",
            "why_needed": "Curitiba ainda carece de ocorrencias oficiais georreferenciadas no pacote observado.",
            "priority": "high",
            "manual_action": "baixar camada ou tabela oficial e colocar no diretorio manual SUSC-13A.",
        },
    ]


def _write_gap_report() -> None:
    rows = _gap_rows()
    write_csv(REQUEST_TABLE, rows, ["region", "institution", "needed_file_type", "suggested_query", "why_needed", "priority", "manual_action"])
    md = """# SUSC-13A - Lacunas de aquisicao manual

Status: `review_only=true` | `can_be_ground_truth=false` | `allowed_for_training=false`

## Recife
Buscar pontos de ocorrencia/alagamento com data e lat/lon ou shapefile/CSV oficial.
Prioridade: Defesa Civil Recife, APAC e Dados Abertos Recife.

## Petropolis
Buscar footprint/ocorrencia 2022 com coordenada/poligono.
Prioridade: CPRM/SGB, Defesa Civil Petropolis, INEA/RJ e Prefeitura de Petropolis.

## Curitiba
Buscar ocorrencias oficiais GeoCuritiba/IPPUC/Defesa Civil com geometria.
Prioridade: GeoCuritiba/IPPUC, Defesa Civil Curitiba, Prefeitura de Curitiba e IAT/Aguas Parana.

## Condicao metodologica
Arquivos manuais devem ser oficiais/tecnicos, rastreaveis e pequenos. Mesmo quando
fortes, permanecem review-only e nao criam ground truth ou treino supervisionado.
"""
    write_markdown(GAP_REPORT, md)


def _tbl(counter: Counter) -> str:
    if not counter:
        return "| nenhum | 0 |"
    return "\n".join(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _write_report(events, links, registry, downloads) -> None:
    by_level = Counter(e["evidence_level"] for e in events)
    by_region = Counter(e["region"] for e in events)
    link_rel = Counter(r["spatial_relation"] for r in links)
    dl_status = Counter(r["download_status"] for r in downloads)
    strong_events = sum(1 for e in events if e["evidence_level"] in STRONG_LEVELS)
    moderate_events = sum(1 for e in events if e["evidence_level"] in MODERATE_LEVELS)
    strong_mod_links = sum(1 for r in links if r["linkage_confidence"] in {"strong", "moderate"})
    eval_links = sum(1 for r in links if r["can_be_used_for_observational_evaluation"] == "true")
    improved_regions = sorted({r["region"] for r in links if r["can_be_used_for_observational_evaluation"] == "true"})
    md = f"""# SUSC-13A - Aquisicao forte de eventos observados reais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

{MANDATORY}

## 1. Objetivo
Fortalecer a camada observacional usada em SUSC-11/12, separando eventos fortes,
moderados, fracos e documentais sem promover nenhum item a rotulo operacional.

## 2. Fontes registradas
Total de fontes-alvo: **{len(registry)}**.

| regiao | fontes |
|---|---|
{_tbl(Counter(r["region"] for r in registry))}

## 3. Politica de aquisicao controlada
Downloads automaticos so ocorrem com URL direta leve, download permitido e
extensao permitida. Raster, Sentinel bruto, API com chave, scraping agressivo e
arquivos acima de 100MB permanecem bloqueados.

## 4. Resultado de downloads
Tentativas registradas: **{len(downloads)}**.

| status | n |
|---|---|
{_tbl(dl_status)}

## 5. Fontes manuais
O diretorio `datasets/suscetibilidade/observed_event_sources_susc13a/` aceita CSV,
GeoJSON, KML/KMZ, SHP ZIP, GPKG, WKT, PDF pequeno, XLSX e TXT com data, geometria,
tipo de evento, fonte e instituicao. Fontes incompletas exigem revisao manual.

## 6. Eventos parseados
Total de registros parseados/reclassificados: **{len(events)}**.

| nivel | n |
|---|---|
{_tbl(by_level)}

## 7. Eventos por regiao
| regiao | n |
|---|---|
{_tbl(by_region)}

## 8. Eventos fortes e moderados
Eventos fortes encontrados: **{strong_events}**.
Eventos moderados encontrados: **{moderate_events}**.
Eventos fortes exigem data/periodo e geometria explicita de alagamento/inundacao.
Registros moderados continuam uteis para avaliacao observacional, mas nao sao GT.

## 9. Linkage patch-evento
Linhas de linkage: **{len(links)}**; links fortes/moderados: **{strong_mod_links}**;
linhas permitidas para avaliacao observacional review-only: **{eval_links}**.

| relacao espacial | n |
|---|---|
{_tbl(link_rel)}

## 10. Regioes com melhora observacional
Regioes com algum vinculo forte/moderado ou avaliavel: **{", ".join(improved_regions) if improved_regions else "nenhuma"}**.
Melhora aqui significa apenas mais aderencia observacional review-only, nao
confirmacao operacional por patch.

## 11. Lacunas restantes
Recife: buscar pontos de ocorrencia/alagamento com data e lat/lon ou shapefile/CSV oficial.
Petropolis: buscar footprint/ocorrencia 2022 com coordenada/poligono.
Curitiba: buscar ocorrencias oficiais GeoCuritiba/IPPUC/Defesa Civil com geometria.

## 12. Governanca e limites
- `can_be_ground_truth=false` em todos os artefatos.
- `allowed_for_training=false` em todos os artefatos.
- `review_only=true` em todos os artefatos.
- Risco, alerta e contexto administrativo nao sao evento observado.
- Nenhum score v7, modelo, treino supervisionado ou confirmacao automatica foi criado.
"""
    write_markdown(REPORT, md)


def main() -> int:
    print("=" * 60)
    print("SUSC-13A Strong Event Patch Linkage")
    print("=" * 60)
    for path in (MATRIX, SCORE, PARSED):
        if not path.exists():
            print(f"STOP: missing input: {rel(path)}")
            return 1

    patches = _load_patches()
    scores = _load_scores()
    events = read_csv(PARSED)
    links = []
    for event in events:
        region = (event.get("region") or "").lower()
        level = event.get("evidence_level", "")
        geom = (event.get("geometry_type") or "").lower()
        bbox = parse_bbox(event.get("bbox", ""))
        patches_here = patches.get(region, [])

        if level not in OBSERVED_LEVELS:
            relation = "same_region_period_context" if event.get("event_date") or event.get("event_period_start") else "insufficient_for_patch_link"
            links.append(_row(event, "REGION_LEVEL_NO_PATCH_RESOLUTION", relation, None, "", "", "Contexto fraco/documental mantido sem resolucao por patch.", "Risco, alerta, administrativo ou documental nao e evento observado."))
            continue

        matched = False
        if geom in {"polygon", "multipolygon", "bbox"} and bbox:
            for pid, pbb, _centroid in patches_here:
                if bbox_intersect(bbox, pbb):
                    score, score_class = scores.get(pid, ("", ""))
                    relation = "strong_polygon_intersects_patch" if level == "strong_observed_flood_polygon" else "moderate_bbox_intersects_patch"
                    links.append(_row(event, pid, relation, None, score, score_class, "Geometria de evento intersecta bbox do patch.", "Interseccao por bbox e aproximacao review-only; nao e ground truth."))
                    matched = True
        else:
            point_links = _point_links(event, patches_here, scores)
            links.extend(point_links)
            matched = bool(point_links)

        if not matched:
            links.append(_row(event, "REGION_LEVEL_NO_PATCH_RESOLUTION", "same_region_period_context", None, "", "", "Evento observado/moderado sem sobreposicao/proximidade de patch nesta matriz.", "Associacao regional/periodo apenas; nao confirma patch."))

    for i, row in enumerate(links):
        row["linkage_id"] = f"S13ALINK_{i:05d}"
    write_csv(LINKAGE, links, FIELDS)

    rel_counts = Counter(r["spatial_relation"] for r in links)
    conf_counts = Counter(r["linkage_confidence"] for r in links)
    eval_links = sum(1 for r in links if r["can_be_used_for_observational_evaluation"] == "true")
    summary_rows = [
        {"metric": "total_events", "value": len(events), "review_only": "true"},
        {"metric": "total_links", "value": len(links), "review_only": "true"},
        {"metric": "links_by_spatial_relation", "value": json.dumps(dict(rel_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "links_by_confidence", "value": json.dumps(dict(conf_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "strong_events", "value": sum(1 for e in events if e["evidence_level"] in STRONG_LEVELS), "review_only": "true"},
        {"metric": "moderate_events", "value": sum(1 for e in events if e["evidence_level"] in MODERATE_LEVELS), "review_only": "true"},
        {"metric": "observational_evaluation_links", "value": eval_links, "review_only": "true"},
        {"metric": "training_allowed_count", "value": sum(1 for r in links if r["allowed_for_training"] == "true"), "review_only": "true"},
        {"metric": "ground_truth_count", "value": sum(1 for r in links if r["can_be_ground_truth"] == "true"), "review_only": "true"},
    ]
    write_csv(SUMMARY, summary_rows, ["metric", "value", "review_only"])

    limitations = {
        "artifact": "SUSC-13A strong observed event patch linkage",
        "n_events": len(events),
        "n_links": len(links),
        "spatial_relation_counts": dict(rel_counts),
        "linkage_confidence_counts": dict(conf_counts),
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "score_v7_created": False,
        "model_persisted": False,
        "key_limitations": [
            "Eventos fortes ainda sao review-only.",
            "Eventos moderados nao sao footprint validado nem rotulo de treino.",
            "Risco, alerta e registros administrativos nao sao evento observado.",
            "Interseccao por bbox e ponto-dentro-bbox sao aproximacoes para revisao.",
            "Sem ground truth, sem treino supervisionado e sem confirmacao operacional automatica.",
        ],
        "mandatory_statement": MANDATORY,
    }
    write_json(LIMITS, limitations)
    _write_gap_report()
    registry = read_csv(REGISTRY) if REGISTRY.exists() else []
    downloads = read_csv(DL_MANIFEST) if DL_MANIFEST.exists() else []
    _write_report(events, links, registry, downloads)

    print(f"linkage rows: {len(links)}")
    print("relations:", dict(rel_counts))
    print("confidence:", dict(conf_counts))
    print(f"observational evaluation links: {eval_links}")
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
