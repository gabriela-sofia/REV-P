"""
SUSC-14A FASE 9/10 - observational diagnostics, readiness and final report.

Reads the FASE 1-8 artifacts, computes conservative event/linkage diagnostics,
writes readiness outputs, and generates the final SUSC-14A report. This script
never creates score v7, ground truth, training data, or persisted models.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"

SCORE = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
CATALOG = DAT / "susc_14a_rescued_observed_event_catalog_v1.csv"
LINKAGE = DAT / "susc_14a_event_patch_linkage_v1.csv"
GEOCODED = DAT / "susc_14a_geocoded_official_occurrences_v1.csv"
REF_REGISTRY = MAN / "susc_14a_official_geocoding_reference_registry_v1.csv"
REF_DL = MAN / "susc_14a_official_reference_download_manifest_v1.csv"
REF_FEATURES = DAT / "susc_14a_official_geocoding_reference_features_v1.csv"
FOOTPRINT_MANIFEST = MAN / "susc_14a_flood_footprint_manifest_v1.csv"
GEOCODE_AUDIT = OUT / "SUSC_14A_official_occurrence_geocoding_audit.csv"

READINESS_CSV = OUT / "SUSC_14A_observational_readiness.csv"
READINESS_MD = OUT / "SUSC_14A_observational_readiness.md"
DIAG_CSV = OUT / "SUSC_14A_score_v6_event_diagnostics.csv"
DIAG_REGION = OUT / "SUSC_14A_score_v6_event_diagnostics_by_region.csv"
DIAG_SUMMARY = OUT / "SUSC_14A_score_v6_event_diagnostics_summary.json"
REPORT = OUT / "SUSC_14A_evidence_rescue_report.md"

MANDATORY = (
    "O SUSC-14A tenta resgatar evidencia observacional a partir de registros oficiais sem coordenada "
    "explicita, usando apenas referencias espaciais oficiais/rastreaveis e footprints publicos quando "
    "disponiveis. Eventos geocodificados por referencia oficial permanecem review-only, nao criam "
    "ground truth, nao liberam treino supervisionado e nao autorizam score v7 automatico."
)

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {
    "moderate_official_occurrence_point",
    "moderate_official_occurrence_geocoded",
    "moderate_official_street_segment_match",
    "moderate_official_flood_bbox",
}
OBSERVED = STRONG | MODERATE
MIN_LINKS = 10
NO_OBSERVED_REASON = "no_strong_or_moderate_patch_links"


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_geom(row):
    return bool(row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref"))


def _has_date(row):
    return bool(row.get("event_date") or row.get("event_period_start"))


def _hit_rate(top, event_patch_ids):
    return round(sum(1 for patch_id in top if patch_id in event_patch_ids) / len(top), 4) if top else 0.0


def _read(path):
    return read_csv(path) if path.exists() else []


def _metric(rows, metric, default="0"):
    for row in rows:
        if row.get("metric") == metric:
            return row.get("value", default)
    return default


def _int_metric(rows, metric):
    try:
        return int(_metric(rows, metric, "0"))
    except ValueError:
        return 0


def _csv_value(value):
    return "null" if value is None else value


def _is_context_link(row):
    return row.get("spatial_relation") in {"same_neighborhood_context", "same_region_period_context"}


def _bool_txt(value):
    return str(bool(value)).lower()


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Observational Diagnostics + Readiness + Report")
    print("=" * 60)

    events = _read(CATALOG)
    pref = [e for e in events if e.get("preferred_record", "true") != "false"]
    links = _read(LINKAGE)
    scores = _read(SCORE)
    geocoded = _read(GEOCODED)
    geocode_audit = _read(GEOCODE_AUDIT)

    strong_events = sum(1 for e in pref if e.get("evidence_level") in STRONG)
    moderate_events = sum(1 for e in pref if e.get("evidence_level") in MODERATE)
    weak_events = sum(1 for e in pref if e.get("evidence_level", "").startswith("weak_"))
    blocked_events = sum(
        1
        for e in pref
        if e.get("evidence_level", "").startswith("rejected_")
        or e.get("georeferencing_status", "").startswith("blocked_")
        or e.get("evidence_level") == "documentary_only"
    )
    observed = strong_events + moderate_events
    geocoded_official = sum(
        1
        for r in geocoded
        if r.get("georeferencing_status") in {"official_address_point_match", "official_street_segment_match"}
    )
    street_matches = sum(1 for r in geocoded if r.get("georeferencing_status") == "official_street_segment_match")
    geocoding_blocked = _int_metric(geocode_audit, "n_blocked_no_reference") + _int_metric(
        geocode_audit, "n_blocked_ambiguous"
    )
    with_date = sum(1 for e in pref if e.get("evidence_level") in OBSERVED and _has_date(e))
    with_geom = sum(1 for e in pref if e.get("evidence_level") in OBSERVED and _has_geom(e))
    strong_links = sum(1 for r in links if r.get("linkage_confidence") == "strong")
    moderate_links = sum(1 for r in links if r.get("linkage_confidence") == "moderate")
    weak_patch_links = sum(1 for r in links if r.get("linkage_confidence") == "weak")
    context_patch_links = sum(1 for r in links if _is_context_link(r))
    strong_mod_links = strong_links + moderate_links
    eval_patches = sorted(
        {
            r.get("patch_id", "")
            for r in links
            if r.get("can_be_used_for_observational_evaluation") == "true"
            and r.get("patch_id") not in {"", "REGION_LEVEL_NO_PATCH_RESOLUTION"}
        }
    )
    context_patches = sorted(
        {
            r.get("patch_id", "")
            for r in links
            if _is_context_link(r) and r.get("patch_id") not in {"", "REGION_LEVEL_NO_PATCH_RESOLUTION"}
        }
    )
    regions_covered = sorted({e.get("region", "") for e in pref if e.get("evidence_level") in OBSERVED and _has_geom(e)})
    observational_blocked = strong_mod_links == 0

    val_by_patch, class_by_patch, ranked_pairs = {}, {}, []
    for row in scores:
        patch_id = row.get("patch_id", "")
        score = _num(row.get("susc_score_v6_candidate"))
        class_by_patch[patch_id] = row.get("susc_class_v6_candidate", "")
        if score is not None:
            val_by_patch[patch_id] = score
            ranked_pairs.append((score, patch_id))
    ranked = [patch_id for _, patch_id in sorted(ranked_pairs, key=lambda item: -item[0])]

    strong_mod = [
        r
        for r in links
        if r.get("linkage_confidence") in {"strong", "moderate"}
        and r.get("patch_id") not in {"", "REGION_LEVEL_NO_PATCH_RESOLUTION"}
    ]
    if len(strong_mod) >= MIN_LINKS:
        event_patch_ids = {r["patch_id"] for r in strong_mod}
        linked_scores = [val_by_patch[p] for p in event_patch_ids if p in val_by_patch]
        mean_v = round(statistics.mean(linked_scores), 4) if linked_scores else None
        median_v = round(statistics.median(linked_scores), 4) if linked_scores else None
        class_dist = Counter(class_by_patch[p] for p in event_patch_ids if p in class_by_patch)
        class_dist_value = "|".join(f"{k}:{v}" for k, v in sorted(class_dist.items())) or "null"
        hr10 = _hit_rate(ranked[:10], event_patch_ids)
        hr20 = _hit_rate(ranked[:20], event_patch_ids)
        hr30 = _hit_rate(ranked[:30], event_patch_ids)
        enrichment = round(len(event_patch_ids) / (len(ranked) or 1), 4)
        diag_status = "computed"
        diag_rows = [
            {"metric": "score_v6_mean_event_links", "value": mean_v},
            {"metric": "score_v6_median_event_links", "value": median_v},
            {"metric": "score_v6_class_distribution_event_links", "value": class_dist_value},
            {"metric": "n_event_linked_patches", "value": len(event_patch_ids)},
            {"metric": "hit_rate_top_10", "value": hr10},
            {"metric": "hit_rate_top_20", "value": hr20},
            {"metric": "hit_rate_top_30", "value": hr30},
            {"metric": "event_enrichment_ratio", "value": enrichment},
            {"metric": "observational_evaluation_blocked", "value": "false"},
            {"metric": "reason", "value": ""},
        ]
        by_region = defaultdict(set)
        for row in strong_mod:
            by_region[row.get("region", "")].add(row["patch_id"])
        region_rows = []
        for region, patch_ids in sorted(by_region.items()):
            region_scores = [val_by_patch[p] for p in patch_ids if p in val_by_patch]
            region_rows.append(
                {
                    "region": region,
                    "n_event_linked_patches": len(patch_ids),
                    "score_v6_mean": round(statistics.mean(region_scores), 4) if region_scores else "null",
                    "review_only": "true",
                }
            )
    else:
        mean_v = median_v = hr10 = hr20 = hr30 = enrichment = None
        class_dist_value = "null"
        diag_status = "not_enough_observed_events"
        diag_rows = [
            {"metric": "status", "value": diag_status},
            {"metric": "strong_or_moderate_links", "value": len(strong_mod)},
            {"metric": "min_required", "value": MIN_LINKS},
            {"metric": "score_v6_mean_event_links", "value": "null"},
            {"metric": "score_v6_median_event_links", "value": "null"},
            {"metric": "score_v6_class_distribution_event_links", "value": "null"},
            {"metric": "n_event_linked_patches", "value": 0},
            {"metric": "hit_rate_top_10", "value": "null"},
            {"metric": "hit_rate_top_20", "value": "null"},
            {"metric": "hit_rate_top_30", "value": "null"},
            {"metric": "event_enrichment_ratio", "value": "null"},
            {"metric": "observational_evaluation_blocked", "value": "true"},
            {"metric": "reason", "value": NO_OBSERVED_REASON},
        ]
        region_rows = [{"region": "n/a", "n_event_linked_patches": 0, "score_v6_mean": "null", "review_only": "true"}]

    for row in diag_rows:
        row["review_only"] = "true"
    write_csv(DIAG_CSV, diag_rows, ["metric", "value", "review_only"])
    write_csv(DIAG_REGION, region_rows, ["region", "n_event_linked_patches", "score_v6_mean", "review_only"])
    write_json(
        DIAG_SUMMARY,
        {
            "artifact": "SUSC-14A event-score diagnostics",
            "status": diag_status,
            "observational_evaluation_blocked": observational_blocked,
            "reason": NO_OBSERVED_REASON if observational_blocked else "",
            "strong_or_moderate_links": len(strong_mod),
            "score_v6_mean_event_links": mean_v,
            "score_v6_median_event_links": median_v,
            "score_v6_class_distribution_event_links": None if class_dist_value == "null" else class_dist_value,
            "hit_rate_top_10": hr10,
            "hit_rate_top_20": hr20,
            "hit_rate_top_30": hr30,
            "event_enrichment_ratio": enrichment,
            "score_v7_created": False,
            "can_be_ground_truth": False,
            "can_be_used_as_ground_truth": False,
            "allowed_for_training": False,
            "review_only": True,
        },
    )

    source_counts = Counter(e.get("source_institution", "") for e in pref if e.get("evidence_level") in OBSERVED)
    single_dominant = bool(source_counts) and max(source_counts.values()) >= 0.8 * sum(source_counts.values())
    ready_12a = strong_mod_links >= 10 and with_date >= 10
    ready_12b = strong_mod_links >= 10
    ready_12c = strong_mod_links >= 20 and len(regions_covered) >= 2
    ready_v7 = strong_mod_links >= 20 and with_geom >= 20 and with_date >= 20 and len(regions_covered) >= 2 and not single_dominant
    readiness_rows = [
        ("strong_events_count", strong_events),
        ("moderate_events_count", moderate_events),
        ("weak_events_count", weak_events),
        ("blocked_events_count", blocked_events),
        ("geocoded_official_occurrences_count", geocoded_official),
        ("geocoding_blocked_count", geocoding_blocked),
        ("street_segment_matches_count", street_matches),
        ("events_with_date_count", with_date),
        ("events_with_geometry_count", with_geom),
        ("strong_patch_links_count", strong_links),
        ("moderate_patch_links_count", moderate_links),
        ("weak_patch_links_count", weak_patch_links),
        ("context_patch_links_count", context_patch_links),
        ("regions_covered", "|".join(regions_covered) or "none"),
        ("regions_covered_count", len(regions_covered)),
        ("patches_with_observational_evaluation", len(eval_patches)),
        ("patches_with_only_context", len(context_patches)),
        ("score_v6_mean_event_links", _csv_value(mean_v)),
        ("score_v6_median_event_links", _csv_value(median_v)),
        ("score_v6_class_distribution_event_links", class_dist_value),
        ("hit_rate_top_10", _csv_value(hr10)),
        ("hit_rate_top_20", _csv_value(hr20)),
        ("hit_rate_top_30", _csv_value(hr30)),
        ("event_enrichment_ratio", _csv_value(enrichment)),
        ("observational_evaluation_blocked", _bool_txt(observational_blocked)),
        ("reason", NO_OBSERVED_REASON if observational_blocked else ""),
        ("single_source_dominant", _bool_txt(single_dominant)),
        ("ready_for_12a_temporal", _bool_txt(ready_12a)),
        ("ready_for_12b_feature_contrast", _bool_txt(ready_12b)),
        ("ready_for_12c_proxy_calibration", _bool_txt(ready_12c)),
        ("ready_for_score_v7", _bool_txt(ready_v7)),
        ("score_v7_status", "blocked" if not ready_v7 else "requires_human_review_before_any_future_action"),
    ]
    write_csv(
        READINESS_CSV,
        [{"metric": key, "value": value, "review_only": "true"} for key, value in readiness_rows],
        ["metric", "value", "review_only"],
    )
    _write_readiness_md(
        strong_events,
        moderate_events,
        weak_events,
        blocked_events,
        geocoded_official,
        geocoding_blocked,
        street_matches,
        with_date,
        with_geom,
        strong_links,
        moderate_links,
        weak_patch_links,
        context_patch_links,
        eval_patches,
        context_patches,
        regions_covered,
        diag_status,
        mean_v,
        median_v,
        class_dist_value,
        hr10,
        hr20,
        hr30,
        enrichment,
        observational_blocked,
        ready_12a,
        ready_12b,
        ready_12c,
        ready_v7,
    )
    _write_report(
        strong_events,
        moderate_events,
        observed,
        weak_events,
        blocked_events,
        geocoded_official,
        street_matches,
        geocoding_blocked,
        with_date,
        with_geom,
        strong_links,
        moderate_links,
        weak_patch_links,
        context_patch_links,
        len(eval_patches),
        len(context_patches),
        regions_covered,
        diag_status,
        mean_v,
        median_v,
        class_dist_value,
        hr10,
        hr20,
        hr30,
        enrichment,
        ready_12a,
        ready_12b,
        ready_12c,
        ready_v7,
        links,
        geocoded,
    )

    print(f"events strong={strong_events} moderate={moderate_events} weak={weak_events} blocked={blocked_events}")
    print(f"official geocoded={geocoded_official} blocked geocoding={geocoding_blocked}")
    print(f"patch links strong={strong_links} moderate={moderate_links} context={context_patch_links}")
    print(f"readiness 12a={ready_12a} 12b={ready_12b} 12c={ready_12c} v7={ready_v7}")
    print("review-only. Score v7 NOT created.")
    return 0


def _ready(value):
    return "PRONTO" if value else "BLOQUEADO"


def _write_readiness_md(
    strong_events,
    moderate_events,
    weak_events,
    blocked_events,
    geocoded_official,
    geocoding_blocked,
    street_matches,
    with_date,
    with_geom,
    strong_links,
    moderate_links,
    weak_patch_links,
    context_patch_links,
    eval_patches,
    context_patches,
    regions_covered,
    diag_status,
    mean_v,
    median_v,
    class_dist_value,
    hr10,
    hr20,
    hr30,
    enrichment,
    observational_blocked,
    ready_12a,
    ready_12b,
    ready_12c,
    ready_v7,
):
    write_markdown(
        READINESS_MD,
        f"""# SUSC-14A - Prontidao observacional

Status: **review-only** | `can_be_ground_truth=false` | `can_be_used_as_ground_truth=false` | `allowed_for_training=false`

## Contagens
- Eventos fortes: **{strong_events}** | moderados: **{moderate_events}** | fracos/contextuais: **{weak_events}** | bloqueados/documentais: **{blocked_events}**
- Ocorrencias oficiais geocodificadas por camada oficial: **{geocoded_official}** (segmento de logradouro: **{street_matches}**)
- Geocodificacoes bloqueadas por referencia ausente/ambigua: **{geocoding_blocked}**
- Eventos observados com data/periodo: **{with_date}** | com geometria: **{with_geom}**
- Links patch-evento fortes: **{strong_links}** | moderados: **{moderate_links}** | fracos: **{weak_patch_links}** | contextuais: **{context_patch_links}**
- Patches com avaliacao observacional: **{len(eval_patches)}** | patches apenas contextuais: **{len(context_patches)}**
- Regioes cobertas por observacao geometrica: **{", ".join(regions_covered) or "nenhuma"}**

## Diagnostico score v6 x eventos
- Status: **{diag_status}**
- Media/mediana v6 em links observacionais: **{mean_v}** / **{median_v}**
- Distribuicao de classe v6: **{class_dist_value}**
- hit@10={hr10} | hit@20={hr20} | hit@30={hr30} | enriquecimento={enrichment}
- Avaliacao observacional bloqueada: **{_bool_txt(observational_blocked)}** | motivo: **{NO_OBSERVED_REASON if observational_blocked else "n/a"}**

## Prontidao
- SUSC-12A temporal: **{_ready(ready_12a)}**
- SUSC-12B contraste de features: **{_ready(ready_12b)}**
- SUSC-12C calibracao de proxy: **{_ready(ready_12c)}**
- Score v7: **{_ready(ready_v7)}**

O SUSC-14A conseguiu medir a rastreabilidade do bloqueio: ha registros oficiais
textuais, mas nao ha geometria oficial/rastreavel suficiente para avaliacao
patch-level. Zero geocodificacoes pode ser resultado fiel quando a camada oficial
disponivel nao cobre os logradouros das ocorrencias, e nao deve ser tratado como
falha automatica. O score v7 permanece bloqueado ate que existam links fortes ou
moderados suficientes, com data e geometria oficial.
""",
    )


def _write_report(
    strong_events,
    moderate_events,
    observed,
    weak_events,
    blocked_events,
    geocoded_official,
    street_matches,
    geocoding_blocked,
    with_date,
    with_geom,
    strong_links,
    moderate_links,
    weak_patch_links,
    context_patch_links,
    eval_patch_count,
    context_patch_count,
    regions_covered,
    diag_status,
    mean_v,
    median_v,
    class_dist_value,
    hr10,
    hr20,
    hr30,
    enrichment,
    ready_12a,
    ready_12b,
    ready_12c,
    ready_v7,
    links,
    geocoded,
):
    refs = _read(REF_REGISTRY)
    downloads = _read(REF_DL)
    features = _read(REF_FEATURES)
    footprints = _read(FOOTPRINT_MANIFEST)
    geocode_audit = _read(GEOCODE_AUDIT)

    n_occ = _metric(geocode_audit, "n_occurrences", "0")
    n_hood = _metric(geocode_audit, "n_neighborhood_only", "0")
    n_blocked_ref = _metric(geocode_audit, "n_blocked_no_reference", "0")
    refs_by_region = Counter(r.get("region", "") for r in refs)
    refs_by_type = Counter(r.get("source_type", "") for r in refs)
    download_candidates = sum(1 for r in refs if r.get("download_candidate") == "true")
    reused_downloads = sum(1 for r in downloads if r.get("download_status") == "reused_local_official")
    blocked_downloads = sum(1 for r in downloads if r.get("download_status") == "network_disabled")
    no_direct_downloads = sum(1 for r in downloads if r.get("download_status") == "endpoint_only")
    large_marked = sum(1 for r in downloads if r.get("blocked_reason"))
    feature_types = Counter(r.get("feature_type", "") for r in features)
    features_latlon = sum(1 for r in features if r.get("lat") and r.get("lon"))
    features_geom = sum(1 for r in features if r.get("bbox") or r.get("wkt") or r.get("geojson_ref"))
    geocode_status = Counter(r.get("georeferencing_status", "") for r in geocoded)
    geocode_levels = Counter(r.get("evidence_level", "") for r in geocoded)
    footprint_status = Counter(r.get("acquisition_status", "") for r in footprints)
    footprint_vector = sum(1 for r in footprints if r.get("geometry_available") == "true")
    strong_mod_links = strong_links + moderate_links
    total_links = len(links)

    write_markdown(
        REPORT,
        f"""# SUSC-14A - Resgate de evidencia observacional

Status: **review-only** | `can_be_ground_truth=false` | `can_be_used_as_ground_truth=false` | `allowed_for_training=false`

{MANDATORY}

## 1. Por que 13C bloqueou
O SUSC-13C encontrou registros oficiais textuais de cheia com data, bairro e
endereco, mas sem coordenada/poligono explicito suficiente. Sem geometria oficial
patch-level, esses registros nao liberam evento forte, treino supervisionado nem
score v7.

## 2. O que o 14A tentou resolver
O 14A reprocessou registros oficiais sem coordenada explicita contra referencias
espaciais oficiais/rastreaveis e footprints publicos disponiveis, sem Google
Maps, sem API com chave, sem geocoding generico, sem centroide de bairro/municipio
e sem coordenada inventada.

## 3. Referencias oficiais descobertas
Total: **{len(refs)}**. Por regiao: **{dict(refs_by_region)}**. Por source_type:
**{dict(refs_by_type)}**. Download candidate=true: **{download_candidates}**.

## 4. Referencias baixadas
Tentativas registradas: **{len(downloads)}**. Reutilizadas localmente:
**{reused_downloads}**. Bloqueadas por rede/offline: **{blocked_downloads}**. Sem
URL direta/endpoint: **{no_direct_downloads}**. Arquivos/endpoints marcados para
revisao manual: **{large_marked}**.

## 5. Referencias parseadas
Features parseadas: **{len(features)}**. Por feature_type: **{dict(feature_types)}**.
Com lat/lon: **{features_latlon}**. Com bbox/wkt/geojson_ref: **{features_geom}**.

## 6. Ocorrencias oficiais sem coordenada reprocessadas
Ocorrencias reprocessadas pelo geocoder oficial: **{n_occ}**.

## 7. Ocorrencias de cheia filtradas
Ocorrencias oficiais filtradas como cheia/inundacao para tentativa de resgate:
**{n_occ}**.

## 8. Matches por endereco/logradouro/camada oficial
Matches oficiais por ponto/segmento de endereco: **{geocoded_official}**.
Street segment matches: **{street_matches}**. Contexto de bairro: **{n_hood}**.
Status de geocoding: **{dict(geocode_status)}**. Evidence level:
**{dict(geocode_levels)}**.

## 9. Casos bloqueados e motivo
Geocodificacoes bloqueadas por referencia ausente/ambigua: **{geocoding_blocked}**.
O principal motivo e `blocked_no_official_reference`: a camada oficial local
parseada cobre pontos especificos e nao cobre os logradouros predominantes das
ocorrencias de cheia. Bloqueadas sem referencia: **{n_blocked_ref}**.

## 10. Footprints descobertos ou ausentes
Fontes registradas: **{len(footprints)}**. Footprints vetoriais disponiveis:
**{footprint_vector}**. Status: **{dict(footprint_status)}**.

## 11. Eventos resgatados fortes/moderados/fracos
Fortes: **{strong_events}**. Moderados: **{moderate_events}**. Observados:
**{observed}**. Fracos/contextuais: **{weak_events}**. Bloqueados/documentais:
**{blocked_events}**.

## 12. Catalogo consolidado
O catalogo consolidado preserva os registros preferenciais e suas flags de
governanca. Todos permanecem review-only, sem ground truth e sem treino.

## 13. Linkage evento-patch
Links totais: **{total_links}**. Links fortes: **{strong_links}**. Links
moderados: **{moderate_links}**. Links fracos: **{weak_patch_links}**. Links
contextuais: **{context_patch_links}**.

## 14. Links fortes/moderados/fracos/contextuais
Links fortes/moderados: **{strong_mod_links}**. Patches com avaliacao
observacional: **{eval_patch_count}**. Patches apenas contextuais:
**{context_patch_count}**.

## 15. Score v6 x eventos
Diagnostico: **{diag_status}**. Media v6: **{mean_v}**. Mediana v6:
**{median_v}**. Distribuicao: **{class_dist_value}**. hit@10={hr10},
hit@20={hr20}, hit@30={hr30}, enriquecimento={enrichment}. Sem links
fortes/moderados suficientes, essas metricas ficam nulas.

## 16. Readiness 12A
SUSC-12A temporal: **{_ready(ready_12a)}**. Requer pelo menos 10 eventos/linkages
fortes ou moderados com data/periodo.

## 17. Readiness 12B
SUSC-12B contraste de features: **{_ready(ready_12b)}**. Requer pelo menos 10
patches fortes/moderados/controlaveis.

## 18. Readiness 12C
SUSC-12C calibracao de proxy: **{_ready(ready_12c)}**. Requer pelo menos 20
patches fortes/moderados em pelo menos duas regioes.

## 19. Readiness score v7
Score v7: **{_ready(ready_v7)}**.

## 20. Por que score v7 continua bloqueado ou nao
O score v7 permanece bloqueado porque nao ha quantidade minima de vinculos
patch-evento fortes/moderados com data e geometria oficial/rastreavel. O SUSC-14A
melhora a rastreabilidade do bloqueio, mas nao autoriza calibracao automatica.

## Achado negativo principal

A tentativa offline de georreferenciar ocorrencias oficiais de cheia a partir da
referencia oficial disponivel nao produziu matches geocodificados suficientes. A
camada oficial local parseada cobre pontos especificos e nao cobre os logradouros
predominantes das ocorrencias de cheia. Portanto, o gargalo de 13C permanece: ha
ocorrencia oficial textual com data/bairro/endereco, mas ainda nao ha geometria
oficial suficiente para validacao patch-level.

## Consequencia para score v7

O score v7 permanece bloqueado porque nao ha quantidade minima de vinculos
patch-evento fortes/moderados com data e geometria oficial/rastreavel. O SUSC-14A
melhora a rastreabilidade do bloqueio, mas nao autoriza calibracao automatica.

## 21. Limitacoes
Ocorrencia geocodificada por logradouro oficial e no maximo moderada; nunca
strong. Street segment match nunca vira strong. Bairro, municipio, risco, alerta
e suscetibilidade nao viram ocorrencia observada patch-level.

## 22. O que nao pode ser afirmado
Nao pode ser afirmado que ha ground truth, treino supervisionado liberado,
validacao operacional, score v7 pronto, ou relacao causal patch-evento.

## 23. O que ja pode ser afirmado
Pode ser afirmado que o gargalo foi reprocessado de forma rastreavel e
fail-closed. A referencia oficial local disponivel e insuficiente para converter
as ocorrencias textuais em geometria patch-level. O achado negativo e
reprodutivel pelos artefatos 14A.

## 24. Proximo marco recomendado
SUSC-14B: aquisicao com rede habilitada da camada completa de logradouros/eixos
de via e bairros (Recife/Curitiba/Petropolis) e de footprints vetoriais oficiais
com data, mantendo review-only e revisao humana antes de qualquer promocao.
""",
    )


if __name__ == "__main__":
    raise SystemExit(main())
