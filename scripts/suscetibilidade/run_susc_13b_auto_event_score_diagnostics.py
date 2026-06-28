"""
SUSC-13B-AUTO event vs score-v6 diagnostics (exploratory, review-only).

If the observational readiness gate has enough strong/moderate patch links, this
computes exploratory diagnostics of score v6 over event-linked patches (means,
class distribution, top-k hit rates, enrichment). If not, it writes a
``not_enough_observed_events`` status without failing. It never creates score v7
and never asserts a predictive claim.

Writes:
  - outputs_public/suscetibilidade/SUSC_13B_auto_event_score_diagnostics.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_event_score_diagnostics_by_region.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_event_score_diagnostics_summary.json
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_event_patch_linkage_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_consolidated_observed_event_catalog_v1.csv"
READINESS = OUT / "SUSC_13B_auto_observational_readiness.csv"
DIAG_CSV = OUT / "SUSC_13B_auto_event_score_diagnostics.csv"
DIAG_REGION = OUT / "SUSC_13B_auto_event_score_diagnostics_by_region.csv"
DIAG_SUMMARY = OUT / "SUSC_13B_auto_event_score_diagnostics_summary.json"
STRONG_REPORT = OUT / "SUSC_13B_AUTO_official_event_discovery_acquisition_report.md"

DISCOVERED = MAN / "susc_13b_auto_discovered_sources_v1.csv"
QUERY_MANIFEST = MAN / "susc_13b_auto_discovery_query_manifest_v1.csv"
DEBUG_LOG = OUT / "SUSC_13B_auto_discovery_debug_log.csv"
DL_MANIFEST = MAN / "susc_13b_auto_download_manifest_v1.csv"
PARSE_AUDIT = OUT / "SUSC_13B_auto_parse_audit.csv"

MANDATORY = (
    "O SUSC-13B-AUTO realiza descoberta e aquisição automática de fontes oficiais/rastreáveis "
    "para fortalecer a camada observacional de alagamento/inundação. Mesmo quando encontra "
    "eventos fortes, a etapa mantém todos os vínculos em modo review-only, não cria ground truth, "
    "não treina modelo supervisionado e não cria score v7 automaticamente."
)
STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}

MIN_LINKS_FOR_DIAGNOSTICS = 10


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _hit_rate(top_patch_ids: list[str], event_patches: set[str]) -> float:
    if not top_patch_ids:
        return 0.0
    hits = sum(1 for pid in top_patch_ids if pid in event_patches)
    return round(hits / len(top_patch_ids), 4)


def _write_not_ready(reason: str, n_links: int) -> None:
    status_rows = [{
        "status": "not_enough_observed_events", "reason": reason,
        "strong_or_moderate_links": n_links, "min_required": MIN_LINKS_FOR_DIAGNOSTICS,
        "review_only": "true",
    }]
    write_csv(DIAG_CSV, status_rows, ["status", "reason", "strong_or_moderate_links", "min_required", "review_only"])
    write_csv(DIAG_REGION, status_rows, ["status", "reason", "strong_or_moderate_links", "min_required", "review_only"])
    write_json(DIAG_SUMMARY, {
        "artifact": "SUSC-13B-AUTO event-score diagnostics",
        "status": "not_enough_observed_events", "reason": reason,
        "strong_or_moderate_links": n_links, "min_required": MIN_LINKS_FOR_DIAGNOSTICS,
        "score_v7_created": False, "can_be_ground_truth": False,
        "allowed_for_training": False, "review_only": True,
    })


def _tbl(counter: Counter) -> str:
    if not counter:
        return "| nenhum | 0 |"
    return "\n".join(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _write_strong_report() -> None:
    sources = read_csv(DISCOVERED) if DISCOVERED.exists() else []
    debug = read_csv(DEBUG_LOG) if DEBUG_LOG.exists() else []
    downloads = read_csv(DL_MANIFEST) if DL_MANIFEST.exists() else []
    catalog = read_csv(CATALOG) if CATALOG.exists() else []
    links = read_csv(LINKAGE) if LINKAGE.exists() else []
    readiness = {r["metric"]: r["value"] for r in read_csv(READINESS)} if READINESS.exists() else {}
    diag = read_json(DIAG_SUMMARY) if DIAG_SUMMARY.exists() else {}

    pref = [e for e in catalog if e.get("preferred_record", "true") != "false"]
    by_method = Counter(s["discovery_method"] for s in sources)
    ckan = sum(1 for s in sources if s["discovery_method"] == "ckan_api")
    arcgis = sum(1 for s in sources if s["discovery_method"] == "arcgis_rest")
    wfs = sum(1 for s in sources if s["discovery_method"] == "wfs_getcapabilities")
    portal = sum(1 for s in sources if s["discovery_method"] == "data_portal_html")
    dl_status = Counter(d["download_status"] for d in downloads)
    downloaded = sum(1 for d in downloads if d["download_status"] == "downloaded")
    strong_events = sum(1 for e in pref if e["evidence_level"] in STRONG)
    moderate_events = sum(1 for e in pref if e["evidence_level"] in MODERATE)
    weak_rej = sum(1 for e in pref if e["evidence_level"] not in (STRONG | MODERATE))
    with_date = sum(1 for e in pref if (e.get("event_date") or e.get("event_period_start")))
    with_geom = sum(1 for e in pref if (e.get("bbox") or (e.get("lat") and e.get("lon")) or e.get("wkt") or e.get("geojson_ref")))
    strong_mod_links = sum(1 for r in links if r.get("linkage_confidence") in {"strong", "moderate"})
    eval_links = sum(1 for r in links if r.get("can_be_used_for_observational_evaluation") == "true")

    def rd(k):
        return readiness.get(k, "n/a")

    diag_line = (f"hit@10={diag.get('hit_rate_top_10')}, hit@20={diag.get('hit_rate_top_20')}, "
                 f"hit@30={diag.get('hit_rate_top_30')}, enriquecimento={diag.get('event_enrichment_ratio_top20')}"
                 if diag.get("status") == "computed"
                 else f"bloqueado ({diag.get('status', 'n/a')}): {diag.get('reason', 'dados insuficientes')}")

    md = f"""# SUSC-13B-AUTO - Descoberta e aquisicao automatica de eventos oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

{MANDATORY}

## 1. Objetivo
Criar um motor automatico de busca, descoberta, download, parsing e validacao de
dados oficiais/rastreaveis de alagamento, inundacao, enxurrada e ocorrencias de
Defesa Civil para Recife, Petropolis e Curitiba, sem depender de upload manual.

## 2. Por que 13A/12B ainda eram insuficientes
O SUSC-13A fechou com 0 eventos fortes, 5 moderados, 1 link patch-evento
moderado, 29 links fracos por buffer e 0 downloads diretos. O gargalo central era
a ausencia de fonte oficial com data + geometria explicita. O 12B/12C nao tinha
contraste observacional robusto o suficiente para calibrar proxy.

## 3. Estrategia de descoberta automatica
Plano deterministico de consultas (CKAN, ArcGIS REST, WFS/GeoServer, portais de
dados abertos, HTML oficial), com probe live opt-in (`SUSC_13B_NETWORK=1`).
Offline e o modo padrao: cada tentativa registra `network_disabled` e nada e
fabricado. robots.txt e respeitado; sem chave de API; sem Google Maps; sem
geocoding generico; sem raster; limite de 250MB.

## 4. Portais consultados
Total de fontes/endpoints candidatos registrados: **{len(sources)}**.

| metodo | n |
|---|---|
{_tbl(by_method)}

## 5. Endpoints encontrados
Probes executados: **{len(debug)}**. Status por probe registrado em
`SUSC_13B_auto_discovery_debug_log.csv`.

## 6. CKANs consultados
Endpoints CKAN registrados: **{ckan}** (Recife, PE, RJ, PR, ANA, dados.gov).

## 7. ArcGIS/FeatureServer consultados
Servicos ArcGIS REST registrados: **{arcgis}** (GeoCuritiba/IPPUC, GeoSGB, INEA, ESIG Recife).

## 8. WFS/GeoServer consultados
Endpoints WFS/GeoServer registrados: **{wfs}** (IAT/Aguas Parana).

## 9. Sitemaps/HTML oficiais consultados
Portais HTML oficiais registrados: **{portal}** (prefeituras, APAC, DRM-RJ, S2iD, CEMADEN).

## 10. Downloads realizados
Tentativas de download: **{len(downloads)}**; concluidos: **{downloaded}**.

| status | n |
|---|---|
{_tbl(dl_status)}

## 11. Downloads bloqueados e motivo
Raster, Sentinel bruto, executavel, tipo desconhecido e arquivos acima de 250MB
sao bloqueados por politica. Offline, todos ficam como `not_attempted_network_disabled`.

## 12. Eventos fortes encontrados
**{strong_events}**. Evento forte exige fonte rastreavel, data/periodo e
geometria/coordenada explicita de alagamento/inundacao.

## 13. Eventos moderados encontrados
**{moderate_events}**.

## 14. Eventos fracos/rejeitados
**{weak_rej}** (risco, alerta, administrativo, documental ou rejeitado).

## 15. Geometrias obtidas
Eventos observados com geometria: **{with_geom}**.

## 16. Datas obtidas
Eventos observados com data/periodo: **{with_date}**.

## 17. Linkage patch-evento
Linhas de linkage: **{len(links)}**; links fortes/moderados: **{strong_mod_links}**;
linhas avaliaveis (observacional review-only): **{eval_links}**.

## 18. Diagnostico score-evento
{diag_line}

## 19. Readiness para 12A/12B/12C
- 12A temporal: **{rd('ready_for_12a_temporal')}**
- 12B contraste de features: **{rd('ready_for_12b_feature_contrast')}**
- 12C calibracao de proxy: **{rd('ready_for_12c_proxy_calibration')}**

## 20. Score v7 continua bloqueado?
Readiness para score v7: **{rd('ready_for_score_v7')}**. Mesmo se PRONTO, o
SUSC-13B-AUTO **nao cria score v7**: a criacao exige revisao humana e etapa
dedicada. Aqui o score v7 permanece bloqueado por governanca.

## 21. Limitacoes
- Offline e o padrao; sem rede nenhum evento novo e adquirido automaticamente.
- Endpoints sao raizes institucionais; nenhum URL de download direto e inventado.
- Risco, alerta, suscetibilidade e registro administrativo nunca sao evento observado.
- Centroide de municipio/bairro, Google Maps e geocoding generico nao sao geometria.
- Eventos fortes exigem data + geometria explicita; nada vira ground truth ou treino.

## 22. Proximo marco
Executar o probe live (`SUSC_13B_NETWORK=1`) em ambiente com rede autorizada para
materializar recursos CKAN/ArcGIS/WFS; ou intake manual de um arquivo oficial com
data + geometria no diretorio de aquisicao, o que reexecuta parse/consolidacao/
linkage/readiness automaticamente. Score v7 segue como marco futuro dedicado,
fora do escopo automatico.
"""
    write_markdown(STRONG_REPORT, md)


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Event vs Score-v6 Diagnostics")
    print("=" * 60)
    links = read_csv(LINKAGE) if LINKAGE.exists() else []
    scores = read_csv(SCORE) if SCORE.exists() else []
    strong_mod = [r for r in links if r.get("linkage_confidence") in {"strong", "moderate"}
                  and r.get("patch_id") not in {"", "REGION_LEVEL_NO_PATCH_RESOLUTION"}]

    if len(strong_mod) < MIN_LINKS_FOR_DIAGNOSTICS:
        reason = (f"apenas {len(strong_mod)} link(s) forte/moderado com patch resolvido; "
                  f"limiar minimo {MIN_LINKS_FOR_DIAGNOSTICS}.")
        _write_not_ready(reason, len(strong_mod))
        _write_strong_report()
        print(f"status: not_enough_observed_events ({len(strong_mod)} links)")
        print("review-only. No score v7. No predictive claim.")
        return 0

    # --- enough data: exploratory diagnostics ---
    event_patches = {r["patch_id"] for r in strong_mod}
    val_by_patch: dict[str, float] = {}
    class_by_patch: dict[str, str] = {}
    ranked_pairs: list[tuple[float, str]] = []
    for r in scores:
        v = _num(r.get("susc_score_v6_candidate"))
        pid = r["patch_id"]
        class_by_patch[pid] = r.get("susc_class_v6_candidate", "")
        if v is not None:
            val_by_patch[pid] = v
            ranked_pairs.append((v, pid))
    linked_scores: list[float] = [val_by_patch[pid] for pid in event_patches if pid in val_by_patch]
    ranked = [pid for _, pid in sorted(ranked_pairs, key=lambda t: -t[0])]

    mean_v = round(statistics.mean(linked_scores), 4) if linked_scores else 0.0
    median_v = round(statistics.median(linked_scores), 4) if linked_scores else 0.0
    class_dist = Counter(class_by_patch[pid] for pid in event_patches if pid in class_by_patch)
    base_rate = round(len(event_patches) / len(ranked), 4) if ranked else 0.0
    hr10, hr20, hr30 = _hit_rate(ranked[:10], event_patches), _hit_rate(ranked[:20], event_patches), _hit_rate(ranked[:30], event_patches)
    enrichment = round(hr20 / base_rate, 3) if base_rate else 0.0

    rows = [
        {"metric": "score_v6_mean_event_links", "value": mean_v},
        {"metric": "score_v6_median_event_links", "value": median_v},
        {"metric": "score_v6_class_distribution_event_links", "value": "|".join(f"{k}:{v}" for k, v in sorted(class_dist.items()))},
        {"metric": "n_event_linked_patches", "value": len(event_patches)},
        {"metric": "base_rate", "value": base_rate},
        {"metric": "hit_rate_top_10", "value": hr10},
        {"metric": "hit_rate_top_20", "value": hr20},
        {"metric": "hit_rate_top_30", "value": hr30},
        {"metric": "event_enrichment_ratio_top20", "value": enrichment},
    ]
    for r in rows:
        r["review_only"] = "true"
    write_csv(DIAG_CSV, rows, ["metric", "value", "review_only"])

    by_region = defaultdict(set)
    for r in strong_mod:
        by_region[r.get("region", "")].add(r["patch_id"])
    region_rows = []
    for region, pids in sorted(by_region.items()):
        rs: list[float] = [val_by_patch[pid] for pid in pids if pid in val_by_patch]
        region_rows.append({
            "region": region, "n_event_linked_patches": len(pids),
            "score_v6_mean": round(statistics.mean(rs), 4) if rs else 0.0,
            "score_v6_median": round(statistics.median(rs), 4) if rs else 0.0,
            "review_only": "true",
        })
    write_csv(DIAG_REGION, region_rows, ["region", "n_event_linked_patches", "score_v6_mean", "score_v6_median", "review_only"])

    write_json(DIAG_SUMMARY, {
        "artifact": "SUSC-13B-AUTO event-score diagnostics", "status": "computed",
        "n_event_linked_patches": len(event_patches), "score_v6_mean_event_links": mean_v,
        "score_v6_median_event_links": median_v, "hit_rate_top_10": hr10, "hit_rate_top_20": hr20,
        "hit_rate_top_30": hr30, "event_enrichment_ratio_top20": enrichment, "base_rate": base_rate,
        "interpretation": "Diagnostico exploratorio review-only; nao e validacao preditiva nem ground truth.",
        "score_v7_created": False, "can_be_ground_truth": False, "allowed_for_training": False, "review_only": True,
    })
    _write_strong_report()
    print(f"status: computed | linked patches {len(event_patches)} | hit@20 {hr20} | enrichment {enrichment}")
    print("review-only. No score v7. No predictive claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
