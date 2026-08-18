"""
SUSC-13C event vs score-v6 diagnostics + observational readiness (review-only).

Computes strong/moderate counts, temporal/geometry coverage, patch-link counts,
score-v6 means and top-k hit rates over event-linked patches, and conservative
readiness flags for SUSC-12A/12B/12C and score v7. Never creates score v7.

Writes:
  - outputs_public/suscetibilidade/SUSC_13C_event_score_diagnostics.csv
  - outputs_public/suscetibilidade/SUSC_13C_event_score_diagnostics_by_region.csv
  - outputs_public/suscetibilidade/SUSC_13C_event_score_diagnostics_summary.json
  - outputs_public/suscetibilidade/SUSC_13C_observational_readiness.csv
  - outputs_public/suscetibilidade/SUSC_13C_observational_readiness.md
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, write_markdown  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_13c_consolidated_observed_event_catalog_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13c_event_patch_linkage_v1.csv"
DIAG_CSV = OUT / "SUSC_13C_event_score_diagnostics.csv"
DIAG_REGION = OUT / "SUSC_13C_event_score_diagnostics_by_region.csv"
DIAG_SUMMARY = OUT / "SUSC_13C_event_score_diagnostics_summary.json"
READINESS_CSV = OUT / "SUSC_13C_observational_readiness.csv"
READINESS_MD = OUT / "SUSC_13C_observational_readiness.md"

FINAL_REPORT = OUT / "SUSC_13C_LIVE_official_event_acquisition_report.md"
HEALTH_JSON = OUT / "SUSC_13C_live_network_healthcheck.json"
LIVE_SOURCES = OUT / "SUSC_13C_live_discovered_sources.csv"
CKAN_CSV = OUT / "SUSC_13C_live_ckan_resources.csv"
ARCGIS_CSV = OUT / "SUSC_13C_live_arcgis_layers.csv"
WFS_CSV = OUT / "SUSC_13C_live_wfs_layers.csv"
HTML_CSV = OUT / "SUSC_13C_live_html_file_links.csv"
DL_MANIFEST = MAN / "susc_13c_live_download_manifest_v1.csv"

MANDATORY = (
    "O SUSC-13C-LIVE executa aquisição online real de fontes oficiais/rastreáveis para tentar "
    "materializar eventos observados de alagamento/inundação com data e geometria. Mesmo quando "
    "eventos fortes ou moderados são encontrados, todos os vínculos permanecem review-only, sem "
    "ground truth, sem treino supervisionado, sem score v7 automático e sem uso operacional preditivo."
)

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}
OBSERVED = STRONG | MODERATE
MIN_LINKS = 10


def _read(path):
    from susc_io import read_csv as _rc
    return _rc(path) if path.exists() else []


def _write_final_report(pref, links, diag_status, hr10, hr20, hr30, mean_v,
                        ready_12a, ready_12b, ready_12c, ready_v7,
                        strong_events, moderate_events, with_date, with_geom,
                        strong_links, moderate_links):
    health = read_json(HEALTH_JSON) if HEALTH_JSON.exists() else {}
    ckan = _read(CKAN_CSV)
    arcgis = _read(ARCGIS_CSV)
    wfs = _read(WFS_CSV)
    html = _read(HTML_CSV)
    dl = _read(DL_MANIFEST)
    reached = health.get("reachable_domains", [])
    blocked = health.get("blocked_domains", [])
    ckan_cands = sum(1 for r in ckan if r.get("is_download_candidate") == "true")
    arcgis_layers = sum(1 for r in arcgis if r.get("layer_name"))
    wfs_layers = sum(1 for r in wfs if r.get("feature_type"))
    html_links = sum(1 for r in html if r.get("file_url"))
    downloaded = sum(1 for r in dl if r.get("download_status") in {"downloaded", "downloaded_by_probe"})
    blocked_dl = sum(1 for r in dl if r.get("download_status") == "blocked")
    weak_rej = sum(1 for e in pref if e["evidence_level"] not in OBSERVED)
    strong_mod_links = strong_links + moderate_links

    def yn(b):
        return "PRONTO" if b else "BLOQUEADO"

    md = f"""# SUSC-13C-LIVE - Aquisicao online real de eventos oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

{MANDATORY}

## 1. Objetivo
Executar de verdade a busca online (rede habilitada) para encontrar e baixar dados
oficiais/rastreaveis de alagamento/inundacao/enxurrada e ocorrencias de Defesa
Civil em Recife, Petropolis e Curitiba.

## 2. 13B-AUTO offline x 13C-LIVE
O 13B-AUTO criou a infraestrutura mas rodou offline (0 downloads). O 13C-LIVE
habilita `SUSC_13B_NETWORK=1` e executa probes reais com status HTTP, URLs,
content-type, timestamps, erros e resultados registrados.

## 3. Healthcheck de rede
Dominios alcancados: **{health.get('n_reachable', '?')}/{health.get('n_domains', '?')}** (minimo {health.get('min_reachable_required', 3)}); rede utilizavel: **{health.get('network_ok')}**.

## 4. Dominios alcancados
{chr(10).join('- ' + u for u in reached) or '- nenhum'}

## 5. Dominios bloqueados
{chr(10).join('- ' + u for u in blocked) or '- nenhum'}

## 6. Queries live executadas
CKAN: 14 termos x 7 bases; ArcGIS: walk de raizes/servicos/layers; WFS:
GetCapabilities; HTML: crawl limitado (depth<=2, <=80 paginas/dominio, robots).

## 7-8. CKANs consultados e recursos encontrados
Fontes CKAN consultadas: 7 bases. Recursos candidatos: **{ckan_cands}** (em {sum(1 for r in ckan if r.get('url'))} recursos listados).

## 9-10. ArcGIS/FeatureServer e layers
Raizes ArcGIS consultadas: 4. Layers com keyword: **{arcgis_layers}**.

## 11-12. WFS/GeoServer e layers
Endpoints WFS consultados: 3. FeatureTypes com keyword: **{wfs_layers}**.

## 13. Links HTML oficiais encontrados
**{html_links}**.

## 14-16. Downloads
Tentativas: **{len(dl)}**; concluidos: **{downloaded}**; bloqueados: **{blocked_dl}**.

## 17-19. Eventos
Fortes: **{strong_events}** | moderados: **{moderate_events}** | fracos/rejeitados: **{weak_rej}**.

## 20-21. Datas e geometrias
Eventos observados com data: **{with_date}** | com geometria: **{with_geom}**.

## 22. Linkage evento-patch
Linhas: **{len(links)}**; fortes/moderados: **{strong_mod_links}**.

## 23. Score v6 x eventos
Diagnostico: **{diag_status}** | hit@10={hr10} hit@20={hr20} hit@30={hr30} | media v6 (links)={mean_v}.

## 24. Readiness 12A/12B/12C
12A: **{yn(ready_12a)}** | 12B: **{yn(ready_12b)}** | 12C: **{yn(ready_12c)}**.

## 25. Score v7 bloqueado?
Readiness v7: **{yn(ready_v7)}**. O 13C-LIVE **nao cria score v7** por governanca,
mesmo se os limiares fossem atingidos.

## 26. Limitacoes
- Eventos oficiais de ocorrencia (ex.: atendimentos da Defesa Civil) muitas vezes
  vem em CSV/tabela sem coordenada explicita: viram moderados/contexto, nao fortes.
- Sem coordenada/poligono explicito + data, nao ha evento forte.
- Risco/alerta/administrativo nunca viram evento observado.
- Tudo permanece review-only; nada vira ground truth, treino ou score v7.

## 27. Proximo marco
Aprofundar recursos com geometria (GeoJSON/ArcGIS/WFS) e cruzar tabelas de
ocorrencia com camadas georreferenciadas oficiais; manter revisao humana antes de
qualquer promocao. Score v7 segue como marco futuro dedicado.
"""
    write_markdown(FINAL_REPORT, md)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _has_geom(e):
    return bool(e.get("bbox") or (e.get("lat") and e.get("lon")) or e.get("wkt") or e.get("geojson_ref"))


def _has_date(e):
    return bool(e.get("event_date") or e.get("event_period_start"))


def _hit_rate(top, eset):
    return round(sum(1 for p in top if p in eset) / len(top), 4) if top else 0.0


def main() -> int:
    print("=" * 60)
    print("SUSC-13C Event-Score Diagnostics + Readiness")
    print("=" * 60)
    events = read_csv(CATALOG) if CATALOG.exists() else []
    links = read_csv(LINKAGE) if LINKAGE.exists() else []
    scores = read_csv(SCORE) if SCORE.exists() else []
    pref = [e for e in events if e.get("preferred_record", "true") != "false"]

    strong_events = sum(1 for e in pref if e["evidence_level"] in STRONG)
    moderate_events = sum(1 for e in pref if e["evidence_level"] in MODERATE)
    observed = strong_events + moderate_events
    with_date = sum(1 for e in pref if e["evidence_level"] in OBSERVED and _has_date(e))
    with_geom = sum(1 for e in pref if e["evidence_level"] in OBSERVED and _has_geom(e))
    strong_links = sum(1 for r in links if r.get("linkage_confidence") == "strong")
    moderate_links = sum(1 for r in links if r.get("linkage_confidence") == "moderate")
    strong_mod_links = strong_links + moderate_links
    regions_covered = sorted({e["region"] for e in pref if e["evidence_level"] in OBSERVED and _has_geom(e)})

    # --- diagnostics (only if enough strong/moderate links) ---
    strong_mod = [r for r in links if r.get("linkage_confidence") in {"strong", "moderate"}
                  and r.get("patch_id") not in {"", "REGION_LEVEL_NO_PATCH_RESOLUTION"}]
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
    ranked = [pid for _, pid in sorted(ranked_pairs, key=lambda t: -t[0])]

    if len(strong_mod) >= MIN_LINKS:
        eset = {r["patch_id"] for r in strong_mod}
        linked = [val_by_patch[p] for p in eset if p in val_by_patch]
        mean_v = round(statistics.mean(linked), 4) if linked else 0.0
        median_v = round(statistics.median(linked), 4) if linked else 0.0
        class_dist = Counter(class_by_patch[p] for p in eset if p in class_by_patch)
        hr10, hr20, hr30 = _hit_rate(ranked[:10], eset), _hit_rate(ranked[:20], eset), _hit_rate(ranked[:30], eset)
        diag_rows = [
            {"metric": "score_v6_mean_event_links", "value": mean_v},
            {"metric": "score_v6_median_event_links", "value": median_v},
            {"metric": "score_v6_class_distribution_event_links", "value": "|".join(f"{k}:{v}" for k, v in sorted(class_dist.items()))},
            {"metric": "n_event_linked_patches", "value": len(eset)},
            {"metric": "hit_rate_top_10", "value": hr10},
            {"metric": "hit_rate_top_20", "value": hr20},
            {"metric": "hit_rate_top_30", "value": hr30},
        ]
        diag_status = "computed"
        by_region = defaultdict(set)
        for r in strong_mod:
            by_region[r.get("region", "")].add(r["patch_id"])
        region_rows = [{
            "region": reg, "n_event_linked_patches": len(pids),
            "score_v6_mean": round(statistics.mean([val_by_patch[p] for p in pids if p in val_by_patch]), 4) if any(p in val_by_patch for p in pids) else 0.0,
            "review_only": "true",
        } for reg, pids in sorted(by_region.items())]
    else:
        diag_status = "not_enough_observed_events"
        diag_rows = [{"metric": "status", "value": "not_enough_observed_events"},
                     {"metric": "strong_or_moderate_links", "value": len(strong_mod)},
                     {"metric": "min_required", "value": MIN_LINKS}]
        region_rows = [{"region": "n/a", "n_event_linked_patches": 0, "score_v6_mean": 0.0, "review_only": "true"}]
        hr10 = hr20 = hr30 = 0.0
        mean_v = median_v = 0.0

    for r in diag_rows:
        r["review_only"] = "true"
    write_csv(DIAG_CSV, diag_rows, ["metric", "value", "review_only"])
    write_csv(DIAG_REGION, region_rows, ["region", "n_event_linked_patches", "score_v6_mean", "review_only"])
    write_json(DIAG_SUMMARY, {
        "artifact": "SUSC-13C event-score diagnostics", "status": diag_status,
        "strong_or_moderate_links": len(strong_mod), "hit_rate_top_10": hr10,
        "hit_rate_top_20": hr20, "hit_rate_top_30": hr30,
        "score_v6_mean_event_links": mean_v, "score_v6_median_event_links": median_v,
        "score_v7_created": False, "can_be_ground_truth": False,
        "allowed_for_training": False, "review_only": True,
    })

    # --- readiness ---
    src = Counter(e.get("source_institution", "") for e in pref if e["evidence_level"] in OBSERVED)
    single_dominant = bool(src) and max(src.values()) >= 0.8 * sum(src.values()) and sum(src.values()) > 0
    ready_12a = observed >= 10 and strong_mod_links >= 10 and with_date >= 10
    ready_12b = strong_mod_links >= 10 and len(regions_covered) >= 1
    ready_12c = strong_mod_links >= 20 and len(regions_covered) >= 2
    ready_v7 = (strong_mod_links >= 20 and with_geom >= 20 and with_date >= 20 and
                len(regions_covered) >= 2 and not single_dominant)
    rd = [
        ("strong_events_count", strong_events), ("moderate_events_count", moderate_events),
        ("events_with_date_count", with_date), ("events_with_geometry_count", with_geom),
        ("strong_patch_links_count", strong_links), ("moderate_patch_links_count", moderate_links),
        ("regions_covered", "|".join(regions_covered) or "none"),
        ("score_v6_mean_event_links", mean_v),
        ("hit_rate_top_10", hr10), ("hit_rate_top_20", hr20), ("hit_rate_top_30", hr30),
        ("ready_for_12a_temporal", str(ready_12a).lower()),
        ("ready_for_12b_feature_contrast", str(ready_12b).lower()),
        ("ready_for_12c_proxy_calibration", str(ready_12c).lower()),
        ("ready_for_score_v7", str(ready_v7).lower()),
    ]
    write_csv(READINESS_CSV, [{"metric": k, "value": v, "review_only": "true"} for k, v in rd],
              ["metric", "value", "review_only"])

    def yn(b):
        return "PRONTO" if b else "BLOQUEADO"

    write_markdown(READINESS_MD, f"""# SUSC-13C - Prontidao observacional (aquisicao live)

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## Contagens (registros preferenciais)
- Eventos fortes: **{strong_events}** | moderados: **{moderate_events}** | observados: **{observed}**
- Com data/periodo: **{with_date}** | com geometria: **{with_geom}**
- Links fortes: **{strong_links}** | moderados: **{moderate_links}** | fortes+moderados: **{strong_mod_links}**
- Regioes cobertas: **{", ".join(regions_covered) or "nenhuma"}**

## Diagnostico score v6 x eventos
- Status: **{diag_status}** | hit@10={hr10} hit@20={hr20} hit@30={hr30} | media score v6 (links)={mean_v}

## Prontidao
- SUSC-12A (temporal): **{yn(ready_12a)}**
- SUSC-12B (contraste de features): **{yn(ready_12b)}**
- SUSC-12C (calibracao de proxy): **{yn(ready_12c)}**
- Score v7: **{yn(ready_v7)}**

Mesmo se PRONTO, o SUSC-13C **nao cria score v7** e nao altera a matriz SUSC-03.
""")

    _write_final_report(pref, links, diag_status, hr10, hr20, hr30, mean_v,
                        ready_12a, ready_12b, ready_12c, ready_v7,
                        strong_events, moderate_events, with_date, with_geom,
                        strong_links, moderate_links)

    print(f"observed: {observed} (strong {strong_events}/moderate {moderate_events}) | strong-mod links: {strong_mod_links}")
    print(f"diag: {diag_status} | ready 12a={ready_12a} 12b={ready_12b} 12c={ready_12c} v7={ready_v7}")
    print("review-only. Score v7 NOT created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
