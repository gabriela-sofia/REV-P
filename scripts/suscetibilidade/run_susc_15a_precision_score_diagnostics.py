"""SUSC-15A FASE 10/11 - precision score diagnostics and final report."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"

GEOCODED = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
CATALOG = DAT / "susc_15a_calibration_eligible_event_catalog_v1.csv"
LINKAGE = DAT / "susc_15a_precision_event_patch_linkage_v1.csv"
SCORE = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
REGISTRY = MAN / "susc_15a_precision_reference_source_registry_v1.csv"
DOWNLOADS = MAN / "susc_15a_precision_reference_download_manifest_v1.csv"

DIAG = OUT / "SUSC_15A_precision_score_event_diagnostics.csv"
DIAG_REGION = OUT / "SUSC_15A_precision_score_event_diagnostics_by_region.csv"
SUMMARY = OUT / "SUSC_15A_precision_score_event_diagnostics_summary.json"
READINESS = OUT / "SUSC_15A_precision_observational_readiness.csv"
READINESS_MD = OUT / "SUSC_15A_precision_observational_readiness.md"
REPORT = OUT / "SUSC_15A_PRECISION_observed_event_coordinate_rescue_report.md"

MANDATORY = (
    "O SUSC-15A-PRECISION tenta obter evidencia espacial precisa para eventos observados usando apenas geometrias "
    "diretas, referencias oficiais de enderecamento/intersecao/linear referencing e footprints rastreaveis. A etapa "
    "nao usa bairro como coordenada, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico."
)


def _rows(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(values):
    return round(statistics.mean(values), 4) if values else None


def _median(values):
    return round(statistics.median(values), 4) if values else None


def _hit(top: list[str], patches: set[str]):
    return round(sum(1 for p in top if p in patches) / len(top), 4) if top else None


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Precision Score Diagnostics + Report")
    print("=" * 60)
    geocoded = _rows(GEOCODED)
    catalog = _rows(CATALOG)
    links = _rows(LINKAGE)
    scores = _rows(SCORE)
    sources = _rows(REGISTRY)
    downloads = _rows(DOWNLOADS)
    eval_links = [r for r in links if r.get("can_be_used_for_observational_evaluation") == "true"]
    patches = sorted({r["patch_id"] for r in eval_links if r.get("patch_id") and r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"})
    regions = sorted({r.get("region", "") for r in eval_links if r.get("region")})
    score_by_patch, class_by_patch, ranked = {}, {}, []
    for row in scores:
        score = _num(row.get("susc_score_v6_candidate"))
        patch = row.get("patch_id", "")
        class_by_patch[patch] = row.get("susc_class_v6_candidate", "")
        if score is not None:
            score_by_patch[patch] = score
            ranked.append((score, patch))
    ranked_patches = [p for _, p in sorted(ranked, key=lambda item: -item[0])]
    linked_scores = [score_by_patch[p] for p in patches if p in score_by_patch]
    mean_score = _mean(linked_scores)
    median_score = _median(linked_scores)
    class_dist = Counter(class_by_patch.get(p, "") for p in patches if p in class_by_patch)
    hit10 = _hit(ranked_patches[:10], set(patches))
    hit20 = _hit(ranked_patches[:20], set(patches))
    hit30 = _hit(ranked_patches[:30], set(patches))
    low_cases = sum(1 for p in patches if score_by_patch.get(p, 1) < 0.33)
    high_cases = sum(1 for p in patches if score_by_patch.get(p, 0) >= 0.67)
    ready_12a = len([r for r in eval_links if r.get("event_date")]) >= 10
    ready_12b = len(patches) >= 10
    ready_12c = len(patches) >= 20 and len(regions) >= 2
    ready_v7 = False
    method_counts = Counter(r.get("geocoding_method", "") for r in geocoded)
    tier_counts = Counter(r.get("precision_tier", "") for r in geocoded)

    diag_rows = [
        ("precision_events_total", len(geocoded)),
        ("calibration_eligible_events", len(catalog)),
        ("precision_patch_links", len(eval_links)),
        ("unique_patches_observational", len(patches)),
        ("score_v6_mean_event_links", "null" if mean_score is None else mean_score),
        ("score_v6_median_event_links", "null" if median_score is None else median_score),
        ("score_v6_class_distribution_event_links", json.dumps(dict(class_dist), ensure_ascii=False, sort_keys=True)),
        ("hit_rate_top_10", "null" if hit10 is None else hit10),
        ("hit_rate_top_20", "null" if hit20 is None else hit20),
        ("hit_rate_top_30", "null" if hit30 is None else hit30),
        ("low_score_event_cases", low_cases),
        ("high_score_event_cases", high_cases),
    ]
    write_csv(DIAG, [{"metric": k, "value": v, "review_only": "true"} for k, v in diag_rows],
              ["metric", "value", "review_only"])
    by_region = defaultdict(list)
    for link in eval_links:
        by_region[link.get("region", "")].append(link)
    write_csv(DIAG_REGION, [
        {"region": region, "precision_patch_links": len(rows),
         "unique_patches_observational": len({r["patch_id"] for r in rows}),
         "review_only": "true"}
        for region, rows in sorted(by_region.items())
    ], ["region", "precision_patch_links", "unique_patches_observational", "review_only"])
    readiness_rows = [
        ("ready_for_12a", str(ready_12a).lower()),
        ("ready_for_12b", str(ready_12b).lower()),
        ("ready_for_12c", str(ready_12c).lower()),
        ("ready_for_score_v7", str(ready_v7).lower()),
        ("score_v7_status", "blocked_review_only_no_automatic_score_v7"),
        ("precision_events_total", len(geocoded)),
        ("calibration_eligible_events", len(catalog)),
        ("precision_patch_links", len(eval_links)),
        ("unique_patches_observational", len(patches)),
    ]
    write_csv(READINESS, [{"metric": k, "value": v, "review_only": "true"} for k, v in readiness_rows],
              ["metric", "value", "review_only"])
    write_json(SUMMARY, {
        "artifact": "SUSC-15A precision score event diagnostics",
        "precision_events_total": len(geocoded),
        "calibration_eligible_events": len(catalog),
        "precision_patch_links": len(eval_links),
        "unique_patches_observational": len(patches),
        "score_v6_mean_event_links": mean_score,
        "score_v6_median_event_links": median_score,
        "score_v6_class_distribution_event_links": dict(class_dist),
        "hit_rate_top_10": hit10,
        "hit_rate_top_20": hit20,
        "hit_rate_top_30": hit30,
        "ready_for_score_v7": ready_v7,
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "can_be_used_as_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    })
    write_markdown(READINESS_MD, f"""# SUSC-15A-PRECISION - prontidao observacional

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

- Eventos de precisao avaliados: **{len(geocoded)}**
- Eventos elegiveis para calibracao: **{len(catalog)}**
- Links precisos patch-evento: **{len(eval_links)}**
- Patches observacionais precisos: **{len(patches)}**

Readiness: 12A={ready_12a}; 12B={ready_12b}; 12C={ready_12c}; score_v7={ready_v7}.
Score v7 nao foi criado.
""")
    write_markdown(REPORT, f"""# SUSC-15A-PRECISION - resgate de coordenadas precisas de eventos

Status: **review-only** | `allowed_for_training=false` | `can_be_ground_truth=false`

{MANDATORY}

## 1. Por que bairro-only nao e suficiente
Bairro-only nao determina coordenada nem footprint de inundacao. No SUSC-15A,
T6 e sempre excluido de calibracao.

## 2. Por que SUSC-14C chegou ao limite
O SUSC-14C ampliou referencias oficiais, mas manteve 0 links moderados
evento-patch. A barreira principal e a ausencia de ponto, poligono, numero,
intersecao ou linear referencing suficiente.

## 3. Estrategia de precisao espacial
A esteira prioriza T0/T1 geometria direta, T2 endereco/lote oficial, T3
intersecao oficial e T4 linear referencing controlado. T5 e candidato fraco.

## 4. Fontes oficiais de precisao descobertas
Fontes registradas: **{len(sources)}**.

## 5. Referencias baixadas
Manifesto de aquisicao: **{len(downloads)}** linhas.

## 6. Eventos com coordenada/poligono direto
Eventos T0/T1: **{tier_counts.get('T0_official_event_polygon', 0) + tier_counts.get('T1_official_event_point', 0)}**.

## 7. Eventos geocodificados por endereco oficial
Eventos T2: **{tier_counts.get('T2_official_address_point_or_parcel', 0)}**.

## 8. Eventos por intersecao oficial
Eventos T3: **{tier_counts.get('T3_official_intersection_point', 0)}**.

## 9. Eventos por linear referencing
Eventos T4: **{tier_counts.get('T4_official_house_number_linear_reference', 0)}**.

## 10. Eventos excluidos por bairro-only
Eventos T6: **{tier_counts.get('T6_neighborhood_context_only', 0)}**.

## 11. Eventos elegiveis para calibracao
Eventos elegiveis: **{len(catalog)}**.

## 12. Linkage patch-evento preciso
Links precisos de avaliacao observacional: **{len(eval_links)}**.

## 13. Score v6 x eventos precisos
Media score v6: **{mean_score}**; mediana: **{median_score}**; hit@10={hit10},
hit@20={hit20}, hit@30={hit30}.

## 14. Readiness
12A={ready_12a}; 12B={ready_12b}; 12C={ready_12c}; score_v7={ready_v7}.

## 15. Se score v7 segue bloqueado, por que
Score v7 segue bloqueado porque a etapa nao cria alvo operacional, nao cria
ground truth e so permitiria futura calibracao humana se houver massa suficiente
de eventos precisos.

## 16. Limitacoes
Nao houve uso de Google Maps, Nominatim como evidencia, centroide de bairro ou
municipio, data inventada, coordenada inventada ou raster pesado.

## 17. Proximo marco
Executar aquisicao/manual review de fontes oficiais que contenham ponto,
poligono, lote, intersecao ou faixa numerica verificavel e entao reprocessar
T0-T4.
""")
    print(f"eligible={len(catalog)} links={len(eval_links)} patches={len(patches)} ready_v7={ready_v7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
