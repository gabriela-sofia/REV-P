"""SUSC-14B FASE 7/8 - diagnostics, readiness and final report."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"

REGISTRY = MAN / "susc_14b_official_spatial_reference_registry_v1.csv"
DOWNLOADS = MAN / "susc_14b_official_spatial_reference_download_manifest_v1.csv"
FEATURES = DAT / "susc_14b_official_spatial_reference_features_v1.csv"
MATCHED = DAT / "susc_14b_matched_official_occurrences_v1.csv"
LINKAGE = DAT / "susc_14b_event_patch_linkage_v1.csv"
SCORE = DAT / "susc_score_v6_candidate_by_patch_v1.csv"

READINESS = OUT / "SUSC_14B_observational_readiness.csv"
READINESS_MD = OUT / "SUSC_14B_observational_readiness.md"
DIAG = OUT / "SUSC_14B_score_v6_event_diagnostics.csv"
DIAG_SUMMARY = OUT / "SUSC_14B_score_v6_event_diagnostics_summary.json"
REPORT = OUT / "SUSC_14B_official_reference_matching_report.md"

MANDATORY = (
    "O SUSC-14B usa referencias espaciais oficiais para tentar associar ocorrencias oficiais sem coordenada "
    "a patches territoriais. Mesmo quando ha match por endereco, logradouro ou eixo viario, o vinculo "
    "permanece review-only, nao cria ground truth, nao libera treino supervisionado e nao autoriza score v7 automatico."
)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows(path):
    return read_csv(path) if path.exists() else []


def _mean(values):
    return round(statistics.mean(values), 4) if values else None


def _hit(top, event_patches):
    return round(sum(1 for p in top if p in event_patches) / len(top), 4) if top else None


def main() -> int:
    print("=" * 60)
    print("SUSC-14B Observational Diagnostics + Report")
    print("=" * 60)
    refs = _rows(REGISTRY)
    downloads = _rows(DOWNLOADS)
    features = _rows(FEATURES)
    matched = _rows(MATCHED)
    links = _rows(LINKAGE)
    scores = _rows(SCORE)

    method_counts = Counter(r.get("match_method", "") for r in matched)
    status_counts = Counter(r.get("georeferencing_status", "") for r in matched)
    feature_counts = Counter(r.get("feature_type", "") for r in features)
    moderate_links = [r for r in links if r.get("linkage_confidence") == "moderate"]
    obs_links = [r for r in links if r.get("can_be_used_for_observational_evaluation") == "true"]
    obs_patches = sorted({r["patch_id"] for r in obs_links if r.get("patch_id") and r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"})
    obs_regions = sorted({r.get("region", "") for r in obs_links})

    score_by_patch, ranked = {}, []
    class_by_patch = {}
    for row in scores:
        patch = row.get("patch_id", "")
        score = _num(row.get("susc_score_v6_candidate"))
        class_by_patch[patch] = row.get("susc_class_v6_candidate", "")
        if score is not None:
            score_by_patch[patch] = score
            ranked.append((score, patch))
    ranked_patches = [p for _, p in sorted(ranked, key=lambda item: -item[0])]
    linked_scores = [score_by_patch[p] for p in obs_patches if p in score_by_patch]
    mean_score = _mean(linked_scores)
    hit10 = _hit(ranked_patches[:10], set(obs_patches))
    hit20 = _hit(ranked_patches[:20], set(obs_patches))
    hit30 = _hit(ranked_patches[:30], set(obs_patches))

    ready_12a = len([r for r in moderate_links if r.get("event_date")]) >= 10
    ready_12b = len(obs_patches) >= 10
    ready_12c = len(obs_patches) >= 20 and len(obs_regions) >= 2
    ready_v7 = ready_12c and mean_score is not None and len(moderate_links) >= 20

    readiness_rows = [
        ("occurrences_input_count", len(matched)),
        ("flood_occurrences_count", len(matched)),
        ("official_reference_features_count", len(features)),
        ("street_segment_reference_count", feature_counts.get("street_segment", 0)),
        ("address_point_reference_count", feature_counts.get("address_point", 0)),
        ("neighborhood_polygon_reference_count", feature_counts.get("neighborhood_polygon", 0)),
        ("exact_matches_count", method_counts.get("exact_street_neighborhood", 0)),
        ("fuzzy_matches_count", method_counts.get("fuzzy_street_neighborhood", 0) + method_counts.get("street_only_unique", 0)),
        ("ambiguous_matches_count", status_counts.get("blocked_ambiguous_address", 0)),
        ("blocked_no_reference_count", status_counts.get("blocked_no_official_reference", 0)),
        ("moderate_patch_links_count", len(moderate_links)),
        ("patches_observational_count", len(obs_patches)),
        ("score_v6_mean_event_links", "null" if mean_score is None else mean_score),
        ("hit_rate_top_10", "null" if hit10 is None else hit10),
        ("hit_rate_top_20", "null" if hit20 is None else hit20),
        ("hit_rate_top_30", "null" if hit30 is None else hit30),
        ("ready_for_12a", str(ready_12a).lower()),
        ("ready_for_12b", str(ready_12b).lower()),
        ("ready_for_12c", str(ready_12c).lower()),
        ("ready_for_score_v7", str(ready_v7).lower()),
        ("score_v7_status", "blocked" if not ready_v7 else "requires_human_review_before_any_future_action"),
    ]
    write_csv(READINESS, [{"metric": k, "value": v, "review_only": "true"} for k, v in readiness_rows],
              ["metric", "value", "review_only"])
    diag_rows = [
        {"metric": "score_v6_mean_event_links", "value": "null" if mean_score is None else mean_score, "review_only": "true"},
        {"metric": "hit_rate_top_10", "value": "null" if hit10 is None else hit10, "review_only": "true"},
        {"metric": "hit_rate_top_20", "value": "null" if hit20 is None else hit20, "review_only": "true"},
        {"metric": "hit_rate_top_30", "value": "null" if hit30 is None else hit30, "review_only": "true"},
        {"metric": "n_observational_patches", "value": len(obs_patches), "review_only": "true"},
    ]
    write_csv(DIAG, diag_rows, ["metric", "value", "review_only"])
    write_json(DIAG_SUMMARY, {
        "artifact": "SUSC-14B score v6 event diagnostics",
        "score_v6_mean_event_links": mean_score,
        "hit_rate_top_10": hit10,
        "hit_rate_top_20": hit20,
        "hit_rate_top_30": hit30,
        "observational_patches": len(obs_patches),
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "can_be_used_as_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    })
    write_markdown(READINESS_MD, f"""# SUSC-14B - prontidao observacional

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

- Ocorrencias reprocessadas: **{len(matched)}**
- Features oficiais parseadas: **{len(features)}** ({dict(feature_counts)})
- Matches exatos: **{method_counts.get('exact_street_neighborhood', 0)}**
- Matches fuzzy/unicos: **{method_counts.get('fuzzy_street_neighborhood', 0) + method_counts.get('street_only_unique', 0)}**
- Bloqueios por ambiguidade: **{status_counts.get('blocked_ambiguous_address', 0)}**
- Bloqueios por falta de referencia: **{status_counts.get('blocked_no_official_reference', 0)}**
- Links moderados: **{len(moderate_links)}**
- Patches observacionais: **{len(obs_patches)}**

Readiness: 12A={ready_12a}; 12B={ready_12b}; 12C={ready_12c}; score_v7={ready_v7}.
Score v7 nao foi criado.
""")
    _write_report(refs, downloads, features, matched, links, feature_counts, method_counts, status_counts,
                  len(moderate_links), len(obs_patches), mean_score, hit10, hit20, hit30,
                  ready_12a, ready_12b, ready_12c, ready_v7)
    print(f"matches exact={method_counts.get('exact_street_neighborhood', 0)} fuzzy={method_counts.get('fuzzy_street_neighborhood', 0) + method_counts.get('street_only_unique', 0)} blocked={status_counts.get('blocked_no_official_reference', 0)}")
    print(f"moderate links={len(moderate_links)} patches={len(obs_patches)} ready_v7={ready_v7}")
    print("review-only. No score v7.")
    return 0


def _write_report(refs, downloads, features, matched, links, feature_counts, method_counts, status_counts,
                  moderate_links, obs_patches, mean_score, hit10, hit20, hit30,
                  ready_12a, ready_12b, ready_12c, ready_v7):
    dl_counts = Counter(r.get("download_status", "") for r in downloads)
    write_markdown(REPORT, f"""# SUSC-14B - matching com referencias espaciais oficiais

Status: **review-only** | `allowed_for_training=false` | `can_be_ground_truth=false`

{MANDATORY}

## 1. Gargalo herdado do 14A
O SUSC-14A reprocessou 4.412 ocorrencias oficiais de cheia, mas teve 0 matches
por endereco/logradouro e manteve score v7 bloqueado por falta de vinculos
patch-evento fortes/moderados.

## 2. Referencias oficiais descobertas
Referencias registradas: **{len(refs)}**.

## 3. Referencias baixadas
Manifesto de aquisicao: **{len(downloads)}** linhas. Status: **{dict(dl_counts)}**.

## 4. Features de logradouro/endereco/bairro parseadas
Features oficiais parseadas: **{len(features)}**. Por tipo: **{dict(feature_counts)}**.

## 5. Ocorrencias de cheia reprocessadas
Ocorrencias reprocessadas: **{len(matched)}**.

## 6. Matches exatos
Matches exatos por logradouro + bairro: **{method_counts.get('exact_street_neighborhood', 0)}**.

## 7. Matches fuzzy
Matches fuzzy/unicos: **{method_counts.get('fuzzy_street_neighborhood', 0) + method_counts.get('street_only_unique', 0)}**.

## 8. Bloqueios por ambiguidade
Bloqueios ambiguos: **{status_counts.get('blocked_ambiguous_address', 0)}**.

## 9. Bloqueios por ausencia de referencia
Bloqueios sem referencia oficial: **{status_counts.get('blocked_no_official_reference', 0)}**.

## 10. Links patch-evento
Links totais: **{len(links)}**. Links moderados: **{moderate_links}**. Patches
observacionais: **{obs_patches}**.

## 11. Score v6 x eventos
Media score v6 nos patches observacionais: **{mean_score}**. hit@10={hit10},
hit@20={hit20}, hit@30={hit30}.

## 12. Readiness
12A={ready_12a}; 12B={ready_12b}; 12C={ready_12c}; score_v7={ready_v7}.

## 13. Se score v7 segue bloqueado, por que
O score v7 segue bloqueado quando nao ha pelo menos 20 patches com evidencia
oficial consistente em duas ou mais regioes e diagnostico positivo. O SUSC-14B
nao cria score v7 automaticamente.

## 14. Limitacoes
Street/address match e no maximo moderado review-only. Bairro sozinho continua
weak/contextual e nao vira patch-level. Falhas de portal, endpoints sem dado
direto e ausencia de camada oficial completa permanecem bloqueios cientificos,
nao autorizacoes para inferir coordenadas.
""")


if __name__ == "__main__":
    raise SystemExit(main())
