"""SUSC-14C FASE 7/8 - diagnostics, readiness and final report."""

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

DAT = ROOT / "datasets" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

REGISTRY_14B = MAN / "susc_14b_official_spatial_reference_registry_v1.csv"
DOWNLOADS = MAN / "susc_14c_additional_reference_download_manifest_v1.csv"
FEATURES_ADD = DAT / "susc_14c_additional_official_reference_features_v1.csv"
FEATURES_UNION = DAT / "susc_14c_official_spatial_reference_features_union_v1.csv"
MATCHED = DAT / "susc_14c_matched_official_occurrences_v1.csv"
RESOLVED = DAT / "susc_14c_resolved_ambiguous_occurrences_v1.csv"
LINKAGE = DAT / "susc_14c_event_patch_linkage_v1.csv"
SCORE = DAT / "susc_score_v6_candidate_by_patch_v1.csv"

READINESS = OUT / "SUSC_14C_observational_readiness.csv"
READINESS_MD = OUT / "SUSC_14C_observational_readiness.md"
DIAG = OUT / "SUSC_14C_score_v6_event_diagnostics.csv"
DIAG_SUMMARY = OUT / "SUSC_14C_score_v6_event_diagnostics_summary.json"
REPORT = OUT / "SUSC_14C_official_reference_expansion_report.md"

MANDATORY = (
    "O SUSC-14C expande a aquisicao de referencias oficiais e reprocessa ocorrencias de cheia para aumentar "
    "vinculos observacionais evento-patch. Todos os vinculos permanecem review-only, sem ground truth, sem "
    "treino supervisionado e sem score v7 automatico."
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


def _hit(top, event_patches):
    return round(sum(1 for p in top if p in event_patches) / len(top), 4) if top else None


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Observational Diagnostics + Report")
    print("=" * 60)
    refs = _rows(REGISTRY_14B)
    downloads = _rows(DOWNLOADS)
    features_add = _rows(FEATURES_ADD)
    features = _rows(FEATURES_UNION)
    matched = _rows(MATCHED)
    resolved = _rows(RESOLVED)
    links = _rows(LINKAGE)
    scores = _rows(SCORE)

    method_counts = Counter(r.get("match_method", "") for r in matched)
    status_counts = Counter(r.get("georeferencing_status", "") for r in resolved)
    input_status_counts = Counter(r.get("georeferencing_status", "") for r in matched)
    feature_counts = Counter(r.get("feature_type", "") for r in features)
    dl_counts = Counter(r.get("download_status", "") for r in downloads)
    resolved_count = sum(1 for r in resolved if r.get("match_method") == "resolved_ambiguous_strict_official_candidate")
    remaining_amb = status_counts.get("blocked_ambiguous_address", 0)
    moderate_links = [r for r in links if r.get("linkage_confidence") == "moderate"]
    obs_links = [r for r in links if r.get("can_be_used_for_observational_evaluation") == "true"]
    obs_patches = sorted({r["patch_id"] for r in obs_links if r.get("patch_id") and r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"})
    obs_regions = sorted({r.get("region", "") for r in obs_links})

    score_by_patch, ranked = {}, []
    for row in scores:
        patch = row.get("patch_id", "")
        score = _num(row.get("susc_score_v6_candidate"))
        if score is not None:
            score_by_patch[patch] = score
            ranked.append((score, patch))
    ranked_patches = [p for _, p in sorted(ranked, key=lambda item: -item[0])]
    linked_scores = [score_by_patch[p] for p in obs_patches if p in score_by_patch]
    mean_score = _mean(linked_scores)
    median_score = _median(linked_scores)
    hit10 = _hit(ranked_patches[:10], set(obs_patches))
    hit20 = _hit(ranked_patches[:20], set(obs_patches))
    hit30 = _hit(ranked_patches[:30], set(obs_patches))

    ready_12a = len([r for r in moderate_links if r.get("event_date")]) >= 10
    ready_12b = len(obs_patches) >= 10
    ready_12c = len(obs_patches) >= 20 and len(obs_regions) >= 2
    ready_v7 = False

    readiness_rows = [
        ("occurrences_input_count", len(matched)),
        ("flood_occurrences_count", len(matched)),
        ("reference_features_total", len(features)),
        ("new_reference_features_count", len(features_add)),
        ("exact_matches_count", method_counts.get("exact_street_neighborhood", 0)),
        ("fuzzy_matches_count", method_counts.get("fuzzy_street_neighborhood", 0)),
        ("resolved_ambiguous_count", resolved_count),
        ("remaining_ambiguous_count", remaining_amb),
        ("blocked_no_reference_count", status_counts.get("blocked_no_official_reference", 0)),
        ("moderate_patch_links_count", len(moderate_links)),
        ("patches_observational_count", len(obs_patches)),
        ("score_v6_mean_event_links", "null" if mean_score is None else mean_score),
        ("score_v6_median_event_links", "null" if median_score is None else median_score),
        ("hit_rate_top_10", "null" if hit10 is None else hit10),
        ("hit_rate_top_20", "null" if hit20 is None else hit20),
        ("hit_rate_top_30", "null" if hit30 is None else hit30),
        ("ready_for_12a", str(ready_12a).lower()),
        ("ready_for_12b", str(ready_12b).lower()),
        ("ready_for_12c", str(ready_12c).lower()),
        ("ready_for_score_v7", str(ready_v7).lower()),
        ("score_v7_status", "blocked_review_only_no_automatic_score_v7"),
    ]
    write_csv(READINESS, [{"metric": k, "value": v, "review_only": "true"} for k, v in readiness_rows],
              ["metric", "value", "review_only"])
    diag_rows = [
        {"metric": "score_v6_mean_event_links", "value": "null" if mean_score is None else mean_score, "review_only": "true"},
        {"metric": "score_v6_median_event_links", "value": "null" if median_score is None else median_score, "review_only": "true"},
        {"metric": "hit_rate_top_10", "value": "null" if hit10 is None else hit10, "review_only": "true"},
        {"metric": "hit_rate_top_20", "value": "null" if hit20 is None else hit20, "review_only": "true"},
        {"metric": "hit_rate_top_30", "value": "null" if hit30 is None else hit30, "review_only": "true"},
        {"metric": "n_observational_patches", "value": len(obs_patches), "review_only": "true"},
    ]
    write_csv(DIAG, diag_rows, ["metric", "value", "review_only"])
    write_json(DIAG_SUMMARY, {
        "artifact": "SUSC-14C score v6 event diagnostics",
        "score_v6_mean_event_links": mean_score,
        "score_v6_median_event_links": median_score,
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
    write_markdown(READINESS_MD, f"""# SUSC-14C - prontidao observacional

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

- Ocorrencias reprocessadas: **{len(matched)}**
- Features oficiais na uniao: **{len(features)}** ({dict(feature_counts)})
- Novas features parseadas no 14C: **{len(features_add)}**
- Matches exatos: **{method_counts.get('exact_street_neighborhood', 0)}**
- Matches fuzzy: **{method_counts.get('fuzzy_street_neighborhood', 0)}**
- Ambiguos resolvidos por criterio estrito: **{resolved_count}**
- Ambiguos remanescentes: **{remaining_amb}**
- Bloqueios por falta de referencia: **{status_counts.get('blocked_no_official_reference', 0)}**
- Links moderados: **{len(moderate_links)}**
- Patches observacionais: **{len(obs_patches)}**

Readiness: 12A={ready_12a}; 12B={ready_12b}; 12C={ready_12c}; score_v7={ready_v7}.
Score v7 nao foi criado.
""")
    write_markdown(REPORT, f"""# SUSC-14C - expansao oficial de referencias e reprocessamento observacional

Status: **review-only** | `allowed_for_training=false` | `can_be_ground_truth=false`

{MANDATORY}

## 1. Escopo
O SUSC-14C parte dos bloqueios do SUSC-14B, audita falhas de matching,
executa lote adicional de aquisicao oficial e recompila a uniao de referencias
espaciais rastreaveis.

## 2. Referencias oficiais
Referencias herdadas do registro 14B: **{len(refs)}**. Manifesto adicional
14C: **{len(downloads)}** linhas. Status de aquisicao: **{dict(dl_counts)}**.

## 3. Features parseadas
Novas features 14C parseadas: **{len(features_add)}**. Features totais na
uniao: **{len(features)}**. Por tipo: **{dict(feature_counts)}**.

## 4. Reprocessamento de ocorrencias
Ocorrencias reprocessadas: **{len(matched)}**. Status inicial: **{dict(input_status_counts)}**.

## 5. Matching
Matches exatos: **{method_counts.get('exact_street_neighborhood', 0)}**.
Matches fuzzy: **{method_counts.get('fuzzy_street_neighborhood', 0)}**.

## 6. Ambiguidade
Ambiguos resolvidos por criterio estrito: **{resolved_count}**. Ambiguos
remanescentes: **{remaining_amb}**. Ambiguidade remanescente continua bloqueada
para patch-level e para avaliacao observacional.

## 7. Bloqueios
Bloqueios sem referencia oficial: **{status_counts.get('blocked_no_official_reference', 0)}**.
Nenhum bloqueio foi preenchido por Google Maps, geocoding generico,
centroide municipal ou bairro como patch-level.

## 8. Links evento-patch
Links totais: **{len(links)}**. Links moderados: **{len(moderate_links)}**.
Patches observacionais: **{len(obs_patches)}**.

## 9. Score v6 x eventos
Media score v6 nos patches observacionais: **{mean_score}**. Mediana:
**{median_score}**. hit@10={hit10}, hit@20={hit20}, hit@30={hit30}.

## 10. Readiness
12A={ready_12a}; 12B={ready_12b}; 12C={ready_12c}; score_v7={ready_v7}.

## 11. Por que score v7 segue bloqueado
O SUSC-14C nao cria score v7 automaticamente. Mesmo quando ha vinculo oficial
por endereco/logradouro, o vinculo e moderado, revisavel e insuficiente para
ground truth ou treino supervisionado.

## 12. Limitacoes
Falhas de portal, endpoints sem arquivos parseaveis, ausencia de geometria
oficial e ambiguidade entre logradouros continuam como bloqueios. Bairro sozinho
permanece contexto fraco e nunca resolve patch-level.
""")
    print(f"moderate links={len(moderate_links)} patches={len(obs_patches)} ready_v7={ready_v7}")
    print("review-only. No score v7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
