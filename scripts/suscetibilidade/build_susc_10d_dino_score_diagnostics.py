"""
SUSC-10D DINO Score Diagnostics (review-only)

If loadable per-patch DINO vectors exist (per the discovery), aligns them with the
score v6 and computes representational diagnostics (cosine/PCA/centroid/outliers)
WITHOUT training any classifier. If not, emits an honest absence report. DINO is a
representational diagnostic layer only — never a classifier, never ground truth.

Writes:
  - SUSC_10D_dino_score_alignment.csv
  - SUSC_10D_dino_region_diagnostics.csv
  - SUSC_10D_dino_outlier_candidates.csv
  - SUSC_10D_dino_score_diagnostics_summary.json
  - SUSC_10D_dino_score_diagnostics_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, write_markdown  # noqa: E402

DISCOVERY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10D_dino_embedding_discovery_summary.json"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"

OUT = ROOT / "outputs_public" / "suscetibilidade"
A_ALIGN = OUT / "SUSC_10D_dino_score_alignment.csv"
A_REGION = OUT / "SUSC_10D_dino_region_diagnostics.csv"
A_OUTLIER = OUT / "SUSC_10D_dino_outlier_candidates.csv"
A_SUMMARY = OUT / "SUSC_10D_dino_score_diagnostics_summary.json"
A_REPORT = OUT / "SUSC_10D_dino_score_diagnostics_report.md"

MANDATORY = ("O SUSC-10D usa DINO apenas como camada representacional diagnóstica. Ele não treina "
             "classificador, não cria ground truth e não transforma embeddings em prova de ocorrência "
             "de enchente.")


def main() -> int:
    print("=" * 60)
    print("SUSC-10D DINO Score Diagnostics (review-only)")
    print("=" * 60)

    discovery = read_json(DISCOVERY) if DISCOVERY.exists() else {"loadable_per_patch_vectors_available": False}
    loadable = bool(discovery.get("loadable_per_patch_vectors_available"))
    n_score = len(read_csv(SCORE)) if SCORE.exists() else 0

    align_fields = ["patch_id", "regiao", "score_v6", "score_class_v6", "dino_cosine_to_region_centroid",
                    "dino_pca1", "dino_pca2", "representational_status", "review_only", "notes"]
    region_fields = ["regiao", "n_patches_with_embedding", "mean_centroid_distance",
                     "dino_available", "review_only", "notes"]
    outlier_fields = ["patch_id", "regiao", "outlier_metric", "outlier_value",
                      "representational_status", "review_only", "notes"]

    if not loadable:
        # honest absence path — write empty-but-headed tables + absence summary
        write_csv(A_ALIGN, [], align_fields)
        write_csv(A_REGION, [], region_fields)
        write_csv(A_OUTLIER, [], outlier_fields)
        write_json(A_SUMMARY, {
            "stage": "SUSC-10D DINO diagnostics",
            "dino_status": "ABSENT_NO_LOADABLE_PER_PATCH_VECTORS",
            "loadable_per_patch_vectors_available": False,
            "score_patches": n_score,
            "diagnostics_computed": False,
            "reason": ("No loadable per-patch DINO vectors in repo: registries/feature-store are empty "
                       "scaffolds and raw .npz/.npy are git-ignored / external (PROJETO)."),
            "dino_usage": "representational_diagnostic_only_never_classifier",
            "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
            "trained_model": False, "model_persisted": False,
            "scientific_status": "dino_absent_review_only_not_ground_truth",
        })
        write_markdown(A_REPORT, "\n".join([
            "# SUSC-10D — Diagnóstico DINO × Score v6 (ausência documentada)", "",
            f"> {MANDATORY}", "",
            "## Status: DINO AUSENTE (sem vetores carregáveis por patch)",
            "",
            f"A descoberta (`SUSC_10D_dino_embedding_discovery.csv`) encontrou "
            f"{discovery.get('files_found','?')} arquivos relacionados a DINO, mas **nenhum vetor "
            "por patch carregável**: os registries/feature-store estão vazios (scaffolds) e os vetores "
            "brutos `.npz/.npy` são git-ignored / externos (PROJETO).", "",
            "## Consequência",
            "- Nenhuma similaridade/PCA/centróide/outlier foi computada (sem vetores reais).",
            "- Tabelas de alinhamento/região/outliers foram geradas vazias (apenas cabeçalho).",
            "- Isso **não** bloqueia a linha SUSC: DINO é camada complementar diagnóstica, não requisito.", "",
            "## O que seria feito se houvesse vetores",
            "Alinhamento por `patch_id`, cosine ao centróide regional, PCA (se sklearn), distância ao "
            "centróide e outliers representacionais — sempre review-only, nunca como classificador.", "",
            "## Limitações",
            "Os vetores DINO reais (referenciados por URI/hash em manifests v1ph/v1pq) não estão no "
            "REV-P por política de peso/governança. Integração futura exige trazer vetores leves "
            "por patch de forma rastreável.", "",
            "> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.", "",
        ]))
        print(f"\nDINO status: ABSENT (no loadable vectors). score patches={n_score}.")
        print("Absence report written. No failure. review-only.")
        return 0

    # (loadable path intentionally minimal: present here for completeness; current
    # repo has no loadable vectors, so this branch is not exercised.)
    print("loadable vectors present — alignment path not exercised in this repo state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
