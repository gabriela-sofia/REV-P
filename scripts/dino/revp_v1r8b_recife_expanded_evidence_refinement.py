"""REV-P v1r8b (SUSC-21b) -- Re-execucao do refinamento de evidencia do v1r7
sobre a cobertura ampliada de Recife (23 -> 52 patches com embedding real).

Este script NAO reimplementa o metodo. Ele funde os dois arquivos de embedding
ja versionados (v1r3 + a expansao v1r8) num insumo unico e chama o
`run()` do proprio `revp_v1r7_dino_evidence_refinement_recife.py`, com os
mesmos limiares pre-registrados. Manter o codigo do metodo intocado e o que
torna o v1r7 e o v1r8b comparaveis lado a lado: a unica coisa que muda entre
as duas rodadas e o numero de patches com embedding.

PRE-REGISTRO (identico ao v1r7, fixado antes desta rodada)
-----------------------------------------------------------
  * phi_H = 0.75, phi_L = 0.25, suporte minimo = 2 pontos v12 por patch.
  * Unidade de analise = patch. Score assinado
    mean_cos(t, sementes positivas) - mean_cos(t, sementes negativas).
  * Bateria de QA identica: permutacao (exata quando o numero de particoes
    permitir, Monte Carlo com N >= 20000 sempre), confundidores (adjacencia
    geografica, proveniencia do patch, n de pontos), leave-one-seed-out,
    redundancia com as 6 features causais sob Bonferroni.
  * Criterio de utilidade identico: so emite fila ORDENADA se a permutacao
    der p < 0.05 E nenhum confundidor for significativo. Resultado nulo com
    mais poder e confirmacao, nao falha -- e nao autoriza mexer em limiar.

Nao cria label. Nao treina classificador. Nao baixa nem gera imagem.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

from revp_v1qg_v1qm_smoke_embedding_common import DATASETS, ROOT, _p

V1R7 = Path(__file__).with_name("revp_v1r7_dino_evidence_refinement_recife.py")
IN_EMB_BASE = _p("REVP_V1R8B_IN_EMB_BASE", DATASETS / "dino_recife_sedec_all_embeddings_v1r3.csv")
IN_EMB_EXPANSION = _p("REVP_V1R8B_IN_EMB_EXPANSION",
                      DATASETS / "dino_recife_coverage_expansion_embeddings_v1r8.csv")
# The merged feature store is a derived intermediate (it duplicates vectors
# that are already versioned in the two inputs), so it lands in local_runs.
MERGED = _p("REVP_V1R8B_MERGED",
            ROOT / "local_runs" / "susc-21b-recife-coverage-expansion"
            / "merged_recife_embeddings_v1r8.csv")


def _merge() -> int:
    rows: list[dict[str, str]] = []
    for path in (IN_EMB_BASE, IN_EMB_EXPANSION):
        with Path(path).open(encoding="utf-8", newline="") as fh:
            rows.extend(csv.DictReader(fh))
    seen: dict[str, dict[str, str]] = {}
    for r in rows:
        seen.setdefault(r["patch_id"], r)
    merged = [seen[k] for k in sorted(seen)]
    emb_cols = [f"embedding_{i:03d}" for i in range(768)]
    fields = ["patch_id", "alias", "region", "relative_path", "model_name",
              "model_path_hash", "embedding_dim", "l2_normalized", "vector_norm",
              "dino_allowed_use", "review_only", "can_create_label",
              "can_train_model", "target_created"] + emb_cols
    Path(MERGED).parent.mkdir(parents=True, exist_ok=True)
    with Path(MERGED).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow(r)
    return len(merged)


def run() -> None:
    n = _merge()
    os.environ["REVP_V1R7_IN_EMB"] = str(MERGED)
    os.environ.setdefault("REVP_V1R7_OUT_SCORES",
                          str(DATASETS / "dino_recife_expanded_refinement_scores_v1r8.csv"))
    os.environ.setdefault("REVP_V1R7_OUT_QUEUE",
                          str(DATASETS / "dino_recife_expanded_refinement_review_queue_v1r8.csv"))
    os.environ.setdefault("REVP_V1R7_OUT_QA",
                          str(DATASETS / "dino_recife_expanded_refinement_qa_v1r8.csv"))
    os.environ.setdefault("REVP_V1R7_OUT_BOUNDARY",
                          str(DATASETS / "dino_evidence_refinement_boundary_matrix_v1r8.csv"))
    os.environ.setdefault("REVP_V1R7_SCH_SCORES",
                          str(ROOT / "datasets" / "schemas"
                              / "dino_recife_expanded_refinement_scores_v1r8_schema.csv"))
    os.environ.setdefault("REVP_V1R7_DOC",
                          str(ROOT / "docs" / "metodologia_cientifica"
                              / "revp_v1r8b_dino_expanded_evidence_refinement_recife.md"))

    spec = importlib.util.spec_from_file_location("revp_v1r7_for_v1r8b", V1R7)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(V1R7.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    print(f"[v1r8b] merged_patches={n}")
    mod.run()


if __name__ == "__main__":
    run()
