"""REV-P v1r8b (SUSC-21b) -- Re-execucao do refinamento de evidencia do v1r7"""
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
