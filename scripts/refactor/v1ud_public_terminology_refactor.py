"""Mapeamentos historicos mantidos para validar a limpeza terminologica."""

from __future__ import annotations


TERMINOLOGY_MAP = {
    "review_gate": "review_gate",
    "requires_reviewer_confirmation": "requires_review_confirmation",
    "visual assisted review": "visual review",
    "assistida": "programatica",
    "revisao supervisora": "revisao supervisora",
    "autonomous automation": "automacao programatica",
    "Claude-based": "baseado em regras documentadas",
}

FILE_RENAMES = {
    "scripts/dino/revp_v1gg_dino_human_review_gate_package.py": "scripts/dino/revp_v1gg_dino_review_gate_package.py",
    "datasets/human_reference_review_registry.csv": "datasets/review_gate_reference_registry.csv",
    "datasets/schemas/human_reference_review_schema.csv": "datasets/schemas/review_gate_reference_schema.csv",
    "docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md": "docs/metodologia_cientifica/protocolo_c_revisao_supervisora_referencia.md",
    "datasets/dino_human_review_queue.csv": "datasets/dino_review_queue.csv",
    "datasets/dino_visual_assisted_review.csv": "datasets/dino_visual_review.csv",
    "docs/dino_human_review.md": "docs/dino_review.md",
    "scripts/dino/revp_dino_assisted_review.py": "scripts/dino/revp_dino_review.py",
}

PRESERVE_TERMS = ["DINO", "Sentinel", "Protocolo C", "QA", "PCA", "UMAP"]
