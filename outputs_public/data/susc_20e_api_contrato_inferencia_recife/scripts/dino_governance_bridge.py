"""SUSC-20E -- ponte entre a API do contrato de inferência e a camada de
governança visual DINOv2 (v2fg).

A ponte é fina de propósito: resolve QUAL embedding real representa a
requisição e delega a decisão para `Dinov2GovernanceEngine`. Ela nunca toca
em `score.value`, `score.confidence_interval`, `features_used` ou nos
coeficientes do Firth -- o motor físico (SUSC-20D) não sabe que esta camada
existe.

Ordem de resolução do embedding de consulta:
  1. `request.visual_patch_id` explícito (patch do corpus v2fg);
  2. índice de bboxes dos patches Sentinel, quando
     `REVP_SUSC20D_SENTINEL_DIR` estiver configurado (mesmo índice que já
     alimenta `evidence.dino_patch_id`);
  3. nada -> estado explícito `no_visual_evidence`.

Nenhum caminho fabrica embedding, medoid ou similaridade.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "dino"))

from revp_v2fg_dinov2_governance_engine import (  # noqa: E402
    Dinov2GovernanceEngine, STATUS_NO_VISUAL_EVIDENCE,
)

_engine: Dinov2GovernanceEngine | None = None


def get_engine(refresh: bool = False) -> Dinov2GovernanceEngine:
    """Engine singleton (carregada uma vez no startup da API)."""
    global _engine
    if _engine is None or refresh:
        _engine = Dinov2GovernanceEngine()
    return _engine


def resolve_patch_id(visual_patch_id: str | None, lat: float | None, lon: float | None,
                     dino_bboxes: dict | None = None) -> str | None:
    """Patch do corpus que representa visualmente esta requisição, ou None."""
    if visual_patch_id:
        return visual_patch_id.strip().upper()
    if not dino_bboxes or lat is None or lon is None:
        return None
    for pid, (xmin, ymin, xmax, ymax) in dino_bboxes.items():
        if xmin <= lon <= xmax and ymin <= lat <= ymax:
            return f"REC_{pid}".upper()
    return None


def evaluate(requested_region: str | None, visual_patch_id: str | None = None,
             lat: float | None = None, lon: float | None = None,
             dino_bboxes: dict | None = None,
             engine: Dinov2GovernanceEngine | None = None) -> dict[str, Any]:
    """Bloco de governança pronto para o contrato. Nunca levanta exceção e
    nunca devolve None -- toda requisição sai com um estado auditável."""
    eng = engine if engine is not None else get_engine()
    patch_id = resolve_patch_id(visual_patch_id, lat, lon, dino_bboxes)
    try:
        result = eng.evaluate(patch_id=patch_id, requested_region=requested_region)
    except Exception as exc:  # a governança nunca derruba a inferência física
        return {
            "status": STATUS_NO_VISUAL_EVIDENCE,
            "query_patch_id": patch_id,
            "cosine_similarity": None,
            "nearest_medoid_patch_id": None,
            "suggested_region": None,
            "requested_region": requested_region,
            "territorial_match": "not_applicable",
            "ood_threshold": None,
            "ranking": [],
            "audit": {},
            "notes": [f"falha_interna_da_governanca_dinov2: {type(exc).__name__}: {exc}"],
        }
    return result


__all__ = ["get_engine", "resolve_patch_id", "evaluate"]
