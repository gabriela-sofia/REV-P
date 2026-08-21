"""Schema de entrada/saída -- implementa literalmente o rascunho v0 do
contrato de inferência (`txtpragab.docx`, extraído em
`revp_fase2_decisoes_design_contrato.md`), com os dois pontos que a Fase 2
já decidiu (CI = bootstrap preditivo; DINO nunca soma ao score).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RegionGeometry(BaseModel):
    geometry: dict = Field(..., description="GeoJSON Polygon ou MultiPolygon")
    crs: str = Field(default="EPSG:4326")


class Period(BaseModel):
    start: str | None = None
    end: str | None = None


class ScoreRequest(BaseModel):
    request_id: str
    region: RegionGeometry
    period: Period | None = None
    requested_layers: list[str] = Field(default_factory=list)
    visual_patch_id: str | None = Field(
        default=None,
        description=(
            "Opcional. patch_id do corpus DINOv2 (v2fg) que representa visualmente "
            "esta requisição, quando o chamador já o conhece. Só alimenta a camada "
            "de governança visual -- não entra no modelo físico nem altera o score."),
    )


class FeatureUsed(BaseModel):
    name: str
    contribution: Literal["high", "medium", "low"]
    stability: Literal["robust", "unstable"]


class ScoreBlock(BaseModel):
    value: float | None
    confidence_interval: list[float] | None
    model_version: str | None


class Evidence(BaseModel):
    observational_points_used: int
    sources: list[str]
    dino_embedding_available: bool = False
    dino_patch_id: str | None = None


DinoGovernanceStatus = Literal[
    "in_domain",              # embedding real dentro do domínio observado do corpus
    "out_of_domain",          # gate OOD acionado (similaridade abaixo do limiar)
    "no_visual_evidence",     # nenhum embedding real pôde ser resolvido
    "invalid_embedding",      # embedding resolvido mas rejeitado na validação
    "governance_unavailable", # manifesto de medoids ausente/incompatível
]


class DinoMedoidSimilarity(BaseModel):
    """Similaridade contra UM medoid territorial. A resposta expõe o ranking
    inteiro, não só o vencedor -- é a evidência bruta da decisão do gate."""

    region: str
    medoid_patch_id: str
    cosine_similarity: float


class DinoGovernanceAudit(BaseModel):
    """Rastro para auditoria: de onde veio cada número. Só caminhos de
    arquivo reais do repositório -- nenhuma URL de auditoria é emitida."""

    model_config = ConfigDict(extra="allow")

    manifest_path: str | None = None
    manifest_version: str | None = None
    manifest_generated_at: str | None = None
    corpus_path: str | None = None
    corpus_candidates: int | None = None
    corpus_valid: int | None = None
    corpus_blocked: int | None = None
    model_name: str | None = None
    embedding_dim: int | None = None
    medoid_definition: str | None = None
    ood_threshold_basis: str | None = None
    ood_threshold_source: str | None = None
    territorial_concordance_in_corpus: float | None = None
    methodological_note: str | None = None


class DinoGovernance(BaseModel):
    """Camada de governança visual DINOv2 (v2fg).

    NUNCA soma ao score: é validação de domínio, similaridade territorial e
    evidência de auditoria. `status` e `territorial_match` são estados
    explícitos -- a API não bloqueia inferência em silêncio por causa do DINO,
    e o gate OOD não altera `score.value` nem `score.confidence_interval`.
    """

    status: DinoGovernanceStatus
    query_patch_id: str | None = None
    cosine_similarity: float | None = None
    nearest_medoid_patch_id: str | None = None
    suggested_region: str | None = None
    requested_region: str | None = None
    territorial_match: Literal["match", "mismatch", "not_applicable"] = "not_applicable"
    ood_threshold: float | None = None
    ranking: list[DinoMedoidSimilarity] = Field(default_factory=list)
    audit: DinoGovernanceAudit = Field(default_factory=DinoGovernanceAudit)
    notes: list[str] = Field(default_factory=list)
    affects_score: Literal[False] = False


class ScoreResponse(BaseModel):
    request_id: str
    status: Literal["ok", "insufficient_data", "region_not_supported"]
    region_maturity: Literal["available", "limited_evidence", "insufficient"]
    score: ScoreBlock
    features_used: list[FeatureUsed]
    evidence: Evidence
    dino_governance: DinoGovernance
    limitations: list[str]
    data_version: str
    generated_at: str
