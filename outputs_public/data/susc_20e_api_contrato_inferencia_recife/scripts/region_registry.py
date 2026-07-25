"""Registro de regiões suportadas -- reflete o que o projeto REALMENTE tem,
nao uma promessa. Ver PLANO_ACAO_produto_v1.md secao 4 e
revp_fase2_decisoes_design_contrato.md gate #8.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionInfo:
    name: str
    bbox_wgs84: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    model_version: str | None  # None => nenhum modelo estatistico treinado ainda
    region_maturity: str  # "available" | "limited_evidence" | "insufficient"
    status_note: str


REGIONS: dict[str, RegionInfo] = {
    "recife": RegionInfo(
        name="recife",
        bbox_wgs84=(-35.05, -8.15, -34.85, -7.90),
        model_version="SUSC-20 v12 (Firth, n=269, LOO-AUC=0.6781)",
        region_maturity="available",
        status_note="Único modelo estatístico real do projeto. MVP roda aqui.",
    ),
    "curitiba": RegionInfo(
        name="curitiba",
        bbox_wgs84=(-49.45, -25.65, -49.10, -25.30),
        model_version=None,
        region_maturity="limited_evidence",
        status_note=(
            "Evidência em processamento: footprint SAR + corpus Sentinel + "
            "corroboração hidrológica real (ANA) + 1 evento MODIS-validado real "
            "(Global Flood Database, DFO_4276/2015) -- ver "
            "susc_curitiba_leadb_ana_estacoes_reais/reports/RELATORIO_fase4_curitiba_leads_abc.md. "
            "Sem modelo Firth próprio; geometria oficial de ocorrência do evento de "
            "2022-01-15/16 ainda ausente (bloqueio SUSC-18C)."
        ),
    ),
    "petropolis": RegionInfo(
        name="petropolis",
        bbox_wgs84=(-43.25, -22.55, -43.05, -22.30),
        model_version=None,
        region_maturity="insufficient",
        status_note=(
            "Dados insuficientes para inferência: mistura enchente/deslizamento "
            "ainda não separada por fenômeno -- bloqueio documentado, não resolvido."
        ),
    ),
}


def find_region_for_bbox(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> str | None:
    """Retorna o nome da região cuja bbox intersecta a geometria de entrada, ou
    None se nenhuma região conhecida for tocada (region_not_supported)."""
    for name, info in REGIONS.items():
        rxmin, rymin, rxmax, rymax = info.bbox_wgs84
        if lon_min <= rxmax and lon_max >= rxmin and lat_min <= rymax and lat_max >= rymin:
            return name
    return None
