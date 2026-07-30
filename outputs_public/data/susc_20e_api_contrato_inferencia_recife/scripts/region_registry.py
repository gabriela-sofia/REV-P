"""Registro de regiões suportadas -- reflete o que o projeto REALMENTE tem,
nao uma promessa. Ver PLANO_ACAO_produto_v1.md secao 4 e
revp_fase2_decisoes_design_contrato.md gate #8.

Formalizacao (revp_proxima_linhagem_programacao_pos_api.md etapa 2): o
registro agora valida a si mesmo -- uma regiao nao pode reportar
`region_maturity="available"` sem `model_version` associado. A violacao
falha na propria construcao do registro (Pydantic), nao so em teste.
O JSON Schema gerado a partir deste modelo fica versionado em
`../schemas/region_registry_schema_v1.json` (ver
`generate_schema_file()` / `tests/test_susc_20e_region_registry_schema.py`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION = "region_registry.v1"

RegionMaturity = Literal["available", "limited_evidence", "insufficient"]


class RegionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    bbox_wgs84: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    model_version: str | None  # None => nenhum modelo estatistico treinado ainda
    region_maturity: RegionMaturity
    status_note: str

    @model_validator(mode="after")
    def _available_requires_model_version(self) -> "RegionInfo":
        if self.region_maturity == "available" and self.model_version is None:
            raise ValueError(
                f"regiao '{self.name}' reporta region_maturity='available' sem "
                "model_version associado -- viola o gate #8"
            )
        return self


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
            "corroboração hidrológica real (ANA, 2 estações em São José dos Pinhais) "
            "-- ver susc_curitiba_leadb_ana_estacoes_reais/reports/RELATORIO_fase4_curitiba_leads_abc.md. "
            "Sem modelo Firth próprio; geometria oficial de ocorrência do evento de "
            "2022-01-15/16 ainda ausente (bloqueio SUSC-18C). "
            "N de pontos-evento adjudicados = 1 (atualizado -- ver "
            "local_runs/curitiba_modelo_v1_diagnostico_lead_a_e_inventario_pontos/"
            "RELATORIO_v8_CANDIDATO_JUVEVE_ADJUDICACAO.md): pixel MODIS MCDWD_L3 "
            "classe Flood=3 (água anômala, não bate com camada de referência "
            "permanente) em 2022-01-17, bairro Juvevê (-25.4177,-49.2557), corroborado "
            "por reportagem real de área cronicamente alagável (fundo de vale, 3 "
            "córregos canalizados). Ressalva: data do sinal é 17/01, não 15-16/01 "
            "(janela original do evento-catálogo); não dá pra confirmar se é o mesmo "
            "sistema de chuva ou evento distinto. N=1 não sustenta nenhuma feature pelo "
            "orçamento EPV (~10 eventos/variável) -- gate de treino continua não "
            "passando, sem Firth. Antes deste achado, os 3 candidatos anteriores "
            "(2 ANA + 1 GFD) foram adjudicados e nenhum virou ponto -- as 2 estações "
            "ANA são corroboração de área/tempo pela mesma regra do Lead B de Recife, "
            "e o candidato GFD DFO_4276/2015 foi REJEITADO (13 dos 14 pixels dentro do "
            "município são a lâmina d'água da Represa do Passaúna: platô DEM de "
            "887,00 m, JRC Global Surface Water occurrence 84-87%, OSM natural=water). "
            "Lead A esgotado por varredura completa do Legisladoc (64/64 dias úteis de "
            "jan-mar/2022) e por S2ID (0 registros federais de Curitiba em jan/2022)."
        ),
    ),
    "petropolis": RegionInfo(
        name="petropolis",
        bbox_wgs84=(-43.25, -22.55, -43.05, -22.30),
        model_version=None,
        region_maturity="insufficient",
        status_note=(
            "N de pontos-evento adjudicados = 0. O único candidato (Valparaíso, "
            "-22.51625,-43.18828, sinal Sentinel-2 de 2022-03-24) foi REBAIXADO A CANDIDATO "
            "FRACO na reavaliação de 2026-07-28 -- ver "
            "docs/metodologia_cientifica/revp_reavaliacao_candidato_petropolis_valparaiso_v1.md. "
            "Motivo: a leitura topográfica de SUSC-20G/2 reprovou o ponto em três critérios "
            "físicos independentes -- HAND=50,88 m, declividade 23,34° (mediana regional 23,57°) "
            "e TWI 5,11 abaixo da mediana regional 5,58 -- assinatura de encosta, não de "
            "acúmulo em fundo de vale. A hipótese de artefato de resolução foi testada e "
            "descartada: a altura sobre o mínimo local (medida que não depende de extração de "
            "drenagem) dá +30 a +50 m em DOIS DEMs independentes (FABDEM 30 m e Hipsometria "
            "SGB 10 m). A corroboração hidrográfica do laudo original também não se sustentou: "
            "consulta OSM refeita mostra o Rio Quitandinha a 688 m (não ~500 m) e 30-47 m "
            "ABAIXO do ponto; o curso d'água mais próximo é o Rio Aureliano, a 635 m. "
            "Não foi REJEITADO porque a queda absoluta de reflectância em NIR+SWIR é real "
            "(B08 0,304->0,122; B11 0,146->0,090) e descarta nuvem; a hipótese alternativa "
            "(sombra de relevo, fonte documentada de confusão) não pôde ser testada porque as "
            "cenas B03/B08/B11 originais de 04/03 e 24/03/2022 não estão em disco -- sem B03 "
            "não dá pra separar água de sombra. Adjudicação original preservada em "
            "RELATORIO_v14_CANDIDATO_PETROPOLIS_VALPARAISO_ADJUDICACAO.md. "
            "Com N=0 nenhuma feature é sustentável (piso EPV >= 20, revisão de literatura "
            "seção 5) -- gate de treino não passa, sem Firth. "
            "Independente disso, a mistura enchente/deslizamento permanece não separável pelo "
            "critério COBRADE nos registros S2ID adquiridos: Petrópolis tem 3 registros em "
            "2022, todos COBRADE 13214 (Tempestade Local/Convectiva - Chuvas Intensas), "
            "incluindo o de 2022-02-15 com 78 mortos -- zero registros nas classes "
            "hidrológicas (12100/12200/12300) e zero em movimento de massa "
            "(11321/11331/11332), porque a classificação S2ID é pelo gatilho meteorológico, "
            "não pelo processo físico. O registro S2ID em si continua sem geometria/"
            "coordenada (municipal). Ver "
            "docs/metodologia_cientifica/revp_petropolis_s2id_aquisicao_real_cobrade.md. "
            "ATUALIZAÇÃO 2026-07-30: as 2 janelas de evento reais de 2023-2026 (05/04/2025 "
            "forte, 22/03/2024 forte mas misto) foram testadas e esgotadas -- ver "
            "outputs_public/data/susc_20j_sentinel1_sar_water_candidates/reports/"
            "susc_20j2_petropolis_dois_eventos_esgotados_report.md. Evento de 05/04/2025: "
            "4 métodos independentes (3 referências ópticas Via B + 1 SAR Via C, todos "
            "normalizados por z-score e restritos ao corredor real dos rios Quitandinha/"
            "Piabanha via OSM) -- nenhum achou candidato no corredor real; hipótese física "
            "principal é drenagem rápida de terreno de serra antes da primeira imagem limpa "
            "disponível (d+2 a d+4). Evento de 22/03/2024: Via B já fechado por nuvem "
            "persistente (99-100% de d+1 a d+6); Via C (SAR) agora também fechado -- achado "
            "estrutural real: existe 1 única órbita Sentinel-1 cobrindo Petrópolis (29 "
            "passagens de 2024 conferidas, todas mesma órbita relativa), com lacuna de "
            "cobertura real sobre o núcleo urbano (100% sem dado na janela "
            "Centro/Quitandinha em 2024-03-21, confirmado em pixel, não só metadado) -- "
            "nenhuma outra data de 2024 contornaria isso. N=0 permanece; não é falta de "
            "tentativa, é resultado real e documentado dos dois métodos disponíveis nas "
            "duas janelas de maior evidência."
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


def registry_as_dict() -> dict:
    """Serializa o registro atual (schema_version + regiões) para uso por
    consumidores não-Python (ex.: futura interface web, etapa 10 do roadmap)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "regions": {name: info.model_dump() for name, info in REGIONS.items()},
    }


def generate_schema_file(path: Path | None = None) -> Path:
    """Escreve o JSON Schema formal de `RegionInfo` em disco (versionado por
    `SCHEMA_VERSION`). Usado por `tests/test_susc_20e_region_registry_schema.py`
    para garantir que o schema versionado no repo está sincronizado com o
    modelo Pydantic que efetivamente valida o registro."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "schemas" / "region_registry_schema_v1.json"
    schema = {"schema_version": SCHEMA_VERSION, **RegionInfo.model_json_schema()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    written = generate_schema_file()
    print(f"schema escrito em {written}")
