"""SUSC-15B step 8: summarize forensic precision readiness."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402


def main() -> int:
    catalog = c.read_csv(c.CATALOG)
    links = c.read_csv(c.LINKAGE)
    eligible = [row for row in catalog if row.get("calibration_eligible") == "true"]
    precise_links = [row for row in links if row.get("can_be_used_for_observational_evaluation") == "true"]
    tiers = c.tier_counts(c.read_csv(c.GEOCODED_15A))
    ready_16a = len(eligible) >= 10 and len(precise_links) >= 10
    ready_v7 = False
    c.write_markdown(c.READINESS, f"""# SUSC-15B - prontidao de resgate forense de precisao

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

{c.MANDATORY}

## Metricas
- Eventos avaliados: {len(catalog)}
- Eventos elegiveis para calibracao: {len(eligible)}
- Links patch-evento precisos: {len(precise_links)}
- Patches observacionais precisos: {len({row.get('patch_id') for row in precise_links})}
- T0 official event polygon: {tiers.get('T0_official_event_polygon', 0)}
- T1 official event point: {tiers.get('T1_official_event_point', 0)}
- T2 official address point or parcel: {tiers.get('T2_official_address_point_or_parcel', 0)}
- T3 official intersection point: {tiers.get('T3_official_intersection_point', 0)}
- T4 official house number linear reference: {tiers.get('T4_official_house_number_linear_reference', 0)}
- T5 street segment candidates: {tiers.get('T5_official_street_segment_candidate', 0)}
- T6 bairro-only/context: {tiers.get('T6_neighborhood_context_only', 0)}

## Readiness
- ready_for_16a: {ready_16a}
- ready_for_score_v7: {ready_v7}
- score_v7_created: False

## Decisao
SUSC-16A segue bloqueado porque nao ha pelo menos 10 eventos elegiveis e 10 links patch-evento precisos. O proximo passo recomendado e revisao manual/aquisicao controlada das fontes listadas no manifesto SUSC-15B.
""")
    c.write_markdown(c.REPORT, f"""# SUSC-15B - Forensic Precision Rescue

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

{c.MANDATORY}

## Resultado
SUSC-15B executou uma rodada densa de auditoria forense sobre campos ocultos, identificadores internos, fontes profundas candidatas, joins oficiais e linkage patch-evento. Nenhuma fonte adicional materializou ponto, poligono, lote, intersecao ou faixa numerica controlada suficiente para calibracao.

## Por que cada familia passou ou falhou
- IDs/codigos internos: retidos para join manual, mas sem tabela oficial auxiliar materializada.
- Colunas escondidas de coordenada/UTM: auditadas; nenhuma nova geometria oficial elegivel foi promovida.
- Tabelas oficiais auxiliares: bloqueadas por ausencia de arquivo/fonte oficial precisa nesta execucao.
- PDFs/anexos: bloqueados por ausencia de anexo local oficial parseavel.
- ArcGIS/WFS/FeatureServer: registrados como alvo, sem download automatico nesta execucao.
- Pontos criticos/156/Defesa Civil: bloqueados sem camada oficial precisa vinculavel.
- Intersecoes oficiais: bloqueadas sem gazetteer oficial materializado.
- Numeracao predial: bloqueada sem faixa/lote oficial controlado.
- Footprints Copernicus/EMS/SGB/INEA/APAC: bloqueados sem footprint oficial vinculavel aos eventos.

## Metricas
- Eventos avaliados: {len(catalog)}
- Eventos elegiveis para calibracao: {len(eligible)}
- Links patch-evento precisos: {len(precise_links)}
- Score v7 criado: False

## Limitacoes
Nao houve push, ground truth, treino supervisionado, modelo persistido, coordenada inventada, data inventada, bairro como coordenada ou score v7 automatico.
""")
    print(f"15B readiness: eligible={len(eligible)} links={len(precise_links)} ready_16a={ready_16a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
