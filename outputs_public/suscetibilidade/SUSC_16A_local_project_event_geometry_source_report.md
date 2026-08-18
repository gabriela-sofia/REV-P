# SUSC-16A - auditoria local de fontes de geometria de evento

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

O SUSC-16A substitui a tentativa de geocodificacao textual por uma estrategia de footprints observacionais, combinando geometrias locais, fontes oficiais/tecnicas e planejamento Sentinel/SAR. A etapa mantem todos os vinculos review-only, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

## Escopo minerado
- Raizes PROJETO encontradas: 1
- REV-P tambem foi escaneado para copias locais.
- Arquivos candidatos registrados: 3977

## Classificacao
{
  "local_event_polygon_candidate": 989,
  "non_event_context": 802,
  "official_flood_footprint_candidate": 134,
  "official_susceptibility_map_only": 1772,
  "official_risk_area_only": 168,
  "official_drainage_context": 30,
  "official_address_reference": 25,
  "local_event_point_candidate": 57
}

## Decisao
O manifesto registra caminhos sanitizados, tamanho e SHA256 quando viavel. Nenhum dado bruto pesado foi copiado para o REV-P, e nenhuma fonte local foi promovida automaticamente a ground truth ou treino.
