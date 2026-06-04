# Protocolo C — Relatório v1uf Station-Resolved Acquisition

**Gerado em:** 2026-06-03T09:38:43.529063  
**Versão:** v1uf  

## Resumo

| Métrica | Valor |
|---------|-------|
| Eventos | 3 |
| Assets de série extraídos | 3 |
| Coordenadas oficiais resolvidas | 3 |
| Métricas de janela computadas | 8 |
| Próximas ações | 17 |

## Tabela por Evento

| Evento | Nível Hidromet | Série | Coord | Precip Evento | Bloqueio |
|--------|---------------|-------|-------|---------------|----------|
| PET_2022_02_15 | BLOCKED_PHENOMENON_SEPARATION_REQUIRED | true | true | true | PHENOMENON_SEPARATION_REQUIRED |
| PET_2024_03_21_28 | BLOCKED_PHENOMENON_SEPARATION_REQUIRED | true | true | true | PHENOMENON_SEPARATION_REQUIRED |
| REC_2022_05_24_30 | BLOCKED_INSUFFICIENT_COVERAGE | true | true | false | INSUFFICIENT_COVERAGE |

## Perguntas-Chave

### Quais ZIPs oficiais foram baixados?
- 2 ZIP(s) oficial(is) único(s) baixado(s) (por SHA256), em local_only/ (não versionado)

### Quais arquivos foram extraídos?
- PET_2022_02_15: INMET_SE_RJ_A610_PICO DO COUTO_01-01-2022_A_31-12-2022.CSV (code=A610)
- PET_2024_03_21_28: INMET_SE_RJ_A610_PICO DO COUTO_01-01-2024_A_31-12-2024.CSV (code=A610)
- REC_2022_05_24_30: INMET_NE_PE_A301_RECIFE_01-01-2022_A_31-12-2022.CSV (code=A301)

### Quais estações foram associadas aos eventos?
- 7 estação(ões) candidata(s) no catálogo v1uf

### Quais coordenadas oficiais foram resolvidas?
- A610 (PICO DO COUTO): lat=-22.46472222 lon=-43.29138888 [FROM_OFFICIAL_CATALOG]
- A610 (PICO DO COUTO): lat=-22.46472222 lon=-43.29138888 [FROM_OFFICIAL_CATALOG]
- A301 (RECIFE): lat=-8.01888888 lon=-34.94416666 [FROM_OFFICIAL_CATALOG]

### Quais métricas de precipitação foram calculadas?
- PET_2022_02_15 / event_core_window: total=2.8mm max_diário=2.8mm (cobertura=1.0)
- PET_2022_02_15 / pre_event_window_3d: total=16.2mm max_diário=12.2mm (cobertura=1.0)
- PET_2022_02_15 / pre_event_window_7d: total=97.6mm max_diário=38.0mm (cobertura=1.0)
- PET_2022_02_15 / post_event_window_3d: total=45.8mm max_diário=22.0mm (cobertura=1.0)
- PET_2024_03_21_28 / event_core_window: total=265.8mm max_diário=80.2mm (cobertura=1.0)
- PET_2024_03_21_28 / pre_event_window_3d: total=66.0mm max_diário=51.8mm (cobertura=1.0)
- PET_2024_03_21_28 / pre_event_window_7d: total=66.2mm max_diário=51.8mm (cobertura=1.0)
- PET_2024_03_21_28 / post_event_window_3d: total=17.0mm max_diário=10.8mm (cobertura=1.0)

### Quais eventos ganharam âncora hidrometeorológica real?
- PET_2022_02_15: SIM (nível=BLOCKED_PHENOMENON_SEPARATION_REQUIRED)
- PET_2024_03_21_28: SIM (nível=BLOCKED_PHENOMENON_SEPARATION_REQUIRED)
- REC_2022_05_24_30: não (nível=BLOCKED_INSUFFICIENT_COVERAGE)

### Quais continuam só com portal genérico?

### Quais continuam bloqueados por fenômeno misto?
- PET_2022_02_15: fenômeno MISTO — separação pendente
- PET_2024_03_21_28: fenômeno MISTO — separação pendente

### Quais continuam bloqueados por ausência de geometria?
- PET_2022_02_15: SEM geometria observada (todos nesta etapa)
- PET_2024_03_21_28: SEM geometria observada (todos nesta etapa)
- REC_2022_05_24_30: SEM geometria observada (todos nesta etapa)

### O que falta para ground reference?
- Geometria observacional oficial (ausente em todos)
- Separação de fenômeno (eventos PET mistos)
- Revisão de supervisor (não executada)
- Overlay patch-evidência (não executado)

### Por que ainda não há ground truth operacional?
- `ground_truth_operational=false` é invariante do protocolo
- Chuva forte no evento melhora plausibilidade temporal, NÃO cria ground reference
- Estação é sensor pontual, não geometria de extensão de inundação
- Score/precipitação alto define apenas próxima ação

## Invariantes — Confirmação Explícita

- **ground_truth_operational** = `False`
- **can_create_ground_reference** = `False`
- **can_create_training_label** = `False`
- **can_reopen_protocol_b** = `False`
- **dino_usage** = `SUPPORT_ONLY`
- **no_overlay_executed** = `True`
- **no_coordinates_invented** = `True`
- **supervisor_review_completed** = `False`
- **estação oficial não é geometria de inundação**
- **precipitação ancora plausibilidade temporal, não patch-level truth**

---
*Relatório gerado por Protocol C v1uf.*
