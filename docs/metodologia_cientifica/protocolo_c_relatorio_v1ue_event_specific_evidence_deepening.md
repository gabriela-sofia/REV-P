# Protocolo C — Relatório v1ue Event-Specific Evidence Deepening

**Gerado em:** 2026-06-03T09:09:39.694976  
**Versão:** v1ue  

## Visão Geral

| Métrica | Valor |
|---------|-------|
| Eventos | 3 |
| Estações candidatas | 7 |
| Datasets resolvidos | 9 |
| Evidências evento-específicas | 0 |
| Evidências ainda genéricas (portal) | 3 |
| Próximas ações | 20 |

## Tabela por Evento

| Evento | Classificação | Temporal | Geometria | Fenômeno | Bloqueios |
|--------|---------------|----------|-----------|----------|----------|
| PET_2022_02_15 | BLOCKED_PHENOMENON_SEPARATION_REQUIRED | 0.8 | 0.0 | 0.1 | PHENOMENON_SEPARATION_REQUIRED|GEOMETRY_MISSING|FORMAL_REQUEST_REQUIRED |
| PET_2024_03_21_28 | BLOCKED_PHENOMENON_SEPARATION_REQUIRED | 0.8 | 0.0 | 0.1 | PHENOMENON_SEPARATION_REQUIRED|GEOMETRY_MISSING |
| REC_2022_05_24_30 | BLOCKED_GEOMETRY_MISSING | 0.8 | 0.0 | 0.5 | GEOMETRY_MISSING |

## Perguntas-Chave

### Quais evidências deixaram de ser homepage genérica?
- Nenhuma ainda totalmente evento-específica. Datasets ano-específicos (INMET) resolvidos mas dependem de download/parse de série.

### Quais eventos ganharam âncora temporal?
- PET_2022_02_15: SIM (score=0.8)
- PET_2024_03_21_28: SIM (score=0.8)
- REC_2022_05_24_30: SIM (score=0.8)

### Quais eventos ganharam estação candidata oficial?
- PET_2022_02_15: 3 estação(ões) candidata(s)
- PET_2024_03_21_28: 3 estação(ões) candidata(s)
- REC_2022_05_24_30: 1 estação(ões) candidata(s)

### Quais eventos continuam sem geometria?
- PET_2022_02_15: SEM geometria observacional
- PET_2024_03_21_28: SEM geometria observacional
- REC_2022_05_24_30: SEM geometria observacional

### Quais eventos continuam sem separação de fenômeno?
- PET_2022_02_15: fenômeno MISTO — separação pendente
- PET_2024_03_21_28: fenômeno MISTO — separação pendente

### Quais fontes exigem pedido formal?
- ANA_HIDROWEB
- CEMADEN_PLUVIOMETROS
- DEFESA_CIVIL
- SGB_CPRM_CARTOGRAFIA

### Qual é o próximo melhor alvo de aquisição?
1. Série anual INMET específica (download direto possível)
2. Pontos de ocorrência georreferenciados da Defesa Civil (geometria observacional)
3. Geodata de campo SGB/CPRM (com separação de fenômeno)

### O que falta para qualquer evento virar ground reference?
- Geometria observacional oficial (ausente em todos)
- Separação de fenômeno para eventos PET (mistos)
- Revisão de supervisor (não executada)
- Overlay patch-evidência (não executado nesta etapa)

### Por que ainda não há ground truth operacional?
- `ground_truth_operational=false` é invariante do protocolo nesta fase
- Score alto define apenas próxima ação, nunca cria label
- Gates G10 (overlay) e G11 (supervisor) permanecem FAIL

## Invariantes — Confirmação Explícita

- **ground_truth_operational** = `False`
- **can_create_training_label** = `False`
- **can_reopen_protocol_b** = `False`
- **dino_usage** = `SUPPORT_ONLY`
- **no_overlay_executed** = `True`
- **no_coordinates_invented** = `True`
- **can_create_ground_reference** = `false` (todos os eventos)
- **supervisor_review_completed** = `false` (todos os eventos)

---
*Relatório gerado por Protocol C v1ue.*
