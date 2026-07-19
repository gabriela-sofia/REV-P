# MV2-15 - Cronograma Engine & Gate Orchestrator

Marco engineering-only, review-only e fail-closed. Nao executa STAC real, nao baixa raster, nao cria crop, feature espectral, label, silver, gold, negativo, split, sandbox ou treino.

## Fechamento da vertente de dados

MV2-10 a MV2-14 mostraram que o gargalo atual e dado/evidencia: ha diagnostico suficiente para formalizar gates, mas nao para promover treino ou operacao.
Lineage local real existe apenas como Protocolo-C/Petropolis; no corpus oficial, MV2-14 manteve 48 candidatos de revisao e 80 sem lineage.

## Estado do cronograma

- Dia 8: PARTIAL - DINOv2/visuais e subset limitado existem, mas vies regional segue alto
- Dia 9: DONE - QA label-free existe como controle review-only; nao promove label
- Dia 10: BLOCKED_DATA - sem raster Sentinel nativo, lineage completo, STAC real, crop ou feature espectral
- Dia 11: DONE - sanity label-free e vizinhanca existem como analise review-only
- Dia 12: PARTIAL - confounder regional/cidade esta documentado e permanece risco alto
- Dia 13: DONE - baseline/controle metodologico existe como referencia nao treinavel
- Dia 14: DONE - pacote publico label-free existe e preserva semantica review-only
- Dia 15: READY_REVIEW_ONLY - politica existe, mas sem negativos formais criados
- Dia 16: BLOCKED_EVIDENCE - nao existem negativos formais; ausencia nao vira negativo
- Dia 17: PARTIAL - ha filas/revisoes candidatas, mas sem fechamento observacional patch-level
- Dia 18: BLOCKED_EVIDENCE - sem evidencia manual/observacional suficiente para desbloqueio
- Dia 19: BLOCKED_EVIDENCE - silver formal permanece zero
- Dia 20: PARTIAL - pacote leve existe, mas reproducibilidade externa ainda depende de dados ausentes
- Dia 21: BLOCKED_METHOD - sem labels/silver/splits treinaveis
- Dia 22: BLOCKED_GUARDRAIL - sandbox supervisionado proibido sem silver formal e split treinavel

## Bloqueios ativos

- Dia 10: GATE_DAY10_BLOCKED - native_raster=0;lineage_strong=0;stac_real=0
- Dia 16: GATE_NO_NEGATIVES_FORMAL - formal_negatives=0
- Dia 18: GATE_DAY18_BLOCKED - ground_truth_operacional_status=ausente
- Dia 19: GATE_DAY19_BLOCKED - silver_formal=0
- Dia 21: GATE_DAY21_BLOCKED - labels=0;silver=0;can_train=false
- Dia 22: GATE_DAY22_BLOCKED - can_train=false;silver_formal=0

## Programacao pesada liberada

- Acoes permitidas agora: 12
- Escopo: consolidacao de engenharia, schemas, testes, CLI, dashboards, registry, validadores e pacote leve reprodutivel.

## Backlog recomendado

- CLI unico revp mv2 status
- CLI unico revp mv2 validate
- CLI unico revp mv2 gates
- CLI unico revp mv2 report
- Consolidar schemas MV2-10 a MV2-14

## Acoes proibidas

- STAC_REAL; RASTER_DOWNLOAD; PRIVATE_CROP; SPECTRAL_FEATURE_EXTRACTION; SUPERVISED_BASELINE; TRAINING; SILVER_PROMOTION; NEGATIVE_CREATION; SPLIT_TRAINABLE; SANDBOX_SUPERVISED

## Proximo pacote recomendado

MV2-16 deve implementar CLI/validadores globais e dashboard de gates usando apenas outputs leves ja existentes.
