# v1pa — Protocol C Observed Evidence Bundle

## Objetivo

Consolidar v1ou-v1oz em manifest, QC e summary científico final. Não recalcula decisões científicas — auditoria apenas.

## Relação com v1og-v1ot

v1og-v1ot fechou recuperação temporal com TEMPORAL_RECOVERY_FAIL_CLOSED: 0 product_dates confirmadas em 2.654 patches. v1ou-v1pa não tenta destravar temporal artificialmente. Toda a camada observacional permanece em regime review-only/contextual.

## O que é evento observado

Evento observado é uma ocorrência de inundação ou deslizamento documentada por fonte rastreável (decreto, boletim oficial, laudo técnico). Neste bloco, os candidatos a eventos de Recife foram identificados em dossiers e registros de gaps — mas nenhum foi confirmado por fonte adquirida, pois G1/G3/G4 permanecem abertos.

## O que é evidência contextual

Evidência contextual é informação geomorfológica, topográfica ou de drenagem que descreve o ambiente mas não confirma evento específico. PE3D MDE e drenagem ESIG são exemplos de contexto que não geram ground truth.

## Por que C3/C4 permanecem bloqueados

C3+ requer scene_date Sentinel confirmada (product_dates_confirmed_real=0 de v1ot). C4 requer negativo formal explícito (formal_negative_count=0 de v1ot). Nenhuma condição foi satisfeita.

## Papel do DINO

DINOv2 with registers é usado exclusivamente para representação estrutural visual — embeddings de revisão sem label, sem target, sem ground truth derivado.

## Texto recomendado para o TCC

A camada observacional do Protocolo C foi estruturada como registro auditável de eventos e evidências externas, separando fonte, data, precisão temporal, localização, precisão espacial e vínculo com patches Sentinel. Essa estrutura não promove automaticamente evidências externas a ground truth operacional. Na ausência de cadeia temporal Sentinel confirmada e de negativos formais, os registros permanecem em regime review-only/contextual, preservando os embeddings DINOv2 como representação visual sem criação de rótulos supervisionados.

## QC

Total de checks: 78. Falhas: 0. Críticos: 0. Status final: OBSERVED_EVIDENCE_REVIEW_ONLY_FAIL_CLOSED.
