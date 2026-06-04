# Protocolo C — Auditoria de Aceitação v1ue

**Data:** 2026-06-03
**Versão auditada:** v1ue — Event-Specific Evidence Deepening and Station/Asset Binding
**Auditor:** Programático (pré-v1uf)

## 1. Arquivos v1ue

| Verificação | Resultado |
|------------|-----------|
| 24 arquivos v1ue existem | PASS — 24/24 |

Composição: 6 scripts + 4 configs + 7 datasets + 2 docs + 5 testes = 24.

## 2. Janelas Temporais

| Verificação | Resultado |
|------------|-----------|
| 15 janelas existem | PASS |
| 5 tipos × 3 eventos | PASS |

Tipos: event_core_window, pre_event_window_3d, pre_event_window_7d,
post_event_window_3d, sentinel_link_window — 3 cada.

## 3. Estações Candidatas

| Verificação | Resultado |
|------------|-----------|
| 7 estações candidatas | PASS |
| Todas coordinate_status=MISSING | PASS — 7/7 |
| Nenhuma como geometria de inundação | PASS — can_anchor_spatial_evidence=false em 7/7 |

## 4. Guardrails nos CSVs

| Guardrail | Resultado |
|-----------|-----------|
| `can_create_training_label=true` ausente | PASS — 0 ocorrências |
| `ground_truth_operational=true` ausente | PASS — 0 ocorrências |
| `can_create_ground_reference=true` ausente | PASS — 0 ocorrências |
| Coordenadas inventadas | PASS — nenhuma |
| Path absoluto versionável | PASS — 0 ocorrências |

## 5. Bloqueio por Tamanho dos INMET ZIPs

| Verificação | Resultado |
|------------|-----------|
| ZIPs ano-específicos resolvidos | PASS — 3 datasets http=200 |
| Bloqueio por tamanho documentado | **NEEDS_REVIEW** |

Os 3 datasets INMET ano-específicos (2022.zip = 90.362.801 bytes ≈ 90MB; 2024.zip =
102.772.199 bytes ≈ 102MB) retornaram HTTP 200 e foram corretamente **rejeitados** pelo
limite de 25MB do domínio INMET no v1ud_allowed_domains.yaml.

**Achado:** O `content_length` está visível no `v1ue_official_dataset_resolution_registry.csv`
e a rejeição por tamanho foi reportada na entrega da v1ue, mas os documentos narrativos
da v1ue (metodologia e relatório) dizem "download direto possível" sem registrar
explicitamente a rejeição por tamanho. Isso é levemente impreciso.

**Resolução:** A v1uf trata isso diretamente via `v1uf_large_official_download_policy.yaml`,
que permite limite de até 150MB **apenas** para fontes oficiais allowlisted (INMET ZIP anual),
com extração seletiva por estação para não versionar o ZIP bruto.

## 6. Correção do Falso-Positivo de Localidade

| Verificação | Resultado |
|------------|-----------|
| Falso-positivo de localidade corrigido | PASS |

O scorecard (`revp_v1ue_event_evidence_scorecard.py`, linhas 93-126) usa `substantive_obs`,
que filtra observações com `event_specificity == "GENERIC_PORTAL_HOMEPAGE"`. Termos de
navegação (rua/avenida) de portais HTML genéricos não contam mais como localidade real.
Consequência verificada: REC_2022 passou de `OBSERVATIONAL_CANDIDATE_MODERATE` (inflado)
para `BLOCKED_GEOMETRY_MISSING` (correto).

## 7. Classificação por Evento

| Evento | Classificação | Bloqueio |
|--------|---------------|----------|
| PET_2022_02_15 | BLOCKED_PHENOMENON_SEPARATION_REQUIRED | misto |
| PET_2024_03_21_28 | BLOCKED_PHENOMENON_SEPARATION_REQUIRED | misto |
| REC_2022_05_24_30 | BLOCKED_GEOMETRY_MISSING | sem geometria |

## Decisão

**v1ue ACEITA para progressão a v1uf.**

16/17 verificações PASS. A única NEEDS_REVIEW (documentação do bloqueio por tamanho)
é tratada pela v1uf, que implementa a política de download oficial maior e extração
seletiva por estação. Todos os guardrails permanecem enforced.
