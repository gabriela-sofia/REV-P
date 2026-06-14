# v2bv — Dossiê formal QA e scaffold de protocolo de negativos comparáveis

Versão: `v2bv`
Modo: auditoria metodológica autônoma. Não cria label, não cria negativo formal, não libera treino.

## 1. Por que o v2bv existe

O v2bu produziu o primeiro sinal geométrico robusto do projeto (REC_00276). O
v2bv consolida esse resultado numa camada formal de decisão metodológica **sem
ultrapassar a fronteira de label**: um dossiê QA positivo, um registro separado
de method-dependent, um scaffold de negativos comparáveis e um gate formal de
ground truth — tudo abaixo da fronteira da verdade de campo.

## 2. Por que REC_00276 é candidato forte, mas não label

REC_00276 é o patch QA-compatível mais robusto (intersecta 4 métodos, incluindo
reconstruções tight; razão de interseção 0.88). Vira um
`FORMAL_QA_POSITIVE_CANDIDATE_DOSSIER` com status
`STRONG_QA_POSITIVE_CANDIDATE_HELD_FOR_FORMAL_FOOTPRINT_VALIDATION`. Mesmo assim:
`formal_gt_ready=false`, `gt_patch_flood_observed=NA`, `allowed_for_training=false`.
O footprint do evento ainda é QA-only derivado de pontos e nenhum protocolo
positivo foi aceito.

## 3. Por que REC_00299 é method-dependent

REC_00299 intersecta apenas as reconstruções permissivas (convex hull + buffer
maior), não as tight. É registrado separadamente como
`METHOD_DEPENDENT_HELD_FOR_TIGHTER_EVENT_GEOMETRY`. Não pode entrar no mesmo
nível do candidato robusto.

## 4. Por que os 35 noncompatible não são negativos

Os patches não compatíveis não intersectam a geometria QA-only do evento, mas
**não-compatibilidade não é negativo, e ausência não é negativo**. Eles são
scaffoldados como **candidatos** a negativo comparável apenas. No estado atual,
14 atingem `COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY` (dentro da banda de distância)
e 21 ficam `NOT_COMPARABLE_NEGATIVE_CANDIDATE_DISTANCE_TOO_FAR`. Nenhum é
negativo formal.

## 5. Como funciona o scaffold de negativos comparáveis

Cada candidato é checado contra critérios rígidos de comparabilidade (mesma
região, mesmo contexto de evento, boundary disponível, distância controlada ao
footprint QA, família de fonte comparável, não method-dependent, nunca
ausência/fundo aleatório). Mesmo um candidato que passa resulta em
`formal_negative_label_created=false`. Como `formal_protocol_exists=false`,
nenhum negativo pode ser criado.

## 6. O que ainda falta para ground truth formal

Footprint oficial revisado do evento, protocolo positivo aceito, protocolo de
amostragem de negativos, definição formal de negativos comparáveis, blocking
espacial e split anti-leakage, e um target de treino aceito — ver
`gt_protocol_gap_analysis_v2bv.csv`.

## 7. O que ainda falta para treino supervisionado

Labels formais, negativos formais, target de treino e protocolo anti-leakage
estão todos ausentes. `can_train_supervised_model=false`. Análises permitidas
agora: revisão do dossiê QA, validação de footprint, design do protocolo de
negativos, auditoria da feature table.

## 8. Por que treino segue bloqueado

`labels_created=false`, `formal_gt_ready=false`, `allowed_for_training_count=0`.
Um dossiê QA e um scaffold de negativos não criam labels nem desbloqueiam treino.

## Outputs

`local_runs/ground_truth/v2bv/` (11 arquivos `.csv`/`.json`/`.md`, leves,
incluindo o dossiê individual `formal_qa_positive_dossier_REC_00276_v2bv.md`).
Nenhum arquivo pesado é gravado ou versionado.
