# SUSC-17D Validacao tecnica de evidencia observacional

## Estado herdado do 17C5

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `ad95ce6`
- Area staged: 0 arquivo(s)
- Geometrias resolvidas no 17C5: 1
- Geometrias nao resolvidas no 17C5: 62
- Vinculos evento-patch herdados: 67
- Vinculos fortes candidatos herdados: 5
- Itens herdados para validacao tecnica: 5
- Estado 17B herdado: bloqueado_parcial_aguardando_validacao_tecnica_17d
- Itens avaliados: 5
- `score_v6` alterado: False
- `score_v7` criado: False

## Metodologia de validacao tecnica

A validacao comparou fonte, data, fenomeno, geometria, vinculo com patch, incerteza espacial, features disponiveis, score v6 e bloqueios metodologicos. A decisao e review-only e nao cria verdade de referencia, treino, score_v7 ou benchmark 17B.

## Criterios de decisao

Um item so e aceito para avaliacao review-only quando tem geometria real resolvida, patch identificado, classe de vinculo forte, temporalidade suficiente, fenomeno compativel, justificativa tecnica e ausencia de bloqueio critico.

## Contagem por status

- aceito_para_avaliacao_review_only: 5

## Itens avaliados

- S17D_VAL_0001 / S17C_REF_0063 / S17C6_CANARY_REC_00002 / aceito_para_avaliacao_review_only
- S17D_VAL_0002 / S17C_REF_0063 / S17C6_CANARY_REC_00001 / aceito_para_avaliacao_review_only
- S17D_VAL_0003 / S17C_REF_0063 / S17C6_CANARY_REC_00003 / aceito_para_avaliacao_review_only
- S17D_VAL_0004 / S17C_REF_0063 / S17C6_CANARY_REC_00004 / aceito_para_avaliacao_review_only
- S17D_VAL_0005 / S17C_REF_0063 / S17C6_CANARY_REC_00005 / aceito_para_avaliacao_review_only

## Itens aceitos, rejeitados e ambiguos

- Aceitos para avaliacao review-only: 5
- Candidatos a calibracao futura: 0
- Rejeitados: 0
- Ambiguos: 0
- Precisam de mais fonte: 0

## Comparacao feature/score

- Linhas de matriz de features: 90
- Features disponiveis: 0
- Observacao: ausencia de feature nao reprova automaticamente, mas reduz prontidao para calibracao.

## Bloqueios criticos

- Linhas de bloqueio: 5
- Principal bloqueio: patches candidatos 17C6 ainda nao possuem features/score v6 materializados e exigem reducao de incerteza antes de calibracao.

## Gate de prontidao 17B

- minimo_3_eventos_datados: passou=false (1 / 3)
- minimo_2_regioes: passou=false (1 / 2)
- minimo_1_geometria_forte_por_regiao: passou=false (1 / 2)
- minimo_20_patch_links_fortes_aceitos: passou=false (5 / 20)
- validacao_tecnica_aceita_suficiente: passou=false (5 / 20)
- ground_truth_false: passou=true (0 / 0)
- trainable_false: passou=true (0 / 0)

## Conclusao

- Desbloqueado: 5 itens para avaliacao review-only.
- Segue bloqueado: calibracao futura, treino, ground truth, score_v7 e benchmark 17B.
- Candidatos a calibracao futura: 0.
- Status 17E: 17E_BLOQUEADO_SEM_CANDIDATO_CALIBRACAO.
- Status 17B: 17B_PARCIAL_COM_VALIDACAO_TECNICA_SEM_PRONTIDAO_BENCHMARK.

17B continua sem prontidao de benchmark porque os aceites tecnicos sao insuficientes em quantidade, estao concentrados em uma regiao e seguem sem ground truth, sem treino e sem score_v7.
