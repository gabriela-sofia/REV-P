# SUSC-15B - prontidao de resgate forense de precisao

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

SUSC-15B e review-only: nao aceita bairro-only como sucesso, nao aceita street segment sem numero/intersecao como calibracao, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

## Metricas
- Eventos avaliados: 4412
- Eventos elegiveis para calibracao: 0
- Links patch-evento precisos: 0
- Patches observacionais precisos: 0
- T0 official event polygon: 0
- T1 official event point: 0
- T2 official address point or parcel: 0
- T3 official intersection point: 0
- T4 official house number linear reference: 0
- T5 street segment candidates: 31
- T6 bairro-only/context: 1598

## Readiness
- ready_for_16a: False
- ready_for_score_v7: False
- score_v7_created: False

## Decisao
SUSC-16A segue bloqueado porque nao ha pelo menos 10 eventos elegiveis e 10 links patch-evento precisos. O proximo passo recomendado e revisao manual/aquisicao controlada das fontes listadas no manifesto SUSC-15B.
