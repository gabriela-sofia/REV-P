# SUSC-13B-AUTO - Prontidao observacional

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## Contagens (apenas registros preferenciais, sem duplicatas)
- Eventos fortes: **0**
- Eventos moderados: **2**
- Eventos observados (forte+moderado): **2**
- Eventos observados com data/periodo: **1**
- Eventos observados com geometria: **2**
- Links patch-evento fortes: **0**
- Links patch-evento moderados: **0**
- Links fortes+moderados: **0**
- Links permitidos para avaliacao observacional: **7**
- Regioes cobertas: **petropolis, recife**

## Prontidao
- SUSC-12A (temporal): **BLOQUEADO** — exige >=10 eventos com data e >=10 links fortes/moderados.
- SUSC-12B (contraste de features): **BLOQUEADO** — exige >=10 patches fortes/moderados e controles na regiao.
- SUSC-12C (calibracao de proxy): **BLOQUEADO** — exige >=20 patches fortes/moderados e >=2 regioes.
- Score v7: **BLOQUEADO** — exige >=20 patches, data/geometria suficientes, 2 regioes e evidencia nao concentrada numa unica fonte fraca.

## Conclusao
A camada observacional automatica ainda nao atinge os limiares minimos; o gargalo segue sendo a ausencia de fonte oficial com data + geometria explicita acessivel automaticamente (offline e o modo padrao).

Esta etapa **nao cria score v7** e nao altera a matriz SUSC-03. Mesmo se PRONTO,
qualquer avanco exige revisao humana e permanece review-only.
