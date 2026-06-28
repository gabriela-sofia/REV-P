# SUSC-14A - Prontidao observacional

Status: **review-only** | `can_be_ground_truth=false` | `can_be_used_as_ground_truth=false` | `allowed_for_training=false`

## Contagens
- Eventos fortes: **0** | moderados: **2** | fracos/contextuais: **27** | bloqueados/documentais: **2**
- Ocorrencias oficiais geocodificadas por camada oficial: **0** (segmento de logradouro: **0**)
- Geocodificacoes bloqueadas por referencia ausente/ambigua: **4024**
- Eventos observados com data/periodo: **1** | com geometria: **2**
- Links patch-evento fortes: **0** | moderados: **0** | fracos: **0** | contextuais: **25**
- Patches com avaliacao observacional: **0** | patches apenas contextuais: **0**
- Regioes cobertas por observacao geometrica: **petropolis, recife**

## Diagnostico score v6 x eventos
- Status: **not_enough_observed_events**
- Media/mediana v6 em links observacionais: **None** / **None**
- Distribuicao de classe v6: **null**
- hit@10=None | hit@20=None | hit@30=None | enriquecimento=None
- Avaliacao observacional bloqueada: **true** | motivo: **no_strong_or_moderate_patch_links**

## Prontidao
- SUSC-12A temporal: **BLOQUEADO**
- SUSC-12B contraste de features: **BLOQUEADO**
- SUSC-12C calibracao de proxy: **BLOQUEADO**
- Score v7: **BLOQUEADO**

O SUSC-14A conseguiu medir a rastreabilidade do bloqueio: ha registros oficiais
textuais, mas nao ha geometria oficial/rastreavel suficiente para avaliacao
patch-level. Zero geocodificacoes pode ser resultado fiel quando a camada oficial
disponivel nao cobre os logradouros das ocorrencias, e nao deve ser tratado como
falha automatica. O score v7 permanece bloqueado ate que existam links fortes ou
moderados suficientes, com data e geometria oficial.
