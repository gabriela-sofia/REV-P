# SUSC-18A Execucao de referencia observacional regional

## Estado herdado do 17I

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `72dec09`
- Status final 17I herdado: 17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS
- Status 17B herdado: 17B_APROXIMACAO_COM_AMOSTRA_REGIONAL_PARCIAL
- `score_v6` alterado: False
- `score_v7` criado: False

## Motivacao da execucao

O 17I mapeou a amostra regional, mas parou no diagnostico com filas executaveis. Esta etapa
executa de fato essas filas e consolida a base regional observacional: reune a referencia forte
de Recife com as features diretas ja extraidas localmente, preserva as referencias parciais
datadas de Curitiba e transforma o que falta em pacotes de execucao concretos (geometria,
separacao de fenomeno e footprint SAR), sem inventar dado.

## Metodologia

Padrao por candidato: registro de evento -> geometria oficial ou footprint tecnico -> vinculo com
patch -> features diretas (somente pre-evento) -> avaliacao somente revisao -> gate 17B. Onde falta
geometria, data, separacao de fenomeno ou raster, gera-se fila executavel com fonte, formato,
campos e comando. Bairro/rua/texto nunca vira geometria forte; alerta/area de risco nunca vira
ocorrencia; feature pos-evento nunca vira feature pre-evento.

## Resultado Recife (referencia)

Recife entrega 5 referencias fortes review-only (canarios), cada uma com
footprint tecnico, vinculo forte (exact_polygon_overlap, herdado do 17D) e features diretas locais
(fisico do 17G, espectral pre-evento do 17C20, chuva pre-evento do 17C25):

- S18A_ITEM_0001 S17C6_CANARY_REC_00001 | vinculo exact_polygon_overlap | fisico+espectral+chuva locais
- S18A_ITEM_0002 S17C6_CANARY_REC_00002 | vinculo exact_polygon_overlap | fisico+espectral+chuva locais
- S18A_ITEM_0003 S17C6_CANARY_REC_00003 | vinculo exact_polygon_overlap | fisico+espectral+chuva locais
- S18A_ITEM_0004 S17C6_CANARY_REC_00004 | vinculo exact_polygon_overlap | fisico+espectral+chuva locais
- S18A_ITEM_0005 S17C6_CANARY_REC_00005 | vinculo exact_polygon_overlap | fisico+espectral+chuva locais

## Resultado Curitiba

Curitiba tem eventos de inundacao datados por fonte oficial, mas nenhuma geometria oficial local.
Nenhum ponto/poligono/shapefile foi encontrado nos artefatos; endereco/rua nao vira geometria forte.
Os eventos datados seguem como referencia parcial e entram na fila de obtencao de geometria
(3 tarefas).

## Resultado Petropolis

Petropolis tem muitos registros, mas predominantemente de fenomeno misto/deslizamento (laudos
geotecnicos de encosta). Sem separacao por ocorrencia, nao podem entrar como inundacao. Apenas o
evento declarado de inundacao segue como evidencia contextual. Gerou-se fila de separacao de
fenomeno (29 tarefas).

## Geometrias encontradas e ausentes

- Recife: footprint tecnico (geometria tecnica) disponivel review-only.
- Curitiba/Petropolis: sem geometria oficial forte local; todas em fila.

- REC (Recife): itens=5; fortes=5; parciais=0; contextuais=0; bloqueados/fila=0
- CUR (Curitiba): itens=11; fortes=0; parciais=2; contextuais=0; bloqueados/fila=9
- PET (Petropolis): itens=30; fortes=0; parciais=0; contextuais=1; bloqueados/fila=29

## Fenomeno por regiao

- REC / inundacao_alagamento_enxurrada: 5
- CUR / inundacao_alagamento_enxurrada: 4
- CUR / fenomeno_misto: 7
- PET / inundacao_alagamento_enxurrada: 1
- PET / fenomeno_misto: 29

## Footprints produzidos ou enfileirados

- Recife: footprint tecnico review-only herdado (nao e ground truth, nao e feature pre-evento).
- Curitiba/Petropolis: footprint SAR em fila de execucao (4 tarefas), dependente de AOI oficial.

## Vinculos patch

- Patch-links fortes review-only: 5 (Recife).
- Curitiba/Petropolis: same_region_only (nao forte) ate obter geometria.

## Features disponiveis

- Referencia forte de Recife com fisico, espectral e chuva locais (somente pre-evento).
- Curitiba/Petropolis: extracao de features em fila (9 tarefas), apos geometria/patch.

## Referencias parciais e contextuais

- S18A_ITEM_0044 (CUR) S17C_REF_0060 | 2022-01-15..2022-01-16 | referencia_observacional_parcial
- S18A_ITEM_0046 (CUR) S17C_REF_0062 | 2024-02-18..2024-02-20 | referencia_observacional_parcial
- S18A_ITEM_0043 (PET) S17C_REF_0059 | 2024-03-21..2024-03-28 | evidencia_contextual

## Gate de prontidao 17B pos-18A

- minimo_3_eventos_distintos_fortes: passou=false (1 / 3)
- minimo_2_regioes_com_referencia_forte: passou=false (1 / 2)
- regioes_com_evidencia_observacional: passou=true (3 / >=2)
- minimo_20_patch_links_fortes: passou=false (5 / 20)
- separacao_temporal_possivel: passou=false (false / true)
- features_diretas_suficientes: passou=true (true / true)
- controles_nao_supervisionados: passou=true (true / true)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)

- Status 17B: **17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS**
- Nenhum benchmark 17B foi criado.

## Lacunas restantes

- Curitiba: geometria oficial datada por evento.
- Petropolis: separacao de fenomeno por ocorrencia (inundacao x deslizamento).
- Todas as regioes: footprint SAR depende de AOI oficial; features regionais dependem de patch-link.

## Conclusao

- O que avancou de verdade: a referencia forte de Recife foi consolidada com features diretas locais
  (fisico/espectral/chuva) e 5 patch-links fortes review-only; Curitiba
  manteve 2 referencias parciais datadas; e todas as lacunas viraram
  filas executaveis concretas (54 tarefas no total).
- O que segue bloqueado: geometria oficial de Curitiba, separacao de fenomeno de Petropolis e a
  ampliacao para mais de uma regiao/evento forte; por isso o 17B permanece em aproximacao regional.
- Proximo marco recomendado: **SUSC-18B** — executar a fila de geometria de Curitiba e a separacao
  de fenomeno de Petropolis, resolver patch-links e extrair features diretas, buscando a segunda
  regiao com referencia forte, sempre somente revisao, sem ground truth, treino ou score_v7.

## Garantias

- ground_truth=false; eligible_for_training=false; score_v7_allowed=false; review_only=true.
- score_v6 intacto (False para alterado); nenhum benchmark 17B criado.
