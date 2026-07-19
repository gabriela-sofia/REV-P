# SUSC-18B Execucao de geometrias regionais e separacao de fenomeno

## Estado herdado do 18A

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `642eae9`
- Status final 18A herdado: 18A_REFERENCIA_REGIONAL_FORTE_MAIS_FILAS_EXECUTAVEIS
- Status 17B herdado: 17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS
- `score_v6` alterado: False
- `score_v7` criado: False

## Objetivo do 18B

Executar as filas regionais mais importantes do 18A para tentar transformar Curitiba ou Petropolis em
segunda regiao com referencia observacional forte somente revisao: resolver geometria, separar fenomeno,
preparar footprint tecnico, resolver vinculos e localizar features, sem criar ground truth, treino,
score_v7 ou benchmark 17B.

## Execucao Curitiba

Busca local exaustiva por geometria de ocorrencia (csv, geojson, shapefile, parquet, xlsx e registries de
protocolo). Achado: o evento datado tem apenas ancoras hidrometeorologicas (timing) e patches candidatos
em nivel regional (region-only, sem overlay); nao ha camada de ocorrencia local. Endereco/rua/bairro e
centroide de municipio nao viram geometria forte.

- S17C_REF_0012 (not_available): rejeitado | patches region-only=0
- S17C_REF_0060 (2022-01-15..2022-01-16): geometria_ausente_com_tarefa_externa | patches region-only=54
- S17C_REF_0061 (2023-10-28..2023-10-30): geometria_textual_bloqueada | patches region-only=0
- S17C_REF_0062 (2024-02-18..2024-02-20): geometria_ausente_com_tarefa_externa | patches region-only=0

Geometrias de ocorrencia resolvidas e normalizadas: 0.

## Execucao Petropolis

Separacao de fenomeno executada item a item. Os laudos CPRM de 2022 sao geotecnicos de encosta (desastre
hidrometeorologico misto), sem separacao por ocorrencia; apenas o evento de inundacao de 2024 pode seguir
como evidencia contextual, e ainda sem geometria.

- inundacao_alagamento_enxurrada: 1
- fenomeno_misto: 29

Candidatos de inundacao separados: 1; bloqueados por fenomeno:
29.

## Geometrias encontradas e ausentes

- Recife: footprint tecnico (geometria tecnica) disponivel review-only.
- Curitiba/Petropolis: sem geometria oficial de ocorrencia local; todas em fila externa especifica.

- REC (Recife): itens=5; fortes=5; parciais=0; contextuais=0; bloqueados/fila=0
- CUR (Curitiba): itens=11; fortes=0; parciais=2; contextuais=0; bloqueados/fila=9
- PET (Petropolis): itens=30; fortes=0; parciais=0; contextuais=1; bloqueados/fila=29

## Fenomenos separados e bloqueados

- Petropolis: 1 inundacao separada, 29 bloqueados (misto/deslizamento), fila de separacao com 29 tarefas.

## Footprints tecnicos produzidos ou enfileirados

- Recife: footprint tecnico review-only herdado (nao e ground truth, nao e feature pre-evento).
- Curitiba/Petropolis: footprint SAR em fila (4 tarefas), dependente de AOI oficial.

## Vinculos patch gerados

- Patch-links fortes review-only: 5 (Recife).
- Curitiba/Petropolis: same_region_only (nao forte); Curitiba com patches candidatos region-only nomeados.

## Features disponiveis

- Referencia forte de Recife com fisico, espectral e chuva locais (somente pre-evento):

- S18B_ITEM_0001 S17C6_CANARY_REC_00001 | exact_polygon_overlap | fisico+espectral+chuva locais
- S18B_ITEM_0002 S17C6_CANARY_REC_00002 | exact_polygon_overlap | fisico+espectral+chuva locais
- S18B_ITEM_0003 S17C6_CANARY_REC_00003 | exact_polygon_overlap | fisico+espectral+chuva locais
- S18B_ITEM_0004 S17C6_CANARY_REC_00004 | exact_polygon_overlap | fisico+espectral+chuva locais
- S18B_ITEM_0005 S17C6_CANARY_REC_00005 | exact_polygon_overlap | fisico+espectral+chuva locais

- Curitiba/Petropolis: extracao de features em fila (9 tarefas), apos geometria/patch.

## Comparacao com Recife

Recife segue como unica regiao forte (1 evento, 5 vinculos fortes, features
completas). Curitiba esta a um passo (evento datado oficial, falta apenas geometria de ocorrencia e
overlay). Petropolis esta a dois passos (falta separar fenomeno e obter geometria). Detalhe em
`comparacao_regional_recife_curitiba_petropolis.csv`.

## Gate de prontidao 17B pos-18B

- minimo_3_eventos_distintos_fortes: passou=false (1 / 3)
- minimo_2_regioes_fortes: passou=false (1 / 2)
- regioes_com_evidencia_observacional: passou=true (3 / >=2)
- minimo_20_patch_links_fortes: passou=false (5 / 20)
- separacao_temporal_possivel: passou=false (false / true)
- features_diretas_suficientes: passou=true (true / true)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)

- Status 17B: **17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS**
- Nenhum benchmark 17B foi criado.

## Conclusao

- O que avancou de verdade: a execucao localizou com precisao o bloqueio de Curitiba (falta somente a
  geometria de ocorrencia; patches candidatos region-only ja identificados) e executou a separacao de
  fenomeno de Petropolis, isolando o unico candidato de inundacao. Foram emitidas filas executaveis
  especificas (45 tarefas: 3 geometria Curitiba,
  29 separacao Petropolis, 4 footprint SAR,
  9 features).
- O que segue bloqueado: nenhuma segunda regiao virou forte com dado local; falta geometria de ocorrencia
  de Curitiba e separacao/geometria de Petropolis. O 17B permanece em aproximacao regional com referencias
  parciais.
- Proxima acao pesada: **SUSC-18C** — obter a geometria oficial de ocorrencia de Curitiba (camada GeoCuritiba/
  IPPUC ou solicitacao a Defesa Civil) e, com ela, executar o overlay de patch e a extracao de features
  diretas pre-evento, buscando a segunda regiao forte, sempre somente revisao e sem ground truth.

## Garantias

- ground_truth=false; eligible_for_training=false; score_v7_allowed=false; review_only=true.
- score_v6 intacto (False para alterado); nenhum benchmark 17B criado.
