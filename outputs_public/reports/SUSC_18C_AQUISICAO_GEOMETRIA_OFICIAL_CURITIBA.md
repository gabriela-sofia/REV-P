# SUSC-18C Aquisicao e resolucao de geometria oficial de ocorrencia em Curitiba

## Estado herdado do 18B

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `72dec09`
- Status final 18B herdado: 18B_BLOQUEIOS_REGIONAIS_REDUZIDOS_COM_FILAS_EXECUTAVEIS
- Status 17B herdado: 17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS
- `score_v6` alterado: False
- `score_v7` criado: False

## Por que Curitiba e o caminho mais curto para a segunda regiao forte

O evento de janeiro de 2022 (CUR_2022_01_15 / S17C_REF_0060) ja tem data oficial, fonte administrativa e
54 patches candidatos region-only identificados. Falta apenas a geometria de ocorrencia para executar o
overlay e, com features, virar referencia forte. Nenhum outro candidato regional esta tao proximo.

## Auditoria local

Auditoria consolidada de 6 entradas. O inventario de geometria de evento do trabalho de
protocolo anterior (v2ca) ja havia concluido, sem inventar nada, CONTEXT_ONLY_NO_GEOMETRY /
NO_LOCAL_GEOMETRY_OR_POINTS para os eventos de Curitiba. Nao ha ponto nem poligono de ocorrencia local.
O v2ca auditou 119 fontes locais, criou
86 bindings candidatos e manteve ready_for_overlay_count=
0. Ha, porem, 43 poligonos
oficiais de patch (EPSG:4326), prontos para o overlay assim que a geometria de ocorrencia chegar.

## Aquisicao oficial tentada

Fontes oficiais leves candidatas: GeoCuritiba, IPPUC, Defesa Civil de Curitiba e Portal de Dados Abertos.
Aquisicao offline-first por padrao (rede desabilitada; opcional via variavel de ambiente dedicada). Artefatos
leves adquiridos nesta execucao: 0.

## Geometrias encontradas ou ausentes

- Geometrias de ocorrencia resolvidas: 0.
- Status v2ca herdado: CONTEXT_ONLY_NO_GEOMETRY / CURITIBA_GEOMETRY_OR_POINT_EVIDENCE_NOT_READY.
- Sem geometria de ocorrencia, endereco/rua/bairro, centroide de bairro/municipio e area administrativa nao
  sao promovidos a geometria forte.

## Vinculos patch

- Patch-links de Curitiba gerados: 4 (fortes: 0).
- Patches candidatos region-only no prelink para CUR_2022_01_15: 54.
- Quando `geometrias_ocorrencia_resolvidas=0`, os patch-links gerados sao linhas de controle
  nao fortes: `insufficient_for_patch_link`, sem `geometry_id` real e sem `patch_id` adjudicado.
  Eles existem para manter o contrato tabular e explicar o bloqueio, nao para afirmar overlay.
- A maquinaria de overlay (poligonos reais dos patches + teste ponto-em-poligono) esta pronta e validada;
  produz vinculo forte assim que houver geometria de ocorrencia.

## Features disponiveis

- Curitiba ainda sem features diretas por patch; extracao em fila (0 tarefas),
  condicionada ao overlay.

## Referencia observacional Curitiba

- S18C_REF_0001 S17C_REF_0012 (not_available): bloqueado_sem_geometria
- S18C_REF_0002 S17C_REF_0060 (2022-01-15..2022-01-16): bloqueado_sem_geometria
- S18C_REF_0003 S17C_REF_0061 (2023-10-28..2023-10-30): bloqueado_sem_geometria
- S18C_REF_0004 S17C_REF_0062 (2024-02-18..2024-02-20): bloqueado_sem_geometria

## Solicitacao formal

Pacote formal executavel emitido: oficio de solicitacao (`solicitacao_geometria_ocorrencia_curitiba.md`),
schema de ingestao (`schema_resposta_esperada_curitiba.json`) e planilha modelo de resposta
(`modelo_planilha_resposta_curitiba.csv`). A resposta oficial, uma vez ingerida, aciona a maquinaria completa
em uma unica execucao.

## Gate Curitiba

- geometria_oficial_de_ocorrencia_encontrada: passou=false (false / true)
- patch_link_forte_criado: passou=false (0 / >=1)
- features_diretas_disponiveis: passou=false (false / true)
- curitiba_segunda_regiao_forte: passou=false (false / true)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)

## Gate 17B pos-18C

- minimo_3_eventos_distintos_fortes: passou=false (1 / 3)
- minimo_2_regioes_fortes: passou=false (1 / 2)
- minimo_20_patch_links_fortes: passou=false (5 / 20)
- separacao_temporal_possivel: passou=false (false / true)
- features_diretas_suficientes: passou=true (true / true)
- controles_nao_supervisionados: passou=true (true / true)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)

- Status 17B: **17B_BLOQUEADO_POR_GEOMETRIA**
- Nenhum benchmark 17B foi criado.

## Conclusao

- O que avancou de verdade: a maquinaria completa de geometria -> overlay -> vinculo -> referencia foi
  construida e validada com os poligonos oficiais reais dos patches de Curitiba, e um pacote formal
  executavel de aquisicao foi emitido. O bloqueio de Curitiba esta reduzido a um unico insumo: a geometria
  oficial de ocorrencia.
- O que segue bloqueado: sem esse insumo, Curitiba ainda nao vira segunda regiao forte; o 17B permanece
  bloqueado por geometria.
- Proxima acao pesada: **SUSC-18D**: protocolar a solicitacao formal e, ao receber a camada oficial de
  ocorrencia, ingeri-la para acionar overlay e extracao de features, consolidando Curitiba como segunda
  regiao forte somente revisao.

## Garantias

- ground_truth=false; eligible_for_training=false; score_v7_allowed=false; review_only=true.
- score_v6 intacto (False para alterado); nenhum benchmark 17B criado; nenhuma coordenada inventada.
