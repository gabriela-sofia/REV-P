# SUSC-17I Ampliacao regional da amostra observacional

## Estado herdado do 17H

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `4bd01ae`
- Status final 17H herdado: 17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA
- Status 17B herdado: 17B_AINDA_SEM_PRONTIDAO_BENCHMARK_AMOSTRA_LOCAL
- `score_v6` alterado: False
- `score_v7` criado: False

## Por que a ampliacao regional e necessaria

A calibracao forte do 17H usou apenas Recife (1 evento, 1 regiao, 5 vinculos). O 17B exige mais
eventos, mais regioes e separacao temporal. Esta etapa amplia para Curitiba e Petropolis.

## Inventario Curitiba/Petropolis

- Candidatos totais: 41 (Curitiba 11, Petropolis 30).

- CUR (Curitiba): candidatos=11; fortes=0; parciais=2; contextuais=0; bloqueados=9
- PET (Petropolis): candidatos=30; fortes=0; parciais=0; contextuais=1; bloqueados=29

## Candidatos encontrados

- S17I_ITEM_0038 (PET): S17C_REF_0059 | inundacao | 2024-03-21..2024-03-28 | candidato_contextual
- S17I_ITEM_0039 (CUR): S17C_REF_0060 | inundacao | 2022-01-15..2022-01-16 | candidato_observacional_parcial
- S17I_ITEM_0041 (CUR): S17C_REF_0062 | inundacao | 2024-02-18..2024-02-20 | candidato_observacional_parcial

## Candidatos fortes/parciais/contextuais

- Fortes: 0
- Parciais: 2
- Contextuais: 1
- Bloqueados sem geometria: 1
- Bloqueados sem data: 1
- Bloqueados por fenomeno misto: 36

## Lacunas por regiao

- CUR / inundacao: 4
- CUR / misto: 7
- PET / inundacao: 1
- PET / misto: 29

Curitiba nao tem geometria oficial nem, na maioria, data exata; os eventos ficam como parciais,
contextuais ou em fila de obtencao de geometria/data. Petropolis tem pontos oficiais datados de
fevereiro de 2022, mas o fenomeno e predominantemente misto (deslizamento e inundacao juntos), o
que exige separacao explicita antes de usar como evidencia de inundacao.

## Comparacao com Recife

Recife tem o unico conjunto com geometria forte (footprint tecnico) e vinculo forte (5 canarios).
Curitiba e Petropolis ainda nao alcancam geometria forte nem vinculo forte de inundacao.

## Gate final 17I

- regioes_processadas: passou=true (2 / 2)
- regioes_com_evidencia_observacional: passou=true (3 / 2)
- eventos_distintos_com_geometria_forte: passou=false (1 / 3)
- vinculos_patch_fortes: passou=false (5 / 20)
- candidatos_fortes_curitiba: passou=true (0 / >=0)
- candidatos_fortes_petropolis: passou=true (0 / >=0)
- candidatos_parciais: passou=true (2 / >=0)
- fila_executavel_criada: passou=true (true / true)
- separacao_temporal_possivel: passou=false (false / true)
- features_completas_por_evento: passou=false (0 / >=1)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_allowed_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)
- caminho_funcional_entregue: passou=true (true / true)
- status_final_17i: passou=true (17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS / enum)

- Status final: **17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS**
- Caminho funcional: **candidatos_parciais_mais_fila_executavel**

## Status 17B pos-17I

- minimo_3_eventos_distintos: passou=false (1 / 3)
- minimo_2_regioes: passou=true (3 / 2)
- minimo_20_vinculos_fortes: passou=false (5 / 20)
- 1_evento_por_regiao_com_data_e_geometria: passou=false (1 / 2)
- separacao_temporal_possivel: passou=false (false / true)
- controles_nao_supervisionados: passou=true (true / true)
- ground_truth_false: passou=true (0 / 0)
- trainable_false: passou=true (0 / 0)

- Status 17B: **17B_APROXIMACAO_COM_AMOSTRA_REGIONAL_PARCIAL**
- Nenhum benchmark 17B foi criado.

## Por que segue sem ground truth

Nenhum candidato confirma ocorrencia no patch; sem verdade de referencia observacional nao ha ground truth.

## Por que segue sem treino

Sem rotulo validado, nenhum candidato alimenta treino supervisionado.

## Por que segue sem score_v7

Nenhum score oficial ou score_v7 e criado; o score_v6 permanece intacto.

## Proximo marco recomendado

SUSC-17J: executar a fila regional (obter geometria oficial datada em Curitiba e separar o fenomeno
de inundacao em Petropolis), resolver vinculos de patch e extrair features, ampliando a amostra
forte para pelo menos duas regioes, sempre somente revisao e sem ground truth.
