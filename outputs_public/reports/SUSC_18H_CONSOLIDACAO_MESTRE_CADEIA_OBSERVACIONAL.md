# SUSC-18H - Consolidacao mestre da cadeia observacional

## Resumo executivo

Esta etapa e uma consolidacao metodologica. Nao cria nova geometria, novo score,
novo benchmark nem novo rotulo. Ela fotografa o estado real da cadeia 17C ate 18G,
organiza a evidencia em matrizes auditaveis e define a proxima frente estrutural.

Guardrails permanentes: ground_truth = falso, treino desabilitado, score_v7 nao
permitido, benchmark 17B nao criado e score_v6 intacto.

## Historico 17C ate 18G

| Marco | Titulo | Status |
| --- | --- | --- |
| SUSC-17C | Aquisicao de referencia forte e canarios observacionais | `17C_REFERENCIA_FORTE_ADQUIRIDA_SEM_PATCH_LINK_SUFICIENTE` |
| SUSC-17C5 | Resolucao de vinculo geometria para patch | `17C5_PATCH_LINKS_PARCIAIS_SEM_VINCULO_OFICIAL_FORTE` |
| SUSC-17D | Validacao tecnica da evidencia observacional | `17D_VALIDACAO_TECNICA_SEM_PRONTIDAO_BENCHMARK` |
| SUSC-17E | Prontidao e calibracao observacional exploratoria | `17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY` |
| SUSC-17F | Extracao fisica topografica dos canarios | `17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO` |
| SUSC-17G | Extracao direta de features fisicas dos canarios | `17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL` |
| SUSC-17H | Calibracao observacional forte somente revisao | `17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA` |
| SUSC-17I | Ampliacao regional da amostra observacional | `17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS` |
| SUSC-18A | Execucao da referencia observacional regional | `18A_REFERENCIA_REGIONAL_FORTE_MAIS_FILAS_EXECUTAVEIS` |
| SUSC-18B | Execucao de geometrias regionais e separacao de fenomeno | `18B_BLOQUEIOS_REGIONAIS_REDUZIDOS_COM_FILAS_EXECUTAVEIS` |
| SUSC-18C | Aquisicao de geometria oficial de Curitiba | `18C_CURITIBA_GEOMETRIA_AUSENTE_COM_SOLICITACAO_FORMAL` |
| SUSC-18D | Protocolo externo de Curitiba com solicitacao formal | `18D_AGUARDANDO_RESPOSTA_OFICIAL` |
| SUSC-18E | Footprint tecnico SAR de Curitiba | `18E_PACOTE_GEE_SENTINEL1_PRONTO` |
| SUSC-18E2 | Execucao controlada Sentinel-1 de Curitiba | `18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO` |
| SUSC-18F | Ingestao e validacao do footprint SAR de Curitiba | `18F_REFERENCIA_TECNICA_SAR_CURITIBA_PARCIAL_POR_PATCH_STATS` |
| SUSC-18G | Recuperacao e compactacao vetorial SAR de Curitiba | `18G_REFERENCIA_TECNICA_SAR_FORTE_COM_OVERLAY` |

## Estado por regiao

- **Recife**: 1 evento com 5 canarios fortes somente revisao; features fisicas
  diretas, espectrais e de chuva; calibracao forte review-only com divergencia
  fisica preservada. Limite: uma unica regiao e um unico evento.
- **Curitiba**: segunda regiao tecnica via SAR Sentinel-1 (patch_stats real de 43
  linhas e footprint tecnico compacto com 2 overlays em CUR_01050 e CUR_01101).
  A geometria oficial de ocorrencia segue ausente; o pacote formal 18D aguarda
  resposta.
- **Petropolis**: fenomeno misto (deslizamento e inundacao) sem separacao e sem
  geometria forte; 1 candidato contextual de 2024. Nao promovido.

## Estado por evidencia

A qualidade da evidencia foi classificada em tiers de A ate G. O canario tecnico
de Recife e o footprint SAR de Curitiba entram como tier C (footprint tecnico
somente revisao). Curitiba oficial e Petropolis misto permanecem em tier G
(insuficiente) ate haver geometria e separacao de fenomeno.

## Status do 17B

Estado mestre: `17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA`. Os cenarios avaliados (oficial
estrito, tecnico review-only, misto, futuro com resposta oficial de Curitiba e
futuro com Petropolis separado) confirmam que os minimos (3 eventos, 2 regioes e
20 vinculos fortes) ainda nao foram atingidos. O 17B nao foi criado.

## Pendencias priorizadas

- **P0**: aguardar e ingerir a resposta oficial de Curitiba (18D); recuperar e
  validar os exports SAR finais; manter patch_stats separado de footprint.
- **P1**: consolidar a interpretacao dos 2 overlays SAR; buscar geometria oficial
  de Curitiba.
- **P2**: separar o fenomeno de Petropolis; buscar evento 2024 com geometria;
  ampliar eventos de Recife.
- **P3**: construir a matriz multimodal escalavel por patch e auditar cobertura.

## Decisao estrategica

1. Parar de depender do footprint como base central.
2. Usar footprint e SAR como canario de evidencia, nao como fundamento.
3. Avancar para a matriz multimodal escalavel por patch como novo eixo estrutural.

## Proximos marcos recomendados

1. **SUSC-18I** - Consolidacao tecnica SAR de Curitiba pos-vetor.
2. **SUSC-19A** - Matriz multimodal escalavel por patch (eixo principal).
3. **SUSC-19B / 19C / 19D** - Cobertura multimodal, avaliacao review-only e
   comunicacao cientifica.

Nenhum desses marcos propoe score_v7.

## Plano de commit seletivo

Entram no commit o codigo, o schema, os testes e as saidas publicas do 18H, alem
da atualizacao de allowlist do `.gitignore`. Nao entram `local_runs/`, rasters,
embeddings nem pastas de marco fora da allowlist. O detalhamento esta em
`plano_commit_seletivo_pos_18g.md`.
