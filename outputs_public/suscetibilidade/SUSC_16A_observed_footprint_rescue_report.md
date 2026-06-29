# SUSC-16A - Footprint observacional preciso por dados locais, fontes oficiais e Sentinel/SAR

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`; `score_v7_created=false`.

O SUSC-16A substitui a tentativa de geocodificacao textual por uma estrategia de footprints observacionais, combinando geometrias locais, fontes oficiais/tecnicas e planejamento Sentinel/SAR. A etapa mantem todos os vinculos review-only, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

## 1. Por que 15A/15B bloquearam
SUSC-15A/15B mantiveram 4412 ocorrencias oficiais reais, mas nao encontraram lat/lon, poligono, lote, intersecao ou faixa numerica controlada suficiente para calibracao. O resultado permaneceu 0 eventos elegiveis e 0 links patch-evento precisos.

## 2. Por que mudar para footprint observacional
A geocodificacao textual chegou ao limite. O SUSC-16A troca a busca por coordenada textual por footprints observacionais: geometrias locais, fontes oficiais/tecnicas e planejamento Sentinel/SAR.

## 3. Dados locais do PROJETO minerados
Fontes locais registradas: 2301. Caminhos publicos foram sanitizados e nenhum bruto pesado foi copiado.

## 4. Geometrias locais encontradas
Candidatos parseados: 403. Candidatos elegiveis permanecem review-only e dependem de geometria direta e data/metodo documentados.

## 5. Fontes externas consultadas
Fontes externas registradas: 8. Todas ficaram bloqueadas para aquisicao automatica por falta de URL vetorial direta pequena e auditavel nesta execucao.

## 6. Footprints oficiais/tecnicos encontrados
Footprints oficiais/tecnicos materializados automaticamente: 12. Fontes externas sem vetor direto foram mantidas como manifestos bloqueados.

## 7. Plano Sentinel/SAR criado
Janelas Sentinel/SAR: 161. Targets Sentinel-1: 161. Stubs GEE/STAC foram criados sem autenticar ou baixar raster.

## 8. Canary SAR local executado ou bloqueado
Candidatos SAR locais: 3. O canary permanece bloqueado quando nao ha raster/array local com contrato de parsing seguro.

## 9. Catalogo de footprints
Linhas no catalogo unificado: 408. O catalogo unifica fontes locais, canary SAR e evidencias existentes SUSC-13C, sem ground truth.

## 10. Linkage footprint-patch
Links totais: 570. Links elegiveis: 65. Patches observacionais unicos: 62.

## 11. Score v6 x footprints
Media score v6: 0.457698. Mediana: 0.430251. hit@10=0.0; hit@20=0.0; hit@30=0.0.

## 12. Readiness
ready_for_16B_score_evaluation=True; ready_for_score_v7_discussion=False.

## 13. O que pode ser afirmado
Podemos afirmar que existe um pipeline auditavel para buscar footprints observacionais e cruzar candidatos com patches em modo review-only.

## 14. O que nao pode ser afirmado
Nao ha ground truth, treino supervisionado, ocorrencia operacional confirmada por patch, score v7, ou validacao automatica de bairro-only como calibracao.

## 15. Proximo marco
Se houver pelo menos 10 links elegiveis, executar SUSC-16B para avaliacao de score v6 com footprints. Caso contrario, priorizar aquisicao manual/controlada dos vetores externos bloqueados e execucao Sentinel/SAR fora do repo publico.
