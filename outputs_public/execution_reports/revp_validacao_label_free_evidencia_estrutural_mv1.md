# Validação label-free e evidência estrutural MV1

## 1. Escopo do marco MV1
Este marco consolida uma análise pública, auditável e label-free dos embeddings DINOv2 congelados, da topologia entre cidades/regiões e da evidência administrativa/contextual como probe externo.

## 2. Por que esta branch é um turning point
A branch `marco/validacao-label-free-evidencia-estrutural-mv1` reúne a virada metodológica: restauração forense v2dz-v2ef, auditoria temporal, bloqueio da Trilha A e adoção da Trilha B como eixo científico imediato.

## 3. Estado herdado da restauração `v2dz-v2ef`
A restauração `v2dz-v2ef` é tratada como trilha forense e não como validação operacional. Ela não cria labels, negativos formais ou ground truth operacional.

## 4. Estado herdado da auditoria temporal
A auditoria temporal registrou `METADADOS_TEMPORAIS_INSUFICIENTES_TRILHA_B_RECOMENDADA`. O estado herdado é de metadados temporais insuficientes para deslocamento temporal amplo.

## 5. Por que a Trilha A está bloqueada agora
A Trilha A depende de séries temporais úteis e metadados de nuvem por patch. O estado atual não oferece esse volume de metadados para o corpus Sentinel/DINO.

## 6. Por que a Trilha B é a trilha imediata
A Trilha B permite avaliar organização estrutural label-free, vizinhança de embeddings e alinhamento exploratório com evidência administrativa/contextual sem transformar esses sinais em label.

## 7. Entradas encontradas
- `outputs_public/tables/table_dino_embedding_inventory.csv`
- `outputs_public/tables/table_dino_similarity_matrix.csv`
- `outputs_public/tables/table_dino_nearest_neighbors.csv`
- `outputs_public/tables/table_external_evidence_summary.csv`
- `outputs_public/tables/protocol_c_cross_region_evidence_scorecard.csv`
- `local_runs/dino_embeddings/v1gv/evidence_coverage_matrix_v1gv.csv`
- `local_runs/dino_embeddings/v1ge/dino_expanded_embedding_manifest_v1ge.csv`
- `local_runs/dino_embeddings/v1ge/dino_expanded_embedding_summary_v1ge.json`
- `outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json`
- `outputs_public/execution_reports/revp_restauracao_manual_v2dz_v2ef.md`

## Entradas ausentes

## 8. Embeddings encontrados e critérios de validade
Foram encontrados 12 embeddings rastreáveis e 12 embeddings válidos. Critérios: patch identificado, dimensão válida, hash presente, matriz de similaridade disponível, sem label e sem alvo supervisionado.

## 9. Metodologia label-free
A metodologia usa distância de cosseno derivada da matriz pública de similaridade, medoids por cidade/região/global, vizinho mais próximo e cobertura contextual como probe externo.

## 10. Topologia entre cidades/regiões
Distância média intra-cidade: `0.287495`. Distância média inter-cidades: `0.309723`. Esses valores descrevem organização estrutural, não classe.

## 11. Vizinhos mais próximos
Consistência de vizinho na mesma cidade: `0.416667`. Consistência na mesma região: `0.416667`.

## 12. Evidência administrativa/contextual como probe externo
12 amostras têm algum indicador contextual disponível. O campo `score_evidencia_eh_label` é sempre `false`.

## 13. Checagem de sanidade de domínio do DINOv2
Com n=12, a checagem é apenas piloto. Há sinal estrutural mensurável por vizinhança e distâncias, mas a suficiência amostral não permite conclusão forte.

## 14. Limitações
- Corpus pequeno para validação ampla.
- Vetores brutos permanecem locais; a saída pública usa métricas derivadas e hashes.
- Evidência administrativa/contextual não é fonte de label operacional.
- Sem positivos formais, negativos formais ou ground truth operacional.

## 15. Decisão metodológica
`PILOTO_LABEL_FREE_COM_LIMITACAO_AMOSTRAL`.

## 16. Guardrails preservados
- sem treino supervisionado
- sem label binário
- sem positivo formal
- sem negativo formal
- sem ground truth operacional
- unknown não vira negativo
- Curitiba não vira negativo formal
- DINOv2 não prova inundação
- evidência administrativa não vira label
- restauração v2dz-v2ef não vira validação operacional

## 17. Próximos passos
Expandir o corpus DINOv2 rastreável, manter a Trilha B como eixo imediato, cruzar topologia estrutural com evidência contextual apenas como priorização de revisão humana e não liberar treino supervisionado.
