# revp_temporal_source_sensor_provenance_mv1

## 1. Escopo
Auditoria publica, offline e deterministica de proveniencia sensor-fonte para assets temporais, DINOv2 e unknown.

## 2. Relacao com o bloqueio da resolucao de sensor
- A etapa anterior ficou bloqueada porque nenhum slot tinha familia Sentinel-2 resolvida.
- Status desta auditoria: `STEP_A_SOURCE_PROVENANCE_PARTIAL_ONLY`.

## 3. Por que DINOv2 nao e sensor de aquisicao
- DINOv2 e derivado. Ele so pode carregar proveniencia Sentinel quando um manifesto explicito conecta o embedding ao asset fonte.
- Mesmo com link explicito, o embedding nao vira sensor fisico de aquisicao.

## 4. Arquivos de entrada
- `outputs_public/tables/revp_temporal_sensor_family_resolution_mv1.csv`
- `outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv`
- `outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv`
- `outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv`
- `outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv`
- `outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv`

## 5. Regras de proveniencia aceitas
- Campo direto de source/input asset.
- Campo direto de produto, cena, tile ou familia Sentinel.
- Manifesto DINO que declare input Sentinel com patch e source asset unicos.
- Padrao Sentinel inequivoco quando nao ha conflito.

## 6. Regras de rejeicao
- Contexto fraco ou diretorio generico nao resolve sensor.
- Sensor ausente, data fonte ausente, conflito S1/S2 ou unknown permanecem bloqueados.
- DINOv2 sem fonte Sentinel explicita permanece bloqueado.

## 7. Source sensors resolvidos
- Total resolvido: 0
- Sentinel-2: 0
- Sentinel-1: 0

## 8. DINO/unknowns bloqueados
- DINOv2 com source explicito: 91
- DINOv2 sem source explicito: 4231
- Unknowns resolvidos: 0
- Unknowns bloqueados: 735

## 9. Impacto potencial sobre requirements/requests
- Slots que poderiam ser desbloqueados: 0
- Patches que poderiam virar acionaveis: 0

## 10. Batches recuperaveis ou inexistentes
- 1 patch: False
- 20 patches: False
- 30 patches: False

## 11. Proximos passos permitidos dentro da Vertente A
- Recuperar manualmente source sensor family e acquisition date somente a partir de evidencia explicita.
- Reconstruir resolucao de sensor apenas se os vinculos fonte ficarem unicos e auditaveis.

## 12. Limitacoes
- Esta auditoria nao consulta backend externo e nao baixa dados.
- Esta auditoria nao cria datas e nao corrige requirements.

## 13. Guardrails preservados
- Branch auditada: `analysis/temporal-asset-readiness-mv1`
- sem download de assets
- sem consulta STAC API ou GEE
- sem gerar URLs de download
- sem criar data real ou sintetica
- sem calcular embedding drift
- sem gerar embeddings novos
- sem treino
- sem rotulo
- sem criar evidencia de ausencia supervisionada
- sem classe-zero
- sem GT operacional
- sem tabela multimodal de atributos
- sem alterar manifestos originais
- sem alterar outputs anteriores
