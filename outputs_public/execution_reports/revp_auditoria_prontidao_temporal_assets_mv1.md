# Auditoria de prontidão temporal de assets MV1

## Escopo
Auditoria apenas de metadados da prontidão temporal de assets Sentinel/DINO para decidir a próxima trilha MV1: deslocamento temporal label-free de embeddings ou correlação topológica/administrativa entre cidades.

## Branch
analise/auditoria-prontidao-temporal-assets-mv1

## Arquivos de entrada encontrados
- `manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv`
- `outputs_public/tables/table_dino_embedding_inventory.csv`
- `datasets/protocolo_c/v2ap_sentinel_asset_inventory.csv`
- `datasets/protocolo_c/v2aa_patch_date_candidate_consolidation.csv`
- `datasets/protocolo_c/v2aa_sentinel_filename_date_extraction.csv`
- `datasets/protocolo_c/v2ag_sentinel_date_linkability_audit.csv`
- `datasets/protocolo_c/v2ag_event_patch_temporal_preview.csv`
- `datasets/official_anchor_sentinel_patch_registry.csv`
- `datasets/official_anchor_sentinel_patch_quality_registry.csv`
- `datasets/event_patch_linkage_registry.csv`

## Arquivos de entrada ausentes

## Campos usados
- `datasets/event_patch_linkage_registry.csv`: patch_candidate_id, pre_scene_date, post_scene_date
- `datasets/official_anchor_sentinel_patch_quality_registry.csv`: reference_patch_id, scene_date, local_cloud_fraction, cloud_metadata_global
- `datasets/official_anchor_sentinel_patch_registry.csv`: reference_patch_id, region, scene_date, cloud_cover_metadata
- `datasets/protocolo_c/v2aa_patch_date_candidate_consolidation.csv`: patch_id, region, candidate_dates, selected_sentinel_date, sentinel_date_recovered
- `datasets/protocolo_c/v2aa_sentinel_filename_date_extraction.csv`: patch_id, extracted_date, extraction_status
- `datasets/protocolo_c/v2ag_event_patch_temporal_preview.csv`: patch_id, preview_sentinel_date, preview_status
- `datasets/protocolo_c/v2ag_sentinel_date_linkability_audit.csv`: patch_id, recovered_date, can_link_sentinel_date
- `datasets/protocolo_c/v2ap_sentinel_asset_inventory.csv`: patch_id_detected, region_detected, date_detected, safe_to_use_as_crosswalk_evidence
- `manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv`: canonical_patch_id, region, dino_input_id, source_asset_id
- `outputs_public/tables/table_dino_embedding_inventory.csv`: patch_id, region, dino_input_id

## Campos ausentes ou insuficientes
- `datasets/event_patch_linkage_registry.csv`: cobertura_nuvens
- `datasets/protocolo_c/v2aa_patch_date_candidate_consolidation.csv`: cobertura_nuvens
- `datasets/protocolo_c/v2aa_sentinel_filename_date_extraction.csv`: regiao, cobertura_nuvens
- `datasets/protocolo_c/v2ag_event_patch_temporal_preview.csv`: regiao, cobertura_nuvens
- `datasets/protocolo_c/v2ag_sentinel_date_linkability_audit.csv`: regiao, cobertura_nuvens
- `datasets/protocolo_c/v2ap_sentinel_asset_inventory.csv`: cobertura_nuvens
- `manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv`: data_aquisicao, cobertura_nuvens
- `outputs_public/tables/table_dino_embedding_inventory.csv`: data_aquisicao, cobertura_nuvens

## Contagens por número de datas
- 1 data: 7
- 2 datas: 1
- 3 ou mais datas: 0

## Contagens por região
- Curitiba: 43
- Petrópolis: 50
- Petrópolis;desconhecido: 2
- Recife: 39
- Recife;desconhecido: 2
- desconhecido: 28

## Patches elegíveis
- Elegíveis para deslocamento temporal: 0

## Decisão A/B
- Decisão global: `METADADOS_TEMPORAIS_INSUFICIENTES_TRILHA_B_RECOMENDADA`
- Critério aplicado: a trilha A exige volume suficiente de patches com 3 ou mais datas limpas e metadados de nuvem rastreáveis; quando o universo Sentinel/DINO principal não possui metadados temporais/de nuvem suficientes, a trilha B permanece recomendada.

## Limitações
- Datas ausentes foram registradas como `ausente`; regiões não documentadas foram registradas como `desconhecido`.
- A auditoria não abre rasters, não lê pixels, não baixa assets e não calcula embeddings.
- Metadados temporais e de nuvem aparecem de forma parcial em registros estruturados, mas não fecham prontidão temporal ampla para MV1.

## Guardrails preservados
- sem_treino_de_modelo
- sem_criacao_de_labels
- sem_negativos_formais
- sem_promocao_de_ground_truth
- sem_tabela_multimodal_de_atributos
- DINOv2_apenas_encoder_congelado_exploratorio
- analise_apenas_metadados
