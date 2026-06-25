# Aquisicao real DATA-06/07 - relatorio inicial

## Branch / worktree
- branch: dados/aquisicao-real-data-06-07
- worktree: REV-P-aquisicao-real-data-06-07
- base: dados/evolucao-real-metadados-data-06-09 (HEAD b0c98a4)

## Filas de aquisicao lidas
- outputs_public/mv2_data_06_09_real_acquisition_queue/mv2_data_06_temporal_window_acquisition_queue.csv
- outputs_public/mv2_data_06_09_real_acquisition_queue/mv2_data_07_sensor_lineage_acquisition_queue.csv
- outputs_public/mv2_data_06_09_real_acquisition_queue/mv2_data_08_metadata_config_checklist.md

## 10 targets atuais
- Recife (alagamento/inundacao maio 2022): REC_00019, REC_00183, REC_00204, REC_00205, REC_00227
- Petropolis (deslizamento/chuva fev 2022): PET_00016, PET_00104, PET_00119, PET_00140, PET_00240
- bbox/CRS vazios na fila (registry sem geometria valida) -> nao inventados.

## Vinculo patch -> evento (fonte interna auditavel committed)
- v2at_event_patch_package_registry.csv:
  - 5 REC -> evento REC_2022_05_24_30, janela 2022-05-24 a 2022-05-30
  - 5 PET -> evento PET_2022_02_15, janela 2022-02-15
- event_sentinel_temporal_window_registry.csv corrobora as mesmas janelas por evento.

## Estrategia
- DATA-06: resolver janela temporal real por evento (interno auditavel) e corroborar com fonte oficial publica.
- DATA-07: resolver lineage sensorial a partir de registries committed (v2at sentinel_sensor_family, manifest v1fu, v2bd), sem inferir sensor por nome visual.

## Garantias
- nenhum input local sera versionado;
- nenhum raster/crop/SCL/treino;
- sem chamada API por padrao;
- chamadas/downloads/rasters/crops: 0/0/0/0.
