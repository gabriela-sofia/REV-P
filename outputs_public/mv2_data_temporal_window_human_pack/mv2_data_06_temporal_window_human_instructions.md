# DATA-06 - Instrucoes de preenchimento da janela temporal

Este pacote e de entrada humana rastreavel. Nenhuma data e preenchida
automaticamente pelo script.

## Como preencher
1. Abra `mv2_data_06_temporal_window_human_template.csv`.
2. Para cada `patch_id`/`asset_id`, localize a janela temporal **apenas** em uma
   fonte oficial ou registro auditavel.
3. Preencha `temporal_window_start` e `temporal_window_end` no formato
   ISO `AAAA-MM-DD`.
4. Preencha `temporal_window_source` (nome da fonte), `source_ref` (link, DOI,
   protocolo ou caminho do registro), `source_type` e `evidence_strength`.
5. So mude `review_status` para `APPROVED`/`REVIEWED`/`CONFIRMED` depois de
   conferir a fonte. Enquanto estiver vazio, mantenha `PENDING_HUMAN_FILL`.
6. Salve o arquivo preenchido em uma pasta local (nao versionada), por exemplo
   `local_only/mv2_data_temporal_window/mv2_data_06_temporal_window_filled.csv`,
   e rode `scripts/mv2_data_06_temporal_window_promotion.py --filled-template <caminho>`.

## Regras inviolaveis
- Datas nunca sao inferidas por bbox.
- Datas nunca sao inferidas por cidade.
- "Evento provavel" sem fonte nao vale.
- Janela so pode existir com fonte rastreavel.
- Sem preenchimento, o estagio permanece `BLOCKED_NO_FILLED_TEMPLATE`.

## Fontes aceitas
- BOLETIM_OFICIAL
- RELATORIO_OFICIAL
- CEMADEN
- DEFESA_CIVIL
- CEMS_COPERNICUS_EMS
- SGB_CPRM
- ANA
- PUBLICACAO_CIENTIFICA
- REGISTRO_INTERNO_AUDITAVEL

## Fontes nao aceitas
- MEMORIA_HUMANA_SEM_REGISTRO
- ESTIMATIVA_VISUAL
- DATA_INVENTADA
- JANELA_ABERTA_SEM_JUSTIFICATIVA
- BBOX_ONLY_SEARCH
