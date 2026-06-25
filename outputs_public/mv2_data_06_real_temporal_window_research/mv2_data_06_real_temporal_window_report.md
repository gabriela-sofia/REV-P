# DATA-06 - pesquisa de janelas temporais reais

## Estado
- targets investigados: 10
- candidatos fortes: 10
- distribuicao por forca: {"STRONG": 10}
- input local criado (nao versionado): True

## Evidencia
- Recife (REC_2022_05_24_30): janela 2022-05-24 a 2022-05-30 (v2at + event_sentinel_temporal_window_registry),
  corroborada por CEMADEN/MCTI e INMET (alerta vermelho 28-29/05/2022).
- Petropolis (PET_2022_02_15): janela 2022-02-15 (v2at + registro interno),
  corroborada por Copernicus EMS (S2 pos-evento 17/02/2022).

## Regras respeitadas
- nenhuma data sem fonte; nenhuma janela de "mes inteiro" sem justificativa;
- nenhum bbox-only; nenhuma noticia generica sem ligacao temporal;
- PDF/HTML bruto pesado nao copiado (apenas URL, titulo, data).

## Side effects
- chamadas/downloads/rasters/crops: 0/0/0/0.
