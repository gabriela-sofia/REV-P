# SUSC-17C25 - CHIRPS real por AOI e matriz multimodal cientifica review-only

## Objetivo
Fazer funcionar CHIRPS real por AOI/janela antecedente e consolidar uma matriz multimodal scientific_review_only combinando Sentinel-2 multitemporal (17C24) com chuva antecedente CHIRPS real.

## Estrategia CHIRPS executada
- Estrategia selecionada: chc_ucsb_daily_file_windowed_read (Rota A).
- Rota A: raster global diario CHC/UCSB (chirps-v2.0.YYYY.MM.DD.tif.gz, 0.05 graus) baixado temporariamente em local_runs; apenas o pixel do AOI de cada patch e lido via rasterio /vsigzip/. Somente CSV/JSON leve derivado e commitado, com SHA256. Nenhum raster global commitado.
- Estrategias tentadas: 1 (Rotas B/C/D nao foram necessarias pois a Rota A obteve CHIRPS real).

## Resultado CHIRPS
- Patches processados: 5.
- Janelas CHIRPS processadas: 15 (CHIRPS_3d/7d/30d x 5 patches).
- Valores diarios reais: 150 (30 dias x 5 patches, 2022-04-24 a 2022-05-23).
- Features de chuva reais: 15.
- Patches com CHIRPS real: 5.
- Manifestos CHIRPS: 10.

## Matriz multimodal
- Matriz multimodal criada: True (5 linhas).
- Features Sentinel-2 na matriz: 8; features CHIRPS na matriz: 9.
- Deltas observacionais nao entram na matriz.

## Guardrails
- CHIRPS usado como evento observado: 0; features promovidas a treino: 0.
- Janela apenas pre-evento (ate 2022-05-23); Ground Reference: 0; ground truth: 0; label: 0; score v6 intacto; score v7 inexistente; 17B bloqueado.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C26 Dossie sensorial multimodal por patch (Sentinel-2 + CHIRPS) para revisao cientifica review-only e requisitos de ground reference
