# SUSC-17C17 - Snapshot de rede para metadados sensoriais e politica de raster leve

## Objetivo
O 17C17 executa a primeira captura real controlada de metadados publicos de CHIRPS e Sentinel-2/CDSE, com opt-in explicito de rede, e congela o resultado em snapshots leves reprodutiveis. O build publico normal faz apenas replay offline desses snapshots ja commitados; a captura real fica atras de `SUSC_17C17_ALLOW_NETWORK=1` no modo `capture-snapshot` da CLI.

## Fontes sensoriais planejadas
- Fontes planejadas: 4.
- Snapshots de rede tentados: 3.
- Snapshots criados: 3.
- Manifestos de snapshot: 3.
- Metadados CHIRPS capturados: 1.
- Metadados Sentinel-2/CDSE capturados: 2.

## Registro de captura
- CHIRPS_OFFICIAL_DATA_PORTAL: capture_status=captured_public_page_snapshot, network_enabled=true.
- CHIRPS_GEE_CATALOG: capture_status=failed_safe, network_enabled=false.
- SENTINEL2_CDSE_DATA_COLLECTION_PAGE: capture_status=captured_public_page_snapshot, network_enabled=true.
- SENTINEL2_CDSE_STAC_API: capture_status=captured_lightweight_metadata, network_enabled=true.

## Planejamento de raster leve
- AOIs de patches candidatos planejadas: 5.
- Linhas de politica de raster leve: 3.
- Linhas de plano de execucao de raster leve: 20.

## Gates
Metadado de catalogo sensorial nao vira evento observado. G4 (vinculo espacial) e G5 (separacao de fenomeno) permanecem bloqueados por design; nenhum Ground Reference Candidate e criado a partir deste snapshot.

## Guardrails
Nenhum raster bruto foi baixado, nenhum tile foi criado, nenhum embedding foi calculado, o score v6 permanece intacto, o score v7 nao foi criado e o 17B continua bloqueado.

## Proximo marco recomendado
SUSC-17C18 Execucao de estatistica zonal CHIRPS e consulta STAC Sentinel-2 por AOI com politica de armazenamento aprovada
