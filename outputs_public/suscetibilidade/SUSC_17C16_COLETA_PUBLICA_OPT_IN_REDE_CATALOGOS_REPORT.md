# SUSC-17C16 - Coleta publica com opt-in de rede para catalogos CHIRPS/CDSE

## Objetivo
O 17C16 especializa a coleta publica controlada para catalogos CHIRPS e Sentinel-2/CDSE. O pacote versionado permanece offline e deterministico; rede real exige `SUSC_17C16_ALLOW_NETWORK=1` nos modos CLI.

## Drivers
ChirpsCatalogProbeDriver, CopernicusSentinel2CatalogProbeDriver, OfficialLightweightArtifactDriver, BlockedHeavyRasterDriver, BlockedAuthenticatedActionDriver

## Resultado do build publico
- Rede habilitada no build: False.
- Canais avaliados: 9.
- Probes publicos tentados: 0.
- Coletas leves tentadas: 0.
- Artefatos leves coletados: 0.
- Catalogos CHIRPS sondados com metadados: 0.
- Catalogos Sentinel-2/CDSE sondados com metadados: 0.

## Gates
Metadado de catalogo nao vira evento observado. Sem artefato real de evento, G4 e G5 permanecem bloqueados e nenhum Ground Reference Candidate e aceito.

## Guardrails
Nao houve submissao externa, protocolo, resposta inventada, contato inventado, raster pesado, tile, embedding, score v7, patch oficial, patch-link oficial, ground truth ou label de treino.

## Proximo marco recomendado
SUSC-17C17 Politica de execucao para raster leve, AOI de patch e intake de metadados coletados com rede
