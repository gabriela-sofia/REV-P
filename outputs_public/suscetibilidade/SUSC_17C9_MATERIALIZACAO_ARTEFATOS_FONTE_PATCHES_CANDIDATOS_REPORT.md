# SUSC-17C9 - Materializacao de artefatos fonte para patches candidatos

O 17C8 revelou que a grade candidata tem geometria e contratos multimodais, mas nao tem artefatos fonte candidato-especificos para marcar features como reais. O resultado de 0 features reais foi metodologicamente correto: sem raster, tile, estatistica zonal, janela temporal e proveniencia por patch candidato, qualquer valor seria copia, inferencia ou placeholder.

Este marco transforma esse bloqueio em plano operacional verificavel. Ele nao extrai feature final, nao executa SAR, nao executa DINO/SatMAE, nao baixa raster pesado e nao cria score v7.

## Artefatos fonte existentes

Existem a grade candidata 17C6 em CSV/GeoJSON, a matriz oficial de features, o manifesto de proveniencia da matriz oficial e manifests de referencia para Sentinel-2, STAC/GEE e DINO. Esses artefatos ajudam a desenhar adapters e requisitos, mas nao substituem artefatos reais por patch candidato.

## Artefatos fonte ausentes

Faltam DEM/HAND, drenagem, derivados de fluxo/TWI, MapBiomas ou cobertura territorial, CHIRPS pre-evento, tile Sentinel-2 real e tile real para DINO/SatMAE cobrindo os 5 patches candidatos.

## Pipelines e adapters

Os pipelines atuais conseguem auditar ou perfilar matrizes oficiais e manifests, mas exigem adapter para aceitar geojson/bbox candidato e para produzir estatistica zonal leve. Parte da cadeia Sentinel-2 aceita geometria/metadata, mas ainda precisa binding especifico da grade candidata e nao deve baixar raster neste marco.

## Exports leves suficientes

A proxima extracao deve preferir CSV de estatistica zonal, pequenos recortes vetoriais, metadata de tile e manifest de input de embedding. Raw raster pesado continua proibido no Git.

## Bloqueios por modalidade

- DINO/SatMAE: falta tile real candidato-especifico e politica de pre-processamento.
- Sentinel-2: falta tile/metadata pre-evento com politica de nuvem.
- Chuva/CHIRPS: falta janela pre-evento agregada por patch candidato.
- Fisicas/urbanas: faltam DEM/HAND/drenagem/MapBiomas ou estatisticas zonais reais.

## Prontidao

- Artefatos inventariados: 16
- Artefatos locais cobrindo a grade candidata: 2
- Artefatos obrigatorios ausentes: 5
- Pipelines auditados: 7
- Exports leves prontos: 0

Score v6 permanece intacto. Score v7 e 17B continuam bloqueados porque nao ha features reais candidato-especificas, QA e politica de promocao.

Proximo marco recomendado: `SUSC-17C10 Pacote de Solicitacao Formal`.
