# SUSC-17C31 - Execucao da fila G4c/G4d/G5d: geometria oficial, SAR metadata e separacao hidrologica

## Objetivo
Executar a fila do 17C30 para atacar diretamente os subgates bloqueados G4c (geometria patch-level), G4d (vinculo patch/buffer) e G5d (separacao hidrologica). Aquisicao real dirigida em cascata (rotas A-D), sem redesenhar fingerprint/policy/gates do 17C30.

## Aquisicao real
- Fila consumida: 36 linhas.
- Tentativas dirigidas: 52.
- Artefatos reais adquiridos: 8 (oficiais/tecnicos: 8); parseados: 8.
- Fontes reais: Dados Recife (CKAN) - GeoJSON de areas de risco (pontos oficiais), recorte de Atendimentos da Defesa Civil 2022 na janela do evento, abrigos temporarios; ASF Sentinel-1 GRD (metadata).

## Geometria e fenomeno
- Geometry candidates: 2 (geocodaveis: 2, com geometria patch-level: 2).
- Phenomenon separation candidates: 42; hidrologicos separados: 14.
- G4c=2 (true), G4d=0 (true), G4_full=0.
- G5d=14 (true), G5_full=14.
- Bairros com fenomeno hidrologico separado (G5d): Afogados, Apipucos, Areias, Brejo de Beberibe, Campina do Barreto, Caxanga, Cordeiro, Estancia, Imbiribeira, Ipsep, Iputinga, Jiquia.

## SAR metadata feasibility (Rota C)
- Cenas Sentinel-1 pre-evento: 3; durante/pos: 4.
- Footprint geravel no futuro: true (raster NAO baixado; requires_future_raster_processing=true).

## Transicao de subgate
- G4c: false -> true; G4d: false -> false; G5d: false -> true.
- G4c aberto: geometria oficial de ponto (areas de risco Recife) e endereco de ocorrencia presente; G5d aberto: ocorrencias oficiais separam bairro so-hidrologico de movimento de massa; G4d permanece bloqueado: geometria de evento sem coordenada patch-level vinculada ao patch/buffer

## Resultado (honesto)
- Resultado A parcial: G4c e G5d avancaram com evidencia oficial real (geometria de ponto de areas de risco + separacao de fenomeno por bairro nas ocorrencias da Defesa Civil).
- G4d permanece bloqueado: a geometria de EVENTO nao tem coordenada patch-level vinculada ao patch/buffer (area de risco != evento observado; enderecos de ocorrencia sem coordenada e/ou distantes do patch). G4_full=false.
- Ground Reference Candidates review-only aceitos: 0. 17B permanece bloqueado.

## Guardrails
- Setor de risco NAO tratado como evento observado; alerta NAO tratado como ocorrencia; SAR metadata NAO virou footprint; nenhum raster pesado; coordenada nunca inventada; OSM/risk-sector centroid apenas geocodificacao de apoio; noticia comercial nao virou Ground Reference; score v6 intacto; score v7 inexistente; nenhum ground truth/label; 17B nao desbloqueado automaticamente.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C32 Geocodificacao oficial de enderecos de ocorrencia proximos ao patch (ponto com coordenada) e recorte tecnico Sentinel-1 leve para tentar G4d/G4_full, mantendo score v6 intacto e 17B fail-closed.
