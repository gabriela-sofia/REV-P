# SUSC-17C32 - Geocodificacao oficial de ocorrencias e vinculo patch-buffer para G4d

## Objetivo (estreito)
Atacar somente o gargalo G4d/G4_full reusando o 17C31: ocorrencia oficial hidrologica -> logradouro/bairro -> geocodificacao controlada -> distancia ao patch/buffer -> G4d. NAO redesenha fingerprint, gate promotion engine, acquisition queue, SAR feasibility, separacao G5d nem politica de gates.

## Ponto cientifico
G5d/G5_full ja existem para 14 bairros hidrologicos (17C31). O gargalo e G4d/G4_full. Nao se busca mais provar o evento; busca-se ponto oficial de ocorrencia hidrologica proximo ao patch.

## Geocodificacao controlada
- Alvos de ocorrencia hidrologica (bairros prioritarios): 12.
- Tentativas de geocodificacao: 12 (resolvidas: 11); pontos geocodificados: 11.
- Geocoder: Nominatim/OSM (NAO oficial); numero do endereco anonimizado (LGPD) -> precisao street-level, incerteza alta.

## Distancia patch/buffer
- Buffer aceitavel: 1500 m. Pontos dentro do buffer: 0.
- Distancia mais proxima ao patch: 3142.1 m; mais distante: 12641.9 m.
- Amostra:
  - Pina / Avenida Republica do Libano: 11031.8 m -> S17C6_CANARY_REC_00001
  - Imbiribeira / Rua Joao Murilo de Oliveira: 12641.9 m -> S17C6_CANARY_REC_00001
  - Afogados / Rua Vicente Ribeiro de Barros: 9161.4 m -> S17C6_CANARY_REC_00001
  - Areias / Rua Joao Paulo Ii: 11520.2 m -> S17C6_CANARY_REC_00001
  - Areias / Rua Jose Firmino Pires: 11311.4 m -> S17C6_CANARY_REC_00001
  - Ipsep / Rua Alvorada: 11993.9 m -> S17C6_CANARY_REC_00001
  - Iputinga / Estrada do Barbalho: 4189.2 m -> S17C6_CANARY_REC_00001
  - Iputinga / Rua Luiz Souto Dourado: 3142.1 m -> S17C6_CANARY_REC_00001

## Resultado (honesto)
- G4d: false -> false (true count = 0).
- G4_full: false -> false (true count = 0).
- Ground Reference Candidates review-only aceitos: 0. 17B permanece bloqueado.
- Classe do resultado: B_honest_block_spatial_mismatch.
- Bloqueio honesto e especifico: (1) enderecos oficiais com numero anonimizado (LGPD) -> geocodificacao street-level de incerteza alta; (2) o patch candidato (AOI Charter758, Recife Antigo/Santo Amaro ~-8.00) NAO coincide espacialmente com os bairros onde o alagamento foi oficialmente documentado (Varzea, Pina, Areias, Iputinga: 9-13 km ao sul/oeste); (3) OSM sozinho nao abre G4d.

## Guardrails
- bairro/centroide nao abriu G4d; OSM sozinho nao abriu G4d; area de risco nao abriu G4d; logradouro sem numero => incerteza alta; coordenada nunca inventada; nenhum ground truth/label; score v6 intacto; score v7 inexistente; 17B nao desbloqueado.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C33 Reancorar patch candidato para bairro com alagamento oficial documentado (ex.: Varzea/Areias) OU obter ponto de endereco oficial de-anonimizado (numero) proximo ao patch atual; manter score v6 intacto e 17B fail-closed.
