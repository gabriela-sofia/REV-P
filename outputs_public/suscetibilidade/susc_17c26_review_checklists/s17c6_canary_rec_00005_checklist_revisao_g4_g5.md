# Checklist de revisão G4/G5 - S17C6_CANARY_REC_00005

## Identificação do patch
- Patch: S17C6_CANARY_REC_00005 | Evento: REC_2022_05_24_30 | BBox: -34.926498328,-7.986973178,-34.917515175,-7.978079592

## Artefatos sensoriais disponíveis
- Sentinel-2 multitemporal (3 cenas pré-evento, medianas temporais).

## Chuva antecedente
- CHIRPS_3d/7d/30d (mm): 20.5343/20.5343/139.6736

## Deltas disponíveis
- Delta pré/pós observacional review-only.

## O que procurar em fonte oficial
- Ocorrência oficial com local; geometria/extensão; classificação de fenômeno.

## Critérios de aceite
- Fonte oficial/técnica, data compatível, local georreferenciável, fenômeno explícito.

## Critérios de rejeição
- Sem data, sem local, sem fonte identificável, apenas notícia não confirmada.

## Campos obrigatórios
- event_date_or_period, observed_location, geometry_or_geocodable_address, phenomenon_class, source_name, source_hash.

## Riscos de falso positivo
- Confundir chuva antecedente com evento; confundir mudança espectral com inundação; usar patch vizinho como prova.

## Declaração
- não chamar de ground truth sem G4/G5
