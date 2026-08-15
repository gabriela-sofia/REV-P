# SUSC-14B - descoberta live de referencias espaciais oficiais

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

- Network gate: `SUSC_13B_NETWORK` = `True`
- Referencias registradas: **575**
- Por regiao: **{'curitiba': 9, 'petropolis': 7, 'recife': 559}**
- Download candidates: **513**

A busca priorizou Recife, mas tambem registrou Curitiba e Petropolis. Apenas
fontes oficiais/rastreaveis foram registradas. Nenhum Google Maps, Nominatim ou
geocoding generico foi usado. O baseline local do SUSC-14A foi mantido como
referencia rastreavel para comparacao e reproducibilidade.
