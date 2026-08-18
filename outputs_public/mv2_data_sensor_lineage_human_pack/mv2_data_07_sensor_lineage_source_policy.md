# DATA-07 - Politica de fontes de sensor lineage

## Evidencia aceita (sensor_source_ref)
- Metadado de produto Sentinel (ex.: product_id, datatake, MGRS tile).
- Manifesto de aquisicao oficial (CDSE, GEE) com referencia verificavel.
- Registro interno auditavel que vincule o asset a um produto de origem.

## Nao aceito
- Inferir Sentinel-2 so porque o PNG "parece" optico.
- Tratar embedding NPZ ou render DINO como raster espectral.
- Deduzir familia pelo nome do arquivo.

## Principio
A familia do sensor so e promovida com `sensor_source_ref` rastreavel. Sem cadeia
de origem, o registro fica `UNKNOWN_BLOCKED` e nao gera elegibilidade espectral.
