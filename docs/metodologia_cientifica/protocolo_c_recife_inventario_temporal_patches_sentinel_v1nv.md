# Protocolo C - Recife inventario temporal Sentinel v1nv

O inventario localiza metadados REC/Sentinel ja versionados e registra datas de cena apenas quando presentes explicitamente.

Quando a data da imagem nao existe no manifest, o status e SENTINEL_DATE_MISSING. O script nao le raster bruto e nao infere data por nome de cidade, evento ou patch.

A matriz de readiness nao cria label nem abre C4.
