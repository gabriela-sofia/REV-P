# Protocolo C - Recife scanner local de metadata Sentinel v1oh

O scanner procura assets Sentinel locais em raizes seguras e registra somente metadados sanitizados.

Rasters nao sao copiados nem versionados; quando rasterio existe, apenas tags/header metadata sao lidos.

Data de modificacao/download nunca e usada como scene_date.
