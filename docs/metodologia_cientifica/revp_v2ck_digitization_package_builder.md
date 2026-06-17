# REV-P v2ck - pacote de digitalizacao/georreferenciamento manual

Este marco prepara uma fila de trabalho para revisao humana. Ele nao executa
digitalizacao automatica, nao transforma imagem em vetor por inferencia e nao cria
ground truth operacional.

Saidas aceitas futuramente: GeoJSON, GPKG, Shapefile e CSV com WKT quando o CRS
estiver explicito. Saidas rejeitadas como geometria validada: PNG isolado, JPEG
isolado, PDF sem georreferenciamento, descricao textual, link sem arquivo local e
coordenada inferida sem fonte.

Todo item exige revisao humana, CRS explicito, proveniencia, hash e vinculo
documental antes de alimentar o validador `v2cl`.

