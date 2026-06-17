# Protocolo manual v2ck

Para cada tarefa, o revisor deve produzir geometria vetorial local com CRS
explicito, proveniencia, hash e vinculo documental. Saidas aceitas: GeoJSON|GPKG|Shapefile|CSV_WKT_COM_CRS_EXPLICITO.
Saidas rejeitadas: PNG_ISOLADO|JPEG_ISOLADO|PDF_SEM_GEORREFERENCIAMENTO|DESCRICAO_TEXTUAL|LINK_SEM_ARQUIVO_LOCAL|COORDENADA_INFERIDA_SEM_FONTE.

Nenhuma coordenada pode ser inferida sem fonte. Nenhum arquivo visual isolado vale
como geometria observada validada.
