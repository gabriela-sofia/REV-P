# Relatorio v1jq - descoberta de fontes negativas oficiais abertas

## Resultado principal

Fontes oficiais abertas auditadas: 12. Fonte formal pronta: 0. C4 mudou: nao.

## Alto potencial

- SRC_V1JQ_DRM_RJ_CARTOGRAFIA_RISCO_FICHAS: DRM-RJ - cartografia de risco e fichas publicas de campo

Essas fontes sao alvos de extracao, nao evidencias prontas. A pagina de fichas/cartografia do DRM-RJ e a rota mais objetiva porque lista documentos tecnicos por localidade; ainda assim, o indice publico nao contem por si uma declaracao negativa formal.

## Fontes contextuais

- SRC_V1JQ_PREF_DEFESA_CIVIL_PLANOS_CONTINGENCIA: Prefeitura de Petropolis - Defesa Civil planos de contingencia
- SRC_V1JQ_DRM_RJ_CARTA_RISCO_DADOS_ABERTOS: Dados Abertos RJ / DRM-RJ - Carta de Risco Petropolis
- SRC_V1JQ_S2ID_ATLAS_DIGITAL: S2ID / Atlas Digital de Desastres
- SRC_V1JQ_DADOS_GOV_CATALOG: Portal Brasileiro de Dados Abertos
- SRC_V1JQ_CEMADEN_REDEGEO_PETROPOLIS: Cemaden/MCTI - RedeGeo Petropolis monitoring equipment
- SRC_V1JQ_IBGE_BDIA_CONTEXT: IBGE BDIA
- SRC_V1JQ_MAPBIOMAS_CONTEXT: MapBiomas
- SRC_V1JQ_INPE_CHARTER_CONTEXT: INPE / International Charter emergency mapping context

Essas fontes ajudam a contextualizar risco, evento, monitoramento ou estratificacao. Elas nao autorizam negativo formal.

## Proibicoes preservadas

Ausencia de registro, baixo risco, area fora de risco, catalogo sem resultado, background e pseudo-ausencia continuam bloqueados para label. DINO permanece congelado e sem papel de label. O treino supervisionado permanece bloqueado.

## Proximo passo real

Targeted extraction from DRM-RJ public field-sheet/cartography files and selected SGB/CPRM/municipal sources for explicit no-occurrence/stability statements.

Esse passo deve baixar/extrair apenas metadados ou textos necessarios dos documentos publicos selecionados, com timeout e sem massa raster ou arquivos pesados no repositorio.
