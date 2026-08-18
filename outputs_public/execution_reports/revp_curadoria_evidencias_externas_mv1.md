# Curadoria de evidencias externas (MV1)

> Passada review-only. Nenhuma fonte externa foi promovida a ground truth operacional, nenhum label foi criado, nenhum negativo formal foi criado e nenhum treino foi liberado.

## 1. Escopo da curadoria externa

Baixar (quando ja disponivel localmente), auditar, organizar e documentar fontes externas oficiais ou institucionalmente confiaveis para Recife, Petropolis e Curitiba, alem de fontes nacionais/internacionais de contexto. Toda evidencia serve como suporte observacional/contextual e NAO vira label automaticamente.

## 2. Relacao com o marco label-free MV1

Esta curadoria e paralela e nao interfere no fechamento do marco `marco/validacao-label-free-evidencia-estrutural-mv1`. Ela apenas prepara um pacote auditavel de evidencias externas para uso futuro, preservando todos os guardrails do REV-P (Sentinel-first, DINOv2 congelado, sem ground truth operacional patch-level, sem classe binaria, sem negativo formal).

## 3. Fontes pesquisadas

Total de 19 fontes inventariadas. Fontes locais ja presentes no repositorio foram auditadas a partir do arquivo real (SHA256 calculado). Fontes nacionais/internacionais foram pesquisadas via busca web (data de acesso 2026-06-18) e registradas como candidatas nao baixadas.

- `FONTE_REC_001` — Defesa Civil do Recife / Prefeitura do Recife — Areas de risco mapeadas - Recife (pontos) (aprovada_com_ressalvas)
- `FONTE_REC_002` — International Charter Space and Major Disasters (Ativacao 758) — Charter 758 - poligono candidato digitalizado (produto MEDIA-871-16) (requer_revisao_humana)
- `FONTE_REC_003` — SEDEC - Secretaria Executiva de Defesa Civil do Recife — Dados vivos SEDEC Recife - ocorrencias (aprovada_para_contexto)
- `FONTE_REC_004` — EMLURB / Servico 156 - Prefeitura do Recife — EMLURB 156 Recife - solicitacoes (aprovada_com_ressalvas)
- `FONTE_PET_001` — SGB/CPRM (Servico Geologico do Brasil) — Avaliacao tecnica pos-desastre: Petropolis, RJ (2022) (aprovada_para_contexto)
- `FONTE_PET_002` — SGB/CPRM (Servico Geologico do Brasil) — Anexos da avaliacao pos-desastre Petropolis 2022 (ZIP) (requer_revisao_humana)
- `FONTE_PET_003` — SGB/CPRM (Servico Geologico do Brasil) — ANEXO I - CPRM Petropolis 19-02-22 Bairro Mosella (aprovada_para_contexto)
- `FONTE_PET_004` — Prefeitura Municipal de Petropolis — Diario Oficial do Municipio de Petropolis - fevereiro/2022 (aprovada_para_contexto)
- `FONTE_CUR_001` — Coordenadoria Estadual da Defesa Civil do Parana — Chuvas fortes causam transtornos em Curitiba e no Litoral (Defesa Civil PR) (aprovada_para_contexto)
- `FONTE_CUR_002` — Fonte oficial (one-hop a partir da Defesa Civil PR) — PDF contextual capturado em crawl de 1 salto (v2ce) (requer_revisao_humana)
- `FONTE_CUR_003` — IPPUC / GeoCuritiba - Prefeitura de Curitiba — Mapeamento de areas de alagamento Curitiba (IPPUC/GeoCuritiba) (bloqueada_sem_origem_rastreavel)
- `FONTE_NAC_001` — CEMADEN/MCTI - Centro Nacional de Monitoramento e Alertas de Desastres Naturais — CEMADEN - Mapa Interativo / GeoRisk (requer_revisao_humana)
- `FONTE_NAC_002` — ANA - Agencia Nacional de Aguas e Saneamento Basico (SNIRH/HidroWeb) — ANA HidroWeb / SNIRH - series hidrometeorologicas (aprovada_para_contexto)
- `FONTE_NAC_003` — MapBiomas — MapBiomas - Colecao 10.1 (uso e cobertura do solo) (aprovada_para_contexto)
- `FONTE_NAC_004` — IBGE - Instituto Brasileiro de Geografia e Estatistica — IBGE - Malha de Setores Censitarios / Downloads Geociencias (aprovada_para_contexto)
- `FONTE_NAC_005` — APAC - Agencia Pernambucana de Aguas e Clima — APAC - boletins de chuva/rios/reservatorios (PE) (requer_revisao_humana)
- `FONTE_NAC_006` — DRM-RJ / NADE - Servico Geologico do Estado do Rio de Janeiro — DRM-RJ/NADE - relatorio pos-desastre Petropolis 2022 (bloqueada_sem_origem_rastreavel)
- `FONTE_INT_001` — Copernicus Emergency Management Service (EMS) Rapid Mapping — Copernicus EMS Rapid Mapping - ativacao Petropolis 2022 (a confirmar) (requer_revisao_humana)
- `FONTE_INT_002` — NHESS (Copernicus Publications) - artigo academico — Deadly disasters... flash floods and landslides of February 2022 in Petropolis (NHESS, 2023) (aprovada_com_ressalvas)

## 4. Fontes baixadas

10 fontes ja presentes no repositorio (area git-ignored), com SHA256 calculado:

- `FONTE_REC_001` — `datasets/external_sources/recife_minimal_tp/raw/recife_defesa_civil_risk_areas_geojson.geojson` — SHA256 `0f8036b19ec60481...`
- `FONTE_REC_002` — `datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758/derived/event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson` — SHA256 `8e833d449effb837...`
- `FONTE_REC_003` — `local_runs/protocolo_c/v1np/raw/recife_dados_vivos_sedec_f3b3a0ab-ac8f-4fe8-a1cc-c3afdd20081a_001.csv` — SHA256 `e26f80ddada0a6c7...`
- `FONTE_REC_004` — `local_runs/protocolo_c/v1np/raw/recife_emlurb_156_031c3ad3-265f-4d9c-830a-7d1f85f830fa_007.csv` — SHA256 `dd2f31e3f53c3aa9...`
- `FONTE_PET_001` — `local_runs/protocolo_c/v1if/raw_official_sources/Relatorio_Tecnico_Petropolis.pdf` — SHA256 `45d1a9802c648570...`
- `FONTE_PET_002` — `local_runs/protocolo_c/v1if/raw_official_sources/anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip` — SHA256 `78a185ba06b6f5d4...`
- `FONTE_PET_003` — `local_runs/protocolo_c/v1mc/extracted/PKG_V1MC_0001_24be39ae/ANEXO-I-CPRM_Relatório_Petrópolis_19-02-22_Bairro_Mosella.pdf` — SHA256 `053fa4cbccac09a6...`
- `FONTE_PET_004` — `local_runs/protocolo_c/v1na/raw/gazette_issue_2022-02-15_4542_0a45c34e.pdf` — SHA256 `3399af178d756a19...`
- `FONTE_CUR_001` — `local_runs/ground_truth/v2cd/downloaded_sources/SRC_4c0debd1596a.html` — SHA256 `8997b1edd6025e46...`
- `FONTE_CUR_002` — `local_runs/ground_truth/v2ce/downloaded_onehop_sources/SRC_9b151d28ab38.pdf` — SHA256 `4db40582b10238be...`

## 5. Fontes nao baixadas e motivo

- `FONTE_CUR_003` — IPPUC / GeoCuritiba - Prefeitura de Curitiba — Sem URL direta no log v1if (OBS_CUR_001). Curitiba nunca vira negativo formal por ausencia de evidencia.
- `FONTE_NAC_001` — CEMADEN/MCTI - Centro Nacional de Monitoramento e Alertas de Desastres Naturais — Tambem: GeoRisk (georisk.cemaden.gov.br) e gov.br/cemaden. Suscetibilidade nao e geometria de evento observado.
- `FONTE_NAC_002` — ANA - Agencia Nacional de Aguas e Saneamento Basico (SNIRH/HidroWeb) — Tambem dadosabertos.ana.gov.br. Serie pontual/estacao; nao e geometria de extensao de evento.
- `FONTE_NAC_003` — MapBiomas — Contexto de cobertura, nao geometria de evento. Verificar versao/licenca antes de uso publico.
- `FONTE_NAC_004` — IBGE - Instituto Brasileiro de Geografia e Estatistica — Geometria administrativa/censitaria (contexto), nao geometria de evento observado.
- `FONTE_NAC_005` — APAC - Agencia Pernambucana de Aguas e Clima — Boletim/serie regional; suporte contextual de chuva, nao geometria de evento.
- `FONTE_NAC_006` — DRM-RJ / NADE - Servico Geologico do Estado do Rio de Janeiro — Log v1if (OBS_PET_003): NO_URL, requer solicitacao formal DRM-RJ/NADE.
- `FONTE_INT_001` — Copernicus Emergency Management Service (EMS) Rapid Mapping — ID exato da ativacao (EMSR) para Petropolis nao foi confirmado na busca; nao inventar. Produto pode delimitar deslizamento, nao flood.
- `FONTE_INT_002` — NHESS (Copernicus Publications) - artigo academico — NHESS vol. 23, p. 1157, 2023. Usar como referencia documental; nao tratar figuras do artigo como geometria auditavel.

## 6. Organizacao dos arquivos em quarentena

A estrutura `local_only/evidencias_externas_quarentena/{recife,petropolis,curitiba,fontes_nacionais,fontes_internacionais}/` foi criada para ingestao de novos downloads. Os arquivos brutos ja existentes permanecem em sua localizacao git-ignored original (`local_runs/...`, `datasets/external_sources/...`) e sao referenciados por caminho relativo no manifesto, para evitar duplicacao de artefatos pesados. Apenas manifestos, auditorias, indices e checksums foram publicados em `outputs_public/`.

## 7. Fontes aprovadas para contexto

- `FONTE_REC_001` — Defesa Civil do Recife / Prefeitura do Recife — Areas de risco mapeadas - Recife (pontos) (aprovada_com_ressalvas)
- `FONTE_REC_003` — SEDEC - Secretaria Executiva de Defesa Civil do Recife — Dados vivos SEDEC Recife - ocorrencias (aprovada_para_contexto)
- `FONTE_REC_004` — EMLURB / Servico 156 - Prefeitura do Recife — EMLURB 156 Recife - solicitacoes (aprovada_com_ressalvas)
- `FONTE_PET_001` — SGB/CPRM (Servico Geologico do Brasil) — Avaliacao tecnica pos-desastre: Petropolis, RJ (2022) (aprovada_para_contexto)
- `FONTE_PET_003` — SGB/CPRM (Servico Geologico do Brasil) — ANEXO I - CPRM Petropolis 19-02-22 Bairro Mosella (aprovada_para_contexto)
- `FONTE_PET_004` — Prefeitura Municipal de Petropolis — Diario Oficial do Municipio de Petropolis - fevereiro/2022 (aprovada_para_contexto)
- `FONTE_CUR_001` — Coordenadoria Estadual da Defesa Civil do Parana — Chuvas fortes causam transtornos em Curitiba e no Litoral (Defesa Civil PR) (aprovada_para_contexto)
- `FONTE_NAC_002` — ANA - Agencia Nacional de Aguas e Saneamento Basico (SNIRH/HidroWeb) — ANA HidroWeb / SNIRH - series hidrometeorologicas (aprovada_para_contexto)
- `FONTE_NAC_003` — MapBiomas — MapBiomas - Colecao 10.1 (uso e cobertura do solo) (aprovada_para_contexto)
- `FONTE_NAC_004` — IBGE - Instituto Brasileiro de Geografia e Estatistica — IBGE - Malha de Setores Censitarios / Downloads Geociencias (aprovada_para_contexto)
- `FONTE_INT_002` — NHESS (Copernicus Publications) - artigo academico — Deadly disasters... flash floods and landslides of February 2022 in Petropolis (NHESS, 2023) (aprovada_com_ressalvas)

## 8. Fontes bloqueadas

- `FONTE_CUR_003` — IPPUC / GeoCuritiba - Prefeitura de Curitiba — Mapeamento de areas de alagamento Curitiba (IPPUC/GeoCuritiba) (bloqueada_sem_origem_rastreavel)
- `FONTE_NAC_006` — DRM-RJ / NADE - Servico Geologico do Estado do Rio de Janeiro — DRM-RJ/NADE - relatorio pos-desastre Petropolis 2022 (bloqueada_sem_origem_rastreavel)

### Fontes que requerem revisao humana

- `FONTE_REC_002` — International Charter Space and Major Disasters (Ativacao 758) — Charter 758 - poligono candidato digitalizado (produto MEDIA-871-16) (requer_revisao_humana)
- `FONTE_PET_002` — SGB/CPRM (Servico Geologico do Brasil) — Anexos da avaliacao pos-desastre Petropolis 2022 (ZIP) (requer_revisao_humana)
- `FONTE_CUR_002` — Fonte oficial (one-hop a partir da Defesa Civil PR) — PDF contextual capturado em crawl de 1 salto (v2ce) (requer_revisao_humana)
- `FONTE_NAC_001` — CEMADEN/MCTI - Centro Nacional de Monitoramento e Alertas de Desastres Naturais — CEMADEN - Mapa Interativo / GeoRisk (requer_revisao_humana)
- `FONTE_NAC_005` — APAC - Agencia Pernambucana de Aguas e Clima — APAC - boletins de chuva/rios/reservatorios (PE) (requer_revisao_humana)
- `FONTE_INT_001` — Copernicus Emergency Management Service (EMS) Rapid Mapping — Copernicus EMS Rapid Mapping - ativacao Petropolis 2022 (a confirmar) (requer_revisao_humana)

## 9. Eventos externos candidatos

8 eventos candidatos; 0 potencialmente revisaveis (apenas para revisao futura, sem liberar treino) e 8 que nao podem virar label.

- `EV_FONTE_REC_002` (Recife, desastre_hidrometeorologico_composto) — pode_label=false — motivo: risco_landslide_vs_flood
- `EV_FONTE_PET_001` (Petropolis, deslizamento) — pode_label=false — motivo: sem_geometria_auditavel;sem_crs_declarado;risco_landslide_vs_flood
- `EV_FONTE_PET_002` (Petropolis, deslizamento) — pode_label=false — motivo: sem_geometria_auditavel;sem_crs_declarado;risco_landslide_vs_flood
- `EV_FONTE_PET_003` (Petropolis, deslizamento) — pode_label=false — motivo: sem_geometria_auditavel;sem_crs_declarado;risco_landslide_vs_flood
- `EV_FONTE_CUR_001` (Curitiba, alagamento) — pode_label=false — motivo: sem_geometria_auditavel;sem_crs_declarado
- `EV_FONTE_NAC_006` (Petropolis, deslizamento) — pode_label=false — motivo: sem_geometria_auditavel;janela_temporal_nao_fechada;sem_crs_declarado;risco_landslide_vs_flood;fonte_bloqueada_na_auditoria
- `EV_FONTE_INT_001` (Petropolis, deslizamento) — pode_label=false — motivo: sem_geometria_auditavel;janela_temporal_nao_fechada;sem_crs_declarado;risco_landslide_vs_flood
- `EV_FONTE_INT_002` (Petropolis, desastre_hidrometeorologico_composto) — pode_label=false — motivo: sem_geometria_auditavel;sem_crs_declarado;risco_landslide_vs_flood

## 10. Geometrias externas candidatas

- `GEO_FONTE_REC_001` (Recife, Point, EPSG:4326) — overlay=false — bloqueio: geometria_de_suscetibilidade_nao_e_evento_observado
- `GEO_FONTE_REC_002` (Recife, MultiPolygon, EPSG:4326 (origem EPSG:32725)) — overlay=false — bloqueio: produto_pode_misturar_deslizamento_e_inundacao_revisao_obrigatoria

## 11. Riscos metodologicos

- Evidencia administrativa/textual nao fecha geometria patch-level.
- Suscetibilidade (areas de risco) nao e o mesmo que extensao de evento observado.
- Produtos de midia/charter podem ser parcialmente digitalizados e nao revisados.

## 12. Risco de circularidade

Fontes derivadas do proprio pipeline ou que reaproveitam saidas internas podem inflar concordancia. Nenhuma fonte de alta circularidade foi aprovada para uso direto; bases de servico urbano (EMLURB 156) exigem filtragem tematica.

## 13. Risco landslide vs flood

As fontes de Petropolis e a ativacao Charter 758 referem-se predominantemente a DESLIZAMENTO. Scar de deslizamento NAO prova flood extent. Deslizamento e inundacao nunca sao combinados automaticamente.

## 14. Risco de evidencia textual sem geometria

A maioria das fontes (relatorios CPRM, diarios oficiais, boletins, noticias) e textual e foi classificada como documental/contextual. Texto nao substitui geometria auditavel.

## 15. Risco de licenca/uso

Varias fontes tem licenca a confirmar (Charter, portais municipais, CEMADEN, APAC). Toda fonte sem licenca clara permanece bloqueada para uso publico direto.

## 16. Como essas fontes podem ajudar o ground truth futuro

- Geometrias candidatas (Charter 758, areas de risco Recife) podem, apos revisao humana e confirmacao de CRS/janela temporal, sustentar candidatos para revisao futura.
- Series ANA/APAC e relatorios CPRM ancoram a janela temporal e a natureza fisica do evento.
- Bases IBGE/MapBiomas fornecem contexto territorial para estratificacao futura.

## 17. O que ainda falta

- Geometria de EVENTO observado, georreferenciada, revisada e com CRS auditavel para as tres cidades.
- Confirmacao de licenca das fontes marcadas como a_confirmar.
- Obtencao formal das fontes bloqueadas (DRM-RJ, IPPUC/GeoCuritiba, Defesa Civil PE).
- Confirmacao do ID de ativacao Copernicus EMS para Petropolis.

## 18. Guardrails preservados

- REV-P permanece review-only; nenhuma fonte externa vira ground truth operacional nesta passada.
- Evidencia externa NAO vira label automaticamente.
- unknown nunca vira negativo; ausencia de evento nunca vira classe 0.
- Curitiba nunca vira negativo formal.
- Desastre documentado nao significa label patch-level fechado.
- Landslide scar nao prova flood extent; deslizamento e flood nunca sao misturados automaticamente.
- Geometria de suscetibilidade nao e geometria de evento observado.
- Fonte textual nao substitui geometria auditavel.
- Fonte sem licenca clara fica bloqueada para uso publico direto.
- Nenhum SHA256, URL ou data foi inventado; valores derivam de arquivo real ou de log/busca registrada.

## 19. Proximos passos recomendados

Submeter geometrias candidatas (Charter 758 / areas de risco Recife) a revisao humana e solicitar formalmente fontes bloqueadas (DRM-RJ, IPPUC, Defesa Civil PE), sem promover nada a ground truth nesta passada.
