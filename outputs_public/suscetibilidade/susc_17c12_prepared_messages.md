# SUSC-17C12 - mensagens preparadas para submissao assistida

Este arquivo contem texto copiavel para revisao humana. Nenhuma mensagem foi enviada e nenhum protocolo foi aberto.

## S17C12_MSG_P0_COMPDEC_RECIFE_PKG_FR_REC_002

- ID da solicitacao: `P0_COMPDEC_RECIFE_PKG_FR_REC_002`
- Fornecedor: COMPDEC/Defesa Civil PE
- Prioridade: P0
- Canal: Pagina oficial da Secretaria Executiva de Defesa Civil do Recife
- Status do canal: `confirmed_official_source`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de pontos, enderecos ou geometria oficial do evento Recife maio 2022

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: pontos, enderecos georreferenciados, setores atingidos ou geometria oficial do pacote PKG_FR_REC_002.

Contexto da solicitacao: complementar o footprint coarse do International Charter com evidencia oficial local.

Campos minimos solicitados: data_do_evento;tipo_do_fenomeno;geometria_ou_coordenada;incerteza_ou_escala;fonte_responsavel;crs_se_houver.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00004;S17C10_ATTACH_00005;S17C10_ATTACH_00006
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P0_COMPDEC_RECIFE_PKG_FR_REC_002`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P0_COMPDEC_RECIFE_PKG_FR_REC_002`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P0_COMPDEC_RECIFE_PKG_FR_REC_002 --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0001 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P0_DRM_RJ_PKG_FR_PET_001

- ID da solicitacao: `P0_DRM_RJ_PKG_FR_PET_001`
- Fornecedor: DRM-RJ/NADE
- Prioridade: P0
- Canal: DRM-RJ e OuveRJ
- Status do canal: `confirmed_official_source`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de geometria oficial do evento Petropolis 2022

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: geometria oficial, ponto, poligono, endereco georreferenciado ou tabela de ocorrencias do pacote PKG_FR_PET_001.

Contexto da solicitacao: destravar evidencia observacional oficial para avaliacao patch-level review-only.

Campos minimos solicitados: data_do_evento;tipo_do_fenomeno;geometria_ou_coordenada;incerteza_ou_escala;fonte_responsavel;crs_se_houver.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00001;S17C10_ATTACH_00002;S17C10_ATTACH_00003
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P0_DRM_RJ_PKG_FR_PET_001`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P0_DRM_RJ_PKG_FR_PET_001`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P0_DRM_RJ_PKG_FR_PET_001 --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0002 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30

- ID da solicitacao: `P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30`
- Fornecedor: CHIRPS/GEE - fonte a confirmar
- Prioridade: P1
- Canal: CHIRPS official data page e catalogo Earth Engine
- Status do canal: `confirmed_official_source`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de janela CHIRPS pre-evento Recife maio 2022

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: acumulados CHIRPS 3d, 7d, 30d e runoff_context por patch candidato.

Contexto da solicitacao: destravar chuva/gatilho pre-evento sem usar dado pos-evento.

Campos minimos solicitados: acumulado_pre_evento;janela_temporal;fonte_resolucao;metodo_agregacao.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00016;S17C10_ATTACH_00017;S17C10_ATTACH_00018
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30 --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0003 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID

- ID da solicitacao: `P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID`
- Fornecedor: GEE/DEM/HAND/drenagem - fonte a confirmar
- Prioridade: P1
- Canal: APAC Monitoramento e Fale Conosco
- Status do canal: `candidate_needs_manual_verification`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de DEM/HAND/drenagem cobrindo a grade candidata

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: DEM/HAND, drenagem, distancia a agua, flow accumulation e TWI por bbox candidata.

Contexto da solicitacao: destravar features fisicas sem copiar valores de patch oficial.

Campos minimos solicitados: DEM_HAND_cobrindo_bbox;crs;resolucao;data_versao;fonte_licenca_formato.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00013;S17C10_ATTACH_00014;S17C10_ATTACH_00015
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0004 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY

- ID da solicitacao: `P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY`
- Fornecedor: Defesa Civil Petropolis / DRM-RJ (a confirmar)
- Prioridade: P1
- Canal: Defesa Civil de Petropolis
- Status do canal: `candidate_needs_manual_verification`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de geometria ou ocorrencias Petropolis 2024 Valparaiso/Floresta

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: geometria, pontos, enderecos ou tabela de ocorrencias para Valparaiso/Floresta.

Contexto da solicitacao: recuperar evidencia observacional oficial para evento PET 2024 citado no 17C3/17C4.

Campos minimos solicitados: data_do_evento;tipo_do_fenomeno;geometria_ou_coordenada;incerteza_ou_escala;fonte_responsavel;crs_se_houver.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00010;S17C10_ATTACH_00011;S17C10_ATTACH_00012
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0005 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P1_SENTINEL2_TILE_CANDIDATE_GRID

- ID da solicitacao: `P1_SENTINEL2_TILE_CANDIDATE_GRID`
- Fornecedor: STAC/CDSE/GEE - fonte a confirmar
- Prioridade: P1
- Canal: Copernicus Data Space Sentinel-2 e STAC
- Status do canal: `confirmed_official_source`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de tile Sentinel-2 real para patches candidatos

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: tile ou metadata Sentinel-2 pre-evento cobrindo bboxes candidatas.

Contexto da solicitacao: destravar NDVI/NDWI/MNDWI/NDBI e tile base para embedding.

Campos minimos solicitados: tile_real_cobrindo_bbox;data_aquisicao;bandas_resolucao;politica_nuvem;formato_export.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00019;S17C10_ATTACH_00020;S17C10_ATTACH_00021
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P1_SENTINEL2_TILE_CANDIDATE_GRID`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P1_SENTINEL2_TILE_CANDIDATE_GRID`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P1_SENTINEL2_TILE_CANDIDATE_GRID --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0006 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P1_SGB_CPRM_VECTOR_OR_COORDINATES

- ID da solicitacao: `P1_SGB_CPRM_VECTOR_OR_COORDINATES`
- Fornecedor: SGB/CPRM
- Prioridade: P1
- Canal: SIC e Ouvidoria SGB
- Status do canal: `confirmed_official_source`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de vetores ou coordenadas associados a PDF/ZIP SGB/CPRM

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: vetor, tabela de coordenadas ou legenda georreferenciada dos anexos e relatorio tecnico.

Contexto da solicitacao: PDF/ZIP sem vetor nao fecha geometria forte nem link patch-level.

Campos minimos solicitados: data_do_evento;tipo_do_fenomeno;geometria_ou_coordenada;incerteza_ou_escala;fonte_responsavel;crs_se_houver.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00007;S17C10_ATTACH_00008;S17C10_ATTACH_00009
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P1_SGB_CPRM_VECTOR_OR_COORDINATES`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P1_SGB_CPRM_VECTOR_OR_COORDINATES`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P1_SGB_CPRM_VECTOR_OR_COORDINATES --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0007 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P2_DINO_SATMAE_INPUT_TILE

- ID da solicitacao: `P2_DINO_SATMAE_INPUT_TILE`
- Fornecedor: Pipeline interno DINO/SatMAE - sem execucao nesta sprint
- Prioridade: P2
- Canal: Sem canal externo oficial confirmado
- Status do canal: `not_found`
- Assunto sugerido: Solicitacao formal REV-P - Preparacao de tile real para DINO/SatMAE

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: manifest de tile real com bandas, resolucao, politica de nuvem e pre-processamento.

Contexto da solicitacao: destravar embedding real sem usar smoke test sintetico.

Campos minimos solicitados: tile_real_cobrindo_bbox;data_aquisicao;bandas_resolucao;politica_nuvem;formato_export.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00022;S17C10_ATTACH_00023;S17C10_ATTACH_00024
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P2_DINO_SATMAE_INPUT_TILE`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P2_DINO_SATMAE_INPUT_TILE`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P2_DINO_SATMAE_INPUT_TILE --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0008 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.

## S17C12_MSG_P2_SAR_RUNTIME_OR_EXPORT_PATH

- ID da solicitacao: `P2_SAR_RUNTIME_OR_EXPORT_PATH`
- Fornecedor: Runtime tecnico SAR - fonte a confirmar
- Prioridade: P2
- Canal: Sem canal externo oficial confirmado
- Status do canal: `not_found`
- Assunto sugerido: Solicitacao formal REV-P - Solicitacao de runtime ou caminho de export SAR review-only

### Mensagem copiavel

Prezadas(os),

Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: runtime, credenciais ou caminho de export SAR com regra anti-leakage documentada.

Contexto da solicitacao: registrar dependencia tecnica SAR sem executar SAR nesta sprint.

Campos minimos solicitados: runtime_credencial;produto_janela_temporal;formato_export.

Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.

Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.

Atenciosamente,
[preencher manualmente]

### Controle operacional

- Anexos recomendados: S17C10_ATTACH_00025;S17C10_ATTACH_00026;S17C10_ATTACH_00027
- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.
- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.
- Comando local sugerido para prepare: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py prepare --request-id P2_SAR_RUNTIME_OR_EXPORT_PATH`
- Comando local sugerido para open-channel: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py open-channel --request-id P2_SAR_RUNTIME_OR_EXPORT_PATH`
- Comando local sugerido para record-submission: `python scripts\suscetibilidade\susc_17c12_submission_orchestrator.py record-submission --request-id P2_SAR_RUNTIME_OR_EXPORT_PATH --submitted-at AAAA-MM-DD --channel-id S17C11_CH_0009 --evidence-note "valor literal informado pela pessoa"`
- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.
