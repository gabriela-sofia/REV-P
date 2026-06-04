# Protocolo C — v1uf Completion Handoff

**Status:** `COMPLETE_REVIEW_ONLY_FAIL_CLOSED`
**Data:** 2026-06-03
**Etapa:** v1uf — Station-Resolved Official Data Acquisition and Hydrometeorological Evidence Extraction

## Resumo Técnico

A v1uf transformou os datasets oficiais ano-específicos resolvidos na v1ue em
evidência hidrometeorológica real **por estação e janela temporal**. Os ZIPs anuais
oficiais do INMET foram baixados (streaming, allowlist, limite 150MB), as séries das
estações-alvo foram extraídas seletivamente, as coordenadas oficiais foram resolvidas
via catálogo INMET (com hash de proveniência), e as métricas de precipitação por janela
foram computadas. Nenhum ground truth, ground reference, label, overlay ou coordenada
inventada foi produzido. Toda a cadeia é fail-closed.

## Arquivos Criados/Modificados

### Scripts (6)
- `scripts/protocolo_c/revp_v1uf_station_resolved_acquisition.py`
- `scripts/protocolo_c/revp_v1uf_inmet_zip_selective_extractor.py`
- `scripts/protocolo_c/revp_v1uf_official_station_catalog_resolver.py`
- `scripts/protocolo_c/revp_v1uf_hydromet_window_metrics.py`
- `scripts/protocolo_c/revp_v1uf_station_evidence_integrity_audit.py`
- `scripts/protocolo_c/revp_v1uf_event_hydromet_scorecard.py`

### Configs (4)
- `configs/protocolo_c/v1uf_large_official_download_policy.yaml`
- `configs/protocolo_c/v1uf_station_catalog_sources.yaml`
- `configs/protocolo_c/v1uf_station_target_binding.yaml`
- `configs/protocolo_c/v1uf_hydromet_metrics_policy.yaml`

### Datasets versionáveis (9)
- `datasets/protocolo_c/v1uf_large_download_manifest.csv`
- `datasets/protocolo_c/v1uf_official_station_catalog_registry.csv`
- `datasets/protocolo_c/v1uf_station_binding_registry.csv`
- `datasets/protocolo_c/v1uf_station_series_asset_registry.csv`
- `datasets/protocolo_c/v1uf_hydromet_window_metrics_registry.csv`
- `datasets/protocolo_c/v1uf_station_evidence_integrity_registry.csv`
- `datasets/protocolo_c/v1uf_event_hydromet_scorecard.csv`
- `datasets/protocolo_c/v1uf_gate_delta_registry.csv`
- `datasets/protocolo_c/v1uf_next_actions_registry.csv`

### Docs (4, incluindo este handoff e o status)
- `docs/metodologia_cientifica/protocolo_c_v1uf_station_resolved_acquisition.md`
- `docs/metodologia_cientifica/protocolo_c_relatorio_v1uf_station_resolved_acquisition.md`
- `docs/metodologia_cientifica/protocolo_c_v1uf_completion_handoff.md` (este)
- `docs/metodologia_cientifica/protocolo_c_status_atual_v1uf.md`

### Testes (5)
- `tests/test_revp_v1uf_inmet_zip_selective_extractor.py`
- `tests/test_revp_v1uf_official_station_catalog_resolver.py`
- `tests/test_revp_v1uf_hydromet_window_metrics.py`
- `tests/test_revp_v1uf_station_evidence_integrity_audit.py`
- `tests/test_revp_v1uf_event_hydromet_scorecard.py`

### Auditoria de aceitação v1ue (2)
- `docs/metodologia_cientifica/protocolo_c_v1ue_acceptance_audit.md`
- `datasets/protocolo_c/v1ue_acceptance_audit.csv`

### Manifesto de artefatos versionáveis (1)
- `datasets/protocolo_c/v1uf_versionable_artifacts_manifest.csv`

## Assets Oficiais Baixados Localmente (não versionados)

Todos em `local_only/protocolo_c/evidence_raw/v1uf/INMET_BDMEP/<event>/`, git-ignored:

| Evento | ZIP | Tamanho | SHA256 (12) | Status |
|--------|-----|---------|-------------|--------|
| PET_2022_02_15 | 2022.zip | 90.362.801 B (~90MB) | 292d997e0589 | DOWNLOADED |
| PET_2024_03_21_28 | 2024.zip | 102.772.199 B (~102MB) | 0a7f89de5742 | DOWNLOADED |
| REC_2022_05_24_30 | 2022.zip | 90.362.801 B (~90MB) | 292d997e0589 | DOWNLOADED_CACHED (dedup por URL) |

## Séries Extraídas (não versionadas)

Em `local_only/protocolo_c/evidence_staging/v1uf/inmet/<event>/`, git-ignored:

| Evento | Estação | Arquivo | Tamanho | file_sha256 (12) |
|--------|---------|---------|---------|------------------|
| PET_2022_02_15 | A610 | INMET_SE_RJ_A610_PICO DO COUTO_..._2022.CSV | 697.712 B | e28299aaaaba |
| PET_2024_03_21_28 | A610 | INMET_SE_RJ_A610_PICO DO COUTO_..._2024.CSV | 750.650 B | 35ec6ac84ef5 |
| REC_2022_05_24_30 | A301 | INMET_NE_PE_A301_RECIFE_..._2022.CSV | 333.724 B | 2dcf4f1fb1bd |

## Coordenadas Oficiais Resolvidas

Via API INMET oficial (`apitempo.inmet.gov.br/estacoes/T`, 679 estações, SHA256 do
catálogo registrado), `coordinate_status=FROM_OFFICIAL_CATALOG`:

| Código | Nome (catálogo oficial) | Latitude | Longitude | Altitude |
|--------|-------------------------|----------|-----------|----------|
| A610 | PICO DO COUTO | -22,46472222 | -43,29138888 | 1777 m |
| A301 | RECIFE | -8,01888888 | -34,94416666 | 10 m |

Nota: o catálogo oficial é autoritativo — A610 é "PICO DO COUTO", não o nome sugerido
no hint da config. Nenhuma coordenada foi inventada.

## Métricas Hidrometeorológicas por Evento (janela central do evento)

| Evento | Total | Max diário | Cobertura | Status |
|--------|-------|-----------|-----------|--------|
| PET_2022_02_15 | 2,8 mm | 2,8 mm | 1.0 | COMPUTED |
| PET_2024_03_21_28 | 265,8 mm | 80,2 mm | 1.0 | COMPUTED |
| REC_2022_05_24_30 | — | — | 0.0 | INSUFFICIENT_COVERAGE |

## Interpretação Científica dos Achados

### PET_2024_03_21_28 — `TEMPORAL_HYDROMET_ANCHOR_CONFIRMED`
Precipitação oficial forte na estação A610/Pico do Couto durante a janela do evento
(265,8 mm acumulados, 80,2 mm de máximo diário). Isso **fortalece a plausibilidade
temporal/hidrometeorológica** do evento. **Não cria ground reference** porque ainda não
há geometria observada, overlay, revisão supervisora nem patch-level truth.

### PET_2022_02_15 — bloqueado (fenômeno misto + geometria ausente)
A série oficial A610/Pico do Couto existe e foi processada. A precipitação observada na
janela central foi **baixa (2,8 mm)** em relação à magnitude documentada do desastre.
A interpretação correta: **a estação oficial disponível não capturou adequadamente o
núcleo espacial/convectivo do evento** que afetou a área urbana de Petrópolis (a estação
fica a 1777 m de altitude, no Pico do Couto, distante do núcleo urbano atingido). Isso
**reforça por que estação meteorológica não pode ser usada como ground truth espacial**.
O evento continua bloqueado por fenômeno misto e ausência de geometria observada que
separe inundação de deslizamento.

### REC_2022_05_24_30 — `INSUFFICIENT_COVERAGE`
A série oficial A301/Recife foi extraída, mas a cobertura na janela do evento foi
insuficiente (sem observações válidas de precipitação no intervalo). O status correto é
`INSUFFICIENT_COVERAGE`. **Não fabricar dado, não preencher lacuna, não promover.** A
ausência de dado na estação não nega o evento — apenas significa que esta estação não
fornece âncora temporal utilizável para esta janela.

## Limitações

- A estação oficial mais próxima pode não representar o núcleo espacial do evento
  (caso PET_2022/Pico do Couto).
- Séries são **ano-específicas**, recortadas por janela; não são evento-específicas
  no sentido geométrico.
- Cobertura pode ser insuficiente por falhas de telemetria (caso REC_2022).
- Nenhuma geometria observada de inundação foi obtida nesta etapa.
- Suscetibilidade, quickview e portal genérico permanecem contexto, não ocorrência.

## Guardrails Permanentes (confirmados)

- `ground_truth_operational = false`
- `can_create_ground_reference = false`
- `can_create_training_label = false`
- `can_reopen_protocol_b = false`
- `dino_usage = SUPPORT_ONLY`
- `no_overlay_executed = true`
- `no_coordinates_invented = true`
- `supervisor_review_completed = false`
- estação oficial **não é** geometria de inundação
- precipitação forte **não é** label
- ausência de precipitação em estação **não nega** o evento
- evidência hidrometeorológica serve como **plausibilidade temporal**
- ground reference exige **geometria observada + overlay + revisão supervisora**
- Protocolo B permanece bloqueado

## Próximos Passos Recomendados — v1ug (não implementada)

`v1ug — Human Review Package and Formal Request Finalization` deve:
- montar pacotes de revisão humana por evento;
- consolidar pedidos formais para SGB/CPRM, DRM-RJ, Defesa Civil Petrópolis,
  COMPDEC Recife, Cemaden;
- buscar geometria observada real (pontos/polígonos/ocorrências georreferenciadas);
- preparar adjudicação supervisora;
- continuar sem label até haver ground reference validado.

## Resultado dos Testes

- v1uf: 29 testes passando
- v1ue (regressão): 35 testes passando
- Total v1ue+v1uf: 64 testes passando
- `git diff --check`: limpo
