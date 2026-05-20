# Protocolo C — Revisão Pré-Overlay Evento–Patch

## Identificação

- **pre_overlay_review_id**: [PRE_OVERLAY_REVIEW_ID]
- **preflight_id**: [PREFLIGHT_ID]
- **observed_event_id**: [OBSERVED_EVENT_ID]
- **Patch(es) no escopo**: [PATCH_IDS]
  - Exemplo: REC_P001; REC_P002; REC_P003; REC_P004
- **Região**: [REGIAO]
- **Data de revisão**: [DATA_DE_REVISAO]
- **Revisor (função)**: [REVISOR_FUNCAO]

---

## 1. Estado das dependências

| Dependência | dependency_id | Status atual |
|---|---|---|
| SOURCE_GEOMETRY | [DEP_ID] | [STATUS] |
| MANUAL_GEOCODING | [DEP_ID] | [STATUS] |
| LICENSE_PROVENANCE | [DEP_ID] | [STATUS] |
| SENTINEL_TEMPORAL_SEARCH | [DEP_ID] | [STATUS] |
| HUMAN_REVIEW | [DEP_ID] | [STATUS] |
| PHENOMENON_SEPARATION (se aplicável) | [DEP_ID] | [STATUS] |

---

## 2. Geometria disponível

- **Geometria vetorial disponível**: [GEOMETRIA_DISPONIVEL]
  - Controles: SIM / NAO / PARCIAL
- **Fonte da geometria**: [FONTE_GEOMETRIA]
- **CRS verificado**: [CRS_VERIFICADO]
  - Controles: SIM / NAO / NAO_APLICAVEL
- **Licença verificada para uso**: [LICENCA_VERIFICADA]
  - Controles: SIM / NAO / PENDENTE

---

## 3. Alinhamento temporal Sentinel

- **Sentinel_status**: [SENTINEL_STATUS]
  - Controles: NOT_ACQUIRED / ASSET_IDENTIFIED_NOT_DOWNLOADED / ASSET_AVAILABLE / CLOUD_RISK_HIGH / UNKNOWN
- **temporal_window_id**: [TEMPORAL_WINDOW_ID]
- **Asset Sentinel-1 identificado**: [ASSET_S1_IDENTIFICADO]
- **Asset Sentinel-2 identificado**: [ASSET_S2_IDENTIFICADO]
- **Cobertura de nuvem verificada**: [COBERTURA_NUVEM]

---

## 4. Evidência espacial

- **event_spatial_precision**: [EVENT_SPATIAL_PRECISION]
- **Evidência espacial documental**: [EVIDENCIA_ESPACIAL_DOCUMENTAL]
- **Localidades geocodificadas**: [LOCALIDADES_GEOCODIFICADAS]
  - Listar geocoding_target_ids com status de cada geocodificação
- **Separação de fenômenos realizada**: [SEPARACAO_FENOMENOS]
  - Controles: SIM / NAO / NAO_APLICAVEL
  - ATENÇÃO: para Petrópolis sempre NAO_APLICAVEL é inválido — separação obrigatória

---

## 5. Avaliação de condições para overlay futuro

- **Pode executar overlay futuro**: [PODE_EXECUTAR_OVERLAY_FUTURO]
  - Controles: SIM / NAO / PENDENTE_DEPENDENCIAS
  - Overlay futuro exige: geometria disponível + CRS verificado + licença verificada + alinhamento temporal confirmado + revisão humana autorizada + separação de fenômenos (se aplicável)

---

## 6. Decisões de promoção — sempre false nesta etapa

- **Pode promover a ground reference**: false
  - Esta revisão pré-overlay não cria as condições para promoção a ground reference. Promoção exige overlay executado; validação de relação patch-evento; gates G5–G9 abertos; protocolo supervisionado específico.
- **Pode gerar label de treino**: false
  - Nenhum label de treino pode ser criado nesta etapa ou como consequência direta desta revisão.
- **Pode reabrir Protocolo B**: false
  - O Protocolo B permanece BLOCKED. Esta revisão não altera esse status.
- **Pipeline multimodal pode avançar**: false
  - O pipeline multimodal permanece em HOLD. Esta revisão não altera esse status.

---

## 7. Decisão da revisão

- **Decisão**: [DECISAO]
  - Opções:
    - `READY_FOR_FUTURE_OVERLAY` — todas as dependências críticas resolvidas; overlay pode ser executado em etapa futura específica
    - `REQUEST_SOURCE_GEOMETRY` — geometria vetorial ainda não disponível ou não verificada; overlay não pode ser agendado
    - `REQUEST_MANUAL_GEOCODING` — localidades ainda não geocodificadas; overlay bloqueado pendente de geocodificação
    - `REQUEST_LICENSE_REVIEW` — licença de geometria ou fonte não verificada para uso operacional
    - `BLOCK_PATCH_LINKING` — evidência espacial insuficiente para qualquer tentativa de overlay; bloqueio formal registrado
    - `BLOCK_OPERATIONAL_USE` — uso operacional bloqueado por limitação metodológica (ex: separação de fenômenos não realizada; G4=PARTIAL não resolvido)

---

## 8. O que esta revisão não pode afirmar

- Sempre incluir: ground truth operacional estabelecido; flood detection confirmado; flood prediction; flood label criado; training label criado; supervised training autorizado; patch validado como inundado; relação patch-evento confirmada; overlay executado; produto Copernicus regional como ground reference local; Sentinel como prova isolada de inundação

---

## 9. Próxima ação

- **Próxima ação**: [PROXIMA_ACAO]
- **Dependências ainda abertas**: [DEPENDENCIAS_ABERTAS]
- **Prazo estimado para retomada**: [PRAZO]

---

## 10. Observações

[OBSERVACOES]
