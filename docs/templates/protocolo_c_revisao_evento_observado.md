# Template de Revisão Humana — Evento Observado Candidato

## Instruções de preenchimento

Este template deve ser preenchido por pesquisador ou especialista após revisão direta da evidência documental do evento.
- Não inserir dado pessoal
- Não inserir path privado
- Não declarar ground truth operacional com base apenas nesta revisão
- Não criar label de treino a partir desta revisão

---

## Identificação da revisão

- **REVIEW_ID**: [REVIEW_ID]
- **OBSERVED_EVENT_ID**: [OBSERVED_EVENT_ID]
- **REGIAO**: [REGIAO]
- **REVISOR_FUNCAO**: [REVISOR_FUNCAO]
  - Ex: METHODOLOGICAL_REVIEWER / GIS_REVIEWER / REMOTE_SENSING_REVIEWER / FUTURE_EXTERNAL_REVIEWER
- **DATA_DA_REVISAO**: [DATA_DA_REVISAO]

---

## Fontes revisadas

- **FONTES_REVISADAS**: [FONTES_REVISADAS]
  - Listar todas as fontes revisadas com URL ou identificador de arquivo local

---

## Confirmações documentais

- **EVENTO_CONFIRMADO_DOCUMENTALMENTE**: [EVENTO_CONFIRMADO_DOCUMENTALMENTE]
  - true / false / parcial
- **DATA_CONFIRMADA**: [DATA_CONFIRMADA]
  - Data ou janela temporal exatamente como descrita na fonte revisada
- **LOCALIDADE_CONFIRMADA**: [LOCALIDADE_CONFIRMADA]
  - Localidade exatamente como descrita na fonte revisada

---

## Evidência espacial e temporal

- **GEOMETRIA_EXISTE**: [GEOMETRIA_EXISTE]
  - true / false / parcial — se existe geometria vetorial adquirida localmente
- **PRECISAO_TEMPORAL**: [PRECISAO_TEMPORAL]
  - EXACT_DATE / SHORT_WINDOW / MULTI_DAY_WINDOW / IMPRECISE
- **PRECISAO_ESPACIAL**: [PRECISAO_ESPACIAL]
  - MUNICIPAL / DISTRICT / NEIGHBORHOOD / STREET_OR_POINT / RIVER_CORRIDOR / TECHNICAL_MAP / PARTIAL

---

## Fenomenologia

- **FENOMENO_PRINCIPAL**: [FENOMENO_PRINCIPAL]
  - Preencher: chuva_extrema_alagamento / chuva_extrema_inundacao / desastre_hidrometeorologico / evento_misto
- **FENOMENOS_CONCORRENTES**: [FENOMENOS_CONCORRENTES]
  - Listar outros processos presentes no evento que devem ser separados metodologicamente
  - Ex: deslizamento; enxurrada; transbordamento de rio; inundação ribeirinha
  - Se presentes, separação metodológica é obrigatória antes de qualquer uso operacional

---

## Decisões de avanço

- **PODE_AVANCAR_PARA_SOURCE_REVIEW**: [PODE_AVANCAR_PARA_SOURCE_REVIEW]
  - true / false
- **PODE_AVANCAR_PARA_PATCH_LINKING**: [PODE_AVANCAR_PARA_PATCH_LINKING]
  - true / false — false sem geometria validada e overlay executado
- **PODE_PROMOVER_GROUND_REFERENCE**: [PODE_PROMOVER_GROUND_REFERENCE]
  - false — proibido nesta etapa sem satisfação de G1–G9
- **PODE_GERAR_LABEL**: [PODE_GERAR_LABEL]
  - false — proibido nesta etapa e exige protocolo supervisionado específico posterior

---

## Justificativa e próxima ação

- **JUSTIFICATIVA**: [JUSTIFICATIVA]
  - Descrever a razão da decisão com base na evidência revisada

- **PROXIMA_ACAO**: [PROXIMA_ACAO]
  - Descrever o próximo passo concreto necessário

---

## Decisão formal

Marcar apenas uma das opções abaixo:

- [ ] ACCEPT_AS_OBSERVED_EVENT_CANDIDATE
- [ ] REQUEST_MORE_SPATIAL_EVIDENCE
- [ ] REQUEST_MORE_TEMPORAL_EVIDENCE
- [ ] REQUEST_LICENSE_REVIEW
- [ ] BLOCK_FOR_PATCH_LINKING
- [ ] BLOCK_FOR_OPERATIONAL_USE
