# Protocolo C — Auditoria de Proveniência da Fotointerpretação (v1ir)

## Contexto e Justificativa

### Por que v1ir existe

v1iq-R2 esgotou a evidência do DBF de camada original de feições poligonais de deslizamento fotointerpretadas:

- **FONTE = `Fotointerpretação`** — método de produção, não instituição
- **OBS completamente vazio** — 444 registros sem qualquer anotação
- **Nenhum campo temporal** — sem DATA, DATE, DT, ANO nos atributos
- **has_event_or_survey_date_in_field = FALSE** — confirmado em 444 × 8 campos

**Pergunta:** Será que os metadados XML (sidecars) do mesmo pacote SIG contêm:
- Data da imagem de satélite/aerofoto usada para fotointerpretação?
- Data do levantamento de campo que produziu as feições de deslizamento?
- Qualquer referência ao evento de 2022-02-15?
- Identidade da imagem base usada?

**v1ir responde:** Auditando sidecars de camadas irmãs do mesmo pacote SIG CPRM/SGB.

### Contexto de v1ir em Relação ao Histórico

| Versão | Resultado | Bloqueio |
|--------|-----------|----------|
| v1ij | Candidato bloqueado | feições de deslizamento cumulativas sem data |
| v1ik | Temporal status: BLOCKED | Sem data específica |
| v1ip | STRONG_COMPOSITE_BUT_WEAK_TEMPORAL | Sem linkage explícito |
| v1iq-R2 | STRONG_COMPOSITE_BUT_TEMPORAL_LINK_WEAK | DBF histórico (2015); sem campo DATA; OBS vazio |
| **v1ir** | **STRONG_COMPOSITE_BUT_TEMPORAL_LINK_WEAK** | **Fotointerpretação de 2013; sem imagem base; sem 2022** |

---

## O Que v1ir Audita

### 1. Sidecars das Camadas Irmãs (Feicoes/)

Camadas do mesmo diretório e mesmo SIG:

| Sidecar | Relação com camada de feições poligonais de deslizamento fotointerpretadas |
|---------|---------------------------|
| `sidecar original de pontos de feições de deslizamento fotointerpretadas` | Proxy direto — mesma família de feições de deslizamento |
| `Feicoes_Erosivas_P.shp.xml` | Irmã — mesmo processo de mapeamento |
| `Deposito_Acumulacao_Encosta_A.shp.xml` | Irmã — geomorfologia |
| `Campo_de_Blocos_A.shp.xml` | Irmã — feições de campo |
| `Lineamento_L.shp.xml` | Irmã — estrutural |
| `Paredao_Rochoso_A.shp.xml` | Irmã — geomorfologia |

**Nota:** `camada original de feições poligonais de deslizamento fotointerpretadas` é o único layer sem `.shp.xml` próprio na pasta Feicoes.

### 2. Sidecars do Pacote Completo

| Sidecar | Conteúdo esperado |
|---------|------------------|
| `Pontos_de_Campo_P.shp.xml` | Levantamento de campo — DATA="Maio/2013" |
| `Movimento_de_Massa_A.shp.xml` | Susceptibilidade — CRS confirma EPSG:31983 |
| `Enxurrada_A.shp.xml` | Susceptibilidade |
| `Padroes_de_Relevo_A.shp.xml` | Relevo |
| `metadata.xml` (MDE) | Modelo Digital de Elevação — Kit_trabalho_2013 |

### 3. Registries Existentes (v1ij, v1iq)

- `cicatriz_area_ground_reference_dossier.csv` → decisão de v1iq
- `cicatriz_area_ground_reference_gate_matrix.csv` → gate_event_or_survey_date

---

## Resultados: O Que Foi Encontrado

### camada original de feições poligonais de deslizamento fotointerpretadas — Sidecar Próprio

**Encontrado:** Não existe. `camada original de feições poligonais de deslizamento fotointerpretadas` é o único layer sem `.shp.xml` na pasta Feicoes.

### sidecar original de pontos de feições de deslizamento fotointerpretadas (Proxy Direto)

- **CreaDate:** 20150122
- **FONTE atribuída:** `Fotointerpretação` (via CalculateField em 20150122)
- **Copy date de feições de deslizamento:** 20130822 — origem: `D:\SUSCETIBILIDADE\Correções_Kits_2013\...`
- **Imagem base:** NÃO documentada — sem referência a imagem, satélite, ortofoto ou aerofoto
- **"2022":** NÃO encontrado

### Pontos_de_Campo_P.shp.xml

- **CreaDate:** 20150123
- **DATA:** `Maio/2013` — levantamento de campo realizado em Maio de 2013
- **Imagem base:** NÃO documentada
- **"2022":** NÃO encontrado

### Feicoes_Erosivas_P.shp.xml

- **CreaDate:** 20150122
- **Process date:** 20130822 (mesmo do contexto de feições de deslizamento)
- **"2022":** NÃO encontrado

### Deposito_Acumulacao_Encosta_A.shp.xml

- **Process date:** 20130412 — início do mapeamento geomorfológico
- **"2022":** NÃO encontrado

### MDE metadata.xml

- **Origem:** `Kit_trabalho_2013`
- **Contém "Imagens/MDE"** — referência a pasta de MDE, não a imagem de fotointerpretação
- **"2022":** NÃO encontrado

---

## Auditoria de Termos por Categoria

| Categoria | Termos Buscados | Encontrados em XMLs | Associados a 2022 |
|-----------|-----------------|--------------------|--------------------|
| **Fotointerpretação** | fotointerpret*, ortofoto, aerofoto, imagem | `Fotointerpretação` em camada de pontos de feições de deslizamento fotointerpretadas | NÃO |
| **Imagem base** | imagem orbital, ecw, tif, satélite, Pleiades | — | NÃO |
| **Temporal** | 2013, 2015, maio/2013, data | `Maio/2013`, `20130822`, `20150122` | NÃO |
| **Evento** | 2022, 15/02/2022, fevereiro | — | NÃO |
| **Instituição** | CPRM, SGB, DICART | Inferido de paths | NÃO explícito |

---

## Decisão de Proveniência (v1ir)

```
provenance_id                    = PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA_PHOTOINTERP_PROVENANCE_V1IR
candidate_asset_name             = camada original de feições poligonais de deslizamento fotointerpretadas
source_field_value               = Fotointerpretação
source_method                    = Fotointerpretação
source_institution               = CPRM_INFERRED_FROM_PACKAGE_PATHS
imagery_or_base_name_sanitized   = NOT_DOCUMENTED
imagery_date_documented          = NOT_DOCUMENTED
mapping_date_documented          = 2013-08-22 (via sidecar original de pontos de feições de deslizamento fotointerpretadas, proxy)
survey_date_documented           = Maio/2013 (via Pontos_de_Campo_P.shp.xml, proxy)
event_date_documented            = NOT_DOCUMENTED
temporal_reference_type          = survey_and_mapping_dates_documented_but_2013
temporal_link_strength           = DOCUMENTED_2013_NOT_2022
package_lineage_strength         = MODERATE_2013_KIT
source_authority_strength        = CPRM_INFERRED_HIGH
phenomenon_match                 = STRONG
region_match                     = STRONG
can_update_ground_reference_status = NO
promotion_decision               = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK
can_be_ground_reference_candidate  = NO
can_be_operational_ground_truth    = NO
can_create_training_label          = NO
can_train_model                    = NO
can_reopen_protocol_b              = NO
primary_blocker                  = Fotointerpretação documentada em 2013 (não em resposta ao evento de 2022); sem imagem base referenciada; sem data post-2022 em nenhum metadado
```

---

## Por Que NÃO é GROUND_REFERENCE_CANDIDATE

### Regra de Promoção

Para ser `GROUND_REFERENCE_CANDIDATE` via auditoria de fotointerpretação, seria necessário:

```
método/fonte documentado      ✓ (Fotointerpretação via sidecar original de pontos de feições de deslizamento fotointerpretadas)
+ imagem base documentada     ✗ (NÃO — nenhum XML referencia imagem)
+ data de imagem documentada  ✗ (NÃO — NOT_DOCUMENTED)
+ evento 2022 confirmado      ✗ (NÃO — "2022" ausente em todos os XMLs)
+ CPRM autoridade             ✓ (inferida de paths)
= GROUND_REFERENCE_CANDIDATE  ✗ (NÃO ATINGIDO)
```

### O Que Foi Encontrado vs. O Que Seria Necessário

| Evidência | Encontrada | Necessária para promoção |
|-----------|-----------|--------------------------|
| Método documentado (Fotointerpretação) | ✓ SIM | ✓ |
| Imagem base identificada | ✗ NÃO | ✓ |
| Data da imagem | ✗ NÃO | ✓ |
| Data de levantamento de campo | ✓ Maio/2013 | ✓ (mas deve ser pós-2022) |
| Data de mapeamento | ✓ 2013-08-22 (proxy) | ✓ (mas deve ser pós-2022) |
| "2022" em algum XML | ✗ NÃO | ✓ |
| Vínculo explícito com 2022-02-15 | ✗ NÃO | ✓ |

---

## Evidência Mínima para Superar o Bloqueio

Documento oficial SGB/CPRM declarando que as feições de deslizamento em camada original de feições poligonais de deslizamento fotointerpretadas foram mapeadas por fotointerpretação de imagem adquirida após 2022-02-15, especificamente para o evento de Petrópolis 2022, com data de imagem e/ou data de levantamento documentadas.

Essa evidência não está disponível com os arquivos auditados.

---

## Invariantes

- ❌ NÃO é ground truth operacional
- ❌ NÃO é label de treino
- ❌ NÃO libera modelo
- ❌ NÃO implica Protocolo B
- ❌ NÃO houve e-mail, solicitação ou vistoria

```
can_be_operational_ground_truth  = NO  (sempre)
can_create_training_label        = NO  (sempre)
can_train_model                  = NO  (sempre)
can_reopen_protocol_b            = NO  (sempre)
can_be_ground_reference_candidate = NO
```

---

## Outputs Gerados

### Locais (`local_runs/protocolo_c/v1ir/`)

| Arquivo | Conteúdo |
|---------|----------|
| `v1ir_photointerpretation_source_inventory.csv` | Inventário do pacote SIG |
| `v1ir_sidecar_metadata_audit.csv` | Metadados extraídos por sidecar |
| `v1ir_package_lineage_audit.csv` | Lineagem do pacote (estatísticas) |
| `v1ir_source_imagery_temporal_audit.csv` | Auditoria temporal por sidecar |
| `v1ir_documentary_provenance_linkage.csv` | Linkagem documental |
| `v1ir_ground_reference_update_decision.csv` | Decisão de promoção |
| `v1ir_summary.json` | Resumo estruturado |
| `v1ir_qa.csv` | Verificações de QA |

### Públicos (`datasets/`)

| Arquivo | Conteúdo |
|---------|----------|
| `cicatriz_area_photointerpretation_provenance_registry.csv` | Registry de proveniência |
| `schemas/cicatriz_area_photointerpretation_provenance_schema.csv` | Schema de campos |

---

**Versão:** v1ir — Photointerpretation Provenance and Source Imagery Audit  
**Resultado:** camada original de feições poligonais de deslizamento fotointerpretadas = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK  
**Fotointerpretação:** documentada em 2013 — levantamento Maio/2013, mapeamento Agosto/2013  
**Imagem base:** NÃO documentada em nenhum XML do pacote  
**Evento 2022:** NÃO encontrado em nenhum XML do pacote  
**Decisão:** bloqueio temporal confirmado por terceira camada de evidência  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
