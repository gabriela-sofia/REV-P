# Protocolo C — Aquisição e Auditoria de Vetores Observados Oficiais (v1if)

**Versão:** v1if  
**Status:** ACTIVE — GROUND_TRUTH BLOCKED  
**Etapa anterior:** v1ie — Auditoria de ground reference (10 candidatos locais, todos BLOCKED)  
**Data:** 2026-05-22  

---

## 1. Objetivo desta etapa

A v1if não cria ground truth. A v1if **busca ground truth real já existente**.

A distinção é fundamental: o Protocolo C não pode inferir, aproximar, derivar ou construir ground truth por proxy. Só pode promover evidência que:
- Seja vetorial (não apenas PDF ou imagem)
- Seja observada (não modelada, não baseada em risco, não suscetibilidade)
- Tenha data compatível com o evento alvo
- Seja rastreável a uma instituição oficial ou técnica
- Tenha fenômeno identificável (inundação/alagamento/enxurrada vs. deslizamento/corrida de massa)
- Permita binding ao nível de patch

Toda evidência que não satisfaça esses critérios permanece fora do escopo de ground truth operacional.

---

## 2. Diferença entre o que foi feito nas etapas anteriores e v1if

| Etapa | O que fez |
|---|---|
| v1ic | Separação fenomenológica textual: DRM-RJ (57p.) + NHESS — resultado: PARTIAL_SEPARATION |
| v1id | Registrou PKG_FR_PET_001 como REQUIRED_NOT_INGESTED |
| v1ie | Auditou 10 candidatos locais SGB/CPRM; todos BLOCKED no Gate 6 (sem data de evento) |
| **v1if** | **Buscou e baixou fontes vetoriais de repositórios oficiais; auditou cada ativo com 11 gates** |

---

## 3. Por que suscetibilidade não é ocorrência observada

**Mapa de suscetibilidade** representa onde um fenômeno *pode ocorrer* — modelagem baseada em relevo, solo, geologia, uso do solo e histórico de eventos. É um produto prospectivo ou retrospectivo geral, produzido para fins de planejamento de risco.

**Ocorrência observada** registra onde o fenômeno *ocorreu de fato*, em data específica, com evidência de campo, sensoriamento remoto ou relato verificável.

Usar suscetibilidade como ground truth de evento específico cria:
- Falsos positivos: áreas com suscetibilidade alta mas que não foram afetadas no evento
- Falsos negativos: áreas com suscetibilidade baixa que foram afetadas por fatores locais

A confusão entre suscetibilidade e ocorrência invalida qualquer claim preditivo ou avaliação de desempenho de modelo.

---

## 4. Por que PDF ou imagem não equivale automaticamente a vetor validado

Um mapa cartográfico oficial publicado como PDF ou imagem pode conter:
- Delimitações de áreas inundadas desenhadas sobre base cartográfica
- Pontos marcados com coordenadas aproximadas
- Isolinhas, polígonos ou hachuras representando fenômenos

Para ser utilizável como ground reference vetorial, esse mapa precisa:
1. Ser **georreferenciado** explicitamente (não apenas sobre base visual)
2. Ter os polígonos/pontos **vetorizados** manualmente
3. Ter os metadados documentados: escala, datum, projeção, data de levantamento
4. Passar por **dupla revisão supervisora** que confirme que a geometria reflete o fenômeno e a data correta
5. Ter a **incerteza posicional** documentada (erro de georeferenciamento manual é tipicamente 10–50m)

Essa cadeia de transformação (PDF → vetor) não é automática e não pode ser feita pelo script. Ela requer decisão supervisora qualificada e pode ser realizada como etapa futura (derivação cartográfica controlada), mas não como ground truth direto.

---

## 5. Como uma referência derivada pode ser construída cientificamente

Se não houver vetor observado disponível, é metodologicamente possível construir referência candidata derivada seguindo o protocolo abaixo:

1. **Georreferenciamento controlado** da imagem/mapa oficial usando pontos de controle conhecidos (cruzamento de ruas, marcos urbanos com coordenadas verificadas)
2. **Vetorização manual** por especialista com documentação do método e erro estimado
3. **Metadados completos**: datum, projeção, data do mapa original, escala, fonte, método de georref, RMSE
4. **Revisão dupla independente**: dois revisores confirmam que o polígono reflete o fenômeno correto e que a data é compatível
5. **Registro de incerteza**: o produto final é classificado como `DERIVED_CARTOGRAPHIC_REFERENCE_CANDIDATE`, não como `GROUND_REFERENCE_AUDITED`
6. **Protocolo de split/leakage**: antes de qualquer uso em ML, aplicar split temporal/espacial e verificar ausência de leakage

Esta cadeia garante rastreabilidade e honestidade epistemológica. Um vetor derivado de PDF ainda não é ground truth operacional — é uma referência candidata com incerteza documentada.

---

## 6. O que os PDFs do ZIP SGB/CPRM representam

O ZIP baixado em v1if (`anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip`, 20.9MB) contém **11 PDFs de avaliação técnica de campo**, um por bairro, produzidos pela equipe do SGB/CPRM entre 19/02/2022 e 02/03/2022 — logo após o evento principal de 15/02/2022.

**Valor metodológico desses PDFs:**
- Confirmam que houve avaliação técnica oficial pós-desastre por bairro
- Fornecem datas de campo que permitem inferir proximidade temporal ao evento
- Provavelmente contêm: fotografias, croquis, descrições de fenômenos por localidade, coordenadas de pontos de interesse
- São `CARTOGRAPHIC_LEAD_ONLY`: úteis para orientar solicitação institucional de vetores, mas não substituem geometria observada

**O que os PDFs não fornecem:**
- Geometria vetorial diretamente utilizável (são PDFs, não SHP/GPKG/KMZ)
- CRS explícito
- Campos de data de ocorrência estruturados
- Separação de fenômeno em formato de atributo consultável

Os PDFs confirmam que o SGB/CPRM possui dados de campo para as seguintes localidades pós-15/02/2022:
- Bairro Mosella (ANEXO-I, 19/02/2022)
- Bairro Moinho Preto (ANEXO-II, 19/02/2022)
- Bairro Serra Velha (ANEXO-III, 20/02/2022)
- Bairro Valparaíso, Rua Eugenio Barcelos (ANEXO-IV, 22/02/2022)
- Rua Teresa e imediações (ANEXO-V, 23/02/2022)
- Bairro Moinho Preto (revisita, ANEXO-VI, 24/02/2022)
- Bairro Mosella (revisita, ANEXO-VII, 24/02/2022)
- Estrada Velha e Vila Felipe (ANEXO-VIII, 25-26/02/2022)
- Bairro Sargento Boening (ANEXO-IX, 28/02/2022)
- Servidão Alépio Gomes da Costa (ANEXO-X, 01/03/2022)
- Bairro Quitandinha (ANEXO-XI, 02/03/2022)

**Ação recomendada**: solicitar ao SGB/CPRM os dados georeferenciados produzidos como base desses relatórios de campo — eles provavelmente existem em formato digital interno (KMZ, SHP ou planilha com coordenadas).

---

## 7. Atlas/S2ID ajuda a confirmar evento, mas não substitui vetor intraurbano

O **Atlas Digital de Desastres no Brasil (Sedec/MIDR)** e o **S2ID (Sistema Integrado de Informações sobre Desastres)** registram eventos por município: tipo de desastre, data de ocorrência, afetados, danos, decretos de emergência/calamidade.

Esses dados:
- **Confirmam** que o evento PET_2022_02_15 foi oficialmente registrado como desastre
- **Permitem** vincular o evento ao código municipal IBGE e à taxonomia COBRADE
- **Não fornecem** geometria intraurbana das áreas afetadas
- **Não permitem** patch-level binding (cada registro é municipal, não bairro/lote)

Para os fins do Protocolo C (binding a patches Sentinel de 10m × 10m), é necessário geometria com resolução espacial ao nível de bairro ou inferior.

---

## 8. O ZIP de anexos SGB/CPRM como pista prioritária

O fato de existirem 11 PDFs de campo, um por bairro, indica que:
1. A equipe SGB/CPRM coletou dados in loco em múltiplos bairros
2. Esses dados provavelmente foram digitalizados em KMZ ou SHP para uso interno
3. A solicitação formal ao SGB/CPRM deve pedir especificamente os **dados georreferenciados de campo** que embasaram os PDFs, não apenas os PDFs em si

Isso torna a solicitação formal ao SGB/CPRM a **próxima ação de maior prioridade** para PET_2022_02_15.

---

## 9. O GeoJSON de Curitiba (candidato local)

Um arquivo `zee_inundacoes_ocorrencia_curitiba.geojson` foi encontrado no workspace local (provável fonte: GeoCuritiba/Defesa Civil).

Auditoria:
- Geometria: POLYGON, WGS84
- Features: **1 polígono** (nível municipal, não bairro)
- Campos: `enxurradas_ocorr`, `enxurradas_afet`, `deslizament_ocorr`, `deslizament_afe` (contadores agregados)
- Data: **ausente** — não tem campo de data de evento

Decisão: BLOCKED no Gate 6 (event_date_available=FAIL) e Gate 11 (spatial_unit_usable_for_patch_binding=FAIL — 1 polígono municipal não resolve ao nível de patch).

Esse arquivo é útil como confirmação de que inundações/enxurradas são fenômenos documentados no município de Curitiba, mas não pode ser usado como ground truth de evento específico.

---

## 10. Gates de ground truth candidato (11 gates)

Para receber `ground_truth_status=CANDIDATE_OBSERVED_GROUND_TRUTH`, um ativo deve passar todos os 11 gates:

| Gate | Descrição |
|---|---|
| 1. official_or_institutional_source | Fonte oficial ou técnico-institucional |
| 2. raw_asset_traceable | URL ou referência auditável do ativo bruto |
| 3. geometry_available | Geometria presente e legível |
| 4. crs_available | Sistema de referência de coordenadas identificado |
| 5. geometry_valid | Geometria sem erros críticos de leitura |
| 6. event_date_available | Campo de data presente no atributo |
| 7. event_date_compatible | Data compatível com o evento alvo (ex: 2022-02-15) |
| 8. phenomenon_available | Campo de fenômeno/tipo presente |
| 9. phenomenon_is_observed_not_risk | Fenômeno é ocorrência observada, não risco/suscetibilidade |
| 10. hydrological_or_mass_movement_separable | Inundação e deslizamento são separáveis como atributo |
| 11. spatial_unit_usable_for_patch_binding | Geometria com granularidade compatível com patch Sentinel |

Se qualquer gate falhar:
- Gate 10 falha isolado → `BLOCKED_UNTIL_PHENOMENON_SEPARATION`
- Qualquer outro gate falha → `BLOCKED`

---

## 11. Invariantes após v1if

```
operational_ground_truth_status     = BLOCKED
ml_label_status                     = BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL
can_create_training_label           = false
can_reopen_protocol_b               = false
multimodal_status                   = HOLD
dino_usage_status                   = SUPPORT_ONLY
```

Se um ativo passar todos os 11 gates em uma iteração futura:
- Status avança para `CANDIDATE_OBSERVED_GROUND_TRUTH`
- Ainda requer revisão supervisora obrigatória
- Ainda requer protocolo de split/leakage
- Ainda não libera treino supervisionado

---

*Documento gerado como parte do Protocolo C de construção de corpus de referência terrestre para eventos de inundação urbana.*  
*Repositório: REV-P. Estágio: v1if. Decisão: BLOCKED. Nenhum claim operacional.*
