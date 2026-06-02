# Relatório Científico: Auditoria de Proveniência da Fotointerpretação (v1ir)

## Resumo Executivo

**RESULTADO PRINCIPAL: camada original de feições poligonais de deslizamento fotointerpretadas PERMANECE COMO STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK**

v1ir executou auditoria dos metadados XML (sidecars) do pacote SIG CPRM/SGB para verificar
se FONTE="Fotointerpretação" possui proveniência suficiente para promover camada original de feições poligonais de deslizamento fotointerpretadas
a GROUND_REFERENCE_CANDIDATE.

**Conclusão:** Fotointerpretação documentada em 2013. Imagem base não documentada. Nenhum XML
referencia o evento de 2022-02-15. O bloqueio temporal é confirmado por terceira camada
independente de evidência.

**Status:** AUDITORIA COMPLETA (v1ir)  
**Candidato:** camada original de feições poligonais de deslizamento fotointerpretadas  
**Região:** Petrópolis, RJ (PET)  
**Evento de referência:** 2022-02-15  
**Decisão:** **STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK**  
**Promoção para GROUND_REFERENCE_CANDIDATE:** NÃO  
**Motivo:** Fotointerpretação de 2013; imagem base não documentada; "2022" ausente em todos os XMLs  
**Camadas de evidência:** 3 (DBF atributos v1iq-R2 + XML sidecars v1ir + registry v1ij)

---

## 1. Pergunta Central de v1ir

> A FONTE="Fotointerpretação" de camada original de feições poligonais de deslizamento fotointerpretadas possui metadado/proveniência
> suficiente para transformá-lo em GROUND_REFERENCE_CANDIDATE?

**Resposta: NÃO.**

---

## 2. O Que Foi Auditado

### 2.1 Sidecars das Camadas Irmãs (Feicoes/)

| Sidecar | CreaDate | Event 2022 | Imagem Base | Observação |
|---------|----------|------------|-------------|------------|
| `sidecar original de pontos de feições de deslizamento fotointerpretadas` | 20150122 | NÃO | NÃO | FONTE="Fotointerpretação"; copy 20130822 |
| `Feicoes_Erosivas_P.shp.xml` | 20150122 | NÃO | NÃO | Process date 20130822 |
| `Deposito_Acumulacao_Encosta_A.shp.xml` | — | NÃO | NÃO | Geomorf 20130412 |
| `Campo_de_Blocos_A.shp.xml` | — | NÃO | NÃO | Mesmo pacote 2013 |
| `Lineamento_L.shp.xml` | — | NÃO | NÃO | Mesmo pacote 2013 |
| `Paredao_Rochoso_A.shp.xml` | — | NÃO | NÃO | Mesmo pacote 2013 |

**`camada original de feições poligonais de deslizamento fotointerpretadas.xml`:** NÃO EXISTE — único layer sem sidecar na pasta Feicoes.

### 2.2 Sidecars do Pacote Completo

| Sidecar | CreaDate | Evidência Chave | Event 2022 |
|---------|----------|----------------|------------|
| `Pontos_de_Campo_P.shp.xml` | 20150123 | **DATA="Maio/2013"** — levantamento de campo | NÃO |
| `Movimento_de_Massa_A.shp.xml` | 20151118 | EPSG:31983 confirmado | NÃO |
| `metadata.xml` (MDE) | — | Kit_trabalho_2013 | NÃO |

### 2.3 Achado Principal

**Em nenhum dos XMLs auditados foi encontrado:**
- Referência a imagem de satélite, ortofoto ou aerofoto usada para fotointerpretação
- Data de imagem/aquisição
- O termo "2022" ou qualquer referência ao evento de 2022-02-15
- Vínculo explícito entre as feições de deslizamento e o evento de Petrópolis

---

## 3. Evidências por Sidecar Chave

### sidecar original de pontos de feições de deslizamento fotointerpretadas (Proxy Direto)

```
CreaDate:         20150122
FONTE:            "Fotointerpretação"  ← método de produção documentado
copy_date:        20130822             ← feições de deslizamento copiadas de Kits_Executores_2013
institution:      CPRM_implied         ← inferido de paths (não explícito)
imagery_base:     NOT_DOCUMENTED       ← nenhuma imagem referenciada
event_2022:       NO
```

**O que confirma:** método (fotointerpretação) e data de mapeamento (proxy Agosto/2013).  
**O que não confirma:** imagem base, data de imagem, evento de 2022.

### Pontos_de_Campo_P.shp.xml

```
CreaDate:         20150123
DATA:             "Maio/2013"  ← levantamento de campo documentado
event_2022:       NO
```

**O que confirma:** levantamento de campo em Maio/2013.  
**O que não confirma:** vínculo com 2022.

---

## 4. Decisão de Proveniência

### Lógica de Promoção

```
GROUND_REFERENCE_CANDIDATE requer:
  (1) método documentado           ✓ Fotointerpretação (via proxy)
  (2) imagem base documentada      ✗ NOT_DOCUMENTED em todos os XMLs
  (3) data de imagem pós-2022      ✗ NOT_DOCUMENTED
  (4) "2022" em metadado           ✗ NÃO encontrado em nenhum XML
  (5) CPRM autoridade              ✓ inferida

→ Promoção: BLOQUEADA (falha em 2, 3, 4)
```

### Resultado

```
promotion_decision_after_provenance_audit = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK
can_be_ground_reference_candidate         = NO
temporal_link_strength                    = DOCUMENTED_2013_NOT_2022
imagery_date_documented                   = NOT_DOCUMENTED
event_date_documented                     = NOT_DOCUMENTED
survey_date_documented                    = Maio/2013 (proxy, 2013 — não 2022)
mapping_date_documented                   = 2013-08-22 (proxy, 2013 — não 2022)
primary_blocker                           = Fotointerpretação documentada em 2013
                                            (não em resposta ao evento de 2022);
                                            sem imagem base referenciada;
                                            sem data post-2022 em nenhum metadado
```

---

## 5. Três Camadas de Evidência do Bloqueio

O bloqueio temporal de camada original de feições poligonais de deslizamento fotointerpretadas é agora confirmado por três camadas independentes:

| Camada | Versão | Evidência | Bloqueio Confirmado |
|--------|--------|-----------|---------------------|
| **Registry v1ij** | v1ij | blocking_reason: feicoes_deslizamento_cumulativas_sem_data_especifica | ✓ |
| **DBF atributos** | v1iq-R2 | 444 registros × 8 campos; OBS vazio; sem campo DATA; has_event_or_survey_date=NO | ✓ |
| **XML sidecars** | v1ir | Fotointerpretação de 2013; sem imagem base; "2022" ausente em todos os XMLs | ✓ |

---

## 6. Evidência Mínima para Superar o Bloqueio

Para ser `GROUND_REFERENCE_CANDIDATE`, seria necessário:

> Documento oficial SGB/CPRM declarando que as feições de deslizamento em camada original de feições poligonais de deslizamento fotointerpretadas foram
> mapeadas por fotointerpretação de imagem adquirida **após 2022-02-15**, especificamente
> para o evento de Petrópolis 2022, com data de imagem e/ou data de levantamento
> documentadas nos metadados.

Essa evidência não está disponível com os arquivos auditados.

---

## 7. Invariantes

```
can_be_operational_ground_truth  = NO  (invariante absoluto — sempre)
can_create_training_label        = NO  (invariante absoluto — sempre)
can_train_model                  = NO  (invariante absoluto — sempre)
can_reopen_protocol_b            = NO  (invariante absoluto — sempre)
can_be_ground_reference_candidate = NO
```

- [x] Nenhum label foi criado
- [x] Nenhum modelo foi treinado
- [x] Protocolo B não foi reaberto
- [x] Sem inventar vínculo
- [x] Sem aceitar nome de pasta como evidência forte
- [x] Sem usar data de sistema como evidência
- [x] Sem aceitar risco como observado
- [x] Sem e-mail, solicitação ou vistoria
- [x] Sem path privado em arquivos públicos
- [x] local_runs/ não versionado

---

## 8. O Que v1ir Acrescentou

1. **Fotointerpretação documentada (2013)** — sidecar original de pontos de feições de deslizamento fotointerpretadas confirma FONTE="Fotointerpretação" e copy date 20130822. O método é real e rastreável. Mas a data é de 2013, não de 2022.

2. **Levantamento de campo documentado (Maio/2013)** — Pontos_de_Campo_P.shp.xml confirma DATA="Maio/2013". O levantamento existiu e está documentado. Mas é de 2013.

3. **Imagem base NÃO documentada** — Nenhum XML do pacote referencia a imagem de satélite, ortofoto ou aerofoto usada. O método (fotointerpretação) está documentado, mas o objeto da interpretação (a imagem) não.

4. **"2022" ausente em todos os XMLs** — Verificado em todos os sidecars do pacote. Nenhum referencia o evento de 2022-02-15 ou qualquer data posterior a 2015.

5. **Terceira camada de confirmação** — O bloqueio temporal, já confirmado via registry v1ij e atributos DBF (v1iq-R2), é agora confirmado independentemente via metadados XML.

6. **Nome de pasta não é evidência** — O diretório é referenciado como "SIG pós-desastre Petrópolis 2022" no RIGeo, mas os dados XML confirmam produção de 2013-2015. Nomes de pasta e títulos de repositório não são equivalentes a data de evento nas feições.

---

## 9. Comparação com v1iq

| Aspecto | v1iq-R2 | v1ir |
|---------|---------|------|
| **Evidência auditada** | 444 registros DBF × 8 campos | Sidecars XML do pacote SIG |
| **FONTE documentada** | `Fotointerpretação` (campo DBF) | `Fotointerpretação` (CalculateField XML) |
| **Data de mapeamento** | Não nos atributos | 2013-08-22 (via copy date proxy) |
| **Data de levantamento** | Não nos atributos | Maio/2013 (via Pontos_de_Campo) |
| **"2022" encontrado** | NÃO | NÃO |
| **Imagem base** | Não auditado | NOT_DOCUMENTED |
| **Decisão** | STRONG_COMPOSITE_BUT_TEMPORAL_LINK_WEAK | STRONG_COMPOSITE_BUT_TEMPORAL_LINK_WEAK |

**O que mudou:** v1ir adiciona datas concretas de 2013 (levantamento e mapeamento) e confirma que a imagem base não está documentada — tornando o bloqueio mais preciso e rastreável.

---

## 10. Status Final e Commitabilidade

**Commitável?** SIM, como registro de análise metodológica.

**O que incluir no commit:**
- `scripts/protocolo_c/revp_v1ir_photointerpretation_provenance_audit.py`
- `tests/test_revp_v1ir_photointerpretation_provenance_audit.py`
- `docs/metodologia_cientifica/protocolo_c_auditoria_proveniencia_fotointerpretacao_v1ir.md`
- `docs/metodologia_cientifica/protocolo_c_relatorio_proveniencia_fotointerpretacao_v1ir.md`
- `datasets/cicatriz_area_photointerpretation_provenance_registry.csv`
- `datasets/schemas/cicatriz_area_photointerpretation_provenance_schema.csv`

**O que NÃO incluir:**
- `local_runs/protocolo_c/v1ir/` (outputs locais)
- Nenhum arquivo `.tif`, `.shp`, `.dbf`, `.xml` do PROJETO

---

**Data de Execução:** 2026-05-24  
**Etapa:** v1ir — Photointerpretation Provenance and Source Imagery Audit  
**Decisão Final:** camada original de feições poligonais de deslizamento fotointerpretadas = STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK  
**Promoção para GROUND_REFERENCE_CANDIDATE:** NÃO  
**Bloqueio:** Fotointerpretação de 2013; imagem base não documentada; "2022" ausente em todos os XMLs  
**Camadas de evidência:** 3 (v1ij + v1iq-R2 + v1ir)  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
