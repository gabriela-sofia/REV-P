# Protocolo C — Relatório de Aquisição e Auditoria de Vetores Observados (v1if)

**Versão:** v1if  
**Status:** BLOQUEADO — Nenhum vetor observado com ground truth operacional  
**Data:** 2026-05-22  

---

## 1. Decisões metodológicas obrigatórias

> **O REV-P não possui ground truth operacional validado neste estágio.**

> **A próxima evolução real é encontrar ou solicitar formalmente geometria observada com vínculo temporal ao evento — não treinar modelo com proxy.**

> **Treino supervisionado permanece bloqueado até haver geometria com vínculo temporal, separação de fenômeno, revisão supervisora e protocolo de split/leakage.**

---

## 2. Fontes verificadas em v1if

| ID | Fonte | Tipo | Status de acesso |
|---|---|---|---|
| OBS_PET_001 | SGB/CPRM — Relatório Técnico Petrópolis PDF | PDF_REPORT | DOWNLOAD_OK (4.3MB) |
| OBS_PET_002 | SGB/CPRM — Anexos ZIP pós-desastre Petrópolis | ZIP_ARCHIVE | DOWNLOAD_OK (20.9MB) |
| OBS_PET_003 | DRM-RJ/NADE — PKG_FR_PET_001 | UNKNOWN | PENDING_FORMAL_REQUEST |
| OBS_PET_004 | Defesa Civil Municipal Petrópolis | UNKNOWN | PENDING_FORMAL_REQUEST |
| OBS_REC_001 | COMPDEC/Defesa Civil PE — PKG_FR_REC_002 | UNKNOWN | PENDING_FORMAL_REQUEST |
| OBS_CUR_001 | GeoCuritiba/Defesa Civil (local) | VECTOR_GEOJSON | ALREADY_LOCAL (auditado) |

---

## 3. O ZIP oficial SGB/CPRM foi encontrado e baixado?

**Sim.** O arquivo `anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip` (20,944,977 bytes / 20.9MB) foi baixado com sucesso da URL oficial:  
`https://rigeo.sgb.gov.br/bitstreams/23d77158-e00c-4a99-87c7-0bb1d3ecb7fd/download`

Repositório de origem: `https://rigeo.sgb.gov.br/handle/doc/22668`  
Item: *Avaliação técnica pós-desastre: Petrópolis, RJ* — SGB/CPRM, 2022.

---

## 4. Quais arquivos havia dentro do ZIP?

O ZIP usa estrutura de streaming (sem Central Directory padrão), sendo necessário parser especializado para extração.

**Conteúdo inventariado: 11 PDFs de avaliação técnica de campo por bairro**

| Anexo | Bairro/Localidade | Data de campo |
|---|---|---|
| ANEXO-I | Bairro Mosella | 19/02/2022 |
| ANEXO-II | Bairro Moinho Preto | 19/02/2022 |
| ANEXO-III | Bairro Serra Velha | 20/02/2022 |
| ANEXO-IV | Bairro Valparaíso, Rua Eugenio Barcelos | 22/02/2022 |
| ANEXO-V | Rua Teresa e imediações | 23/02/2022 |
| ANEXO-VI | Bairro Moinho Preto (revisita) | 24/02/2022 |
| ANEXO-VII | Bairro Mosella (revisita) | 24/02/2022 |
| ANEXO-VIII | Estrada Velha e Vila Felipe | 25–26/02/2022 |
| ANEXO-IX | Bairro Sargento Boening | 28/02/2022 |
| ANEXO-X | Servidão Alépio Gomes da Costa | 01/03/2022 |
| ANEXO-XI | Bairro Quitandinha | 02/03/2022 |

---

## 5. Havia vetor no ZIP?

**Não.** Todos os 11 arquivos extraídos são PDFs. Nenhum shapefile, geopackage, KMZ, KML ou GeoJSON foi encontrado no ZIP.

---

## 6. Havia CRS nos ativos?

Não aplicável — os ativos são PDFs. Nenhum CRS vetorial identificado.

---

## 7. Havia campo de data?

Não. Os PDFs são documentos de campo sem campos de data estruturados que permitam auditoria programática.

---

## 8. Havia separação de fenômeno?

Não — para fins de auditoria automática. Os PDFs provavelmente contêm descrições textuais de fenômenos por bairro, mas não campos de atributo consultáveis.

---

## 9. Havia inundação observada?

Não como vetor. Os PDFs cobrem bairros com histórico de inundação e deslizamento (Mosella, Moinho Preto, Quitandinha, Serra Velha), mas o conteúdo não é auditável programaticamente como geometria observada.

---

## 10. Havia movimento de massa observado?

Não como vetor, pelos mesmos motivos acima.

---

## 11. Algum ativo pode virar candidato a ground truth?

**Não diretamente.** Nenhum dos 6 registros passou todos os 11 gates.

| ID | Gate bloqueante | Status |
|---|---|---|
| OBS_PET_001 | geometry_available=FAIL (PDF) | BLOCKED |
| OBS_PET_002 | geometry_available=NOT_APPLICABLE (ZIP/PDF) | PENDING_VECTOR_AUDIT |
| OBS_PET_003 | raw_asset_traceable=FAIL (sem URL pública) | BLOCKED |
| OBS_PET_004 | raw_asset_traceable=FAIL (sem URL pública) | BLOCKED |
| OBS_REC_001 | raw_asset_traceable=FAIL (sem URL pública) | BLOCKED |
| OBS_CUR_001 | event_date_available=FAIL; spatial_unit=FAIL | BLOCKED |

Os PDFs do ZIP são `CARTOGRAPHIC_LEAD_ONLY`: confirmam que o SGB/CPRM tem dados de campo, mas não fornecem geometria vetorial diretamente utilizável.

---

## 12. O que continua bloqueado?

| Bloqueio | Causa | Resolução necessária |
|---|---|---|
| Ground truth operacional PET_2022_02_15 | Nenhum vetor observado com data compatível encontrado | Solicitar SGB/CPRM dados de campo georref; obter PKG_FR_PET_001 DRM-RJ |
| Ground truth REC_2022_05 | PKG_FR_REC_002 não público | Solicitar COMPDEC/Defesa Civil PE |
| Ground truth CUR_* | GeoJSON local é municipal, sem data de evento | Buscar eventos CUR com data e geometria intraurbana |
| ml_label_status | Sem ground reference auditado + sem split/leakage protocol | Sequencial: primeiro ground reference, depois split |

---

## 13. Que instituição precisa ser acionada formalmente?

### Prioridade 1 — SGB/CPRM (máxima prioridade)
**O quê:** Dados georeferenciados de campo que embasaram os 11 PDFs de avaliação pós-desastre de Petrópolis 2022  
**Por quê:** A equipe CPRM visitou 11 bairros entre 19/02/2022 e 02/03/2022 — os dados digitais internos (KMZ, SHP, planilhas GPS) provavelmente existem  
**Como:** Solicitação formal via RIGeo (repositório) ou contato direto com autores dos relatórios (Filipe Modesto, Leandro Kuhlmann, Patrícia Jacques, Rafael Ribeiro, Thiago Santos)

### Prioridade 2 — DRM-RJ/NADE
**O quê:** PKG_FR_PET_001 — geometria cartográfica separada por fenômeno e bairro  
**Como:** Solicitação formal ao DRM-RJ com especificação de shapefile/geopackage

### Prioridade 3 — INPE / International Charter
**O quê:** Produto de sensoriamento remoto de emergência para PET_2022_02_15  
**Por quê:** A Charter foi ativada para o evento e pode ter produtos de mapeamento  
**Como:** Consultar portal International Charter e contato com INPE/OBT

### Prioridade 4 — Defesa Civil Municipal Petrópolis
**O quê:** Laudos com coordenadas de áreas evacuadas e afetadas  
**Como:** LAI ou contato direto com Secretaria Municipal de Defesa Civil

---

## 14. Estado invariante após v1if

```
operational_ground_truth_status     = BLOCKED
ml_label_status                     = BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL
can_create_training_label           = false
can_reopen_protocol_b               = false
multimodal_status                   = HOLD
dino_usage_status                   = SUPPORT_ONLY
```

---

## 15. Próximo passo concreto para v1ig

Criar pacote de solicitação institucional completo e rastreável para:
1. SGB/CPRM: dados georref de campo (KMZ/SHP) que embasaram os PDFs do ZIP
2. DRM-RJ/NADE: PKG_FR_PET_001 (geometria cartográfica pós-desastre Petrópolis 2022)
3. INPE / International Charter: produto de mapeamento de emergência Petrópolis 2022
4. Defesa Civil Municipal Petrópolis: coordenadas de áreas evacuadas e afetadas

O objetivo de v1ig seria executar, registrar e rastrear essas solicitações formais, mantendo o blockeio metodológico enquanto não houver resposta com dado vetorial auditável.

---

*Relatório gerado como parte do Protocolo C de construção de corpus de referência terrestre para eventos de inundação urbana.*  
*Repositório: REV-P. Estágio: v1if. Decisão: BLOCKED. Nenhum claim operacional. Nenhum ground truth estabelecido.*
