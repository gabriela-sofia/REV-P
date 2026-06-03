# Protocolo C — Consolidação de Referências Observacionais Candidatas (v1ib)

**Versão:** v1ib  
**Status:** ACTIVE  
**Etapa anterior:** v1ia — Inventário de Fontes Espaciais Autoritativas  
**Data de definição:** 2026-05-21  

---

## 1. Por que o Protocolo C não é mais só bloqueio

As etapas anteriores do Protocolo C foram construindo uma infraestrutura de controle: bloqueios espaciais, registros de prontidão, preflight de geocodificação, lacunas autoritativas. Essa infraestrutura é metodologicamente necessária — ela impede que afirmações indevidas apareçam nos artefatos públicos.

Mas infraestrutura de controle não é suficiente. O objetivo do Protocolo C é construir uma camada de referência observacional candidata para eventos de inundação urbana. Para isso, é preciso responder uma pergunta diferente:

**O que já sabemos positivamente sobre cada evento candidato?**

A v1ib responde essa pergunta. Ela lê todos os registros produzidos nas etapas anteriores e sintetiza o estado de evidência positiva de cada evento — o que foi confirmado, por quais fontes, com que força, e até qual nível de promoção o evento pode avançar no estado atual.

---

## 2. Como o Protocolo C promove evidência em níveis

A promoção não é binária. Um evento não passa de "nada" para "ground reference" — ele acumula evidência em camadas:

| Nível | Significado |
|---|---|
| `LEVEL_0_CITED_ONLY` | Evento mencionado em contexto (notícia, cadastro), mas sem fonte acessível confirmando |
| `LEVEL_1_EVENT_CONFIRMED` | Pelo menos uma fonte confirma que o evento ocorreu na data aproximada |
| `LEVEL_2_TEMPORAL_CONFIRMED` | Período específico do evento confirmado por fonte |
| `LEVEL_3_PHENOMENON_CONFIRMED` | Tipo de fenômeno (inundação, deslizamento, misto) confirmado por fonte |
| `LEVEL_4_LOCALITY_CONFIRMED` | Localidades específicas afetadas identificadas por fonte ou triangulação documentada |
| `LEVEL_5_SPATIAL_TRIAGE_READY` | Fonte espacial autoritativa de triagem disponível (limite de bairro ou geometria contextual) |
| `LEVEL_6_READY_FOR_CONTROLLED_GEOCODING` | Método de geocodificação definido, fonte identificada, alvo registrado, pronto para execução controlada |
| `LEVEL_7_GROUND_REFERENCE_CANDIDATE_FUTURE` | Após geocodificação controlada + overlay + revisão supervisora: candidato a ground reference futuro |
| `BLOCKED_CONTEXTUAL_ONLY` | Evidência somente contextual; nenhuma fonte específica confirma evento, data, fenômeno ou localidade |

A promoção de nível nunca implica em geocodificação automática, overlay ou criação de label. Cada transição de nível requer evidência documentada, e as transições acima de LEVEL_6 requerem decisão supervisora explícita.

---

## 3. Hierarquia de evidência: evento candidato, referência observacional, ground reference — ground truth operacional não estabelecido nesta etapa

**Evento observado candidato**: Evento identificado em fontes externas como potencialmente relevante para o corpus REV-P. Pode ser apenas mencionado em notícia. Nível mínimo: LEVEL_0.

**Referência observacional candidata**: Evento com evidência suficiente para ser tratado como candidato a referência — confirmado por fonte, com data, fenômeno e localidade identificados. Nível mínimo: LEVEL_3 ou LEVEL_4. Esta designação *não* implica em geocodificação, overlay ou promoção a ground reference. É uma camada de qualificação que indica que o evento tem substância documental.

**Referência observacional candidata forte**: Evento com múltiplas fontes confirmadas, data precisa, fenômeno identificado, localidades documentadas e fonte espacial de triagem disponível. Nível mínimo: LEVEL_5. Pode avançar para geocodificação controlada após resolução de bloqueios técnicos (ex.: separação de fenômeno).

**Referência observacional candidata secundária**: Evento confirmado por pelo menos uma fonte, com triagem espacial disponível, mas com lacuna de fenômeno ou evidência espacial parcial. Nível: LEVEL_4 a LEVEL_6 com ressalvas.

**Ground reference**: Posição georreferenciada validada de evento de inundação observado, após geocodificação controlada + overlay + confirmação por revisão supervisora. Não existe ainda neste corpus. Requer etapas futuras separadas.

**Ground truth operacional**: Conjunto de geometrias validadas para treino supervisionado ou avaliação de modelo. **Não existe e não pode ser criado nesta etapa.** `can_be_called_ground_truth_operational=false` para todos os eventos. Invariante permanente no estado atual.

---

## 4. Por que alguns eventos avançam

Um evento avança de nível quando a evidência acumulada satisfaz o critério do nível seguinte. O critério é sempre baseado em fontes verificadas e acessadas — não em inferência, nome de localidade ou contexto genérico.

**PET_2022_02_15** avança para LEVEL_5 porque:
- Três fontes foram confirmadas localmente (DRM-RJ/NADE/Thalweg, NHESS/Copernicus Image of the Day);
- Data específica (15/02/2022) é confirmada por múltiplas fontes;
- Fenômeno misto (inundação + deslizamento) é documentado explicitamente;
- Seis localidades afetadas (Alto da Serra, Centro, Chácara Flora, São Sebastião, Floresta, Quitandinha) são identificadas nas fontes confirmadas;
- Fonte espacial autoritativa de triagem (limites IBGE/Petrópolis) existe e é identificável.

**PET_2024_03_21_28** avança para LEVEL_6 porque:
- Uma fonte confirmada (Prefeitura de Petrópolis, HTML local) documenta o evento;
- Dois alvos de geocodificação (Valparaíso, Floresta) estão classificados como READY_FOR_TRIAGE_ONLY;
- Limites IBGE estão identificados como fonte de triagem disponível.

---

## 5. Por que outros continuam bloqueados

**PET_2022_02_15 não avança para LEVEL_6** porque todos os seis alvos de geocodificação estão em WAITING_PHENOMENON_SEPARATION: o documento técnico DRM-RJ que separa inundação de deslizamento por localidade é obrigatório antes de qualquer geocodificação controlada, e esse documento está em solicitação formal pendente (PKG_FR_PET_001).

**REC_2022_05_24_30 não passa de LEVEL_4** porque nenhuma fonte foi confirmada com força: o boletim de ocorrência COMPDEC (fonte primária obrigatória) requer solicitação formal (PKG_FR_REC_002); o decreto de emergência existe como metadata mas não foi lido nem sua localidade foi confirmada programaticamente.

**CTB_2023_10_28_30 fica em LEVEL_1** porque nenhuma fonte foi acessada e confirmada: os portais da Prefeitura de Curitiba estavam inacessíveis, os boletins da Defesa Civil requerem solicitação formal, e os produtos de portal homepage adquiridos não confirmaram o evento.

**Eventos sem fontes confirmadas** (REC_2023, PET_2022_03, etc.) ficam em LEVEL_0 ou BLOCKED_CONTEXTUAL_ONLY porque não há base documental suficiente para qualificá-los como candidatos.

---

## 6. Por que isso ainda não cria label

A criação de label supervisionado requer:
1. Geocodificação controlada validada com incerteza documentada;
2. Overlay geoespacial entre geometria do evento e patch do corpus;
3. Confirmação por revisão supervisora da correspondência evento–patch;
4. Validação da qualidade da geometria de referência;
5. Decisão explícita de que a geometria é suficientemente precisa para uso como label.

Nenhuma dessas etapas foi executada. Nenhum evento, independentemente do nível atingido, gera automaticamente um label. `can_create_training_label=false` é invariante permanente para todos os eventos nesta etapa.

---

## 7. Por que isso ainda não reabre o Protocolo B

O Protocolo B está bloqueado porque não existe geometria de referência observacional válida para cruzamento com os patches. A reabertura do Protocolo B requer:
1. Ground reference estabelecido (geocodificado, overlay executado, revisão supervisora realizada);
2. Correspondência patch–evento validada com grau de sobreposição documentado;
3. Decisão metodológica formal sobre se a qualidade da referência é suficiente para uso no pipeline.

A promoção de eventos para referência observacional candidata é uma pré-condição necessária, mas insuficiente. `can_reopen_protocol_b=false` para todos os eventos nesta etapa.

---

## 8. Relação com próximas etapas

Os eventos promovidos a LEVEL_5 ou LEVEL_6 são candidatos para o próximo protocolo de execução de geocodificação controlada. Este protocolo ainda não existe e deve ser criado separadamente.

Para PET_2022_02_15, o caminho bloqueante imediato é o PKG_FR_PET_001 (DRM-RJ). Enquanto não houver separação documentada de fenômeno por localidade, nenhuma geocodificação é metodologicamente válida para os alvos deste evento.

Para PET_2024_03_21_28, os alvos Valparaíso e Floresta podem avançar para geocodificação de triagem após definição do protocolo de execução.

Mesmo após geocodificação: overlay e promoção a ground reference permanecem etapas separadas, com decisão supervisora explícita, e `can_create_training_label` permanece `false`.

---

## 9. Estado após v1ic

A v1ic executou a separação fenomenológica máxima alcançável com fontes textuais para PET_2022_02_15. Resultado: PARTIAL_SEPARATION — nenhuma localidade atingiu HYDROLOGICAL_CONFIRMED. O bloqueio de geocodificação da v1ib permanece ativo. A lacuna crítica (PKG_FR_PET_001) permanece aberta. Os guardrails desta metodologia (`can_be_called_ground_truth_operational=false`, `can_create_training_label=false`, `can_reopen_protocol_b=false`, `multimodal_status=HOLD`) são invariantes e não foram alterados pela v1ic.

---

## 10. Estado após v1id e v1ie

A v1id registrou PKG_FR_PET_001 como REQUIRED_NOT_INGESTED e preparou a ingestão formal.

A v1ie executou busca local real e auditou 10 candidatos SGB/CPRM e FBDS com geometria confirmada (SIRGAS 2000). Nenhum passou o Gate 6 (event_date_compatible): mapas de suscetibilidade (`Inundacao_A.shp`, `Movimento_de_Massa_A.shp`) são produtos modelados, não ocorrências do evento de 15/02/2022; feições de deslizamento históricas (`camada original de feições poligonais de deslizamento fotointerpretadas`) não têm campo de data. PKG_FR_PET_001 permanece não encontrado.

Estado invariante após v1ie:
- `operational_ground_truth_status = BLOCKED`
- `can_create_training_label = false`
- `can_be_called_ground_truth_operational = false`
- `can_reopen_protocol_b = false`
- `multimodal_status = HOLD`
- `dino_usage_status = SUPPORT_ONLY`

---

## 11. Estado após v1if

A v1if buscou e baixou fontes vetoriais de repositórios oficiais. O ZIP SGB/CPRM (`anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip`, 20.9MB) foi baixado com sucesso. Conteúdo: 11 PDFs de avaliação técnica de campo por bairro (Mosella, Moinho Preto, Serra Velha, Valparaíso, Quitandinha e outros), sem vetores. O GeoJSON de Curitiba encontrado localmente é 1 polígono municipal sem campo de data — BLOCKED pelos gates 6 e 11.

Nenhum ativo passou os 11 gates. Ground truth operacional permanece BLOCKED.

**Achado positivo da v1if**: os PDFs do ZIP confirmam que a equipe SGB/CPRM realizou avaliações de campo em pelo menos 11 bairros de Petrópolis entre 19/02/2022 e 02/03/2022. Os dados georref internos que embasaram esses relatórios (KMZ, SHP, planilhas GPS) são o próximo alvo prioritário de solicitação institucional.

Estado invariante após v1if:
- `operational_ground_truth_status = BLOCKED`
- `can_create_training_label = false`
- `can_be_called_ground_truth_operational = false`
- `can_reopen_protocol_b = false`
- `multimodal_status = HOLD`
- `dino_usage_status = SUPPORT_ONLY`

---

*Documento gerado como parte do Protocolo C de construção de corpus de referência terrestre para eventos de inundação urbana.*  
*Repositório: REV-P. Estágio: v1ib (atualizado pós-v1if). Status: pré-geocodificação. Nenhum claim operacional.*
