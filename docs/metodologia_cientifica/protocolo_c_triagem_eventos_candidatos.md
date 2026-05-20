# Protocolo C — Triagem de Eventos Candidatos por Região

## Objetivo desta etapa (v1hn)

Esta etapa cria uma camada metadata-only para organizar eventos candidatos de inundação e alagamento por região do REV-P (Recife, Petrópolis, Curitiba), priorizar quais eventos devem ser investigados primeiro e conectar cada evento candidato aos gates do Protocolo C.

Nenhum dado bruto é baixado nesta etapa. Nenhum raster é lido. Nenhum label é criado. Nenhuma referência operacional é declarada.

---

## O que esta etapa faz

- Cria um registro de eventos candidatos reais por região, com identificadores, descrição, prioridade de busca e status atual
- Cria um backlog de fontes a pesquisar por evento candidato, conectando cada evento às fontes institucionais identificadas no Protocolo C (v1hl, v1hm)
- Cria um registro de escopo de triagem por patch: quais patches do corpus de 12 (v1fz) estão no perímetro de busca de cada evento candidato
- Documenta explicitamente os gates do Protocolo C que cada evento candidato pode eventualmente ajudar a fechar — e os gates que permanecem bloqueados independentemente

---

## O que esta etapa NÃO faz

- Não baixa raster, imagem ou dado geoespacial
- Não executa pipeline espacial ou temporal
- Não confirma evento como observado — eventos permanecem como EVENT_SEARCH_TARGET ou PENDING_SOURCE_REVIEW até aquisição e revisão humana
- Não declara ground truth operacional
- Não gera label supervisionado
- Não associa DINO como suporte a gate de evento, temporalidade, espacialidade ou ground truth
- Não avança para Protocolo B
- Não desbloqueia multimodal

---

## Fontes da triagem

A triagem de eventos candidatos parte das fontes documentadas nas etapas anteriores:

- **v1hl**: `observational_evidence_acquisition_plan.csv` e `regional_ground_reference_readiness.csv` — fontes alvo por região e prioridade de aquisição
- **v1hm**: `evidence_acquisition_tracker.csv` e `evidence_source_intake_registry.csv` — tracker operacional de aquisição e intake
- **Protocolo C (etapa de aquisição)**: `flood_event_candidate_registry.csv` — registro anterior com placeholders por região

Esta etapa não substitui esses registros. Ela os complementa com uma camada de triagem que opera sobre eventos candidatos específicos, não sobre fontes abstratas.

---

## Status dos eventos candidatos

Os eventos candidatos do REV-P são organizados em três status:

**EVENT_SEARCH_TARGET**
Evento mencionado em contexto regional ou histórico como período relevante de busca. Não há fonte institucional confirmada no repositório. A pesquisa de fonte ainda não foi iniciada.

**PENDING_SOURCE_REVIEW**
Evento que possui indícios de documentação institucional (menção em relatórios públicos, referências em bases metodológicas, contexto nacional documentado) mas cuja fonte específica ainda não foi adquirida nem revisada. A designação reflete plausibilidade de documentação, não confirmação.

**CONFIRMED_BY_SOURCE**
Evento confirmado por fonte institucional adquirida, revisada por humano e documentada no registro de proveniência. Nenhum evento atingiu este status nesta etapa.

**BLOCKED**
Evento cuja busca está explicitamente bloqueada por bloqueador identificado (acesso restrito, sensibilidade, conflito de licença).

---

## Prioridade de busca

Os eventos candidatos são classificados por prioridade de busca:

**HIGH**
Evento com indícios de documentação institucional, relevância territorial direta para patches do corpus e pelo menos uma fonte-alvo identificada. Busca de fonte deve ser iniciada na próxima etapa operacional.

**MEDIUM**
Evento relevante mas com cobertura temporal ou espacial incerta, ou com fonte-alvo menos específica. Busca pode ser iniciada após conclusão das buscas HIGH.

**LOW**
Evento com período temporal ou cobertura espacial marginal para os patches do corpus. Busca adiada.

**METHOD_REFERENCE_ONLY**
Evento ou período utilizado como referência metodológica em fontes externas (ex: Sen1Floods11, Copernicus GFM). Não é alvo de aquisição direta para o REV-P.

---

## Gates do Protocolo C conectados à triagem

Cada evento candidato é conectado ao conjunto de gates que pode eventualmente ajudar a fechar, conforme o Protocolo C:

| Gate | Nome | O que requer |
|------|------|-------------|
| G1 | EVENT_CONFIRMATION | Confirmação de que o evento ocorreu — fonte institucional documentada |
| G2 | SOURCE_AVAILABILITY | Disponibilidade de fonte com cobertura do evento |
| G3 | TEMPORAL_ALIGNMENT | Alinhamento temporal entre evento e asset Sentinel disponível |
| G4 | SPATIAL_ALIGNMENT | Sobreposição espacial entre evento e patches do corpus |
| G5 | SOURCE_STRENGTH | Força metodológica da fonte (observação direta vs. produto algorítmico) |
| G6 | UNCERTAINTY_AND_LIMITATIONS | Documentação de incerteza da fonte |
| G7 | HUMAN_REVIEW | Revisão humana ou especialista da evidência |
| G8 | INDEPENDENT_CORROBORATION | Corroboração independente por segunda fonte |
| G9 | PROMOTION_DECISION | Decisão formal de promoção a referência operacional |

**O DINOv2 não fecha nenhum desses gates.** Os embeddings DINO são review-only e não constituem evidência observacional de evento.

Nenhum evento candidato desta etapa fecha qualquer gate. A triagem documenta quais gates são relevantes por evento e quais fontes precisariam ser adquiridas para avançar na cadeia.

---

## Regiões e eventos candidatos

### Recife

Área urbana costeira com histórico documentado de eventos de inundação e alagamento associados a chuvas intensas, especialmente no período maio–julho. As fontes-alvo principais são: COMPDEC Recife (Defesa Civil), CPRM (Serviço Geológico), PE3D (plataforma GIS municipal), Sentinel-2 pós-evento, Copernicus GFM.

Eventos candidatos identificados: período maio–julho de 2021 e período equivalente de 2022. Ambos aparecem como EVENT_SEARCH_TARGET ou PENDING_SOURCE_REVIEW com base nas referências institucionais documentadas em v1hl e v1hm.

### Petrópolis

Área serrana com histórico de deslizamentos e inundações em eventos de chuva extrema. O evento de fevereiro de 2022 tem documentação nacional extensiva e é referenciado como contexto metodológico por fontes externas. A separação entre inundação e deslizamento é obrigatória para qualquer uso metodológico. As fontes-alvo principais são: Defesa Civil Municipal, CPRM (laudo pós-evento), Sentinel-2 pós-evento.

Evento candidato identificado: fevereiro de 2022. Classificado como PENDING_SOURCE_REVIEW dado contexto nacional amplamente documentado, embora fonte local específica ainda não tenha sido adquirida no repositório.

### Curitiba

Área urbana com histórico de alagamentos urbanos em eventos de chuva intensa, contexto de contraste metodológico para o REV-P. As fontes-alvo principais são: Defesa Civil Municipal, GeoCuritiba (plataforma GIS municipal), Sentinel-2 pós-evento.

Eventos candidatos identificados: períodos de 2022 e 2023. Ambos aparecem como EVENT_SEARCH_TARGET com base nas referências de v1hl e v1hm.

---

## Relação com multimodal

Multimodal permanece em hold. A triagem de eventos candidatos não altera essa condição. Condições de desbloqueio permanecem as mesmas documentadas em v1ft e v1hl: recuperação do stack Recife, balanceamento regional, aprovação de revisor, ground reference operacional estabelecido.

---

## Registros desta etapa

- `datasets/event_candidate_screening_registry.csv` — eventos candidatos por região com status, prioridade e gates relevantes
- `datasets/event_source_search_backlog.csv` — fontes a pesquisar por evento candidato
- `datasets/event_patch_screening_scope.csv` — escopo de triagem por patch: quais patches estão no perímetro de busca de cada evento

Esses três registros são metadata-only. Nenhum dado bruto é referenciado. Nenhuma aquisição é executada.

## Próximas etapas

A etapa v1ho complementa a triagem com dossiês de evidência por evento candidato: especifica o pacote mínimo de evidências necessário, o estado atual de cada requisito crítico e a decisão de continuidade. Veja [`protocolo_c_dossies_eventos_candidatos.md`](protocolo_c_dossies_eventos_candidatos.md).

A etapa v1hp transforma os dossiês em ação concreta: organiza planos de busca externa por região, pacotes de solicitação formal a instituições, perguntas de busca mapeadas a gates G1–G9 e matriz de prioridade regional. Veja [`protocolo_c_busca_externa_solicitacao_regional.md`](protocolo_c_busca_externa_solicitacao_regional.md).

A etapa v1hq inicia a primeira camada documental de eventos observados candidatos: 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial. Esta etapa fecha evidência de existência do evento e fonte rastreável, mas mantém bloqueados ground truth operacional, Protocolo B, multimodal e labels supervisionados. Os dados externos brutos necessários estão catalogados em `datasets/manual_external_evidence_needed_registry.csv`. Veja [`protocolo_c_referencias_observacionais_candidatas.md`](protocolo_c_referencias_observacionais_candidatas.md).
