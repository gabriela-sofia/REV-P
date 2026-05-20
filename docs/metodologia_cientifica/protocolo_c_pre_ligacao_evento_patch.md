# Protocolo C — Pré-ligação Evento–Patch

## 1. Motivação

A etapa v1hq fechou a primeira camada documental de eventos observados candidatos do Protocolo C: 9 eventos (3 por região) com G1/G2/G3 fechados por fonte primária rastreável e G4 em triagem espacial de bairro ou localidade. Essa camada prova a existência documental do evento, a fonte e a temporalidade — mas não estabelece nenhuma ligação espacial validada com os patches Sentinel do corpus REV-P.

A etapa v1hr prepara essa ligação sem executá-la. Ela organiza, por evento observado candidato, quais patches do corpus estão no escopo regional de triagem, quais localidades precisam ser geocodificadas manualmente, quais janelas temporais Sentinel são candidatas para busca futura e quais dependências precisam ser resolvidas antes de qualquer overlay patch-level real.

Esta etapa é metadata-only. Nenhum overlay é executado. Nenhuma geocodificação automática é realizada. Nenhum dado bruto é baixado. Nenhum patch é validado como inundação.

---

## 2. O que esta etapa faz

- **Organiza escopos de preflight evento–patch**: para cada evento observado candidato, identifica quais patches do corpus DINO estão no escopo regional de triagem, com status explícito de que nenhuma relação patch-level foi validada
- **Cria alvos de geocodificação manual**: lista as localidades citadas nas fontes primárias e secundárias da v1hq que precisam ser geocodificadas por revisão humana ou fonte oficial antes de qualquer overlay
- **Define janelas temporais Sentinel metadata-only**: para cada evento, calcula períodos de pré-evento, evento e pós-evento como alvo de busca futura de assets Sentinel — sem baixar imagens nem verificar disponibilidade real
- **Registra dependências para overlay futuro**: documenta o que precisa estar resolvido antes de qualquer patch-linking real — geometria da fonte, geocodificação, CRS, licença, busca Sentinel, revisão humana, separação de fenômenos (onde aplicável)
- **Mantém todos os bloqueios metodológicos** herdados da v1hq: ground truth operacional não estabelecido, Protocolo B bloqueado, multimodal em hold, DINO support-only

---

## 3. O que esta etapa não faz

Esta etapa **não executa** e **não permite**:

- overlay espacial entre geometria do evento e patches Sentinel
- geocodificação automática de localidades
- criação de coordenadas ou geometrias, mesmo aproximadas
- confirmação de que qualquer patch foi inundado
- confirmação de que qualquer patch é afetado pelo evento
- declaração de ground truth operacional
- criação de flood label
- criação de training label
- reabertura do Protocolo B
- avanço do pipeline multimodal
- uso de DINOv2 como evidência observacional de evento
- uso de NDWI/NDBI/SAR como confirmação de inundação
- promoção de evento ou patch a referência operacional
- download de imagem Sentinel, shapefile, GeoTIFF ou dado pesado

---

## 4. Diferença entre G4 de triagem e patch-linking real

### G4 em triagem (v1hq)

G4_SPATIAL_ALIGNMENT_TRIAGE foi fechado documentalmente na v1hq quando havia evidência espacial de nível de localidade — bairro, rua, comunidade, corredor de rio, área descrita em laudo técnico ou produto Copernicus regional. O fechamento documental de G4 significa: *há evidência textual ou cartográfica de que o evento ocorreu na mesma área geográfica geral dos patches do corpus*.

Esse fechamento **não** significa:
- que a geometria exata do evento é conhecida
- que o evento tem polígono de área afetada georreferenciado
- que a sobreposição entre o evento e os patches foi computada
- que algum patch foi confirmado como afetado

### Patch-linking real (futuro)

Patch-linking real exige, no mínimo:
1. Geometria vetorial da área afetada (shapefile, GeoJSON) com CRS documentado
2. Interseção ou distância calculada entre essa geometria e os bounding boxes dos patches
3. Alinhamento temporal confirmado — asset Sentinel disponível na janela do evento
4. Licença e proveniência verificadas para uso operacional
5. Revisão humana ou especialista da relação evento–patch
6. Nenhum confundidor de processo não separado (ex: deslizamento em Petrópolis)

A v1hr está antes dessa cadeia. Ela prepara as condições sem satisfazê-las.

---

## 5. Alvos de geocodificação manual

Localidades citadas nas fontes primárias e secundárias dos eventos observados candidatos precisam ser geocodificadas antes que qualquer ligação patch-evento possa ser estabelecida. Geocodificação manual significa:

- identificar a localidade por nome exato conforme a fonte
- buscar geometria oficial (shapefile municipal, base GIS, mapa técnico institucional) ou ponto de referência rastreável
- documentar a fonte da geocodificação com URL e data de acesso
- registrar o CRS esperado
- verificar licença de redistribuição
- não usar geocodificação aproximada de serviço online como verdade operacional

**O que não é geocodificação válida neste projeto:**
- coordenada estimada por buscador online sem verificação oficial
- centroide de município como representação de área afetada específica
- inferência de localização a partir de nome de rua sem fonte oficial
- geometria aproximada de mapa genérico

Todos os alvos de geocodificação da v1hr têm `geocoding_status=NOT_GEOCODED` ou `NEEDS_MANUAL_REVIEW` — nenhum está completo nesta etapa.

---

## 6. Janela temporal Sentinel

Para cada evento observado candidato, a v1hr define três janelas temporais candidatas para busca futura de assets Sentinel:

**pre_event_window**: período anterior ao evento — 14 dias antes do início — para baseline de estado pré-chuva (cobertura vegetal, solo, corpos d'água sem evento)

**event_window**: período do evento — do início ao fim documentado — alvo primário de busca de imagem pós-chuva ou durante evento

**post_event_window**: período posterior ao evento — até 14 dias depois do fim — para análise de recuperação ou persistência de inundação

**Restrições importantes:**
- Sentinel-1 (SAR) tem relevância HIGH para todos os eventos de chuva intensa, pois SAR penetra nuvens
- Sentinel-2 (óptico) tem relevância MEDIUM e risco de nuvem HIGH em eventos de chuva — cobertura de nuvem precisa ser verificada na busca real
- As janelas são metadata-only: nenhum asset foi buscado, nenhuma disponibilidade foi verificada, nenhuma imagem foi baixada
- A existência de janela temporal não garante disponibilidade de imagem com baixa cobertura de nuvem

---

## 7. Estados de pré-ligação

**NOT_READY_FOR_PATCH_LINKING**
Evento ainda sem evidência espacial suficiente para iniciar geocodificação ou busca de geometria. G4 pode ter triagem de nível regional mas falta localidade específica.

**READY_FOR_MANUAL_GEOCODING**
Evento tem localidades citadas nas fontes que podem ser geocodificadas manualmente. Ainda não há geometria validada.

**READY_FOR_SOURCE_GEOMETRY_REVIEW**
Evento tem fonte com geometria (laudo técnico, produto Copernicus, mapa oficial) que precisa ser revisada, verificada em licença e georreferenciada antes de uso.

**READY_FOR_FUTURE_OVERLAY**
Evento tem localidades geocodificadas ou geometria com fonte rastreável, janela temporal definida e licença avaliada — mas overlay patch-level ainda não foi executado e aguarda revisão humana.

**BLOCKED_PENDING_GEOMETRY**
Evento bloqueado para pré-linking por ausência de geometria, impossibilidade de geocodificação manual por falta de informação espacial ou conflito de licença.

**BLOCKED_PENDING_LICENSE**
Evento tem geometria disponível mas licença da fonte de geometria não foi verificada para uso operacional.

**METHOD_REFERENCE_ONLY**
Evento referenciado apenas para contexto metodológico — não é alvo de pré-linking.

---

## 8. Relação com Protocolo B e multimodal

O **Protocolo B** permanece **BLOCKED**. A pré-ligação evento–patch não cria as condições para reabertura do Protocolo B. O Protocolo B exige ground reference validada com labels curados, protocolo supervisionado, negativos confiáveis, splits temporais ou espaciais e validação independente — nenhum desses elementos existe na v1hr.

O pipeline **multimodal** permanece em **HOLD**. A preparação de janelas temporais Sentinel não autoriza combinação de modalidades.

Patch-linking futuro — mesmo quando completamente validado — **não equivale automaticamente a label de treino**. A criação de label de treino exigiria etapa posterior específica com: definição de negativos, protocolo de balanceamento, controle de leakage e validação supervisionada independente.

---

## 9. Saída da etapa

A v1hr produz os seguintes artefatos metadata-only:

- `datasets/event_patch_linking_preflight_registry.csv` — preflight de pré-ligação evento–patch: escopo regional, status de overlay, bloqueios e guardrails por linha
- `datasets/manual_geocoding_target_registry.csv` — alvos de geocodificação manual: localidades a geocodificar por evento, tipo, fonte e status
- `datasets/event_sentinel_temporal_window_registry.csv` — janelas temporais Sentinel por evento: períodos pré/evento/pós, relevância de sensor e status de aquisição
- `datasets/patch_linking_dependency_registry.csv` — dependências para patch-linking real: o que precisa ser resolvido antes de overlay, human review e ground reference
- `docs/templates/protocolo_c_ficha_geocodificacao_manual.md` — template de ficha de geocodificação manual
- `docs/templates/protocolo_c_revisao_pre_overlay_evento_patch.md` — template de revisão pré-overlay
- `tests/test_revp_v1hr_event_patch_prelinking_audit.py` — testes de auditoria da camada v1hr

**Nenhum overlay foi executado. Nenhuma geocodificação foi realizada. Nenhuma coordenada foi criada. Nenhum dado bruto foi baixado. Nenhum patch foi validado como inundação.**
