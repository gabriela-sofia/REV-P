# SUSC-GT09 - Esteira Canária de Revisão Visual, Pareamento Pré/Pós e Pré-Execução SAR

## 1. Escopo do marco

Este marco composto transforma os 5 canários Recife em uma **esteira completa de
pré-execução observacional**: melhor par Sentinel-1 pré/pós por canário, dossiê
técnico-visual, inventário de ativos locais, contrato de renderização futura, dry-run do
pipeline SAR, handoff para QA humano, readiness gate e guardrails de leakage. É **offline
e review-only**: não usa internet, não consulta STAC/API remota, não baixa Sentinel nem
raster, não abre nem gera imagem, não roda SAR/GEE, não cria footprint, não cria geometria
real, não executa QA, não altera o `score_v6`, não cria `score_v7`, não treina modelo e não
promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01 a GT08

O GT08 encontrou, offline, AOI forte e um catálogo Sentinel-1 local real com pares pré/pós
para os 5 canários. O GT09 consolida essa fronteira em uma esteira auditável, **sem
cruzá-la**. O SAR continua sendo canário review-only, não a base central do dataset.

## 3. Canários e pares pré/pós selecionados

Entraram **5** canários; **5** com par
pré/pós selecionado, sendo **5** de mesma órbita/path/frame.
Pares alternativos: **5**.

| par | patch | cena pré | cena pós | mesma órbita | score |
| --- | --- | --- | --- | --- | --- |
| PP_0547 | S18A_PATCH_0301 | S1A_IW_GRDH_1SDV_20220516T075410 | S1A_IW_GRDH_1SDV_20220528T075411 | true | 85 |
| PP_0548 | S18A_PATCH_0302 | S1A_IW_GRDH_1SDV_20220516T075410 | S1A_IW_GRDH_1SDV_20220528T075411 | true | 85 |
| PP_0549 | S18A_PATCH_0303 | S1A_IW_GRDH_1SDV_20220516T075410 | S1A_IW_GRDH_1SDV_20220528T075411 | true | 85 |
| PP_0550 | S18A_PATCH_0304 | S1A_IW_GRDH_1SDV_20220516T075410 | S1A_IW_GRDH_1SDV_20220528T075411 | true | 85 |
| PP_0551 | S18A_PATCH_0305 | S1A_IW_GRDH_1SDV_20220516T075410 | S1A_IW_GRDH_1SDV_20220528T075411 | true | 85 |

O par é escolhido preferindo mesma órbita/path/frame, polarização VV+VH e menor distância
temporal ao evento; os demais pares ficam como alternativos em
`susc_gt09_pares_alternativos_canarios.csv`.

## 4. Dossiê por canário

`susc_gt09_dossie_canarios.csv` reúne, por canário, patch/AOI, bairro, data, fonte,
fenômeno, cenas pré/pós, deltas temporais (pré→evento, evento→pós), razão do par e
requisitos para revisão visual e QA futuro.

## 5. Inventário de ativos locais

Foram inventariados **42** ativos **já versionados**
(**30** visuais e **12**
geométricos), sem gerar nenhuma imagem nem baixar nada. Incluem prévias de canário
(pré/pós) e camadas GeoJSON de bbox de patch.

## 6. Contrato de renderização futura

`susc_gt09_contrato_renderizacao_futura.csv` define nove painéis por canário (pré SAR, pós
SAR, diferença pré/pós, overlay AOI/patch, overlays de água permanente, HAND/slope e
exclusões urbanas, máscara candidata e painel de QA), todos com `renderizacao_executada=false`,
`raster_baixado=false` e `footprint_criado=false`.

## 7. Dry-run do pipeline SAR futuro

`susc_gt09_dry_run_pipeline_sar_futuro.csv` traz onze etapas lógicas por canário (confirmar
AOI; confirmar par; validar GRD/IW/VV/VH; calibração; speckle; razão pré/pós; máscara de
água; HAND/slope; exclusão urbana; polígono candidato; QA humano), cada uma com input,
output, critério de sucesso e critério fail-closed. **Nenhuma** foi executada
(`executado_agora=false`).

## 8. Handoff para QA humano

`susc_gt09_handoff_qa_canarios.csv` mapeia cada grupo do checklist do GT07: quais itens
ficam **desbloqueáveis após visualização**, quais **dependem de footprint** e quais
**dependem de QA humano**. Nenhum canário é marcado como aceito/rejeitado; `accepted`
futuro só liberaria avaliação review-only.

## 9. Readiness gate observacional

| gate_state | canários | podem baixar (futuro) | podem QA visual (futuro) |
| --- | --- | --- | --- |
| pronto_para_renderizacao_visual_futura | 5 | 5 | 5 |

Um canário só é `pronto_para_renderizacao_visual_futura` com AOI forte, par pré/pós com
scene_id/product_id, janela válida, fonte/fenômeno e guardrails review-only.

## 10. Guardrails de leakage

`susc_gt09_guardrails_leakage_evidencia.csv` registra, por canário, que a **cena pós-evento
é referência de avaliação futura, nunca feature pré-evento**
(`post_scene_used_as_pre_event_feature=false`), e que features pré-evento só vêm da janela
pré.

## 11. Por que nada foi baixado ou processado

Este marco é de **pré-execução**: prepara a esteira a partir do que já está versionado, sem
rede, download, SAR ou geração de imagem. A execução fica para marcos futuros controlados.

## 12. Confirmação explícita

**Não** houve internet, STAC/API remota, download
(`download_executado=0`), SAR/GEE, imagem gerada
(`imagem_gerada=0`), footprint criado
(`footprint_criado=0`), geometria criada, QA executado
(`qa_executado=0`), accepted/rejected atual
(`accepted_atual=0`, `rejected_atual=0`),
alteração do `score_v6` (`score_v6_changed=false`) nem
promoção a `positive_strong`
(`positive_strong_promovidos=0`). Etapas de dry-run
executadas agora: 0. Contagens de controle:
`eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável, e nenhuma evidência vira ground truth supervisionado sem
política, QA e incerteza.

## 13. Próximo passo recomendado

**GT10 - Renderizacao Visual Local Controlada**. Com AOI forte, pares pré/pós de mesma órbita e ativos
visuais locais já disponíveis para os canários de Recife, o próximo passo é a renderização
visual local controlada, mantendo tudo review-only e sem download.
