# SUSC-GT08 - Preparação Integrada de AOI e Busca Sentinel-1 para Canários

## 1. Escopo do marco

Este marco transforma os canários selecionados nos marcos anteriores em um **pacote técnico
completo para busca Sentinel-1 metadata-only** (apenas metadados, sem baixar imagem):
resolve AOI (Area of Interest, área de interesse) offline, gera especificações de consulta,
faz replay de catálogos Sentinel-1 locais já versionados, monta o contrato de pareamento
pré/pós-evento e o plano de execução SAR futura. É **offline e review-only**: não usa
internet, não consulta STAC/API remota, não baixa Sentinel nem raster, não roda SAR nem GEE,
não cria footprint, não cria geometria real, não executa QA, não altera o `score_v6`, não
cria `score_v7`, não treina modelo e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01 a GT07

O GT01–GT07 formalizaram a política, aplicaram-na, montaram a fila, resolveram datas,
prepararam geometria, selecionaram os canários e prepararam o QA humano. O GT08 avança a
engenharia até a fronteira da aquisição, **sem cruzá-la**.

## 3. Por que SAR entra como canário e nada é baixado/processado

O SAR é apenas um teste de aderência em **poucos** canários; a generalização do REV-P segue
por features escaláveis por patch. Toda consulta Sentinel-1 aqui é **especificação futura**
(`metadata_only=true`, `no_download=true`, `remote_execution_allowed_now=false`). O footprint
futuro pós-evento será **referência de avaliação**, nunca feature pré-evento.

## 4. Canários e AOI

Entraram **5** canários. Distribuição de AOI:
aoi_forte_por_bbox_ou_geometria_patch=5.

| alvo | patch | cidade/bairro | AOI | pré/pós locais | pareamento | pronto p/ busca |
| --- | --- | --- | --- | --- | --- | --- |
| S1_0547 | S18A_PATCH_0301 | recife/Pina | aoi_forte_por_bbox_ou_geometria_patch | 2/1 | par_completo_pre_pos | true |
| S1_0548 | S18A_PATCH_0302 | recife/Imbiribeira | aoi_forte_por_bbox_ou_geometria_patch | 2/1 | par_completo_pre_pos | true |
| S1_0549 | S18A_PATCH_0303 | recife/Afogados | aoi_forte_por_bbox_ou_geometria_patch | 2/1 | par_completo_pre_pos | true |
| S1_0550 | S18A_PATCH_0304 | recife/Areias | aoi_forte_por_bbox_ou_geometria_patch | 2/1 | par_completo_pre_pos | true |
| S1_0551 | S18A_PATCH_0305 | recife/Areias | aoi_forte_por_bbox_ou_geometria_patch | 2/1 | par_completo_pre_pos | true |

A AOI forte vem do **bbox** (retângulo envolvente) dos patches no registro versionado
`susc_18a_unified_patch_registry.csv`; nenhuma geometria nova foi criada.

## 5. Catálogos Sentinel-1 locais

Catálogos locais versionados encontrados: **1**;
cenas avaliadas: **35**.

| catálogo | arquivo | região | cenas | casa evento do canário |
| --- | --- | --- | --- | --- |
| CAT_001 | asf_sentinel1_sentinel1_grd_metadata_recife_event.json | recife | 7 | true |

## 6. Especificações de consulta Sentinel-1

Uma especificação por canário em `susc_gt08_especificacoes_consulta_sentinel1.csv`, com
política fixa: coleção `sentinel1_grd`, instrumento `SAR`, produto `GRD`, modo `IW`,
polarização `VV,VH`, órbita `any`, janelas pré/pós do GT04, `metadata_only=true`,
`no_download=true` e `remote_execution_allowed_now=false`.

## 7. Cenas candidatas locais (replay)

Do catálogo local, foram classificadas **10** cenas
`pre_event_candidate` e **5** `post_event_candidate` (as demais
como `outside_window`, `wrong_product`, `wrong_mode` ou `insufficient_overlap`). Exemplos:
pré = `S1A_IW_GRDH_1SDV_20220516T075410_20220516T075435_043233_0529C8_8330` (2022-05-16T07:54:10Z); pós = `S1A_IW_GRDH_1SDV_20220528T075411_20220528T075436_043408_052EF7_33C5` (2022-05-28T07:54:11Z). Cada cena é apenas metadado (`metadata_only=true`,
`no_download_in_this_milestone=true`).

## 8. Contrato de pareamento pré/pós

Em `susc_gt08_contrato_pareamento_pre_pos.csv`: exige ≥1 cena pré e ≥1 pós, ambas
intersectando a AOI, produto GRD, modo IW e polarização VV/VH, preferindo a mesma órbita.
Canários com par completo pré/pós: **5**.

## 9. Plano de execução SAR futura

Em `susc_gt08_plano_execucao_sar_futura.csv`, onze passos por canário (confirmar AOI; buscar
metadados; selecionar pré; selecionar pós; **baixar apenas em marco futuro**; pré-processar
SAR; máscara de água permanente; filtro HAND/slope; exclusões urbanas; footprint candidato;
QA humano). Todos com `no_execution_now=true`.

## 10. Bloqueios

Em `susc_gt08_bloqueios_sentinel1.csv`: AOI insuficiente, janela ausente e par pré/pós
incompleto quando aplicável.

## 11. Por que nada foi baixado ou processado

Este marco é **metadata-only**: prepara a busca e o pareamento a partir do que já está
versionado, sem rede e sem download. A aquisição real fica para marcos futuros controlados.

## 12. Confirmação explícita dos bloqueios

**Não** houve internet, STAC/API remota, download de Sentinel/raster
(`download_executado=0`), SAR/GEE executado
(`sar_executado=0`), footprint criado
(`footprint_criado=0`), geometria criada, QA executado,
alteração do `score_v6` (`score_v6_changed=false`) nem
promoção a `positive_strong`
(`positive_strong_promovidos=0`). Buscas/downloads
agora: `can_execute_search_now_true_count=0`,
`can_download_now_true_count=0`. Contagens de
controle: `eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 13. Próximo passo recomendado

**GT09 - Pacote Visual Manual**. Com AOI forte e cenas locais pré/pós já identificadas para
os canários de Recife, o próximo passo natural é consolidar o pacote de revisão visual dos
pares localizados, mantendo tudo review-only e sem download.
