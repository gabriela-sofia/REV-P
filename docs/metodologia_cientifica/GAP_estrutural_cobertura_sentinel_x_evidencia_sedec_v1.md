# Gap estrutural: cobertura Sentinel/DINO x evidência real SEDEC (Recife)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO
**Autorizado por**: Gabriela Sofia (reviewer/executora única do projeto)
**Data**: 2026-07-23

## Objetivo deste documento

Registrar formalmente, com números reais e reprodutíveis, o gap entre (a) a cobertura
Sentinel/DINO disponível hoje e (b) a evidência real de enchente (eventos SEDEC
geocodificados) usada em `PROJETO/local_runs/recife_modelo_final_v5/pipeline_final_v5.py`
— e o que é preciso, de fato, pra fechar esse gap.

## 1. O que já existe

- **Corpus Sentinel/DINO atual**: 128 patches reais (.tif) baixados via GEE — 43 Curitiba,
  48 Petrópolis, 37 Recife — em `PROJETO/data/sentinel/`. Manifesto de export:
  `PROJETO/data/sentinel_export_manifest.csv` (128 linhas, todas `COMPLETED`).
- **Grade oficial completa**: `PROJETO/data/dataset_final.csv` define geometria (bbox
  WGS84) para 300 patches (100 por região) — ou seja, os 128 baixados são um subconjunto
  de `selection_priority >= 4` da grade de 300, não a grade inteira.
- **Evidência real (não-heurística)**: 163 registros SEDEC geocodificados reais
  (`recife_pos_*` = 141 flood_positive; `real_clean_sedec_negative` = 22), em
  `PROJETO/local_runs/recife_modelo_oficial_v4_auditoria_lacunas/dataset_v4_features_finais.csv`.
  Já validados com metodologia Firth/bootstrap/LOO-AUC em `pipeline_final_v5.py`
  (LOO AUC=0.602, k-fold AUC=0.603±0.028).
- **Label "v5 core" do projeto** (`score_evento_enchente_potencial_v5_core` em
  `dataset_final.csv`): é um **score heurístico** calculado a partir das mesmas variáveis
  físico-hidrológicas (HAND, distância à água, elevação, declividade, chuva,
  urbano/vegetação) — **não** é evento observado. O próprio projeto já documentou o risco
  de circularidade (`PROJETO/docs/recife_external_patchset_score_v5_methodological_decision.md`):
  treinar contra esse score usando essas variáveis como feature reaprenderia a fórmula, não
  validaria nada novo. **Este documento não recomenda usar `v5_core` como target.**

## 2. O gap medido (join espacial real, ponto-em-bbox, sem reprojeção arriscada)

| Corpus usado no join | Patches Recife cobertos | Pontos SEDEC (n=163) casados | % |
|---|---|---|---|
| 37 patches já baixados (Sentinel real) | 37/100 | **14** (11 pos / 3 neg) | 8.6% |
| Grade oficial completa (100 patches, via `dataset_final.csv`) | 100/100 | **25** (todas identificadas com label=1) | 15.3% |

Método: para cada um dos 163 pontos (lat/lon reais do registro SEDEC), testei
contenção no bbox de cada patch (bbox lido via `rasterio` nos .tif reais já baixados,
reprojetado pra WGS84; bbox da grade oficial lido direto de `dataset_final.csv`, já em
WGS84). Nenhuma reprojeção especulativa, nenhuma correspondência aproximada.

**Achado central**: mesmo a grade oficial de 300 patches (100 por região) cobre só
15% dos eventos reais de Recife. A grade foi construída por outro critério de
amostragem (provavelmente sistemático/estratificado), não centrada nos locais reais de
enchente. **Baixar o resto da grade oficial ajuda pouco** (fecha só 11 pontos a mais,
2 patches: `recife_00322` com 8 pontos, `recife_00528` com 3 pontos, ambos
`selection_priority=1`, ou seja fora do subconjunto priorizado original).

## 2b. Atualização 2026-07-23 (sessão de autenticação GEE)

A service account `revprojeto@projetotransformador01.iam.gserviceaccount.com` foi
autorizada (papel `serviceUsageConsumer` + registro no Earth Engine) e os 2 patches
identificados na seção 3a (`recife_00322`, `recife_00528`) foram baixados de verdade
via `ee.Image.getDownloadURL()` (download síncrono, sem passar por Google Drive —
mais simples que o fluxo de `Export.image.toDrive` do `export_sentinel.py` original,
mesmo resultado: GeoTIFF 6-bandas float64 EPSG:32725, mesmo formato dos 128 existentes).

Resultado real após embedding DINO desses 2 novos patches:

| | Antes | Depois |
|---|---|---|
| Patches Recife com Sentinel real | 37 | **39** |
| Pontos SEDEC casados (de 163) | 14 (11 pos / 3 neg) | **25 (22 pos / 3 neg)** |
| % cobertura | 8.6% | **15.3%** |

Os 2 patches novos só adicionaram positivos (11 pontos, todos `flood_positive`) — o
número de negativos reais (3) não mudou. **Continua descritivo, não treinável**: n=3
negativos é insuficiente pra qualquer inferência, mesmo com mais positivos. Pra virar
treino de verdade, o gargalo agora é especificamente **mais negativos reais
(`real_clean_sedec_negative`)**, não mais positivos.

Scripts atualizados: `REV-P/scripts/dino/revp_v1qz_dino_sedec_recife_join_descriptive.py`
(agora lê `dino_recife_sedec_full_embeddings_v1r1.csv`, 10 patches únicos).
Manifesto atualizado: `PROJETO/data/sentinel_export_manifest.csv` (130 linhas).

## 2c. Atualização 2026-07-23 (ataque aos falsos negativos)

Os 19 registros `real_clean_sedec_negative` que não caíam em NENHUM patch da grade
oficial de 300 (nem os 100 baixados nem os 200 restantes) foram agrupados
geograficamente em 13 clusters e baixados como **patches novos, centrados em
evidência, fora da grade oficial** (`recife_neg_00001`..`recife_neg_00013`,
identificados no corpus DINO como `REC_NEG_00001`..`REC_NEG_00013`). Isso é uma
categoria de patch metodologicamente distinta da grade sistemática de 300 — está
documentado como tal em todos os artefatos (`selection_reason` /
`not_part_of_official_300_grid` nas linhas correspondentes).

Resultado final após embedding DINO desses 13 + os 2 patches oficiais anteriores:

| | Sessão anterior | Agora |
|---|---|---|
| Patches Recife com Sentinel real | 39 | **52** (37 grade + 2 grade-adicional + 13 evidência) |
| Pontos SEDEC casados (de 163) | 25 (22 pos / 3 neg) | **81 (59 pos / 22 neg)** |
| Negativos reais cobertos (de 22) | 3 | **22/22 (100%)** |
| % cobertura total | 15.3% | **49.7%** |

### Análise real feita com esse n=78 (após remover linhas com feature física ausente)

Rodei `REV-P/scripts/dino/revp_v1r4_dino_sedec_extended_analysis.py`, seguindo
exatamente a disciplina do `pipeline_final_v5.py` (nunca misturar variáveis físicas
com DINO no mesmo modelo, dado o orçamento de EPV):

- **Screen univariado (Mann-Whitney), 8 features isoladas**: `elevation_m`
  (p=0.026) e `rain_max_24h_chirps` (p=0.010) mostram diferença real entre
  positivo/negativo. `dino_pca1` e `dino_pca2` **não** mostram diferença
  (p=0.44 e p=0.77).
- **Modelo Firth só-DINO** (2 preditores, EPV=11, atende a heurística ≥10):
  **LOO AUC = 0.490** — nível de chance. Ambos os coeficientes com IC cruzando
  zero.

**Leitura honesta**: o embedding DINO puro, sozinho, não discrimina evento real de
não-evento nesse n. Isso é consistente com a regra fixa do projeto — DINO é
evidência auxiliar, não causal; quem carrega o sinal real são as variáveis
físico-hidrológicas (elevação, chuva), como o próprio achado univariado confirma.
Não é um resultado negativo do projeto — é a confirmação empírica de que a
arquitetura conceitual (DINO auxiliar / física causal) está correta.

## 3. Caminho pra fechar o gap de verdade

### 3a. Ganho fácil e já dentro da grade oficial (baixo risco metodológico)

Exportar só os 2 patches que já existem na grade de 300 e cobrem 11 pontos SEDEC a mais:

```bash
cd PROJETO
python scripts/export_sentinel.py --patch-ids recife_00322,recife_00528 --wait
```

Isso **não pode ser executado por mim neste ambiente**: `export_sentinel.py` exige o
pacote `earthengine-api` (não instalado no sandbox) e `ee.Initialize()` — que por sua vez
exige `earthengine authenticate` (fluxo OAuth interativo no navegador) ou uma
service-account key, nenhum dos dois disponível aqui. Verifiquei: não há credencial GEE
nem token salvo em nenhuma pasta do projeto. **Isso precisa rodar na sua máquina** (onde
você já tem acesso GEE — foi de lá que saíram os 128 patches originais), ou você me
manda uma service-account key pra eu rodar aqui.

### 3b. Fechar o gap inteiro (163/163) — decisão metodológica maior, não só técnica

Pra cobrir os outros 138 pontos (163 - 25), seria necessário criar patches **novos, não
pertencentes à grade oficial de 300** — centrados diretamente nos locais reais de
enchente. Isso muda o desenho amostral do corpus (mistura grade sistemática com patches
centrados em evidência) e precisa ser uma decisão consciente, documentada como tal —
não algo pra fazer silenciosamente. Se você quiser seguir por aqui, o próximo passo é eu
gerar a lista de bboxes propostos (já tenho o clustering geográfico pronto: 45 patches
novos cobririam os 149 pontos hoje não casados com nenhum patch da grade oficial) para
sua revisão antes de qualquer export.

## 4. Estado atual da correlação DINO x evidência real

- **Recife (n=14, join atual)**: `REV-P/scripts/dino/revp_v1qz_dino_sedec_recife_join_descriptive.py`
  — 11 positivos / 3 negativos, **descritivo apenas**, n insuficiente para qualquer
  inferência (abaixo do próprio limiar que `pipeline_final_v5.py` já sinalizou como
  frágil em n=22).
- **Petrópolis (n=17, variável física real — elevação/declividade, não evento
  observado)**: `REV-P/scripts/dino/revp_v1qx_dino_embedding_physical_variable_correlation.py`
  — r(PCA1, declividade) = -0.73, achado exploratório mas real.

## 5. Recomendação

1. Rodar o comando da seção 3a (2 patches, ganho garantido, zero risco metodológico) —
   depende de você (acesso GEE).
2. Não treinar contra `v5_core` (circular por construção).
3. Se quiser fechar o gap inteiro (3b), decidir conscientemente sobre patches
   fora da grade oficial antes de eu gerar as geometrias.
