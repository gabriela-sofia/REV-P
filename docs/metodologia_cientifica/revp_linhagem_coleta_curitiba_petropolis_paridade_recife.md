# Linhagem de coleta — Curitiba e Petrópolis rumo à paridade com Recife (v1)

**Contexto**: após a investigação real desta sessão (17 fontes tentadas em Curitiba, 15+ em
Petrópolis), cada região tem N=1 ponto-evento positivo adjudicado (Curitiba: Juvevê,
17/01/2022, MODIS MCDWD_L3; Petrópolis: Valparaíso, 24/03/2022, Sentinel-2). Isto documenta o
que falta, de fato, pra chegar a um modelo comparável ao de Recife (v12, Firth, N=269,
LOO-AUC=0,6781) — não é uma promessa de prazo, é o mapa real do trabalho.

## 1. Quanto falta, em número (orçamento EPV real)

Regra já usada nesta linhagem (~10 eventos/variável, mesmo padrão do teste A/B do DINO
v1r5/v1r6): com N=1, zero features cabem. Três patamares reais, não estimativa otimista:

| Patamar | N mínimo | O que destrava |
|---|---|---|
| Piso técnico | ~20-30 | 2-3 features mais fortes (rain_decay, TWI — já são as mais fortes em Recife por evidência real) |
| Confortável | ~50-80 | 3-4 features com folga real de bootstrap |
| Paridade com Recife | ~150-270 | Conjunto completo (6-8 features), mesmo rigor de v12 |

Hoje: Curitiba N=1, Petrópolis N=1. Faltam, no mínimo, ~19-29 pontos positivos adjudicados
por região só pro piso técnico — e pontos negativos correspondentes (ver seção 4).

## 2. Protocolo de aquisição de positivos — agora reutilizável, não mais tentativa às cegas

Esta sessão descobriu e validou, com resultado real, um protocolo de 2 vias que funciona:

**Via A — MODIS MCDWD_L3 (quando o tile não está mascarado)**
1. Calcular tile pela grade geográfica real do produto (`h=floor((lon+180)/10)`,
   `v=floor((90-lat)/10)`) — **não** pela fórmula sinusoidal padrão do MODIS (erro cometido e
   corrigido nesta sessão; este produto usa `GCTP_GEO`, grade lat/lon simples).
2. Baixar `.hdf` via LAADS DAAC (token EOSDIS + autorização OAuth uma vez no navegador).
3. Ler camada `Flood_1Day_250m` com `pyhdf`: classe `3`=enchente real (não bate com água de
   referência), `2`=enchente recorrente, `255`=sem classificação (não é "sem enchente").
4. Testar 3+ dias ao redor da data suspeita (nuvem pode esconder o sinal em 1-2 dias e abrir
   depois — foi exatamente o caso de Curitiba, dia 3 revelou o cluster).
5. **Checar antes de tudo se o produto cobre a região**: escanear o bbox do município inteiro
   — se 100% dos pixels ficam `255` mesmo com `ValidCounts` real presente, é máscara
   estrutural de relevo (caso Petrópolis) e esta via está fechada pra aquela região,
   independente da data.

**Via B — Sentinel-2 bandas cruas (quando MODIS está mascarado, ex. terreno íngreme)**
1. **Nunca** usar a camada "NDWI" pré-colorida do Copernicus Browser (é visualização
   renderizada, contamina qualquer comparação quantitativa). Baixar sempre B03, B08, B11 como
   **Raw**, formato TIFF 32-bit float, aba "Analytical".
2. Escolher a data de referência ("antes") usando `cloudCover` real da API OData
   (`catalogue.dataspace.copernicus.eu/odata/v1/Products?...$expand=Attributes`), não
   suposição — um dia "3 dias antes" pode ainda estar dentro da mesma tempestade.
3. Calcular NDWI `(B3-B8)/(B3+B8)` e MNDWI `(B3-B11)/(B3+B11)`.
4. **Critério físico obrigatório**: só aceitar pixel como candidato se a reflectância
   **absoluta** (não só o índice relativo) for baixa em B08 E B11 no dia do evento (ex.
   `<0,15`) e não era assim antes. Índice relativo sozinho pega nuvem (nuvem sobe em todas as
   bandas; água desce em NIR/SWIR) — erro cometido e corrigido nesta sessão.
5. Adjudicar: dentro do município + rio/córrego real mapeado próximo (Overpass:
   `way["waterway"](around:500,lat,lon)`) + corroboração de fonte externa (notícia real sobre
   a área ser propensa a alagar).

## 3. Onde buscar mais datas de evento (além das já esgotadas)

Já esgotadas nesta sessão: Curitiba jan/2022 (Legisladoc completo, S2ID, ANA, GFD, Copernicus
EMS, SIAC156, CEMADEN, Atlas Digital — só MODIS deu candidato). Petrópolis: 07/01/2022
(inviável, nuvem 97-99%), 15-16/02/2022 (Sentinel-2 contaminado, MODIS mascarado
estruturalmente), 22-24/03/2022 (candidato achado).

**Próximas datas reais a checar** (não tentadas ainda nesta linhagem):
- Petrópolis: evento de 1988 e 2011 (mencionados em reportagem real, sem verificação de
  geolocalização ainda) — pré-Sentinel/MODIS moderno, exigiria Landsat histórico.
- Petrópolis 2024-2026: verificar S2ID/Legisladoc equivalente por novos eventos recentes
  (não verificado nesta sessão).
- Curitiba: mesmo protocolo (S2ID + notícia real) pra qualquer temporal de verão 2023-2026,
  já que Curitiba tem alagamento recorrente documentado (Juvevê, mas também outros bairros
  citados nas buscas: Alto da Glória, Cristo Rei, Hugo Lange).

## 4. Lacuna não resolvida ainda: pontos negativos

Toda esta sessão buscou só positivos. Um modelo Firth precisa de controle negativo real
(local confirmado não-enchente), do mesmo jeito que Recife tem. **Isto não foi endereçado
ainda** — é o próximo item real de metodologia, não só quantidade. Precisa definir o mesmo
critério usado em Recife (ver `recife_modelo_v9_bairro_matched_new_negatives`) e replicar:
amostragem de pontos dentro do município, fora de qualquer registro de evento, idealmente
pareados por bairro/uso do solo com os positivos.

## 5. Lacuna de infraestrutura: HAND/TWI nunca foi salvo como script reutilizável

Achado real (auditoria desta rodada): o cálculo real de HAND/TWI D-infinity usado no v12 de
Recife (WhiteboxTools: `fill_depressions_wang_and_liu`, `d_inf_flow_accumulation`, `slope`,
`wetness_index`, `elevation_above_stream`) só existe documentado em prosa
(`improvement2_hand_twi_dinf_report.md`) — o script em si nunca foi salvo, só um
amostrador de pontos que lê rasters já prontos. Existe uma implementação alternativa (D8,
numpy puro, sem WhiteboxTools) já feita especificamente pra Petrópolis em
`PROJETO/local_runs/treino_exploratorio_diagnostico_v3/step1_lithology_hand_petropolis/`,
mas é D8, não D-infinity — não é o mesmo método do v12, mudaria a comparabilidade.

**DTMs reais já confirmados localmente, prontos pra uso**:
- Curitiba: `PROJETO/data/raw/curitiba/sgb_cprm/produtos_mde_curitiba_pr.zip` (722MB, Esri
  ArcInfo Binary Grid, pasta `01.MDS_Hipsometria/mdt/`)
- Petrópolis: `PROJETO/data/raw/petropolis/sgb_cprm/produtos_mde_petropolis_rj.zip` (230MB,
  mesmo formato, pasta `MDE/pt_sirgas_utm/`)

**Próximo passo de infraestrutura**: reescrever o script D-infinity (WhiteboxTools) como
unidade testável e genérica (aceita caminho de DTM como parâmetro), validar rodando sobre o
DTM de Recife e comparando contra `hand_dinf.tif`/`twi_dinf.tif` já existentes (teste de
regressão real, não confiança cega), só depois aplicar aos DTMs de Curitiba/Petrópolis.

## 6. Ordem real de próximos passos

1. Reescrever/testar script D-infinity genérico, validado contra Recife (infraestrutura,
   não depende de N crescer).
2. Definir critério de ponto negativo (metodologia, replicando Recife).
3. Continuar aquisição de positivos com o protocolo A/B desta seção, região por região,
   documentando N real a cada rodada — sem pular pra treino antes do piso EPV.
4. Só quando N ≥ piso técnico (~20-30) por região: rodar Firth reduzido (2-3 features),
   nunca antes.

## Regras que continuam valendo, sem exceção

Uma tarefa por vez. Nunca fabricar ponto. Sucesso = tentativa real + resultado real. Nenhuma
região passa a `region_maturity="available"` sem `model_version` real (bloqueado no schema
Pydantic).
