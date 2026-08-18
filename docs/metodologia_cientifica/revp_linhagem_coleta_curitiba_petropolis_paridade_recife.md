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

Hoje: **Curitiba N=1, Petrópolis N=0**. Petrópolis voltou a zero em 2026-07-28: o candidato de
Valparaíso foi rebaixado a candidato fraco na reavaliação topográfica — ver
`revp_reavaliacao_candidato_petropolis_valparaiso_v1.md`. Faltam, no mínimo, ~19 pontos
positivos adjudicados em Curitiba e ~20 em Petrópolis só pro piso técnico — e pontos negativos
correspondentes (ver seção 4).

**Correção do piso, pela revisão de literatura** (`revp_revisao_literatura_alinhamento_metodos_v1.md`,
seção 5): a regra de EPV~10 tem base empírica fraca; **EPV ≥ 20** é o número que a evidência
recente sustenta. O piso técnico de ~20-30 já documentado nesta tabela não é conservadorismo —
é o mínimo recomendado.

## 2. Protocolo de aquisição de positivos — agora reutilizável, não mais tentativa às cegas

Esta sessão descobriu e validou, com resultado real, um protocolo de 2 vias que funciona:

**Via A — MODIS MCDWD_L3 (quando o tile não está mascarado)**
1. Calcular tile pela grade geográfica real do produto (`h=floor((lon+180)/10)`,
   `v=floor((90-lat)/10)`) — **não** pela fórmula sinusoidal padrão do MODIS (erro cometido e
   corrigido nesta sessão; este produto usa `GCTP_GEO`, grade lat/lon simples).
2. Baixar `.hdf` via LAADS DAAC. **Bloqueio confirmado em 2026-07-28 (SUSC-20H)**: a listagem
   (`.../MCDWD_L3/2022/<doy>.csv`) é pública e funciona sem credencial, mas o download do `.hdf`
   sem token cai no OAuth da Earthdata (HTTP 200 com 10.783 bytes de HTML, não HDF). Não há token
   EOSDIS neste ambiente, então nem foi possível testar se o aceite de licença já registrado
   destrava o download programático. Enquanto isso, esta via só roda com arquivo baixado à mão.
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
   renderizada, contamina qualquer comparação quantitativa). Baixar sempre **B03, B08, B11 e
   B12** como **Raw**, formato TIFF 32-bit float, aba "Analytical". (B12 entrou em SUSC-20H:
   AWEI_nsh precisa de SWIR2, e sem ele o consenso do passo 3 degenera no par NDWI+MNDWI que a
   literatura considera insuficiente sozinho.)
2. Escolher a data de referência ("antes") usando `cloudCover` real da API OData
   (`catalogue.dataspace.copernicus.eu/odata/v1/Products?...$expand=Attributes`), não
   suposição — um dia "3 dias antes" pode ainda estar dentro da mesma tempestade.
3. Calcular NDWI `(B3-B8)/(B3+B8)`, MNDWI `(B3-B11)/(B3+B11)` e AWEI_nsh
   `4*(B3-B11) - (0,25*B8 + 2,75*B12)` (Feyisa et al. 2014), e exigir **concordância de 2 dos
   3** cruzando o limiar de mudança pré→pós. Implementado e testado em
   `outputs_public/data/susc_20h_sentinel2_water_candidates/scripts/detect_water_candidates.py`.
4. **Critério físico obrigatório**: só aceitar pixel como candidato se a reflectância
   **absoluta** (não só o índice relativo) for baixa em B08 E B11 no dia do evento (ex.
   `<0,15`) e não era assim antes. Índice relativo sozinho pega nuvem (nuvem sobe em todas as
   bandas; água desce em NIR/SWIR) — erro cometido e corrigido nesta sessão.
5. Adjudicar: dentro do município + rio/córrego real mapeado próximo (Overpass:
   `way["waterway"](around:500,lat,lon)`) + corroboração de fonte externa (notícia real sobre
   a área ser propensa a alagar). **Só depois de passar pelo filtro topográfico da seção 2.1.**

### 2.1 Filtro topográfico obrigatório — antes da adjudicação SUSC-20A

Passo novo, criado em SUSC-20H depois do caso Valparaíso. **Todo candidato bruto, venha da Via A
ou da Via B, passa primeiro por leitura de HAND, TWI e declividade** nos rasters já existentes
de `outputs_public/data/susc_20g_hand_twi_dinfinity_generico/` (script
`scripts/read_hand_twi_slope_at_point.py`, rasters em `local_runs/susc_20g_hand_twi_dinfinity_generico/<região>/`).
Só é adjudicado formalmente o candidato que **não** diverge nos três critérios ao mesmo tempo:

| Critério | O que se espera de ponto de enchente | Divergência |
|---|---|---|
| HAND | baixo — perto do nível de drenagem | alto em termos absolutos **e** sem ser explicado por resolução |
| TWI | acima da mediana regional — convergência de fluxo | abaixo da mediana regional |
| Declividade | suave — fundo de vale/piemonte | encosta, próxima da mediana de terreno de serra |

Divergência **simultânea nos três** → o candidato não é adjudicado; vira candidato fraco, com o
motivo registrado por escrito. Divergência em um ou dois critérios não reprova sozinha: vira
ressalva anotada na adjudicação, porque relevo de serra e pixel de 30 m degradam a leitura.

Duas verificações auxiliares que o caso Valparaíso mostrou serem necessárias, e que custam pouco:

- **Altura sobre o mínimo local** (raios de 150/300/500 m), que não depende do limiar de extração
  de drenagem — é ela que separa "HAND alto de verdade" de "artefato de resolução".
- **Distância e desnível reais até o curso d'água mapeado** (Overpass + o próprio MDT), em vez de
  aceitar a distância estimada no laudo.

**Precedente real**: o candidato Petrópolis/Valparaíso (Sentinel-2, 24/03/2022) foi adjudicado
em v14 pelo critério SUSC-20A sem nenhuma checagem topográfica e, quando a leitura foi aplicada
retroativamente, deu HAND 50,88 m, declividade 23,34° (mediana regional 23,57°) e TWI 5,11
(mediana 5,58) — os três divergindo juntos. As duas verificações auxiliares confirmaram: +30 a
+50 m sobre o mínimo local em dois DEMs independentes, e o Rio Quitandinha a 688 m (não os ~500 m
do laudo) e 30-47 m abaixo do ponto. Resultado: rebaixado a candidato fraco, Petrópolis de volta
a N=0. Ver `revp_reavaliacao_candidato_petropolis_valparaiso_v1.md`. Este passo existe para que o
custo dessa descoberta seja pago **antes** da adjudicação, não depois.

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

## 5. Infraestrutura HAND/TWI — resolvida (SUSC-20G)

Status anterior (resolvido nesta rodada): o cálculo real de HAND/TWI D-infinity usado no v12
de Recife só existia documentado em prosa (`improvement2_hand_twi_dinf_report.md`) — o script
em si nunca tinha sido salvo, só um amostrador de pontos que lê rasters já prontos.

**Reconstruído e validado**: `outputs_public/data/susc_20g_hand_twi_dinfinity_generico/scripts/`
(`compute_hand_twi_dinfinity.py`, `prepare_region_dtm.py`, `compare_rasters.py`) — MDT de
entrada e diretório de saída como parâmetros, não hardcoded pra região nenhuma. WhiteboxTools
`fill_depressions_wang_and_liu` → `d_inf_flow_accumulation` → `slope` → `wetness_index` →
`elevation_above_stream`, mesma sequência do v12.

**Validação contra Recife: reprodução bit a bit idêntica** (Pearson r=1,0000000, diferença
máxima 0,0 m em `hand_dinf.tif` e `twi_dinf.tif`, 3,48M/3,58M células comparadas; limiar P98 =
1122,7383 bate com o documentado 1122,74). Teste pytest real em
`tests/test_susc_20g_hand_twi_dinfinity.py` (12 passaram, inclui a regressão contra Recife —
pula com motivo explícito se os rasters de referência não estiverem montados, não passa por
omissão).

**Aplicado às duas regiões** (rasters em `local_runs/susc_20g_hand_twi_dinfinity_generico/`,
git-ignored, não comitados por regra):
- Curitiba: MDT nativo 2,5 m agregado a 10 m (paridade com Recife), EPSG:31982, 4.357.585
  células, cobertura HAND 96,79%.
- Petrópolis: **ressalva real, não contornada** — o ZIP do SGB/CPRM não tem pasta de MDT
  (só Declividade/Fusão/Hipsometria/MDE/Relevo_sombreado); o caminho usado é modelo de
  **superfície**, resolução nativa 30,13 m, não reamostrado pra 10 m (interpolar não cria
  informação). Com célula de 30 m o HAND de Petrópolis (mediana 103,8 m) **não é
  numericamente comparável** ao de Recife (7,1 m) — funciona como raster de prontidão, não
  como feature pronta pra comparação direta entre regiões ainda.

**DTMs reais usados**:
- Curitiba: `PROJETO/data/raw/curitiba/sgb_cprm/produtos_mde_curitiba_pr.zip` (722MB, Esri
  ArcInfo Binary Grid, pasta `01.MDS_Hipsometria/mdt/`)
- Petrópolis: `PROJETO/data/raw/petropolis/sgb_cprm/produtos_mde_petropolis_rj.zip` (230MB,
  mesmo formato, pasta `MDE/pt_sirgas_utm/` — é MDS, não MDT, ver ressalva acima)

### 5.1 MDT verdadeiro de Petrópolis — resolvido (SUSC-20G/2)

A pendência acima foi fechada. Relatório completo:
`outputs_public/data/susc_20g_hand_twi_dinfinity_generico/reports/susc_20g2_petropolis_mdt_terreno_nu_report.md`.

- **Local, antes de baixar nada**: as 3 cópias do ZIP do SGB/CPRM são o mesmo arquivo (sha256
  idêntico). Achado novo dentro dele: `Hipsometria/hip_pt_utm23/` é grade de **10 m** — mas é a
  **mesma superfície** do `MDE/` (diferença média 0,35 m quando agregada a 30 m), não terreno nu.
  `bc_petropolis_rj.zip` não tem curva de nível nem ponto cotado; `ibge_ana_auxiliary/` está
  vazio. Nada local resolvia.
- **Copernicus DEM GLO-30 — testado e rejeitado**: é DSM por definição, e contra o MDE do SGB não
  há deslocamento de escala de dossel (média −2,17 m, desvio 10,32 m). Trocaria superfície por
  superfície.
- **FABDEM V1-2 — aceito**: Copernicus DEM com floresta e edificação removidas, tile S23W044,
  `data.bris.ac.uk`, aberto e **sem login/token**; lido por range request dentro do zip de 10°
  (~2,6 MB em vez de 1,05 GB). Terreno nu **comprovado**, não assumido: estratificando FABDEM −
  GLO-30 por MapBiomas 2022, a queda é −7,64 m em formação florestal e −9,77 m em silvicultura,
  contra −0,79 m em pastagem e +0,31 m em água. Licença CC BY-NC-SA 4.0 (citar em publicação).
- **Erro corrigido no caminho**: Petrópolis é **EPSG:31983** (UTM 23S, meridiano central −45°),
  não 31984 como rotulado na rodada anterior. Rasters regerados. Curitiba (31982) conferida.
- **Ressalva que continua**: FABDEM é 30 m; Recife e Curitiba são 10 m. Paridade de **tipo de
  superfície** resolvida, paridade de **resolução** não — e o grid local de 10 m disponível é
  superfície, então a escolha é excludente. Nenhuma combinação dos dois foi feita (subtrair um do
  outro para "fabricar" um MDT de 10 m seria dado inventado).

## 6. Ordem real de próximos passos

1. ~~Reescrever/testar script D-infinity genérico, validado contra Recife~~ — **feito**
   (SUSC-20G, seção 5). ~~Pendência: MDT verdadeiro de Petrópolis~~ — **feito** (SUSC-20G/2,
   seção 5.1, FABDEM V1-2). Resta a diferença de resolução (30 m contra 10 m), documentada.
2. ~~Definir critério de ponto negativo (metodologia, replicando Recife)~~ — **feito**
   (`docs/metodologia_cientifica/revp_criterio_ponto_negativo_recife_e_replicacao_curitiba_petropolis.md`).
   Amostragem em si continua não executada pra Curitiba/Petrópolis (N=1 positivo não sustenta
   pareamento por bairro ainda — decisão deliberada, não pendência esquecida).
3. Continuar aquisição de positivos com o protocolo A/B desta seção, região por região,
   documentando N real a cada rodada — sem pular pra treino antes do piso EPV.
4. Só quando N ≥ piso técnico (~20-30) por região: rodar Firth reduzido (2-3 features),
   nunca antes.

## Regras que continuam valendo, sem exceção

Uma tarefa por vez. Nunca fabricar ponto. Sucesso = tentativa real + resultado real. Nenhuma
região passa a `region_maturity="available"` sem `model_version` real (bloqueado no schema
Pydantic).
