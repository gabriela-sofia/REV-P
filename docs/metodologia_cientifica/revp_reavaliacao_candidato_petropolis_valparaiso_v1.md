# Reavaliação do candidato Petrópolis/Valparaíso à luz da evidência topográfica (v1)

**Data**: 2026-07-28 | **Decisão**: **REBAIXADO A CANDIDATO FRACO** — Petrópolis volta a
**N = 0 pontos-evento adjudicados**.

Este documento existe porque a adjudicação original (v14, critério SUSC-20A) e a leitura
topográfica de SUSC-20G/2 chegam a conclusões opostas sobre o mesmo ponto, e a regra do projeto
é decidir por escrito, não por omissão.

**O ponto**: lat −22,51625, lon −43,18828, sinal de 2022-03-24 (Sentinel-2 L2A, B03/B08/B11
cruas), bairro Valparaíso, Petrópolis/RJ.

---

## 1. O que pesa a favor de manter

1. **A queda de reflectância é real e absoluta, não índice relativo.** B08 caiu de 0,304 (04/03)
   para 0,122 (24/03); B11 caiu de 0,146 para 0,090. Água tem reflectância baixa em NIR e SWIR;
   nuvem sobe nas três bandas. Foi exatamente esse critério que separou este cluster dos 1600+
   clusters de ruído da primeira tentativa (v14, "Correção de método que destravou isto"). O
   falso-positivo de nuvem **está** descartado.
2. **Concentração espacial coerente**: 4 clusters vizinhos (111, 95, 86, 66 pixels) na mesma
   área, não pixels espalhados.
3. **Coerência temporal**: 24/03 é 2 dias depois do registro S2ID de 22/03/2022 — água que ainda
   não baixou, mesma lógica aceita em Curitiba.
4. **Dentro do município**, geocodificação real (Nominatim), e contexto externo verdadeiro
   (Agência Brasil sobre transbordamento recorrente dos rios do centro de Petrópolis).

## 2. O que pesa contra

### 2.1 Três critérios físicos independentes falham juntos

Leitura de SUSC-20G/2 (`susc_20g2_petropolis_mdt_terreno_nu_report.md`), MDT FABDEM 30 m:

| Métrica | Valor no ponto | Referência regional | Leitura |
|---|---:|---:|---|
| HAND | 50,88 m | mediana 109,31 m | 50 m acima da drenagem |
| Declividade | 23,34° | mediana 23,57° | encosta típica de serra, não fundo de vale |
| TWI | 5,11 | mediana 5,58 (percentil 31,9 %) | **abaixo** da mediana — sem convergência de fluxo |

Enchente acumula onde a água converge e o terreno é suave. Aqui não há nem uma coisa nem outra.

### 2.2 Não é artefato de resolução nem do limiar de drenagem

Esta era a ressalva mais forte contra confiar no HAND: pixel de 30 m, bbox recortada, fundo de
vale estreito. Foi testada com uma medida que **não depende** de extração de drenagem nem de
limiar — altura sobre o mínimo local — em **dois DEMs independentes e duas resoluções**:

| Raio | FABDEM 30 m (MDT) | Hipsometria SGB 10 m (MDS) |
|---|---:|---:|
| 150 m | +38,4 m | +30,2 m |
| 300 m | +50,9 m | +46,1 m |
| 500 m | +56,5 m | +50,2 m |

Os dois concordam: o ponto está algumas dezenas de metros acima de qualquer fundo de vale
próximo. **A ressalva de resolução não explica o achado** — ela foi levantada honestamente na
rodada anterior e agora está descartada por medição.

### 2.3 A corroboração hidrográfica é mais fraca do que o registrado

O v14 registra "Rio Quitandinha mapeado a ~500 m". Consulta Overpass/OSM refeita hoje sobre o
ponto exato, raio de 800 m — **só 2 cursos d'água mapeados**:

| Curso d'água | Distância | Elevação (FABDEM / Hipsometria) | Desnível até o candidato |
|---|---:|---:|---:|
| Rio Aureliano | **635 m** | 842,0 / 855,7 m | +39,1 / +34,5 m |
| Rio Quitandinha | **688 m** | 834,1 / 860,0 m | +47,0 / +30,2 m |

O candidato está a 881,1 m (FABDEM) / 890,2 m (Hipsometria). Ou seja: o rio corroborador não
está a ~500 m, está a ~690 m, e **30 a 47 m abaixo** do ponto. Para o cluster ser lâmina d'água
do Quitandinha, o rio teria que ter subido dezenas de metros verticais — o que não é o que uma
enchente urbana de fundo de vale faz.

### 2.4 A hipótese alternativa nunca foi testada — e não pode ser testada agora

A revisão de literatura (seção 3) registra que sombra — de nuvem **ou de relevo** — é fonte
documentada de confusão com água justamente por rebaixar NIR/SWIR. Uma encosta de 23° em vale
encaixado de serra, em imagem de fim de março, é candidata natural a sombra de relevo. O teste
que separaria as duas hipóteses é simples: se **B03 também caiu** junto com B08 e B11, é sombra
(sombra derruba todas as bandas); se B03 se manteve enquanto B08/B11 caíram, é água.

**Bloqueio real**: o v14 reporta os valores de B08 e B11, mas **não** os de B03. E as 6 cenas
originais (B03/B08/B11 de 04/03 e 24/03/2022) **não estão em disco** — varredura de
`PROJETO/` e `REV-P/` encontra só o corpus legado `petropolis_*__sentinel__B3B8B11__10m__v1.tif`,
que é outro conjunto (patches de 10 m de outra linhagem), não as cenas desta adjudicação. O teste
decisivo fica **pendente por indisponibilidade de dado**, não por falta de método.

## 3. Decisão

**Rebaixado a candidato fraco. Petrópolis volta a N = 0 pontos-evento adjudicados.**

Por que não "manter com ressalva": três critérios físicos independentes falham simultaneamente,
a única defesa disponível (artefato de resolução) foi testada e descartada com dois DEMs, e a
corroboração hidrográfica registrada estava otimista em ~190 m e ignorava um desnível de 30-47 m.
Um ponto que sobrevive só por reflectância, com toda a geomorfologia contra, não é base honesta
para nada.

Por que não "rejeitado": a queda absoluta de NIR+SWIR é real e o falso-positivo de nuvem está
genuinamente descartado. A hipótese que explicaria o sinal sem ser água — sombra de relevo — é
plausível mas **não foi demonstrada**, porque as cenas sumiram. Declarar rejeição seria afirmar
algo que não posso mostrar. O candidato fica registrado, não apagado.

**Consequência prática, sem suavizar**: Petrópolis não tem nenhum ponto-evento positivo utilizável.
Não é N=1 frágil — é N=0. Curitiba segue com N=1. Nenhuma das duas regiões chega perto do piso
EPV ≥ 20 que a literatura sustenta (revisão, seção 5), e `region_maturity` de Petrópolis continua
`insufficient` com `model_version=None`.

## 4. O que reabriria o caso

1. **Recuperar B03 das cenas de 04/03 e 24/03/2022** (reaquisição via Copernicus OData, mesma
   rota do protocolo Via B) e testar a hipótese de sombra. Se B03 se manteve enquanto B08/B11
   caíram, o candidato volta a pleno.
2. Máscara de sombra de relevo calculada do MDT (ângulo solar real da data) sobre o mesmo
   recorte — descarta ou confirma sombra topográfica sem depender de banda extra.
3. Qualquer registro oficial com geometria em Valparaíso para 22-24/03/2022.

Nada disso é objetivo desta rodada; fica registrado como caminho concreto, não como promessa.

## 5. O que muda no repositório

- `outputs_public/data/susc_20e_api_contrato_inferencia_recife/scripts/region_registry.py`:
  `status_note` de Petrópolis reescrito para N=0 e candidato fraco.
- Nenhuma mudança em `region_maturity` (já era `insufficient`) nem em `model_version` (já era
  `None`).
- A adjudicação original (v14) **não é apagada nem reescrita** — continua como registro do que
  foi decidido com a informação daquele momento. Este documento é a camada de revisão.
