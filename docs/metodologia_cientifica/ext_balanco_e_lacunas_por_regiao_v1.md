# Balanço da frente externa e lacunas por região — v1

**Data**: 2026-08-07
**Status**: diagnóstico honesto de cobertura; nenhuma decisão de modelagem
**Pergunta que responde**: o dado externo adquirido resolveu os problemas das
regiões do projeto, ou resolveu outra coisa?

---

## 1. Resposta curta

**Não resolveu Recife, Curitiba nem Petrópolis.** Resolveu o Rio Grande do
Sul, que não era região do projeto, e produziu um piloto inglês que é forte
metodologicamente mas é de outro país.

Isso não torna o trabalho inútil — mas é preciso nomear com precisão o que
foi ganho, porque confundir isso levaria a declarar resolvido um gate que
continua aberto.

---

## 2. O que foi efetivamente ganho

### 2.1. Um negativo formal por observação, no Brasil (EMSR720, RS)

| | |
|---|---|
| AOI declarada e analisada | 257,3 km² |
| Cobertura por imagem | 100% da AOI |
| Inundação observada | 36,44 km² (1.490 feições) |
| Deslizamento | 4,27 km² (497 feições) |
| **Negativo observado estrito** | **216,55 km²** (razão 5,94 : 1) |

Isto é **registro de ausência**, não ausência de registro. É a primeira vez
que o projeto tem essa categoria. Vale só para o RS.

### 2.2. Um piloto com escala e disciplina que o projeto nunca teve (Inglaterra)

7.476 pontos, 3.738 positivos e 3.738 negativos pareados por cobertura do
solo, **201 eventos independentes**, com agrupamento por evento na validação.
Contra os 278 pontos do v12 de Recife. Serve para testar o método, não para
resolver as regiões brasileiras.

### 2.3. Um argumento, não um dado, para Petrópolis

É aqui que o mal-entendido mora. O dado britânico **não supre Petrópolis**.
O que ele fornece é a demonstração de que a separação enchente/deslizamento é
um **atributo de origem** — a Environment Agency traz `fluvial_f`/`coastal_f`/
`tidal_f`, o NOAA traz `Debris Flow` como classe própria, o CEMS traz
`event_type` com `5-Flood` e `6-Mass Movement`. Isso sustenta a afirmação de
que Petrópolis é insolúvel *sem essa informação na origem*, o que é uma
conclusão científica legítima. Mas conclusão não é dado: para modelar
Petrópolis é preciso um inventário brasileiro que faça a separação.

---

## 3. Lacunas por região e fontes candidatas

### 3.1. Petrópolis — falta separar mecanismo

**O que existe e não foi buscado.** Após fevereiro de 2022, CPRM, DRM-RJ,
PUC-Rio, UERJ e UFRJ atuaram junto à Defesa Civil de Petrópolis e produziram:

- **611 cicatrizes de deslizamento identificadas**
- **402 polígonos de risco remanescente**
- 283 laudos técnicos
- **15 Mapas de Risco Remanescente na escala 1:5.000**

Isso é exatamente o inventário que falta — e com separação de mecanismo por
construção, já que cicatriz de deslizamento é geometria de deslizamento, não
de inundação. Escala 1:5.000 é melhor que qualquer coisa que a frente externa
trouxe.

**Situação de acesso**: não confirmada como dado aberto. O relatório técnico
do DRM-RJ é público em PDF; os vetores, não se sabe. O repositório já tem um
modelo de pedido via LAI (`docs/metodologia_cientifica/
modelo_pedido_lai_defesa_civil_petropolis_2022_v1nj.md`) que nunca foi usado.

**Prioridade**: alta. É a única rota que destrava Petrópolis de verdade.

### 3.2. Recife — falta negativo

Rota mais madura do projeto (v12, n=278, LOO-AUC 0,6781), mas com negativo
construído por ausência.

**Fonte candidata não explorada**: o portal `dados.recife.pe.gov.br` publica
conjuntos da Secretaria Executiva de Defesa Civil — solicitações de
atendimento, áreas de risco, vistorias realizadas, colocação de lona. É CKAN,
com API em `/api/3/action/package_search`.

O valor está na estrutura: solicitação de atendimento é **evento datado e
localizado**, e a existência de solicitações permite raciocinar sobre onde
houve demanda e onde não houve — que é matéria-prima de negativo com
denominador, na mesma lógica das apólices do NFIP norte-americano.

**Ressalva**: solicitação de atendimento tem viés de reporte forte (depende de
alguém ligar). Não é negativo por observação; é melhor que ausência pura, pior
que AOI declarada.

**Prioridade**: alta. É a região mais madura e o ganho seria imediato.

### 3.3. Curitiba — falta explicar o colapso temporal

Firth existe (LOO-AUC 0,6459) mas colapsa em holdout temporal 2026
(AUC 0,5246). Sete diagnósticos internos não explicaram.

**Fontes candidatas não exploradas**:
- IPPUC — download de dados geográficos em shapefile (`ippuc.org.br/geodownloads`)
- Portal de dados abertos da Prefeitura (ArcGIS Hub, com CSV/GeoJSON)
- Coordenadoria Estadual de Defesa Civil do PR — Mapas de Ocorrências e GEODC

**Prioridade**: média. O problema de Curitiba pode não ser de dado, e sim de
método — o piloto inglês, com 201 eventos e holdout temporal real, é o teste
mais direto dessa hipótese e já está montado.

### 3.4. Rio Grande do Sul — nova região, decidida

Entra como **quarta região** do projeto. É a única com negativo formal por
observação e mecanismo separado na origem. Falta: adquirir DEM, HAND, TWI,
WorldCover e chuva para a AOI do EMSR720, replicando o que foi feito para a
Inglaterra.

---

## 4. Estado real do plano de níveis

| Nível | Fonte | Status |
|---|---|---|
| 1 | Sen1Floods11 | ✅ 446 chips; reduzido a 45.340 pontos |
| 1 | UFO | ✅ 215 chips; reduzido a 25.800 pontos |
| 1 | Copernicus EMS | ⚠️ inventário completo; vetores só do EMSR720 |
| 1 | Global Flood Database | ⚠️ inventariado (913 obj., 16,3 GB); **não baixado por decisão** |
| 2 | EA Recorded Flood Outlines (Inglaterra) | ✅ 31.672 outlines, pipeline completo |
| 2 | NOAA Storm Events (EUA) | ❌ não iniciado |
| 2 | USGS high-water marks | ❌ não iniciado |
| 2 | FEMA NFIP claims + policies | ❌ não iniciado |
| 2 | IdroGEO / Itália | ❌ não iniciado |
| 2 | HOWAS21 / Alemanha | ❌ não iniciado |
| 2 | HANZE / Europa | ❌ não iniciado |
| 2 | BDHI / França, CNIH / Espanha | ❌ não iniciado |
| 2 | India Flood Inventory / INDOFLOODS | ❌ não iniciado |
| 2 | PetaBencana / Indonésia | ❌ não iniciado |
| 3 | EM-DAT/GDIS, DFO, DesInventar | ❌ e **não deve ser feito** — granularidade incompatível |

**Menos da metade do Nível 2 foi tocado.** Mas isso não é atraso: as fontes do
Nível 2 são substitutas entre si, não complementares. Uma região externa bem
feita vale mais que oito mal feitas. A questão certa não é "quantas faltam" e
sim "qual delas cobre uma lacuna que a Inglaterra não cobre".

Por esse critério, as que ainda valem:

- **NOAA Storm Events**: única com `Debris Flow` como classe irmã de `Flash
  Flood` — o análogo direto do problema de Petrópolis, com 1950–2026.
- **PetaBencana**: pluvial urbano denso com altura de lâmina, e clima tropical.
  O recorte pluvial inglês é frágil (56 pontos a 200 m).
- **India Flood Inventory**: análogo climático tropical/monçônico do Brasil.

As demais europeias adicionariam pouco sobre o que a Inglaterra já deu.

---

## 5. Links para verificação manual

- Lista completa de ativações CEMS: <https://mapping.emergency.copernicus.eu/activations/EMSR/>
  (renderizada por JavaScript; não é acessível por requisição simples)
- Portal de download de ativação: `https://rapidmapping.emergency.copernicus.eu/<CODIGO>`
- Dados abertos Recife: <https://dados.recife.pe.gov.br/organization/secretaria-executiva-de-defesa-civil>
- IPPUC geodownloads: <https://ippuc.org.br/geodownloads/geo.htm>
- Relatório DRM-RJ Petrópolis 2022: <https://www.rj.gov.br/drm/sites/default/files/arquivos_paginas/RL_09.2022.01-MTDLG-PETROPOLIS.pdf>

---

## 6. Declaração

Nenhum gate foi alterado. `C4_BLOCKED_NO_FORMAL_NEGATIVES` permanece aberto
para Recife, Curitiba e Petrópolis. O EMSR720 fornece negativo formal apenas
para a AOI do Rio Grande do Sul.
