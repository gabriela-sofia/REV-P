# Holdout temporal (E4/M3) — execução sobre a tabela única

**Data**: 2026-08-20
**Artefatos**: `local_runs/mod-prosp-02/`
**Script**: `scripts/suscetibilidade/mod_prosp02_holdout_temporal_ds05.py`
**Testes**: `tests/test_mod_prosp02_holdout_temporal.py` (22 passam)
**Precede**: `ext_validacao_prospectiva_e_mecanismo_v1.md` (MOD-PROSP-01, 12/08)
**Critérios**: fixados em `ext_criterios_de_acerto_v1.md` (09/08), antes desta rodada

---

## 1. O que estava em aberto

O E4 aparecia no plano de entrega como "teste ainda não realizado". Isso
deixou de ser verdade em 12/08/2026, quando o MOD-PROSP-01 rodou e deu
`PROSPECTIVAMENTE_ESTAVEL`. A frase sobreviveu por inércia de texto, não por
falta de resultado — e junto com ela sobreviveu um número errado: o plano
falava em "~158 datas", estimativa nunca conferida. **São 110 datas
distintas**, em 201 eventos independentes e 401 grupos de validação, entre
01/06/2000 e 01/01/2025.

Mas o MOD-PROSP-01 também não fechava o E4, por três motivos que só ficaram
visíveis ao reler o artefato:

**1. Rodou na base errada.** Lê `ds-01-multirregiao`, montada antes da
harmonização. A base congelada do E2/M1 é a tabela única (`ds-05`, 16/08).
Ali o piloto inglês tem 401 grupos, não 328.

**2. A data do negativo é artefato.** Os grupos do piloto são puros: 201
eventos só-positivos (`EV_*`) e 200 blocos espaciais de 5 km só-negativos
(`NEG_r_c`). O bloco negativo não tem data de evento — carrega as datas dos
positivos para os quais foi amostrado, mediana de 11,5 datas distintas por
bloco, até 35. Ordenar bloco por data mínima joga quase todo negativo para os
cortes antigos. A consequência, medida na reconstrução do MOD-PROSP-01:

| corte | pos | neg | prevalência do teste |
|---|---:|---:|---:|
| 2000-10-29 | 259 | 612 | 0,297 |
| 2012-06-22 | 521 | 280 | 0,650 |
| **2020-10-06** | **431** | **31** | **0,933** |
| 2025-01-01 | 336 | 162 | 0,675 |

O fold de 2020 mede AUC contra 31 negativos.

**3. Não tem intervalo de confiança.** A regra U2 de
`ext_uk_adjudicacao_negativo_v1.md` exige IC por reamostragem no nível do
grupo em todo desempenho reportado. O MOD-PROSP-01 reporta AUC nua.

---

## 2. O desenho, declarado antes de rodar

Janela expansiva, mesma regra de janela do precedente
(`max(8, n_unidades // 10)`), sobre o pool fluvial da tabela única.

**Estrato primário: o piloto inglês**, escolhido por **cobertura de
calendário** — critério que não depende de nenhum resultado: ele cobre 21
anos e as demais fontes cobrem 3 ou 4. As outras rodam como estratos
secundários para que essa escolha seja conferível em
`viabilidade_por_fonte.csv`, e não afirmada.

**Duas variantes**, porque a natureza do negativo muda o que a data significa:

- **`herança`** — réplica literal do MOD-PROSP-01 na base nova. Herda o
  defeito 2. Existe para responder "o resultado de 12/08 sobrevive à troca
  de base?" e nada além disso.
- **`bloco`** — o eixo temporal se aplica só ao positivo, que é quem tem data
  de evento. Os blocos negativos entram por sorteio determinístico (semente
  20260820), nenhum bloco dos dois lados do mesmo fold, e o teste recebe
  blocos até aproximar 1:1. Justificativa física: suscetibilidade do negativo
  é propriedade do terreno, não da data — o bloco não inundou em nenhuma das
  110 datas.

Qual variante vale para qual fonte é decidido pela **pureza dos grupos**, que
é estrutura do dado e não arbítrio: grupo puro significa negativo amostrado à
parte, com data herdada (piloto inglês, `exclusao_qualificada`) → vale a
`bloco`; grupo misto significa negativo observado na mesma AOI e na mesma
data do evento (CEMS, `observado`) → a data é real e vale a `herança`.

**Trava de EPV**: mínimo de `10 × n_variáveis` grupos **em cada classe** do
treino. IC95 por bootstrap percentil de grupos, N=1000. Cada fold declara
pontos, grupos, positivos e negativos dos dois lados (regra U3).

---

## 3. Resultado — estrato primário

Quatro combinações, todas com o mesmo veredito.

| conjunto | variante | folds | AUC médio | mín | máx | tendência | prevalência do teste | veredito |
|---|---|---:|---:|---:|---:|---:|---|---|
| TERRENO (4) | herança | 2 | 0,7974 | 0,7743 | 0,8206 | — | 0,90–0,99 | `PROSPECTIVAMENTE_ESTAVEL` |
| TERRENO (4) | **bloco** | **8** | **0,7992** | 0,7309 | 0,8559 | +0,120 | **0,46–0,50** | `PROSPECTIVAMENTE_ESTAVEL` |
| COMPLETO (6) | herança | 1 | 0,8058 | — | — | — | 0,92 | `PROSPECTIVAMENTE_ESTAVEL` |
| COMPLETO (6) | **bloco** | **7** | **0,7874** | 0,7076 | 0,9148 | −0,631 | 0,47–0,50 | `PROSPECTIVAMENTE_ESTAVEL` |

Nenhum fold abaixo de 0,60 em nenhuma combinação.

Trajetória da variante `bloco`, conjunto TERRENO — a que responde o E4:

| corte | treino | teste | AUC | IC95 |
|---|---|---|---:|---|
| 2002-06-13 | 40 ev | 284+/301− | 0,8403 | [0,7616; 0,9096] |
| 2004-08-20 | 60 ev | 234+/272− | 0,7928 | [0,6616; 0,8853] |
| 2010-07-14 | 80 ev | 368+/374− | 0,7992 | [0,6821; 0,8966] |
| 2016-06-11 | 100 ev | 613+/618− | 0,7710 | [0,7121; 0,8234] |
| 2020-02-09 | 120 ev | 425+/444− | 0,7309 | [0,6320; 0,8258] |
| 2023-10-20 | 140 ev | 256+/259− | 0,7779 | [0,6646; 0,8786] |
| 2025-01-01 | 160 ev | 321+/324− | 0,8253 | [0,7762; 0,8788] |
| 2025-01-01 | 180 ev | 399+/400− | 0,8559 | [0,8104; 0,8992] |

### O que isso sustenta, e o que não sustenta

**Sustenta**: a relação terreno→inundação não caduca no horizonte medido. Um
modelo ajustado só com eventos até 2002 ordena eventos que aconteceram até
vinte e três anos depois acima do acaso: nos oito cortes o limite inferior do
IC95 fica acima de 0,63, e os oito AUC caem dentro da faixa 0,70–0,88 fixada
antes.

**Sustenta também**, e este é o ponto que interessa ao caso de Curitiba: o
colapso temporal **não é propriedade do método**. Com a mesma rota linear, as
mesmas variáveis e um horizonte cinco vezes maior, o piloto não colapsa.

**Não sustenta**: nada sobre clima tropical de serra. É um país só, e o
negativo é por exclusão qualificada, não por observação de não-ocorrência.

**A tendência não deve ser lida.** Com 4 variáveis ela dá +0,120 nesta semente
e +0,834 na outra; com 6, dá −0,631 nesta e +0,286 na outra — troca de sinal
sem que nada no dado mude. É estatística de oito pontos sensível à alocação — não é medida de deriva, e não se afirma nada a
partir dela. O que sustenta a leitura é o piso: nenhum fold abaixo de 0,60 em
nenhuma das combinações.

**A correção do defeito 2 não mudou a conclusão, mudou a base dela.** A
variante `herança` sobrevive à troca de base, mas com 2 folds e prevalência
de teste entre 0,90 e 0,99 — ou seja, praticamente sem negativo para medir. A
`bloco` entrega 8 folds com prevalência 0,46–0,50. O veredito é o mesmo; o
que se pode dizer dele é que agora ele se apoia em oito medições equilibradas
e não em duas desequilibradas.

### Sensibilidade à alocação dos blocos negativos

A alocação é decisão, não dado — então foi medida em duas sementes. A rodada
corrente usa 20260820; a primeira execução, antes de acertar a data, usou
20260819. Os artefatos gravados são os da corrente; os da anterior estão
transcritos aqui para que a sensibilidade seja um número e não uma ressalva.

| conjunto | semente | folds | AUC médio | mín | máx | na faixa | veredito |
|---|---|---:|---:|---:|---:|---|---|
| TERRENO | 20260819 | 8 | 0,7890 | 0,6660 | 0,8802 | 6/8 | `PROSPECTIVAMENTE_ESTAVEL` |
| TERRENO | **20260820** | 8 | 0,7992 | 0,7309 | 0,8559 | 8/8 | `PROSPECTIVAMENTE_ESTAVEL` |
| COMPLETO | 20260819 | 7 | 0,8000 | 0,7397 | 0,8453 | 7/7 | `PROSPECTIVAMENTE_ESTAVEL` |
| COMPLETO | **20260820** | 7 | 0,7874 | 0,7076 | 0,9148 | 6/7 | `PROSPECTIVAMENTE_ESTAVEL` |

O AUC médio move-se cerca de um centésimo; o veredito não se move. **O que se
move de verdade é o fold individual**: o primeiro corte do TERRENO vai de
0,6660 a 0,8403 conforme quais blocos negativos caem no teste. Isso é o
tamanho real da incerteza de um fold com 40 eventos de treino, e é a razão de
o IC estar em todos eles — o IC de 0,6660 era [0,5318; 0,8146], que contém
0,8403.

---

## 4. Estratos secundários — por que só o piloto responde o E4

| fonte | anos | grupos | pos | neg | pts/grupo | variante | folds | leitura |
|---|---:|---:|---:|---:|---:|---|---:|---|
| **uk** | **21** | 401 | 3.738 | 3.738 | 18,6 | bloco | **8** | estrato primário |
| cems | 3 | 119 | 10.523 | 13.392 | 201,0 | herança | 6 | horizonte curto |
| curitiba | 4 | 1.157 | 1.238 | 114 | 1,2 | bloco | **0** | negativo insuficiente |
| sen1floods11 | 4 | 11 | 12.949 | 17.482 | 2.766,5 | herança | **0** | 11 grupos em 11 datas |

**CEMS** sustenta 6 folds com AUC médio 0,7229, mas com **tendência −0,673**
dentro de uma janela de três anos. Não dispara `DEGRADACAO_TEMPORAL` porque
nenhum fold cai abaixo de 0,60, e três anos não são horizonte — o que esses
folds medem é variação dentro de um mesmo regime. Fica registrado como sinal
a observar, não como conclusão.

**Curitiba não sustenta o teste**, e a razão é numérica: 114 negativos contra
1.238 positivos, com 1,2 ponto por grupo. Esse achado só apareceu porque a
primeira versão desta rodada **produziu** folds para Curitiba — três, com AUC
0,52–0,55 e veredito `DEGRADACAO_TEMPORAL`. Ao conferir a composição, o
treino desses folds tinha **2 grupos negativos**: a alocação 1:1 do teste
consumia quase todo o negativo disponível. Um ajuste assim produz número e
não significa nada.

A correção foi aplicar a regra de EPV como ela sempre significou — eventos da
classe rara por variável, **nas duas classes** — em vez de contar só o total
de grupos. Conferido: os 18 folds do estrato primário são idênticos antes e
depois da trava; ela derruba apenas os folds degenerados. O resultado honesto
para Curitiba é `SEM_FOLD_COM_EPV_SUFICIENTE`, e o AUC 0,52 de 2026 continua
sem explicação por esta via.

**Por que não o pool inteiro (65.070 pontos).** No pool, corte temporal é
corte de fonte. Um corte em 2020 põe Sen1Floods11+UK no treino e
CEMS+UK+Curitiba no teste, e faz o estrato ÍNGREME saltar de 775 para 5.762
pontos. Isso mede transferência entre fontes — que o MOD-MEC-03 já mede com
`leave_one_source_out` — e não estabilidade temporal. A matriz ano × fonte
está em `confusao_fonte_periodo.csv`.

---

## 5. Estado do E4 contra o que o plano exige

| exigência do plano | estado |
|---|---|
| entregável: desempenho por corte temporal | `folds.csv`, 26 folds com IC (18 no estrato primário, 8 no CEMS) |
| evidência: cortes definidos antes da execução | desenho declarado no cabeçalho do script; regra de janela herdada do precedente |
| não exige aquisição nova | confirmado: só a tabela única |

**E4 fechado** para o estrato onde ele é mensurável, com a limitação
declarada: um país, negativo por exclusão qualificada, sem contraparte
tropical de serra.

---

## 6. Limitações que ficam, e não ficam escondidas

- **Um país só.** A estabilidade prospectiva vale para o noroeste da
  Inglaterra. Ela refuta "o colapso é do método"; não prova estabilidade em
  Recife, Curitiba ou Petrópolis.
- **O negativo é por exclusão qualificada.** Não é observação de
  não-ocorrência. Herda o aviso da própria Environment Agency de que ausência
  de registro não é ausência de inundação.
- **A alocação dos blocos é uma decisão, não um dado.** A semente está
  declarada, o teste de determinismo protege a reprodução e a seção 3 mede a
  sensibilidade em duas sementes: o veredito não muda, o fold individual muda
  bastante.
- **O fold mais antigo é o mais fraco.** 40 eventos de treino, IC com
  amplitude de quase 0,15 — é o preço de começar a janela cedo, e a trava de
  EPV é o que impede começar antes ainda.
- **Curitiba segue sem holdout temporal próprio na base harmonizada** — e
  isso agora é um número (114 negativos), não uma impressão.

---

## 7. O que este resultado não destrava

O E3/M2 continua aberto: a tabela de coeficientes **por classe de relevo**
sobre a tabela única não existe. O que existe é `mod-serra-01`, ajustado na
base antiga (`ds-01`), e a avaliação por relevo do `mod-mec-03`
(`transferencia_relevo`: ÍNGREME 0,8005 [0,7539; 0,8464], PLANO 0,6877
[0,6689; 0,7051]). O gargalo do E3 são
os **24 grupos íngremes** do pool, todos estrangeiros, que pela trava de EPV
comportam no máximo 2 variáveis — e não tem relação com a ausência de
negativos em Petrópolis, que não tem nenhuma linha na tabela única.
