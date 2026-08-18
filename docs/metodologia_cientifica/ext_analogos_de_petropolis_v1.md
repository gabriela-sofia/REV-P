# Análogos de Petrópolis — o que estava errado e o que fazer

**Data**: 2026-08-10
**Artefatos**: `local_runs/cems-02-analogos-v2/`
**Substitui operacionalmente**: o ranking de `cems01_analogos_por_regiao.py` (v1)

---

## 0. A tese está certa; a execução falhou

A rota "achar região do mundo com as mesmas características e propagar" é
válida e é a rota principal. Este documento não a contesta — documenta que ela
**não foi executada**, por um erro mensurável, e mostra como executar.

---

## 1. O perfil físico de Petrópolis, medido

Lido do Copernicus DEM GLO-30 numa janela de ~18 km em torno de −22,505 /
−43,178, com os mesmos controles do projeto:

| | elev. mediana | relevo local (p95−p5) | decliv. mediana | % área >15° | % área >25° |
|---|---|---|---|---|---|
| **Petrópolis** | **907 m** | **1.152 m** | **24,8°** | **81%** | **49%** |
| Curitiba | 919 m | 100 m | 4,1° | 3% | 0% |
| Recife | 3 m | 37 m | 0,7° | 3% | 0% |
| Piloto inglês | 89 m | 163 m | 3,9° | 4% | 1% |

Petrópolis não é "uma região um pouco mais íngreme". É outra categoria: **quatro
quintos da área acima de 15°**, contra 3–4% em tudo o mais que o projeto usa.
Note que Curitiba tem a mesma elevação de Petrópolis e relevo onze vezes menor —
**elevação não distingue serra de planalto; relevo local distingue.**

---

## 2. O erro: relevo lido no centroide da ativação

O `cems01` v1 pontuava cada ativação lendo o DEM numa janela de ~16 km em torno
do **centroide da ativação**. O centroide é a média das AOIs. Quando a ativação
cobre um país inteiro, ele cai onde não há AOI nenhuma.

Verificação nas AOIs reais (mediana das 3 maiores AOIs de cada ativação, medida
onde os pontos foram efetivamente amostrados):

| ativação | relevo no centroide (que rankeou) | relevo nas AOIs (real) | decliv. real | veredito |
|---|---|---|---|---|
| EMSR867 Madagascar | 513 m / 19,2° | **70 m** / 2,8° | AOIs a 4,7 e 11,5 m de altitude | **falso análogo** |
| EMSR790 La Réunion | 1.476 m / 19,1° | **1.108 m** / 6,0° | 26% acima de 15° | **análogo real** |
| EMSR851 Sri Lanka | 245 m / 5,2° | 28 m / 3,0° | planície | falso análogo |
| EMSR870 Equador | 29 m / 4,5° | 113 m / 2,4° | planície | falso análogo |
| EMSR754 Vietnã | 15 m / 1,6° | 23 m / 1,8° | planície | falso análogo |
| EMSR857 Moçambique | 28 m / 0,9° | 47 m / 0,7° | planície | falso análogo |

**O primeiro colocado do ranking de Petrópolis — Madagascar, EMSR867 — tem suas
AOIs no litoral, a 5 e 12 metros de altitude.** O centroide caía no planalto
central, a centenas de quilômetros das áreas mapeadas.

Cinco dos seis análogos baixados são planícies de inundação. Contra os 1.152 m
de relevo de Petrópolis, eles entregam de 23 a 113 m.

### Consequência, medida nas 56 AOIs do conjunto

Aplicando o critério da seção 3 a **todas as 56 AOIs** do dataset multirregião:

```
AOIS_PERFILADAS=56   APROVADAS_COMO_INGREME=5
```

| AOI | país | pontos | relevo local | decliv. | % >15° |
|---|---|---|---|---|---|
| EMSR851_AOI04 | Sri Lanka | 240 | 936 m | 15,1° | 50% |
| EMSR857_AOI17 | Moçambique | 360 | 716 m | 11,4° | 35% |
| EMSR851_AOI09 | Sri Lanka | 255 | 566 m | 16,2° | 55% |
| EMSR851_AOI03 | Sri Lanka | 240 | 490 m | 14,3° | 47% |
| EMSR790_AOI02 | La Réunion | 423 | 1.315 m | 7,2° | 34% |

**Cinco de cinquenta e seis.** 1.518 pontos, 5 grupos — EPV de **1,25** com 4
features, contra o mínimo de 10 que o próprio projeto exige. Não dá para
treinar nem interpretar nada sobre terreno íngreme com isso.

Há um achado positivo dentro do negativo: **as terras altas centrais do Sri
Lanka (EMSR851) são o melhor material de serra tropical que já temos**, e
EMSR851 estava classificado como análogo de *Recife* — planície costeira. A
ativação certa estava no conjunto, rotulada para a região errada, porque a
pontuação era por ativação e não por AOI.

O conjunto de treino, portanto, não é vazio de terreno íngreme — é
**insuficiente por uma ordem de grandeza**. O modelo não pode ter aprendido a
física de encosta que Petrópolis exige.

---

## 3. Regra metodológica que decorre disso

> **Semelhança medida no centroide é CANDIDATO. Só a verificação por AOI
> confirma analogia.**

O centroide é geometria administrativa da ativação, não do fenômeno. A unidade
física é a AOI — é onde o analista olhou, onde o polígono foi desenhado e onde
os pontos são amostrados. Toda métrica de comparação deve ser calculada aí.

Isso vale como critério de aceitação: uma AOI só entra no conjunto declarado
como análogo de Petrópolis se seu relevo local ficar dentro de uma faixa
declarada antes da leitura. Proposta: **relevo local ≥ 400 m e ≥ 25% da área
acima de 15°.** Petrópolis dá 1.152 m e 81%; La Réunion dá 1.108 m e 26%; todas
as planícies ficam de fora por larga margem.

---

## 4. A fila corrigida — agora medida na geometria real

O que travava tudo era não haver como obter a geometria das AOIs sem baixar o
pacote vetorial. Existe:

```
dashboard-api/aois/?activation__code=EMSR867
```

Devolve o **polígono de cada AOI**. São 1.109 AOIs em todo o catálogo público,
771 em ativações de Flood/Storm/Mass movement, obtidas em **uma requisição**.
A analogia deixa de ser estimada e passa a ser medida onde o fenômeno foi
mapeado.

Cuidado registrado: `?activation=` e `?code=` são **aceitos e ignorados** pela
API — devolvem as 1.109 sem filtrar. Só `?activation__code=` filtra. É a mesma
família de erro da paginação do EA, e o `cems04` confere linha a linha.

### Resultado do perfilamento das 769 AOIs

```
AOIS_PERFILADAS=769   INGREMES=143 (18,6%)
INGREMES TROPICAIS (|lat| <= 25) = 43   ->  EPV com 4 features = 10,8
```

**43 AOIs de serra tropical existem no catálogo.** EPV 10,8 fica acima do
mínimo de 10 do projeto. A rota é viável — só não tinha sido percorrida.

| ativação | país | AOIs íngremes | relevo | decliv. | % >15° | já baixado |
|---|---|---|---|---|---|---|
| **EMSR847** | Haiti, Cuba, Jamaica | **7** | 573 m | 15,8° | 53% | não |
| **EMSR796** | Equador | **6** | 721 m | 24,8° | 86% | não |
| EMSR702 | Vanuatu | 5 | 608 m | 13,5° | 45% | não |
| EMSR851 | Sri Lanka | 4 | 563 m | 15,6° | 52% | **sim** |
| EMSR778 | Honduras | 3 | 455 m | 9,7° | 31% | não |
| EMSR734 | Caribe | 2 | 609 m | 17,9° | 61% | não |
| EMSR789 | Equador | 2 | 773 m | 24,3° | 86% | não |
| EMSR805 | México | 2 | 754 m | 16,5° | 55% | não |
| EMSR790 | La Réunion | 2 | 1.069 m | 11,7° | 40% | **sim** |
| EMSR813 | Equador | 1 | 790 m | 23,4° | 74% | não |

Prioridade: **EMSR847 → EMSR796 → EMSR702 → EMSR778**. Haiti é o análogo mais
próximo do problema social e físico de Petrópolis — serra tropical úmida com
ocupação densa em encosta.

### O método se corrigindo

A fila anterior deste documento, feita por centroide, punha **EMSR813 em
primeiro e EMSR778 com 12 AOIs**. Medido por AOI, EMSR813 tem **uma** AOI
íngreme e EMSR778 tem **três**, não doze. Os primeiros colocados reais —
Haiti e Equador/EMSR796 — não apareciam na lista por centroide.

Fora dos trópicos há mais 100 AOIs íngremes (Noruega EMSR683 com 20/23,
EMSR775 com 14/14, Eslovênia EMSR680 com 9/15). Servem para testar se a
relação HAND→inundação é estável em serra temperada — o que é um teste de
transferibilidade, não um análogo de Petrópolis.

---

## 5. Duas descobertas colaterais que mudam a leitura do modelo

### 5.1. As features de chuva só existem na Inglaterra

Cobertura de `rain_max_24h` e `rain_decay_index_api` por região:

```
UK_noroeste                          n=7.476   OK
todas as 6 ativações CEMS            SEM CHUVA
sen1floods11 / UFO                   SEM CHUVA
```

O modelo de 6 features é, hoje, exclusivamente inglês. Qualquer modelo
multirregião roda com **4 features**. Isso precisa estar declarado em qualquer
tabela de resultado que misture regiões.

### 5.2. A chuva quase não discrimina — e isso é do desenho, não defeito

Na Inglaterra, a mediana de `rain_max_24h` é **14,3 mm nos positivos e 13,8 mm
nos negativos**. Praticamente idêntica.

O motivo é estrutural: positivo e negativo são amostrados **no mesmo evento**,
logo compartilham a mesma chuva. Dentro de um evento a chuva é quase constante;
o que varia é o terreno. O coeficiente de chuva só é identificado pela variação
*entre* eventos.

Isso explica por que o modelo de 6 features (AUC 0,7999) supera o de 4
(0,7927) por apenas 0,007 — e leva a uma conclusão que precisa entrar no artigo:

> **O modelo do projeto é um modelo de suscetibilidade espacial, não de
> deflagração.** Ele responde "onde inunda quando chove", não "quando chove o
> bastante para inundar".

Para Petrópolis isso é bom: prever *onde* depende de terreno, e terreno é
global e disponível. Mas reforça que o análogo de terreno é tudo — se o terreno
de treino não se parece com o de Petrópolis, não sobra nada.

E impõe uma correção futura: se algum dia a chuva for usada entre regiões, **mm
brutos não transferem**. 79 mm em 24 h é extremo no noroeste inglês e ordinário
na serra fluminense em janeiro. O que transfere é a anomalia padronizada pela
climatologia local.

---

## 6. O que fica pendente

- Baixar e verificar EMSR778, EMSR813, EMSR850 pelo `cems02` (pipeline já existe).
- Marcar cada AOI do dataset com `classe_relevo` derivada da seção 3, para que
  o modelo possa ser treinado e avaliado por classe de terreno.
- Repetir a comparação de definição de negativo (`mod_neg01`) **dentro da classe
  de terreno íngreme**, quando houver AOI suficiente.
- Fontes fora do CEMS para o mecanismo de movimento de massa, que o CEMS quase
  não cobre (só 3 ativações, todas alpinas e minúsculas): COOLR/NASA, IFFI
  (Itália, 620 mil registros, 78% com polígono), ENTLI (Hong Kong) e, no Brasil,
  o S2iD com tipologia COBRADE separando enxurrada de deslizamento.
