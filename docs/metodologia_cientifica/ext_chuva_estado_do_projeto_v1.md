# O estado da chuva no projeto inteiro

**Data**: 2026-08-20
**Artefatos**: `local_runs/aud-chuva-01/`, `local_runs/aud-chuva-02/`
**Scripts**: `aud_chuva01_fontes_incompativeis.py`, `aud_chuva02_escala_do_contraste.py`
**Testes**: `tests/test_aud_chuva02_escala.py` (11), `tests/test_ds03_ds05_tabela_unica.py` (invariantes de chuva)
**Precede**: `ext_chuva_fonte_unica_recife_v1.md` (correção de Recife, 16/08)

---

## 1. Resposta curta

A correção de Recife virou propriedade do projeto inteiro: **as seis fontes da
tabela única usam o mesmo produto de precipitação, a mesma janela e o mesmo
fator de decaimento**, com 99,99% de cobertura. A pergunta de procedência está
fechada e agora tem teste que a protege.

Mas fechar procedência abriu a pergunta seguinte, que ninguém tinha feito: **a
chuva varia na escala em que o modelo compara?** A resposta, medida em doze
pares fonte × variável, é **não em nenhum deles**. E o único lugar onde a chuva
ainda pesa num modelo — Recife — é o único onde positivos e negativos quase não
dividem datas, o que faz o coeficiente descrever *quando*, não *onde*.

---

## 2. Procedência: resolvida, e agora em toda a base

A `aud_chuva01` de 13/08 dizia que CEMS, Sen1Floods11 e UFO estavam `SEM_CHUVA`
e que o piloto inglês usava `chirps_v3_rnl`. Isso deixou de valer quando o
`chuva04_adquirir_era5_global.py` reextraiu a precipitação de toda a base. O
artefato ficou desatualizado por quatro dias; foi reexecutado em 20/08 e hoje
reflete a base.

| fonte | n | produto | período | veredito |
|---|---:|---|---|---|
| cems | 25.249 | Open-Meteo/ERA5-Land | 2024-07 → 2026-03 | `FONTE_UNICA` |
| curitiba | 1.680 | Open-Meteo/ERA5-Land | 2023-01 → 2026-07 | `FONTE_UNICA` |
| recife | 278 | Open-Meteo/ERA5-Land | 2008-03 → 2025-01 | `FONTE_UNICA` |
| sen1floods11 | 30.586 | Open-Meteo/ERA5-Land | 2016 → 2019 | `FONTE_UNICA` |
| ufo | 25.800 | Open-Meteo/ERA5-Land | 2017 → 2021 | `FONTE_UNICA` |
| uk | 7.476 | Open-Meteo/ERA5-Land | 2000-06 → 2025-01 | `FONTE_UNICA` |

Mesma fórmula em todas: janela de 14 dias, decaimento 0,85/dia, grade de 0,1°,
fuso local do ponto. Nenhuma fonte com mistura. Cobertura: só 9 pontos de
Recife ficam sem valor, e é lacuna pré-existente (pontos sem `event_date` na
origem).

**O custo dessa escolha, declarado**: ERA5-Land é reanálise, não produto com
estação. O piloto inglês perdeu a validação por pluviômetro que o CHIRPS
gauge-blend dava, em troca de fonte única em toda a base. A troca foi
deliberada e está registrada no `chuva04`.

**O invariante agora tem guarda.** O teste que existia protegia só Recife —
nasceu quando só Recife tinha o problema. Foram acrescentados dois: um verifica
que nenhuma fonte tem mais de um produto na mesma coluna, outro que a base
inteira tem produto único. Se alguém reextrair uma fonte com outro produto, os
testes falham antes de o número aparecer num modelo.

---

## 3. Escala: a pergunta que faltava

O projeto define suscetibilidade como a predisposição do **terreno** a acumular
água sob um dado forçamento de chuva. A pergunta de pesquisa é sobre **quais
lugares** inundam. O modelo, coerente com isso, compara pontos: dentro do mesmo
evento, quais inundaram e quais não.

A chuva do projeto vive em células de 0,1° — cerca de 11 km — e varia por dia.
O contraste do modelo vive a dezenas ou centenas de metros, dentro do mesmo
dia. **São escalas diferentes**, e é isso que a `aud_chuva02` mede.

### 3.1. Quanto da variação da chuva está entre grupos

| fonte | células de 0,1° | valores distintos | pontos por valor | variância entre grupos |
|---|---:|---:|---:|---:|
| recife | **4** | 123 | 2,2 | 100,0% |
| curitiba | **10** | 302 | 5,6 | 100,0% |
| uk | 85 | 323 | 23,1 | 66,3% |
| ufo | 128 | 108 | 222,6 | 99,2% |
| sen1floods11 | 642 | 348 | 87,4 | 81,7% |
| cems | 797 | 586 | 40,8 | 96,5% |

Recife inteiro cabe em 4 células de chuva; Curitiba, em 10. Nessas duas
regiões a chuva **não é uma variável espacial**: é essencialmente o valor do
dia, repetido para a cidade toda.

### 3.2. A chuva separa dentro do grupo?

| fonte | grupos com as duas classes | com chuva constante | utilizáveis | AUC dentro do grupo |
|---|---:|---:|---:|---:|
| cems | 94 | 9 | 85 | **0,4934** |
| sen1floods11 | 11 | 0 | 11 | **0,4737** |
| ufo | 215 | 114 | 101 | **0,5113** |
| recife | 0 | — | 0 | grupo é o ponto |
| curitiba | 0 | — | 0 | grupo é o ponto |
| uk | 0 | — | 0 | grupos puros |

Onde dá para medir, a chuva é indistinguível do acaso na escala em que o modelo
decide. Onde não dá, não dá por construção: em Recife e Curitiba o grupo de
validação **é** o ponto; no piloto inglês os grupos são puros (evento
só-positivo, bloco só-negativo).

**Placar final: 0 de 12 pares fonte × variável mostram sinal de chuva na escala
do modelo.**

### 3.3. O caso de Recife — o achado que pede decisão

| fonte | datas com as duas classes | % dos pontos | AUC global | AUC só nas datas compartilhadas |
|---|---:|---:|---:|---|
| **recife** | **5 de 205** | **3,7%** | 0,6282 | 0,4600 (n=10) |
| curitiba | 108 de 642 | 29,6% | 0,5155 | 0,5070 (n=497) |
| uk | 106 de 110 | 99,9% | 0,5171 | 0,5173 (n=7.472) |
| cems | 12 de 12 | 100% | 0,5179 | 0,5179 (n=23.915) |

Em Recife, positivos e negativos praticamente **não dividem datas**. Comparar
chuva entre eles é comparar dias, não lugares — e "choveu mais no dia em que
houve enchente" é quase tautológico quando o negativo foi amostrado por
pareamento de bairro, em outros dias.

Restrita às 5 datas compartilhadas, a AUC cai de 0,6282 para 0,4600. **Isso não
prova nada com n=10** — e é justamente esse o ponto: o desenho da amostra não
deixa a pergunta ser respondida. O que se pode afirmar é o desenho, não o
número.

É a mesma espécie de confundimento que a `aud_chuva01` achou na procedência, um
nível acima: lá a variável carregava qual campanha extraiu o dado; aqui carrega
em que dia o ponto foi amostrado.

Curitiba é o contraste útil: 29,6% dos pontos em datas compartilhadas, e a AUC
do `rain_decay_index` **sobrevive** à restrição (0,6195 → 0,6094, n=497). Ali a
chuva antecedente diz alguma coisa que não é só a data — coerente com o
SUSC-20M, que já tinha achado a chuva antecedente como o único termo estável.

---

## 4. O que a chuva faz dentro dos modelos hoje

| modelo | papel da chuva | número |
|---|---|---|
| `mod-mec-03` (pool fluvial, 63.174 pts) | ganho de AUC ao acrescentar as duas variáveis | **−0,0006** |
| `mod-mec-03` | `rain_max_24h` | coef −0,1847, IC [−0,383; +0,023] — cruza zero **com sinal invertido** |
| `mod-mec-03` | `rain_decay_index` | coef +0,2376, IC [+0,003; +0,454] — encosta em zero |
| `mod-prosp-02` (E4) | TERRENO vs COMPLETO | 0,7992 vs 0,7874 — acrescentar chuva **não melhora** |
| `mod-pluv-01` (Recife) | só chuva | LOO-AUC 0,5813 |
| `mod-pluv-01` (Recife) | só terreno | LOO-AUC 0,5795 |
| `mod-pluv-01` (Recife) | completo | LOO-AUC 0,6339 (v12 publicado: 0,6781) |
| `mod-pluv-01` (Recife) | `rain_decay_index` | coef +0,6396, IC [+0,221; +1,149] |

O único lugar do projeto onde uma variável de chuva tem IC longe de zero é o
`rain_decay_index` de Recife — e é exatamente o lugar onde a §3.3 mostra que o
contraste é temporal. **Os dois fatos precisam ser lidos juntos**, e é isso que
impede tratar aquele coeficiente como propriedade do terreno.

Note também que `rain_max_24h` aparece com **sinal negativo** em Recife
(−0,2532) e no pool fluvial (−0,1847). Fisicamente, mais chuva de pico
associada a menos inundação não se sustenta; o IC cruza zero nos dois casos, o
que é a leitura correta — a variável não está medindo o que o nome diz.

---

## 5. O que fazer para a chuva funcionar no projeto todo

Três rotas, com custo e consequência diferentes. **Nenhuma foi executada; a
escolha é sua.**

### Rota A — chuva como forçamento, não como feature (recomendada)

É o que a própria definição do projeto já diz: suscetibilidade é a predisposição
do terreno **sob um dado forçamento de chuva**. O forçamento é a *condição* em
que o mapa vale, não uma coluna do vetor de variáveis.

Na prática: o modelo de lugar usa só terreno; a chuva entra no contrato de
inferência como cenário declarado ("este mapa vale para um evento de X mm em
24 h"). Isso resolve a incoerência de escala por construção, em vez de
disfarçá-la.

**Custo**: baixo — é reorganização, não aquisição. **Evidência a favor**: o
ganho da chuva no pool é −0,0006 e o E4 não muda ao incluí-la; tirar a chuva do
vetor não custa desempenho em lugar nenhum, exceto Recife, onde o que ela
acrescenta é o confundimento temporal. **Contra**: o TCC descreve seis
variáveis; passar a quatro exige reescrever a Tabela I e o parágrafo de
variáveis.

### Rota B — dar contraste temporal à amostra de Recife

Amostrar negativos **nas mesmas datas** dos positivos, no lugar do pareamento
só por bairro. Aí a comparação vira "mesmo dia, lugares diferentes", que é a
pergunta certa, e a chuva passa a poder ser interpretada.

**Custo**: médio — é redesenho de amostragem sobre o inventário SEDEC, não
aquisição nova de chuva. **Consequência**: invalida a comparabilidade direta com
o v12 publicado, porque muda o conjunto de negativos.

### Rota C — precipitação em resolução compatível

Trocar ERA5-Land (11 km) por produto de radar ajustado por pluviômetro —
MERGE/CPTEC no Brasil, NIMROD no Reino Unido — que chega a 1 km ou menos.

**Custo**: alto — é aquisição nova, com cobertura temporal menor e sem garantia
de existir para todos os períodos das seis fontes. **E resolve menos do que
parece**: mesmo a 1 km, a chuva continua praticamente constante dentro de um
evento urbano de poucos quilômetros. Melhora a §3.1, não necessariamente a
§3.2.

**Recomendação**: A, e B se Recife for reaberto. C só faria sentido depois de A
e B, e para uma pergunta diferente da atual.

---

## 6. O que já foi corrigido nos textos

- `main.tex` dizia que as fontes de chuva eram "CHIRPS e ERA5-Land" e a figura
  do pipeline dizia `chirps`. As duas coisas ficaram **factualmente erradas**
  depois do `chuva04`; foram corrigidas para fonte única ERA5-Land, mantendo a
  citação do CHIRPS como o produto que a auditoria retirou.
- A ressalva de chuva na Tabela I passou a declarar a incompatibilidade de
  escala, em vez de só dizer "recuperada para 100% da base".
- `ext_validacao_prospectiva_e_mecanismo_v1.md` §5 afirmava que "a fonte de
  chuva difere entre as regiões brasileiras" e que "isso não foi resolvido" —
  ficou obsoleto e recebeu nota.

---

## 7. O que este documento não faz

Não remove variável de nenhum script, não reajusta nenhum modelo e não altera a
tabela única. As três rotas da §5 são propostas. O `mod-pluv-01`, o `mod-mec-03`
e o `mod-prosp-02` continuam exatamente como estão, com as seis variáveis, até
que a decisão seja tomada.

E não afirma que a chuva seja irrelevante para enchente — afirma que, **na
escala em que este projeto compara pontos**, ela não discrimina, e que num único
caso onde parecia discriminar o que estava sendo medido era a data.
