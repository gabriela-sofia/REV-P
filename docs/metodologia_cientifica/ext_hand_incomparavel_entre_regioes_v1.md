# O `hand_m` das três regiões brasileiras não é a mesma variável

**Data**: 2026-08-12
**Artefatos**: `local_runs/ter-01-cadeia-harmonizada/{recife,curitiba,petropolis}_harmonizado/`
**Consequência**: afeta pooling entre regiões, comparação de coeficientes e a propagação para Petrópolis. **Não** afeta o v12 de Recife internamente.

---

## 1. O achado

Os manifestos das três derivações brasileiras registram todos
`stream_percentile = 98,0`. Convertendo o limiar para área contribuinte —
grandeza física — eles não são o mesmo critério:

| região | pixel | limiar (células) | **área contribuinte** | relativo a Recife |
|---|---|---|---|---|
| Recife | 10 m | 1.122,7 | **0,1123 km²** | 1,0× |
| Curitiba | 10 m | 847,4 | **0,0847 km²** | 0,75× |
| Petrópolis | 30 m | 826,8 | **0,7507 km²** | **6,7×** |

HAND é a altura acima do canal mais próximo. O limiar de acumulação **define
onde há canal**. Três limiares diferentes produzem três redes de drenagem
diferentes e, portanto, três variáveis diferentes com o mesmo nome de coluna.

A rede de Petrópolis é **6,7 vezes mais esparsa** que a de Recife: só vira canal
quem drena 0,75 km², contra 0,11 km². Menos canais, distâncias maiores até o
canal mais próximo, HAND sistematicamente maior.

Isto é anterior à frente externa. Está na cadeia brasileira do projeto desde a
sua construção.

---

## 2. O efeito, medido

As três regiões foram re-derivadas com a cadeia harmonizada (`ter01`): 30 m,
limiar fixo em 0,1123 km², WhiteboxTools v2.4.0. Comparação por amostragem
aleatória na área de sobreposição, com nodata mascarado.

| região | resolução | área do canal | HAND antigo | HAND novo | razão | pearson |
|---|---|---|---|---|---|---|
| **Recife** | 10 → 30 m | 0,1123 → **0,1123** | 7,13 m | 4,47 m | **1,60×** | 0,888 |
| Curitiba | 10 → 30 m | 0,0847 → 0,1123 | 8,81 m | 8,07 m | 1,09× | 0,848 |
| **Petrópolis** | **30 → 30 m** | 0,7507 → 0,1123 | 105,01 m | 65,53 m | **1,60×** | 0,726 |

O desenho isola os dois efeitos, porque em cada caso extremo só uma coisa muda:

- **Recife**: o limiar é idêntico nas duas derivações. A única diferença é a
  resolução. Logo **1,60× é o efeito puro da resolução** de 10 m para 30 m.
- **Petrópolis**: a resolução é 30 m nas duas. A única diferença é o limiar.
  Logo **1,60× é o efeito puro da definição de canal**.

Em Curitiba os dois efeitos atuam em sentidos opostos — a rede fica mais
esparsa (HAND sobe) e a resolução cai (HAND desce) — e quase se cancelam, o que
produz a razão de 1,09 e é a leitura mais enganosa das três.

### O que isso significa em números

O HAND de Petrópolis estava **60% acima** do que uma definição consistente com
Recife produz. Um modelo treinado em dados com rede densa e aplicado a
Petrópolis leria toda a cidade como sistematicamente mais alta acima da
drenagem do que ela é — e, como o coeficiente de HAND é negativo, prediria
**menos** suscetibilidade do que o correto, justamente na cidade em que o erro
é mais caro.

O resultado pareceria plausível. Nada no mapa denunciaria.

---

## 3. Por que ninguém tinha visto

Porque cada região é internamente consistente. Dentro de Recife, o HAND está
certo e o v12 é válido — nada aqui invalida `LOO-AUC = 0,678`. O problema só
aparece quando duas regiões são comparadas ou unidas, e até agora o projeto
tinha modelado uma região por vez.

E porque `stream_percentile = 98,0` aparece igual nos três manifestos. O campo
registrado dava a impressão de critério comum. O critério comum não existia: o
percentil é relativo às células da janela, então o mesmo "98" vira uma área
contribuinte diferente em cada região, conforme o tamanho e o relevo do
recorte.

---

## 4. O que fica válido e o que não

**Continua válido**

- O v12 de Recife e todas as métricas internas de Recife.
- O diagnóstico do colapso temporal de Curitiba: é interno a Curitiba, com
  limiar constante, então não é explicado por isto.
- A comparação serra × planície de `ext_modelo_de_encosta_v1.md`: os dois lados
  usam a mesma cadeia (o produto global), então são comparáveis entre si.

**Não é válido até refazer**

- Juntar Recife, Curitiba e Petrópolis num único modelo.
- Comparar coeficiente de HAND entre as três regiões.
- Aplicar a Petrópolis um modelo treinado nas outras — que era exatamente o
  plano de propagação.

---

## 5. O que foi feito

As três regiões estão re-derivadas com a cadeia harmonizada e manifesto
completo (WBT v2.4.0, CRS, shape, parâmetros, sha256 por saída):

```
local_runs/ter-01-cadeia-harmonizada/recife_harmonizado/
local_runs/ter-01-cadeia-harmonizada/curitiba_harmonizado/
local_runs/ter-01-cadeia-harmonizada/petropolis_harmonizado/
```

Pendente: re-extrair as features nos pontos das três regiões e refazer o v12 de
Recife sobre a versão harmonizada, para medir o que muda no modelo — não só no
raster.

---

## 6. Regra que decorre

> **Limiar de canal declarado em área contribuinte, nunca em percentil.**

Percentil é estatística da janela; área é grandeza física. Um manifesto que
registra percentil não permite saber se duas derivações usaram o mesmo
critério — e foi por isso que a incompatibilidade sobreviveu tanto tempo com o
campo aparentemente idêntico nos três arquivos.

A partir do `ter01`, o manifesto grava **os dois**: a área (critério) e o
percentil equivalente (diagnóstico de densidade de drenagem). Nas 21 AOIs
íngremes o percentil equivalente variou de 90,5 a 95,5 — contra 98 em Recife.
A densidade de drenagem realmente difere entre regiões, e é isso que o
percentil confunde com critério.

---

## 7. Nota sobre um bug encontrado no caminho

A primeira versão desta comparação devolveu correlações próximas de zero. A
causa era minha: o filtro descartava apenas valores não finitos, e os dois
nodata em jogo — **−9999** na cadeia antiga e **−32768** no WhiteboxTools — são
números finitos que passavam direto.

Verificado no dataset harmonizado: a contaminação foi **zero** em 4.154 pontos,
porque todos caem dentro da AOI e o `ter01` usa margem de 0,02°. Os resultados
do `mod-serra-02` não são afetados. A correção em `ter02` é para que continue
zero quando a AOI mudar.

Registro porque é a mesma família dos outros erros deste projeto: um valor
inválido que não parece inválido, passando por uma verificação que só checava
uma das formas de invalidez.
