# Grade de suscetibilidade por região — E5/M4

**Data**: 2026-08-20
**Artefatos**: `local_runs/svc-03-grade/`
**Script**: `scripts/servico/svc03_grade_suscetibilidade.py`
**Testes**: `tests/test_svc03_grade.py` (19)
**Depende de**: `ext_servico_contrato_inferencia_v1.md` (o mapa é o contrato)

---

## 1. O que faltava, e por que já não falta

O E5 pede mapa por região. Até aqui o projeto só tinha os pontos **rotulados**
de cada região — 278 em Recife, 1.680 em Curitiba, zero em Petrópolis. Mapa
exige varrer o território, não os pontos que alguém reportou.

A grade saiu do que já existia: a cadeia de terreno harmonizada cobre **as três
regiões** a 30 m, na mesma convenção (D-infinity, canal de 0,1123 km², WBT
2.4.0). Nenhuma aquisição nova.

| região | raster | células a 120 m | com terreno |
|---|---|---:|---:|
| Recife | 1.081 × 853 | 57.994 | 56.666 (97,7%) |
| Curitiba | 1.268 × 852 | 67.521 | 65.275 (96,7%) |
| Petrópolis | 1.676 × 1.688 | 176.818 | 172.015 (97,3%) |

A grade é **subamostragem** do pixel de 30 m, um a cada quatro. Não é
reamostragem nem suavização: cada célula é um valor derivado, não uma média
inventada.

---

## 2. A decisão que o E5 desempatou

Petrópolis estava em impasse entre dois documentos do projeto: o esboço de telas
declarava `region_not_supported`, e `ext_criterios_de_acerto_v1.md` §6 dizia que
"para PREDIZER em Petrópolis não falta nada; o que falta é a validação".

**O próprio E5 resolve.** Ele manda levar o modelo às três regiões, e a
evidência que exige não é acerto — é que não se afirme acerto onde falta
inventário. Servir com maturidade declarada atende as duas coisas; recusar
impediria o E5.

Petrópolis entra com maturidade **`transferencia_sem_referencia_local`**, um
nível novo, criado para não confundi-la com Curitiba: Curitiba tem inventário
local que o projeto decidiu não usar como critério de aprovação; Petrópolis não
tem nem isso. Chamar as duas de "transferência caracterizada" esconderia que
numa delas existe caminho para verificar depois, e na outra não existe nem isso.

---

## 3. Resultado por região

| região | modelo | maturidade | células servíveis | escore mediano |
|---|---|---|---:|---:|
| Recife | `recife_pluvial` (6 var) | `mvp_local` | 99,9% | 0,5853 * |
| Curitiba | `fluvial_planicie` (4 var) | `transferencia_caracterizada` | 92,2% | 0,2429 |
| Petrópolis | `fluvial_serra` (1 var) | `transferencia_sem_referencia_local` | 91,3% | 0,3460 |

\* no cenário de chuva mediana observada — ver §4.

**A conferência contra o contrato bateu 25/25 em todas as três.** Cada célula
sorteada passou por `inferir()` isoladamente e devolveu o mesmo escore da grade.
O mapa e o serviço são a mesma coisa, não duas implementações que se parecem.

---

## 4. A chuva entra como cenário, não como camada

Recife usa o modelo pluvial, que tem duas variáveis de chuva. Mas a chuva **não
é propriedade do lugar nesta escala**: Recife inteiro cabe em 4 células de 0,1°
do produto de precipitação (`ext_chuva_estado_do_projeto_v1.md`). Pôr chuva como
camada no mapa daria a impressão de que ela varia no espaço; ela não varia.

Então ela entra como **cenário declarado**, tirado da distribuição dos eventos
observados na própria Recife (n=145 positivos):

| cenário | `rain_max_24h` | escore mediano | escore p90 | células servíveis |
|---|---:|---:|---:|---:|
| mediana observada | 13,6 mm | 0,5853 | 0,7943 | 99,9% |
| p90 observado | 34,4 mm | 0,7174 | 0,8741 | 99,9% |
| **máximo observado** | **100,2 mm** | 0,8155 | 0,9251 | **79,9%** |

Isto é literalmente a definição de suscetibilidade do projeto — predisposição do
terreno **sob um dado forçamento**. E como a chuva é constante na grade, ela
desloca o escore inteiro e **não muda o ordenamento**: o mapa ordena terreno, e
o cenário diz em que nível ele opera.

**O cenário máximo derruba a cobertura para 79,9%, e isso é o portão
funcionando.** 100,2 mm está fora da faixa 5–95% que o modelo viu no ajuste;
onde a combinação com o terreno leva a célula para fora do domínio, o serviço
recusa em vez de extrapolar. Um cenário extremo cobre menos território — é a
resposta honesta, não uma falha.

---

## 5. Distância de domínio — o entregável explícito do E5

### Curitiba: a elevação está fora, e mais fora do que se sabia

| variável | dif. padronizada | dentro da faixa do ajuste |
|---|---:|---:|
| **`elevation_m`** | **+5,054** | **0,0%** |
| `slope_deg` | +0,450 | 92,3% |
| `hand_m` | +0,291 | 89,8% |
| `twi_dinf` | −0,528 | 85,5% |

O `metodo_aplicacao_sem_rotulo_local_v1.md` já tinha medido +2,76 desvios nos
pontos rotulados. **Sobre o território inteiro é +5,05, e nenhuma célula cai
dentro da faixa.** Curitiba fica a ~900 m de altitude e as fontes de evidência
real estão perto do nível do mar.

As três variáveis causais estão dentro (85–92%). Como só uma de quatro
extrapola, o portão libera a célula e escreve a extrapolação nas limitações —
que é o comportamento declarado, e a razão de `elevation_m` não ser grandeza
causal comparável entre cidades de altitude de base diferente.

### Petrópolis: o terreno cabe no domínio do modelo de serra

| variável | dif. padronizada | dentro da faixa do ajuste |
|---|---:|---:|
| `hand_m` | +0,445 | 91,3% |

Resultado não óbvio e que vale registrar: **91,3% do território de Petrópolis
cai dentro da faixa de HAND que o modelo de serra viu nas 22 AOIs europeias.**
A serra brasileira não é, nesta variável, um domínio estranho ao que foi
ajustado. Isso não valida nada — continua sem inventário local —, mas é a
diferença entre extrapolar e interpolar.

### Recife: dentro em tudo que importa

| variável | dif. padronizada | dentro da faixa |
|---|---:|---:|
| `elevation_m` | +0,685 | 67,3% |
| `slope_deg` | −0,088 | 72,4% |
| `hand_m` | −0,110 | 96,5% |
| `twi_dinf` | +0,274 | 85,1% |

Esperado: o modelo de Recife foi ajustado em Recife. A grade cobre território
que os 278 pontos rotulados não cobriam, e ainda assim 96,5% dela está dentro da
faixa de HAND dos pontos — os pontos rotulados representam bem o território
nessa variável.

---

## 6. Célula recusada fica vazia, e isso é decisão

O portão de domínio é aplicado **célula a célula**, com a mesma regra do
contrato. Célula recusada **não recebe escore** — fica vazia no mapa e é contada
à parte.

Zero é um escore: significa "muito pouco suscetível". Recusa não é isso — é
"não sei falar sobre esta célula". Confundir os dois pintaria de seguro
justamente o lugar sobre o qual o modelo não tem o que dizer. Um teste guarda
essa distinção, comparando as células vazias do CSV com as recusadas pelo
portão.

Sem isso o mapa mostraria número onde o serviço recusaria responder — e um mapa
que responde mais que o contrato é um mapa que mente.

---

## 7. O que estes mapas são, e o que não são

**São**: escore de predisposição do terreno a acumular água, por célula de
120 m, com IC95 por bootstrap de grupos, maturidade declarada por região e
distância de domínio medida variável a variável.

**Não são**:

- **previsão de evento.** Nenhum destes mapas diz que vai encher. Diz que o
  terreno predispõe, sob o forçamento declarado.
- **afirmação de acerto.** Nenhuma das três regiões tem inventário local usado
  como critério de aprovação. Recife tem rótulo, e o modelo dele não atinge o
  critério de leitura do projeto; Curitiba tem rótulo que o projeto decidiu não
  usar; Petrópolis não tem rótulo.
- **mapa operacional.** Não passou por revisão de Defesa Civil, não tem validação
  de campo e não substitui carta de risco.

---

## 8. O que ainda falta para o E5 virar validação

- **Recife**: negativo melhor. O modelo é `mvp_local` porque o negativo é por
  ausência de registro e o LOO-AUC fica abaixo da faixa.
- **Curitiba**: a elevação absoluta sai, ou o modelo servido para lá passa a ser
  o de duas variáveis causais. A decisão está encostada em `mod_mec03`, que
  publicou a comparação TERRENO-vs-COMPLETO com `elevation_m` dentro.
- **Petrópolis**: inventário. O pedido LAI para a Defesa Civil, modelado em
  `modelo_pedido_lai_defesa_civil_petropolis_2022_v1nj.md`, nunca foi usado — é
  o que transforma predição declarada em predição verificada.
