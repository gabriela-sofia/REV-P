# O modelo de encosta — HAND e TWI em terreno íngreme

**Data**: 2026-08-11
**Artefatos**: `local_runs/mod-serra-01/`, `scripts/suscetibilidade/mod_serra01_ingreme_2features.py`
**Depende de**: `ext_analogos_de_petropolis_v1.md` (correção do matching), `ext_criterios_de_acerto_v1.md` (critérios fixados antes)

---

> **Nota de 20/08/2026 — substituido pela v2.** Este documento vale como
> registro do MOD-SERRA-01, que rodou na `ds-01` (base anterior a
> harmonizacao) e contou EPV por grupos totais. Na tabela unica o estrato
> ingreme tem 19 grupos positivos, e pela regra de EPV da classe minoritaria
> comporta uma variavel, nao duas. A conclusao qualitativa se mantem; os
> coeficientes mudaram. Ver `ext_modelo_de_encosta_v2.md`.

## 1. A pergunta

Propagar o modelo para Petrópolis só faz sentido se a relação HAND → inundação
se comportar em encosta como se comporta em planície. Se a física for a mesma,
treinar onde há dado e aplicar na serra é legítimo. Se não for, a serra precisa
de modelo próprio — e aí não há dado suficiente para construí-lo.

Com as sete ativações novas há, pela primeira vez, os dois lados para comparar.

## 2. Desenho

| | n | positivos | negativos | AOIs (grupos) | EPV |
|---|---|---|---|---|---|
| **Serra** | 4.475 | 1.835 | 2.640 | 22 | **11,0** |
| **Planície** | 20.774 | 9.134 | 11.640 | 97 | 48,5 |

Serra = AOI com relevo local ≥ 400 m e ≥ 25% da área acima de 15°, critério
declarado em `cems03` antes de qualquer leitura.

**Duas features: `hand_m` e `twi_dinf`.** A escolha não é por desempenho. Com
22 grupos, quatro features dariam EPV 5,5 — abaixo do mínimo de 10 do projeto,
que proíbe interpretar modelo nessa condição. E as duas descartadas são
justamente as que já haviam sido medidas invertendo de sinal entre fontes
(`ext_o_que_nao_e_enchente_v2.md`, seção 5): `elevation_m` e `slope_deg`
carregam região, não processo. `hand_m` e `twi_dinf` são os dois termos causais
do balanço — quanto precisa subir para alcançar, e para onde a água converge.

Validação: `GroupKFold` por AOI. IC por bootstrap **reamostrando grupos**, não
linhas (regra U2 — reamostrar linhas infla a precisão por pseudo-replicação).

---

## 3. Achado 1 — a mesma física, em escala dez vezes maior

Contraste entre a mediana do negativo e a do positivo, dentro de cada subconjunto:

| | `hand_m` | `twi_dinf` |
|---|---|---|
| **Serra** | positivo 3,58 m / negativo **31,44 m** → **+27,86 m** | pos 7,75 / neg 6,35 |
| **Planície** | positivo 0,24 m / negativo 3,18 m → **+2,95 m** | pos 9,32 / neg 8,00 |

O contraste de HAND na serra é **quase dez vezes** o da planície, e isso é
exatamente o que a hidrologia prevê: numa encosta o terreno sobe rápido, então
um ponto não inundado está dezenas de metros acima da drenagem. Num fundo de
vale plano, três metros já bastam para não inundar.

O fenômeno é o mesmo; a escala em metros não é. Isso tem consequência direta
para leitura de resultado: **um contraste de HAND de +28 m não é sinal de
negativo fabricado quando o terreno é de serra** — a régua de 2–3 m fixada em
`ext_criterios_de_acerto_v1.md` (seção 4.3) vale para planície e precisa ser
qualificada por classe de relevo.

---

## 4. Achado 2 — o peso troca entre as duas variáveis

Coeficientes sobre features padronizadas, com IC95 por bootstrap de grupos:

| feature | **serra** | **planície** |
|---|---|---|
| `hand_m` | −0,8173 [−1,8094; −0,4204] | **−3,2979** [−4,6454; −2,3198] |
| `twi_dinf` | **+0,5044** [+0,2899; +0,6840] | +0,1782 [+0,0805; +0,2589] |

Nenhum IC cruza zero. Os dois sinais obrigatórios estão corretos nos dois
subconjuntos.

Mas o peso relativo se inverte. Na planície o modelo é **dominado por HAND**
(4× o coeficiente da serra). Na serra, **TWI pesa quase três vezes mais** que
na planície e a decisão é repartida entre as duas.

Leitura física: numa planície de inundação praticamente tudo converge, então
TWI carrega pouca informação e o que decide é a altura acima da drenagem. Numa
serra, a convergência é o que separa o talvegue que enche do que não enche —
qual vale recebe a água importa tanto quanto quão fundo ele é.

Isso não é ruído amostral: os IC das duas features não se sobrepõem entre
subconjuntos em nenhum dos dois casos.

---

## 5. Achado 3 — a transferência funciona, e numa direção só

| treinado em | avaliado em | AUC | AUC nativo do alvo | variação |
|---|---|---|---|---|
| **planície** | **serra** | **0,7748** | 0,7622 | **+0,0126** |
| serra | planície | 0,7018 | 0,7478 | −0,0460 |

**O modelo treinado em planície aplicado à serra atinge 0,7748 — acima do que
o próprio modelo da serra consegue em validação cruzada.** Ele nunca viu
terreno íngreme e mesmo assim funciona lá.

A explicação honesta não é que planície ensine melhor: é que planície tem
20.774 pontos e 97 grupos contra 4.475 e 22. O modelo está melhor estimado. E
o fato de ele transferir sem perda para um domínio que não viu é a evidência
de que **a relação HAND/TWI → inundação é a mesma nos dois terrenos.**

O caminho inverso perde 0,046, o que é coerente com a mesma explicação: o
modelo da serra é o menos bem estimado dos dois.

---

## 6. Veredito contra os critérios fixados antes

Critérios de `ext_criterios_de_acerto_v1.md`, declarados antes desta rodada:

| critério | serra | passa |
|---|---|---|
| AUC com CV agrupada em 0,70–0,88 | 0,7622 | ✅ |
| AUC abaixo de 0,95 (vazamento) | 0,7622 | ✅ |
| `hand_m` negativo | −0,8173 | ✅ |
| `twi_dinf` positivo | +0,5044 | ✅ |
| IC95 sem cruzar zero | ambos | ✅ |
| gap treino-validação < 0,15 | −0,0028 | ✅ |
| nenhum fold ≥ 0,999 | — | ✅ |
| EPV ≥ 10 | 11,0 | ✅ |

```
VEREDITO = COERENTE_COM_CRITERIOS
```

---

## 7. O que isto autoriza a escrever — e o que não

**Autoriza:**

- "A relação entre altura acima da drenagem, convergência topográfica e
  ocorrência de inundação foi verificada em 22 áreas de interesse de terreno
  íngreme e em 97 de planície, com sinais físicos concordantes e intervalos de
  confiança que excluem zero em ambos."
- "Um modelo ajustado em terreno de planície transfere para terreno íngreme sem
  degradação de desempenho (AUC 0,7748 contra 0,7622 do modelo nativo),
  sustentando a aplicação a regiões de serra não representadas no treino."

**Não autoriza:**

- Nenhuma afirmação de acerto em Petrópolis. Transferência medida entre os
  subconjuntos de treino não é validação em Petrópolis, que continua sem
  inventário local.
- Nenhuma afirmação sobre movimento de massa. Os 1.592 pontos de classe 2 estão
  no dataset e 87% deles caem em AOI íngreme, mas **não entraram neste modelo** —
  aqui só há classe 0 e 1.
- Nenhuma afirmação envolvendo chuva. As features pluviométricas não existem em
  nenhuma ativação CEMS; este modelo é puramente topográfico.

---

## 8. Limitações declaradas

**Duas features, não quatro.** Imposto pelo EPV. Quando houver ~40 AOIs
íngremes, refazer com quatro e verificar se `elevation_m` e `slope_deg`
continuam invertendo.

**A avaliação cruzada não é validação cruzada.** `planície→serra` é ajuste em
todo um subconjunto e avaliação em todo o outro. É legítimo como teste fora de
domínio — o modelo nunca viu aquele dado — mas não tem a estrutura de folds do
número nativo, e os dois não são estritamente comparáveis.

**Serra e planície não são regiões pareadas.** Vêm de ativações diferentes, em
países diferentes. A diferença de coeficiente da seção 4 é entre terrenos, mas
carrega também diferença de geografia. O teste da seção 5 é o menos vulnerável
a essa confusão, por ser transferência.

**Sem controle temporal.** Nenhum dos dois modelos foi submetido a holdout por
data. Esse continua sendo o teste mais forte não realizado do projeto.
