# Adjudicação do negativo — AOI Inglaterra (EXT-UK), v1

**Data**: 2026-08-07
**Status**: PROPOSTA METODOLÓGICA — nenhum ponto foi promovido a negativo
**Escopo**: define o critério; a execução é tarefa própria e posterior

---

## 1. O problema que este documento resolve

O REV-P nunca teve negativo formal. O gate `C4_BLOCKED_NO_FORMAL_NEGATIVES`
existe porque todo o Protocolo C constrói a classe negativa por **ausência de
registro** — "não há notícia de enchente aqui, logo aqui não enchente". Isso é
frágil por construção, e a fragilidade não é hipotética: a própria Environment
Agency escreve, na documentação do Recorded Flood Outlines, que

> "a ausência de cobertura por Recorded Flood Outlines numa área não significa
> que a área nunca inundou, apenas que não temos registro de inundação nela."

Qualquer negativo construído só por ausência herda essa contaminação.

---

## 2. A evidência disponível (EXT-UK-01 a EXT-UK-05)

Área de interesse: bloco BNG E 350–400 km / N 350–450 km, 5.000 km²,
noroeste da Inglaterra (eixo Manchester–Warrington–Wigan–Bolton–Blackburn).
Escolhida por evidência em EXT-UK-03: é a janela #1 tanto no ranking geral
quanto no ranking pluvial isolado.

Camadas adquiridas, ambas Environment Agency, ambas OGL v3:

| Camada | O que é | Volume na AOI |
|---|---|---|
| Recorded Flood Outlines | inundação **ocorrida**, com data e mecanismo | 1.550 outlines |
| Flood Map for Planning (FZ2/FZ3) | inundação **modelada** | 17.199 polígonos |

Contabilidade sobre grade regular de 200 m (125.000 pontos):

| Classe | Pontos | % da AOI | Área |
|---|---|---|---|
| `POS_CAND` — dentro de outline registrado | 649 | 0,5% | ~26 km² |
| `EXCLUIDO` — fora de outline, dentro de Flood Zone | 6.411 | 5,1% | ~256 km² |
| `NEG_CAND` — fora de ambos | 117.940 | 94,4% | ~4.718 km² |

Por mecanismo: `POS_CAND` fluvial = 559, pluvial = 56.

Distância de cada `NEG_CAND` até a área inundável mais próxima:

| Faixa | Pontos | % |
|---|---|---|
| < 200 m | 0 | 0,0% |
| < 400 m | 21.148 | 17,9% |
| < 600 m | 43.515 | 36,9% |
| < 1.000 m | 73.387 | 62,2% |
| < 2.000 m | 111.564 | 94,6% |

Mediana 721 m, p90 1.697 m, **máximo 4.669 m**.

---

## 3. O que esses números dizem

**3.1. O filtro de Flood Zone é barato e vale a pena.** Ele remove apenas 5,1%
da AOI, mas remove justamente a faixa fisicamente plausível de inundar sem
registro — que é a fonte da contaminação. Custo baixo, ganho conceitual alto.

**3.2. Não existe "longe" nesta AOI.** O ponto mais distante de qualquer área
inundável está a 4,7 km. Isso é bom: significa que negativo e positivo
compartilham a mesma região, o mesmo clima e a mesma bacia — o modelo não pode
separá-los por contexto macro. Em Curitiba e Recife essa garantia nunca existiu.

**3.3. Mas o buffer não pode ser grande.** Exigir 2 km de afastamento deixa
apenas 5,4% dos candidatos, e esses 5,4% são sistematicamente o terreno mais
alto e mais seco. O modelo aprenderia "alto e seco versus baixo e úmido" — que
é tautológico, porque `elevation` e `HAND` já são features. Isso infla AUC sem
significar nada.

**3.4. O desequilíbrio é real mas administrável.** 649 positivos contra 117.940
candidatos a negativo, na grade de 200 m. A grade é escolha de amostragem, não
limite do dado: numa grade de 30 m, compatível com as features, os positivos
seriam da ordem de 28 mil. O `n` não é o gargalo aqui — o gargalo é o critério.

**3.5. O pluvial é escasso.** Só 56 pontos `POS_CAND` pluviais a 200 m. É o
recorte mais parecido com Recife e o mais frágil em volume. Precisa de grade
mais fina e provavelmente de AOI ampliada para a segunda janela vizinha.

---

## 4. Critério proposto

Um ponto é **candidato a negativo** se, e somente se, satisfizer todas:

- **N1.** Está fora de todo Recorded Flood Outline, sem restrição de data
  (não só dos elegíveis — se inundou em 1953, não serve como negativo).
- **N2.** Está fora de toda Flood Zone modelada, FZ2 e FZ3.
- **N3.** Está a **pelo menos 400 m** de qualquer área das condições N1/N2.
  Justificativa do valor: 400 m é o primeiro limiar com massa não trivial
  (17,9% dos candidatos estão abaixo dele) e corresponde a duas células da
  grade de 200 m, o que absorve imprecisão de digitalização dos outlines.
  Abaixo disso o risco de contaminação por borda é alto; acima de 1 km o
  candidato vira sistematicamente terreno alto.
- **N4.** Está **pareado por contexto construído** com os positivos.
  Esta é a condição que ainda não pode ser executada — falta a camada de
  área construída (GHSL built-up ou ESA WorldCover). Sem ela, a amostra
  negativa seria dominada por campo aberto e o classificador aprenderia
  "urbano versus rural", não suscetibilidade.

**Nenhuma dessas condições transforma um ponto em negativo por si só.** O
conjunto delas define um *candidato*; a promoção exige a execução documentada
e o registro do número final por estrato.

---

## 4-bis. Unidade de análise: o evento, não o ponto

**Esta regra é fixada aqui, antes de qualquer modelo rodar, e não pode ser
revista depois de ver resultado.** Fixá-la a posteriori seria decisão pós-hoc.

### A regra

A unidade de informação independente é o **evento** (`rec_grp_id`), não o
ponto. Disso decorrem três obrigações:

- **U1.** Toda validação cruzada é **agrupada por `rec_grp_id`**. Nenhum ponto
  de um evento pode aparecer em treino e teste simultaneamente.
- **U2.** Todo intervalo de confiança e todo erro-padrão é calculado com
  reamostragem **no nível do evento** (bootstrap de eventos, não de pontos).
- **U3.** Todo relato de tamanho amostral declara os dois números: número de
  pontos e número de eventos independentes. Relatar só o `n` de pontos é
  considerado relato incompleto.

### Por que

Os 1.550 outlines da AOI vêm de 262 eventos. Numa grade de 30 m, esses mesmos
262 eventos gerariam algo da ordem de 28 mil pontos positivos. O `n` cresce
cem vezes; a informação independente não cresce nada.

O efeito da pseudorreplicação é traiçoeiro porque **se parece com sucesso**:
p-valores despencam, intervalos de confiança encolhem, AUC sobe. Nada disso é
real — é o mesmo evento contado muitas vezes. Com pontos espacialmente
autocorrelacionados, o erro-padrão do Firth fica sistematicamente otimista.

Isso não é hipótese neste projeto. O teste A/B do DINO
(`revp_fase1_conclusao_dino_ab_test.md`) já foi comprometido exatamente por
pseudorreplicação, e a sensibilidade com erro robusto por cluster
(`revp_v1r6_dino_v12_cluster_robust_sensitivity.py`) existe por causa disso.
Repetir o erro numa base maior o tornaria mais difícil de detectar, não menos.

### Consequência prática para o dimensionamento

"Mais dados" para esta frente **não** significa mais pontos por evento.
Significa mais eventos independentes. É por isso que a AOI foi ranqueada por
`rec_grp_id` em EXT-UK-03 e não por contagem de polígonos — e foi essa decisão
que evitou escolher a janela do Tâmisa, que tem 1.394 outlines mas apenas 63
eventos.

O ativo real desta AOI são os **262 eventos em ~158 datas distintas ao longo
de 25 anos**. É esse número que precisa aparecer no artigo, ao lado do número
de pontos.

---

## 5. Por que não usar amostragem aleatória simples

A literatura recente é explícita sobre o viés: amostragem aleatória global não
considera heterogeneidade espacial e produz amostra negativa que carrega mais
informação de inundação do que a realidade. Estratégias alternativas
documentadas incluem o método de buffer (reduz a taxa de erro dos negativos
selecionados) e *inverse-occurrence sampling*, que seleciona menos pontos
negativos em áreas com mais registros. O critério N3 acima é a aplicação do
método de buffer; N4 é o pareamento que evita o atalho urbano/rural.

Também está registrado que forçar proporção 1:1 entre positivos e negativos,
só para evitar desequilíbrio, pode levar a sobreajuste. A proporção deve ser
decidida com o orçamento de EPV do Firth, não por estética.

---

## 6. O que falta para executar

1. **Camada de área construída** na AOI (GHSL built-up 10 m ou ESA WorldCover).
   É download — não está adquirida. Sem ela, N4 não roda.
2. **Definição da grade final** (200 m é grade de contabilidade; a grade de
   modelagem deve acompanhar a resolução das features).
3. **Decisão sobre o recorte pluvial**: 56 pontos a 200 m é pouco; avaliar
   ampliar a AOI para a janela vizinha antes de fixar.

---

## 7. Relação com o Nível 1 (negativo por observação)

O critério acima é **negativo por exclusão qualificada** — melhor que ausência
pura, mas ainda não é observação direta. O Nível 1 (Sen1Floods11, UFO,
Copernicus EMS, GFD) fornece a categoria mais forte: área que foi olhada e
registrada como não inundada.

Os dois não competem, se validam. Quando as fontes do Nível 1 estiverem
adquiridas, o teste natural é verificar a taxa de concordância entre elas nas
áreas de sobreposição. Discordância alta seria evidência de que o critério de
exclusão está contaminado e precisa endurecer.

---

## 8. Declaração de não promoção

Nada neste documento promove qualquer ponto a rótulo. Nenhuma feature foi
extraída, nenhum modelo foi treinado, nenhum gate foi alterado. O
`C4_BLOCKED_NO_FORMAL_NEGATIVES` permanece como está.

---

## Fontes

- Environment Agency, *Recorded Flood Outlines* — https://environment.data.gov.uk/dataset/8c75e700-d465-11e4-8b5b-f0def148f590
- Environment Agency, *Flood Map for Planning — Flood Zones* — https://environment.data.gov.uk/dataset/04532375-a198-476e-985e-0579a0a11b47
- *Data Uncertainty of Flood Susceptibility Using Non-Flood Samples*, Remote Sensing, 2025 — https://www.mdpi.com/2072-4292/17/3/375
- *Contrast or Diversity: Non-Flood sampling in urban flood susceptibility modelling*, Journal of Hydrology, 2025 — https://www.sciencedirect.com/science/article/abs/pii/S0022169425003919
- Artefatos internos: `local_runs/ext-uk-0{1..5}-*`, `scripts/externo/ext_uk0{1..5}_*.py`
