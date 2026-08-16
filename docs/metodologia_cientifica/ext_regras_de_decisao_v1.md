# Regras de decisão metodológica

**Data**: 2026-08-16
**Companheiro de**: `ext_criterios_de_acerto_v1.md` §4 (faixas fixadas antes de rodar)

As regras reunidas aqui têm uma coisa em comum: todas nasceram de um erro real
cometido neste projeto, e todas dizem **qual métrica não pode decidir uma
escolha**. O `ext_criterios_de_acerto_v1.md` fixa as faixas de aceitação; este
documento guarda as regras sobre o *processo* de decidir.

---

## Regra 0 — a que já existia: critério não se ajusta por resultado

Fixada em `ext_criterios_de_acerto_v1.md` §4 e repetida em cada script de
modelo: as faixas de aceitação são declaradas **antes** de rodar, e a
classificação de mecanismo é declarada **com a evidência que a sustenta, não
inferida de desempenho**. Classificar por resultado seria circular.

As três regras abaixo são a mesma ideia aplicada a decisões que não são de
modelo.

---

## Regra 1 — folga de derivação de terreno decide-se por métrica física, nunca por contagem de cobertura

### O erro que a originou

Em 2026-08-16, 2.002 pontos do UFO em 51 chips estavam sem `hand_m`. A hipótese
foi que a folga do recorte (0,02 grau) era pequena demais: o ponto cai perto da
borda, o roteamento de fluxo não alcança canal nenhum, e o HAND sai nodata
mesmo com elevação e declividade válidas.

A folga foi aumentada para 0,08 grau. **A cobertura piorou** — 68.432 para
68.364 pontos com as quatro variáveis. A janela maior muda a rede de drenagem,
o limiar de área contribuinte passa a criar canais em outros lugares, e o
recorte alcança mais máscara de mar. Refeito na folga padrão, ficou em 68.799.

### Por que a contagem de cobertura é o critério errado

Cobertura é um número que sobe quando o HAND deixa de ser nodata, e ele deixa
de ser nodata tanto quando o canal certo entra na janela quanto quando um canal
*errado* entra. Os dois casos elevam a cobertura de forma idêntica; só o
primeiro melhora o dado. Otimizar cobertura é, portanto, otimizar uma métrica
que não distingue acerto de artefato — o análogo exato de escolher hiperparâmetro
por AUC de treino.

O caso é agravante porque HAND é *definido* como altura acima do canal mais
próximo: um canal espúrio dentro da janela não produz erro de medida, produz uma
variável diferente com o mesmo nome. É o mesmo mecanismo que
`ext_hand_incomparavel_entre_regioes_v1.md` documentou quando o limiar era
declarado em percentil.

### A regra

> A folga de recorte na derivação de terreno é aceita ou rejeitada por
> **evidência de que a rede de drenagem da janela contém o canal fisicamente
> relevante para os pontos amostrados**. Contagem de cobertura pode ser
> reportada; não pode decidir.

Critérios de aceite admissíveis, em ordem de força:

1. o canal encontrado coincide com drenagem conhecida de referência na AOI
2. a rede tem densidade compatível com a região (o `ter01` já grava
   `n_celulas_canal` e o percentil equivalente para isso)
3. o HAND concorda com produto independente — no Nível 1 o pearson contra o
   produto global é 0,946, e foi ele que sustentou que a janela de ~9 km não
   comprometeu a variável

A guarda de rede degenerada (menos de 50 células de canal ⇒ HAND anulado) é a
forma mínima desta regra já implementada: prefere ausência declarada a número
plausível.

### O que fica em aberto, declarado

Os pontos restantes sem HAND estão majoritariamente em célula costeira anulada
pela máscara de mar (células ≤ 0 m viram nodata). Isso é limitação declarada
desde a máscara existir, e **não** foi resolvido — resolver exige distinguir mar
de terra abaixo do nível do mar, que é outro problema.

---

## Regra 2 — ausência de registro não é negativo

### O que ela diz

> "Não há registro de enchente aqui" é afirmação sobre o **registro**, não sobre
> o lugar. Pode não ter inundado; pode ter inundado sem ninguém anotar. Tratar
> como classe 0 ensina ao modelo que o lugar é seguro por uma razão que é do
> arquivo, e não do terreno.

Ausência é **lacuna de dado**. Na terminologia de PU-learning é *unlabeled*, não
*negative*. A hierarquia do `ds01` já dizia "ausência não entra"; o que muda é
que ela deixa de ser tratada como um negativo fraco e passa a não ser negativo.

### A medida que a sustenta

O `aud_provenance01`, rodado sobre a tabela única, compara os três níveis entre
si. Os estratos não são a mesma população com confiança diferente:

| par | variável que mais separa | AUC | separação |
|---|---|---|---|
| ausência × observado | `elevation_m` | 0,1913 [0,166; 0,217] | 0,617 |
| ausência × observado | `rain_decay_index` | 0,7962 [0,782; 0,811] | 0,592 |
| exclusão qualificada × observado | `rain_max_24h` | 0,8120 [0,807; 0,817] | 0,624 |
| ausência × exclusão qualificada | `elevation_m` | 0,2040 [0,173; 0,238] | 0,592 |

E o agravante: **cada nível vive numa região diferente, sem sobreposição.**
`observado` só existe em CEMS, Sen1Floods11 e UFO; `exclusao_qualificada` só no
piloto inglês; `ausencia` só em Curitiba e Recife. Nesses dados, nível de
negativo e região são a mesma variável — não há como separar o efeito de um do
outro.

### Consequência que precisa ficar dita

Aplicada a regra, **Recife e Curitiba ficam com zero negativos utilizáveis**. O
modelo pluvial de Recife é positivo contra não-rotulado, e seu LOO-AUC de 0,6339
não é discriminação contra negativo observado. Isso não invalida o número; muda
o que ele significa, e o registro de regiões precisa refletir isso.

O caminho que a literatura oferece para esse caso é avaliação PU — positivo
contra fundo da própria AOI — em vez de AUC com negativo. Não está implementado.

---

## Regra 3 — estimador de decisão vem com intervalo

### O erro que ela evita

A matriz de prontidão do `mvp01` decide por limiar: transferência ≥ 0,60 aprova
a região. Com valores pontuais, 0,7382 e 0,6100 parecem igualmente sólidos, e
não são — o segundo pode ter intervalo cruzando o limiar. Sem intervalo, o
limiar decide sozinho e não informa com quanta confiança.

### A regra

> Todo número que entra numa condição de aceite carrega IC95 por bootstrap, e a
> reamostragem é de **grupo**, não de linha.

A reamostragem por grupo segue Ploton et al. (2020, *Nature Communications*
11:4540): com dado espacialmente autocorrelacionado, tratar pontos vizinhos como
observações independentes estreita o intervalo artificialmente. É o mesmo
mecanismo que infla validação cruzada aleatória — e este projeto já usa CV
agrupada exatamente por isso, então usar bootstrap por linha seria inconsistente
com a própria validação.

### Estado atual

| estimador | onde | IC por |
|---|---|---|
| coeficiente | `mod_mec02`, `mod_mec03` | bootstrap de grupo |
| `loso_auc` | `mod_mec03` | bootstrap de grupo do conjunto de teste, modelo fixo |
| `melhor_separacao` | `mvp01` | bootstrap de grupo |
| AUC por estrato | `aud_provenance01` | bootstrap de linha (sem grupo definido no par) |

No LOSO o modelo fica **fixo** durante o bootstrap: o que se mede é a
variabilidade do estimador sobre a amostra de teste, não a instabilidade do
ajuste — essa já está no IC dos coeficientes.

Com os intervalos, as três candidatas a MVP deixam de ser indistinguíveis:

| região | transferência | IC95 | posição frente ao limiar |
|---|---|---|---|
| UK | 0,7336 | [0,7066; 0,7605] | intervalo inteiro acima |
| CEMS | 0,7156 | [0,6943; 0,7363] | intervalo inteiro acima |
| Sen1Floods11 | 0,7048 | [0,6703; 0,7347] | intervalo inteiro acima |

Nenhuma é marginal — o que só se pode afirmar porque o intervalo existe.

---

## Referência

Ploton, P., Mortier, F., Réjou-Méchain, M. et al. Spatial validation reveals
poor predictive performance of large-scale ecological mapping models. *Nature
Communications* 11, 4540 (2020).
