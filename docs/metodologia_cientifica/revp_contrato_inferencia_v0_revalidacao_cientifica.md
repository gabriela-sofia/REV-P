# Revalidação científica: `txtpragab.docx` (contrato de inferência REV-P, rascunho v0)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO — documento de estudo/resposta, não altera nenhum script, dado ou gate existente.
**Data**: 2026-07-23

## 1. O que o documento propõe (síntese neutra)

Quem escreveu propõe separar o projeto em camadas — dados físico-hidrológicos (causal) →
eventos reais (validação) → DINO/Sentinel/SAR (evidência auxiliar) → modelo estatístico →
API → interface → LLM (só explica, nunca decide) — e formaliza isso num "contrato de
inferência": um schema de entrada/saída com **gates obrigatórios** (geometria válida, CRS,
DEM, declividade, HAND, TWI, chuva, modelo validado pra região) que, se não atendidos,
retornam `insufficient_data` em vez de inventar um score. Propõe MVP só em Recife (único
lugar com modelo estatístico real hoje), Curitiba como "evidência em processamento" e
Petrópolis como "dados insuficientes" (por causa da mistura enchente/deslizamento já
conhecida). E propõe testar DINO como feature incremental (Modelo A = físico, Modelo B =
físico + DINO) em vez de treinar um classificador à parte, para não violar a regra de
não ter treino supervisionado no REV-P.

Isso está alinhado com a arquitetura conceitual que já está em vigor no projeto (DINO
auxiliar, física causal, nada de score/threshold como feature) e não contradiz nenhuma
regra fixa do `CLAUDE.md`. É, na prática, um plano de produto que **usa** o motor
científico já existente, não uma mudança nele.

## 2. Cruzamento com o que já rodamos de verdade

O documento propõe "Modelo A (físico) vs Modelo B (físico + DINO)" como o teste que
decide se DINO entra no produto. Isso **já foi parcialmente respondido** nesta sessão:
`revp_v1r4_dino_sedec_extended_analysis.py` (n=78, 56 pos / 22 neg) rodou um modelo
Firth **só com DINO** (2 componentes PCA) e obteve LOO AUC = 0.490 (nível de chance),
enquanto o screen univariado mostrou `elevation_m` (p=0.026) e `rain_max_24h_chirps`
(p=0.010) com sinal real. Isso é consistente com a intuição do documento — mas **não é
ainda o teste A/B que ele pede**. O que rodamos foi um "Modelo C" (DINO sozinho), não a
comparação físico vs. físico+DINO no mesmo modelo. Isso importa porque revela um gap real
que o rascunho não menciona (seção 4 abaixo).

## 3. Revalidação item a item com literatura real

### 3.1 Gates de DEM/declividade/HAND/TWI — bem fundamentado

HAND (Height Above the Nearest Drainage) foi formalizado por Rennó et al. (2008) como
atributo de terreno derivado de DEM que normaliza a topografia em relação à rede de
drenagem, e é hoje um dos preditores de suscetibilidade a enchente mais usados em regiões
com poucos dados hidrodinâmicos — inclusive com extensões específicas para o Brasil
(Lucas do Rio Verde, MT). TWI foi formalizado por Beven & Kirkby (1979) no TOPMODEL e
segue em uso corrente para identificar zonas de acúmulo de água a partir de área de
drenagem e declividade. Exigir DEM + declividade + HAND + TWI como pré-condição para
qualquer inferência é, portanto, uma decisão bem ancorada na literatura hidrológica — não
é uma escolha arbitrária do documento.

### 3.2 Comparação Modelo A vs Modelo B (DINO incremental) — correta em espírito, com uma ressalva estatística real

A ideia geral (comparar modelo com e sem a feature nova) é o padrão certo — é
exatamente o que a literatura de "incremental value of new biomarkers" recomenda em vez
de simplesmente olhar se o coeficiente é significativo. Mas há uma armadilha documentada:
comparar AUC diretamente entre modelos aninhados via teste de DeLong é conhecido como uso
incorreto (Vickers & Cronin apontam que a hipótese nula "sem ganho incremental" não
corresponde à hipótese nula "AUCs iguais", e o teste de DeLong é excessivamente
conservador nesse cenário — menos poderoso que teste de razão de verossimilhança ou Wald).
**Recomendação**: se/quando essa comparação A vs B for rodada de verdade, usar razão de
verossimilhança entre os dois modelos Firth (não comparar ΔAUC bruto como critério de
decisão), com o ΔLOO-AUC reportado apenas como evidência descritiva complementar — igual
já é feito em `pipeline_final_v5.py`.

### 3.3 SAR — achado negativo é coerente com a literatura, mas por um motivo que vale registrar

A literatura de SAR para enchente se divide claramente em duas famílias: (a) mapeamento
de extensão via *change detection* multitemporal (backscatter baixo em água, comparação
antes/depois do evento) e (b) uso como camada estática de suscetibilidade. A maioria da
literatura recente trata SAR como ferramenta de (a) — evidência de evento observado, não
como atributo de terreno estável. Isso explica tecnicamente por que testar um valor
único de SAR por patch como feature de um modelo de suscetibilidade pode dar sinal na
direção errada: o valor depende das condições de umidade/chuva no momento exato da
captura, não é uma propriedade fixa do terreno como HAND/TWI. **Achado**: se o SAR
voltar a fazer sentido em Curitiba (como o documento sugere), a forma tecnicamente mais
sólida — e mais alinhada à literatura — é reintroduzi-lo como evidência de evento
(diferença pré/pós-evento pareada a um evento real conhecido), não como feature estática
recorrente na mesma regressão de suscetibilidade onde já foi descartado em Recife.

### 3.4 Contrato de inferência (gates + `insufficient_data`) — padrão de governança reconhecido

O padrão "declarar as pré-condições, recusar inferência se faltarem, documentar
limitações e versão do modelo" é essencialmente o que Mitchell et al. (2019), *Model
Cards for Model Reporting*, formalizaram como prática de documentação responsável de ML
(hoje padrão de fato — inclusive adotado em massa no Hugging Face). E o mecanismo de
"recusar responder quando a confiança/cobertura de dados é insuficiente" é a literatura
de *selective prediction* / *reject option classifiers*: a ideia central é o trade-off
explícito entre cobertura (quantas regiões o sistema aceita avaliar) e risco (erro nas
que aceita) — exatamente o papel do campo `region_maturity` proposto. O rascunho está
bem alinhado com as duas linhas de literatura mais relevantes pra esse tipo de
governança de produto.

### 3.5 Um ponto que o rascunho não resolve: semântica do `confidence_interval`

O contrato propõe `score.confidence_interval` por requisição. Mas o que
`pipeline_final_v5.py` calcula hoje via bootstrap são **intervalos de confiança dos
coeficientes** do modelo Firth (incerteza sobre o efeito de cada variável), não um
intervalo preditivo por região nova. Transformar isso num CI por score individual exige
uma decisão de design ainda não tomada: bootstrap do preditor linear (reamostrar e
reajustar o modelo N vezes, projetar cada score, tomar percentis) ou erro-padrão
assintótico de Firth propagado pela função logito (delta method). São escolhas
metodologicamente diferentes (a primeira mais custosa mas mais fiel ao n pequeno; a
segunda mais barata mas depende de aproximação assintótica que é discutivelmente frágil
exatamente no regime de n pequeno que motivou usar Firth em primeiro lugar). Isso não
invalida o contrato — só significa que o campo `confidence_interval` no schema ainda não
tem uma implementação definida por trás, e essa decisão precisa ser tomada antes de
qualquer código de API, não depois.

## 4. O gap real que o documento assume resolvido, mas não é: orçamento de EPV para o Modelo A/B

O rascunho diz que a comparação A/B "roda dentro do REV-P usando o embedding DINO como
uma feature a mais na regressão de Firth que já existe". Isso está certo em espírito, mas
esbarra num limite que o próprio projeto já usa como critério de responsabilidade
estatística (heurística EPV ≥ 10, já aplicada em `pipeline_final_v5.py` e em
`revp_v1r4`): hoje, no subconjunto com DINO + física (n=78, 56 positivos), já usamos 6
variáveis físicas (EPV≈13, dentro do limite) OU 2 componentes DINO isoladamente
(EPV≈11.7, dentro do limite) — mas um Modelo B combinando as 6 físicas + 2 DINO dá 8
preditores, EPV≈9.75, **abaixo** do próprio limiar que o projeto adota. Ou seja: o teste
A vs B que o documento propõe como decisivo para "DINO entra no produto ou não" não pode
ser rodado de forma estatisticamente responsável com o n atual sem violar uma regra que
o próprio projeto já se impôs. As saídas possíveis, sem inventar dado: (a) reduzir a
física a 1–2 variáveis já sabidamente mais fortes (elevação, chuva) antes de somar DINO,
mantendo EPV≥10; ou (b) esperar o n crescer (mais evidência real SEDEC/ANA/Global Flood
Database) antes de rodar o modelo combinado. Isso é uma decisão de dado, não de código —
vale levar de volta pra quem escreveu o rascunho antes de aceitar o contrato como pronto.

## 5. Onde o documento está bem ancorado vs. onde falta decisão

Bem ancorado na literatura e no estado real do projeto: os gates físico-hidrológicos
(HAND/TWI/DEM), a arquitetura em camadas (física causal / evidência auxiliar / LLM só
explica), o padrão de `insufficient_data` como recusa honesta, a leitura do achado
negativo de SAR, e o escopo de MVP restrito a Recife.

Ainda em aberto, precisa de decisão explícita antes de virar código: a semântica exata do
`confidence_interval` por score (bootstrap preditivo vs. delta method), e o orçamento de
EPV para o teste Modelo A vs B com DINO (não dá pra rodar hoje sem ferir a própria regra
EPV≥10 do projeto, com o n atual de 78).

## 6. Implicação organizacional

Este documento descreve uma camada nova — produto/API/contrato — que não está coberta por
nenhum gate ou regra existente do REV-P (que hoje é só pesquisa/auditoria, fail-closed,
sem menção a servir inferência). Não é incompatível com as regras fixas, mas é um escopo
de trabalho novo. Seguindo a regra de "uma tarefa por vez": antes de gerar qualquer
código de API/contrato, faz sentido primeiro (a) decidir a semântica do
`confidence_interval` e (b) decidir como respeitar o EPV no teste A/B do DINO — só depois
disso o contrato de entrada/saída do rascunho vira algo implementável sem reabrir
ambiguidade estatística no meio da implementação.

## Referências

- Rennó, C.D. et al. (2008). "HAND, a new terrain descriptor using SRTM-DEM." *Remote Sensing of Environment*.
- Beven, K.J. & Kirkby, M.J. (1979). "A physically based, variable contributing area model of basin hydrology." *Hydrological Sciences Bulletin* (TOPMODEL/TWI).
- Vickers, A.J. & Cronin, A.M. (2011). "Everything you always wanted to know about evaluating prediction models (but were too afraid to ask)." / "Misuse of the DeLong test to compare AUCs for nested models" (PMC).
- Pepe, M.S. et al. "On the assessment of the added value of new predictive biomarkers." *PMC3733611*.
- Mitchell, M. et al. (2019). "Model Cards for Model Reporting." arXiv:1810.03993.
- Literatura de *selective prediction / reject option classifiers* — trade-off cobertura-risco (arXiv:2508.07556 e correlatos).
- Literatura de mapeamento SAR de enchente por *change detection* multitemporal (MDPI Remote Sensing 2024; Clement et al. 2018, *J. Flood Risk Management*).
