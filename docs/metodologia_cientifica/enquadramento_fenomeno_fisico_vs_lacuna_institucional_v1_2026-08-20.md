# Do vazio institucional ao fenômeno físico: revisitando o eixo estruturante da pesquisa

**Data**: 2026-08-20
**Status**: memorando de reflexão metodológica — não altera modelo, dado, feature ou gate algum. Não é execução, é releitura do que já existe.
**Pergunta que originou este documento**: agora que o projeto tem negativos e positivos reais (EMSR720 no Brasil, piloto Inglaterra, CEMS multirregião), faz sentido a pesquisa continuar estruturada em torno do "reconhecimento da defasagem institucional", ou existe uma ancoragem científica mais forte para não depender só disso — mantendo a defasagem como motivação, não como fundamento?

---

## 1. A dúvida não é fraca — é uma distinção que faltava nomear

A frase que você quer revisar faz duas coisas ao mesmo tempo: justifica moralmente por que a pesquisa foca em áreas vulneráveis e pouco vigiadas, e define o que estrutura cientificamente a investigação. Essas são funções diferentes, e empilhá-las numa frase só é o que está soando "esquisito" — com razão.

Um vazio de dado institucional é um fato sobre o mundo, não uma lei física. Ele explica por que validar positivos e negativos deu tanto trabalho no projeto, mas não explica por que uma enchente acontece. Isso é entrar em tensão direta com a regra já fixada neste projeto: "o modelo NÃO deve 'descobrir' enchentes; deve refletir relações físicas conhecidas" e "variáveis físico-hidrológicas são a base causal do fenômeno". Se essa é a regra que já rege o modelo, o texto que apresenta a pesquisa devia se estruturar na mesma base — o processo físico —, e não no estado da vigilância institucional sobre ele.

A boa notícia é que você não precisa abandonar nada do que já escreveu para corrigir isso. Precisa separar as duas frases que hoje estão fundidas.

---

## 2. Os seus próprios dados já provaram por que isso importa (mod-neg-01, v2)

Em 2026-08-09 uma rodada interna do projeto (`docs/metodologia_cientifica/ext_o_que_nao_e_enchente_v2.md`, que corrigiu e substituiu a v1) comparou dois jeitos de definir negativo: por **exclusão** (a lógica que "reconhecimento de defasagem institucional" naturalmente produz — nenhum registro de enchente ali, então é negativo) e por **observação** (um analista olhou e confirmou que não inundou — Copernicus EMS). O teste foi treinar em cada fonte e avaliar na outra:

| Treinado em | Avaliado em | AUC obtido | AUC próprio | Diferença |
|---|---|---|---|---|
| exclusão | observação | 0,6222 | 0,7812 | **−0,159** |
| observação | exclusão | **0,7798** | 0,6634 | **+0,116** |

O modelo com o AUC próprio mais alto (exclusão, 0,7812) é o que **menos** generaliza — perde quase 16 pontos de AUC quando confrontado com negativo real. O modelo com AUC próprio mais baixo (observação, 0,6634) é o que quase não perde nada ao ser testado fora. A leitura registrada no próprio documento é direta: *"o AUC alto do modelo de exclusão não mede competência sobre o fenômeno; mede ajuste ao próprio critério de construção do negativo."*

Isto é a prova empírica, com dado seu, exatamente do risco que você está pressentindo: estruturar a pesquisa em torno da lacuna institucional tende a levar a construir negativo por ausência de registro — e ausência de registro, medido contra o mundo real, ensina o modelo a reconhecer o critério de vigilância, não a física da enchente. A própria Environment Agency (fonte do dado inglês) avisa isso na própria documentação: *"a ausência de cobertura por Recorded Flood Outlines numa área não significa que a área nunca inundou, apenas que não temos registro de inundação nela"* (citada em `ext_uk_adjudicacao_negativo_v1.md`, §1).

Isso não invalida a preocupação moral que você quer manter — pelo contrário, dá a ela um mecanismo científico preciso: é exatamente nas áreas vulneráveis e pouco vigiadas que a diferença entre "não inundou" e "não foi registrado" é maior, e é por isso que tratar a validade de positivo/negativo como etapa de primeira ordem (não como pressuposto) é metodologicamente necessário ali — não porque a lacuna institucional seja o fenômeno, mas porque ela é a fonte de viés mais perigosa contra medir o fenômeno corretamente.

---

## 3. O que a literatura já levantada nas suas próprias sessões aponta como estrutura alternativa

Três frentes já estão parcial ou totalmente documentadas nos seus arquivos, e as três empurram na mesma direção — ancorar no processo físico-hidrológico conhecido, tratando a lacuna de dado como problema metodológico de amostragem, não como fundamento do estudo:

**3.1 — Ancoragem físico-causal explícita.** HAND (altura acima da drenagem), TWI (convergência) e o pulso de chuva já são, pela regra fixa do projeto, a base causal do fenômeno — isso só precisa aparecer como frase estruturante do texto, não só como escolha de feature. O achado do próprio mod-neg-01 reforça isso de outro ângulo: mesmo com a inflação do negativo por exclusão, `hand_m` foi a única variável que manteve o sinal de coeficiente entre as duas fontes — elevação e declividade carregam a região, não o processo.

**3.2 — Presence-only / target-group background / PU-learning.** É um corpo de literatura estabelecido (originado em modelagem de distribuição de espécies, hoje adaptado especificamente a suscetibilidade a enchente) que trata a ausência de negativo confiável como um problema de desenho amostral resolvível — não como algo que precise de uma narrativa institucional para se justificar. Já está citado em `PLANO_ACAO_produto_v1.md` (revisão de literatura do SUSC-20Q): o artigo PBLC, *"A Positive-Unlabeled Learning Algorithm for Urban Flood Susceptibility Modeling"* (Land, 2022), usa quase a mesma frase que o projeto já usa para descrever seu próprio limite — *"case-control sampling with contaminated controls"*. Some-se a isso o framework de amostragem por distância *"Contrast or Diversity"* (J. Hydrology, 2025) e, na origem ecológica do método, Phillips et al. (2009, *Ecological Applications*) sobre viés de seleção em modelos de presença-apenas — a mesma lógica documentada na sua síntese de referências como V10.

**3.3 — Modelos de sub-registro (under-reporting).** Agostini, Pierson & Garg (AAAI-24, arXiv:2312.11754), já citado no seu `PLANO_ACAO_produto_v1.md` para o SIAC-156 de Curitiba, ataca exatamente a distinção que sustenta sua postura moral: separa estatisticamente "evento não ocorreu" de "evento ocorreu mas não foi reportado", usando correlação espacial para estimar probabilidade real de ocorrência em vez de tratar ausência de queixa como ausência de evento. Isso transforma a sua preocupação com áreas vulneráveis e não vistas de motivação retórica em ferramenta metodológica nomeada — o que é mais defensável numa banca do que "os dados institucionais são insuficientes, então pesquiso isso".

Um quarto título apareceu numa busca desta sessão e não pôde ser verificado além do resumo (*"Accelerated and Interpretable Flood Susceptibility Mapping Through Explainable Deep Learning with Hydrological Prior Knowledge"*, Remote Sensing, 2025) — sinalizo como pista a explorar depois, não como achado confirmado, porque não consegui abrir o texto completo nesta sessão.

---

## 4. Proposta concreta de reformulação

**Frase atual:**
> "É no reconhecimento dessa defasagem institucional que esta pesquisa se estrutura, direcionando-se à investigação de positivos e negativos válidos."

**Proposta** (separa o que estrutura do que motiva/justifica o método):
> "Esta pesquisa estrutura-se no conjunto de relações físico-hidrológicas — o volume de chuva que chega, a convergência da drenagem, a altura do ponto acima do curso d'água — que definem cientificamente o fenômeno da enchente urbana. É no reconhecimento da defasagem institucional de registro, particularmente aguda em áreas urbanas vulneráveis e pouco monitoradas, que se justifica tratar a validade de positivos e negativos como etapa metodológica de primeira ordem, e não como pressuposto: ausência de registro não pode ser tomada como ausência do fenômeno, sob risco de o modelo aprender o critério de vigilância institucional em vez do processo físico que pretende representar."

A primeira frase dá à pesquisa o eixo causal que a banca vai reconhecer como ciência hidrológica estabelecida — coerente com a regra fixa do projeto e com o que os seis features do v12 já operacionalizam (`ext_criterios_de_acerto_v1.md`, §1). A segunda preserva integralmente seu cunho moral e vira, ao mesmo tempo, a justificativa metodológica para todo o trabalho de auditoria de negativo que o projeto já fez (EXT-UK, CEMS, mod-neg-01) — sem precisar carregar sozinha o peso de "estruturar" a pesquisa inteira.

---

## 5. O que muda na prática

Muito pouco, e nada urgente. Isto é reformulação de enquadramento textual — provavelmente na introdução/justificativa do artigo ou do documento de planejamento —, não uma mudança de modelo, dado ou pipeline. Nenhuma regra do projeto é tocada; nenhum gate muda; nenhuma execução é necessária. As duas frentes que já estão logadas como próximo passo real continuam sendo o caminho concreto para operacionalizar essa reformulação:

- **SUSC-25** (auditoria de circularidade da amostragem de negativos, já em andamento segundo a síntese de 2026-08) é literalmente o teste desta ideia contra os dados de Recife.
- **V10** (target-group background), já registrada na sua síntese de referências como correção sugerida caso o SUSC-25 encontre circularidade, é a aplicação prática da literatura da seção 3.2 acima.

Ou seja: a reformulação conceitual que você está propondo não é uma ideia nova a perseguir — é o nome certo para um caminho que o projeto já começou a trilhar sozinho, só ainda não tinha virado frase estruturante do texto.

---

## Referências

**Internas** (já existentes no projeto, releitas para este memorando):
- `REV-P/docs/metodologia_cientifica/ext_o_que_nao_e_enchente_v2.md` — achado da assimetria de transferência entre negativo por exclusão e por observação
- `REV-P/docs/metodologia_cientifica/ext_uk_adjudicacao_negativo_v1.md` — critério N1–N4 e citação da Environment Agency
- `REV-P/docs/metodologia_cientifica/ext_criterios_de_acerto_v1.md` — decomposição física das seis features do v12
- `REV-P/docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md` — revisão de literatura do SUSC-20Q (PBLC, Contrast-or-Diversity, Agostini et al.)
- `PROJETO/REV-P_sintese_conversa_referencias_caminhos_v1.md` — vertentes V7–V11, seção 2.1 (escassez de dado como condição estrutural do campo, não falha de execução)

**Externas**:
- Phillips, S. J. et al. (2009). "Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data." *Ecological Applications*. https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/07-2153.1
- "A Positive-Unlabeled Learning Algorithm for Urban Flood Susceptibility Modeling" (PBLC). *Land*, 2022. https://doi.org/10.3390/land11111971
- "Contrast or Diversity: Non-Flood sampling in urban flood susceptibility modelling." *Journal of Hydrology*, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0022169425003919
- Agostini, Pierson & Garg. "A Bayesian Spatial Model to Correct Under-Reporting in Urban Crowdsourcing." AAAI-24. https://arxiv.org/abs/2312.11754
- "Data Uncertainty of Flood Susceptibility Using Non-Flood Samples." *Remote Sensing*, 2025. https://www.mdpi.com/2072-4292/17/3/375
- "Accelerated and Interpretable Flood Susceptibility Mapping Through Explainable Deep Learning with Hydrological Prior Knowledge." *Remote Sensing*, 2025 — título/resumo apenas, não verificado em texto completo nesta sessão. https://www.mdpi.com/2072-4292/17/9/1540
