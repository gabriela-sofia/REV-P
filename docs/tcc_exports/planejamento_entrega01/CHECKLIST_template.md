# Lista de conferência — Documento de Planejamento (Entrega 01)

Extraída do que o template dos professores efetivamente pede, item a item, com o
lugar do documento onde cada exigência é atendida e o que ainda depende de decisão
sua. As frases entre aspas são do próprio template.

---

## 0. Exigências globais (valem para o documento inteiro)

| # | O que o template exige | Onde está | Verificar |
|---|---|---|---|
| 0.1 | "must not exceed three pages in total, including all artifacts (references, figures, tables, annexes)" | 3 páginas exatas | Reconferir depois de qualquer edição sua. O controle de espaço mais sensível é a largura da Fig. 1 (`\resizebox`) — ver README |
| 0.2 | "must strictly follow the provided template formatting" | `IEEEtran` classe `conference` | Se os professores derem o `.tex` deles, trocar só o preâmbulo |
| 0.3 | "written entirely in either English or Portuguese" | Português, sem mistura | Termos técnicos em inglês estão em itálico, não traduzidos |
| 0.4 | Cabeçalho: título, curso, turma, equipe, nomes, e-mails | Título + PUCPR + Turma/Equipe + e-mail | **Preencher número da equipe**; confirmar turma |
| 0.5 | Figuras em formato vetorial ("prioritize vector formats such as PDF or SVG") | Fig. 1 é TikZ nativo; Fig. 2 é PDF vetorial | Nenhuma imagem rasterizada no documento |

---

## 1. Seção I — Descrição do Projeto

| # | O que o template exige | Onde está | Verificar |
|---|---|---|---|
| 1.1 | "up to five paragraphs and no more than 500 words" | 5 parágrafos, **495 palavras** (v2) | Recontar se você reescrever |
| 1.2 | **Contexto do trabalho** | §1, parágrafo 1 | — |
| 1.3 | **Problema central a ser investigado** | §1, parágrafo 3: a ablação da entrega anterior e as três condições que ela impõe | — |
| 1.4 | **Objetivos principais** | §1, parágrafo 4 | Cinco objetivos na v2: variáveis reprodutíveis, negativo observado, efeito da definição de negativo, transferência entre classes de relevo, produto auditável |
| 1.5 | "situate the topic within the state of the art, including references that support the statements" | HAND, TWI, D-infinity, PU-learning, sub-reporte | Toda afirmação de contexto tem citação |
| 1.6 | "The chosen approach should also be indicated in general terms" | §1, parágrafo 5 | Detalhe técnico foi deslocado para a Seção II, como pedido |
| 1.7 | "Technical details (datasets, architectures, resources) will be addressed in Section II" | Nenhum nome de biblioteca ou tamanho de amostra na Seção I | — |
| 1.8 | **Figura ilustrativa do problema** ("schematics, flowcharts, or visual examples") | Fig. 1 — fluxograma da arquitetura com os dois limites | — |

### Autoavaliação da Seção I (as 4 perguntas do template)

- [ ] **O problema central está claramente definido e sua relevância justificada por referências?** → As três condições estão nomeadas e cada uma tem referência.
- [ ] **Os objetivos estão claramente formulados e são consistentes com o problema?** → Quatro objetivos específicos, cada um respondendo a uma das condições.
- [ ] **A pergunta de pesquisa é clara, específica e alinhada aos objetivos?** → Está em negrito, isolada, e tem duas partes verificáveis (discriminação sob negativo observado com validação agrupada; transferência para terrenos não representados).
- [ ] **A seção apresenta, mesmo em termos gerais, como os objetivos podem ser alcançados?** → Último parágrafo, sem detalhe técnico.

---

## 2. Seção II — Materiais e Métodos

| # | O que o template exige | Onde está | Verificar |
|---|---|---|---|
| 2.1 | **Datasets potenciais** | §II-A e §II-B | Cada base com origem, tamanho e função |
| 2.2 | **Arquiteturas a implementar** | §II-C | Firth como rota primária; não lineares como diagnóstico |
| 2.3 | **Métodos e conceitos que sustentam o trabalho** | §II-C e §II-D | EPV, bootstrap, walk-forward, contrato de inferência |
| 2.4 | **Linguagens, ambientes, bibliotecas** | Tabela I, linha "Ambiente" | Versões que são requisito real estão declaradas |
| 2.5 | **Hardware e outros recursos** | §II-E + Tabela I | Números medidos, não estimados |
| 2.6 | **Figura com amostras representativas dos datasets**, evidenciando "possible challenges" | Fig. 2, quatro painéis | Cada painel mostra um desafio nomeado no texto |
| 2.7 | "assess whether the team truly has the necessary infrastructure... considering computational resources, experience, and workload distribution" | §II-E | **É trabalho individual**: a distribuição de carga vira sequenciamento, e isso está dito |
| 2.8 | "If significant limitations are identified, some aspect of the proposal must be adjusted" | §II-E declara os dois ajustes já feitos por limitação real | — |
| 2.9 | **Tabela-resumo de Materiais e Métodos** | Tabela I, três linhas | Deliberadamente enxuta: a contextualização mora no texto da Seção II, e a legenda diz isso. O template autoriza adaptar o formato |

### Autoavaliação da Seção II

- [ ] **Os datasets são apropriados para investigar a pergunta de pesquisa?** → O negativo observado é a base que torna a pergunta respondível; isso está argumentado, não só listado.
- [ ] **Arquiteturas e métodos alinhados aos objetivos e ao estado da arte?** → Firth justificado por interpretabilidade e amostra pequena, com referência.
- [ ] **A equipe tem os recursos de hardware e software?** → Sim, com tempo de ajuste medido em segundos e o gargalo real (versão de biblioteca) nomeado.
- [ ] **Tabela e figura dão visão suficientemente clara?** → A tabela resume; o texto contextualiza. Nenhuma informação existe só na tabela.

---

## 3. Seção III — Etapas e Marcos Físicos

| # | O que o template exige | Onde está | Verificar |
|---|---|---|---|
| 3.1 | "organize project development into well-defined stages" | E0 a E7 | — |
| 3.2 | "clearly describing which activities will be carried out in each phase" | Cada etapa abre com a ação | — |
| 3.3 | "which milestones will serve as references to monitor progress" | M1 a M7, marcados em cada etapa | Consistentes com o cronograma interno do REV-P |
| 3.4 | "relating them to concrete deliverables (partial report, prepared dataset, defined protocol, source code, preliminary results, near-final manuscript)" | Campo *Entregável* em cada etapa | Todo entregável é um arquivo, não um estado |
| 3.5 | "avoiding overly generic descriptions" | Cada etapa cita número, arquivo ou critério | Nenhuma etapa diz apenas "melhorar" ou "analisar" |
| 3.6 | "indicate not only what will be done but also **which evidence will demonstrate completion**" | Campo *Evidência* em cada etapa | Este é o item que o template mais cobra e o mais fácil de perder na revisão |
| 3.7 | Complementaridade com a Seção IV | Mesmos rótulos E0–E7 nas duas seções | Se você renomear uma etapa, renomear nas duas |

### Autoavaliação da Seção III

- [ ] **As etapas são claras e progressivas, sem generalidades?**
- [ ] **Cada etapa tem entregável ou marco bem definido?**
- [ ] **Há critério objetivo para avaliar a conclusão de cada etapa?**
- [ ] **As etapas são viáveis considerando recursos, tempo e cronograma?**
- [ ] **Há consistência entre as etapas daqui e as do cronograma?**

---

## 4. Seção IV — Cronograma

| # | O que o template exige | Onde está | Verificar |
|---|---|---|---|
| 4.1 | "all project stages, from the initial submission to the final submission" | Tabela II, de 09/08 a 17/11 | — |
| 4.2 | "essential that it aligns with the Checkpoints and deadlines defined in the syllabus" | Linha *Checkpoints* com as quatro datas | **Conferir as datas contra o plano de ensino** |
| 4.3 | "Activities must be distributed realistically" | Quinzenas, uma frente por vez | — |
| 4.4 | "group members must work in parallel" | Trabalho individual: o texto explica que o paralelismo vira sobreposição de etapas | Se a equipe voltar a ter mais gente, esta frase muda |
| 4.5 | Etapa mínima: **levantamento do estado da arte** | Linha 1 da Tabela II | O template lembra que "should already have been started" — o texto diz isso |
| 4.6 | Etapa mínima: **processamento do dataset** | Linha 2 | — |
| 4.7 | Etapa mínima: **experimentação** | Linha 3 | — |
| 4.8 | Etapa mínima: **análise de resultados** | Linha 4 | — |
| 4.9 | Etapa mínima: **redação do manuscrito** | Linha 6 | — |
| 4.10 | Tabela no modelo da Tabela III (ou figura vetorial de cronograma) | Tabela II | — |

### Autoavaliação da Seção IV

- [ ] **O cronograma inclui todas as etapas mínimas recomendadas, mesmo que indiretamente?**
- [ ] **Os marcos-chave (checkpoints e entregas) estão incorporados?**
- [ ] **A alocação de tempo é realista dado o prazo?**
- [ ] **O planejamento permite execução paralela?**
- [ ] **As atividades estão logicamente sequenciadas, sem lacunas ou sobrecarga?**
- [ ] **O cronograma é consistente com as etapas da Seção III?**

---

## 5. Tom e escrita (Zobel, *Writing for Computer Science*)

Os professores entregaram esse livro junto com o template; vale usar os critérios
dele como régua de revisão.

| Princípio | O que fazer | Onde isso aparece no documento |
|---|---|---|
| **Economia** (cap. 6) | "Every sentence should be necessary." Cortar palavras que não mudam o sentido. | O documento cabe em 3 páginas sem apêndice e sem enchimento |
| **Tom** (cap. 6) | "Aim for austerity, not pomposity." Uma ideia por parágrafo, frases curtas, específico em vez de vago. | Cada parágrafo da Seção II tem um assunto só, anunciado em negrito |
| **Não obscurecer** (cap. 6) | Trocar "melhorias levam a melhor desempenho" por número real. | "0,8855 → 0,4834 → 0,4689"; "0,056 s"; "216,55 km²"; "5,94:1" |
| **Não exagerar em ressalvas** (cap. 6) | Overqualifying esconde a afirmação. Dizer o resultado e depois o limite. | O colapso de 2026 é afirmado e depois delimitado, não diluído |
| **Sem espantalhos** (cap. 6) | Contrastar com o atual, não com uma alternativa impossível. Opinião marcada como opinião. | A crítica é ao próprio resultado anterior do projeto, com número |
| **Referências** (cap. 6) | Relevantes, necessárias, primárias; nada de encher bibliografia. | 10 referências, todas citadas no texto, todas fonte primária |
| **Planejamento de pesquisa** (cap. 2) | "Components should be identified in advance, but do not necessarily have to be completed in turn"; planejar perguntando "que evidência preciso coletar para convencer um leitor cético?"; sobrepor etapas; tudo demora mais do que o planejado. | Campo *Evidência* em cada etapa; parágrafo de risco e revisão ao fim da Seção III; nota de flexibilidade na Seção IV |

### Decisão de estilo que é sua

O Zobel recomenda voz ativa e uso de "I"/"we" para separar o que é seu do que é da
literatura. O seu artigo anterior é inteiramente impessoal. O documento atual mantém
a forma impessoal do PT01, com primeira pessoa apenas onde há uma decisão sua a
declarar. Se preferir uniformizar para um dos dois extremos, é uma troca de poucas
frases.

---

## 6. O que este documento deliberadamente **não** faz

Registrado aqui para você conferir se concorda, porque cada item foi uma escolha:

1. **Não apresenta resultados como resultados.** Os números do trabalho já feito
   aparecem como estado de partida e como caracterização de dados, nunca como achado
   desta entrega — os resultados são da Entrega 02.
2. **Não promete que os problemas em aberto serão resolvidos.** O colapso de 2026 em
   Curitiba entra como limite conhecido, com etapa de encerramento, não de solução.
3. **Não usa "validado", "confirmado" ou "ground truth"** fora do que os artefatos
   sustentam.
4. **Não esconde o que pode mudar.** A Seção III termina declarando quais etapas têm
   risco real e qual é a rota alternativa de cada uma.
5. **Não trata a assimetria entre regiões como falha.** Ela é condição documentada,
   como no PT01.

---

## 7. Onde cada exigência de contextualização foi atendida no texto corrido

A crítica de que o documento tratava a primeira entrega de forma abrangente demais
foi endereçada nestes pontos específicos, todos em prosa e não em tabela:

| Onde | O que o texto agora diz |
|---|---|
| §I, parágrafo 2 | A ordem real em que o projeto foi construído: leitura territorial, heurísticas e conjuntos tabulares, recorte em \emph{patches}, evidências externas, manifests/registries/QA, Protocolo C com quatro portões e quatro decisões, e só então DINOv2 congelado. Seguido do peso conferível: 59 \emph{patches}, 128 ativos, 36 conjuntos, 87 manifests, 43 testes, 9 eventos candidatos. Termina com o teto que aquela entrega declarou por conta própria |
| §I, parágrafo 3 | O achado que dá função a essa base: os três valores da ablação e a leitura de que o classificador reconstruía critérios do próprio projeto |
| §II, parágrafo 1 | Separação explícita entre o que já está construído e auditado e o que este plano ainda vai construir. É o parágrafo que sustenta a viabilidade do cronograma |
| §II, último parágrafo | Os dois ajustes de escopo que limitações reais já forçaram, como o template pede em "some aspect of the proposal must be adjusted" |
| §III, último parágrafo | Riscos e rotas alternativas de E2, E3 e E5 — o que muda se algo não funcionar, declarado agora |
| §IV | Por que o cronograma se sobrepõe em vez de correr em paralelo, e por que prevê folga |


---

## 8. Revisão v2 (2026-08-11)

Itens que mudaram de estado em relação à v1 e que valem reconferência:

| # | O que mudou | Onde conferir |
|---|---|---|
| 8.1 | Correção factual sobre Petrópolis e sobre o negativo brasileiro | `NOTA_versoes.md` §1 |
| 8.2 | Seção I ganhou um parágrafo inteiro de contextualização física (HAND, TWI, D-infinity em linguagem acessível) | §I, parágrafo 2 |
| 8.3 | Objetivos passaram de quatro para cinco | §I, parágrafo 4 |
| 8.4 | Seção II reorganizada com foco em treino; hierarquia do negativo declarada em três níveis | §II, blocos 1–4 |
| 8.5 | Figura 2 refeita: composição por conjunto, sobreposição de classes, efeito da definição de negativo, régua de HAND por relevo | `fig/make_fig2.py` |
| 8.6 | Etapas fechadas por evidência, não por data; E4 (holdout temporal) é nova | §III |
| 8.7 | Referência nova: Tarboton (1997), fonte primária do D-infinity | bibliografia, item 4 |
| 8.8 | Pendência declarada: curadoria de AOI/conjunto ainda não executada | `NOTA_versoes.md` §3 |

---

## 9. Revisão v3 (2026-08-13) — realinhamento do problema central

| # | O que mudou | Onde conferir |
|---|---|---|
| 9.1 | Problema central deixou de ser a classe negativa e passou a ser a heterogeneidade da evidência | §I, par. 4; `NOTA_versoes.md` §6 |
| 9.2 | Harmonização virou o núcleo da Seção II e a etapa E2 | §II, primeiro bloco; §III |
| 9.3 | Documento descreve um percurso único, não duas frentes | §I, par. 5; Fig. 1 |
| 9.4 | Figura 1: coluna 2 é Harmonização, coluna 4 é Serviço | Fig. 1 |
| 9.5 | Figura 2, painel (b): matriz de variável disponível por fonte | `fig/make_fig2.py` |
| 9.6 | Etapas E5 (aplicação às regiões brasileiras) e E6 (serviço e explicação) são novas | §III |
| 9.7 | Título alinhado ao arco: da evidência heterogênea ao serviço de inferência auditável | cabeçalho |
| 9.8 | Pendência: curadoria/harmonização ainda não executada — é o entregável de E2 | `NOTA_versoes.md` §3 |

---

## 10. Pontos de dados e programação — o que a Seção II já assume como resolvido (E2)

Isolado a pedido seu: só o que é trabalho de dado/código implícito nas suas
próprias notas em Materiais e Métodos, na ordem em que o texto as levanta. Cada
bloco cita a frase/nota de origem, decompõe em passo programável, e diz o que
conta como pronto — no seu padrão de "critério de prova"
(`PLANO_ACAO_produto_v1.md`), não "deveria funcionar".

> **Status em 2026-08-14, conferido contra o commit `e627af3` (`ds03`-`ds05`,
> `ter02`/`ter04`/`ter05`, `mod_mec02`, `aud_chuva01`)**: Bloco 1 **fechado** —
> rodei o pipeline e o teste, artefatos batem com o relatório número a número.
> Bloco 3 **fechado** — `nivel_negativo=ausência` para Curitiba confirmado no
> dado. Bloco 2 **fechado para Curitiba, aberto para Petrópolis** — a derivação
> de terreno existe na convenção nova, mas não há comparação bit a bit
> registrada porque Petrópolis ainda tem zero pontos rotulados para comparar.
> Achados novos da validação (não estavam no relatório): ver `## 12` abaixo.
>
> **`main.tex` já foi atualizado para v5** para refletir este status (E2
> marcada concluída, Curitiba/Petrópolis diferenciados, base integrada tirada
> da lista de pendências) — ver `NOTA_versoes.md` §8. Os achados do `## 12`
> abaixo (mistura de chuva, TWI, teste de determinismo) **não** entraram no
> `main.tex`: são achado novo ainda sem correção, e o documento desta entrega
> não apresenta achado como resultado (regra §6.1).

### Bloco 1 — A tabela única (o "dataset final/tabela de pontos")

**Origem**: a nota que você deixou depois de "HAND das fontes externas vem de
produto global de 30m — limitação declarada, não equivalência": *"eu faço esse
ajuste de escopo, diminuindo e transformando todos os dados no dataset
final/tabela de pontos antes de entregar o planejamento, vou deixar tudo
perfeito no quesito dados e metodologia antes da entrega"*. É a mesma coisa que
o parágrafo de abertura da Seção II já promete: *"Toda fonte passa por uma
redução a tabela de pontos, e cada linha declara procedência, resolução
nativa, unidade de agrupamento, classe de relevo e mecanismo."* Hoje isso não
existe como um arquivo único — Recife (v12), Curitiba (SIAC 156) e as fontes
externas (CEMS, EA/UK) vivem em três pipelines separados, cada um com seu
próprio schema.

| # | Passo | Entrada | Saída / critério de pronto |
|---|---|---|---|
| 1.1 | **Congelar o esquema-alvo** antes de programar qualquer redução (uma tarefa por vez: definir antes de popular) | as 7 colunas que o texto já promete: `fonte`, `AOI`, `nivel_negativo` (observado/exclusão/ausência), `unidade_agrupamento` (evento/AOI), `classe_relevo` (serra/planície), `mecanismo`, + as 6 variáveis físicas (elevação, declividade, HAND, TWI, `rain_max_24h`, `rain_decay_index`) | um arquivo de contrato (schema/dataclass/CSV header) versionado, sem nenhuma linha de dado ainda |
| 1.2 | **Script de redução por fonte**, um por vez: Recife (v12, 278 pts) → Curitiba (SIAC 156, 1.471 unidades) → Copernicus EMS (EMSR720 RS + análogos serra/planície, 25.249 pts) → EA/UK (7.476 pts) | os datasets já existentes de cada fonte, cada um no seu schema atual | 4 arquivos (um por fonte) já no esquema-alvo, com `nivel_negativo` e `mecanismo` preenchidos por linha; onde uma variável não existe na fonte (ex.: chuva fora do piloto inglês), o campo fica nulo declarado, não estimado |
| 1.3 | **Filtro de admissão dos três critérios** que o texto já define: dentro da AOI declarada; variáveis computadas na mesma cadeia; grupo de validação identificável | os 4 arquivos do passo 1.2 | cada linha rejeitada tem motivo nomeado registrado (fora da AOI / variável faltando / sem unidade de agrupamento) — isso é literalmente o "restante fica como proveniência, com o motivo do descarte nomeado" que o texto promete |
| 1.4 | **Checagem de duplicidade entre fontes** (nenhum ponto pode existir em duas fontes) | os 4 arquivos filtrados | relatório de zero duplicatas, ou lista de conflitos resolvidos com critério declarado |
| 1.5 | **Relatório de contagem** — quantos pontos entram e quantos saem por fonte, com o motivo | passos 1.3 e 1.4 | é o *Entregável* que a Seção III já promete para E2: "contagem do que entra e do que sai por fonte, com o motivo nomeado" — se esse relatório existir, E2 tem prova |
| 1.6 | **Consolidação final** num único arquivo versionado | os 4 conjuntos admitidos | a "base de conhecimento geoespacial única" que a Seção II descreve deixa de ser frase e vira arquivo com caminho e hash/data |

### Bloco 2 — Auditoria bit a bit da cadeia de terreno (Curitiba e Petrópolis)

**Origem**: nota depois de "reproduzida *bit a bit* contra o raster de
referência de Recife": *"eu to auditando isso das demais regiões que eu tenho,
[...] quero colocar aqui o resultado mais atual que eu tiver nesse aspecto"*.
Hoje só Recife tem essa auditoria (SUSC-20F: HAND/TWI batem exato contra o DTM
PE3D merged; elevação/declividade com ~2,7m/5,6° de diferença histórica
documentada, raster original não recuperável). Curitiba e Petrópolis usam o
mesmo motor genérico (`susc_20g`, HAND/TWI/D-infinity) sem essa checagem
independente registrada.

| # | Passo | Critério de pronto |
|---|---|---|
| 2.1 | Escolher a fonte de terreno de referência de cada região (Tabela I já lista GLO-30 e SGB/CPRM como as outras duas, ao lado do PE3D 10m de Recife — confirmar qual serve de referência para Curitiba e qual para Petrópolis) | fonte nomeada e versão/data registradas, uma por região |
| 2.2 | Rodar `susc_20g` numa amostra de células conhecidas de cada região e comparar HAND/TWI contra o cálculo de referência (mesmo método usado para validar Recife) | número real de erro (m ou %), não "deveria bater" |
| 2.3 | Registrar o resultado no mesmo formato do achado de Recife (match exato ou diferença documentada + causa) | um arquivo/linha por região, pronto para entrar na Seção II quando você decidir atualizar o texto — não mexo nisso até você confirmar o número |

### Bloco 3 — Curitiba: o dataset de 1.471 unidades ainda não está no esquema-alvo

**Origem**: implícito no mesmo parágrafo do Bloco 1 — o texto lista "o conjunto
de Curitiba, com 1.471 unidades" ao lado do que já está auditado, mas o SIAC
156/SUSC-20N vive num pipeline próprio (`outputs_public/.../susc_20k`), sem os
campos `nivel_negativo`/`unidade_agrupamento`/`classe_relevo` do esquema novo.
Isso é o passo 1.2 aplicado especificamente a Curitiba, mas separo porque é
onde a harmonização é mais trabalhosa: Curitiba não tem hoje um negativo
*observado* (é ausência de registro, como Recife), então a coluna
`nivel_negativo` para as 1.471 unidades precisa vir declarada como tal, não
inferida.

| # | Passo | Critério de pronto |
|---|---|---|
| 3.1 | Mapear as colunas atuais do dataset Curitiba (SUSC-20N) para o esquema-alvo do Bloco 1 | tabela de correspondência coluna-antiga → coluna-nova, sem perda de campo |
| 3.2 | Declarar `nivel_negativo = ausência` para as 1.471 unidades, coerente com a hierarquia que a Seção II já define no texto | nenhuma linha de Curitiba entra como "observado" sem uma fonte real que justifique |

**Como usar os Blocos 1-3**: são trabalho real de dado/código, não edição de
texto — não faço eles por você. Quando concluir um passo, me diga qual, que eu
confiro contra o artefato gerado (arquivo, contagem, número de erro) e ajusto
só o ponto correspondente do `main.tex`, do mesmo jeito que fiz com as notas
do rascunho anterior.

---

## 11. Outras pendências (não é dado, não é programação — fora do escopo pedido acima, registrado à parte)

| # | Item | O que fazer |
|---|---|---|
| 11.1 | Posição do parágrafo de limitações: Materiais e Métodos (onde está) ou Descrição do Projeto? | Pedir ao orientador o documento de expectativas por seção; me diga o que ele disser e eu ajusto |
| 11.2 | Pergunta já registrada em `NOTA_versoes.md` §4 ("plano por evidência vs. plano por data") | Conferir se já foi respondida pelo professor |
| 11.3 | Turma e equipe ainda em vermelho como placeholder (`7A`/`N`) | `main.tex` linha 46 |
| 11.4 | E-mail institucional — confirmar se é o correto | `main.tex` linha 46 |
| 11.5 | Limite de 3 páginas — ambíguo no meu ambiente (sem hifenização de português) | Recompilar no Overleaf |
| 11.6 | Seção I com **573 palavras** (recontado após v7), acima do limite de 500 (regra 1.1) — igual à v6: a v7 corrigiu "a resolução segue o mecanismo" (decisão revertida em `90ecb5c`) por uma frase mais longa e correta no parágrafo 3, então o total não mudou | Cortar ~70-75 palavras antes de entregar. Ainda não cortei nada por conta própria (regra desta conversa); candidato mais barato continua o parágrafo 3 ("O plano se apoia..."), que soma decisões já implícitas no parágrafo 1 — a correção da v7 tornou uma de suas frases um pouco mais longa, não elimina o candidato |
| 11.7 | `Fig.~\ref{fig:datasets}` — compilou normal aqui após duas passadas | Confirmar no Overleaf |
| 11.8 | Grep por overclaim ("validado", "confirmado", "comprovado", "ground truth") | Já rodado: zero ocorrências, sem ação |

---

## 12. Validação do commit `e627af3` (2026-08-14) — achados que o relatório não tinha

Rodei o pipeline de ponta a ponta neste ambiente (`ds03`→`ds05`,
`aud_chuva01`, e a suíte de teste) e conferi manifesto/CSV contra o texto de
`ext_tabela_unica_e_pool_harmonizado_v1.md`. Os números do relatório batem
exatamente com os artefatos (116.992/33.349/33.071, contagem por fonte,
vereditos do `aud_chuva01` — só Recife é `MISTURA_DE_FONTES`, Curitiba e UK
são `FONTE_UNICA`). Quatro coisas que a validação encontrou e o relatório não
menciona:

| # | Achado | Gravidade | O que fazer |
|---|---|---|---|
| 12.1 | **`test_consolidacao_e_deterministica` FALHOU aqui.** Rodei `pytest` na suíte inteira: 18/19 passam, esse falha. Investiguei a fundo: não é bug de dado — comparei os dois CSVs por conteúdo (ordenando por `ponto_id`) e são **idênticos**. O que muda é só a **ordem das linhas**, o que já basta para trocar o sha256. `carregar()` em `ds05` concatena as fontes numa ordem fixa, mas nada garante a ordem interna de cada arquivo do `ds04` — provavelmente herdada de alguma operação sem `sort_values`/chave estável rio acima. Rodei `ds05` três vezes seguidas aqui: as três bateram entre si, só divergiram do hash já gravado no manifesto (gerado antes, possivelmente noutra versão de pandas/numpy — `environment.yml` não fixa versão de nenhum dos dois) | **Alta** — é exatamente o tipo de coisa que o teste foi desenhado para pegar, e hoje ele pega até quando não há erro real, o que é tão ruim quanto não pegar | Ordenar explicitamente por `ponto_id` (ou outra chave estável) antes de gravar cada CSV em `ds04` e em `ds05`, e então re-gravar o hash de referência uma vez. Sem isso, o teste vai continuar quebrando em qualquer ambiente novo (você já recriou esse conda do zero uma vez em 06/08) mesmo com o pipeline correto |
| 12.2 | Comentário no código de `ds05` (linha ~195) diz "209 linhas repetem unidade de observação" em Curitiba; o valor real no manifesto é **812**. Comentário desatualizado, não afeta o dado | Baixa | Atualizar o comentário para 812 quando for mexer nesse arquivo de novo |
| 12.3 | Bloco 2 (auditoria bit a bit) **não fechou para Petrópolis**, e é estrutural, não esquecimento: `ter04` deriva o terreno de Petrópolis na mesma convenção (registrado em `registro_derivacoes.csv`), mas a comparação bit a bit só existe para Recife e Curitiba porque ela é feita **nos pontos rotulados**, e Petrópolis tem zero. Não dá pra fechar esse item sem rótulo — fica dependente de C4/decisão de Petrópolis, não de rodar mais um script | Informativa | Nenhuma ação de dado possível agora; só rastrear |
| 12.4 | O próprio relatório já eleva isto a prioridade, e concordo pela medida: a fonte de chuva de Recife (CHIRPS × ERA5-Land na mesma coluna) tem AUC de indicador de fonte (0,826) maior que a própria chuva (0,738) — o preditor principal da trilha pluvial está parcialmente medindo proveniência, não precipitação | **Alta** | Reamostrar uma fonte única de chuva pros 278 pontos de Recife, como o relatório §4 já registra como pendente |

**Reproduzido aqui**: `python -m pytest tests/test_ds03_ds05_tabela_unica.py -q`
→ 18 passed, 1 failed (`test_consolidacao_e_deterministica`). Os outros 18,
incluindo os que checam contrato, duplicidade cruzada e admissão pelos três
critérios, passam limpos.

---

## 13. Validação do commit `90ecb5c` (2026-08-15) — resolução única de 30 m

Este é o commit sem push que trabalha a lacuna de 30 m em Recife (não é
nenhum dos seis que você colou — aqueles são trabalho posterior: gate #8,
Protocolo C, feature store DINO, DINO×SEDEC, reexecução do pipeline oficial.
`90ecb5c` vem antes de todos eles).

**Veredito**: a estratégia é sólida e resolve a lacuna. Não é o caso de ir
atrás de dado alternativo em 30 m nativo — a solução (b) que você cogitou,
resolver cientificamente, é a que já está no commit e ela é suficiente.
Confirmei cada número do relatório contra o artefato real, não contra o
texto:

| Número citado | Artefato conferido | Bate? |
|---|---|---|
| 89.065 admitidos / 64.989 pool / 4 fontes no pool / 0 em cadeia global | `local_runs/ds-05-tabela-unica/manifesto_v1.json` | Sim, exato |
| 782 derivações; 658 chips Nível 1 | `local_runs/ter-04-registro-auditoria/registro_auditoria.json` | Sim, exato |
| 658/661 chips, 0 rede degenerada, pearson elevação 1,000 / HAND 0,946 | `local_runs/_ter06.log` | Sim, exato |
| AUC 0,7241 (ESTRITO) / 0,7198 (AMPLIADO); HAND −1,3636; TWI +0,4888; declividade +0,0310 [IC cruza zero] | `local_runs/mod-mec-02/resultado.json` (regravado 5 min antes do commit) | Sim, exato |
| LOSO Curitiba 0,4997; melhor separação isolada 0,186 | idem | Sim, exato |
| Transferência planície→serra 0,8018; serra→planície 0,6881 | idem, `leave_one_relevo_out` | Sim, exato |
| `recife__variante_nativa_10m.csv` existe, não é descartada | `local_runs/ds-04-reducao/` — presente, 95.926 bytes, diferente do canônico (94.748 bytes) | Sim |

Por que o argumento central se sustenta, e não é "Recife colapsando": o HAND
do v12 de Recife tem coeficiente −0,0001 com p = 0,978 — o modelo pluvial não
usa terreno como preditor, usa chuva. As duas variáveis que se degradam com
30 m (declividade, TWI) são exatamente as duas que aquele modelo nunca
consultou. Isso não é uma racionalização pós-hoc: é o mesmo coeficiente que
já estava registrado antes desta decisão. Nenhuma capacidade "parou de poder
ser usada" — ao contrário, Sen1Floods11 e UFO **ganharam** `slope_deg` e
`twi_dinf`, que nunca tinham existido nessas fontes (conferido no log do
`ter06`: só há comparação de pearson para elevação e HAND porque não havia
valor anterior de declividade/TWI para comparar). Não há, portanto, ferramenta
perdida para substituir.

Conferi também o `diff` de `ds04_reduzir_por_fonte.py` linha a linha: a
hierarquia do negativo (ausência nunca entra como negativo observado) está
intacta e comentada explicitamente no código; o mapeamento de AOI por chip no
Nível 1 é razoável e documentado (separa unidade de validação cruzada —
`grupo_cv`, o país — da unidade de harmonização de terreno — o chip). Nada a
apagar por erro científico.

**Um achado, e é o mesmo bug do item 12.1, ainda não corrigido:**

| # | Achado | Gravidade | O que fazer |
|---|---|---|---|
| 13.1 | `test_consolidacao_e_deterministica` **continua falhando**, mesma causa de 12.1 (ordem de linha, não conteúdo — reconferi agora rodando `ds05` de novo e comparando os CSVs ordenados por `ponto_id`: idênticos). A correção que ficou pendente em 12.1 (ordenar por `ponto_id` antes de gravar em `ds04`/`ds05`, regravar o hash de referência) não foi aplicada entre os dois commits | **Alta** | A mesma recomendação de 12.1, ainda de pé |
| 13.2 | A mensagem do commit `90ecb5c` afirma **"21 testes passando, um pulado"** — ou seja, zero falhas. Rodei a suíte agora: **20 passed, 1 failed (13.1), 1 skipped**. A contagem de testes bate (22 = 21 + 1, se o pulado for o de rede degenerada — não há chip degenerado nesta execução, então esse teste específico não tem o que avaliar e pula, o que é comportamento correto); a falha é que não deveria haver falha nenhuma e a mensagem do commit não registra a que houve | **Alta, mas só de rastro/commit-hygiene** — não é erro de dado nem de estratégia, é uma afirmação no commit que não é verificável como está | Antes do push: aplicar o fix de 13.1 (resolve o teste e a mensagem passa a ser verdadeira) **ou**, no mínimo, corrigir a alegação da mensagem para registrar a falha conhecida |

A pendência de chuva (12.4) continua igual — este commit não a toca, e o
próprio relatório (`ext_resolucao_unica_30m_v2.md` §7) registra isso
explicitamente como prioridade em aberto.

**Reproduzido aqui**: `python -m pytest tests/test_ds03_ds05_tabela_unica.py -q`
→ 20 passed, 1 failed (`test_consolidacao_e_deterministica`), 1 skipped.

---

## 14. Pendência de chuva resolvida — commit `13fcea2` (2026-08-16)

A pendência 12.4 (fonte de chuva misturada em Recife, CHIRPS×ERA5-Land) foi
fechada nesta sessão. Decisão tomada com você: padronizar os 278 pontos em
Open-Meteo/ERA5-Land — não CHIRPS, apesar de ser o produto cientificamente
preferível (tem estação real), porque o `aq_chirps3_v3.py` deste projeto
documenta bloqueio por scraping do servidor da CHC, e os 97 pontos a
reamostrar cobrem 85 datas distintas (2015–2022), não um evento concentrado.
Open-Meteo já tinha 3 usos bem-sucedidos no projeto sem esse risco.

| # | O que foi feito | Resultado |
|---|---|---|
| 14.1 | `chuva02_padronizar_fonte_unica_recife.py` reamostrou os 181 pontos que estavam em CHIRPS, mesma fórmula das fontes Open-Meteo já existentes | 181/181 reamostrados; 269/278 com chuva no total (os 9 sem valor já eram assim antes, sem `event_date` na origem — não é regressão) |
| 14.2 | `aud_chuva01` re-executado | Recife passa de `MISTURA_DE_FONTES` para `FONTE_UNICA` — nenhuma fonte do projeto tem mais mistura de produto de precipitação. Corrigido de passagem um bug preexistente e não relacionado (`TypeError` em `periodo_por_fonte` com `data_evento` misturando string e NaN) |
| 14.3 | `mod_recife03_pluvial_fonte_unica.py` repete a metodologia exata do v12 publicado (Firth, 6 features, LOO, mesma semente), só trocando a chuva | LOO-AUC 0,6781 → 0,6276; coeficiente da chuva +0,9896 (p<0,0001) → +0,4910 (p=0,0005) — continua dominante e significativo, só que metade da magnitude antiga media proveniência, não precipitação. HAND segue não significativo. `twi_dinf` perde a significância marginal que tinha (0,046→0,123) — registrado, não é ação imediata mas cite se usar `twi_dinf` isoladamente sobre Recife |
| 14.4 | Teste de regressão invertido | `test_recife_continua_registrado_com_mistura...` (que existia para *garantir* a mistura, com aviso explícito no docstring para virar quando corrigida) virou `test_recife_tem_fonte_de_chuva_unica`, guardando contra a mistura voltar |
| 14.5 | Suíte inteira | 21 passed, 1 skipped (rede degenerada, não aplicável nesta execução) — zero falhas |

**Não é o modelo de Recife colapsando — é o modelo ficando mais honesto**: a
chuva continua o preditor dominante, com sinal certo e p<0,001; só a metade
inflada pelo confundimento de fonte saiu.

Commitado localmente (`13fcea2`, branch `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`),
**sem push** — mesma regra interna do REV-P. Detalhe completo, tabela de
coeficientes e ordem de reprodução em
`docs/metodologia_cientifica/ext_chuva_fonte_unica_recife_v1.md`.
