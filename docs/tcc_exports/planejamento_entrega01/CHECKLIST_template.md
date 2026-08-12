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
| 8.1 | Correção factual sobre Petrópolis e sobre o negativo brasileiro | `NOTA_v1_para_v2.md` §1 |
| 8.2 | Seção I ganhou um parágrafo inteiro de contextualização física (HAND, TWI, D-infinity em linguagem acessível) | §I, parágrafo 2 |
| 8.3 | Objetivos passaram de quatro para cinco | §I, parágrafo 4 |
| 8.4 | Seção II reorganizada com foco em treino; hierarquia do negativo declarada em três níveis | §II, blocos 1–4 |
| 8.5 | Figura 2 refeita: composição por conjunto, sobreposição de classes, efeito da definição de negativo, régua de HAND por relevo | `fig/make_fig2.py` |
| 8.6 | Etapas fechadas por evidência, não por data; E4 (holdout temporal) é nova | §III |
| 8.7 | Referência nova: Tarboton (1997), fonte primária do D-infinity | bibliografia, item 4 |
| 8.8 | Pendência declarada: curadoria de AOI/conjunto ainda não executada | `NOTA_v1_para_v2.md` §3 |
