# Do rascunho v1 para a v2 — o que mudou e por quê

**Data**: 2026-08-11
**Escopo**: reescrita do `main.tex` do Documento de Planejamento (Entrega 01).
Nenhum modelo foi rodado, nenhum gate alterado, nenhum dado promovido. As
mudanças são de texto, de figura e de correção factual contra os artefatos.

---

## 1. Correção factual — o ponto mais importante

A v1 escrevia que a ativação EMSR720 do Copernicus EMS era "a informação cuja
ausência mantinha Petrópolis bloqueada" e que, com ela, "o bloqueio deixou de ser
estrutural para ser de execução".

Isso não se sustenta contra `ext_balanco_e_lacunas_por_regiao_v1.md`, que é
explícito em três pontos:

1. O EMSR720 é do **Rio Grande do Sul**. Fornece negativo formal por observação
   **apenas para aquela AOI**, e entra no projeto como quarta região.
2. O que o dado externo dá a Petrópolis é **um argumento, não um dado**: mostra
   que a separação enchente/movimento de massa é atributo de origem. Conclusão
   não é inventário.
3. `C4_BLOCKED_NO_FORMAL_NEGATIVES` **permanece aberto** para Recife, Curitiba e
   Petrópolis.

A v2 reescreve essa passagem como hierarquia declarada de negativo — observação,
exclusão qualificada, ausência de registro — e afirma, em texto e em tabela, que
nenhuma região brasileira tem hoje o nível de observação. Some junto o tom de
"virada do negativo": não há virada a narrar, há um nível de evidência que
existe fora do Brasil e ainda não existe dentro.

Ressalva registrada, porque muda o que se pode escrever mais adiante:
`ext_criterios_de_acerto_v1.md` §6 corrige a leitura de que Petrópolis estaria
bloqueada para tudo. **Predizer** em Petrópolis é possível hoje (as features são
globais); **validar** é que exige o inventário local. A v2 não afirma nem uma
coisa nem outra sobre Petrópolis — o documento é de planejamento —, mas o
objetivo de transferência entre classes de relevo existe justamente para
sustentar essa distinção na Entrega 02.

---

## 2. Nota a nota — onde cada crítica sua foi endereçada

| Sua nota | O que a v2 faz |
|---|---|
| "as pessoas que lerão não têm o entendimento hidro-geográfico; não dá pra soltar D-infinity sem elaborar" | O parágrafo 2 da Seção I passou a existir só para isso: enuncia o fenômeno ("chega mais água do que o ponto escoa ou armazena"), decompõe em três grandezas, e explica HAND como "a lâmina que o rio precisa ganhar para chegar ali" e D-infinity como repartir o fluxo entre direções contínuas em vez de forçá-lo a uma das oito células vizinhas. Nenhum conceito aparece sem tradução física. |
| "quero começar de um jeito original e impactante, que exale o peso da vulnerabilidade urbana e social" | Abertura reescrita: "A chuva cai sem escolher; a água, depois de cair, escolhe" → o ponto mais baixo da cidade brasileira → quem não teve escolha de onde morar → a segunda desigualdade, a de registro → "Bairro sem chamado não é bairro seco." O peso vem da estrutura do argumento, não de adjetivo. |
| "diz no texto que nossos problemas são os que nós não temos mais" | Corrigido na Seção 1 acima. O portão de negativo aparece como aberto, não como resolvido. |
| "não gosto do tom da virada do negativo, como se fosse de entendimento geral" | O termo saiu. O negativo entra como hierarquia de três níveis com definição explícita de cada um, na Seção II, onde é detalhe técnico — não como reviravolta narrativa na Seção I. |
| "o DINO está inútil e congelado; deixar só citado, e falar dos resultados de ablação junto" | Uma frase na Seção I: rota de *embeddings* auto-supervisionados, testada para dar generalização entre cidades, encerrada por três resultados nulos consecutivos — colada ao parágrafo da ablação, como você pediu. Na Figura 1 ela aparece como ramo tracejado com a marca "nunca vira *feature*". Some da Tabela I. |
| "na introdução elaborar que usamos reports reais para os positivos, e regiões externas buscando métricas para implementar nas regiões brasileiras, já que o Brasil peca em dados" | O parágrafo 1 nomeia a origem administrativa do positivo (Defesa Civil, 156, decreto) e por que ela é frágil; o parágrafo 4 põe "constituir conjuntos de treino com negativo observado, **onde essa observação exista**" e "verificar se a relação terreno–inundação transfere entre classes de relevo, **condição para aplicá-la a regiões sem inventário local**". É esse par que justifica cientificamente usar Inglaterra e CEMS para chegar ao Brasil. |
| "não quero ditar os erros discursivamente, só sintetizar resultados e por que mudei de rumo" | Não há relato de percurso. Há três números de ablação, uma frase de encerramento da rota visual, e o problema que restou. Nenhuma menção a tentativa, sessão ou cronologia interna. |
| "os objetivos não são necessariamente só esses" | De quatro para cinco, e dois deles são novos: medir o efeito da definição de negativo **como resultado**, e verificar transferência entre classes de relevo. Os dois vêm de trabalho já feito (`mod-neg-01`, `mod-serra-01`) e não estavam representados. |
| "quero foco no treino nos materiais e métodos" | Seção II reorganizada em seis blocos, e os quatro primeiros são sobre treino: curadoria do que entra, hierarquia do rótulo, gates de admissão e critérios fixados antes de rodar, e o que a caracterização dos dados já impõe. Produto e infraestrutura ficaram por último. |
| "preciso de curadoria real do que está na AOI certa, o que uso e o que abandono" | O documento declara o critério de entrada em três testes (estar na AOI declarada da fonte, ter as variáveis computadas na mesma cadeia, ter unidade de agrupamento identificável) e diz que o resto fica como proveniência. **A curadoria em si não foi feita** — é a pendência da Seção 3 abaixo. |
| "ver se vale escrever em inglês" | Mantido em português, por sua escolha. O argumento do documento é sobre lacuna de dado e pesquisa no Brasil, com fontes nacionais; em inglês ele perderia coerência retórica. Os termos técnicos ficam em itálico, sem tradução forçada. |
| "plano versátil ou concreto e datado?" | Por sua escolha, o plano é fechado por **evidência**: cada etapa declara o arquivo conferível que a encerra, e as datas aparecem só nos *checkpoints* da disciplina. A Seção III abre dizendo isso e a Seção IV repete: "o que muda de posição é a etapa, nunca o critério de evidência que a encerra". |

---

## 3. Pendência aberta — curadoria de AOI e de conjunto

Esta é a única nota sua que o documento **declara** mas não **executa**, e é
deliberado: é tarefa própria, não texto.

O que precisa ser produzido, uma tabela só, antes da Entrega 02:

- uma linha por ponto, com `fonte`, `AOI`, `nivel_negativo`
  (`observado` | `exclusao` | `ausencia`), `unidade_agrupamento`
  (`evento` ou `AOI`), `classe_relevo` (`serra` | `planicie`) e cobertura de
  cada uma das seis variáveis;
- contagem por fonte do que **entra** e do que **fica de fora**, com o motivo
  do descarte nomeado (fora da AOI declarada, variável faltando, sem unidade de
  agrupamento);
- verificação de que nenhum ponto aparece em duas fontes.

É o entregável de E2 no documento. Sem ele, "temos mais de 200 disso e 200
daquilo" continua sendo volume, não amostra.

---

## 4. A pergunta para o professor, formulada

Você registrou a dúvida de se a estratégia de planejamento é válida quando o
resultado de um teste pode mudar a rota — como aconteceu com os *embeddings*.
Vale perguntar assim:

> O documento de planejamento define cada etapa pelo critério de evidência que
> a encerra, e não por uma data de conclusão, mantendo as datas fixas apenas nos
> *checkpoints* da disciplina. A justificativa é que, em pesquisa experimental,
> um resultado negativo encerra uma frente e abre outra — foi o que aconteceu
> com a linha de *embeddings* deste projeto, encerrada por três resultados nulos
> consecutivos. Cada etapa de risco já declara sua rota alternativa no próprio
> documento. Essa forma atende ao que a disciplina espera de "etapas e marcos
> físicos", ou vocês esperam datas de conclusão por etapa, mesmo sabendo que
> algumas serão revistas?

Se a resposta for "datas por etapa", o ajuste é pequeno: a Tabela II já tem a
distribuição por quinzena, e basta mover as datas do texto de apoio para dentro
de cada bloco E0–E7.

---

## 5. O que a v2 deliberadamente não faz

1. Não apresenta resultado como resultado. Os números de `mod-neg-01` e
   `mod-serra-01` aparecem como **caracterização de dados** — o que a amostra
   impõe ao desenho —, nunca como achado desta entrega.
2. Não promete resolver o colapso de 2026 em Curitiba. Ele entra como limite
   conhecido.
3. Não usa "validado", "confirmado" ou "ground truth" fora do que os artefatos
   sustentam.
4. Não trata a assimetria entre regiões como falha. Ela é condição documentada,
   e agora está visível na Figura 2(a).
5. Não afirma nada sobre movimento de massa. O modelo de encosta usou apenas as
   classes de inundação e de não-inundação observada.
