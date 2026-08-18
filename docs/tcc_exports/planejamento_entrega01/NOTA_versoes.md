# Do rascunho v1 à v3 — o que mudou e por quê

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


---

## 6. v3 (2026-08-13) — realinhamento do problema central

A v2 ainda descrevia o problema como sendo a classe negativa. Isso estava
errado por dois motivos: o diagnóstico do rótulo já foi feito e já orientou a
estratégia, e foi justamente a combinação de positivos oficialmente registrados
com negativo observado que produziu todos os resultados que hoje existem.

O problema central da v3 é a **heterogeneidade da evidência**. Modelo de terreno
a 10\,m numa região e 30\,m em outra, o que muda a escala em que HAND e TWI são
lidos; evidência documental sem endereço geocodificável; AOIs declaradas que não
coincidem com o recorte urbano de interesse; cenas com cobertura de nuvem;
milímetros de chuva que não transferem entre climatologias; e áreas cujo
mecanismo dominante é movimento de massa. Organizar, geocodificar, conferir
contra o já validado e só então promover a dado de treino — esse é o trabalho.

Consequências no documento:

| Onde | O que mudou |
|---|---|
| Título | "da evidência heterogênea ao serviço de inferência auditável" |
| §I, par. 3 | Estado real herdado da primeira entrega (sem validação operacional em nível de *patch*), três valores de ablação, e a camada orbital descrita pelo que faz bem: semelhança entre áreas vizinhas, coerência territorial, fila de revisão humana; reservada como apoio visual à explicação |
| §I, par. 4 | Problema central trocado da classe negativa para a heterogeneidade |
| §I, par. 5 | Objetivos reordenados: consolidar a evidência vem primeiro |
| Fig. 1 | Coluna 2 passou de "Derivação física" a **Harmonização**; coluna 4 passou de "Produto" a **Serviço** |
| Fig. 2 | Painel (b) trocado: era sobreposição de classes em Recife, agora é a matriz de variável disponível por fonte |
| §II | Novo bloco de abertura sobre harmonização; percurso único em vez de duas frentes |
| §III | E2 passou a ser Harmonização; E5 é Aplicação às regiões brasileiras; E6 é Serviço e explicação |
| Tabela I | Linha 2 passou a se chamar "Harmonização, variáveis e modelos" |
| Tabela II | Linhas do cronograma renomeadas para as etapas novas |

---

## 7. v4 (2026-08-13) — nota a nota sobre a revisão em cima da v3

**Origem**: `planejamento02.pdf`, um rascunho trabalhado em cima da v3 além do
que estava commitado localmente (o `main.tex` commitado tinha sido salvo às
14h18; o rascunho é das 20h19 do mesmo dia). O rascunho reescreveu a Seção I do
zero — em alguns pontos de forma mais elaborada, com anotações próprias
marcando o que ainda incomodava — e deixou notas pontuais na Seção II.

**Decisão editorial, declarada antes da tabela**: boa parte das notas do
rascunho aponta problemas que a v3 **já tinha resolvido** (o parágrafo da
suscetibilidade já traduz HAND/TWI/D-infinity sem jargão; a abertura já liga
a estatística de exposição ao registro ausente numa frase só). Nesses casos a
v4 não adotou a reescrita nova — manteve o texto da v3, que é mais curto e já
passou pela mesma revisão de estilo antes. Só entraram mudanças onde a nota
apontava algo que a v3 de fato ainda não tinha.

| Sua nota (localização) | O que a v4 faz |
|---|---|
| "adicione um conector entre essas duas frases" (frase 1→2 da abertura) | Nada a fazer: a v3 já fundiu as duas frases numa só, com "e atingem primeiro quem não escolheu onde morar:" fazendo a ligação. Mantido. |
| "ache um jeito de deixar mais fluído aqui, interligue essas duas informações" (estatística → território) | Nada a fazer: a v3 já resolve com "concentrada nessas áreas" → "um território" no período seguinte, sem quebra. Mantido. |
| "deixe claro que não é só pq as pessoas n ligam é pq nao existe força e incentivo publico pra ir ajudar essas pessoas perifericas" | **Aplicado.** §I agora diz que o registro nasce de quem consegue reportar *e ser atendido* — "a ausência não é de quem liga, é de poder público que alcance as áreas periféricas." |
| "isso aqui ta cientifico elaborado demais... o que que é hand, o que é D-Infinity etc" | Nada a fazer no corpo: a v3 já traduz os três conceitos em uma frase cada, sem jargão solto. **Aplicado como reforço pequeno**: HAND ganhou o nome por extenso em inglês (*Height Above the Nearest Drainage*) entre parênteses, ideia que veio do seu próprio rascunho alternativo. |
| "ache outra palavra pra evolui" | **Aplicado.** "O plano evolui do artigo anterior" virou "O plano se apoia no artigo anterior" — a v3 tinha esquecido de trocar essa palavra apesar de já ter corrigido o resto do parágrafo. |
| "(os testes... chega com quatro pontos fixos)" | Nada a fazer: é confirmação de conteúdo, não crítica. A v3 já lista cinco achados (não quatro) porque inclui a unidade evento/AOI contra pseudorreplicação, que o rascunho novo tinha deixado de fora. Mantida a versão mais completa. |
| "toda essa frase ta muito grande... elabore menos" / "não gostei de mascarando" | Sem ação necessária: essas notas estão dentro do parágrafo-duplicata (a reescrita alternativa da abertura, com marcas `[cite: 1]`) que a v4 não adotou — o conteúdo já está coberto pela abertura da v3, mais enxuta. |
| "outra palavra pra harmonização, ela não transmite o que eu efetivamente to fazendo, eu to organizando, validando" | Nada a fazer: no rascunho novo o rótulo da Seção II tinha virado só "Harmonização"; a v3 já usa "**Integração e curadoria**" como abertura da seção, que é exatamente o par organizar+validar que você pediu. Mantido — "Harmonização" continua só como nome da Fig. 1/etapa E2, onde uma palavra só cabe melhor. |
| "a gente tem que deixar clara todas essas limitações mas eu n sei se é nessa sessão de materiais e métodos... se alinhe com o documento do meu orientador" | **Não resolvido — decisão sua, não minha.** Não tenho acesso ao documento de expectativas por seção do seu orientador. O parágrafo de limitações (escala 10m/30m, evidência não geocodificável, AOI, nuvem, climatologia, movimento de massa) continua em §II, onde já estava. Se o orientador esperar isso em "Descrição do Projeto", é mover um parágrafo — me diga e eu movo. |
| "[daqui pra frente eu gosto]" | Confirmação, não crítica — nenhuma ação. O trecho seguinte ("Toda fonte passa por uma redução a tabela de pontos...") é idêntico entre o rascunho e a v3. |
| "eu to auditando isso das demais regiões... quero o resultado mais atual" (após "raster de Recife") | **Aplicado, com honestidade sobre o que existe.** A v4 declara que a auditoria bit a bit está feita para Recife e que a equivalente para Curitiba/Petrópolis (mesmo motor de derivação) ainda não foi concluída — não inventei um resultado que você ainda não tem. |
| "[só? inserir todas]" (fontes externas) | **Aplicado.** "Externas são as bases que publicam observação declarada de não-inundação" agora nomeia as duas que o projeto de fato usa: Copernicus EMS e Environment Agency do Reino Unido. Não inclui as fontes do Nível 1 (Sen1Floods11, UFO, GFD) porque essas ainda estão pendentes de download, não integradas — dizer que são "material externo já auditado" seria overclaim. |
| "isso aqui é o que engloba o produto final/serviço, tenho que elaborar mais" | **Aplicado.** O parágrafo de serviço ganhou três frases: o que o *model card* declara, o que "auditável" significa em termos concretos (todo escore remonta ao *gate* que o liberou), o terceiro status do contrato (`region_not_supported`) e o escopo real de hoje (MVP local, só Recife). |
| "eu faço esse ajuste de escopo, diminuindo e transformando todos os dados... dps a gente altera ou une essa parte a sessão do modelo" | **Não é nota sobre o texto — é lembrete seu de tarefa futura.** Não entrou na v4 porque não é conteúdo de manuscrito; se ficar no rascunho do Overleaf, vale apagar antes de compilar de novo. |
| "A Fig.??reúne" (referência quebrada) | Não é conteúda a corrigir por texto — é artefato de compilação (a v4, como a v3, usa `\ref{fig:datasets}`, que resolve normalmente em duas passadas de `pdflatex`). Provável causa: o rascunho no Overleaf rodou só uma passada antes de exportar o PDF. Sem ação nesta revisão; confira ao recompilar. |

**Verificação de espaço feita**: compilei a v3 e a v4 no mesmo ambiente
sandbox (sem os padrões de hifenização de português, mesma limitação já
registrada no README) para isolar o efeito das minhas edições. As duas
saíram com 4 páginas **nesse ambiente limitado** — ou seja, o estouro não é
introduzido pela v4: a v3 já estourava aqui pela mesma razão de hifenização.
A diferença de tamanho entre as duas fontes compiladas é de 558 bytes (~2%),
o que sugere que a v4 deve continuar cabendo nas 3 páginas no Overleaf (onde
a v3 cabia), mas isso não foi confirmado num ambiente com hifenização real —
recompile no Overleaf antes de entregar para ter certeza. Se estourar por
pouco, os dois controles já documentados no README (largura da Fig. 1,
`figsize` da Fig. 2) resolvem sem cortar texto.

**O que ficou de fora, deliberadamente**: os dois parágrafos-duplicata da
Seção I do rascunho (a reescrita alternativa com marcas `[cite: 1]`, incluindo
a expansão "Primeiro/Segundo/Terceiro/Quarto/Por fim" com a palavra
"comprovada") não entraram. Cobrem o mesmo conteúdo que a v3 já cobre, de
forma mais longa, e "comprovada" é mais forte do que os artefatos sustentam —
a mesma regra que baniu "validado"/"confirmado" na v2 (ver §5 acima).

---

## 8. v5 (2026-08-14) — status de E2 e auditoria de terreno atualizado (só estado, não texto novo)

**Gatilho**: commit `e627af3` (13-14/08) fechou de fato o que a v4 já
descrevia como plano (`ds03`-`ds05`, `ter04`) — validei os artefatos
pessoalmente antes de editar (rodei o pipeline, o manifesto bate número a
número, 18 de 19 testes passam). Esta revisão só corrige afirmações que
ficaram desatualizadas pelo trabalho real; não é a mesma coisa que incorporar
achado novo como resultado — isso continua fora do escopo da Entrega 01 (regra
do `CHECKLIST_template.md` §6.1).

| Onde | Antes (v4) | Agora (v5) | Por quê |
|---|---|---|---|
| §II, "Material próprio, material externo" | "auditoria equivalente para Curitiba e Petrópolis ainda não concluída" | Curitiba auditada (Pearson 0,913-0,998 conforme a variável); Petrópolis segue sem ponto rotulado para comparar --- não é a mesma pendência | A afirmação da v4 tratava as duas regiões como igualmente pendentes; não são mais: uma tem prova, a outra é estruturalmente bloqueada por N=0 |
| §II, mesma frase | "Falta a base integrada" | Removido --- a tabela única existe, testada e versionada | Deixou de ser verdade assim que `ds05` rodou |
| §II, mesma frase | Externas citadas só por nome (Copernicus EMS, EA/UK) | Declarado que Sen1Floods11 e UFO também estão integradas ao esquema, hoje sem contribuir ponto (cadeia de terreno global) | Reflete que 6 fontes estão de fato no pipeline, não só 2 --- e diz por que as outras duas ainda não contam, em vez de omitir |
| §III, E2 | Descrita em tempo futuro/plano, sem marca de conclusão | Marcada "(M1, concluída)", com o número real do entregável (33.071 pontos elegíveis ao ajuste fluvial) e o resultado real da evidência (18/19 testes) | É a mesma etapa que E0/E1 já tratam assim; E2 atingiu o mesmo critério de prova |

**O que eu decidi não trazer para o texto agora, e por quê** (para não fazer
essa escolha por você sem avisar):

1. **Os números de Tabela I e da Fig. 2** (25.249 pontos/119 AOIs, contraste
   de HAND 2,95-27,86 m, queda de 0,159) continuam do `mec01` (12/08) — o
   pipeline novo (`mec02`) já tem números maiores e diferentes (33.071 pontos,
   28.684 na planície, coeficientes Firth próprios). Não troquei porque isso
   exige regenerar a Fig. 2 a partir de fonte nova (`fig/make_fig2.py`), é uma
   tarefa própria, não uma frase — fica pendente, registrada.
2. **O achado da mistura de fonte de chuva em Recife** (indicador de fonte
   discrimina mais que a própria chuva) não entrou no corpo do texto. É
   achado novo desta semana, ainda sem correção aplicada (reamostragem
   pendente), e colocá-lo como frase corrida arriscaria ler como resultado
   desta entrega, o que o `CHECKLIST_template.md` §6.1 proíbe explicitiamente
   ("não apresenta resultados como resultados"). Fica registrado no
   `CHECKLIST_template.md` §12 até a reamostragem decidir o que escrever.
3. **A regra do TWI** ("ou todo TWI vem de 30 m, ou TWI sai do pool") está
   escrita no relatório de harmonização mas ainda não está aplicada no
   critério de elegibilidade do `ds05`. Não mudei o texto porque o código
   ainda não mudou — texto e código continuam consistentes entre si.

**Verificação de espaço**: recompilei no mesmo ambiente sandbox (sem
hifenização de português) --- 4 páginas aqui, como a v3 e a v4, mesma
limitação já registrada, não regressão desta edição. Tamanho do `.tex`
cresceu ~220 bytes; segue a mesma pendência já registrada em
`CHECKLIST_template.md` §11.5/11.6 de confirmar 3 páginas no Overleaf.

---

## 9. v6 (2026-08-14) — problema de terreno, referência do Brasil, AOI, DINOv2 na Fig. 1

Pedido seu, direto: generalizar o problema de terreno além de "cota
rebaixada"; somar referência brasileira à de Tellman; justificar por que a
AOI é a unidade independente; nomear o DINOv2 na Fig. 1 com seta até a
explicação. Executei os quatro; as perguntas abertas do mesmo pedido
(curadoria em Descrição ou Materiais e Métodos; Fig. 2 útil ou não; imagem de
interface) estão respondidas na conversa, não no texto — ver resumo abaixo.

| Onde | O que mudou |
|---|---|
| §I, par. 1 | "terreno baixo, drenagem insuficiente" → "relevo que não escoa a água e relevo que cede sob ela", cobrindo enchente e movimento de massa como a mesma vulnerabilidade física; a cadeia causal física→urbano→social ficou explícita ("da hostilidade desse relevo nasce a ocupação urbana precária, e dela, a vulnerabilidade social"), no lugar de "convergem" |
| §I, par. 1 | Referência brasileira somada à de Tellman: IBGE/CEMADEN, "População em áreas de risco no Brasil" (2018) — 8,3 milhões de pessoas, oficial, mesma família institucional das fontes que o projeto já usa (Defesa Civil, SEDEC) |
| Bibliografia | Novo item `ibge2018risco` |
| §II, "Rótulo positivo e hierarquia do negativo" | Adicionada a frase que faltava: a AOI é a unidade independente também porque é a unidade que o contrato de inferência devolve — validar no mesmo grão que se entrega é o que sustenta "auditável" |
| Fig. 1, coluna 2 | "Semelhança entre áreas vizinhas" → "Similaridade via DINOv2 entre áreas" |
| Fig. 1 | Nova seta pontilhada de "Evidência visual" até "Camada de explicação", rotulada "DINOv2 ilustra a resposta" — renderizei e conferi visualmente (`pdftoppm`), sem colisão com "Limite 2" |
| Legenda Fig. 1 | Passou a nomear DINOv2 e a dizer que a evidência chega à explicação como ilustração, nunca como *feature* |

**Verificação de espaço**: 4 páginas no sandbox (mesma causa de sempre,
hifenização). `.tex` cresceu de 25,4 KB para ~26 KB — a lista de pendência de
página/palavra em `CHECKLIST_template.md` §11.5/11.6 só cresce a cada
rodada; vale resolver antes da próxima.

**O que não entrou no texto, porque é diagnóstico pedido, não instrução
direta** (respondido em conversa): posição do parágrafo de limitações
(Descrição vs. Materiais e Métodos), necessidade de detalhar curadoria,
futuro da Fig. 2, e se vale criar imagem de protótipo de interface.

---

## 10. v7 (2026-08-16) — o texto alcança o pipeline: resolução única e chuva de fonte única

Pedido seu: "atualize o artigo de planejamento com o estado atual do
projeto, cheque todas as seções". Não é revisão de estilo — dois eventos
reais no REV-P desde a v6 tornaram cinco trechos factualmente errados:
`90ecb5c` (resolução única de 30 m, reversão declarada de "a resolução segue
o mecanismo") e a correção de fonte de chuva única em Recife (commit
`13fcea2`, mesma sessão). Revisei as quatro seções inteiras contra os
artefatos correntes; só entrou o que uma delas tornou obsoleto.

| Onde | Estava | Passou a estar | Por quê |
|---|---|---|---|
| §I, par. 3 | "a resolução segue o mecanismo" | "na mesma resolução para toda a base — inclusive onde isso custa detalhe que o próprio modelo não usa" | decisão citada foi revertida em `90ecb5c`; manter o texto antigo contradiria o `ext_resolucao_unica_30m_v2.md` |
| §II, "Material próprio, material externo" | "Sen1Floods11 e UFO... sem contribuição ainda por cadeia de terreno global" | Sen1Floods11 contribui ao ajuste desde que a cadeia própria passou a cobri-lo; UFO tem a mesma cadeia, mas segue fora por mecanismo misto, não mais por lacuna de variável | `ter06` derivou 658/661 chips; a exclusão do UFO nunca foi por terreno, e o texto antigo confundia as duas causas |
| §II, mesmo parágrafo | — | Recife citado com "fonte de chuva única após auditoria de confundimento"; tabela única com "64.989 pontos... nenhum deles mais em cadeia de terreno global" | número e status estavam desatualizados (33.071 → 64.989) e a correção de chuva é achado novo, resolvido nesta sessão |
| §II, "O que a caracterização dos dados impõe" | "as bases de rótulo por imagem não têm derivadas direcionais" | removida; a frase agora liga a ausência de chuva multirregião à causa real (nenhuma fonte do ajuste fluvial tem chuva, não a falta de declividade/TWI) | Sen1Floods11/UFO ganharam `slope_deg`/`twi_dinf` em `90ecb5c`; a frase antiga é falsa hoje |
| §II, "Infraestrutura, carga e ajustes de escopo" | "o HAND das fontes externas vem de produto global de 30 m" | "permanece uma fração residual... majoritariamente água permanente no Copernicus EMS... o restante... passou para a cadeia própria" | a limitação generalizada não existe mais; sobra uma fração residual por nodata, não a fonte inteira |
| §III, evidência de E2 | "33.071 pontos elegíveis ao ajuste fluvial... 18 de 19 testes... instável entre ambientes" | "64.989 pontos... quatro fontes... 21 de 22 testes... sensível à versão de biblioteca" | números desatualizados; "instável" trocado por descrição mais precisa da causa (já diagnosticada, não corrigida no código ainda) |
| Fig. 2(b), `make_fig2.py` | Nível 1 (Sen1Floods11+UFO) marcado sem declividade/TWI | marcado com as quatro variáveis de terreno disponíveis | matriz era hardcoded contra um fato que deixou de ser verdade em `90ecb5c`; figura regenerada e reconferida visualmente |

**O que não mudou, verificado e mantido**: Tabela I (positivos/negativos por
fonte, Curitiba 1.045/426/1.471 unidades — números de uma fonte diferente,
não tocada nesta sessão); a hierarquia do negativo; os parágrafos de modelo,
*gates* e critérios; o cronograma; a bibliografia. Não fui atrás de reauditar
números que este ciclo de trabalho não tocou — só o que os dois eventos
citados tornaram obsoleto.

**Contagem de palavras da Seção I**: 573, igual à v6 — a correção do
parágrafo 3 trocou uma frase curta e errada por uma mais longa e certa; não
ajuda nem piora o excesso sobre o limite de 500, que continua em aberto
(`CHECKLIST_template.md` §11.6).

**Verificação de espaço**: 4 páginas no sandbox, mesma causa de sempre
(hifenização ausente aqui). Figura 1 e Figura 2 renderizadas e conferidas
visualmente (`pdftoppm`) — sem colisão, painel (b) da Fig. 2 mostra a matriz
nova corretamente.
