# Cronograma científico e plano de trabalho — TCC Enchentes / REV-P (ago–dez 2026)

**Status**: PLANO_DE_TRABALHO_NAO_CANONICO — documento de planejamento, não é gate nem
substitui `docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md`. Serve para alinhar o
calendário acadêmico fixo da disciplina com o estado científico real do projeto.

**Criado**: 2026-08-04. **Ponto de partida auditado**: HEAD real do REV-P em `c7d7c6f`
(02/08/2026, SUSC-21A), não a narrativa do README.md nem dos guias internos do REV-P/PROJETO —
ambos desatualizados em pontos específicos (ver seção 1).

**Regra que rege este cronograma**: uma tarefa por vez. Nenhuma semana abaixo abre mais de
uma frente de trabalho simultânea. Isso é deliberado — é a mesma regra fixa do projeto
(TCC ENCHENTES) e do REV-P, e é também a causa raiz de por que a série de diagnósticos
SUSC-20 chegou a resultados tão limpos: cada rodada testou uma coisa e documentou o
resultado antes de abrir a próxima.

---

## 1. Onde o projeto realmente está (auditado no código, não em documentação parada)

Três documentos descrevem o projeto em momentos diferentes e **não concordam entre si**.
Isso não é um erro a corrigir agora — é só preciso deixar explícito qual é a fonte válida
para este cronograma:

| Fonte | O que diz | Validade |
|---|---|---|
| Guia interno do PROJETO (v1fn, 11/05/2026) | Descreve uma frente de *patch grounding*/binding TIF (59 patches, gates B1–B7 todos bloqueados) | Frente lateral e antiga. Não é o estado do SUSC. |
| Guia interno do REV-P | Descreve "Sentinel-first + DINOv2 with registers" como rota principal validada | Superado em 01–02/08/2026: a linha DINO/patch estático foi **fechada definitivamente** como candidata a feature ou score (SUSC-22/v1r9). DINOv2 permanece só como evidência visual auxiliar, nunca causal — conforme a regra fixa do projeto. |
| `README.md` do REV-P | Descreve modo "review-only", sem classificador supervisionado, sem ground truth declarado | Descreve a narrativa pública consolidada em torno do Protocolo C/DINOv2. Não reflete a frente causal SUSC (Firth penalizado com eventos reais SEDEC/ANA/Diário Oficial), que é supervisionada por desenho e é a rota mais madura do projeto (`v12`, Recife, LOO-AUC=0,6781). Precisa de reconciliação editorial antes da entrega final — sinalizado como tarefa, não resolvido aqui. |
| `docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md` (atualizado 02/08/2026) | Estado de execução real, fase a fase, com número/arquivo conferível em cada entrada | **Fonte válida para este cronograma.** |

**Síntese do estado real (02/08/2026)**:

- **Recife** — rota mais madura. Modelo causal Firth (`v12`, n=278: 154 pos/124 neg, 6
  features físico-hidrológicas, LOO-AUC=0,6781), motor de inferência local auditado
  (SUSC-20D) e API MVP com geoprocessamento sob demanda (SUSC-20E/20F) já entregues e
  testados ponta a ponta contra os 278 rótulos reais. DINO testado como feature (Fase 1,
  teste A/B contra o `v12`) e **descartado** — fica só como evidência visual, nunca soma
  ao score.
- **Curitiba** — em diagnóstico avançado, não fechado. Modelo causal Firth existe
  (SUSC-20N, LOO-AUC=0,6459 em CV embaralhada), mas colapsa em holdout temporal real
  (2023–25 → 2026 nunca visto): AUC cai para 0,5246. Sete diagnósticos independentes
  (SUSC-20O a 20T) descartaram vazamento espacial, sazonalidade, ruído de amostra, deriva
  administrativa e a hipótese ENOS/El Niño como explicação. Não-linearidade real e
  generalizável foi confirmada (SUSC-20U a 21A: GBM chega a AUC=0,5888; a versão com
  restrição monotônica causal, SUSC-21A, é o único modelo não-linear com 100% de
  conformidade causal por construção). **Nenhuma dessas linhas resolve o colapso de 2026**
  — ele permanece uma propriedade real do dado, ainda sem causa identificada. Rota
  primária declarada continua sendo o modelo linear/Firth interpretável.
- **Petrópolis** — bloqueado. Mistura enchente/deslizamento não separada nas fontes;
  status honesto = "dados insuficientes para inferência", não avança para modelo até essa
  separação ter um produto oficial de referência (DRM-RJ).
- **Linha DINO/patch estático (Fase 1b/1c)** — encerrada de vez em 01–02/08/2026 (três
  tentativas independentes — A/B com pseudorreplicação, refinamento de evidência nulo em
  23 e 52 patches, Clay bloqueado por falta de instante de aquisição — convergem na mesma
  causa estrutural: patch estático de composição multi-mês não carrega assinatura de
  evento pontual). Não é mais revisitada como candidata a feature.

---

## 2. Objetivo final do projeto (o que este cronograma serve para entregar)

Por regra fixa do projeto: um modelo de suscetibilidade urbana a enchentes
**físico-hidrológico como base causal**, com evidência orbital (Sentinel/DINOv2) **apenas
auxiliar**, nunca causal e nunca virando feature/score. O modelo não deve "descobrir"
enchente — deve refletir relação física já conhecida (daí a prioridade por
interpretabilidade sobre performance, e por que a rota primária de Curitiba continua
linear/Firth mesmo com o GBM tendo AUC maior).

A entrega final do TCC precisa comunicar isso com honestidade científica: Recife como
estudo de caso maduro e auditável ponta a ponta; Curitiba como estudo de caso de
diagnóstico rigoroso de um limite real (colapso temporal não resolvido, documentado como
achado, não como falha escondida); Petrópolis como estudo de caso de bloqueio
metodológico documentado (mistura de fenômenos sem fonte oficial de separação).

---

## 3. Cronograma consolidado (datas fixas da disciplina + uma tarefa por vez)

### Fase de Planejamento (04/08 – 29/08)

| Data | Tarefa única da semana | Entregável |
|---|---|---|
| 04/08 (hoje) | Formação de grupo/escopo já dado. Consolidar por escrito a síntese da seção 1 deste documento como ponto de partida oficial do planejamento — decidir explicitamente que fonte (PLANO_ACAO, não documentos internos antigos) rege o TCC. | Síntese de estado assinada (este documento) |
| 11/08 | Decisão formal sobre Petrópolis: excluir do modelo quantitativo final e documentar como estudo de caso de bloqueio, **ou** tentar 1 fonte externa de separação de fenômeno (DRM-RJ) se o prazo permitir. Uma decisão, não uma investigação nova em paralelo. | Parágrafo de decisão + justificativa (mesmo padrão dos outros registros do PLANO_ACAO) |
| 18/08 | Redigir a estrutura do Documento de Planejamento do Projeto: introdução, motivação, pergunta de pesquisa, metodologia consolidada das 3 regiões (com os números reais da seção 1), cronograma de escrita (baseado neste documento), limitações já conhecidas. | Rascunho do Documento de Planejamento |
| 25/08 | Revisão do rascunho; ajuste de escopo se a decisão de Petrópolis (11/08) mudar alguma seção. | Documento de Planejamento pronto para entrega |
| **29/08 (sáb)** | — | **Entrega 01: Documento de Planejamento do Projeto [TDE 12h]** |

### Fase de Execução (01/09 – 27/10)

| Data | Tarefa única da semana | Entregável |
|---|---|---|
| 01/09 | Congelar formalmente a rota primária de Curitiba como linear/Firth interpretável. Documentar GBM monotônico (SUSC-21A) e a série 20U–21A como achado de apêndice, não como candidato a produção. Nenhum novo diagnóstico de não-linearidade além do já feito (a série está saturada — 15 rodadas, retornos decrescentes). | Nota de encerramento da série SUSC-20/21 (não-linearidade) |
| 08/09 | *Feriado local (Nossa Senhora da Luz dos Pinhais) — sem tarefa nova.* | — |
| 15/09 | Consolidar Fase 4 de Curitiba: gerar os mesmos artefatos públicos que Recife já tem (model card, relatório final, tabelas/figuras em `outputs_public/`), com a limitação do colapso 2026 declarada explicitamente — não escondida. | `outputs_public` de Curitiba equivalente ao de Recife |
| 22/09 | Executar a decisão de Petrópolis tomada em 11/08 (documentar bloqueio final **ou** rodar a tentativa de fonte externa já decidida). Uma tarefa, sem abrir frente nova. | Status final de Petrópolis registrado |
| 29/09 | Consolidação final de `outputs_public` das 3 regiões: tabelas/figuras finais, declaração de ausência de modelo operacional atualizada para refletir Recife (que tem motor+API) vs. Curitiba (limitação documentada) vs. Petrópolis (bloqueado). | `outputs_public` consolidado, 3 regiões coerentes entre si |
| 06/10 | Auditoria de reprodutibilidade: `pytest tests -q` completo, `git status` limpo, checklist do REV-P (sem dados pesados, sem paths privados, sem overclaiming), congelar a versão do pipeline que vai para o artigo. | Checklist de auditoria preenchido + tag/commit de congelamento |
| 13/10 | *Recesso acadêmico e administrativo — sem tarefa nova.* | — |
| 20/10 | Rascunho das seções de Metodologia e Resultados do artigo científico, usando só os artefatos já congelados em 06/10 (nenhum experimento novo nesta fase). | Rascunho Metodologia + Resultados |
| 27/10 | Revisão do rascunho de Metodologia/Resultados; fechamento para a Entrega 02. | Metodologia + Resultados prontos |
| **31/10 (sáb)** | — | **Entrega 02: Codificação, Metodologia Implementada e Resultados [TDE 12h] (online)** |

### Fase de Comunicação (03/11 – 17/11)

| Data | Tarefa única da semana | Entregável |
|---|---|---|
| 03/11 | Redação completa do artigo científico (introdução, discussão, limitações, conclusão) a partir da Metodologia/Resultados já fechados; design do pôster em paralelo só depois do texto do artigo estar estável. | Artigo científico completo |
| **09/11** | — | **Entrega 03: Artigo Científico e Pôster (online)** |
| 10/11 | Revisão por pares (atividade da disciplina); ajustes finais de apresentação a partir do feedback recebido. | Apresentação ajustada |
| **17/11** | — | **Apresentação dos Trabalhos (Poster Session), presencial** |

### Contingência (não planejada como trabalho novo)

| Data | Natureza |
|---|---|
| 24/11 | Recuperação Parcial — buffer, conforme necessidade individual |
| 01/12 | Recuperação Estendida — buffer, conforme necessidade individual |

---

## 4. Marcos próprios (distintos do calendário genérico da disciplina)

Mesmo critério de "feito" já usado no `PLANO_ACAO_produto_v1.md`: um marco só fecha quando
produz um artefato conferível, nunca por estar "provavelmente pronto".

| Marco | Descrição | Alvo | Depende de |
|---|---|---|---|
| **M1** | Encerramento formal da série de diagnósticos de não-linearidade de Curitiba (SUSC-20U–21A) | 01/09 | Decisão humana de não abrir mais rodadas nessa série |
| **M2** | Decisão e execução do status final de Petrópolis | 11/08 (decisão) → 22/09 (execução) | Fonte oficial DRM-RJ, se aplicável |
| **M3** | `outputs_public` de Curitiba paritário ao de Recife (model card + relatório + figuras) | 15/09 | M1 |
| **M4** | Congelamento do pipeline (nenhuma mudança de metodologia depois desta data — só documentação/escrita) | 06/10 | M2 + M3 |
| **M5** | `outputs_public` final das 3 regiões, coerente e sem overclaiming | 29/09 | M2 + M3 |
| **M6** | Metodologia + Resultados do artigo fechados | 27/10 | M4 |
| **M7** | Artigo científico + pôster prontos | 09/11 | M6 |

---

## 5. Regras de proteção durante a execução (não negociáveis)

Repetindo o que já rege o projeto, porque a fase de execução (set–out) é onde é mais fácil
violar sem perceber: nunca usar score, threshold, proxy ou variável derivada do label como
feature; nunca misturar validação científica com otimização de performance (a rota
primária de Curitiba continua linear mesmo com o GBM tendo AUC maior — isso é intencional,
não um resultado pendente de "corrigir"); nunca tratar DINOv2/Sentinel como preditor,
mesmo que a tentação apareça ao escrever a seção de resultados; documentar cada limitação
como achado, não escondê-la para a banca. Nenhuma fase deste cronograma promete que um
problema científico em aberto (o colapso de 2026 em Curitiba, a mistura de fenômenos em
Petrópolis) será resolvido dentro do prazo — o prazo é da entrega acadêmica, não da
ciência.
