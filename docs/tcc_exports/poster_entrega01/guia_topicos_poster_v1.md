# Guia de Tópicos do Pôster — TCC Vulnerabilidade Urbana a Enchentes

**Status**: rascunho de conteúdo, não é o pôster final. Uso: base para você desenvolver o texto
e escolher a forma de falar. Cruza (1) a estrutura obrigatória observada nos 3 pôsteres-exemplo
dados pelos professores e (2) o estado real do projeto em 2026-08-12 (fonte:
`REV-P/docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md` + auditoria direta do git, não o
documentos internos desatualizados).

Cada bloco abaixo dá o tópico + o que ele precisa comunicar. O texto final do pôster deve ser
bem mais curto que estas descrições — os 3 exemplos usam frases telegráficas de 1-2 linhas, não
parágrafos. Trate isto como o material-fonte, não como o texto a colar.

---

## 0. O que os 3 exemplos exigem — estrutura obrigatória

Os três pôsteres (todos do curso de Ciência da Computação) seguem exatamente o mesmo esqueleto,
o que sugere que é um template fixo do curso, não escolha de cada grupo:

- **Cabeçalho fixo**: logo da instituição + selo do curso (canto superior esquerdo), título do
  trabalho em caixa colorida (centro), nomes dos autores em lista (canto superior direito).
- **Exatamente 3 painéis de conteúdo** lado a lado, mesma largura, fundo branco sobre fundo
  colorido escuro. Nos três exemplos os painéis são variações de: **Contextualização/Proposta →
  Método(s)/Resultados → Conclusão**. Um dos exemplos separa Método e Resultados em painéis
  distintos (4 blocos no total) — ou seja, 3-4 painéis é a faixa aceitável, não um número rígido.
  **Confirme com os professores se o seu grupo segue este mesmo template** (cores, logos, número
  de painéis) ou se é livre — os 3 exemplos usam o mesmo padrão visual PUCPR, o que indica
  template obrigatório.
- **Texto em fragmentos curtos**, não frases completas longas — cada afirmação é uma linha ou
  duas. Números vêm em tabela ou gráfico, nunca só narrados em texto corrido.
- **Toda tabela/gráfico tem legenda curta embaixo** (ex.: "Table 1: Resultados de Dice e IoU por
  dataset e sequência", "Falling4U Dataset").
- **Pelo menos um painel é dominado por imagem** (arquitetura do modelo, exemplos de entrada/
  saída, gráficos de resultado) — nenhum dos três é só texto.
- **A pergunta de pesquisa aparece explícita e em destaque** na contextualização (não fica
  implícita) — os 3 exemplos escrevem a pergunta como frase separada, geralmente a última do
  bloco de contexto.

---

## 1. Cabeçalho

- **Título**: precisa comunicar o quê (vulnerabilidade a enchente), como (modelo causal
  físico-hidrológico) e onde (cidades brasileiras) em uma linha. Sugestão de rascunho, ajuste
  livremente: *"Suscetibilidade Causal a Enchentes em Áreas Urbanas Brasileiras: um Modelo
  Físico-Hidrológico Interpretável"*. Evite qualquer palavra que sugira que o projeto "descobre"
  enchente a partir de imagem — o rótulo já é físico-hidrológico por definição do projeto, isso é
  ponto fixo, não achado.
- **Autores + curso/instituição**: conforme exigido pelo template.

---

## 2. Painel 1 — Contextualização

- **Abertura pelo peso social, sem adjetivo inflado.** O peso vem da estrutura do argumento:
  enchente urbana no Brasil atinge desproporcionalmente população em áreas de risco conhecido,
  e a ausência de modelos causais e interpretáveis limita a capacidade de antecipar risco de
  forma defensável (em vez de reativa). Não abra com estatística de impacto genérica solta —
  ancore na lacuna que o projeto ataca.
- **A lacuna real**: não é falta de sensor ou de imagem de satélite — é falta de **rótulo formal
  de enchente/não-enchente** com base física conhecida. A maioria dos dados brasileiros
  disponíveis (boletins de ocorrência, queixa cidadã) mistura fenômenos (enchente e
  deslizamento) e não tem um "negativo" com a mesma qualidade do "positivo" (data e local sem
  evento também precisa de evidência, não pode ser arbitrária).
- **Por que não é um problema de "descobrir padrão em imagem de satélite"**: o projeto define o
  rótulo antes de rodar qualquer modelo — é físico-hidrológico-urbano, fixo. O papel de imagens
  orbitais (Sentinel/CBERS) é auxiliar/evidência visual, nunca variável causal ou score. Essa é
  uma decisão metodológica deliberada, útil para explicar por que o projeto não usa abordagens
  de deep learning direto sobre imagem como caminho principal.
- **Pergunta de pesquisa central** (adapte a redação, mas mantenha os 3 elementos): *"Um
  conjunto pequeno de variáveis físico-hidrológicas causais explica suscetibilidade a enchente de
  forma interpretável, e esse modelo se sustenta ao ser testado em cidades brasileiras com
  geomorfologias distintas (planície costeira, relevo serrano, urbano de contraste) e ao longo do
  tempo?"*

---

## 3. Painel 2 — Método

- **O rótulo é físico-hidrológico-urbano e é ponto de partida, não resultado.** Isso é regra fixa
  do projeto — vale registrar explicitamente no pôster, porque diferencia o trabalho de
  abordagens que "descobrem" enchente via padrão de imagem.
- **As variáveis causais (base física, traduza cada uma — a banca não tem formação
  hidro-geográfica)**:
  - *elevação* e *declividade*: relevo bruto do terreno.
  - **HAND** (*Height Above Nearest Drainage*): a lâmina que o terreno precisa ganhar em altura
    para alcançar a drenagem mais próxima — quanto menor o HAND, mais perto fisicamente a água
    já está de alcançar aquele ponto.
  - **TWI** (*Topographic Wetness Index*, índice topográfico de umidade): combina declividade e
    área de contribuição a montante para indicar onde a água tende a se acumular por gravidade.
  - **D-infinity**: o algoritmo usado para calcular HAND/TWI — reparte o fluxo de água entre
    direções contínuas do terreno em vez de forçá-lo a uma única das oito células vizinhas
    (método mais antigo e mais grosseiro).
  - *chuva máxima em 24h* e *índice de decaimento de chuva* (fonte: CHIRPS, estimativa de
    precipitação por satélite): intensidade e persistência do evento de chuva.
- **Papel do orbital (Sentinel/CBERS)**: evidência visual auxiliar (mostrar o patch, dar contexto
  territorial) — nunca entra como feature do modelo nem define o rótulo. Vale uma frase
  explícita nesse sentido: protege contra a leitura de que o projeto é "deep learning em
  imagem de satélite".
- **Modelo estatístico**: regressão logística com **penalização de Firth** — necessária porque
  eventos reais de enchente são raros no dado disponível, o que causa separação quase-perfeita
  entre classes em regressão logística comum (o modelo "explode" sem essa correção).
  Interpretabilidade (coeficiente com sinal e intervalo de confiança por variável) é prioridade
  de desenho, não meta de performance — coerente com a regra do projeto de não misturar
  validação científica com otimização de métrica.
- **Validação honesta**: *leave-one-out* (cada ponto testado fora do próprio treino, uma vez) e
  *GroupKFold* (impede vazamento — pontos do mesmo grupo/bairro nunca aparecem em treino e teste
  ao mesmo tempo). Regra de tamanho mínimo de amostra por variável (**EPV ≥ 10**, eventos por
  variável) para não sobreajustar um modelo com poucos parâmetros.
- **Rota de imagem testada e encerrada (uma frase, sem narrar tentativas)**: representações de
  imagem de satélite (DINOv2) foram testadas como possível variável adicional ao modelo causal e
  descartadas — patch estático (composição de vários meses) não carrega assinatura de evento
  pontual de enchente. Permanece só como evidência visual auxiliar na interface, nunca como
  variável do score.

---

## 4. Painel 3 — Resultados por região (mostrar como estão, inclusive o que está em aberto)

Você pediu para listar tudo mesmo que ainda não esteja corrigido, porque vai resolver antes da
apresentação — por isso cada bloco abaixo já vem com o que responder se perguntarem sobre o que
está aberto.

### Recife — rota mais madura
- Modelo causal Firth com **6 variáveis físicas**, **n=278** pontos (154 positivos / 124
  negativos), eventos reais (SEDEC/ANA/Diário Oficial).
- **LOO-AUC = 0,6781** — dentro da faixa esperada para suscetibilidade espacial com validação
  agrupada (0,70–0,88 seria o teto de referência usado internamente; abaixo disso ainda é
  informativo, acima de 0,95 seria sinal de vazamento).
- Coeficientes com sinal fisicamente coerente (HAND reduz risco, TWI aumenta, etc. — extrair os
  sinais exatos de `primaria_v12_firth_multivariate_coefs.csv` na hora de montar o texto final).
- Motor de inferência local e API MVP já auditados ponta a ponta contra os 278 rótulos reais
  (não é só treino — a saída do motor foi conferida contra o que o modelo já publicou).
- **Se perguntarem "e o negativo, é confiável?"**: resposta honesta — o negativo de Recife (como
  o de Curitiba) ainda não é um negativo *formal* (baseado em evidência observacional
  independente tipo mapa de inundação), é data sem registro de queixa. Essa é justamente a
  lacuna que a frente externa (Reino Unido/CEMS) foi aberta para atacar.

### Curitiba — modelo existe, colapsa em teste temporal real de 2026 (problema aberto, não resolvido)
- Modelo causal Firth (mesma base física) com **LOO-AUC = 0,6459** em validação embaralhada —
  parece funcionar.
- **Mas em teste temporal real (treino 2023–2025, teste 2026 nunca visto): AUC cai para
  0,5246** — nível de acaso.
- **Sete diagnósticos já descartaram como causa**: vazamento espacial (GroupKFold por 73 bairros
  deu o mesmo resultado), sazonalidade (holdout casado por estação deu o mesmo), ruído de
  amostra (intervalo de confiança exclui o valor original — a queda é real, não sorte), deriva
  administrativa visível (metadado estável ano a ano), hipótese climática ENOS/El Niño (índice
  ONI real não bate com o padrão esperado), lançamento de app municipal (colapso não se
  concentra depois do lançamento), redesenho de amostragem negativa (PU-learning testado, sem
  diferença).
- **O que resta como explicação real**: em 2023–2025 as duas variáveis de chuva tinham relação
  forte e consistente com o rótulo (p<0,01); **em 2026 essa relação cai a zero** — é uma mudança
  real na relação chuva↔queixa específica de 2026, causa física ainda não identificada com o
  dado atual.
- **Achado complementar honesto**: existe não-linearidade genuína nos dados (modelos não-lineares
  ficam consistentemente acima do modelo linear, inclusive em 2026), mas isso amortece a queda,
  não resolve — e um modelo não-linear sacrifica a interpretabilidade que é prioridade do
  projeto. Rota primária declarada continua sendo o modelo linear interpretável.
- **Se perguntarem "então o método não funciona?"**: resposta — o método funciona (a mesma base
  física generaliza bem em 2024 e 2025, testado por *walk-forward*); o que falha é uma
  propriedade real e ainda não explicada do ano de 2026 especificamente, isolada por eliminação
  rigorosa de 7 causas alternativas. Isso é resultado metodológico válido por si só, não uma
  falha escondida.

### Petrópolis — bloqueado
- Status honesto: **"dados insuficientes para inferência"**, não "modelo ruim".
- Causa: a fonte de eventos mistura enchente e deslizamento sem separar o mecanismo na origem —
  rodar um modelo de suscetibilidade a enchente exige poder distinguir os dois fenômenos
  primeiro.
- Distinção importante para explicar: **predizer** já é tecnicamente possível hoje (as mesmas 6
  variáveis físicas existem para Petrópolis); **validar** é que está bloqueado, porque exige
  inventário de referência (CPRM/DRM-RJ) que ainda não foi incorporado.

### (Opcional, evidência de apoio) Piloto Reino Unido
- Frente aberta para atacar a lacuna do "negativo formal" (não é Petrópolis — cuidado com um erro
  fácil de cometer: a ativação Copernicus EMS **EMSR720 é do Rio Grande do Sul**, não resolve
  Petrópolis).
- Piloto real na Inglaterra: **7.476 pontos, 201 eventos independentes, AUC agrupada = 0,7927**,
  negativo por exclusão qualificada com evidência real (buffer de 400m + pareamento por uso do
  solo, não data arbitrária).
- **Achado mais forte para o argumento causal do projeto**: comparando duas definições diferentes
  de negativo (por exclusão vs. por observação direta), **HAND é a única variável que se mantém
  praticamente igual nas duas definições** (+4,89 vs. +4,65); elevação e declividade mudam bastante
  entre as duas (+34 vs. +24; +0,51 vs. +2,68). Leitura: HAND carrega a física do fenômeno,
  elevação/declividade carregam mais o efeito da região específica — evidência a favor de que a
  base causal físico-hidrológica (e não a região) é o que sustenta o modelo.
- **Ressalva a não esconder se usar este resultado**: é validação em outro país, com hidrologia
  e regime de chuva diferentes do Brasil — entra como evidência de apoio ao raciocínio causal,
  não como prova de que o modelo brasileiro está validado.

---

## 5. Painel de Conclusão / próximos passos

- **Síntese sem tom de "resolvido"**: a base físico-hidrológica funciona e é interpretável onde
  há dado suficiente e negativo defensável (Recife). O desafio real não está na escolha das
  variáveis — está (1) na generalização temporal em Curitiba, isolada mas ainda não explicada
  causalmente, e (2) na separação de mecanismo em Petrópolis.
- **O que já está em andamento para resolver os pontos acima** (cite sem prometer data como
  resultado — só como calendário institucional): frente externa (Reino Unido/CEMS) ampliando o
  treino para aprender melhor a fronteira sim/não; próximos marcos internos do projeto
  (documentados em `REV-P/docs/cronograma_cientifico_planejamento_2026.md`) tratam o fechamento
  do diagnóstico de Curitiba e a decisão sobre Petrópolis como prioridades imediatas.
- **Datas fixas da disciplina** (não são resultado do projeto, são prazo externo — pode aparecer
  como rodapé/cronograma, não como conclusão científica): Entrega 01 Planejamento 29/08/2026,
  Entrega 02 Metodologia+Resultados 31/10/2026, Entrega 03 Artigo+Pôster 09/11/2026,
  Apresentação 17/11/2026.

---

## 6. Sugestões de imagem

Todos os 3 exemplos têm pelo menos um painel dominado por imagem/gráfico — isto não é opcional.
Separado em pronto pra usar vs. precisa gerar, com a fonte real de cada um.

| # | Imagem | Status | Fonte |
|---|--------|--------|-------|
| 1 | Mapa das 3 cidades com contraste geomorfológico (Recife planície fluvial/costeira, Petrópolis relevo serrano, Curitiba urbano de contraste com microdrenagem) | **Criar do zero** | Conceito já esboçado em `local_runs/figure_plan.md` (Figura 1, painel B) — reaproveitar a ideia, não existe arquivo pronto |
| 2 | Diagrama conceitual de HAND e TWI (ilustração simples: altura até a drenagem; acúmulo de fluxo por declividade) | **Criar do zero** | Nenhum asset existe — é o mais importante do painel de Método, porque traduz os dois termos técnicos mais centrais visualmente |
| 3 | Patch Sentinel-2 real, RGB + falsa-cor, um por cidade (evidência visual auxiliar, deixar claro na legenda que não é variável causal) | **Já existe, só selecionar** | `REV-P/local_runs/figures_meeting_pack_20260526_173013/04_SENTINEL_BANDAS_INDICES_RENDER_TECNICO/` — Recife (REC_00505), Petrópolis (PET_00360), Curitiba (CUR_00695) |
| 4 | Gráfico de coeficientes do modelo Firth de Recife (forest plot: cada variável com sinal e intervalo de confiança) | **Gerar a partir de dado real** | `outputs_public/data/susc_20c_modelagem_validacao_estatistica_rigorosa_recife/results/primaria_v12_firth_multivariate_coefs.csv` + `primaria_v12_bootstrap_coefs.csv` |
| 5 | Gráfico de barras comparando LOO-AUC entre cenários (Recife 0,678; Curitiba embaralhado 0,646 vs. holdout real 2026 0,525) | **Gerar** (poucos números, fácil montar) | Números já documentados neste guia e no `PLANO_ACAO_produto_v1.md` |
| 6 | Gráfico de linha *walk-forward* de Curitiba por ano-corte (2024 = 0,63; 2025 = 0,67; 2026 = 0,52) — mostra visualmente que só 2026 colapsa | **Gerar** | `outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/reports/susc_20q_...` e `susc_20w_...` |
| 7 | (Se incluir Reino Unido) Gráfico comparando HAND vs. elevação/declividade entre definição de negativo por exclusão e por observação — mostra a invariância de HAND | **Gerar** | `local_runs/mod-neg-01/resumo.json` |

**Evite**: reaproveitar as figuras antigas de `local_runs/Fig1_draft.png` a `Fig6_draft.png` ou do
pipeline DINOv2 sem revisar — foram feitas para um artigo com enquadramento "Sentinel-first"
anterior ao fechamento da linha DINO como candidata a feature (decisão de 2026-08-01/02). Se
usar alguma como pano de fundo do painel de evidência auxiliar, a legenda precisa deixar
explícito que é ilustração de pipeline auxiliar, não do modelo causal atual.

---

## 7. Checklist final contra as regras fixas do projeto

Antes de fechar o texto do pôster, confira que nenhuma frase viola as regras que já regem o
projeto inteiro:

- [ ] Em nenhum lugar o texto sugere que o modelo "descobre" enchente a partir de imagem — o
      rótulo é físico-hidrológico-urbano, fixo desde o início.
- [ ] Sentinel/CBERS aparecem só como evidência/contexto visual, nunca como variável causal do
      score.
- [ ] Nenhuma variável do modelo é score, threshold, proxy ou derivada do próprio rótulo.
- [ ] Curitiba e Petrópolis aparecem com status honesto (aberto/bloqueado), sem linguagem de
      "resolvido" ou "validado" onde ainda não está.
- [ ] Termos técnicos (HAND, TWI, D-infinity, Firth, EPV) aparecem traduzidos em pelo menos uma
      frase cada, não soltos.
