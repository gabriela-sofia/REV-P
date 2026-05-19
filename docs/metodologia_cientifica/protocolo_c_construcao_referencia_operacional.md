# Protocolo C: construção graduada de referência operacional

## 1. Motivação

O REV-P não parte de ground truth observado. Mas também não trata evidência contextual como "nada". Há um espaço metodológico real entre esses dois extremos: evidências institucionais, topografia regional, infraestrutura de drenagem, padrões de uso do solo e diagnósticos estruturais não são observações de evento, mas tampouco são ruído.

O Protocolo C existe para organizar esse espaço de forma auditável. Ele define como evidências se acumulam em direção a uma possível referência operacional, sem declarar que essa referência existe antes das condições necessárias serem satisfeitas.

A contribuição concreta do Protocolo C é tornar explícita a diferença entre contexto, proxy, candidato de referência e validação operacional — e registrar, por patch e por fonte, onde cada evidência se situa nessa escada.

Ground truth operacional continua bloqueado no estado atual do REV-P porque faltam, para todos os patches candidatos: evento observado documentado, alinhamento temporal entre imagem e evento, sobreposição espacial confirmada entre patch e fonte, e revisão humana ou de especialista. O Protocolo C não muda isso — ele documenta o bloqueio com precisão.

---

## 2. Diferença entre evidência, referência e ground truth

Em sensoriamento remoto, esses termos são frequentemente usados de forma imprecisa, criando equívocos de avaliação e overclaim em publicações. O Protocolo C adota as seguintes definições:

**Evidência contextual**
Qualquer fonte que forneça informação física-ambiental sobre um local ou região, sem ter sido coletada especificamente para documentar um evento ou fenômeno. Exemplos: topografia regional, mapas históricos de drenagem, dados de uso do solo derivados de classificação de imagens. Evidência contextual não valida fenômeno — ela fornece contexto para interpretação.

**Proxy de referência**
Uma métrica construída sobre evidências contextuais múltiplas, documentada e reproduzível, que permite comparação estrutural entre patches sem afirmar observação de fenômeno. Exemplos: índice GIS multicritério de exposição hídrica, índice de coerência estrutural Sentinel. Um proxy pode orientar triagem e priorização, mas não serve como rótulo de treinamento.

**Reference data**
Dados coletados ou anotados com o propósito explícito de servir como referência para um produto específico. Inclui anotações manuais, produtos de sensoriamento remoto de alta resolução interpretados por especialista, ou dados de campo vinculados a um evento e local. A qualidade de reference data depende diretamente da cadeia de evidência que a sustenta.

**Validation reference**
Subconjunto de reference data com qualidade suficiente para uso em avaliação quantitativa de desempenho de um modelo ou produto. Requer alinhamento temporal, espacial, controle de viés e, normalmente, revisão independente.

**Ground reference**
Evidência de campo ou equivalente funcional, com vínculo documentado a um evento e local específico. Pode incluir observação direta, imagem de maior resolução interpretada por especialista, ou produto operacional validado externamente com incerteza documentada.

**Ground truth operacional**
O estado final de uma referência que sustenta decisões operacionais. Em flood mapping, isso normalmente significa: evento observado com data e geometria documentadas, fonte rastreável e validação independente. É raro e custoso — exatamente por isso não deve ser declarado por conveniência.

**Regra central do Protocolo C:**
Todo ground truth operacional pode ser tratado como referência, mas nem toda referência pode ser promovida a ground truth. A promoção só ocorre quando todas as condições de alinhamento, observação e validação são satisfeitas.

---

## 3. Lições metodológicas da literatura

O campo de flood mapping tem avançado na construção de datasets de referência, mas os desafios de alinhamento temporal, viés espacial e escalabilidade permanecem abertos.

### O que datasets fortes têm em comum

Datasets bem construídos de mapeamento de inundação compartilham algumas características que o REV-P ainda não possui para nenhum patch:

- **Evento específico documentado**: há uma data de ocorrência, uma geometria afetada e uma fonte de registro do evento (relatório oficial, notícia verificada, produto de defesa civil).
- **Imagem próxima ao evento**: a imagem de satélite usada para anotação ou mapeamento foi adquirida dentro de uma janela temporal compatível com o evento — horas, no melhor caso; dias, no limite aceitável.
- **Anotação ou produto validado**: as áreas inundadas foram anotadas por especialista, ou um produto operacional com incerteza documentada foi usado como referência.
- **Alinhamento espacial verificado**: a geometria do evento e o recorte espacial do dado de satélite foram confirmados como sobrepostos.

### Fontes metodológicas comparáveis

As referências abaixo são mencionadas como exemplos da literatura de flood mapping que informam o Protocolo C. Não são citadas com DOI porque o repositório não as verificou diretamente. Devem ser incluídas com citação formal na versão acadêmica do trabalho.

**Sen1Floods11**
Dataset de mapeamento de inundação com Sentinel-1 e Sentinel-2 abrangendo 11 eventos globais. Usa labels manuais de alta confiança produzidos por especialistas em interpretação de imagens SAR. É uma referência metodológica relevante porque demonstra o custo de anotação necessário para ground truth válido: especialista + imagem próxima ao evento + evento documentado. Advertências na literatura incluem viés espacial nos eventos selecionados e questões de generalização geográfica. Referência metodológica a ser citada na versão acadêmica.

**Kuro Siwo**
Dataset SAR multitemporal com anotações manuais para eventos globais de inundação. Relevante porque trata a dimensão temporal explicitamente: antes do evento, durante e depois. Evidencia que ground truth de inundação exige série temporal, não snapshot estático. Referência metodológica a ser citada na versão acadêmica.

**Urban Flood Observations (UFO)**
Dataset com PlanetScope (~3 m de resolução) e eventos urbanos específicos com anotação humana de inundação/não inundação. Relevante porque trabalha com escala urbana — mais próxima da escala de patch do REV-P. A resolução de 3 m permitiu anotação mais fina que Sentinel-2 (10 m). Evidencia que resolução e escala importam na definição de ground truth urbano. Referência metodológica a ser citada na versão acadêmica.

**Copernicus Emergency Management Service (EMS) / Global Flood Monitoring (GFM)**
Produto operacional que gera mapas de inundação automatizados para eventos em escala global. Útil como fonte externa com cobertura ampla, mas deve ser tratado como produto algorítmico com incerteza associada — não como verdade observacional automática. Produtos GFM incluem camadas de probabilidade (likelihood) que documentam incerteza. Para uso como referência, é necessário documentar: qual versão do produto, qual evento, qual período, qual limiar de probabilidade aplicado, e se houve validação independente do produto naquela região. Referência metodológica a ser citada na versão acadêmica.

**Accuracy assessment em sensoriamento remoto**
A literatura clássica de avaliação de acurácia em sensoriamento remoto (matriz de confusão, Kappa, métricas F1, precision/recall) pressupõe que o dataset de teste é de referência confiável, espacialmente independente do dado classificado, e temporalmente alinhado. Quando qualquer dessas condições não é satisfeita, as métricas perdem significado. O REV-P reconhece explicitamente que nenhuma dessas condições é satisfeita atualmente para nenhum patch.

### O que o REV-P tem vs. o que falta

| Critério | Estado no REV-P |
|---|---|
| Evento documentado | Ausente para todos os patches |
| Alinhamento temporal imagem-evento | Ausente; imagens Sentinel são snapshots estáticos |
| Anotação manual ou de especialista | Ausente; revisão humana planejada mas não concluída |
| Fonte com incerteza documentada | Parcial: PE3D, SGB, GeoCuritiba têm limitações documentadas |
| Alinhamento espacial patch-fonte | Bloqueado por CRS incompatível (B1) |
| Revisão independente | Ausente |
| CRS reconciliado | Bloqueado |

Essa tabela não é um problema a esconder — é a documentação honesta do estado atual, que o Protocolo C torna explícita e auditável.

---

## 4. Critérios de promoção

Para que um patch avance de um nível de referência para o seguinte, é necessário satisfazer os critérios acumulados. Os critérios abaixo são organizados por nível de exigência:

### De CONTEXTUAL_EVIDENCE para AUDITABLE_REFERENCE_PROXY

- CRS da fonte declarado explicitamente (mesmo que incompatível)
- Coordenadas do patch auditadas e registradas
- Linhagem do patch documentada (não placeholder)
- Fonte externa identificada e categorizada (institucional ou tecnicamente justificável)
- Coerência hidrogeomorfológica declarada (mesmo que UNVERIFIED, deve ser registrada)
- Limitações da fonte explicitadas

### De AUDITABLE_REFERENCE_PROXY para STRONG_REFERENCE_CANDIDATE

Todos os anteriores, mais:
- CRS reconciliado entre patch e fonte (ou bloqueio documentado com plano de resolução)
- Sobreposição espacial entre patch e fonte confirmada ou estimada
- Coerência Sentinel-embedding avaliada (ALIGNED ou DIVERGENT, não apenas UNVERIFIED)
- Revisão humana executada (pelo menos parcial)
- Pelo menos duas evidências independentes apontando para a mesma propriedade
- Discordâncias entre fontes documentadas e explicadas

### De STRONG_REFERENCE_CANDIDATE para EVENT_OBSERVED_REFERENCE_ELIGIBLE

Esta é uma elegibilidade futura, não um status aplicado agora. Requer:
- Evento específico confirmado (nome, data, fonte do registro)
- Imagem Sentinel ou similar adquirida dentro da janela temporal do evento
- Correspondência espacial entre geometria do evento e footprint do patch confirmada
- Anotação por especialista ou produto validado externo com incerteza documentada
- Revisão independente (segundo revisor ou segundo conjunto de evidências)
- Fonte rastreável com cadeia de custódia documentada

### OPERATIONAL_GROUND_TRUTH_BLOCKED → desbloqueio

O desbloqueio de promoção para ground truth operacional requer todos os critérios acima mais:
- Aprovação de stakeholder responsável (pesquisador orientador, defesa civil, ou equivalente)
- Teste de alinhamento CRS completo e documentado
- Separação de processos confirmada (inundação vs. alagamento vs. escorregamento)
- Avaliação de viés espacial e temporal do conjunto

---

## 5. Bloqueadores

Os bloqueadores a seguir impedem promoção de qualquer patch para ground truth operacional. Cada bloqueador deve ser documentado no registro com razão específica.

**Bloqueadores críticos (impedem qualquer promoção):**
- Ausência de evento observado documentado
- Ausência de data do evento
- Imagem fora da janela temporal compatível com o evento
- CRS inconsistente entre patch e fonte sem plano de reconciliação
- Fonte apenas modelada tratada como observação (ex.: mapa de suscetibilidade usado como label)

**Bloqueadores intermediários (impedem promoção acima de AUDITABLE_REFERENCE_PROXY):**
- Evidência GIS parcial sem estimativa de cobertura por patch
- Patch fora da cobertura espacial da fonte
- Dependência exclusiva de embedding DINO para caracterização
- Dependência exclusiva de cluster não supervisionado
- Dependência exclusiva de índice espectral (NDWI, NDBI)
- Ausência de revisão humana executada

**Bloqueadores de qualidade (impedem promoção para STRONG_REFERENCE_CANDIDATE):**
- Conflito entre fontes sem explicação documentada
- Apenas uma evidência sem independência de fonte
- Alinhamento temporal desconhecido (NO_EVENT_DATA) sem estimativa de compatibilidade
- Ausência de teste de sobreposição espacial

---

## 6. Estados metodológicos

Os estados abaixo preservam os status definidos na camada de referência contextual e adicionam contexto metodológico baseado nos critérios do Protocolo C.

### CONTEXTUAL_EVIDENCE

Evidência que fornece contexto físico-ambiental sobre um local ou região, sem vínculo confirmado com fenômeno observado.

**Permite afirmar:**
- Que o patch está em contexto de [propriedade contextual específica], conforme [fonte + limitações]
- Que a topografia/drenagem/cobertura sugere [característica ambiental], sujeita a [limitação]
- Que a fonte [X] cobre [região] mas não validou cobertura por patch individual

**Não permite afirmar:**
- Que o patch é inundável, vulnerável, suscetível ou apresenta risco (sem qualificação)
- Que a evidência valida qualquer fenômeno observado
- Que a fonte pode ser usada como label, classe ou target de treinamento

### AUDITABLE_REFERENCE_PROXY

Índice ou proxy construído sobre múltiplas evidências contextuais, documentado e reproduzível. É a contribuição técnica mais forte que o REV-P pode fazer no estado atual para patches sem evento observado.

**Permite afirmar:**
- Que o índice [X] classifica patches em [tiers/scores] com base em [componentes explícitos]
- Que o proxy serve para triagem, priorização e comparação estrutural
- Que as limitações do proxy são [lista explícita]

**Não permite afirmar:**
- Que o proxy valida suscetibilidade ou risco de forma operacional
- Que o score prediz probabilidade de inundação
- Que o proxy é equivalente a ground truth

### STRONG_REFERENCE_CANDIDATE

Convergência de múltiplas evidências independentes em direção a uma propriedade específica, com revisão humana e bloqueadores documentados. Não é ground truth — é o estado mais avançado de referência que um patch pode alcançar sem evento observado.

**Permite afirmar:**
- Que a convergência de [fonte 1] + [fonte 2] + revisão humana apoia a hipótese de [propriedade]
- Que o patch é candidato forte a validação adicional e que os critérios restantes para promoção são [lista específica]
- Que as discordâncias encontradas são [descrição] e foram explicadas por [raciocínio]

**Não permite afirmar:**
- Que o patch foi validado como [propriedade] operacionalmente
- Que o candidato pode ser usado como label de treinamento sem revisão adicional

### OPERATIONAL_GROUND_TRUTH_BLOCKED

Estado explícito de bloqueio. Toda promoção para ground truth operacional está bloqueada no estado atual do REV-P. Este status é aplicado quando há evidência relevante acumulada, mas falta pelo menos um critério crítico para uso operacional.

**O que registrar:**
- Quais critérios foram satisfeitos
- Quais critérios faltam
- Qual ação específica desbloquearia a promoção

### INSUFFICIENT_REFERENCE

Evidência insuficiente, pendente de aquisição ou com incompletude severa. Não pode servir nem como contextual sem tratamento adicional.

**O que registrar:**
- Razão da insuficiência
- Ação pendente para uso futuro

### EVENT_OBSERVED_REFERENCE_ELIGIBLE (elegibilidade futura)

Não é um status aplicado a nenhum patch atual. É o nível que descreveria um patch que satisfez todos os critérios de evento observado, alinhamento temporal, alinhamento espacial, anotação qualificada e revisão independente.

Este conceito está documentado para orientar trabalho futuro, não para classificar patches existentes. Aplicá-lo a qualquer patch sem evidência observacional direta seria overclaim.

---

## 7. Relação com DINOv2

DINOv2 é usado no REV-P como encoder visual congelado para extração de representações estruturais de patches Sentinel. Isso tem papel metodológico claro e contribuição real — mas fora do escopo de construção de referência operacional.

**O que DINOv2 oferece ao Protocolo C:**
- Diagnóstico de coerência estrutural: um embedding muito discrepante pode indicar anomalia visual que orienta revisão humana
- Triagem de candidatos: outliers e medoids no espaço de embeddings sugerem quais patches merecem atenção prioritária
- Evidência de consistência: embeddings estáveis sob perturbação controlada indicam que o patch tem representação estrutural robusta

**O que DINOv2 não oferece:**
- Ground truth de nenhum tipo
- Validação de fenômeno observado
- Label de inundação, suscetibilidade ou risco
- Promoção de referência por si só

Cluster de embedding não vira classe. Vizinhança no espaço latente não vira rótulo de inundação. Outlier estrutural não é confirmação de fenômeno. Esses limites não diminuem a contribuição dos diagnósticos DINO — eles simplesmente definem o papel correto de cada ferramenta dentro do protocolo.

---

## 8. Relação com Protocolo A e Protocolo B

O REV-P opera com três modos metodológicos, cada um com papel distinto:

**Protocolo A — modo científico contextual**
Análise estrutural read-only: embeddings, diagnósticos, índice GIS, comparação de representações. É o modo válido no estado atual. Não requer ground truth e não produz afirmações preditivas. Toda análise publicável do REV-P na fase atual é Protocolo A.

**Protocolo C — construção de referência operacional**
Organiza a hierarquia de evidências de referência: inventário de fontes, critérios de promoção, registro de bloqueadores, documentação de claims permitidos/proibidos. O Protocolo C prepara o caminho metodológico para Protocolo B, sem acionar supervisão antes da hora.

**Protocolo B — supervisão com referência validada**
Bloqueado no estado atual. Só pode ser desbloqueado quando houver referência operacional forte suficiente: evento observado, alinhamento temporal e espacial, anotação qualificada e revisão independente. O Protocolo C não desbloqueia supervisão automaticamente — ele documenta o que falta.

O Protocolo C não é intermediário entre A e B no sentido de "meio caminho para supervisão". É a linhagem metodológica que torna auditável a distância entre o que o projeto tem e o que precisaria ter para supervisão legítima.

---

## 9. Saída metodológica esperada

O resultado do Protocolo C não é um modelo treinado, um mapa de inundação ou uma previsão. É documentação auditável:

- **Matriz de evidência por patch**: quais fontes cobrem quais patches, com qual qualidade declarada e quais bloqueadores ativos
- **Registry de fontes de referência**: inventário de datasets potencialmente úteis, categorizados por family, tipo, evidência observacional, incerteza documentada e allowed_use
- **Status de promoção/bloqueio por patch**: onde cada patch se situa na hierarquia e o que impede sua promoção
- **Claims permitidos e proibidos por patch**: o que pode e o que não pode ser afirmado para cada combinação patch × fonte
- **Trilha de auditoria para escrita acadêmica**: documentação suficiente para que um revisor externo reconstrua o raciocínio de classificação sem acesso aos dados pesados

---

## Referências internas

- [`docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md`](protocolo_c_fechamento_evidencias_ground_reference.md) — etapa de fechamento: gates de promoção (G0–G9), níveis de evidência, matriz de lacunas e relação com revisão humana e Protocolo B
- [`docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md`](protocolo_c_revisao_humana_referencia.md) — protocolo de revisão humana: decisões possíveis, critérios de bloqueio, registro obrigatório e relação com anotação futura
- [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](protocolo_c_aquisicao_ground_reference.md) — etapa de aquisição: qualificação metadata-only de eventos candidatos e vínculos patch-evento-fonte
- [`docs/metodologia_cientifica/camada_referencia_contextual_validada.md`](camada_referencia_contextual_validada.md) — hierarquia de status e guardrails por patch
- [`datasets/contextual_reference_layer_registry.csv`](../../datasets/contextual_reference_layer_registry.csv) — registro de referências por patch
- [`datasets/ground_reference_evidence_source_registry.csv`](../../datasets/ground_reference_evidence_source_registry.csv) — inventário de fontes de referência
- [`datasets/flood_event_candidate_registry.csv`](../../datasets/flood_event_candidate_registry.csv) — registro de eventos candidatos (etapa de aquisição)
- [`datasets/patch_event_reference_link_registry.csv`](../../datasets/patch_event_reference_link_registry.csv) — vínculos patch-evento-fonte (etapa de aquisição)
- [`datasets/schemas/ground_reference_evidence_source_schema.csv`](../../datasets/schemas/ground_reference_evidence_source_schema.csv) — schema de campos do inventário de fontes
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](patch_lineage_and_grounding.md) — linhagem e bloqueadores por patch
- [`docs/metodologia_cientifica/research_datasets_and_artifacts.md`](research_datasets_and_artifacts.md) — cadeia de rastreabilidade geral
