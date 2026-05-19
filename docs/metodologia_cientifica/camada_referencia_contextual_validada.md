# Camada de referência contextual validada do REV-P

## Introdução

A camada de referência contextual validada fornece um framework metodológico para organizar, auditar e distinguir evidências externas, proxies heurísticos e afirmações sobre ground truth sem promover análise exploratória a verdade operacional.

Este documento define:
- O que é ground truth no contexto do REV-P
- Por que ground truth não pode ser binário ou observado sem validação estruturada
- Como construir uma camada de referência validada
- Uma escada de status que fortaleça evidências progressivamente
- Guardrails explícitos contra overclaim

---

## Por que "ground truth" não é binário

Em projetos de detecção de inundação tradicionais, "ground truth" frequentemente significa "rótulo observado de um evento" — uma classificação binária (inundado/não inundado) anotada manualmente sobre imagens ou relatório de campo.

O REV-P não pode usar essa definição porque:

1. **Não há observação de evento**: Os patches Sentinel abrangem períodos multi-temporais (diferentes datas de captura). Uma imagem Sentinel-2 de 2021 não é observação de um evento real — é um snapshot estático que pode ou não capturar um fenômeno.

2. **Temporal não está alinhada**: Mesmo que um evento histórico de inundação tenha sido documentado (ex.: chuvas de março de 2022), a correspondência entre "data do evento" e "data da imagem Sentinel disponível para o patch" é um problema aberto não resolvido.

3. **Delimitação espacial é incompleta**: Fontes externas (PE3D, SGB, GeoCuritiba) cobrem regiões geograficamente, não patches individuais. A intersecção entre footprint de fonte e bounding box de patch é uma operação espacial — bloqueada pelo bloqueador B1 (discrepância CRS).

4. **Não há separação obrigatória de processos**: Uma inundação, um alagamento e um movimento de massa têm origens geomorfológicas distintas. Dados topográficos de SGB/CPRM (Petrópolis) separam explicitamente esses processos. Sem essa separação no dataset integrado, chamar qualquer coisa de "ground truth de suscetibilidade" é conflação de fenômenos distintos.

5. **Evidência modelada não é observada**: Camadas de uso do solo (FBDS, MapBiomas) são produtos de modelagem cartográfica, não observação direta. Classificação supervisionada de pixels MODIS ou Landsat é um proxy — valioso, mas não observação.

---

## Escada de evidência no REV-P

O projeto articula evidências sobre um eixo de progressão, não como presença/ausência de "verdade":

```
EVIDÊNCIA CONTEXTUAL
    ↓
PROXY DE REFERÊNCIA AUDITÁVEL
    ↓
REFERÊNCIA FORTE CANDIDATA
    ↓
GROUND TRUTH OPERACIONAL ← BLOQUEADO NO ESTADO ATUAL
```

---

## Definições de status de referência

### Status 1: CONTEXTUAL_EVIDENCE

**O que significa:**
Uma fonte externa ou um indicador indireto que fornece contexto de suporte, sem pretender descrever uma propriedade observada do patch.

**Exemplos:**
- Topografia regional de SGB/CPRM (contexto de risco geomorfológico)
- Densidade viária local (proxy interpretável de urbanização)
- Cartas históricas de hidrografia municipal (contexto de estrutura de drenagem)
- Uso do solo FBDS (padrão de ocupação territorial)

**O que permite afirmar:**
- "Este patch está em um contexto de densidade viária elevada"
- "A topografia regional sugere um vale fluvial"
- "O uso do solo é predominantemente residencial"
- "A infraestrutura de drenagem está documentada nas fontes municipais"

**O que NÃO permite afirmar:**
- "Este patch é inundável" (requer validação temporal + espacial)
- "Este patch sofrerá enchente" (requer verdade observada)
- "Este patch é suscetível" (requer evento observado ou modelo validado)
- "Este padrão de uso do solo causa inundação" (causalidade não comprovada localmente)

**Evidências mínimas necessárias:**
- Fonte identificada e referenciada
- CRS explícito (ou anotação de incompletude CRS)
- Cobertura geográfica declarada
- Limitações explícitas

---

### Status 2: AUDITABLE_REFERENCE_PROXY

**O que significa:**
Uma métrica ou índice heurístico construído a partir de múltiplas evidências contextuais, documentado e reproduzível, que oferece proxy auditável de comparação sem afirmar descrição de fenômeno.

**Exemplos:**
- Índice GIS multicritério (distância ao rio + densidade viária + uso do solo): proxy interpretável para "exposição estrutural ao contexto hídrico", não "suscetibilidade observada"
- Coerência Sentinel-embedding (medida de estabilidade estrutural visual entre embeddings DINO e imagens): proxy de "coerência estrutural", não "representatividade de verdade de campo"
- Score de revisão humana determinístico (baseado em outlier + cobertura CRS + limitação temporal): triagem de candidatos, não label

**O que permite afirmar:**
- "O índice GIS classifica esses patches em três tiers de exposição estrutural"
- "O proxy de coerência Sentinel quantifica a estabilidade visual relativa entre patches"
- "A triagem de revisão priorizou 5 patches para inspeção baseado em critérios determinísticos"
- "Este proxy identifica patches com cobertura CRS incompleta"

**O que NÃO permite afirmar:**
- "O índice GIS prediz suscetibilidade" (falta validação de desempenho)
- "A coerência Sentinel prova vulnerabilidade" (coerência ≠ risco)
- "Este patch é crítico para inundação" (requer observação de evento)

**Evidências mínimas necessárias:**
- Definição explícita da métrica/componentes
- Documentação de pesos ou lógica de agregação
- Resultados reproduzíveis localmente
- Explicitação de quais variáveis estão bloqueadas ou ausentes
- Teste de limites conhecidos (ex.: "O índice não cobre Petrópolis porque FBDS está incompleto")

---

### Status 3: STRONG_REFERENCE_CANDIDATE

**O que significa:**
Uma evidência ou agregação de evidências que aponta fortemente para uma propriedade específica, mas ainda aguarda validação externa ou reconciliação de bloqueadores técnicos para promoção a operacional.

**Exemplos:**
- Uma combinação de (PE3D topografia + ESIG drenagem + revisão humana de foto satélite) que demarca uma zona de baixa altitude com drenagem deficiente → candidata forte a "zona de acúmulo de água" mas aguardando (1) reconciliação CRS, (2) validação espacial pixel-level, (3) temporalidade de evento observado
- Um patch com concordância entre (embedding DINO outlier + índice GIS baixo + feedback humano "aspecto estruturalmente distinto") → candidato forte a validação adicional mas não decisão operacional ainda

**O que permite afirmar:**
- "Esta agregação de evidências aponta para [propriedade específica]"
- "A concordância entre [fonte 1] e [fonte 2] fortalece a hipótese de [propriedade]"
- "Este candidate é recomendado para validação adicional [motivo específico]"
- "As limitações conhecidas são [lista explícita]; resolução dessas limitações elevaria o status"

**O que NÃO permite afirmar:**
- "Este patch é [propriedade] operacionalmente" (falta resolução de bloqueadores)
- "Este patch será inundado" (falta temporalidade de evento)
- "Usar este como rótulo de treinamento" (falta auditorias finais)

**Evidências mínimas necessárias:**
- Agregação documentada de ≥2 evidências independentes
- Explicitação de quais bloqueadores (CRS, temporal, espacial) ainda estão abertos
- Critérios para promoção a operacional (ex.: "reconciliação CRS + validação spatial de sobreposição + revisão humana de feedback")
- Análise de discordância se múltiplas evidências divergem

---

### Status 4: OPERATIONAL_GROUND_TRUTH_BLOCKED

**O que significa:**
Ground truth operacional é a verdade de campo que sustentaria decisões operacionais (ex.: evacuação, alocação de recursos, política pública). No estado atual, TODAS as promoções a operacional estão bloqueadas.

**Por que está bloqueado:**
1. **Ausência de evento observado**: Não há correspondência confirmada entre data de imagem Sentinel e data de evento de inundação documentado
2. **Ausência de temporalidade compatível**: Os patches Sentinel cobrem períodos estáticos, não séries temporais de captura de evento
3. **Ausência de delimitação espacial oficial**: Nenhuma fonte oficial demarca "essa bounding box é inundável" ou "esse polígono sofreu inundação em [data]"
4. **Cobertura GIS parcial**: Operações espaciais bloqueadas por discrepância CRS; sobresposição patch-fonte não confirmada
5. **Dependência exclusiva de embedding/cluster**: Modelos visuais não validados como preditores; clustering não supervisionado não é validação
6. **Inconsistência CRS/coordenada**: Patches WGS84 UTM, fontes SIRGAS 2000 UTM — reconciliação não executada

**O que permite afirmar:**
- "O status de operacional está bloqueado por [lista específica de razões]"
- "As condições para desbloqueio incluem [critérios técnicos concretos]"
- "Este candidato não pode ser usado em decisões operacionais nesta fase"

**O que NÃO permite afirmar:**
- "Este patch é [propriedade] na prática" (bloqueador ativo)
- "Usar este como rótulo de treinamento" (vedado completamente)
- "Este patch será inundado" (vedado completamente)

**Condições para desbloqueio futuro:**
- Reconciliação CRS completa e validação de sobreposição spatial
- Vinculação temporal confirmada de patch Sentinel a evento observado documentado
- Separação obrigatória de processos (inundação vs. alagamento vs. escorregamento)
- Revisão humana executada e documentada
- Métricas de concordância entre fontes ≥ limiar operacional definido e auditado
- Aprovação explícita de stakeholder autorizado (defesa civil, pesquisador responsável)

---

### Status 5: INSUFFICIENT_REFERENCE

**O que significa:**
Uma fonte ou evidência com limitações tão severas que não pode ser usada nem como contextual nem como proxy sem aquisição/tratamento adicional.

**Exemplos:**
- MapBiomas não adquirido (EXISTS_PARTIAL no registro)
- Dados de defesa civil Curitiba não disponibilizados (MISSING)
- Uma fonte com CRS desconhecido e sem metadados de georeferenciação

**O que permite afirmar:**
- "Esta fonte está pendente [aquisição | tratamento | resolução]"
- "Quando resolvida, poderá fornecer [tipo de evidência]"

**O que NÃO permite afirmar:**
- Qualquer coisa sobre o conteúdo da fonte

**Ação esperada:**
Documentar em registry, marcar com promotion_allowed=false e uma ação de future work.

---

## Como construir e auditar a camada de referência

### 1. Inventário de evidências

Cada evidência externa deve ser registrada com:
- ID único (ex.: `recife_pe3d_mde`, `petropolis_sgb_rigeo`)
- Fonte oficial ou justificativa técnica
- Status local (EXISTS_COMPLETE / EXISTS_PARTIAL / MISSING)
- CRS declarado (ou anotação de incompletude)
- Cobertura geográfica
- Limitações explícitas

Exemplo: Veja [`datasets/external_evidence_registry.csv`](../../datasets/external_evidence_registry.csv)

### 2. Associação evidência → patch

Cada patch do corpus DINO pode estar associado a zero ou mais evidências. A associação deve registrar:
- patch_id (ex.: `REC_01`)
- evidence_id
- status de cobertura (FULLY_COVERED / PARTIALLY_COVERED / NOT_COVERED / UNVERIFIED_SPATIAL)
- CRS reconciliation status (COMPATIBLE / INCOMPATIBLE / UNRESOLVED)
- Temporal alignment status (UNKNOWN / COMPATIBLE / INCOMPATIBLE / NO_EVENT_DATA)

### 3. Índices e proxies

Agregações de múltiplas evidências devem ser documentadas com:
- Nome e versão (ex.: `gis_multicriteria_index_v1gq`)
- Componentes (quais evidências, quais pesos)
- Método de cálculo (determinístico, fórmula)
- Resultados (quais patches, quais scores)
- Limitações (quais componentes estão bloqueados, por quê)

Exemplo: Veja [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](patch_lineage_and_grounding.md) seção "Contexto GIS"

### 4. Rastreabilidade

Cada decisão de status deve ser rastreável:
- Qual script ou ação produziu o resultado?
- Qual evidência é citada?
- Qual bloqueador (se houver) impede promoção?
- Qual revisão humana foi executada (se houver)?

Use campo `notes` no registry para citação de manifests/estágios.

---

## Guardrails contra overclaim

### Termos e claims proibidos no estado atual

**Nenhum dos seguintes pode ser afirmado:**
- "predição de enchente" (falta validação preditiva)
- "detecção de enchente" (falta temporalidade de evento)
- "ground truth observado" (falta dado de evento confirmado)
- "flood prediction" (vedado)
- "flood detection" (vedado)
- "operational ground truth" (status bloqueado)
- "este patch é inundável" sem qualificador "candidato a revisão"
- "machine learning detectou suscetibilidade" (vedado; DINO não é classificador)
- "este clustering prova agrupamento de risco" (clustering não é validação)
- "o índice GIS prediz vulnerabilidade" (é proxy interpretável, não preditor validado)

### Termos e claims permitidos

- "este patch exibe [propriedade contextual] conforme [fonte + limitações]"
- "o proxy de [métrica] classifica patches em [tiers], útil para [propósito específico]"
- "a concordância entre [fonte 1] e [fonte 2] apoia a hipótese de [propriedade], sujeita a [limitação]"
- "este patch é candidato a validação adicional por [razão específica]"
- "as limitações bloqueando operacionalização são: [lista]"
- "para elevar o status de [patch], seriam necessários: [ações específicas]"

---

## Integração com o pipeline REV-P

A camada de referência contextual validada conecta-se aos estágios existentes:

| Estágio | Entrada | Saída | Status no registro |
|---|---|---|---|
| v1fu — Sentinel manifest | 128 patches candidatos | — | Topo da cadeia |
| v1gq — GIS multicritério | 12 patches + evidências externas | Índice GIS | AUDITABLE_REFERENCE_PROXY |
| v1gt — Cobertura uso do solo | 128 patches + FBDS/MapBiomas | Cobertura auditada | CONTEXTUAL_EVIDENCE (bloqueadores anotados) |
| v1gv — Matriz de cobertura externa | Evidências externas | Status por patch | Mapeamento CONTEXTUAL_EVIDENCE ↔ patch |
| v1gw — Human review candidate | Índices + embedding outliers | Triagem determinística | AUDITABLE_REFERENCE_PROXY (triagem) |
| v1hb — Human review execution | Candidatos + feedback humano | Notas de revisão | Elevação para STRONG_REFERENCE_CANDIDATE (se aprovado) |

---

## Exemplo concreto: Recife PE3D

### Inventário
- **evidence_id**: `recife_pe3d_mde`
- **source**: PE3D/MDE rasters (Pernambuco 3D)
- **region**: Recife
- **status local**: EXISTS_PARTIAL (48 MDE tiles + 7 orthophoto)
- **CRS**: EPSG:31985 (SIRGAS 2000 UTM zone 25S)
- **patch_coverage**: 18 (Recife canônico)
- **evidence_tier**: STRONG (quando completamente inspecionada)
- **limitations**: "Stack parcial; sem leitura pixel-level; sem temporalidade de evento; header não valida cobertura individual de patch"

### Associação a patches
Para cada patch REC_01–REC_18:
- **evidence_id**: `recife_pe3d_mde`
- **crs_status**: INCOMPATIBLE (patch WGS84 UTM32722, fonte SIRGAS 31985) → bloqueador B1 ativo
- **spatial_coverage**: UNVERIFIED_SPATIAL (falta operação spatial para confirmar sobreposição)
- **temporal_alignment**: UNKNOWN (PE3D é snapshot estático; sem dado de evento para REC_01–18)
- **reference_status**: CONTEXTUAL_EVIDENCE (não PROXY ainda, porque CRS incompatível)

### Condições para promoção
REC_01 poderia ser elevado a AUDITABLE_REFERENCE_PROXY se:
1. CRS reconciliado e sobreposição patch-PE3D confirmada spacialmente
2. Revisão humana executada (v1hb stage)

REC_01 poderia ser elevado a STRONG_REFERENCE_CANDIDATE se:
1. + Correlação com evento observado documentado (ex.: chuva de março 2022, imagem Sentinel de março 2022, feedback de defesa civil Recife)
2. + Concordância com evidências independentes (ex.: ESIG drenagem + embedding outlier)

REC_01 poderia ser elevado a OPERATIONAL_GROUND_TRUTH se:
1. + Tudo acima + aprovação explícita de stakeholder

---

## Linhagem metodológica do Protocolo C

O Protocolo C nasceu no REV-P como resposta à necessidade de organizar uma linhagem metodológica para construção de ground reference em um projeto sem ground truth observado de inundação.

A versão inicial dessa camada tratava evidências contextuais de forma qualitativa, sem critérios explícitos de promoção ou bloqueio entre níveis. Isso gerava risco de que evidências contextuais migrassem, por omissão ou conveniência, para posições de referência que não sustentam.

Após revisão da literatura de sensoriamento remoto e flood mapping, a formulação foi refinada: a camada não declara ground truth operacional, mas organiza uma hierarquia auditável de evidências de referência. Ground truth passa a ser tratado como estado final condicionado — requerendo evento observado, alinhamento temporal e espacial, anotação qualificada e revisão independente — não como premissa disponível.

A camada permite distinguir:
- evidência contextual (fornece contexto físico-ambiental, não valida fenômeno);
- proxy auditável (índice heurístico documentado, não rótulo operacional);
- candidato forte de referência (convergência multi-fonte, ainda sem validação completa);
- ground truth operacional bloqueado (estado explícito de bloqueio com razões auditáveis).

A distinção importa porque um projeto que não a faz corretamente corre o risco de citar como validação o que é, na melhor das hipóteses, evidência de contexto. O Protocolo C é a resposta metodológica a esse risco dentro do REV-P.

Veja [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](protocolo_c_construcao_referencia_operacional.md) para a formulação completa.

---

## Próximas fases previstas

1. **Fase atual**: Documentação metodológica da camada de referência e Protocolo C
2. **Fase seguinte**: Aquisição de registros de revisão humana (v1hb) para patches candidatos
3. **Fase futura**: Reconciliação CRS e validação spatial de sobreposição (requer desbloqueio B1)
4. **Fase operacional**: Definição de limiar de concordância inter-fonte para elevação a operacional (requer aprovação stakeholder e evidência observacional direta)

---

## Referências internas

- [`datasets/contextual_reference_layer_registry.csv`](../../datasets/contextual_reference_layer_registry.csv) — registro de referências por patch
- [`datasets/schemas/contextual_reference_layer_schema.csv`](../../datasets/schemas/contextual_reference_layer_schema.csv) — schema de campos
- [`datasets/external_evidence_registry.csv`](../../datasets/external_evidence_registry.csv) — inventário de evidências externas
- [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](protocolo_c_construcao_referencia_operacional.md) — Protocolo C: formulação madura com lições da literatura
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](patch_lineage_and_grounding.md) — linhagem e bloqueadores
- [`docs/metodologia_cientifica/research_datasets_and_artifacts.md`](research_datasets_and_artifacts.md) — cadeia de rastreabilidade
