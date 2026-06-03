# Protocolo C — aquisição e qualificação de candidatos a ground reference

## 1. Motivação

A etapa anterior do Protocolo C definiu a hierarquia de evidências: o que conta como evidência contextual, proxy auditável, candidato forte de referência e bloqueio operacional. Essa hierarquia organizou o vocabulário e os guardrails. Esta etapa começa a construir a trilha material para encontrar referências fortes por evento, fonte e patch.

O objetivo não é declarar ground truth — é preparar o caminho para saber quando uma referência pode ser promovida. Isso significa registrar quais eventos são candidatos, quais fontes cobrem cada evento, e qual é o vínculo potencial entre evento, fonte e patch. Tudo isso como metadados, sem baixar rasters pesados, sem executar pipeline espacial e sem gerar labels.

A distinção entre esta etapa e operacionalização é explícita: ao final desta etapa, o projeto terá um registro auditável de candidatos e bloqueadores ativos por patch. Não terá ground truth.

---

## 2. O que conta como candidato a ground reference

Um candidato a ground reference não é qualquer evidência de suscetibilidade ou contexto físico-ambiental. É uma combinação específica de elementos que, quando todos reunidos, permite afirmar que um patch pode ser associado a um evento observado com qualidade metodológica suficiente.

Os elementos necessários para um candidato ser considerado elegível para busca de referência:

**Evento confirmado**
Há documentação de que um evento de inundação, alagamento ou fenômeno hídrico ocorreu em uma data e área que potencialmente cobrem o patch.

**Janela temporal**
A data ou intervalo do evento é suficientemente preciso para que uma imagem de satélite possa ser identificada como próxima ao evento (antes, durante ou imediatamente após).

**Fonte externa**
Existe pelo menos uma fonte que documenta o evento: relatório oficial, produto operacional, dataset anotado, registro de defesa civil.

**Cobertura espacial**
A fonte cobre ou potencialmente cobre a área geográfica do patch — mesmo que a sobreposição exata ainda não tenha sido verificada.

**Tipo de fonte**
A fonte pertence a uma família com força metodológica suficiente para servir como referência. Fontes apenas contextuais ou apenas modeladas não são suficientes isoladamente.

**Incerteza documentada**
As limitações e o grau de confiança da fonte estão documentados. Uma fonte sem incerteza declarada é suspeita, não forte.

**Revisão supervisora**
Há previsão de revisão por especialista ou humano qualificado antes de uso como referência operacional.

**Vínculo com patch**
O evento e a fonte podem ser razoavelmente associados ao patch por geometria, nome de município, bacia hidrográfica ou equivalente.

A ausência de qualquer elemento acima bloqueia a promoção do candidato.

---

## 3. Tipos de fonte por força metodológica

### FIELD_OBSERVATION
Observação direta de campo, relatório georreferenciado, nota de vistoria de defesa civil com coordenadas.

**Permite afirmar:** que um fenômeno foi observado diretamente em um ponto ou área específica.
**Não permite afirmar:** generalização para patches sem correspondência espacial verificada.

### OFFICIAL_OBSERVED_FLOOD_MAP
Mapa oficial de área inundada produzido por órgão governamental (defesa civil, proteção civil, agência ambiental) para um evento específico.

**Permite afirmar:** que a área mapeada foi classificada como inundada pelo órgão emissor para aquele evento.
**Não permite afirmar:** que a classificação é perfeita (podem existir omissões, erro de data, escala).

### EXPERT_ANNOTATED_HIGH_RES_IMAGE
Imagem de alta resolução (≤3 m) interpretada e anotada por especialista para delimitação de área inundada.

**Permite afirmar:** que a geometria anotada reflete julgamento especializado sobre a extensão do evento.
**Não permite afirmar:** que a anotação está livre de erro ou que cobre 100% da área afetada.

### OPERATIONAL_FLOOD_PRODUCT
Produto gerado automaticamente por algoritmos de detecção em tempo quase-real (ex.: Copernicus EMS Rapid Mapping, Global Flood Monitoring baseado em Sentinel-1). Inclui camadas de likelihood e incerteza.

**Permite afirmar:** que o algoritmo classificou a área como inundada com determinado grau de confiança para determinado período, com as limitações do algoritmo documentadas.
**Não permite afirmar:** que o produto é equivalente a observação de campo. O produto é algorítmico — erros de comissão e omissão existem e devem ser tratados.

### HAND_LABELED_REMOTE_SENSING_DATASET
Dataset de pesquisa com imagens de satélite manualmente anotadas por especialistas para treinamento ou validação. Exemplos: Sen1Floods11, Kuro Siwo, UFO.

**Permite afirmar:** que os patches desse dataset têm labels de alta confiança para aqueles eventos específicos, produzidos com protocolo de anotação documentado.
**Não permite afirmar:** que os labels transferem diretamente para outros patches, regiões ou datas sem validação de transferibilidade.

### MODELLED_SUSCEPTIBILITY_LAYER
Camada de suscetibilidade ou risco produzida por modelo (hidrológico, hidrodinâmico, multicritério). Pode ser útil para triagem e contexto.

**Permite afirmar:** que o modelo classifica a área como de maior/menor suscetibilidade conforme os parâmetros e hipóteses do modelo.
**Não permite afirmar:** que a área de fato sofreu inundação em qualquer evento, nem que a classificação equivale a observação.

### HYDROGEOMORPHOLOGICAL_CONTEXT
Dados topográficos, de drenagem, uso do solo, hidrografia e geologia que fornecem contexto físico-ambiental.

**Permite afirmar:** contexto de exposição estrutural (terreno baixo, próximo a curso d'água, cobertura impermeável).
**Não permite afirmar:** que um evento ocorreu, que o patch foi inundado, ou que o contexto se traduz em suscetibilidade validada.

### SENTINEL_STRUCTURAL_EVIDENCE
Embeddings DINOv2, diagnósticos estruturais, coerência Sentinel, índice GIS multicritério.

**Permite afirmar:** características estruturais visuais dos patches Sentinel e comparação relativa entre patches.
**Não permite afirmar:** que o patch foi inundado, que o embedding corresponde a um estado de inundação, ou que o índice mede vulnerabilidade observada.

### REVIEW_GATE_EVIDENCE
Feedback de especialista ou revisor humano sobre patches candidatos, registrado no protocolo de revisão supervisora do REV-P.

**Permite afirmar:** que o revisor observou características visuais ou contextuais consistentes com a hipótese de exposição hídrica.
**Não permite afirmar:** que a revisão substitui evidência observacional de evento.

---

## 4. Critérios mínimos para elegibilidade operacional

Para que um par patch-evento seja considerado elegível para uso como referência operacional, todos os seguintes critérios precisam ser satisfeitos:

| Critério | Descrição |
|---|---|
| Evento confirmado | Há confirmação documentada de que o evento ocorreu |
| Data ou intervalo do evento | Precisão temporal mínima: intervalo de dias (não apenas mês ou ano) |
| Aquisição compatível | Imagem ou produto adquirido dentro da janela temporal do evento |
| Geometria disponível | Delimitação espacial do evento disponível ou interpretável |
| CRS conhecido | Sistemas de referência coordenados compatíveis ou reconciliados |
| Resolução compatível | Resolução espacial suficiente para discriminar patch individual |
| Cobertura do patch | A fonte cobre a bounding box do patch (confirmado por sobreposição) |
| Fonte rastreável | Origem, versão, data e emissor da fonte documentados |
| Incerteza documentada | Limitações e grau de confiança da fonte declarados |
| Revisão supervisora | Revisão por especialista ou protocolo qualificado executado |

A ausência de qualquer item acima gera bloqueio com razão documentada no registry de vínculos.

---

## 5. Bloqueios de promoção

Os seguintes estados bloqueiam um vínculo patch-evento-fonte de avançar para ground reference candidate ou operacional:

**Bloqueios críticos** — impedem qualquer promoção:
- Fonte apenas contextual sem evento associado
- Fonte apenas modelada tratada como observação
- Ausência de evento confirmado
- Ausência de data do evento
- Ausência de geometria do evento
- Ausência de cobertura do patch pela fonte
- Temporalidade incompatível (imagem muito distante do evento)
- CRS inconsistente sem plano de reconciliação

**Bloqueios intermediários** — impedem promoção acima de CONTEXTUAL_ONLY:
- Patch fora da área de cobertura da fonte
- Dependência exclusiva de embedding DINO/cluster
- Dependência exclusiva de índice espectral (NDWI, NDBI, MNDWI)
- Dependência exclusiva de índice GIS multicritério
- Ausência de revisão supervisora
- Fonte não adquirida localmente sem alternativa documentada

**Bloqueios de qualidade** — impedem promoção para operacional:
- Incerteza da fonte não documentada
- Conflito entre fontes independentes sem resolução
- Dependência de uma única fonte sem corroboração
- Alinhamento espacial apenas estimado, não verificado
- Revisão de qualidade não concluída

---

## 6. Relação com os datasets de referência da literatura

O campo de flood mapping tem produzido datasets que o REV-P pode usar como modelo metodológico — não como dado aplicável diretamente, mas como exemplo de como construir referência válida.

**Sen1Floods11**
Usa Sentinel-1 e Sentinel-2 com labels manuais de alta confiança para 11 eventos globais. Mostra que ground truth de inundação requer: evento específico + imagem próxima + anotação especializada + protocolo documentado. O custo metodológico é real. O REV-P deve usar Sen1Floods11 como parâmetro de qualidade mínima esperada, não como dado transferível. Referência metodológica a ser citada na versão acadêmica.

**Kuro Siwo**
Dataset SAR multitemporal com anotações manuais para múltiplos eventos globais. Ressalta a importância da dimensão temporal: ground truth de inundação requer cobertura temporal antes, durante e após o evento — não apenas um snapshot estático. O REV-P atualmente opera apenas com snapshots Sentinel estáticos, o que é um bloqueador temporal explícito. Referência metodológica a ser citada na versão acadêmica.

**Urban Flood Observations (UFO)**
Dataset urbano com PlanetScope (~3 m) e anotação humana de inundação/não inundação para eventos específicos. Relevante porque trabalha na escala urbana de patch, próxima ao REV-P. Mostra que resolução e escala importam: 3 m permite discriminação que 10 m (Sentinel-2) dificulta. Para o REV-P, o uso de Sentinel-2 a 10 m é uma limitação real para anotação fina. Referência metodológica a ser citada na versão acadêmica.

**Copernicus EMS / Global Flood Monitoring (GFM)**
Produto operacional Sentinel-1 que gera mapas de inundação com likelihood por evento. Pode ser uma fonte candidata para eventos específicos nas três regiões do REV-P — desde que o uso seja feito com: versão do produto documentada, evento e período específicos, limiar de likelihood explicitado, e reconhecimento de que é produto algorítmico (não observação direta). Não equivale a verdade observacional automática. Referência metodológica a ser citada na versão acadêmica; aquisição não executada localmente.

**Accuracy assessment em sensoriamento remoto**
A avaliação de acurácia clássica pressupõe que amostras de referência têm qualidade, independência espacial, alinhamento temporal e procedência documentada. Sem esses elementos, métricas de desempenho (F1, recall, IoU, Kappa) perdem significado. O REV-P registra explicitamente que nenhuma dessas condições é satisfeita atualmente para nenhum patch — e esta etapa começa a identificar se e como elas podem ser satisfeitas no futuro.

O REV-P usa esses exemplos como modelo metodológico do que é necessário para construir referência operacional, não como claim de que já possui dados equivalentes.

---

## 7. Saída desta etapa

A saída desta etapa é um conjunto de registros metadata-only que organiza:

- **Registry de eventos candidatos** (`flood_event_candidate_registry.csv`): lista de eventos de inundação que podem ser relevantes para as três regiões do REV-P, com status de confirmação, fonte confirmatória e elegibilidade para busca de referência. Inclui referências metodológicas externas marcadas como METHOD_REFERENCE_ONLY.

- **Registry de vínculos patch-evento-fonte** (`patch_event_reference_link_registry.csv`): mapeamento entre patches do corpus DINO, eventos candidatos e fontes de referência. Para cada combinação, registra: alinhamento temporal, alinhamento espacial, força da observação, status de revisão supervisora, se DINO foi usado apenas como suporte, status de candidatura e bloqueadores ativos.

- **Status de elegibilidade por vínculo**: cada linha do registry de vínculos tem `reference_candidate_status` e `promotion_allowed`. No estado atual, todos os vínculos permanecem com `promotion_allowed=false` e `reference_candidate_status` conservador.

- **Claims permitidos e proibidos por vínculo**: campo `allowed_claim` com o que pode ser afirmado e `forbidden_claim` com o que está vedado para cada combinação específica.

- **Trilha de auditoria**: os campos de cada registry permitem que um revisor externo entenda, sem acesso aos dados pesados, por que um candidato está bloqueado e o que seria necessário para desbloquear.

---

## 8. O que esta etapa ainda não faz

Esta etapa é explicitamente metadata-only. Ela não:

- Baixa rasters pesados, GeoTIFFs de evento ou produtos SAR
- Gera labels supervisionados de nenhum tipo
- Cria dataset de treinamento
- Executa classificação ou inferência de modelo
- Desbloqueia Protocolo B
- Declara ground truth operacional para nenhum patch
- Executa pipeline espacial pesado (sobreposição, reprojection, rasterização)
- Transforma embedding DINO em label
- Transforma cluster em classe
- Promove suscetibilidade contextual a verdade observada

O que a etapa faz é criar a documentação auditável que torna visível a distância entre o estado atual do projeto e o estado necessário para referência operacional. Essa documentação é a contribuição desta etapa — não mais.

---

## Referências internas

- [`docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md`](protocolo_c_fechamento_evidencias_ground_reference.md) — etapa seguinte: gates de promoção, matriz de lacunas e decisão formal de promoção
- [`docs/metodologia_cientifica/protocolo_c_revisao_supervisora_referencia.md`](protocolo_c_revisao_supervisora_referencia.md) — protocolo de revisão supervisora que se aplica quando houver candidatos suficientes
- [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](protocolo_c_construcao_referencia_operacional.md) — hierarquia de evidências e critérios gerais
- [`docs/metodologia_cientifica/camada_referencia_contextual_validada.md`](camada_referencia_contextual_validada.md) — guardrails por status e claims por patch
- [`datasets/flood_event_candidate_registry.csv`](../../datasets/flood_event_candidate_registry.csv) — registry de eventos candidatos
- [`datasets/patch_event_reference_link_registry.csv`](../../datasets/patch_event_reference_link_registry.csv) — registry de vínculos patch-evento-fonte
- [`datasets/schemas/flood_event_candidate_schema.csv`](../../datasets/schemas/flood_event_candidate_schema.csv) — schema do registry de eventos
- [`datasets/schemas/patch_event_reference_link_schema.csv`](../../datasets/schemas/patch_event_reference_link_schema.csv) — schema do registry de vínculos
