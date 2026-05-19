# Protocolo C — fechamento de evidências para ground reference

## 1. Motivação

A etapa anterior do Protocolo C mapeou eventos candidatos, fontes e vínculos patch-evento-fonte. O resultado foi um conjunto de registros auditáveis que organizam o que existe, o que está ausente e por que nenhum patch pode ainda ser associado a referência operacional. Essa documentação é a base — mas não o destino.

Esta etapa define como as lacunas serão fechadas. Não é uma declaração de que evidência será encontrada — é a organização metodológica de quais portas precisam ser abertas, em que sequência, e com que tipo de evidência, para que um vínculo patch-evento-fonte possa eventualmente ser considerado elegível para promoção a ground reference.

A contribuição desta etapa é dupla: define os gates de promoção em sequência auditável e documenta, para cada região e par patch-fonte, onde exatamente o processo está bloqueado e o que seria necessário para avançar. O objetivo não é declarar que o REV-P já tem ground truth operacional — é tornar visível o que falta para que isso seja possível no futuro.

Ground truth operacional não está estabelecido no estado atual do REV-P. Esta etapa não muda esse fato. Ela organiza a trilha que tornaria a mudança auditável.

---

## 2. O que significa "fechar evidência"

Fechar evidência não é declarar ground truth. É satisfazer, de forma documentada e auditável, um conjunto de critérios mínimos que permitem afirmar que um par patch-evento-fonte tem qualidade metodológica suficiente para ser considerado candidato a referência operacional.

Os critérios de fechamento são:

**Evento confirmado**
Existe documentação de que um fenômeno hídrico específico (inundação, alagamento, enchente) ocorreu em data e área que cobrem ou podem cobrir o patch. Documentação por fonte oficial, acadêmica, operacional ou documental rastreável.

**Fonte rastreável**
A fonte que documenta o evento existe, tem proveniência conhecida, versão documentada, emissor identificado e condições de uso registradas. Fontes apenas contextuais ou apenas modeladas não fecham este critério isoladamente.

**Janela temporal compatível**
A imagem, o produto ou a fonte foram adquiridos dentro de uma janela temporal compatível com o evento — antes (pré-evento), durante ou imediatamente após. Snapshots estáticos sem vínculo temporal com evento não fecham este critério.

**Cobertura espacial do patch**
A fonte cobre ou potencialmente cobre o bounding box do patch. Cobertura confirmada por sobreposição geométrica ou, em estágio anterior, estimada por região e verificada por especialista.

**CRS compatível**
Os sistemas de referência coordenados do patch e da fonte são compatíveis ou foram reconciliados de forma documentada. CRS incompatível sem plano de reconciliação é bloqueador crítico.

**Geometria ou interpretação espacial**
O evento tem geometria disponível — polígono de área afetada, raster de likelihood, bounding box rastreável — ou há interpretação espacial qualificada por especialista.

**Tipo de observação**
A fonte é observacional (campo, anotação especializada) ou operacional algorítmica com incerteza documentada. Fontes apenas modeladas ou apenas contextuais não fecham isoladamente o critério de observação.

**Incerteza documentada**
As limitações da fonte estão declaradas: resolução, cobertura, erros de omissão/comissão, hipóteses do modelo, período de validade. Fonte sem incerteza declarada não pode ser tratada como forte.

**Revisão humana ou de especialista**
Um revisor qualificado leu o patch, leu a fonte, verificou alinhamento temporal e espacial, e registrou decisão motivada. Revisão não executada não fecha este critério.

**Decisão auditada**
A decisão de promover ou bloquear é registrada formalmente, com gates satisfeitos e falhados documentados, allowed_claim e forbidden_claim explicitados, e reasoning permanente no registry.

A ausência de qualquer critério acima gera um gate aberto. A promoção só ocorre quando todos os gates críticos estiverem fechados.

---

## 3. Gates de promoção

Os gates são a sequência de verificações que cada par patch-evento-fonte precisa satisfazer para avançar na hierarquia de candidatura. Cada gate tem pré-condições, o que satisfaz e o que bloqueia.

### G0_PATCH_LINEAGE

**O que verifica:** o patch existe como entidade geográfica com região, bounding box definida, origem documentada e manifest rastreável no repositório.

**O que satisfaz:** presença no manifest canônico do corpus; região e município identificados; origin_dataset documentado.

**O que bloqueia:** patch sem geometria canônica (ex.: placeholders CUR_08–14); ausência de manifest; patch sem rastreabilidade de origem.

**O que G0 permite afirmar:** o patch é uma unidade auditável com existência territorial documentada.

**O que G0 não permite afirmar:** que o patch foi afetado por qualquer evento; que o patch é caso positivo de inundação.

---

### G1_EVENT_CONFIRMATION

**O que verifica:** existe documentação de que um fenômeno hídrico ocorreu em data e área potencialmente compatíveis com o patch.

**O que satisfaz:** relatório oficial de defesa civil; produto operacional Copernicus EMS com evento associado; registro municipal; publicação acadêmica com evento específico; cobertura de imprensa rastreável com data e local.

**O que bloqueia:** ausência de evento documentado; evento apenas hipotético ou implícito; evento não separado de outros fenômenos (ex.: deslizamento registrado como inundação em Petrópolis); placeholder de busca.

**O que G1 permite afirmar:** houve fenômeno documentado na região que pode ser candidato ao vínculo com o patch.

**O que G1 não permite afirmar:** que o patch foi inundado; que o evento afetou especificamente aquele patch.

---

### G2_SOURCE_AVAILABILITY

**O que verifica:** a fonte que documenta o evento existe, é rastreável e tem condições de uso conhecidas — mesmo que ainda não adquirida localmente.

**O que satisfaz:** identificação de produto Copernicus GFM para o evento; localização de registro de defesa civil; identificação de dataset acadêmico específico.

**O que bloqueia:** fonte completamente desconhecida; busca não executada; fonte apenas mencionada sem referência rastreável.

**O que G2 permite afirmar:** existe fonte candidata que documenta o evento, com proveniência identificada.

**O que G2 não permite afirmar:** que a fonte é suficiente para referência operacional; que a fonte está disponível localmente.

---

### G3_TEMPORAL_ALIGNMENT

**O que verifica:** a imagem Sentinel ou o produto externo está dentro de uma janela temporal compatível com o evento — antes (pré-evento), durante ou imediatamente após.

**O que satisfaz:** imagem Sentinel com data dentro de janela de dias em relação ao evento; produto operacional com timestamp do evento documentado.

**O que bloqueia:** imagem estática sem vínculo temporal com evento; distância temporal superior à janela de validade do evento; evento com data UNKNOWN; série temporal ausente (apenas snapshot pré-evento sem cobertura durante/após).

**O que G3 permite afirmar:** existe ou pode existir cobertura temporal compatível entre imagem e evento.

**O que G3 não permite afirmar:** que a imagem capturou o evento; que a inundação está visível na imagem.

---

### G4_SPATIAL_ALIGNMENT

**O que verifica:** a fonte cobre o bounding box do patch ou tem geometria compatível com a localização do patch.

**O que satisfaz:** sobreposição geométrica confirmada; polígono de evento que intersecta ou inclui o patch; produto raster com cobertura da célula do patch.

**O que bloqueia:** fonte com cobertura regional mas sem verificação de sobreposição com o patch específico; patch fora da área documentada do evento; resolução da fonte incompatível com discriminação de patch individual.

**O que G4 permite afirmar:** há sobreposição potencial ou confirmada entre a fonte e a área do patch.

**O que G4 não permite afirmar:** que o patch está classificado como inundado pela fonte; que a sobreposição implica rótulo.

---

### G5_SOURCE_STRENGTH

**O que verifica:** a natureza da observação da fonte — observacional direta, anotação especializada, produto operacional algorítmico, modelado ou apenas contextual.

**O que satisfaz:** fonte observacional (vistoria de campo, anotação por especialista em imagem de alta resolução); produto operacional com likelihood documentado.

**O que bloqueia como fonte isolada:** fonte apenas modelada (suscetibilidade); fonte apenas contextual (topografia, drenagem, uso do solo); embedding DINO; índice GIS; índice espectral.

**O que G5 permite afirmar:** a natureza da observação da fonte é conhecida e documentada.

**O que G5 não permite afirmar:** que a força da fonte é suficiente sozinha; que modelos ou contexto são equivalentes a observação.

---

### G6_UNCERTAINTY_AND_LIMITATIONS

**O que verifica:** as limitações da fonte estão declaradas — resolução, cobertura, erros potenciais, hipóteses do modelo, período de validade.

**O que satisfaz:** documentação de incerteza no registro da fonte; limitações explicitadas no allowed_use/forbidden_use; resolução espacial e temporal declaradas.

**O que bloqueia:** fonte sem incerteza declarada; fonte tratada como verdade automática; produto operacional usado sem reconhecimento de erros de comissão/omissão.

**O que G6 permite afirmar:** a incerteza da fonte está documentada e pode ser propagada para a avaliação.

**O que G6 não permite afirmar:** que a incerteza foi eliminada; que a fonte é perfeita.

---

### G7_HUMAN_REVIEW

**O que verifica:** um revisor qualificado leu o patch, leu a fonte, verificou alinhamentos e registrou decisão motivada com allowed_claim e forbidden_claim explicitados.

**O que satisfaz:** revisão executada com protocolo documentado (ver `protocolo_c_revisao_humana_referencia.md`); decisão registrada no registry de revisão humana; feedback coerente com gates anteriores.

**O que bloqueia:** revisão não executada; revisão sem protocolo documentado; revisão baseada exclusivamente em DINO ou índice GIS; decisão sem justificativa; conflito de evidência sem resolução registrada.

**O que G7 permite afirmar:** há avaliação humana qualificada sobre a coerência da evidência.

**O que G7 não permite afirmar:** que a revisão substitui evidência observacional; que revisão visual é equivalente a evento documentado.

---

### G8_INDEPENDENT_CORROBORATION

**O que verifica:** mais de uma fonte independente apoia a referência — eventos corroborados por fontes distintas (relatório oficial + produto operacional + registro local, por exemplo).

**O que satisfaz:** presença de pelo menos duas fontes independentes que confirmam o mesmo evento na mesma área temporal.

**O que bloqueia em uso operacional:** dependência exclusiva de uma única fonte; corroboração apenas entre fontes do mesmo tipo (ex.: dois modelos, dois índices, dois embeddings).

**O que G8 permite afirmar:** a referência tem suporte em múltiplas fontes independentes.

**O que G8 não permite afirmar:** que fontes múltiplas eliminam a incerteza; que corroboração equivale a observação direta.

*Nota: G8 não é bloqueador crítico para EVIDENCE_PARTIALLY_CLOSED, mas é requisito para STRONG_REFERENCE_READY_FOR_EXTERNAL_VALIDATION.*

---

### G9_PROMOTION_DECISION

**O que verifica:** existe uma decisão formal e auditada de promoção ou bloqueio, com todos os gates documentados, claim permitido e proibido registrado, e reasoning permanente no registry de decisão.

**O que satisfaz:** entrada no `reference_promotion_decision_registry.csv` com passed_gates, failed_gates, final_reference_status, allowed_claim e forbidden_claim preenchidos; revisão de consistência metodológica executada.

**O que bloqueia:** ausência de decisão registrada; decisão sem justificativa; decisão que promove ground truth sem satisfazer G1, G3, G4 e G7.

**O que G9 permite afirmar:** o estado metodológico do par patch-evento-fonte foi formalmente avaliado e registrado.

**O que G9 não permite afirmar:** que a decisão é permanente sem revisão; que o registro substitui evidência observacional.

---

## 4. Níveis de fechamento

Os níveis de fechamento traduzem o conjunto de gates satisfeitos em um status de candidatura legível e auditável.

### EVIDENCE_OPEN

Há lacunas relevantes em múltiplos gates críticos. Tipicamente: sem evento confirmado (G1 aberto), sem fonte rastreável (G2 aberto), sem alinhamento temporal (G3 aberto). O par patch-evento-fonte está em estágio inicial de busca.

**Claims permitidos:** o patch está em região com contexto físico-ambiental documentado; a busca de referência ainda não foi concluída.

**Claims proibidos:** qualquer afirmação de observação, inundação, suscetibilidade validada ou candidatura de referência.

---

### EVIDENCE_PARTIALLY_CLOSED

Há evidência contextual documentada (G0, G5 parcialmente, G6 parcialmente) e pelo menos uma fonte rastreável identificada (G2), mas faltam temporalidade (G3), espacialidade (G4), evento confirmado (G1) ou revisão humana (G7).

**Claims permitidos:** o patch está em região coberta por fontes contextuais documentadas; há candidato de fonte identificado que pode ser explorado; a busca de referência está em andamento.

**Claims proibidos:** afirmação de evento observado; flood label; ground truth operacional; candidatura confirmada de referência.

---

### REFERENCE_CANDIDATE_READY_FOR_REVIEW

Gates G0–G6 satisfeitos ou parcialmente satisfeitos. Existe evento identificado (G1), fonte rastreável (G2), alinhamento temporal estimado (G3), sobreposição espacial estimada (G4), força de fonte documentada (G5), incerteza documentada (G6). Falta revisão humana (G7) e decisão formal (G9).

**Claims permitidos:** há candidato de referência suficiente para revisão humana estruturada; a candidatura está aguardando avaliação especializada.

**Claims proibidos:** equivalência com ground truth; uso como label de treinamento sem revisão; afirmação de inundação observada.

---

### STRONG_REFERENCE_READY_FOR_EXTERNAL_VALIDATION

Todos os gates críticos satisfeitos (G0–G7), com corroboração de fonte independente (G8). Falta apenas validação externa independente ou decisão formal (G9). Esta situação não ocorre para nenhum patch do REV-P no estado atual.

**Claims permitidos:** evidência forte documentada; candidato para validação externa ou revisão de par.

**Claims proibidos:** uso operacional sem validação; equivalência automática com observação de campo.

---

### OPERATIONAL_GROUND_TRUTH_NOT_ESTABLISHED

Estado de bloqueio explícito. Qualquer requisito crítico está ausente: G1 não satisfeito (sem evento confirmado), ou G3 (sem temporalidade), ou G4 (sem espacialidade confirmada), ou G7 (sem revisão humana). Nenhum patch do REV-P sai deste estado no momento atual.

**Claims permitidos:** nenhum claim de referência, candidatura, ou observação de evento é permitido. O estado metodológico é de bloqueio ativo.

**Claims proibidos:** flood detection; flood prediction; predição de enchente; detecção de enchente; ground truth operacional; operational ground truth; flood label; label de enchente; training label; supervised class; qualquer claim de suscetibilidade validada.

---

## 5. Relação com revisão humana e anotação

A revisão humana no Protocolo C não é "opinião livre" de quem olha para o patch. É uma etapa protocolada com entradas, critérios, decisões possíveis e registro obrigatório.

A sequência mínima da revisão é:

1. **Leitura do patch Sentinel:** qual região, quais bandas, qual data de aquisição.
2. **Leitura da fonte candidata:** qual evento, qual data, qual organismo emissor, qual resolução.
3. **Verificação de temporalidade:** a imagem do patch é compatível temporalmente com o evento da fonte? Há janela de dias plausível?
4. **Verificação de cobertura espacial:** a fonte cobre o bounding box do patch? Há sobreposição confirmada ou estimada?
5. **Registro de incerteza:** o revisor documenta dúvidas, conflitos e limitações identificadas.
6. **Decisão motivada:** o revisor registra uma das decisões possíveis (ver `protocolo_c_revisao_humana_referencia.md`) com justificativa.
7. **Registro de claim:** o revisor documenta o allowed_claim e o forbidden_claim resultantes.

Se houver conflito entre fontes ou entre imagem e fonte, a revisão não pode promover. O conflito deve ser registrado como bloqueador e permanecer no registry de revisão.

Revisão humana nunca substitui evento documentado. Revisor que vê estrutura visual consistente com inundação sem evento confirmado documenta sua observação como contextual — não como label ou ground truth.

A anotação manual futura (quando e se houver imagem pós-evento de alta resolução e evento confirmado) é uma etapa distinta. Não é consequência automática de revisão visual. Deve ser tratada como nova fase com protocolo próprio.

---

## 6. Relação com DINOv2

O DINOv2 é usado no REV-P como encoder visual congelado para extração de representações estruturais de patches Sentinel. No contexto do Protocolo C de fechamento de evidências, o papel do DINO é limitado e explícito:

**O que DINO pode fazer na etapa de fechamento:**
- Sugerir similaridade estrutural visual entre patches — útil para identificar candidatos com padrão visual semelhante ao de patches mais documentados.
- Apoiar auditoria visual em G7 como informação de suporte — o revisor pode usar diagnósticos de embedding para orientar atenção.
- Identificar outliers estruturais que merecem prioridade na busca de referência.

**O que DINO não pode fazer — em nenhum gate:**
- Fechar G1_EVENT_CONFIRMATION: embedding não confirma evento. Nenhuma representação estrutural documenta que houve inundação.
- Fechar G3_TEMPORAL_ALIGNMENT: similaridade entre embeddings de datas distintas não é alinhamento temporal com evento.
- Fechar G5_SOURCE_STRENGTH como observação direta: embedding é representação — não é observação de campo nem anotação especializada.
- Ser fonte de ground truth em qualquer gate.
- Promover candidato de referência isoladamente.

DINO entra apenas como suporte em G7 — como parte do material que o revisor humano pode usar, não como substituto da revisão. Quando DINO for usado, `dino_used_as_support_only=true` deve estar registrado no vínculo correspondente e a limitação deve ser documentada explicitamente.

Cluster não vira classe. Embedding não vira label. Distância no espaço latente não vira proximidade a evento.

---

## 7. Relação com Protocolo B

O Protocolo B — supervisão com referência validada — permanece bloqueado no estado atual do REV-P. O Protocolo C de fechamento de evidências não desbloqueia automaticamente o Protocolo B.

Para que o Protocolo B seja reavaliado, é necessário:
- Conjunto de pares patch-evento com referência forte documentada (STRONG_REFERENCE_CANDIDATE ou STRONG_REFERENCE_READY_FOR_EXTERNAL_VALIDATION).
- Cobertura suficiente nas três regiões para training e validação com referência independente.
- Ausência de dependência exclusiva de fontes contextuais, modeladas ou estruturais.
- Revisão humana executada e registrada para os candidatos de treinamento e validação.
- Decisão formal de promoção no registry de promoção.

Nenhuma dessas condições é satisfeita no estado atual. O Protocolo C documenta exatamente onde cada condição está bloqueada e o que seria necessário para satisfazê-la.

O Protocolo C não é um "meio caminho para treino". É a documentação auditável da distância entre o que o projeto tem e o que precisaria ter para que supervisão fosse metodologicamente legítima.

---

## 8. Saída da etapa

A saída desta etapa é um conjunto de registros metadata-only que organizam:

- **Matriz de lacunas** (`ground_reference_gap_matrix.csv`): para cada região/par patch-fonte, quais gates estão abertos, qual a evidência faltante, qual é a ação necessária, qual é o risco metodológico e quais são os próximos passos permitidos.

- **Protocolo de revisão humana** (`protocolo_c_revisao_humana_referencia.md`): como a revisão humana será conduzida quando houver candidatos suficientes, com decisões possíveis, critérios de bloqueio e registro obrigatório.

- **Schema de revisão humana** (`schemas/human_reference_review_schema.csv`): campos para registrar cada revisão executada, com reviewer_role, decisão, confidence_level e claims permitidos/proibidos.

- **Registry de revisão humana** (`human_reference_review_registry.csv`): linhas de revisão executadas ou placeholders metodológicos, todas com promotion_allowed=false no estado atual.

- **Schema de decisão de promoção** (`schemas/reference_promotion_decision_schema.csv`): campos para registrar a decisão formal de promoção ou bloqueio por par patch-evento-fonte, com gates satisfeitos/falhados documentados.

- **Registry de decisão de promoção** (`reference_promotion_decision_registry.csv`): decisões formais registradas, todas com promotion_allowed=false e protocol_b_reassessment_allowed=false no estado atual.

---

## 9. O que esta etapa ainda não faz

O parâmetro de qualidade que orienta o Protocolo C é definido por datasets de referência da literatura de flood mapping — Sen1Floods11, Kuro Siwo e UFO — que combinam evento confirmado, sensor específico, anotação qualificada e protocolo documentado. Esses datasets mostram o que seria necessário para referência de alta qualidade. O REV-P não tem equivalente local ainda. Esta etapa documenta exatamente essa distância.

Esta etapa é explicitamente metadata-only. Ela não:

- Baixa rasters pesados, GeoTIFFs de evento ou produtos SAR.
- Gera labels supervisionados de nenhum tipo.
- Cria dataset de treinamento.
- Executa classificação ou inferência de modelo.
- Desbloqueia Protocolo B.
- Declara ground truth operacional para nenhum patch.
- Executa pipeline espacial pesado (sobreposição, reprojeção, rasterização).
- Transforma embedding DINO em label.
- Transforma cluster em classe.
- Promove suscetibilidade contextual a verdade observada.
- Declara que qualquer gate está fechado sem evidência verificada.

O que esta etapa faz é criar a documentação auditável dos gates de promoção, das lacunas por região e dos próximos passos necessários para aquisição futura. Essa documentação não é ground truth — é o mapa do que falta para que ground truth seja metodologicamente possível no futuro.

---

## Referências internas

- [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](protocolo_c_aquisicao_ground_reference.md) — etapa anterior: registro de eventos e vínculos candidatos
- [`docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md`](protocolo_c_revisao_humana_referencia.md) — protocolo de revisão humana
- [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](protocolo_c_construcao_referencia_operacional.md) — Protocolo C: formulação completa
- [`docs/metodologia_cientifica/camada_referencia_contextual_validada.md`](camada_referencia_contextual_validada.md) — hierarquia de status e guardrails por patch
- [`datasets/ground_reference_gap_matrix.csv`](../../datasets/ground_reference_gap_matrix.csv) — matriz de lacunas por região
- [`datasets/human_reference_review_registry.csv`](../../datasets/human_reference_review_registry.csv) — registry de revisões humanas
- [`datasets/reference_promotion_decision_registry.csv`](../../datasets/reference_promotion_decision_registry.csv) — registry de decisões de promoção
- [`datasets/flood_event_candidate_registry.csv`](../../datasets/flood_event_candidate_registry.csv) — registry de eventos candidatos
- [`datasets/patch_event_reference_link_registry.csv`](../../datasets/patch_event_reference_link_registry.csv) — registry de vínculos patch-evento-fonte
