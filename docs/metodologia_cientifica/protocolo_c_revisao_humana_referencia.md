# Protocolo C — revisão humana de referência

## 1. Papel da revisão humana

A revisão humana no Protocolo C é uma etapa de auditoria e curadoria metodológica — não uma forma de criar labels arbitrários ou atribuir classes de forma subjetiva. Ela existe para verificar, de forma qualificada e documentada, se a evidência reunida nos gates anteriores é coerente o suficiente para avançar na hierarquia de candidatura a ground reference.

A revisão humana não substitui evento documentado. Não é equivalente a observação de campo. Não cria ground truth por si só. O revisor verifica a consistência entre a evidência disponível e a candidatura proposta — e registra sua decisão com justificativa, claim permitido e claim proibido.

O resultado da revisão é sempre um registro auditável: quem revisou (em termos de papel metodológico), o que foi avaliado, qual foi a decisão, e por que. Revisões sem registro não têm validade metodológica no Protocolo C.

A revisão humana corresponde ao gate G7 da sequência de promoção definida em `protocolo_c_fechamento_evidencias_ground_reference.md`. G7 não pode ser satisfeito sem revisão executada e registrada.

---

## 2. Entradas da revisão

Para que uma revisão humana possa ser conduzida, o revisor precisa ter acesso a:

**Patch Sentinel**
O recorte de imagem Sentinel associado ao patch — bandas relevantes, data de aquisição, região e identificador canônico (ex.: REC_01). O revisor deve poder visualizar e interpretar a imagem.

**Evento candidato**
O evento de inundação, alagamento ou fenômeno hídrico documentado no `flood_event_candidate_registry.csv` — com nome, data, região, município e status de confirmação.

**Fonte externa**
A fonte que documenta o evento — relatório oficial, produto operacional, dataset acadêmico. O revisor deve ter acesso à fonte ou à sua descrição rastreável no `ground_reference_evidence_source_registry.csv`.

**Metadados de data**
Data de aquisição da imagem Sentinel e data do evento (início e fim), para verificação de alinhamento temporal.

**Metadados espaciais**
Bounding box do patch e, quando disponível, geometria do evento — para verificação de sobreposição espacial.

**Incerteza documentada**
Limitações da fonte, resolução, cobertura e erros potenciais — documentados no registry de fontes.

**Suporte DINO (opcional, nunca exclusivo)**
Diagnósticos estruturais do embedding DINOv2, se disponíveis — outlier score, vizinhança, medoid status — como informação de suporte para orientar atenção visual. O uso de DINO como suporte deve ser registrado com `dino_support_used=true` e limitação explicitada.

Revisão baseada exclusivamente em DINO, cluster ou índice GIS não satisfaz G7. DINO é suporte, não substituto de evidência observacional.

---

## 3. Decisões possíveis

O revisor registra uma das seguintes decisões no `human_reference_review_registry.csv`:

**ACCEPT_AS_CONTEXTUAL_REFERENCE**
A evidência disponível é coerente como referência contextual — fornece contexto físico-ambiental rastreável, mas não documenta evento observado. Resultado: o par patch-fonte permanece em CONTEXTUAL_EVIDENCE. Nenhuma promoção acima deste nível.

**ACCEPT_AS_AUDITABLE_PROXY**
A evidência disponível inclui proxy rastreável (ex.: índice GIS multicritério com fontes documentadas) que permite comparação estrutural entre patches, sem afirmar observação de fenômeno. Resultado: o par avança para AUDITABLE_REFERENCE_PROXY. Não habilita treino supervisionado.

**MARK_AS_STRONG_REFERENCE_CANDIDATE**
Evento confirmado, fonte rastreável, alinhamento temporal estimado, sobreposição espacial estimada, incerteza documentada, sem conflito relevante. O par tem evidência suficiente para ser considerado candidato forte, aguardando validação externa. Resultado: o par avança para STRONG_REFERENCE_CANDIDATE. Ainda não habilita treino supervisionado.

**BLOCK_OPERATIONAL_PROMOTION**
Qualquer gate crítico está aberto: sem evento confirmado, sem temporalidade, sem cobertura espacial, conflito não resolvido, fonte apenas contextual ou modelada. A promoção está bloqueada. O bloqueio e sua razão são registrados.

**REQUEST_ADDITIONAL_EVIDENCE**
A evidência disponível é insuficiente para decisão, mas há caminho identificado para coleta de evidência adicional. O revisor documenta o que está faltando e o que seria necessário para revisão subsequente.

**REJECT_AS_INSUFFICIENT**
A evidência disponível é insuficiente e não há caminho claro para coleta adicional. O par é rejeitado como candidato no estado atual. A razão da rejeição é documentada.

**METHOD_REFERENCE_ONLY**
A linha é referência metodológica externa e não se aplica diretamente a patches do REV-P. Usado para linhas de Sen1Floods11, Kuro Siwo, UFO e Copernicus/GFM no registry.

---

## 4. Critérios de bloqueio

Os seguintes critérios impõem BLOCK_OPERATIONAL_PROMOTION obrigatoriamente:

**Conflito temporal**
A data da imagem Sentinel e a data do evento têm distância incompatível com alinhamento válido — ex.: imagem de 2020 e evento de 2022, ou snapshot estático sem série antes-durante-depois.

**Conflito espacial**
A fonte cobre região diferente do patch, ou a sobreposição espacial é incompatível com a bounding box do patch.

**Fonte apenas modelada**
A única fonte disponível é um modelo de suscetibilidade, modelo hidrológico ou camada de risco — sem observação direta, anotação especializada ou produto operacional associado a evento específico.

**Ausência de evento**
Não há evento confirmado. A revisão não pode promover candidatura sem G1 satisfeito.

**Cobertura parcial sem resolução**
A fonte cobre apenas parte do patch, sem possibilidade de determinar se a parte afetada é relevante para o fenômeno.

**Incerteza não documentada**
A fonte não tem incerteza declarada. Revisão não pode promover fonte opaca.

**Revisão visual inconclusiva**
O revisor não consegue determinar, pela imagem, qualquer informação consistente com o fenômeno candidato. Inconclusividade não promove — é registrada como bloqueio ou solicitação de evidência adicional.

**Dependência exclusiva de DINO, cluster ou índice**
A única "evidência" disponível é embedding, cluster, NDWI/NDBI, MNDWI ou índice GIS. Nenhuma dessas fontes fecha G1, G3 ou G4. Revisão baseada exclusivamente nessas fontes resulta em BLOCK_OPERATIONAL_PROMOTION.

---

## 5. Registro da decisão

Toda revisão executada deve produzir uma entrada no `human_reference_review_registry.csv` com os seguintes campos mínimos preenchidos:

**reviewer_role:** papel metodológico do revisor — METHODOLOGICAL_REVIEWER, DOMAIN_REVIEWER, GIS_REVIEWER, REMOTE_SENSING_REVIEWER ou FUTURE_EXTERNAL_REVIEWER. Identificação pessoal não é armazenada no registro público.

**review_date:** data da revisão em formato ISO 8601 (YYYY-MM-DD). Se revisão não executada, o campo recebe NOT_EXECUTED.

**reviewed_materials:** lista dos materiais efetivamente consultados durante a revisão — imagem Sentinel, fonte, relatório, produto. Revisão sem materiais documentados não é válida.

**temporal_consistency:** avaliação de alinhamento temporal — CONSISTENT, PARTIAL, INCONSISTENT ou NOT_ASSESSED.

**spatial_consistency:** avaliação de sobreposição espacial — CONSISTENT, PARTIAL, INCONSISTENT ou NOT_ASSESSED.

**source_consistency:** avaliação de coerência da fonte com o evento — CONSISTENT, PARTIAL, INCONSISTENT ou NOT_ASSESSED.

**visual_consistency:** avaliação de consistência visual da imagem Sentinel com o fenômeno candidato — CONSISTENT, PARTIAL, INCONSISTENT ou NOT_ASSESSED.

**dino_support_used:** se DINO foi consultado como suporte (true/false). Se true, `dino_support_limitation` deve deixar claro que DINO não cria label nem ground truth.

**review_decision:** uma das decisões definidas na seção 3.

**confidence_level:** LOW, MODERATE, HIGH ou NOT_APPLICABLE — avaliação do revisor sobre sua própria confiança na decisão.

**promotion_allowed:** true ou false — derivado da decision. Apenas MARK_AS_STRONG_REFERENCE_CANDIDATE pode resultar em promotion_allowed=true, e apenas se todos os gates críticos (G1, G3, G4, G6) estiverem satisfeitos.

**blocked_reason:** razão do bloqueio quando promotion_allowed=false — obrigatório.

**allowed_claim:** o que pode ser afirmado a partir desta revisão.

**forbidden_claim:** o que está categoricamente proibido a partir desta revisão.

---

## 6. Relação com anotação futura

Se, no futuro, houver imagem pós-evento de alta resolução disponível para um patch específico — ou produto operacional confirmado com evento documentado — pode haver anotação manual assistida por especialista.

Essa anotação futura é uma nova etapa metodológica, distinta da revisão humana descrita neste documento. Ela não é consequência automática de uma revisão positiva. Deve ser tratada com:

- Protocolo de anotação documentado.
- Especificação da imagem usada (data, sensor, resolução).
- Especificação do evento de referência.
- Identificação do anotador (papel metodológico).
- Controle de qualidade e revisão independente.
- Registro de incerteza da anotação.

Mesmo que a anotação futura seja executada, o resultado é STRONG_REFERENCE_CANDIDATE ou STRONG_REFERENCE_READY_FOR_EXTERNAL_VALIDATION — não ground truth operacional automático. A promoção final para uso operacional requer validação independente e decisão formal no registry de promoção.

A anotação futura não retroativamente valida patches do estado atual. Ela cria novos vínculos com novas entradas no registry, rastreáveis de forma independente.

---

## 7. Papel da revisão no contexto do Protocolo C

A revisão humana não é o último passo do Protocolo C. É o penúltimo: ela satisfaz G7 e habilita G9 (decisão formal de promoção). A sequência é:

```
G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7 (revisão humana) → G8 (corroboração) → G9 (decisão)
```

Sem G7 satisfeito, G9 não pode ser executado. Sem G9 executado, nenhuma promoção é formal.

No estado atual do REV-P, nenhum patch tem G1, G3, G4 ou G7 satisfeitos. O Protocolo B permanece bloqueado. Esta etapa de revisão humana existe como protocolo preparado para quando os gates anteriores forem satisfeitos — não como declaração de que eles já foram.

---

## Referências internas

- [`docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md`](protocolo_c_fechamento_evidencias_ground_reference.md) — gates de promoção e níveis de fechamento
- [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](protocolo_c_aquisicao_ground_reference.md) — etapa de aquisição: eventos e vínculos candidatos
- [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](protocolo_c_construcao_referencia_operacional.md) — Protocolo C: formulação completa
- [`datasets/human_reference_review_registry.csv`](../../datasets/human_reference_review_registry.csv) — registry de revisões humanas executadas ou placeholder
- [`datasets/schemas/human_reference_review_schema.csv`](../../datasets/schemas/human_reference_review_schema.csv) — schema de campos do registry de revisão humana
- [`datasets/reference_promotion_decision_registry.csv`](../../datasets/reference_promotion_decision_registry.csv) — registry de decisões de promoção
