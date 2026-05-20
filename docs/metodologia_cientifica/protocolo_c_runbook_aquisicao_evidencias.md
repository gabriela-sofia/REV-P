# Runbook de aquisição de evidências do Protocolo C

Este runbook é um guia operacional passo a passo para quando a coleta real de evidências observacionais começar. Ele deve ser seguido por qualquer pesquisador que execute a aquisição, garantindo rastreabilidade, conformidade com licenças e preservação dos guardrails metodológicos.

**Esta etapa (v1hm) não executa aquisição real.** O runbook documenta o processo para execução futura autorizada.

---

## 1. Antes de buscar a fonte

**Objetivo**: garantir que a busca seja deliberada, priorizada e conectada a um gate específico do Protocolo C.

### Passo 1.1 — Conferir o registry regional de prontidão
Abrir `datasets/regional_ground_reference_readiness.csv`. Identificar qual região está em foco e qual gate tem maior urgência de fechamento.

Perguntas a responder:
- Qual o `event_confirmation_readiness` da região?
- Qual o `source_availability_readiness`?
- Qual a lacuna crítica registrada em `strongest_missing_evidence`?

### Passo 1.2 — Conferir o plano de aquisição
Abrir `datasets/observational_evidence_acquisition_plan.csv`. Filtrar por região e `acquisition_priority=HIGH`.

Identificar:
- Qual `target_source_name` está no topo da fila?
- Qual `related_protocol_c_gate` ela pode fechar?
- Qual `access_mode` se aplica (PUBLIC_DOWNLOAD, FORMAL_REQUEST, etc.)?

### Passo 1.3 — Conferir o tracker de aquisição
Abrir `datasets/evidence_acquisition_tracker.csv`. Verificar o `acquisition_status` da fonte-alvo.

Estados que indicam que a busca pode prosseguir:
- `NOT_STARTED`: início de primeira busca
- `IDENTIFIED`: fonte já localizada, busca de acesso prossegue
- `REQUEST_REQUIRED`: solicitação formal pendente

Estados que indicam espera:
- `REQUESTED`: aguardando resposta
- `RECEIVED_METADATA_ONLY`: metadado já em mãos, aguardar dado bruto ou prosseguir com metadado

---

## 2. Ao encontrar a fonte

**Objetivo**: documentar imediatamente o que foi encontrado, antes de qualquer julgamento.

### Passo 2.1 — Registrar referência
Anotar URL, referência bibliográfica, número de publicação ou outro identificador único. Nunca anotar apenas "vi no site" — a referência precisa ser reproducível.

### Passo 2.2 — Registrar instituição
Identificar o provedor primário: qual órgão, universidade ou empresa publicou ou mantém a fonte. Se a fonte foi intermediada, registrar tanto o provedor primário quanto o intermediário.

### Passo 2.3 — Registrar licença
Localizar a licença ou termos de uso da fonte. Pode estar em:
- Seção de metadados do portal
- Rodapé ou página de termos do site
- Documentação do dataset (README, ficha técnica)
- Resposta formal da instituição

Preencher `license_status` com valor controlado:
- `PUBLIC_REUSE_ALLOWED` se a licença permite reutilização explícita
- `PUBLIC_VIEW_ONLY` se permite apenas visualização
- `REQUEST_REQUIRED` se exige pedido formal
- `RESTRICTED` se há restrições específicas documentadas
- `UNKNOWN` se a licença não foi encontrada

**Atenção**: `UNKNOWN` é estado temporário, não permanente. É necessário esforço ativo para esclarecer.

### Passo 2.4 — Registrar data
Qual a data da fonte? Qual o período de referência dos dados? Se for produto operacional, qual a data de processamento?

### Passo 2.5 — Registrar escala/resolução
Para rasters: resolução espacial (metros), projeção, número de bandas.
Para vetores: escala de captura, tipo de geometria, método de levantamento.
Para relatórios: área de cobertura, data de levantamento de campo.

### Passo 2.6 — Registrar geometria disponível
A fonte tem geometria associada (shapefile, GeoJSON, WKT, bounding box)? Preencher `geometry_available`:
- `TRUE`: geometria acessível e documentada
- `PARTIAL`: geometria parcialmente documentada
- `FALSE`: sem geometria
- `UNKNOWN`: não avaliado

### Passo 2.7 — Registrar incerteza
Existe documentação de acurácia, taxa de erro, intervalo de confiança ou limitações metodológicas? Preencher `uncertainty_available`:
- `TRUE` se sim
- `FALSE` se não há documentação de incerteza

---

## 3. Se houver download

**Objetivo**: proteger a separação entre dado bruto local e metadado público.

### Passo 3.1 — Não baixar nesta etapa
A etapa v1hm não autoriza download de dados. Apenas documentação e planejamento.

### Passo 3.2 — Quando autorizado futuramente
Somente após aprovação explícita e verificação de licença:
- Salvar dado bruto exclusivamente em `local_only/ground_reference/raw/`
- Nunca salvar em pasta versionada
- Nunca nomear arquivo com path absoluto no manifest público

### Passo 3.3 — Criar manifest público de metadados
Após ter dado bruto local, criar manifest público com:
- `source_id`, `region`, `institution`, `acquisition_date`
- `license_status`, `redistribution_status`
- `protocol_c_gates_supported`
- `intake_decision`

Nunca incluir no manifest público:
- Path local do arquivo
- Conteúdo do dado bruto
- Geometria específica se for de uso restrito
- URLs de acesso não-público

### Passo 3.4 — Atualizar tracker
Mudar `acquisition_status` para `RECEIVED_RAW_LOCAL_ONLY` ou `RECEIVED_METADATA_ONLY` conforme o caso.

---

## 4. Ao preparar para revisão

**Objetivo**: encaminhar fonte elegível para revisão humana sem automatizar promoção.

### Passo 4.1 — Criar ou verificar vínculo source-event-patch
Abrir `datasets/patch_event_reference_link_registry.csv`. Verificar se já existe vínculo para esta fonte, evento e patches relevantes.

Condições para criar novo vínculo:
- Fonte tem `event_link_status=LINKED` ou `PARTIAL`
- Evento tem data conhecida
- Patch está na mesma região geográfica
- Alinhamento temporal é possível (cobertura de data pelo evento)
- `promotion_allowed=false` enquanto não houver revisão humana

### Passo 4.2 — Criar entrada em revisão humana
Criar placeholder em `datasets/human_reference_review_registry.csv` com:
- `review_status=PENDING`
- `reviewer=TBD`
- `materials_reviewed=PENDING`
- `allowed_claim` e `forbidden_claim` conservadores

Não assumir resultado da revisão antes dela acontecer.

### Passo 4.3 — Não promover automaticamente
Mesmo que a fonte pareça forte, não definir `promotion_allowed=true` sem revisão documentada. O Protocolo C proíbe promoção automática.

---

## 5. Ao encontrar bloqueio

**Objetivo**: registrar bloqueio de forma auditável e manter guardrails.

### Passo 5.1 — Identificar razão específica do bloqueio
Usar os critérios da seção 8 do pacote operacional. Razões de bloqueio válidas:
- `LICENSE_UNKNOWN`
- `LICENSE_RESTRICTED`
- `NO_EVENT_LINK`
- `TEMPORAL_MISMATCH`
- `NO_GEOMETRY`
- `SPATIAL_COVERAGE_INSUFFICIENT`
- `CONTEXTUAL_SOURCE_PROMOTED`
- `RAW_DATA_REDISTRIBUTION_FORBIDDEN`
- `SOURCE_CONFLICT_UNRESOLVED`
- `DINO_ONLY_INSUFFICIENT`
- `MISSING_HUMAN_REVIEW`

### Passo 5.2 — Registrar no intake registry
Preencher:
- `intake_decision=BLOCK_USE`
- `blocked_reason=[razão específica]`
- `forbidden_use=[lista de usos proibidos]`

### Passo 5.3 — Manter promotion_allowed=false
No vínculo correspondente em `patch_event_reference_link_registry.csv`, garantir que `promotion_allowed=false`.

### Passo 5.4 — Registrar próxima ação
Definir `next_action` no tracker: o que seria necessário para desbloquear? Reunião com órgão, esclarecimento de licença, busca de fonte alternativa?

---

## 6. Critérios para encaminhar ao próximo ciclo

Uma fonte pode ser encaminhada para o próximo ciclo de aquisição (e potencialmente para revisão humana) quando:

| Critério | Requerido |
|---|---|
| Fonte rastreável | Sim |
| Licença compreendida (não UNKNOWN) | Sim |
| Evento associado identificado | Sim para gates G1, G3, G4 |
| Data da fonte compatível com evento | Sim para gate G3 |
| Geometria disponível ou estimada | Sim para gate G4 |
| Vínculo possível com patch REV-P | Sim para gates G4–G8 |
| Revisão humana agendada ou pendente | Sim para gates G7, G9 |

Fontes que satisfazem todos os critérios relevantes devem ter:
- `acquisition_status=RECEIVED_METADATA_ONLY` ou `RECEIVED_RAW_LOCAL_ONLY`
- `intake_decision=ACCEPT_METADATA_ONLY` ou `ACCEPT_LOCAL_ONLY`
- Entrada em `human_reference_review_registry.csv` com `review_status=PENDING`

---

## Apêndice — Valores controlados de referência rápida

**acquisition_status**: NOT_STARTED, IDENTIFIED, REQUEST_REQUIRED, REQUESTED, RECEIVED_METADATA_ONLY, RECEIVED_RAW_LOCAL_ONLY, REJECTED, BLOCKED, METHOD_REFERENCE_ONLY

**intake_decision**: ACCEPT_METADATA_ONLY, ACCEPT_LOCAL_ONLY, REQUEST_MORE_INFORMATION, BLOCK_USE, METHOD_REFERENCE_ONLY

**license_status**: PUBLIC_REUSE_ALLOWED, PUBLIC_VIEW_ONLY, REQUEST_REQUIRED, RESTRICTED, UNKNOWN, METHOD_REFERENCE_ONLY

**redistribution_status**: PUBLIC_METADATA_ONLY, PUBLIC_REUSABLE, LOCAL_ONLY_LICENSED, REDISTRIBUTION_FORBIDDEN, UNKNOWN, METHOD_REFERENCE_ONLY

---

## Relação com a camada de busca externa (v1hp)

A camada de busca externa e solicitação regional (v1hp) complementa este runbook com estruturas prontas para ação: planos de busca por região (`regional_external_search_plan.csv`), pacotes de solicitação formal a instituições (`source_request_package_registry.csv`), perguntas de busca mapeadas a gates G1–G9 (`gate_search_question_registry.csv`) e matriz de prioridade regional (`regional_request_priority_matrix.csv`). Ao executar os passos deste runbook, use esses registros para identificar qual fonte consultar primeiro, qual formulário enviar e qual pergunta específica precisa ser respondida para fechar o gate alvo. Veja [`protocolo_c_busca_externa_solicitacao_regional.md`](protocolo_c_busca_externa_solicitacao_regional.md).

A etapa v1hq inicia a primeira camada documental de eventos observados candidatos: 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial. O inventário de dados externos que precisam ser trazidos manualmente está em `datasets/manual_external_evidence_needed_registry.csv`. Ground truth operacional não está estabelecido. Protocolo B permanece bloqueado. Multimodal permanece hold.
