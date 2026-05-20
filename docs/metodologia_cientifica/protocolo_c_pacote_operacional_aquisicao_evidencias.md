# Protocolo C — pacote operacional de aquisição de evidências

## 1. Motivação

A etapa v1hl definiu o que buscar: fontes-alvo por região, prioridades de aquisição, força metodológica de cada tipo de fonte e a prontidão regional para os gates do Protocolo C.

A etapa v1hm define **como** operar essa aquisição: como registrar, verificar acesso, documentar licença, triar entrada, organizar staging local e encaminhar para revisão humana — sem promover claims indevidos, sem baixar dados pesados e sem declarar ground truth onde ele não existe.

O REV-P ainda não possui ground truth operacional observado. O Protocolo C organiza a construção graduada de referência: contexto → proxy → candidato de referência → validação operacional. Nenhuma etapa dessa progressão é automática. Esta etapa coloca em prática os mecanismos operacionais para que o plano de aquisição da v1hl possa ser executado com rastreabilidade, documentação e guardrails auditáveis.

---

## 2. Princípios operacionais

Toda ação nesta etapa e nas subsequentes segue estes princípios invioláveis:

### metadata-first
O repositório público contém apenas metadados seguros: IDs, nomes de fonte, datas, atributos de licença, status de aquisição, campos de controle metodológico. Dados brutos, rasters, shapefiles e qualquer arquivo pesado ficam exclusivamente em armazenamento local.

### public registry, raw data local-only
O que entra no GitHub é rastreabilidade — não dados. O que entra no workspace local é o dado. Essa separação é inviolável durante toda a vida do projeto.

### rastreabilidade de fonte
Toda fonte usada deve ter instituição identificável, referência citável, data conhecida e método documentado. Fonte sem rastreabilidade não pode fechar gate.

### licença/restrição documentada
Nenhuma fonte pode ser usada sem que seu status de licença e redistribuição seja registrado. Licença desconhecida bloqueia promoção.

### nenhuma promoção automática
Nenhum dado, produto ou resultado é automaticamente promovido a ground truth ou referência operacional. Toda promoção requer revisão humana documentada e satisfação de gates.

### nenhum label supervisionado
Esta etapa não cria, não importa e não usa labels de treinamento. Não há e não haverá rótulo de inundação, não-inundação ou qualquer classe supervisionada.

### nenhum Protocolo B
Detecção, segmentação e predição de inundações permanecem bloqueadas. O Protocolo B não pode ser executado nesta etapa.

### multimodal em hold
A integração multimodal permanece em espera até que a camada de ground reference esteja metodologicamente fechada e revisada.

---

## 3. Fluxo de aquisição

O fluxo a seguir governa como uma fonte-alvo passa de identificada a elegível para revisão.

### Passo 1 — Identificar fonte-alvo
Consultar `datasets/observational_evidence_acquisition_plan.csv`. Verificar qual gate a fonte-alvo pode fechar. Confirmar prioridade de aquisição (HIGH/MEDIUM/LOW/METHOD_REFERENCE_ONLY).

### Passo 2 — Registrar fonte no tracker
Criar ou atualizar linha em `datasets/evidence_acquisition_tracker.csv`. Preencher todos os campos obrigatórios. Definir `acquisition_status=IDENTIFIED` ou mais avançado, conforme real.

### Passo 3 — Verificar acesso e licença
Consultar portal ou documentação da fonte. Registrar `license_status` e `redistribution_status`. Se licença for `UNKNOWN`, registrar `current_blocker=LICENSE_UNKNOWN`. Não usar fonte com licença desconhecida.

### Passo 4 — Registrar modo de aquisição
Definir se é `PUBLIC_DOWNLOAD`, `PUBLIC_PORTAL_REVIEW`, `FORMAL_REQUEST`, `MANUAL_REVIEW`, `FUTURE_ACQUISITION` ou `METHOD_REFERENCE_ONLY`.

### Passo 5 — Se houver dado bruto, manter local-only
Dado bruto vai para `local_only/ground_reference/raw/`. Nunca versionar. Nunca copiar para o repositório público.

### Passo 6 — Criar manifest público apenas com metadados seguros
O manifest público contém: `source_id`, `region`, `institution`, `acquisition_date`, `license_status`, `protocol_c_gates_supported`, `intake_decision`. Nunca incluir path local, dado bruto ou conteúdo restrito.

### Passo 7 — Avaliar gates do Protocolo C
Verificar quais gates a fonte pode satisfazer. Registrar em `protocol_c_gates_supported` do intake registry. Avaliar se a fonte está suficientemente forte para cada gate específico.

### Passo 8 — Encaminhar para revisão humana se houver elegibilidade
Se a fonte satisfizer as pré-condições de evento, temporalidade, espacialidade e força metodológica, criar entrada em `human_reference_review_registry.csv` (etapa anterior). Revisor humano decide sobre promoção.

### Passo 9 — Registrar bloqueio se faltar condição
Se faltar evento, temporalidade, espacialidade, licença ou revisão, registrar `intake_decision=BLOCK_USE` e `blocked_reason` específico. Manter `promotion_allowed=false`.

---

## 4. Tipos de entrada aceitos

### Portal público
Dados publicados em portais abertos (CPRM/RIGeo, GeoCuritiba, PE3D, CEMS). Aceito como metadado para staging. Dado bruto requer avaliação de licença antes de download.

### Solicitação formal
Dados acessíveis mediante pedido formal a órgão público (Defesa Civil, prefeitura, universidade). Requer `request_required=true`, registro de status e resposta.

### Relatório técnico
Relatórios de evento, análises pós-desastre, publicações técnicas. Aceito como referência documental. Não pode sozinho fechar gate de validação espacial.

### Produto operacional
GFM/CEMS, produtos SAR de detecção de mudança. Aceito com `cannot_support_operational_ground_truth_alone=true`. Incerteza deve ser documentada.

### Dataset acadêmico
Sen1Floods11, Kuro Siwo, UFO. Aceito como `METHOD_REFERENCE_ONLY` enquanto não houver aplicação direta validada aos patches REV-P.

### Mapa oficial observado
Produto de mapeamento pós-evento com metodologia publicada, cobertura documentada e acurácia estimada. Pode fechar múltiplos gates, dependendo de qualidade e independência.

### Camada modelada/suscetibilidade
Dados de suscetibilidade geomorfológica ou hidrológica. Aceito como contexto. Não pode fechar gate de confirmação de evento específico.

### Imagem Sentinel pós-evento
Sentinel-1 SAR ou Sentinel-2 optical. Requer data precisa, cobertura aceitável e anotação humana para close de gate de validação espacial.

### Anotação humana futura
Anotação especializada sobre imagens pós-evento. É a fonte de maior força metodológica potencial, mas ainda não foi executada no estado atual.

---

## 5. Propriedade, licença e redistribuição

A licença de uma fonte determina o que pode ser feito com ela no contexto do REV-P.

### O que pode ir ao GitHub
Metadados seguros: IDs, nomes, datas, atributos de licença, status de aquisição, fields de controle. Nunca: dado bruto, path local, conteúdo restrito, geometria específica de fonte privada.

### O que fica local-only
Todo dado bruto (raster, shapefile, GeoJSON, ZIP, CSV pesado com coordenadas específicas) que tenha restrição de redistribuição ou que simplesmente seja pesado demais. A regra padrão é: quando em dúvida, local-only.

### Tipos de licença e consequências

| license_status | O que fazer |
|---|---|
| `PUBLIC_REUSE_ALLOWED` | Pode usar e referenciar metadados publicamente |
| `PUBLIC_VIEW_ONLY` | Pode visualizar e referenciar, mas não redistribuir dado bruto |
| `REQUEST_REQUIRED` | Deve solicitar formalmente antes de usar |
| `RESTRICTED` | Não pode usar sem autorização explícita; bloqueia promoção |
| `UNKNOWN` | Bloqueia promoção até esclarecimento |

### Redistribuição
Redistribuição de dado bruto só é permitida quando `redistribution_status=PUBLIC_REUSABLE`. Qualquer outro status (`LOCAL_ONLY_LICENSED`, `REDISTRIBUTION_FORBIDDEN`, `UNKNOWN`) impede publicação do dado no GitHub.

---

## 6. Staging local seguro

As pastas a seguir são definidas por caminho relativo (sem path privado). Não devem ser versionadas.

```
local_only/
└── ground_reference/
    ├── raw/          ← dados brutos locais (nunca versionar)
    ├── derived/      ← dados derivados locais (nunca versionar)
    └── manifests_private/  ← manifests com paths/dados sensíveis (nunca versionar)

local_runs/
└── ground_reference_audit/  ← outputs de auditorias locais (nunca versionar)
```

Essas pastas devem estar no `.gitignore`. Qualquer arquivo dentro delas que seja commitado acidentalmente deve ser removido com `git rm --cached` imediatamente.

---

## 7. Intake de fonte

Toda fonte recebida ou acessada precisa ter os seguintes campos registrados no `evidence_source_intake_registry.csv`:

| Campo | Obrigatório | Descrição |
|---|---|---|
| `intake_id` | Sim | ID único do registro de intake |
| `acquisition_id` | Sim | Referência ao tracker de aquisição |
| `source_id` | Sim | ID único da fonte |
| `region` | Sim | Região REV-P (RECIFE, PETROPOLIS, CURITIBA) |
| `source_name` | Sim | Nome descritivo da fonte |
| `source_family` | Sim | Família tipológica (valores controlados) |
| `source_type` | Sim | Tipo específico dentro da família |
| `provider` | Sim | Instituição ou provedor |
| `event_id` | Sim | ID do evento associado (ou NOT_LINKED) |
| `event_link_status` | Sim | Status de vínculo com evento |
| `acquisition_date` | Sim | Data de acesso/aquisição ou PENDING |
| `source_date_or_period` | Sim | Data/período da fonte original |
| `geometry_available` | Sim | Geometria disponível? |
| `crs_available` | Sim | CRS documentado? |
| `temporal_information_available` | Sim | Informação temporal disponível? |
| `spatial_coverage_status` | Sim | Cobertura espacial dos patches |
| `uncertainty_available` | Sim | Incerteza/acurácia documentada? |
| `license_status` | Sim | Status de licença |
| `redistribution_status` | Sim | Status de redistribuição |
| `local_asset_status` | Sim | Status do ativo local |
| `public_registry_safe` | Sim | Seguro para publicação em repositório público? |
| `protocol_c_gates_supported` | Sim | Gates que esta fonte pode suportar |
| `intake_decision` | Sim | Decisão de intake |
| `blocked_reason` | Se bloqueado | Razão do bloqueio |
| `allowed_use` | Sim | Usos permitidos |
| `forbidden_use` | Sim | Usos proibidos |
| `notes` | Não | Observações adicionais |

---

## 8. Critérios de rejeição ou bloqueio

Uma fonte deve ser bloqueada (`intake_decision=BLOCK_USE`) quando:

- **Sem licença**: `license_status=UNKNOWN` ou `RESTRICTED` sem autorização documentada
- **Sem fonte rastreável**: provedor não identificável, referência não citável
- **Sem evento**: fonte contextual que não pode ser vinculada a evento específico quando gate exige confirmação de evento
- **Sem temporalidade**: data da fonte incompatível com data do evento
- **Sem geometria**: fonte sem geometria quando gate exige alinhamento espacial
- **Sem cobertura do patch**: fonte não cobre a área geográfica dos patches da região
- **Uso indevido de fonte contextual**: fonte contextual (suscetibilidade, geomorfologia, NDWI) sendo promovida como ground truth
- **Dado bruto com redistribuição proibida**: fonte local-only sendo referenciada como pública
- **Conflito entre fontes**: fontes divergentes sem critério de resolução documentado
- **Dependência exclusiva de DINO, cluster, NDWI/NDBI ou GIS modelado**: essas fontes não podem ser usadas como única evidência para qualquer gate de validação observacional

---

## 9. Saída da etapa

A etapa v1hm produz os seguintes artefatos públicos (metadados apenas):

- **Tracker de aquisição** (`evidence_acquisition_tracker.csv`): estado atual de cada fonte-alvo
- **Intake registry** (`evidence_source_intake_registry.csv`): registro de fontes acessadas/recebidas com decisão de intake
- **Registry de proveniência/licenciamento** (`evidence_license_provenance_registry.csv`): licença, redistribuição e proveniência por fonte
- **Checklist de triagem de fonte** (template): avaliação pré-uso de qualquer fonte
- **Template de solicitação** (template): modelo para pedidos formais a órgãos
- **Runbook de aquisição** (documento): guia passo a passo para execução futura
- **Regras de staging local** (neste documento, seção 6): estrutura local-only sem path privado
- **Bloqueios explícitos** (em cada registry): `forbidden_use`, `blocked_reason`, `intake_decision=BLOCK_USE`

## Etapas subsequentes

A triagem de eventos candidatos (v1hn) e os dossiês de evidência (v1ho) complementam este pacote com camadas mais específicas sobre o que buscar e o que é necessário por evento. Os dossiês especificam o pacote mínimo de evidências, requisitos críticos e decisões de continuidade com `can_reassess_protocol_b=false` e `can_start_multimodal=false`. Veja [`protocolo_c_triagem_eventos_candidatos.md`](protocolo_c_triagem_eventos_candidatos.md) e [`protocolo_c_dossies_eventos_candidatos.md`](protocolo_c_dossies_eventos_candidatos.md).

A camada de busca externa e solicitação regional (v1hp) transforma os dossiês em ação concreta: planos de busca por região com fonte-alvo, gate e modo de acesso; pacotes de solicitação formal a instituições; perguntas de busca mapeadas a gates G1–G9; e matriz de prioridade regional consolidando a ordem de execução. Veja [`protocolo_c_busca_externa_solicitacao_regional.md`](protocolo_c_busca_externa_solicitacao_regional.md).

A etapa v1hq inicia a primeira camada documental de eventos observados candidatos: 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial. Ground truth operacional não está estabelecido. Protocolo B permanece bloqueado. Multimodal permanece hold. Dados externos brutos que precisam ser trazidos manualmente estão catalogados em `datasets/manual_external_evidence_needed_registry.csv`.
