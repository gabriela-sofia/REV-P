# Protocolo C — Dossiês de Evidência por Evento Candidato

## 1. Motivação

A etapa v1hn organizou eventos candidatos de inundação e alagamento por região do REV-P, registrou prioridades de busca e conectou cada evento aos gates do Protocolo C. Essa triagem responde à pergunta: *quais eventos precisam ser investigados primeiro?*

Esta etapa (v1ho) responde à pergunta seguinte: *exatamente o que precisa ser encontrado para que um evento candidato possa futuramente apoiar ground reference?*

Para cada evento candidato registrado em `event_candidate_screening_registry.csv`, um dossiê metodológico especifica:
- o pacote mínimo de evidências necessário;
- o estado atual de cada requisito;
- as lacunas que impedem avanço;
- os bloqueadores ativos;
- a decisão de continuidade.

O dossiê não confirma o evento. Não declara ground truth. Não baixa dados. É um registro metodológico que prepara a busca e a aquisição futuras de forma auditável.

---

## 2. O que é um dossiê de evento candidato

Um dossiê reúne, em formato auditável, os elementos necessários para avaliar se um evento candidato pode futuramente apoiar uma referência de ground reference no REV-P:

- **Evento candidato**: identificador, região, município, período e tipo de evento
- **Fontes-alvo**: quais instituições ou produtos precisariam ser consultados
- **Evidência temporal esperada**: data ou janela temporal do evento, confirmada por fonte rastreável
- **Evidência espacial esperada**: mapa ou geometria de ocorrência, com cobertura sobre os patches do corpus
- **Produto ou sensor esperado**: Sentinel-2, SAR, GFM, laudo oficial ou outro
- **Vínculo possível com patches**: quais patches do corpus DINO estão no perímetro de busca
- **Lacunas atuais**: quais requisitos mínimos ainda não foram satisfeitos
- **Bloqueadores ativos**: o que impede o avanço no estado atual
- **Decisão de continuidade**: continuar busca, solicitar formalmente, aguardar ou bloquear

O dossiê é um instrumento de planejamento metodológico — não um documento de validação.

---

## 3. Evidência mínima por evento

Um evento candidato só pode avançar na cadeia do Protocolo C se reunir, no mínimo, os seguintes elementos:

1. **Fonte rastreável**: pelo menos uma instituição ou produto identificável que documente o evento
2. **Data ou janela temporal**: período do evento confirmado por fonte — não apenas inferido de contexto regional
3. **Documentação do evento**: relatório, laudo, comunicado ou produto que registre a ocorrência
4. **Fonte espacial ou produto interpretável**: mapa de área afetada, produto de detecção ou imagem com cobertura temporal próxima
5. **Possibilidade de conexão com patch**: ao menos um patch do corpus dentro do perímetro plausível do evento
6. **Licença e proveniência compreendidas**: o uso da fonte para fins acadêmicos precisa ser explicitamente permitido ou clarificado
7. **Revisão humana futura**: nenhuma promoção pode ocorrer sem revisão por pesquisador ou especialista

A ausência de qualquer um desses elementos bloqueia a promoção do evento a candidato de referência.

---

## 4. Pacote ideal de evidência

O pacote forte de evidência para um evento candidato inclui:

- **Fonte oficial ou acadêmica** com documentação do evento (Defesa Civil, CPRM, publicação revisada)
- **Mapa observado ou produto operacional com incerteza documentada** (laudo geológico, GFM com metadados de confiança)
- **Imagem Sentinel-2 ou SAR próxima ao evento** com baixa cobertura de nuvem e alinhamento temporal confirmado
- **Geometria ou polígono de ocorrência** com referência de CRS e cobertura sobre os patches do corpus
- **Anotação humana ou de especialista** (se disponível), documentada no registro de revisão humana
- **Fonte independente de corroboração** — segunda fonte que confirme a ocorrência do evento de forma independente

Nenhum desses elementos é suficiente isoladamente para declarar ground truth operacional. A promoção exige satisfação de todos os gates G1–G9 e revisão formal (Protocolo C — etapa de fechamento).

---

## 5. Estados do dossiê

O estado do dossiê reflete o grau de completude das evidências reunidas para um evento candidato:

**DOSSIER_NOT_STARTED**
O dossiê foi criado mas nenhum elemento de evidência foi identificado ou buscado. Todos os requisitos estão em NOT_ASSESSED.

**DOSSIER_OPEN**
O dossiê está ativo: pelo menos uma fonte-alvo identificada no backlog. Busca ainda não iniciada. A maioria dos requisitos está MISSING ou NOT_ASSESSED.

**DOSSIER_PARTIAL**
Parte dos requisitos foi avaliada. Pode haver evidência contextual (PENDING_SOURCE_REVIEW) ou fontes identificadas mas ainda não adquiridas. Requisitos críticos permanecem MISSING.

**DOSSIER_READY_FOR_SOURCE_REVIEW**
O dossiê tem fonte-alvo clara e caminho de acesso definido. A próxima etapa é a revisão real da fonte (consulta ao portal, solicitação formal). Nenhum dado foi baixado ainda.

**DOSSIER_READY_FOR_HUMAN_REVIEW**
Evidência suficiente foi adquirida e documentada para justificar revisão por pesquisador ou especialista. Este estado requer que fontes concretas tenham sido recebidas e registradas — não existe nesta etapa.

**DOSSIER_BLOCKED**
Um bloqueador crítico impede qualquer avanço: ausência de fonte, conflito de licença, sensibilidade, ou outros bloqueadores identificados.

**METHOD_REFERENCE_ONLY**
O dossiê é apenas referência metodológica — o evento não é alvo de aquisição direta para o REV-P.

---

## 6. Critérios de bloqueio

Um dossiê é bloqueado quando ocorre pelo menos um dos seguintes:

- Ausência de fonte rastreável para o evento
- Ausência de data ou janela temporal confirmada
- Ausência de evidência espacial que permita conexão com patches do corpus
- Ausência de cobertura possível de algum patch do corpus pelo evento
- Licença desconhecida ou incompatível com uso acadêmico
- Fonte apenas contextual tentando fechar gates de observação direta
- Dependência exclusiva de embedding DINO, índice GIS ou clustering como confirmação de evento
- Conflito entre fontes sobre ocorrência ou extensão do evento
- Ausência de revisão humana quando exigida pelo gate

---

## 7. Relação com gates G1–G9

O dossiê organiza os requisitos de evidência de acordo com os gates do Protocolo C:

| Gate | Nome | O que o dossiê registra |
|------|------|------------------------|
| G1 | EVENT_CONFIRMATION | Fonte que confirma a ocorrência do evento |
| G2 | SOURCE_AVAILABILITY | Disponibilidade de fonte com cobertura do evento |
| G3 | TEMPORAL_ALIGNMENT | Alinhamento entre evento e asset Sentinel disponível |
| G4 | SPATIAL_ALIGNMENT | Sobreposição entre evento e patches do corpus |
| G5 | SOURCE_STRENGTH | Força metodológica da fonte (observação direta vs. produto algorítmico) |
| G6 | UNCERTAINTY_AND_LIMITATIONS | Documentação de incerteza da fonte |
| G7 | HUMAN_REVIEW | Revisão humana ou especialista documentada |
| G8 | INDEPENDENT_CORROBORATION | Segunda fonte independente |
| G9 | PROMOTION_DECISION | Decisão formal de promoção a referência operacional |

**O DINOv2 não satisfaz nenhum desses gates.** Os embeddings DINO são review-only estrutural e não constituem evidência observacional de evento, temporalidade, espacialidade ou ground truth.

Nenhum evento desta etapa satisfaz qualquer gate. O dossiê registra quais gates são relevantes e quais requisitos precisariam ser satisfeitos para avançar.

---

## 8. O que esta etapa não faz

Esta etapa (v1ho) explicitamente não faz:

- **Não confirma ground truth operacional**: nenhum evento tem ground truth estabelecido
- **Não cria flood label**: nenhum pixel é rotulado
- **Não cria training label**: nenhum alvo supervisionado é gerado
- **Não treina modelo**: nenhum pipeline de aprendizado é executado
- **Não executa Protocolo B**: multimodal, treinamento e predição permanecem bloqueados
- **Não avança multimodal**: multimodal permanece em hold aguardando recuperação de stack, balanceamento regional e aprovação de revisor
- **Não baixa dados**: nenhum raster, shapefile, GeoJSON ou dado pesado é adquirido
- **Não usa DINO como confirmação de evento**: os embeddings DINO são review-only e não fecham gates G1–G9

---

## Registros desta etapa

- `datasets/event_evidence_dossier_registry.csv` — dossiês por evento candidato com status, lacunas e decisão de continuidade
- `datasets/event_evidence_requirements_registry.csv` — requisitos mínimos de evidência por evento candidato, conectados aos gates G1–G9
- `datasets/event_dossier_decision_registry.csv` — decisões de continuidade por dossiê: próximos passos permitidos e proibidos

Esses três registros são metadata-only. Nenhum dado bruto é referenciado. Nenhuma aquisição é executada.

Os templates em `docs/templates/` documentam como preencher um dossiê e registrar uma busca manual futura.
