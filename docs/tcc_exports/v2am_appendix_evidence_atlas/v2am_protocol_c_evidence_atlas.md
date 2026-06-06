# Protocolo C v2am - Evidence Atlas

Atlas tecnico de evidencia por eixo cientifico. Documento de apoio e auditoria,
nao e o capitulo final do TCC. Corpus review-only, sem ground truth operacional.

## candidatos review-only
172 pacotes preservados como candidatos revisaveis, sem promocao operacional.

- Fonte: datasets/protocolo_c/v2ah_candidate_reference_review_queue.csv
- Status: candidates_only_pending_review
- Interpretacao permitida: Descrever corpus de candidatos para revisao futura.
- Interpretacao proibida: Nao ler como deteccao, classe, label ou referencia validada.
- Local sugerido: apendice: fila de revisao
- Pergunta de banca: O que significam os 172 candidatos?

## blockers de promocao
172 pacotes com promotion_allowed=false e blockers explicitos.

- Fonte: datasets/protocolo_c/v2ai_safe_promotion_blockers.csv
- Status: promotion_blocked
- Interpretacao permitida: Explicar por que nenhum candidato e promovido.
- Interpretacao proibida: Nao ler como problema resolvido nem validacao pendente trivial.
- Local sugerido: apendice: claims e guardrails
- Pergunta de banca: Entao como validam o projeto?

## revisao humana pendente
344 slots de revisao humana preparados, ainda pendentes.

- Fonte: datasets/protocolo_c/v2ai_review_assignment_registry.csv
- Status: human_review_pending
- Interpretacao permitida: Mostrar estrutura de revisao futura.
- Interpretacao proibida: Nao ler como revisao concluida nem identidade real de revisor.
- Local sugerido: apendice: fila de revisao
- Pergunta de banca: A revisao humana foi feita?

## adjudicacao pendente
172 itens aguardando adjudicacao apos revisao futura.

- Fonte: datasets/protocolo_c/v2ai_adjudication_queue.csv
- Status: adjudication_pending
- Interpretacao permitida: Mostrar plano de adjudicacao.
- Interpretacao proibida: Nao ler como consenso atingido.
- Local sugerido: apendice: fila de revisao
- Pergunta de banca: Houve adjudicacao?

## claims permitidos/proibidos
Matriz com 10 claims separando linguagem segura e proibida.

- Fonte: datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv
- Status: claims_separated
- Interpretacao permitida: Usar como guardrail de escrita.
- Interpretacao proibida: Nao reintroduzir claim proibido como afirmacao.
- Local sugerido: apendice: claims e guardrails
- Pergunta de banca: Que afirmacoes voces podem fazer?

## limitacoes metodologicas
9 limitacoes documentadas como controle metodologico.

- Fonte: datasets/protocolo_c/v2aj_methodological_limitations_export.csv
- Status: limitations_documented
- Interpretacao permitida: Apresentar limitacoes como delimitacao controlada.
- Interpretacao proibida: Nao ler como falha descontrolada do projeto.
- Local sugerido: apendice: limitacoes
- Pergunta de banca: A ausencia de ground truth invalida o trabalho?

## integracao segura no manuscrito
Bundles Markdown/LaTeX e matriz de insercao preparados para revisao manual.

- Fonte: datasets/protocolo_c/v2al_section_insertion_matrix.csv
- Status: integration_prepared_manual_review
- Interpretacao permitida: Inserir secoes apos revisao humana.
- Interpretacao proibida: Nao inserir automaticamente nem promover candidato.
- Local sugerido: apendice: indice e catalogo
- Pergunta de banca: Como o Protocolo C entra no texto?

## captions e tabelas
Legendas seguras de governanca/revisao para tabelas do TCC.

- Fonte: datasets/protocolo_c/v2al_table_caption_export.csv
- Status: captions_safe
- Interpretacao permitida: Usar legendas de governanca e revisao.
- Interpretacao proibida: Nao usar caption de acuracia ou validacao operacional.
- Local sugerido: apendice: catalogo de tabelas e figuras
- Pergunta de banca: As tabelas mostram desempenho?

## guardrails
Regressoes de guardrail fail-closed mantidas em todas as etapas.

- Fonte: datasets/protocolo_c/v2ak_safe_language_glossary.csv
- Status: guardrails_active
- Interpretacao permitida: Mostrar que linguagem e governanca foram auditadas.
- Interpretacao proibida: Nao ler como validacao de desempenho.
- Local sugerido: apendice: claims e guardrails
- Pergunta de banca: Como garantem que nao houve overclaim?

