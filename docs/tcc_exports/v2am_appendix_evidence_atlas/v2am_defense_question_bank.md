# Protocolo C v2am - banco de perguntas de banca

Perguntas e respostas seguras para a defesa. As respostas negam explicitamente
qualquer afirmacao operacional; termos proibidos aparecem apenas como termo a evitar.

## Voces tem ground truth?
- Resposta curta: Nao temos ground truth operacional patch-level.
- Resposta tecnica: Nao ha referencia operacional patch-level; a busca foi parada em GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE e os pacotes permanecem candidatos revisaveis.
- Evidencia a mostrar: datasets/protocolo_c/v2ah_ground_truth_search_stop_gate.csv
- Termos seguros: ausencia de ground truth operacional; candidato revisavel
- Termo a evitar (unsafe): ground truth validado

## Entao como validam o projeto?
- Resposta curta: Nao ha validacao operacional; o trabalho e review-only e auditavel.
- Resposta tecnica: A contribuicao e metodologica: organizacao de candidatos, blockers, claims e guardrails; a validacao de desempenho nao e reivindicada.
- Evidencia a mostrar: datasets/protocolo_c/v2ai_safe_promotion_blockers.csv
- Termos seguros: review-only; governanca metodologica
- Termo a evitar (unsafe): validacao operacional

## O que significam os 172 candidatos?
- Resposta curta: Sao 172 pacotes candidatos revisaveis, sem promocao.
- Resposta tecnica: Cada pacote e um candidato de evidencia contextual que permanece bloqueado para promocao ate revisao humana e nova fonte qualificada.
- Evidencia a mostrar: datasets/protocolo_c/v2ah_candidate_reference_review_queue.csv
- Termos seguros: candidatos revisaveis; review-only
- Termo a evitar (unsafe): classe positiva

## DINOv2 detecta enchente?
- Resposta curta: Nao. DINOv2 nao realiza deteccao de enchente; e suporte estrutural de triagem.
- Resposta tecnica: DINOv2 fornece embeddings estruturais para triagem; nao produz deteccao nem classe de inundacao observada.
- Evidencia a mostrar: datasets/protocolo_c/v2ak_safe_language_glossary.csv
- Termos seguros: suporte estrutural; review-only
- Termo a evitar (unsafe): deteccao de enchente

## GIS virou label?
- Resposta curta: Nao. GIS e contexto territorial, nao cria label nem classe.
- Resposta tecnica: O GIS fornece contexto territorial externo; nao foi convertido em label, classe ou target.
- Evidencia a mostrar: datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv
- Termos seguros: contexto territorial; sem label
- Termo a evitar (unsafe): label operacional

## Por que nao treinaram o modelo?
- Resposta curta: Porque nao ha referencia qualificada; treino esta bloqueado.
- Resposta tecnica: Sem ground truth operacional e com blockers ativos, treino supervisionado nao e justificavel; o escopo permanece estrutural e review-only.
- Evidencia a mostrar: datasets/protocolo_c/v2ai_safe_promotion_blockers.csv
- Termos seguros: treino bloqueado; review-only
- Termo a evitar (unsafe): treinamento supervisionado pronto

## O que falta para o Protocolo B?
- Resposta curta: Falta nova fonte qualificada e revisao humana concluida.
- Resposta tecnica: O Protocolo B permanece fechado ate evidencia observacional qualificada, revisao humana e adjudicacao reais; nada disso foi simulado.
- Evidencia a mostrar: datasets/protocolo_c/v2ah_ground_truth_search_stop_gate.csv
- Termos seguros: Protocolo B bloqueado; pendente
- Termo a evitar (unsafe): ground truth validado

## A ausencia de ground truth invalida o trabalho?
- Resposta curta: Nao. A ausencia e uma limitacao controlada e documentada.
- Resposta tecnica: A contribuicao metodologica (governanca, blockers, claims, rastreabilidade) e valida independentemente; a limitacao delimita o que pode ser afirmado.
- Evidencia a mostrar: datasets/protocolo_c/v2aj_methodological_limitations_export.csv
- Termos seguros: limitacao controlada; governanca
- Termo a evitar (unsafe): validacao operacional

## O que exatamente v2ah/v2ai/v2aj/v2ak/v2al provaram?
- Resposta curta: Provaram organizacao review-only auditavel, nao desempenho operacional.
- Resposta tecnica: v2ah parou a busca e consolidou candidatos; v2ai preparou revisao/adjudicacao com blockers; v2aj separou claims e limitacoes; v2ak gerou drafts seguros; v2al preparou integracao manual. Nenhuma etapa criou ground truth, label ou predicao.
- Evidencia a mostrar: datasets/protocolo_c/v2am_traceability_dag_edges.csv
- Termos seguros: rastreabilidade; review-only
- Termo a evitar (unsafe): modelo preditivo

