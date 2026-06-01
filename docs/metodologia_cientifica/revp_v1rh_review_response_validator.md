# v1rh — Review Response Validator

## Objetivo

Validar respostas A/B preenchidas manualmente (REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH). Sem o arquivo, REVIEW_RESPONSES_WAITING_MANUAL_INPUT. Uma linha por checagem por (sample, slot).

## Checagens

packet_id_exists, reviewer_slot_valid, all_required_questions_answered, confidence_in_range_0_4, recommended_decision_allowed, source_reference_when_event_supported, no_absolute_path, no_synthetic_unless_sandbox, no_label_target_ground_truth_true.

## Resultado

Status: REVIEW_RESPONSES_WAITING_MANUAL_INPUT. Grupos: 0. Checagens: 0 (passou 0, falhou 0).

## Guardrails

review_only=true. Fixture/synthetic so com sandbox explicito. Nenhum label/target/ground truth/negativo formal.
