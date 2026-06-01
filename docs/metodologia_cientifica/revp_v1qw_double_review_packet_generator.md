# v1qw — Double-Review Packet Generator

## Objetivo

Gerar pacotes de revisao dupla A/B. Cada review_sample_id gera dois slots (REVIEWER_A, REVIEWER_B) e formularios em branco com perguntas obrigatorias.

## Perguntas obrigatorias

evidence_visible, event_supported, location_supported, timing_supported, source_quality, independent_source_present, uncertainty_level, recommended_decision, uncertainty_notes.

## Resultado

Amostras: 8. Pacotes A/B: 16. Formularios: 144.

## Guardrails

Nenhuma resposta final e preenchida; apenas placeholders <TO_BE_FILLED_BY_HUMAN_REVIEWER>. dino_validates_event=false. Nenhum label.
