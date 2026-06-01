# v1rg — Review Response Intake Template

## Objetivo

Gerar template preenchivel para respostas de Review A/B a partir dos pacotes v1qw. Nenhuma resposta e preenchida; nenhuma evidencia e criada.

## Preenchimento seguro

1) Cada revisor preenche apenas answer_value/confidence/notes do seu slot. 2) Usar pseudonimo, nunca PII. 3) Nao colar paths absolutos nem referencias a diretorios locais. 4) source_reference obrigatoria quando event_supported=sim. 5) Apontar REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH para o CSV preenchido e rodar v1rh.

## Resultado

Pacotes A/B: 16. Linhas de template (pacote x pergunta): 144.

## Guardrails

review_only=true. Nenhuma resposta preenchida automaticamente. Nenhum label/target/ground truth operacional.
