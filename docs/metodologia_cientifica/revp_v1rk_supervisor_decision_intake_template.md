# v1rk — Supervisor Decision Intake Template

## Objetivo

Gerar template preenchivel para a decisao final do supervisor a partir dos pacotes v1rj. Nenhuma decisao e preenchida; nenhuma evidencia e criada.

## Preenchimento seguro

Preencher supervisor_decision com uma acao permitida, decision_confidence_0_4 (0-4), e notas. Apontar REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH para o CSV preenchido e rodar v1rl.

## Resultado

Pacotes para decisao: 0.

## Guardrails

Aprovar C3 candidate permanece review-only: can_create_operational_label=false, ground_truth_operational=false. C4 nunca aberto sem fonte formal negativa.
