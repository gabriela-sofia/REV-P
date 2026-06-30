# REV-P v2da

Prepara plano de download seguro; download real exige flag e licenca explicita.

## Guardrails

- Offline-first.
- Network only when an explicit flag is passed and only for registered URLs.
- Downloads are not executed by default.
- Raw external files are never written to `outputs_public`.
- Outputs are review-only and cannot be interpreted as operational ground truth.
- Missing evidence remains blocked instead of inferred.

## Allowed claim

Uso permitido apenas como prontidao cientifica review-only; nao fecha TP2, TP3, treino ou validacao operacional.

## Forbidden claim

ground_truth_operacional|label_binario|negativo_formal|dataset_treino|claim_deteccao|claim_predicao|intersecao_observada_automatica
