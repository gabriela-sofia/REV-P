# Relatorio v1jl - Evidencia para negativos e controles

## Resultado

A v1jl auditou controles candidatos, camadas externas e taxonomia de corpus. Nenhum item foi promovido a negativo formal.

## Negativos formais

Negativos formais encontrados: 0.

O motivo e direto: nao ha registro oficial explicito dizendo que uma area, em uma janela temporal compativel, estava estavel ou sem o fenomeno analisado. Sem essa evidencia, controle candidato nao vira negativo.

## Controles fortes

Os controles temporais do mesmo anchor sao fortes para revisao, porque permitem comparar o proprio local antes e depois. Mas eles carregam risco de leakage e nao sao amostras independentes. Por isso permanecem review-only.

## Controles fracos e hipoteses invalidas

Patches de fundo, camadas externas e material cross-region sao contexto. Eles ajudam a interpretar o lote, mas nao documentam ausencia.

A hipotese "sem registro = negativo" fica explicitamente bloqueada.

## Status do treino

O gate final e:

- `SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES`;
- `can_create_training_label=false`;
- `can_train_model=false`;
- `can_unfreeze_dino_for_scientific_claim=false`.

O projeto continua review-only. Para sair desse estado, precisa de protocolo formal de ausencia/estabilidade, negativos auditaveis, labels positivos formalizados, split fechado e validacao independente.
