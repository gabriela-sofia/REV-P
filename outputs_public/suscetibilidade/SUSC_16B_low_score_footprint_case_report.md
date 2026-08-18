# SUSC-16B - casos footprint com score baixo/medio

Status: review-only. Estes casos nao sao ground truth e nao liberam treino.

Casos auditados: 57.

## Possiveis razoes
- score_weight_underestimates_local_factor: 10
- spectral_feature_temporal_mismatch: 47

## Leitura
Quando o score v6 baixo/medio diverge de um footprint elegivel, a divergencia
fica registrada para revisao. Nenhum peso foi alterado e nenhum score v7 foi
criado neste marco.
