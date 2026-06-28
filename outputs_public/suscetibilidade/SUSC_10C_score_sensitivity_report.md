# SUSC-10C — Sensibilidade e Ablação do Score v6 (review-only)

> O SUSC-10C avalia a estabilidade de um score determinístico candidato. A análise de sensibilidade não constitui treinamento, validação supervisionada ou confirmação de evento real.

## 1. O que foi testado
11 cenários: remoção de grupos (chuva/urbano-espectral/topo-hidro/evidência), pesos iguais, hidrologia/urbano/chuva pesados, e thresholds estrito/frouxo de classe.

## 2. Estabilidade global
- Estabilidade de classe média (fração de cenários que mantêm a classe do baseline): 0.7048.
- Menor Spearman vs baseline entre cenários: 0.4151.
- Patches robustos: 114; instáveis: 38.
- (O range bruto do score por patch é inflado pelo reescalonamento por cenário; a métrica honesta é a estabilidade de classe/rank.)

## 3. Estabilidade por região
Ver `SUSC_10C_score_sensitivity_by_region.csv`.

## 4. Patches robustamente altos
48 patches `high` no baseline e `robust` sob cenários.

## 5. Patches instáveis
Top 25 em `SUSC_10C_score_sensitivity_top_unstable_patches.csv`.

## 6. Qual grupo mais altera o ranking
Ablação mais impactante: **no_topography_hydrology** (menor Spearman vs baseline).

## 7. Implicação metodológica
O score é uma composição determinística; remover topo/hidrologia tende a alterar mais o ranking, coerente com o peso alto desse grupo (escolha documentada, não aprendida).

## 8. Limitações
Pesos e thresholds são escolhas metodológicas. Sensibilidade não é treino nem validação; evidência de suporte é review-only.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
