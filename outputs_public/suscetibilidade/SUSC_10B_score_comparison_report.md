# SUSC-10B — Comparação Score v6 × Proxies × Baseline (review-only)

> O SUSC-10B compara scores e proxies internos de suscetibilidade em modo review-only. A concordância entre score v6, proxy v5, baseline ou evidência documental não constitui validação supervisionada, ground truth ou confirmação de ocorrência real por patch.

## 1. Objetivo
Comparar o score v6 candidato (determinístico) com o proxy v5, o baseline (06A/06B) e a evidência espacial review-only (07B/08/09A). Diagnóstico, não validação.

## 2. Diferença entre score v6, score v5, proxy e baseline
- **v6 candidato:** determinístico, pesos documentados, 19 features aprovadas, review-only.
- **v5 proxy:** score heurístico composto (SUSC-06B mostrou circularidade alta).
- **baseline 06A:** ajuste interpretável contra o proxy v5 (recuperabilidade, não predição).

## 3. Comparação global
- n=300; Pearson(v6,v5)=0.8489; Spearman(v6,v5)=0.8797.
- top-15 overlap=0.6; top-30 overlap=0.5667; concordância de classe=0.7967.

## 4. Comparação por região
Ver `SUSC_10B_score_region_diagnostics.csv` (Pearson/Spearman/diferença por região).

## 5. Top patches v6
Ver `SUSC_10A_score_v6_candidate_top_patches.csv` e o rank shift em `SUSC_10B_score_rank_shift_by_patch.csv`.

## 6. Patches com maior concordância
Patches com classe v6 = classe v5 (tercis) — ver `SUSC_10B_score_class_agreement.csv`.

## 7. Patches com maior divergência
Top 25 por |rank shift| em `SUSC_10B_score_disagreement_cases.csv`.

## 8. Relação com casos espaciais review-only
- Patches com evidência espacial (07B): ['petropolis_00467', 'recife_00019', 'recife_00229', 'recife_00276', 'recife_00299', 'recife_00322'].
- Combinações em `SUSC_10B_high_score_spatial_evidence_cases.csv` (alto+evidência / alto sem evidência / evidência sem alto).

## 9. Limitações
Correlação alta v6×v5 reflete uso parcial dos mesmos condicionantes físicos (não validação). Evidência espacial é review-only; classes por tercis sensíveis à amostra.

## 10. O que NÃO pode ser afirmado
- Que concordância v6×v5 valida o score. Que evidência espacial confirma enchente no patch. Que algo aqui é ground truth.

## 11. O que JÁ pode ser afirmado
- v6 e v5 têm relação mensurável (review-only); há patches de alta suscetibilidade com aderência espacial review-only que merecem revisão humana.

## 12. Próximo passo
SUSC-10C (sensibilidade) e SUSC-10D (DINO diagnóstico) + pacote visual SUSC-11A.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
