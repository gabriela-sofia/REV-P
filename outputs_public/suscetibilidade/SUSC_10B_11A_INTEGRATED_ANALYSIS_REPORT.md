# SUSC-10B→11A — Relatório de Análise Integrada (review-only)

> A análise integrada SUSC-10B→11A consolida diagnósticos review-only de suscetibilidade urbana a enchentes. Ela não cria ground truth, não valida ocorrência real por patch, não treina modelo supervisionado e não autoriza uso operacional preditivo.

---

## 1. Estado inicial

SUSC-01→10A commitados; score v6 candidato (300 patches, 19 features, classes low 100/medium 99/high 101); 9 relações espaciais review-only (07B/08); workspace humano (09A); aquisição externa offline (09B).

## 2. Score v6 candidato

Determinístico, explicável, escala 0–1, top patch `recife_00506` (1.00). Subíndices: topo-hidro (0.40), chuva (0.25), urbano-espectral (0.20), vegetação-mitigação (−0.10), evidência-suporte (0.05). Médias por região: recife 0.573, curitiba 0.515, petrópolis 0.480.

## 3. Comparação v6 × v5/proxy (SUSC-10B)

- Pearson(v6,v5) = **0.85**; Spearman = **0.88**; top-15 overlap = **0.60**; top-30 = **0.57**; concordância de classe = **0.80**.
- A alta correlação é **esperada e não é validação**: v6 e v5 compartilham condicionantes físicos (v5 é circular, conforme SUSC-06B).

## 4. Comparação v6 × baseline proxy (SUSC-06A/06B)

O baseline 06A ajustava o proxy v5 (R²_CV 0.92 = recuperabilidade, não predição). O v6 é determinístico e **não** ajustado a target — substitui o ajuste circular por composição transparente de condicionantes.

## 5. Sensibilidade (SUSC-10C)

11 cenários. Estabilidade medida por **consistência de classe** (não pelo range bruto reescalado): ~38 patches instáveis, ~48 patches `high` robustos. Ablação mais impactante: **`no_topography_hydrology`** (coerente com o maior peso). O score é estável em rank/classe sob mudanças moderadas; remover o grupo topo-hidro é o que mais altera o ranking.

## 6. DINO (SUSC-10D)

**Ausente**: a descoberta encontrou ~348 arquivos DINO (registries/scaffolds), mas **nenhum vetor por patch carregável** (`.npz/.npy` git-ignored/externos). Relatório de ausência honesto; nenhuma similaridade/PCA computada. DINO permanece camada complementar futura, nunca classificador.

## 7. Casos espaciais review-only

9 relações (5 bbox_overlap + 4 near_patch_buffer) em 6 patches; 0 exact. 3 `moderate_candidate`, 9 `weak_contextual`, 1 `insufficient`, **0 strong**. Patches com score alto + evidência espacial review-only estão em `SUSC_10B_high_score_spatial_evidence_cases.csv` para revisão humana.

## 8. Pacote visual (SUSC-11A)

9 figuras SVG (distribuição, por região, top-15, sensibilidade, v6×v5, casos espaciais, pipeline, matriz conceitual, DINO), 5 tabelas, 8 slides, GeoJSON. Toda figura traz rodapé `review-only — NÃO ground truth`.

## 9. O que é defensável no TCC

- Cadeia metodológica auditável e governada (sem GT/treino/modelo).
- Score v6 determinístico, explicável por feature, com sensibilidade documentada.
- Aderência espacial review-only entre suscetibilidade e geometria rastreável (Recife/Petrópolis).
- Separação rigorosa: suscetibilidade ≠ evento observado ≠ ground truth.

## 10. O que deve ser dito com cautela

- Concordância v6×v5 (não é validação; é recuperação parcial dos mesmos condicionantes).
- Os 3 casos `moderate_candidate` (pontos de risco/coord oficial; não footprint).
- Estabilidade do score (boa em classe/rank, mas pesos são escolha metodológica).

## 11. O que NÃO pode ser dito

- Que algum patch teve enchente confirmada.
- Que o score v6 prediz enchente ou é operacional.
- Que evidência documental/espacial é ground truth.
- Que DINO classifica ou valida.

## 12. Próximo marco recomendado

**SUSC-11B / SUSC-12**: preenchimento do form de revisão humana (09A), aquisição oficial direta (GeoCuritiba/APAC/CPRM) para destravar 09B/10D, e — só após footprint validado sob revisão — discutir critério de referência. Tudo permanece review-only até decisão humana documentada.

---

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
