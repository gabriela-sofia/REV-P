# SUSC-08 — Revisão Humana/Metodológica dos Casos Espaciais

> O SUSC-08 não cria ground truth de enchente por patch. Ele organiza uma revisão humana/metodológica dos casos espaciais encontrados no SUSC-07B e preserva a distinção entre aderência espacial, evidência documental, suscetibilidade e validação supervisionada.

---

## 1. Objetivo do SUSC-08

Criar um pacote auditável para revisão humana dos **9 casos espaciais reais** do SUSC-07B, separando: caso forte para defesa metodológica, caso útil com cautela, caso apenas contextual e caso bloqueado por conflito geométrico/temporal. Não cria ground truth — é uma camada de validação qualitativa.

## 2. Estado herdado do SUSC-07B

13 coordenadas reais rastreáveis; 9 relações espaciais (5 `bbox_overlap` + 4 `near_patch_buffer_candidate`, 0 `exact_patch_overlap`) em 6 patches únicos; todas as coordenadas `requires_manual_review=true`.

## 3. Por que os 9 casos precisam de revisão humana

As geometrias são **candidatas/digitalizadas, de contexto (AOI), próprias fronteiras de patch, ou pontos de localidade/risco** — nenhuma é footprint oficial validado de enchente. O vínculo evidência↔coordenada é regional/manual. Logo, aderência espacial ≠ ocorrência confirmada; cada caso precisa de julgamento humano antes de qualquer uso.

## 4. Critérios de força de caso

- `strong_candidate`: exige footprint **oficial validado** + data compatível → **nenhum** disponível.
- `moderate_candidate`: ponto de risco/coord oficial dentro/próximo do patch, com cautela.
- `weak_contextual`: geometria de contexto (AOI), fronteira do próprio patch, ou bbox sem evento.
- `blocked_conflict` / `insufficient`: conflito de fontes ou sem coordenada.

## 5. Casos `bbox_overlap` (5)

- `recife_00019` ← `patch_boundary_REC_00019` — **fronteira do próprio patch** (quase circular) → `weak_contextual`.
- `recife_00229`, `recife_00276`, `recife_00299`, `recife_00322` ← `recife_digitization_aoi_context` — **AOI de digitalização (contexto)**, não footprint de evento → `weak_contextual`.

## 6. Casos `near_patch_buffer_candidate` (4)

- `recife_00276` ← `recife_defesa_civil_risk_locations` / `risk_areas` (30 pontos cada) → `moderate_candidate` (concentração de pontos de risco; **pontos de risco ≠ evento**).
- `recife_00276` ← `recife_risk_areas_context` → `weak_contextual` (camada nomeada contexto).
- `petropolis_00467` ← `official_coordinate_recovery_hardened_registry` (1 coord oficial CPRM) → `moderate_candidate` (vínculo evento→patch a revisar).

## 7. Casos regionais/documentais preservados

3 casos de contexto **Petrópolis 2022** (`same_region_period`) mantidos como contexto secundário (`weak_contextual`, `context_only`). A camada documental completa permanece no SUSC-07A/07B.

## 8. Lacuna Curitiba

1 caso `insufficient` — **nenhuma coordenada real extraída para Curitiba** nesta passada → `acquire_geometry_first` (GeoCuritiba/IPPUC/Defesa Civil PR).

## 9. Conflitos conhecidos

- `geometry_source_disagreement`: Charter 758 (polígono) × pontos Defesa Civil (Recife) — fontes discordam; o polígono Charter sequer sobrepôs patches.
- `geometry_is_context_not_event`: AOI de digitalização usada como geometria.
- `geometry_is_patch_self_boundary`: fronteira do próprio patch.
- `event_to_patch_link_requires_review` / `missing_or_weak_temporal_link`: Petrópolis CPRM.

## 10. Casos recomendados para defesa metodológica (forte)

**Nenhum** caso atinge `strong_candidate`/`tcc_strong_example` — não há footprint oficial validado. Honestidade metodológica: a defesa do TCC deve apresentar a **cadeia de aderência espacial review-only**, não uma "validação de enchente por patch".

## 11. Casos recomendados apenas com cautela (`tcc_example_with_caution`)

3 `moderate_candidate`: `recife_00276` (Defesa Civil, 2 fontes de pontos) e `petropolis_00467` (coord oficial CPRM). Usar como **exemplos ilustrativos de aderência espacial review-only**, sempre com a limitação explícita (pontos de risco/coords oficiais ≠ ocorrência por patch).

## 12. Casos bloqueados / contexto / lacuna

9 `weak_contextual` (AOI/fronteira/regionais) → `context_only`; 1 `insufficient` (Curitiba) → `acquire_geometry_first`. Nenhum `blocked_conflict` hard nesta passada (conflito Charter registrado como limitação, não como caso espacial — o Charter não sobrepôs patches).

## 13. Relação com score v6

Os patches dos casos `moderate` (ex.: `recife_00276` proxy 0.37; `petropolis_00467` proxy 0.69) podem servir de **contexto qualitativo** para inspeção do score v6, nunca como rótulo. `approved_for_score_v6_context` no formulário começa vazio (decisão humana).

## 14. Relação com baseline proxy

O SUSC-06B mostrou que o baseline é circular (recuperabilidade do heurístico). Os casos SUSC-08 não validam o baseline; oferecem âncoras espaciais review-only para discutir, com a banca, onde suscetibilidade alta coincide (ou não) com pontos de risco documentados.

## 15. Relação com DINO

DINO permanece camada latente complementar (SUSC-05/06B). Nos casos `moderate`, DINO poderá futuramente **comparar** padrões visuais dos patches aderentes vs não-aderentes — nunca como detector/ground truth.

## 16. O que ainda NÃO pode ser afirmado

- Que algum patch teve enchente confirmada.
- Que `bbox_overlap`/`near_patch_buffer_candidate` são ocorrência observada.
- Que pontos de risco da Defesa Civil ou polígonos candidatos são footprints validados.
- Que qualquer caso é ground truth ou rótulo de treino.

## 17. O que JÁ pode ser afirmado

- Existem 9 aderências espaciais reais (review-only), 3 com força `moderate`.
- A cadeia coordenada→geometria→patch é rastreável a arquivo/fonte.
- O pacote separa, de forma auditável, o que serve como exemplo-com-cautela, contexto ou lacuna.
- Toda a governança review-only/não-GT/não-treino foi preservada.

## 18. Próximo marco recomendado

**SUSC-09 — Execução da revisão humana** (preencher `SUSC_08_manual_review_form.csv`) + aquisição oficial de footprint validado (CPRM/APAC/INEA) e geometria Curitiba, antes de qualquer discussão de critério de referência. Mesmo aprovação humana **não** transforma automaticamente em ground truth.

---

## Disclaimer obrigatório

> O SUSC-08 não cria ground truth de enchente por patch. Ele organiza uma revisão humana/metodológica dos casos espaciais encontrados no SUSC-07B e preserva a distinção entre aderência espacial, evidência documental, suscetibilidade e validação supervisionada.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
