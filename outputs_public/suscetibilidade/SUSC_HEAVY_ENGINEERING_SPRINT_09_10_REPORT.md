# SUSC — Relatório Mestre da Sprint Pesada (SUSC-08-C + 09A + 09B + 10A)

> Esta sprint avançou a infraestrutura operacional do REV-P para revisão humana, aquisição espacial controlada e score determinístico candidato. Nenhum artefato produzido cria ground truth, desbloqueia treinamento supervisionado ou transforma aderência documental/espacial em ocorrência confirmada por patch.

---

## 1. Estado antes da sprint

SUSC-01→07B commitados; SUSC-08 executado e validado mas **untracked** (13 casos: 9 espaciais reais — 5 bbox_overlap, 4 near_patch_buffer, 0 exact; 3 moderate, 9 weak, 1 insufficient, 0 strong). 13 coordenadas reais rastreáveis (Recife 8, Petrópolis 5). Branch `[ahead 8]`.

## 2. SUSC-08 commitado

Commit `707d035 feat: cria pacote de revisao espaco-temporal SUSC-08` (10 arquivos no escopo `suscetibilidade/`).

## 3. Hardening técnico feito

Camada comum reutilizável criada: `susc_io.py` (CSV/JSON/MD/SHA256/escrita determinística/containment), `susc_governance.py` (constantes + verificação de governança), `susc_aliases.py` (resolve aliases → `SUSC_09A_alias_resolution_report.csv`, 7 aliases: 4 exatos + 3 closest), `susc_geometry.py` (bbox/point-in-bbox/intersect/haversine/GeoJSON/bounds Brasil+regiões; shapely opcional), `susc_downloads.py` (downloader seguro: bloqueia raster/>100MB/sem-chave), `susc_common.py` (paths SUSC + matrix SHA256 + detecção de modelo). Novos scripts 09A/09B/10A usam esses módulos. Validador + teste de hardening **PASSED** (9 testes).

## 4. Workspace humano criado (SUSC-09A)

`outputs_public/suscetibilidade/SUSC_09A_human_review_workspace/` com **13 casos**: índice (md/csv), manifest, form pré-preenchido (`approved_for_ground_truth=false`) + instruções, **13 dossiês**, **13 mapas SVG**, **3 camadas GeoJSON** (patch bbox: 6 features; event geometry: 9 features; review cases: 9 features). `machine_pre_review` ∈ {candidate_for_tcc_with_caution, context_only, blocked_pending_geometry, blocked_source_conflict, needs_human_review} — nunca `approved/strong/ground_truth/training_ready`. Validador + 7 testes **PASSED**.

## 5. Aquisição externa tentada/realizada (SUSC-09B)

Registry de **13 fontes oficiais** (GeoCuritiba/IPPUC, Prefeitura/Defesa Civil Curitiba, APAC, Prefeitura/Defesa Civil Recife, SGB/CPRM, Defesa Civil Petrópolis, INEA, ANA/Hidroweb, CEMADEN, INMET, MapBiomas). **Downloads tentados: 13; bem-sucedidos: 0** (offline, sem URL direta → `not_attempted_no_direct_url_or_no_network`). Manifesto com URL/status/tamanho/sha256/motivo. Validador + 7 testes **PASSED**.

## 6. Geometrias novas encontradas ou bloqueadas

**0 geometrias externas novas** parseadas nesta passada (offline). Parser pure-Python pronto para GeoJSON/CSV/WKT/KML/KMZ; SHP/PDF registrados como metadata (sem lib). Lacuna de aquisição oficial direta permanece (manual). As geometrias reais já usadas no overlay vêm do SUSC-07B (Charter 758, Defesa Civil, CPRM), todas review-only.

## 7. Score v6 candidato criado (SUSC-10A)

`susc_score_v6_candidate_by_patch_v1.csv` — 300 patches, **19 features aprovadas**, determinístico (winsorize 1/99 → robust min-max → orientação por direção). Subíndices: topography_hydrology (0.40), rainfall_trigger (0.25), urban_spectral (0.20), vegetation_mitigation (−0.10), evidence_support (0.05). Classes por tercis globais: low 100 / medium 99 / high 101. **4 features sinalizadas (SUSC-06B) fora do score principal** (diagnóstico): curvature_laplacian_mean, rain_3d_7d_ratio, flow_acc_log_p75, water_occurrence_patch. Validador + 9 testes **PASSED**.

Top patches: `recife_00506` (1.00), `petropolis_00140` (0.978), `petropolis_00070` (0.976), `recife_00227` (0.969), `petropolis_00068` (0.950). Médias por região: recife 0.573, curitiba 0.515, petrópolis 0.480.

## 8. O que melhorou de verdade

- Linha SUSC agora tem **camada comum testada** (io/governança/geometria/downloads/aliases).
- Aliases divergentes resolvidos e auditados.
- Workspace navegável para a banca/revisor humano (dossiês + SVG + GeoJSON QGIS).
- Score v6 **determinístico e explicável** (top-5 contribuições por patch), sem treino e sem proxy.

## 9. O que continua bloqueado

- Aquisição oficial direta (GeoCuritiba/APAC/CPRM/INEA) — offline, pendente manual.
- Geometria de evento validada (footprint oficial) — inexistente; só candidatos review-only.
- Curitiba sem coordenada de evento real.
- SHP/GPKG/PDF exigem biblioteca/parse manual.

## 10. O que ainda NÃO pode ser afirmado

- Que algum patch teve enchente confirmada.
- Que o score v6 prediz ocorrência (é candidato determinístico review-only).
- Que aderência espacial é ground truth.

## 11. O que JÁ pode ser afirmado

- Há um score v6 candidato transparente, auditável e explicável por feature.
- Há um workspace completo para revisão humana dos casos espaciais.
- Toda a infraestrutura preserva governança (sem GT, sem treino, sem modelo).

## 12. Próximos passos

- Preencher o form de revisão humana (SUSC-09A) e adquirir footprint oficial validado.
- Aquisição oficial Curitiba (GeoCuritiba/IPPUC) com URL direta → re-parse/re-overlay (09B).
- SUSC-10B: comparação score v6 candidato × baseline proxy × DINO (sob revisão), ainda review-only.

---

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
