# Protocolo C v2am - limitacoes (apendice)

Limitacoes formuladas como controle metodologico, nao falha descontrolada.

## no_operational_patch_ground_truth
- Explicacao segura: Controlled limitation: no operational patch-level reference is claimed.
- Nao implica: The pipeline failed.
- Mitigacao: Stop gate and blockers documented.
- Trabalho futuro: Use new qualified source and review.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## no_explicit_anchor_sentinel_date_crosswalk
- Explicacao segura: Dates remain unlinkable without explicit crosswalk.
- Nao implica: Dates can be inferred by region.
- Mitigacao: v2ag and v2ah documented block.
- Trabalho futuro: Find versionable crosswalk source.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## no_observed_geometry
- Explicacao segura: Geometry absence is tracked as blocker.
- Nao implica: No spatial work was done.
- Mitigacao: Geometry blocker matrix exists.
- Trabalho futuro: Acquire institutional geometry.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## no_occurrence_coordinates
- Explicacao segura: Contextual coordinates do not create patch truth.
- Nao implica: Context coordinates are event coordinates.
- Mitigacao: Coordinate blockers retained.
- Trabalho futuro: Acquire verified coordinates.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## heterogeneous_external_evidence
- Explicacao segura: Regional heterogeneity is disclosed and routed.
- Nao implica: Evidence is inconsistent or invalid.
- Mitigacao: Regional registries preserve status.
- Trabalho futuro: Normalize future source intake.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## pending_human_review
- Explicacao segura: Review package is prepared but pending.
- Nao implica: Review was simulated.
- Mitigacao: v2ai assignments/templates exist.
- Trabalho futuro: Execute real review.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## pending_adjudication
- Explicacao segura: Adjudication queue waits for completed reviews.
- Nao implica: Consensus exists.
- Mitigacao: v2ai adjudication queue exists.
- Trabalho futuro: Run adjudication after reviews.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## dino_support_only
- Explicacao segura: DINO can support review routing only.
- Nao implica: DINO validates events.
- Mitigacao: DINO guardrails retained.
- Trabalho futuro: Keep DINO support-only.
- Secao recomendada: limitacoes_e_trabalhos_futuros

## gis_context_only
- Explicacao segura: GIS context informs review without labels.
- Nao implica: GIS creates labels.
- Mitigacao: Safe-use registries retained.
- Trabalho futuro: Use GIS with explicit source evidence.
- Secao recomendada: limitacoes_e_trabalhos_futuros

