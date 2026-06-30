# SUSC-16C calibration design matrix

Status: review-only. A matriz nao cria score v7 e nao altera pesos do score v6.

| feature_or_component | observed_issue | proposed_action | eligible_for_16D | eligible_for_score_v7_future |
|---|---|---|---|---|
| hand_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| slope_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| elevation_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| distance_to_water_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| twi_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| flow_accumulation_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| urban_prop | stable_support | increase_weight | true | false |
| vegetation_prop | contradictory | block_due_to_insufficient_evidence | false | false |
| water_prop | contradictory | block_due_to_insufficient_evidence | false | false |
| ndvi_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| mndwi_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| ndbi_mean | contradictory | block_due_to_insufficient_evidence | false | false |
| chirps_3d_mm | contradictory | block_due_to_insufficient_evidence | false | false |
| chirps_7d_mm | contradictory | block_due_to_insufficient_evidence | false | false |
| chirps_30d_mm | contradictory | block_due_to_insufficient_evidence | false | false |
| runoff_context_7d | contradictory | block_due_to_insufficient_evidence | false | false |
| hydrological_component | component_requires_review | keep_unchanged | true | false |
| urban_component | component_underweighted_candidate | increase_weight | true | false |
| rainfall_runoff_component | component_underweighted_candidate | increase_weight | true | false |
| spectral_component | component_requires_review | keep_unchanged | true | false |
| topographic_component | component_requires_review | keep_unchanged | true | false |
| documentary_component | component_requires_review | keep_unchanged | true | false |
