# v2ax - Handoff para operador

Abra o CSV de intake, preencha `source_type`, `geometry_value` ou `geometry_path`, `crs`,
proveniencia, documento, licenca, operador e status de revisao. Salve e rode
`python scripts/run_v2ax_recife_geometry_intake_pack.py`. Leia
`datasets/v2ax_recife_manual_intake_validation.csv`; corrija o blocker indicado. Somente exports
com linhas validadas podem alimentar v2aw/v2av/v2au.
