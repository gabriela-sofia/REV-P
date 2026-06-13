# v2ax - Fluxo de intake geometrico Recife

A v2ax prepara os 55 patches Recife P1 e os eventos Recife comprovados para preenchimento manual.
Recife P1 e prioridade porque os pacotes `candidate_reference` seguem bloqueados sem boundary.
O checkout atual comprova apenas 1 evento Recife nos pacotes, embora a configuracao espere 3. A v2ax registra a divergencia e nao inventa eventos.

Preencha os CSVs em `datasets/manual_intake/recife_p1/`, rode
`python scripts/run_v2ax_recife_geometry_intake_pack.py`, revise os blockers e use somente os
exports validados. Depois execute v2aw, v2av e v2au, nessa ordem. O fluxo nao cria label,
ground truth final, treino ou C4 automatico.
