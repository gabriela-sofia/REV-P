# MV2-13 — Sumário Executivo

1. **O que foi feito:** construída camada programática de binding que reconcilia 281 âncoras Sentinel com inventários de 128 assets e 128 patches, resolvendo geometria/AOI/CRS, temporalidade/scene lineage e gates STAC e Dia 10 — sem baixar raster, sem STAC e sem crop.
2. **O que foi descoberto:** os dados de binding estão em dois conjuntos **disjuntos** — 128 âncoras têm o lado espacial (patch+asset+bbox+CRS) mas nenhuma cena/data; 10 têm cena Sentinel-2 mas nenhum patch/asset/geometria. Nenhuma âncora tem a cadeia completa.
3. **Força de binding:** STRONG=0, MEDIUM=128, WEAK=10, NONE=141, CONFLICT=0, INVALID=2.
4. **STAC dry-run:** 0 elegível formal; 128 rascunhos espaciais não-executados (would_download=false, would_create_crop=false).
5. **STAC real autorizado:** 0 (nenhum binding forte).
6. **Dia 10 desbloqueado:** não — 0 raster Sentinel nativo e 0 binding forte → `can_unlock_day10_now=false`.
7. **O que falta:** scene_id + acquisition_datetime para os 128 patches espacialmente prontos; patch+asset+geometria para as 2 cenas T23KPR; e raster nativo (externo) para o Dia 10.
8. **Mais próximas de STAC real:** os 128 MEDIUM (falta só temporal/cena); depois as 10 cenas T23KPR (falta vínculo espacial).
9. **Recuperação prioritária (local, sem download):** histórico de tasks GEE → destrava scene_id+datetime+cloud dos 128 patches numa única ação.
10. **Próxima ação:** recuperar o lineage temporal GEE dos 128 patches e validar (revisão humana) se as cenas T23KPR pertencem a Petrópolis — sem auto-junção por tile. Guardrails fail-closed 100% preservados; nada treinável foi criado.
