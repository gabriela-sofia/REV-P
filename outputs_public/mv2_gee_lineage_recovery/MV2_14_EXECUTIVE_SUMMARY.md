# MV2-14 — Sumário Executivo

1. **O que foi feito:** camada de recuperação de lineage GEE que varre fontes locais, extrai candidatos de cena/data/cloud/collection e reconcilia com os 128 bindings MEDIUM do MV2-13 — sem baixar raster, sem STAC, sem crop.
2. **O que foi descoberto:** 841 fontes GEE/export relevantes geraram 111 candidatos de lineage; existe lineage GEE local **real** (scene_id, data, cloud, `COPERNICUS/S2_SR_HARMONIZED`), mas pertence ao track de **âncora oficial do Protocolo C** (Petrópolis, tile T23KPR), num **namespace diferente** do corpus; e só há cenas locais da **zona 23** (Petrópolis), nenhuma das zonas 22 (Curitiba) ou 25 (Recife).
3. **Força de lineage:** STRONG=0, PARTIAL=0, CANDIDATE_REVIEW=48 (Petrópolis), CONFLICT=0, INVALID=0, NOT_FOUND=80 (Curitiba 43 + Recife 37).
4. **STAC dry-run:** 0 ; **STAC real review-required:** 0 (nenhum binding ficou STRONG porque nenhum candidato referencia o `patch_id` do corpus).
5. **Cenas T23KPR auto-vinculadas:** **0** — coincidência de tile/zona é hipótese, nunca vínculo (`auto_joins_performed=0`).
6. **Dia 10 desbloqueado:** não — `can_unlock_day10_now=false` em todos (0 raster nativo, 0 binding forte).
7. **O que falta:** scene_id+datetime+cloud+proveniência **keyed ao patch do corpus**; para Petrópolis, confirmar humano quais cenas T23 correspondem a quais patches; para Curitiba/Recife, recuperar o export GEE próprio.
8. **Recuperação manual GEE:** histórico de tasks → script de export → PRODUCT_ID/scene_id → data/cloud/tile, exportando só metadado leve (ver fila e template).
9. **Sucesso mínimo atingido:** fila objetiva de recuperação GEE + template preenchível keyed a cada patch real (lineage vazio por design).
10. **Próxima ação:** preencher o template após checagem no GEE; ao fechar scene_id+datetime por patch, o binding vira STRONG → habilita STAC dry-run (ainda sem download). Guardrails fail-closed 100% preservados; nada treinável criado.
