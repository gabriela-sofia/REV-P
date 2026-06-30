# SUSC-17C5 - Patch Grid Expansion Review

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `official_patch_created=false`; `official_patch_link_created=false`; `eligible_for_17b_now=false`.

## 1. O que o SUSC-17C4 descobriu

O SUSC-17C4 encontrou uma geometria oficial candidata real do Charter 758 para `REC_2022_05_24_30`: `MultiPolygon`, CRS EPSG:4326, `qa_status=needs_review`, `review_only=true`, `ground_truth=false`. A tabela 17C4 de patch-links permaneceu vazia.

## 2. Por que Charter 758 e real, mas ainda nao utilizavel para 17B

A geometria existe e e auditavel como candidata observacional, mas ainda nao tem QA aceito e nao intersecta a grade SUSC Recife atual. Sem `qa_status=accepted` e sem patch-link SUSC valido, o benchmark 17B continua bloqueado.

## 3. Por que cair fora da grade impede patch-link

A grade SUSC Recife atual tem `100` patches, bbox `-35.142094;-8.2271;-34.944465;-8.013666`. O Charter 758 intersecta `0` patches, com patch mais proximo `recife_00552` a `1398.4` m. Criar patch-link com intersecao zero inventaria cobertura.

## 4. Por que REC_00019 nao pode ser assumido como patch SUSC

REC_00019 pertence ao namespace historico do Protocolo C. A grade SUSC usa ids `recife_*`; nao ha match de string nem manifesto de ponte entre namespaces. Status: `namespace_incompatible`.

## 5. Expandir, manter ou apenas auditar

Nesta sprint, apenas auditar. A expansao candidata e metodologicamente aceitavel como proximo marco se for feita por protocolo proprio, manifesto de patches candidatos, extracao de features e QA humana. Ela nao e permitida como alteracao silenciosa da grade oficial.

## 6. Riscos metodologicos

Os riscos principais sao misturar namespace Protocolo C e SUSC, transformar AOI candidato em patch oficial, criar patch-link sem patch real, recomputar score sem protocolo separado, usar Charter como ground truth e executar 17B antes de QA + patch-link valido.

## 7. Artefatos necessarios para patches candidatos futuros

Sao necessarios protocolo de expansao de grade, manifesto de patches candidatos, regra de alinhamento ao grid atual, features hidrologicas e Sentinel para os candidatos, plano de QA humana e protocolo separado de recomputo de score se algum dia houver promocao.

## 8. Por que score v6 nao pode ser recalculado nesta sprint

O objetivo e revisar cobertura e risco, nao alterar o dataset oficial. Recalcular score v6 contaminaria a linha de base antes de existir patch candidato validado, features extraidas e protocolo de score proprio. O score v7 permanece inexistente.

## 9. Opcoes permitidas e bloqueadas

Permitidas agora: do_not_expand_context_only, acquire_compdec_pkg_before_expansion.

Bloqueadas agora: expand_grid_candidate_review_only, create_protocolc_to_susc_bridge_only, run_sar_runtime_on_existing_grid_only.

## 10. Proximo marco recomendado

`SUSC-17C6 Patch Grid Candidate Generation`. Em paralelo, manter a solicitacao formal de pacotes P0, especialmente `PKG_FR_REC_002`, para reduzir risco antes de qualquer expansao oficial futura.
