# Plano de commit seletivo pos-18G - SUSC-18H

Este plano descreve o que deve entrar no commit. Nao executa commit.

## 1. Arquivos que devem entrar no commit (codigo e schema)

- `scripts/suscetibilidade/build_susc_18h_consolidacao_mestre_cadeia_observacional.py`
- `scripts/suscetibilidade/validate_susc_18h_consolidacao_mestre_cadeia_observacional.py`
- `scripts/suscetibilidade/susc_18h_consolidacao_common.py`
- `schemas/suscetibilidade/susc_18h_consolidacao_mestre_schema_v1.json`
- `tests/suscetibilidade/test_susc_18h_consolidacao_mestre_cadeia_observacional.py`
- `.gitignore` (nova allowlist do 18H)

## 2. Saidas publicas do 18H que devem entrar

- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/inventario_marcos_susc_17c_18g.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/linhagem_mestre_evidencia_observacional.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/matriz_qualidade_evidencia.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/matriz_regional_recife_curitiba_petropolis.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/diagnostico_prontidao_17b_mestre.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/matriz_pendencias_priorizadas.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/plano_proximos_marcos_pos_18g.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/resumo_por_regiao.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/resumo_por_status.csv`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/summary.json`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/preflight.json`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/resumo_tecnico_para_artigo.md`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/resumo_visual_para_slides.md`
- `outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/plano_commit_seletivo_pos_18g.md`
- `outputs_public/reports/SUSC_18H_CONSOLIDACAO_MESTRE_CADEIA_OBSERVACIONAL.md`
- `schemas/suscetibilidade/susc_18h_consolidacao_mestre_schema_v1.json`

## 3. Pastas de marcos anteriores versionaveis (ja no historico)

- `outputs_public/data/susc_17c5_geometry_to_patch_linkage_resolver`
- `outputs_public/data/susc_17c_strong_reference_acquisition_canary`
- `outputs_public/data/susc_17d_validacao_tecnica_evidencia_observacional`
- `outputs_public/data/susc_17e_prontidao_calibracao_observacional_exploratoria`
- `outputs_public/data/susc_17f_extracao_fisica_topografica_canarios_observacionais`
- `outputs_public/data/susc_17g_extracao_direta_features_fisicas_canarios`
- `outputs_public/data/susc_17h_calibracao_observacional_forte_somente_revisao`
- `outputs_public/data/susc_17i_ampliacao_regional_amostra_observacional`
- `outputs_public/data/susc_18a_execucao_referencia_observacional_regional`
- `outputs_public/data/susc_18b_execucao_geometrias_regionais_separacao_fenomeno`
- `outputs_public/data/susc_18c_aquisicao_geometria_oficial_curitiba`
- `outputs_public/data/susc_18d_protocolo_externo_curitiba`
- `outputs_public/data/susc_18e2_execucao_controlada_sentinel1_curitiba`
- `outputs_public/data/susc_18e_footprint_tecnico_sar_curitiba`
- `outputs_public/data/susc_18f_ingestao_validacao_footprint_sar_curitiba`
- `outputs_public/data/susc_18g_recuperacao_compactacao_vetorial_sar_curitiba`

## 4. Saidas que NAO devem entrar (ignoradas pelo .gitignore)

Pastas de marcos fora da allowlist do .gitignore (17E, 17F, 17G, 17H, 17I) e
qualquer conteudo de `local_runs/`, rasters e artefatos pesados:

- `outputs_public/reports/SUSC_18D_PROTOCOLO_EXTERNO_CURITIBA.md` (ignorado pelo .gitignore)

- `local_runs/**` nunca entra.
- Rasters `.tif`, `.tiff`, `.vrt` e embeddings `.npy`, `.npz` nunca entram.
- Nenhum path privado ou credencial entra.

## 5. Comando git add sugerido

```
git add scripts/suscetibilidade/build_susc_18h_consolidacao_mestre_cadeia_observacional.py
git add scripts/suscetibilidade/validate_susc_18h_consolidacao_mestre_cadeia_observacional.py
git add scripts/suscetibilidade/susc_18h_consolidacao_common.py
git add schemas/suscetibilidade/susc_18h_consolidacao_mestre_schema_v1.json
git add tests/suscetibilidade/test_susc_18h_consolidacao_mestre_cadeia_observacional.py
git add outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/
git add outputs_public/reports/SUSC_18H_CONSOLIDACAO_MESTRE_CADEIA_OBSERVACIONAL.md
git add .gitignore
```

## 6. Ordem de commit sugerida

1. `.gitignore` com a allowlist do 18H.
2. Modulo comum, build e validator do 18H.
3. Schema e testes do 18H.
4. Saidas publicas do 18H em `outputs_public/data/...`.
5. Relatorio publico do 18H em `outputs_public/reports/...`.

## 7. Antes do commit

- `git status --short` para conferir o que esta staged.
- Garantir staging seletivo: nada de `local_runs/`, raster ou artefato pesado.
- Rodar o validator e o pytest do 18H.

Estado consolidado do 17B: `17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA` (17B nao criado).
