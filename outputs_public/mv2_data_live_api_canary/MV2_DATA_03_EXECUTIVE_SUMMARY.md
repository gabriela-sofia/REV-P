# MV2-DATA-03 — Sumário Executivo

**Vertente B (desbloqueio científico de dados).** Testamos, de forma controlada, a transição
do harness fail-closed (DATA-02) para a camada REAL de API/metadados Sentinel + um raster
canary privado opcional — só sobre as 2 âncoras oficiais (Protocolo-C), nunca o corpus.

## Em uma frase

Sem config local privada, o sucesso é **bloquear corretamente**: 0 chamadas, 0 downloads, 0
raster, corpus intacto. Com config/env explícitos, o máximo permitido é **1** raster canary
privado oficial, validado, que **não** desbloqueia o corpus nem o Dia 10.

## Números (execução atual, CONFIG_MISSING)

- config local: **não encontrada** → runbook gerado, nenhuma rede
- flags: allow_network/metadata/raster_download/canary = **todas false**
- candidates oficiais: **2** (Petrópolis 2022 / CPRM, TECHNICAL_CANARY_ONLY)
- probes GEE/STAC/OData: **0 / 0 / 0**
- consenso: REGISTRY_ONLY_STRONG **2**, API_CONFIRMED_STRONG 0, conflitos 0
- raster canary executado: **0** · downloads **0** · raster nativo **0** · vazamento **0**
- validação privada: `NO_RASTER_EXECUTED_OK`
- corpus targets baixados: **0** · Dia 10: **bloqueado** · sandbox: **bloqueado** · treino: **bloqueado**

## Guardrails

config local obrigatória p/ rede · credencial só por env var (valores nunca logados) · máx 1
canary · canary só com consenso forte + `REV_P_ALLOW_RASTER_CANARY=YES` · raster só em
diretório privado git-ignored · nenhum segredo/caminho privado/raster em `outputs_public` ·
`can_use_for_corpus_day10=false` sempre.

## Dia 10 do corpus

**Continua bloqueado.** O canary oficial prova o mecanismo técnico (auth → probe → download
privado → checksum → bandas/SCL); ele não recupera o lineage dos 128 nem desbloqueia o corpus.
