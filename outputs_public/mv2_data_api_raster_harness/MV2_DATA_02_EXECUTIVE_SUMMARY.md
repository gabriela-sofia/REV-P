# MV2-DATA-02 — Sumário Executivo

**Vertente B (desbloqueio científico de dados).** Construímos a infraestrutura REAL de
API/raster Sentinel (GEE / CDSE STAC / CDSE OData) + executor de raster canary, toda com
guardrails fail-closed FORTES.

## Em uma frase

O harness está pronto para consultar metadados Sentinel e preparar um download/crop canary
controlado, mas por padrão **não** chama rede, **não** baixa nada, **não** grava raster
público e **não** desbloqueia o Dia 10 — tudo isso só acontece com config + credencial +
confirmação explícita por variável de ambiente.

## Números (execução default)

- 128 targets · 3 providers registrados
- chamadas GEE/STAC/OData: **0 / 0 / 0**
- lineage API: 128 `NOT_EXECUTED` (0 strong/partial/review/conflict/not_found)
- canary candidates: **2** (âncoras oficiais PET 2022, scene_id documentado) — bloqueados por config
- canary executados: **0** · downloads **0** · crops **0** · raster nativo **0**
- vazamento público de raster: **0** · rasters privados validados: **0**
- labels/silver/gold/negativos: **0** · `can_train`: false · sandbox: bloqueado

## Guardrails

config default fail-closed · credencial só por env var · bulk download sempre proibido ·
canary só com lineage forte + `REV_P_ALLOW_RASTER_CANARY=YES` · raster só em diretório
privado git-ignored · nenhum segredo/caminho privado em `outputs_public`.

## Dia 10

**Continua bloqueado.** Só destrava após: lineage forte real via API → canary opt-in para
diretório privado → raster privado validado (bandas + SCL + checksum + CRS).
