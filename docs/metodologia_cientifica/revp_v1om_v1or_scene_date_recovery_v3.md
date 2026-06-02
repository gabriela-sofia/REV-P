# Protocolo C v1om-v1or — Sentinel Scene Date Recovery v3

## O que foi tentado

O bloco v1om-v1or realiza recuperação de datas Sentinel exclusivamente via fontes de metadado
legítimas: arquivos SAFE, MTD XML, STAC JSON, manifest.safe, sidecars textuais, e IDs de produto
Sentinel com timestamp embutido.

## Fontes que contam como scene_date confirmada

- Nome de produto SAFE Sentinel-2: `S2A_MSIL1C_YYYYMMDDTHHMMSS_...SAFE`
- Nome de produto SAFE Sentinel-1: `S1A_IW_GRDH_..._YYYYMMDDTHHMMSS_...SAFE`
- MTD XML: campos `PRODUCT_START_TIME`, `PRODUCT_STOP_TIME`, `SENSING_TIME`,
  `DATATAKE_SENSING_START`, `GENERATION_TIME`
- STAC JSON/GeoJSON: campos `datetime`, `start_datetime`, `end_datetime`,
  `properties.datetime`
- ID de produto Sentinel genérico com timestamp compacto embutido (20YYMMDDTHHMMSS)

Todos exigem vínculo patch→asset→metadado oficial para resultar em
`PRODUCT_DATE_CONFIRMED` e `can_unlock_temporal=true`.

## Fontes bloqueadas

As seguintes fontes foram explicitamente bloqueadas e nunca produzem `scene_date`:

- `MANIFEST_FIELD` genérico (ex: `manifestCreationDate`, `processingDate`)
- IDs de evento REC (ex: `REC-20220415`, `RECIFE_00123`)
- Janela temporal de evento (`event_window_*`)
- Nome derivado de patch (`patch_derived_*`, `REC_YYYY_MM_DD`)
- Data de modificação de arquivo (mtime)
- Data de execução da pipeline
- `YYYYMMDD` isolado sem contexto de produto Sentinel

## Por que filename/manifest/event window não confirma cena Sentinel

O nome de arquivo de patch (`REC_2022_04_15_01.tif`) é gerado internamente pela pipeline e não
carrega proveniência de produto Sentinel. O campo `manifestCreationDate` em `manifest.safe` é
a data de empacotamento, não a data de aquisição. Uma janela de evento (ex: ±30 dias do evento)
é uma estimativa, não uma data de cena observada. Apenas o nome do produto SAFE ou campos
de sensing/aquisição em MTD XML/STAC são considerados datas de cena válidas.

## Resultado v1om-v1or

| Métrica | Valor |
|---|---|
| Sidecars com data permitida | 0 |
| Produto dates confirmadas (parser) | 0 |
| Produto dates prováveis/review-only | 0 |
| Blocked non-scene-date | 0 |
| Patches com scene_date confirmada | 0 |
| Patches que desbloqueiam temporal | 0 |
| C3+ review candidates | 0 |
| C4 aberto | false |
| Formal negatives | 0 |
| Fila DINO | 0 |
| Status DINO | REVIEW_ONLY_REPRESENTATION |

## Implicação para C3, C4 e DINO

- **C3+**: só existe se `scene_date_status == PRODUCT_DATE_CONFIRMED`, regra temporal
  satisfeita (strong/moderate/contextual) e `formal_negative_count > 0`.
  C3+ requer revisão humana — não é label operacional.
- **C4**: fechado enquanto `formal_negative_count == 0`. A presença de scene_date confirmada
  não abre C4 por si só.
- **DINO**: permanece `REVIEW_ONLY_REPRESENTATION`. A fila DINO contém apenas patches com
  scene_date confirmada ainda em revisão. DINO não cria label, não valida evento,
  não alimenta treino.

## Guardrails

`can_create_operational_label`, `can_train_model` e `ground_truth` são sempre `false`
em todos os outputs deste bloco. Nenhum pixel foi lido. Nenhum path absoluto está presente
nos outputs versionáveis.
