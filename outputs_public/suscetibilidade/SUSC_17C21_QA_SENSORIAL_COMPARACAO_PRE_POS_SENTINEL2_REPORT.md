# SUSC-17C21 - QA sensorial e comparacao pre/pos-evento Sentinel-2 review-only

## Objetivo
Fazer a segunda metade operacional do fluxo sensorial: QA dos recortes pre-evento reais do 17C20, selecao e materializacao de cena pos-evento equivalente, comparacao pre/pos (deltas) e auditoria anti-leakage, tudo review-only.

## QA pre-evento
- QA concluido para 5 patches; aprovado para 5.
- Cada artefato 17C20 foi verificado: existencia, SHA256, tamanho, formato, bandas B03/B04/B08/B11, indices NDVI/NDWI/MNDWI/NDBI, valid pixel count, leitura do preview, marcacao de fallback e cena pre-evento.

## Selecao e materializacao pos-evento
- Cenas pos-evento selecionadas: 5 (cena canonica pos-evento: S2B_MSIL1C_20220601T124319_N0510_R009_T25MBM_20240612T164504.SAFE).
- Patches com artefato pos-evento: 5; artefatos pos-evento: 15.
- Mesma estrategia do 17C20: CDSE/OData e produto completo (bloqueado como primario), materializacao via fallback STAC/COG Earth Search por janela HTTP range, sem baixar produto.

## Comparacao pre/pos
- Patches com delta pre/pos: 5; features de delta: 40.
- delta = pos-evento - pre-evento, para B03/B04/B08/B11_mean e NDVI/NDWI/MNDWI/NDBI_mean.
- Todo delta e `observational_change_review_only`: nao e feature pre-evento, nao e label, nao e ground truth, nao e score.

## Guardrails
- Pos-evento usado como feature pre-evento: 0.
- Delta usado como label: 0.
- Produtos completos baixados: 0; `.SAFE`/ZIP: 0; raster pesado commitado: 0; embeddings: 0; Ground Reference: 0; ground truth: 0; label: 0.

## Score v6, score v7 e 17B
Score v6 intacto, score v7 inexistente, 17B bloqueado ate existir artefato de evento com geometria e fenomeno, aceite formal de QA e revisao de politica do fallback.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C22 Revisao humana formal do QA sensorial e das mudancas observacionais pre/pos antes de qualquer politica de patch candidato
