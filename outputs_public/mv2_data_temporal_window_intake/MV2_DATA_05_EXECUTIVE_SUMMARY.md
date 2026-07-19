# MV2-DATA-05 — Sumário Executivo

**Vertente B (desbloqueio científico de dados).** Intake, validação e gate de promoção das
janelas temporais por target do corpus, a partir do template DATA-04 — sem inventar datas.

## Em uma frase

Sem template preenchido, o resultado correto é **0 janelas promovidas** e **DATA-06
bloqueado**, mas com gates de promoção e template de correção prontos para o preenchimento humano.

## Números (execução atual)

- template preenchido encontrado: **não** (`NO_FILLED_TEMPLATE_FOUND`)
- linhas de entrada: **15** (batch DATA-04)
- evidência: 15 `EMPTY` (strong/partial/review/conflict/invalid = 0)
- promovidas: STRONG **0**, PARTIAL **0**
- probe-ready GEE/STAC/OData: **0 / 0 / 0**
- API GEE/STAC/OData executadas: **0 / 0 / 0**
- downloads/crops/rasters/features: **0** · labels/silver/gold/negatives: **0**
- Dia 10: **bloqueado** · sandbox: **bloqueado** · treino: **bloqueado**
- template de correção: 15 linhas (15 precisam de correção)

## Validação fail-closed

STRONG exige target+asset/patch válidos, datas não-futuras, start≤end, janela ≤45d (ou
justificada), fonte+source_ref e review aprovado. Rejeita data futura, start>end, sem fonte,
target inexistente, asset/patch mismatch. GEE/STAC só com promoção; OData nunca aqui.

## Próximo passo (DATA-06)

Depositar `local_only/mv2_data_temporal_window/mv2_data_05_temporal_window_filled.csv` com a
janela temporal por target (evidência real, nunca inventada). Janelas STRONG/PARTIAL promovem
e liberam o metadata probe do corpus.
