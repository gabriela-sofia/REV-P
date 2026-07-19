# SUSC-18G - Recuperacao e compactacao vetorial SAR de Curitiba

## Estado herdado do 18F

- Patch_stats real: `43` linhas.
- Flood vector local herdado: `false`
- Status herdado 18F: `18F_REFERENCIA_TECNICA_SAR_CURITIBA_PARCIAL_POR_PATCH_STATS`

## Recuperacao e compactacao

- Drive acessivel ou rota local disponivel: `true`
- Vetor compacto produzido: `true`
- Vetor compacto valido: `true`
- Metodo: `compact_vector_by_patch`
- Numero de feicoes: `2`
- Resultado da compactacao: `vetor local existente`

## Overlay tecnico

- Overlays patch criados: `2`
- Vinculos tecnicos fortes somente revisao: `2`
- Status 18G: `18G_REFERENCIA_TECNICA_SAR_FORTE_COM_OVERLAY`
- Status 17B: `17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA`

## Guardrails

Sem ground truth, sem treino, sem score_v7, score_v6 intacto e sem benchmark 17B.
O footprint tecnico SAR compacto nao substitui geometria oficial de ocorrencia.

## Proxima acao pesada

Revisar o vetor compacto por patch e, se necessario, recuperar o GeoJSON exportado
do Drive para comparar com a compactacao local.
