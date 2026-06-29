# SUSC-15B - descoberta de camadas/eventos nao obvios

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

SUSC-15B e review-only: nao aceita bairro-only como sucesso, nao aceita street segment sem numero/intersecao como calibracao, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

## Fontes atacadas
- IDs/codigos internos em CSVs de ocorrencia: 29 campos com valor.
- Colunas escondidas de coordenada/UTM: 23 campos coordinate-like com valor ja auditados ou bloqueados.
- Tabelas oficiais auxiliares: dependem de correspondencia manual por identificador.
- PDFs/anexos de pontos de alagamento: nao foram materializados em arquivo publico novo.
- ArcGIS/WFS/FeatureServer com nomes nao obvios: mantidos como alvo de aquisicao manual, sem download automatico nesta execucao.
- Intersecoes oficiais, numeracao predial e footprints Copernicus/EMS/SGB/INEA/APAC: requerem evidencia oficial precisa antes de T0-T4.

## Estado herdado do SUSC-15A
- Eventos avaliados: 4412
- Eventos elegiveis para calibracao: 0
- Links patch-evento precisos: 0
- Patches observacionais precisos: 0

## Decisao
SUSC-15B permanece bloqueado para calibracao automatica ate que uma fonte oficial forneca ponto, poligono, lote, intersecao ou faixa numerica controlada. Bairro-only e segmento de rua sem numero/intersecao continuam excluidos.
