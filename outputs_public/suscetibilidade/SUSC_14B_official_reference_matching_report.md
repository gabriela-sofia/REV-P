# SUSC-14B - matching com referencias espaciais oficiais

Status: **review-only** | `allowed_for_training=false` | `can_be_ground_truth=false`

O SUSC-14B usa referencias espaciais oficiais para tentar associar ocorrencias oficiais sem coordenada a patches territoriais. Mesmo quando ha match por endereco, logradouro ou eixo viario, o vinculo permanece review-only, nao cria ground truth, nao libera treino supervisionado e nao autoriza score v7 automatico.

## 1. Gargalo herdado do 14A
O SUSC-14A reprocessou 4.412 ocorrencias oficiais de cheia, mas teve 0 matches
por endereco/logradouro e manteve score v7 bloqueado por falta de vinculos
patch-evento fortes/moderados.

## 2. Referencias oficiais descobertas
Referencias registradas: **575**.

## 3. Referencias baixadas
Manifesto de aquisicao: **575** linhas. Status: **{'not_direct_data': 62, 'reused_cached_download': 120, 'not_attempted_batch_cap': 392, 'reused_local_official': 1}**.

## 4. Features de logradouro/endereco/bairro parseadas
Features oficiais parseadas: **100000**. Por tipo: **{'address_point': 1378, 'river_line': 345, 'street_segment': 98225, 'drainage_line': 7, 'neighborhood_polygon': 45}**.

## 5. Ocorrencias de cheia reprocessadas
Ocorrencias reprocessadas: **4412**.

## 6. Matches exatos
Matches exatos por logradouro + bairro: **0**.

## 7. Matches fuzzy
Matches fuzzy/unicos: **5**.

## 8. Bloqueios por ambiguidade
Bloqueios ambiguos: **67**.

## 9. Bloqueios por ausencia de referencia
Bloqueios sem referencia oficial: **146**.

## 10. Links patch-evento
Links totais: **4414**. Links moderados: **4**. Patches
observacionais: **2**.

## 11. Score v6 x eventos
Media score v6 nos patches observacionais: **0.5679**. hit@10=0.0,
hit@20=0.0, hit@30=0.0.

## 12. Readiness
12A=False; 12B=False; 12C=False; score_v7=False.

## 13. Se score v7 segue bloqueado, por que
O score v7 segue bloqueado quando nao ha pelo menos 20 patches com evidencia
oficial consistente em duas ou mais regioes e diagnostico positivo. O SUSC-14B
nao cria score v7 automaticamente.

## 14. Limitacoes
Street/address match e no maximo moderado review-only. Bairro sozinho continua
weak/contextual e nao vira patch-level. Falhas de portal, endpoints sem dado
direto e ausencia de camada oficial completa permanecem bloqueios cientificos,
nao autorizacoes para inferir coordenadas.
