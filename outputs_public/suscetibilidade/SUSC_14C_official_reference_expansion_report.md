# SUSC-14C - expansao oficial de referencias e reprocessamento observacional

Status: **review-only** | `allowed_for_training=false` | `can_be_ground_truth=false`

O SUSC-14C expande a aquisicao de referencias oficiais e reprocessa ocorrencias de cheia para aumentar vinculos observacionais evento-patch. Todos os vinculos permanecem review-only, sem ground truth, sem treino supervisionado e sem score v7 automatico.

## 1. Escopo
O SUSC-14C parte dos bloqueios do SUSC-14B, audita falhas de matching,
executa lote adicional de aquisicao oficial e recompila a uniao de referencias
espaciais rastreaveis.

## 2. Referencias oficiais
Referencias herdadas do registro 14B: **575**. Manifesto adicional
14C: **392** linhas. Status de aquisicao: **{'reused_cached_download': 52, 'downloaded': 28, 'not_attempted_batch_cap': 312}**.

## 3. Features parseadas
Novas features 14C parseadas: **50000**. Features totais na
uniao: **12158**. Por tipo: **{'address_point': 1244, 'river_line': 249, 'street_segment': 10369, 'drainage_line': 7, 'neighborhood_polygon': 289}**.

## 4. Reprocessamento de ocorrencias
Ocorrencias reprocessadas: **4412**. Status inicial: **{'blocked_no_official_reference': 66, 'official_neighborhood_only': 4326, 'blocked_ambiguous_address': 19, 'official_address_point_match': 1}**.

## 5. Matching
Matches exatos: **0**.
Matches fuzzy: **1**.

## 6. Ambiguidade
Ambiguos resolvidos por criterio estrito: **0**. Ambiguos
remanescentes: **19**. Ambiguidade remanescente continua bloqueada
para patch-level e para avaliacao observacional.

## 7. Bloqueios
Bloqueios sem referencia oficial: **66**.
Nenhum bloqueio foi preenchido por Google Maps, geocoding generico,
centroide municipal ou bairro como patch-level.

## 8. Links evento-patch
Links totais: **4412**. Links moderados: **0**.
Patches observacionais: **0**.

## 9. Score v6 x eventos
Media score v6 nos patches observacionais: **None**. Mediana:
**None**. hit@10=0.0, hit@20=0.0, hit@30=0.0.

## 10. Readiness
12A=False; 12B=False; 12C=False; score_v7=False.

## 11. Por que score v7 segue bloqueado
O SUSC-14C nao cria score v7 automaticamente. Mesmo quando ha vinculo oficial
por endereco/logradouro, o vinculo e moderado, revisavel e insuficiente para
ground truth ou treino supervisionado.

## 12. Limitacoes
Falhas de portal, endpoints sem arquivos parseaveis, ausencia de geometria
oficial e ambiguidade entre logradouros continuam como bloqueios. Bairro sozinho
permanece contexto fraco e nunca resolve patch-level.
