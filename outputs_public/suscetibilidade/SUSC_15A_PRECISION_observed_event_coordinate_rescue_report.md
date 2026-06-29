# SUSC-15A-PRECISION - resgate de coordenadas precisas de eventos

Status: **review-only** | `allowed_for_training=false` | `can_be_ground_truth=false`

O SUSC-15A-PRECISION tenta obter evidencia espacial precisa para eventos observados usando apenas geometrias diretas, referencias oficiais de enderecamento/intersecao/linear referencing e footprints rastreaveis. A etapa nao usa bairro como coordenada, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

## 1. Por que bairro-only nao e suficiente
Bairro-only nao determina coordenada nem footprint de inundacao. No SUSC-15A,
T6 e sempre excluido de calibracao.

## 2. Por que SUSC-14C chegou ao limite
O SUSC-14C ampliou referencias oficiais, mas manteve 0 links moderados
evento-patch. A barreira principal e a ausencia de ponto, poligono, numero,
intersecao ou linear referencing suficiente.

## 3. Estrategia de precisao espacial
A esteira prioriza T0/T1 geometria direta, T2 endereco/lote oficial, T3
intersecao oficial e T4 linear referencing controlado. T5 e candidato fraco.

## 4. Fontes oficiais de precisao descobertas
Fontes registradas: **360**.

## 5. Referencias baixadas
Manifesto de aquisicao: **360** linhas.

## 6. Eventos com coordenada/poligono direto
Eventos T0/T1: **0**.

## 7. Eventos geocodificados por endereco oficial
Eventos T2: **0**.

## 8. Eventos por intersecao oficial
Eventos T3: **0**.

## 9. Eventos por linear referencing
Eventos T4: **0**.

## 10. Eventos excluidos por bairro-only
Eventos T6: **1598**.

## 11. Eventos elegiveis para calibracao
Eventos elegiveis: **0**.

## 12. Linkage patch-evento preciso
Links precisos de avaliacao observacional: **0**.

## 13. Score v6 x eventos precisos
Media score v6: **None**; mediana: **None**; hit@10=0.0,
hit@20=0.0, hit@30=0.0.

## 14. Readiness
12A=False; 12B=False; 12C=False; score_v7=False.

## 15. Se score v7 segue bloqueado, por que
Score v7 segue bloqueado porque a etapa nao cria alvo operacional, nao cria
ground truth e so permitiria futura calibracao humana se houver massa suficiente
de eventos precisos.

## 16. Limitacoes
Nao houve uso de Google Maps, Nominatim como evidencia, centroide de bairro ou
municipio, data inventada, coordenada inventada ou raster pesado.

## 17. Proximo marco
Executar aquisicao/manual review de fontes oficiais que contenham ponto,
poligono, lote, intersecao ou faixa numerica verificavel e entao reprocessar
T0-T4.
