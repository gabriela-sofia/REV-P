# SUSC-11B / SUSC-11C — Observed Event Acquisition and Event-to-Patch Linkage

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

O SUSC-11B/11C organiza ocorrências reais/documentais de alagamento/inundação e testa sua ligação espacial com patches. Esses vínculos são evidência observacional review-only e não constituem ground truth supervisionado, predição operacional ou confirmação automática de ocorrência por patch.

## 1. Objetivo
Buscar, baixar quando possivel, parsear e normalizar dados reais de alagamento/
inundacao/enxurrada; construir um catalogo observacional priorizando fontes
oficiais/tecnicas com coordenada, geometria, data e tipo de evento; e testar a
ligacao espacial desses eventos com os patches do REV-P, classificando a forca do
vinculo. Tudo permanece evidencia observacional review-only.

## 2. Fontes buscadas (registry)
23 fontes-alvo registradas por regiao (Recife, Petropolis, Curitiba):
APAC, Defesa Civil, Prefeituras, CEMADEN, CPRM/SGB, ANA/Hidroweb, S2iD, INEA/RJ,
GeoCuritiba/IPPUC, IAT/Aguas Parana.

| regiao | fontes |
|---|---|
| curitiba | 8 |
| recife | 8 |
| petropolis | 7 |

## 3. Fontes baixadas
Downloads bem-sucedidos nesta passada: **0**. As fontes-alvo do registry
nao possuem URL direta de arquivo (`url_is_direct_download=false`); a aquisicao
oficial direta permanece manual. O catalogo reusa geometria local ja rastreavel
(SUSC-07B): Charter758 (mancha candidata), Defesa Civil (risco), registros de
ocorrencia e catalogos de estacao.

## 4. Fontes nao baixadas e motivo
| status | n |
|---|---|
| not_attempted_no_direct_url_or_no_network | 23 |

`not_attempted_no_direct_url_or_no_network`: fonte oficial documentada, sem URL
direta de arquivo no registry (aquisicao manual pendente). Nenhum raster, nenhum
arquivo >100MB, nenhuma API com chave, nenhum Google Maps.

## 5. Eventos observados extraidos
Total no catalogo: **13** registros (geometria rastreavel, sem
coordenada inventada).

## 6. Eventos por regiao
| regiao | eventos |
|---|---|
| recife | 8 |
| petropolis | 5 |

## 7. Eventos com geometria
13 de 13 registros tem geometria (bbox/coordenada) rastreavel.

## 8. Eventos com data
2 de 13 registros tem data/periodo explicito.

## 9. Niveis de evidencia
| nivel | n |
|---|---|
| official_occurrence_point_moderate | 4 |
| administrative_record_only | 3 |
| risk_area_context | 3 |
| documentary_context_only | 2 |
| observed_flood_bbox_moderate | 1 |

`*_strong` exige footprint validado com geometria explicita (nao disponivel
localmente). O unico poligono de mancha e digitalizacao candidata (Charter),
classificado como `observed_flood_bbox_moderate`. Setor/ponto de risco da Defesa
Civil -> `risk_area_context`; estacoes -> `administrative_record_only`.

## 10. Eventos linkados a patch (relacoes espaciais)
| relacao espacial | n linhas |
|---|---|
| event_point_near_patch | 40 |
| event_point_inside_patch | 4 |
| documentary_context_only | 2 |
| same_region_only | 1 |
| same_region_same_period | 1 |

Linhas usaveis como evidencia observacional: **0**;
patches distintos com evidencia observacional: **0**.

## 11. Links fortes / moderados / fracos
| forca | n |
|---|---|
| strong | 0 |
| moderate | 1 |
| weak | 47 |

Distribuicao de `linkage_confidence`:
| weak | 43 |
| very_weak | 4 |
| moderate | 1 |

## 12. Limitacoes
- Nenhum vinculo e ground truth ou rotulo de treino supervisionado.
- Links fortes exigiriam footprint de evento validado; nao ha localmente.
- Poligono de mancha e digitalizacao candidata (Charter) -> no maximo link moderado.
- Ponto de ocorrencia/risco != footprint de area alagada por patch.
- Sobreposicao de bbox e aproximacao; proximidade (~550m) nao confirma ocorrencia.
- `score_v6` do patch e candidato heuristico review-only, nao ocorrencia confirmada.
- Alerta != ocorrencia; setor de risco != evento; registro administrativo != patch-level.

## 13. O que NAO pode ser afirmado
- NAO se pode afirmar que um patch teve alagamento confirmado.
- NAO se pode usar estes vinculos como ground truth ou alvo de treino.
- NAO se pode tratar score_v6 alto como ocorrencia.
- NAO se pode tratar setor de risco/alerta como evento observado.

## 14. O que JA pode ser afirmado
- Existe geometria de evento rastreavel (Charter, mancha candidata) com data 2022-05.
- Ha registros oficiais de ocorrencia com coordenada (review-only) em Recife/Petropolis.
- Esses elementos tem aderencia espacial mensuravel a patches especificos (review-only).
- A cadeia de aquisicao->parsing->catalogo->linkage e reproduzivel e auditavel.

## 15. Proximo marco: SUSC-12A / 12B
Revisao humana das ligacoes moderadas (Charter x patch; ocorrencias x patch),
aquisicao oficial direta de footprint validado (APAC/INEA/Defesa Civil) e desenho
de criterio de referencia sob revisao — ainda sem ground truth automatico.
