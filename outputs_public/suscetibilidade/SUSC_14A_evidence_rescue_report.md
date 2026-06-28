# SUSC-14A - Resgate de evidencia observacional

Status: **review-only** | `can_be_ground_truth=false` | `can_be_used_as_ground_truth=false` | `allowed_for_training=false`

O SUSC-14A tenta resgatar evidencia observacional a partir de registros oficiais sem coordenada explicita, usando apenas referencias espaciais oficiais/rastreaveis e footprints publicos quando disponiveis. Eventos geocodificados por referencia oficial permanecem review-only, nao criam ground truth, nao liberam treino supervisionado e nao autorizam score v7 automatico.

## 1. Por que 13C bloqueou
O SUSC-13C encontrou registros oficiais textuais de cheia com data, bairro e
endereco, mas sem coordenada/poligono explicito suficiente. Sem geometria oficial
patch-level, esses registros nao liberam evento forte, treino supervisionado nem
score v7.

## 2. O que o 14A tentou resolver
O 14A reprocessou registros oficiais sem coordenada explicita contra referencias
espaciais oficiais/rastreaveis e footprints publicos disponiveis, sem Google
Maps, sem API com chave, sem geocoding generico, sem centroide de bairro/municipio
e sem coordenada inventada.

## 3. Referencias oficiais descobertas
Total: **13**. Por regiao: **{'recife': 5, 'curitiba': 3, 'petropolis': 5}**. Por source_type:
**{'official': 7, 'state': 5, 'federal': 1}**. Download candidate=true: **9**.

## 4. Referencias baixadas
Tentativas registradas: **13**. Reutilizadas localmente:
**1**. Bloqueadas por rede/offline: **9**. Sem
URL direta/endpoint: **3**. Arquivos/endpoints marcados para
revisao manual: **0**.

## 5. Referencias parseadas
Features parseadas: **400**. Por feature_type: **{'address_point': 388, 'river_line': 12}**.
Com lat/lon: **400**. Com bbox/wkt/geojson_ref: **400**.

## 6. Ocorrencias oficiais sem coordenada reprocessadas
Ocorrencias reprocessadas pelo geocoder oficial: **4412**.

## 7. Ocorrencias de cheia filtradas
Ocorrencias oficiais filtradas como cheia/inundacao para tentativa de resgate:
**4412**.

## 8. Matches por endereco/logradouro/camada oficial
Matches oficiais por ponto/segmento de endereco: **0**.
Street segment matches: **0**. Contexto de bairro: **388**.
Status de geocoding: **{'blocked_no_official_reference': 4020, 'official_neighborhood_only': 388, 'blocked_ambiguous_address': 4}**. Evidence level:
**{'rejected_not_observed_event': 4024, 'weak_official_neighborhood_context': 388}**.

## 9. Casos bloqueados e motivo
Geocodificacoes bloqueadas por referencia ausente/ambigua: **4024**.
O principal motivo e `blocked_no_official_reference`: a camada oficial local
parseada cobre pontos especificos e nao cobre os logradouros predominantes das
ocorrencias de cheia. Bloqueadas sem referencia: **4020**.

## 10. Footprints descobertos ou ausentes
Fontes registradas: **10**. Footprints vetoriais disponiveis:
**0**. Status: **{'network_disabled': 9, 'not_a_vector_candidate': 1}**.

## 11. Eventos resgatados fortes/moderados/fracos
Fortes: **0**. Moderados: **2**. Observados:
**2**. Fracos/contextuais: **27**. Bloqueados/documentais:
**2**.

## 12. Catalogo consolidado
O catalogo consolidado preserva os registros preferenciais e suas flags de
governanca. Todos permanecem review-only, sem ground truth e sem treino.

## 13. Linkage evento-patch
Links totais: **31**. Links fortes: **0**. Links
moderados: **0**. Links fracos: **0**. Links
contextuais: **25**.

## 14. Links fortes/moderados/fracos/contextuais
Links fortes/moderados: **0**. Patches com avaliacao
observacional: **0**. Patches apenas contextuais:
**0**.

## 15. Score v6 x eventos
Diagnostico: **not_enough_observed_events**. Media v6: **None**. Mediana v6:
**None**. Distribuicao: **null**. hit@10=None,
hit@20=None, hit@30=None, enriquecimento=None. Sem links
fortes/moderados suficientes, essas metricas ficam nulas.

## 16. Readiness 12A
SUSC-12A temporal: **BLOQUEADO**. Requer pelo menos 10 eventos/linkages
fortes ou moderados com data/periodo.

## 17. Readiness 12B
SUSC-12B contraste de features: **BLOQUEADO**. Requer pelo menos 10
patches fortes/moderados/controlaveis.

## 18. Readiness 12C
SUSC-12C calibracao de proxy: **BLOQUEADO**. Requer pelo menos 20
patches fortes/moderados em pelo menos duas regioes.

## 19. Readiness score v7
Score v7: **BLOQUEADO**.

## 20. Por que score v7 continua bloqueado ou nao
O score v7 permanece bloqueado porque nao ha quantidade minima de vinculos
patch-evento fortes/moderados com data e geometria oficial/rastreavel. O SUSC-14A
melhora a rastreabilidade do bloqueio, mas nao autoriza calibracao automatica.

## Achado negativo principal

A tentativa offline de georreferenciar ocorrencias oficiais de cheia a partir da
referencia oficial disponivel nao produziu matches geocodificados suficientes. A
camada oficial local parseada cobre pontos especificos e nao cobre os logradouros
predominantes das ocorrencias de cheia. Portanto, o gargalo de 13C permanece: ha
ocorrencia oficial textual com data/bairro/endereco, mas ainda nao ha geometria
oficial suficiente para validacao patch-level.

## Consequencia para score v7

O score v7 permanece bloqueado porque nao ha quantidade minima de vinculos
patch-evento fortes/moderados com data e geometria oficial/rastreavel. O SUSC-14A
melhora a rastreabilidade do bloqueio, mas nao autoriza calibracao automatica.

## 21. Limitacoes
Ocorrencia geocodificada por logradouro oficial e no maximo moderada; nunca
strong. Street segment match nunca vira strong. Bairro, municipio, risco, alerta
e suscetibilidade nao viram ocorrencia observada patch-level.

## 22. O que nao pode ser afirmado
Nao pode ser afirmado que ha ground truth, treino supervisionado liberado,
validacao operacional, score v7 pronto, ou relacao causal patch-evento.

## 23. O que ja pode ser afirmado
Pode ser afirmado que o gargalo foi reprocessado de forma rastreavel e
fail-closed. A referencia oficial local disponivel e insuficiente para converter
as ocorrencias textuais em geometria patch-level. O achado negativo e
reprodutivel pelos artefatos 14A.

## 24. Proximo marco recomendado
SUSC-14B: aquisicao com rede habilitada da camada completa de logradouros/eixos
de via e bairros (Recife/Curitiba/Petropolis) e de footprints vetoriais oficiais
com data, mantendo review-only e revisao humana antes de qualquer promocao.
