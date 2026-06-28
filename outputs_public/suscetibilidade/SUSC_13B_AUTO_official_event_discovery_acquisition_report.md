# SUSC-13B-AUTO - Descoberta e aquisicao automatica de eventos oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

O SUSC-13B-AUTO realiza descoberta e aquisição automática de fontes oficiais/rastreáveis para fortalecer a camada observacional de alagamento/inundação. Mesmo quando encontra eventos fortes, a etapa mantém todos os vínculos em modo review-only, não cria ground truth, não treina modelo supervisionado e não cria score v7 automaticamente.

## 1. Objetivo
Criar um motor automatico de busca, descoberta, download, parsing e validacao de
dados oficiais/rastreaveis de alagamento, inundacao, enxurrada e ocorrencias de
Defesa Civil para Recife, Petropolis e Curitiba, sem depender de upload manual.

## 2. Por que 13A/12B ainda eram insuficientes
O SUSC-13A fechou com 0 eventos fortes, 5 moderados, 1 link patch-evento
moderado, 29 links fracos por buffer e 0 downloads diretos. O gargalo central era
a ausencia de fonte oficial com data + geometria explicita. O 12B/12C nao tinha
contraste observacional robusto o suficiente para calibrar proxy.

## 3. Estrategia de descoberta automatica
Plano deterministico de consultas (CKAN, ArcGIS REST, WFS/GeoServer, portais de
dados abertos, HTML oficial), com probe live opt-in (`SUSC_13B_NETWORK=1`).
Offline e o modo padrao: cada tentativa registra `network_disabled` e nada e
fabricado. robots.txt e respeitado; sem chave de API; sem Google Maps; sem
geocoding generico; sem raster; limite de 250MB.

## 4. Portais consultados
Total de fontes/endpoints candidatos registrados: **25**.

| metodo | n |
|---|---|
| data_portal_html | 11 |
| ckan_api | 8 |
| arcgis_rest | 5 |
| wfs_getcapabilities | 1 |

## 5. Endpoints encontrados
Probes executados: **25**. Status por probe registrado em
`SUSC_13B_auto_discovery_debug_log.csv`.

## 6. CKANs consultados
Endpoints CKAN registrados: **8** (Recife, PE, RJ, PR, ANA, dados.gov).

## 7. ArcGIS/FeatureServer consultados
Servicos ArcGIS REST registrados: **5** (GeoCuritiba/IPPUC, GeoSGB, INEA, ESIG Recife).

## 8. WFS/GeoServer consultados
Endpoints WFS/GeoServer registrados: **1** (IAT/Aguas Parana).

## 9. Sitemaps/HTML oficiais consultados
Portais HTML oficiais registrados: **11** (prefeituras, APAC, DRM-RJ, S2iD, CEMADEN).

## 10. Downloads realizados
Tentativas de download: **0**; concluidos: **0**.

| status | n |
|---|---|
| nenhum | 0 |

## 11. Downloads bloqueados e motivo
Raster, Sentinel bruto, executavel, tipo desconhecido e arquivos acima de 250MB
sao bloqueados por politica. Offline, todos ficam como `not_attempted_network_disabled`.

## 12. Eventos fortes encontrados
**0**. Evento forte exige fonte rastreavel, data/periodo e
geometria/coordenada explicita de alagamento/inundacao.

## 13. Eventos moderados encontrados
**2**.

## 14. Eventos fracos/rejeitados
**6** (risco, alerta, administrativo, documental ou rejeitado).

## 15. Geometrias obtidas
Eventos observados com geometria: **8**.

## 16. Datas obtidas
Eventos observados com data/periodo: **2**.

## 17. Linkage patch-evento
Linhas de linkage: **14**; links fortes/moderados: **0**;
linhas avaliaveis (observacional review-only): **7**.

## 18. Diagnostico score-evento
bloqueado (not_enough_observed_events): apenas 0 link(s) forte/moderado com patch resolvido; limiar minimo 10.

## 19. Readiness para 12A/12B/12C
- 12A temporal: **false**
- 12B contraste de features: **false**
- 12C calibracao de proxy: **false**

## 20. Score v7 continua bloqueado?
Readiness para score v7: **false**. Mesmo se PRONTO, o
SUSC-13B-AUTO **nao cria score v7**: a criacao exige revisao humana e etapa
dedicada. Aqui o score v7 permanece bloqueado por governanca.

## 21. Limitacoes
- Offline e o padrao; sem rede nenhum evento novo e adquirido automaticamente.
- Endpoints sao raizes institucionais; nenhum URL de download direto e inventado.
- Risco, alerta, suscetibilidade e registro administrativo nunca sao evento observado.
- Centroide de municipio/bairro, Google Maps e geocoding generico nao sao geometria.
- Eventos fortes exigem data + geometria explicita; nada vira ground truth ou treino.

## 22. Proximo marco
Executar o probe live (`SUSC_13B_NETWORK=1`) em ambiente com rede autorizada para
materializar recursos CKAN/ArcGIS/WFS; ou intake manual de um arquivo oficial com
data + geometria no diretorio de aquisicao, o que reexecuta parse/consolidacao/
linkage/readiness automaticamente. Score v7 segue como marco futuro dedicado,
fora do escopo automatico.
