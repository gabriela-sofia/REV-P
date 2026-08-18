# SUSC-13C-LIVE - Aquisicao online real de eventos oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

O SUSC-13C-LIVE executa aquisição online real de fontes oficiais/rastreáveis para tentar materializar eventos observados de alagamento/inundação com data e geometria. Mesmo quando eventos fortes ou moderados são encontrados, todos os vínculos permanecem review-only, sem ground truth, sem treino supervisionado, sem score v7 automático e sem uso operacional preditivo.

## 1. Objetivo
Executar de verdade a busca online (rede habilitada) para encontrar e baixar dados
oficiais/rastreaveis de alagamento/inundacao/enxurrada e ocorrencias de Defesa
Civil em Recife, Petropolis e Curitiba.

## 2. 13B-AUTO offline x 13C-LIVE
O 13B-AUTO criou a infraestrutura mas rodou offline (0 downloads). O 13C-LIVE
habilita `SUSC_13B_NETWORK=1` e executa probes reais com status HTTP, URLs,
content-type, timestamps, erros e resultados registrados.

## 3. Healthcheck de rede
Dominios alcancados: **12/13** (minimo 3); rede utilizavel: **True**.

## 4. Dominios alcancados
- https://dados.gov.br/
- https://www.gov.br/mdr/pt-br
- https://www.sgb.gov.br/
- https://www.cemaden.gov.br/
- https://www.ana.gov.br/
- https://dados.recife.pe.gov.br/
- https://www.apac.pe.gov.br/
- https://www.recife.pe.gov.br/
- https://www.inea.rj.gov.br/
- https://www.curitiba.pr.gov.br/
- https://geocuritiba.ippuc.org.br/
- https://www.defesacivil.pr.gov.br/

## 5. Dominios bloqueados
- https://www.petropolis.rj.gov.br/

## 6. Queries live executadas
CKAN: 14 termos x 7 bases; ArcGIS: walk de raizes/servicos/layers; WFS:
GetCapabilities; HTML: crawl limitado (depth<=2, <=80 paginas/dominio, robots).

## 7-8. CKANs consultados e recursos encontrados
Fontes CKAN consultadas: 7 bases. Recursos candidatos: **28** (em 31 recursos listados).

## 9-10. ArcGIS/FeatureServer e layers
Raizes ArcGIS consultadas: 4. Layers com keyword: **2**.

## 11-12. WFS/GeoServer e layers
Endpoints WFS consultados: 3. FeatureTypes com keyword: **0**.

## 13. Links HTML oficiais encontrados
**461**.

## 14-16. Downloads
Tentativas: **33**; concluidos: **33**; bloqueados: **0**.

## 17-19. Eventos
Fortes: **0** | moderados: **2** | fracos/rejeitados: **7**.

## 20-21. Datas e geometrias
Eventos observados com data: **1** | com geometria: **2**.

## 22. Linkage evento-patch
Linhas: **15**; fortes/moderados: **0**.

## 23. Score v6 x eventos
Diagnostico: **not_enough_observed_events** | hit@10=0.0 hit@20=0.0 hit@30=0.0 | media v6 (links)=0.0.

## 24. Readiness 12A/12B/12C
12A: **BLOQUEADO** | 12B: **BLOQUEADO** | 12C: **BLOQUEADO**.

## 25. Score v7 bloqueado?
Readiness v7: **BLOQUEADO**. O 13C-LIVE **nao cria score v7** por governanca,
mesmo se os limiares fossem atingidos.

## 26. Limitacoes
- Eventos oficiais de ocorrencia (ex.: atendimentos da Defesa Civil) muitas vezes
  vem em CSV/tabela sem coordenada explicita: viram moderados/contexto, nao fortes.
- Sem coordenada/poligono explicito + data, nao ha evento forte.
- Risco/alerta/administrativo nunca viram evento observado.
- Tudo permanece review-only; nada vira ground truth, treino ou score v7.

## 27. Proximo marco
Aprofundar recursos com geometria (GeoJSON/ArcGIS/WFS) e cruzar tabelas de
ocorrencia com camadas georreferenciadas oficiais; manter revisao humana antes de
qualquer promocao. Score v7 segue como marco futuro dedicado.
