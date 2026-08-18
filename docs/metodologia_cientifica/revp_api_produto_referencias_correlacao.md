# API como núcleo do produto — correlação entre txtpragab, o estado científico do REV-P e referências externas

**Status**: DOCUMENTO_DE_CORRELACAO_NAO_CANONICO — não é gate, não altera modelo/dado/label.
Complementa `PLANO_ACAO_produto_v1.md`, `revp_fase2_decisoes_design_contrato.md` e
`RELATORIO_susc_20e_api_contrato_inferencia.md`/`RELATORIO_susc_20f_pipeline_sob_demanda.md`.
Objetivo: checar, contra literatura e produtos reais publicados, se o desenho de API já
decidido (rascunho `txtpragab.docx` + Fases 1-5) está alinhado com o que a área considera
maduro — e onde evoluir a partir daqui.

---

## 1. Por que a API é mesmo o ponto de alavancagem

O "motor científico" (Firth + 6 features físicas + bootstrap, LOO-AUC=0,6781) já existia
antes do SUSC-20E. O que ele não tinha era uma interface que aceitasse uma pergunta
arbitrária ("essa coordenada aqui") e devolvesse uma resposta estruturada, com gate,
incerteza e limitação — em vez de só existir como script rodado manualmente sobre 278
linhas fixas de um CSV. É essa camada (contrato de entrada/saída + gates + geoprocessamento
sob demanda) que transforma "resultado de pesquisa reproduzível" em "coisa que um sistema
externo consegue consultar". Isso é literalmente a definição operacional de produto aqui:
não é o modelo que muda, é a superfície de acesso a ele.

## 2. O que produtos reais de risco geoespacial confirmam sobre o desenho já feito

**Google Flood Hub / Flood Forecasting API** — sistema operacional real (LSTM + modelo de
inundação, publicado em Nature/HESS por Nevo et al.) que expõe previsões via API pública.
O ponto relevante para o REV-P não é o modelo (é hidrologia de vazão, fenômeno diferente),
é a **prática de maturidade por localização**: o Flood Hub classifica cada ponto coberto por
nível de confiança (gauge verificado com métricas históricas vs. cobertura só por modelo,
sem estação local) em vez de tratar toda a cobertura como igualmente confiável. Isso é
exatamente o que `region_maturity: available | limited_evidence | insufficient` já faz no
contrato do REV-P — a diferença é que o Flood Hub tem isso publicado e testado em produção
em >5000 localizações/100 países, o que funciona como validação externa de que esse padrão
de design é o certo, não uma cautela exagerada nossa.

**OGC API - Features/Processes e STAC** — o padrão emergente (adotado por OGC, usado por
Planet, Microsoft Planetary Computer, etc.) para expor processamento geoespacial via API usa
exatamente a forma que o rascunho já escolheu por conta própria: geometria de entrada em
GeoJSON, período temporal, execução de um "processo" contra essa geometria, resposta
estruturada. O contrato do REV-P (`region.geometry`/`crs`, `period`, `requested_layers`) já
está na forma que a área padronizou — o que sugere que, se algum dia o produto precisar
interoperar com QGIS ou outro cliente GIS genérico, o caminho de menor atrito é convergir
para conformidade formal com OGC API - Processes, não redesenhar o contrato do zero.

**Literatura consolidada de flood susceptibility mapping** (reviews 2023-2025) — o conjunto
de fatores usado nesses trabalhos (elevação, declividade, TWI, HAND, distância a
drenagem/rio, chuva, uso do solo) é quase idêntico às 6 features físicas do v12. Isso
confirma, de fora, que a decisão fixa do projeto de tratar física/hidrologia como base
causal e sensoriamento remoto (SAR/DINO/orbital) como auxiliar não é uma escolha arbitrária
do REV-P — é o consenso do campo. Os estudos que usam ML mais pesado (Random Forest,
XGBoost, CNN) tipicamente adicionam essas variáveis de sensoriamento remoto como
complementares aos fatores topo-hidrológicos, nunca como substituto deles.

## 3. O que a literatura de governança de ML confirma sobre o contrato

**Model Cards for Model Reporting** (Mitchell et al., 2019) — propõe documentar, por modelo:
uso pretendido, fatores relevantes, métricas de avaliação, considerações éticas, limitações
e recomendações. O `contract_schema.py` já produz isso **por resposta**: `features_used`
(fator + contribuição + estabilidade), `evidence.sources`, `limitations[]`,
`model_version`/`data_version`. Ou seja, cada chamada à API já devolve um model card
serializado. O que falta (item de roadmap abaixo) é a versão estática/agregada — um
documento único, legível por humano, que uma banca ou revisor possa ler sem precisar chamar
a API.

**Datasheets for Datasets** (Gebru et al.) — mesma lógica aplicada ao dado de treino
(proveniência, composição, coleta, manutenção). O REV-P já tem isso em prosa espalhado
(SUSC-20B/20C, `dataset_v12_final.csv` com `rain_data_source` registrado por linha), mas
não como um documento único referenciável. Vale consolidar quando a Fase de documentação
de defesa chegar — não é urgente agora.

**Selective prediction / "aprender a abster-se"** (survey de reject option, Franc et al. e
correlatos) — a literatura trata como padrão maduro um sistema que se recusa a responder
quando a confiança/dado é insuficiente, em vez de forçar uma predição. Os status
`insufficient_data` e `region_not_supported` do contrato **são exatamente esse mecanismo**,
com a distinção adicional (já decidida no design) entre "fora da cobertura conhecida" e
"dentro da cobertura, mas sem dado suficiente nesse ponto exato" — uma granularidade que a
própria literatura de reject option recomenda (não tratar todo "não sei" como a mesma coisa).
O fail-closed do SUSC-20F (`sample_terrain_features()` retorna `None` fora do bbox do DTM,
sem interpolar) é a implementação literal disso.

## 4. O que a literatura de fusão multimodal confirma sobre o teste do DINO

A pergunta central do rascunho ("o embedding visual agrega informação além da física?") é
uma pergunta padrão em pesquisa clínica multimodal (radiologia + clínica, patologia +
genômica etc.): compara-se modelo com e sem a modalidade nova via razão de verossimilhança
e métricas de ganho incremental (NRI, IDI, decision curve analysis). A metodologia usada em
v1r5/v1r6 — LRT + correção por erro-padrão cluster-robusto quando as observações não são
independentes — é exatamente esse padrão, e é **mais rigorosa** que boa parte da literatura
clínica revisada aqui, que frequentemente reporta ΔAUC sem checar pseudorreplicação. O fato
de a correção cluster-robusta ter revertido um resultado que parecia significante
(p=0,0048 → p=0,1752) é, sob esse padrão, o comportamento esperado quando a pseudorreplicação
é real — não um "quase que dava certo", é o teste funcionando.

Um precedente direto do próprio domínio: em *Flood-DamageSense* (2025, SAR+óptico+Mamba
multimodal), o estudo de ablação encontrou que a camada de risco físico prévio
("inherent-risk feature") foi o maior contribuinte de desempenho, com sensoriamento remoto
entrando como complemento, não substituto. É o mesmo padrão qualitativo do achado do REV-P:
física/hidrologia domina, evidência visual é secundária. DINO ter ficado como evidência
auxiliar em vez de feature do score não é uma limitação do projeto — é consistente com o que
a área encontra quando testa isso com rigor.

## 5. Correlação ponto a ponto: regra fixa do projeto ↔ decisão já tomada ↔ referência

| Regra fixa do projeto | Onde já está implementada | Referência externa que confirma |
|---|---|---|
| Variáveis físico-hidrológicas são a base causal | 6 features do v12 no `score_engine` | Consenso dos reviews de flood susceptibility mapping |
| Orbital (Sentinel/DINO) é só auxiliar, nunca causal | `evidence.dino_embedding_available`, nunca em `features_used` | Ablation do Flood-DamageSense; achado A/B do próprio REV-P |
| Nunca usar score/threshold/derivado do label como feature | `score` é saída, não entra em `on_demand_feature_engine.py` | Separação entrada/saída padrão de qualquer API de inferência |
| Não misturar validação científica com otimização de performance | Fase 2 escolheu bootstrap preditivo (não delta method) por consistência metodológica, não por custo | Prática de reporting em Model Cards (métrica de avaliação declarada e justificada) |
| Priorizar interpretabilidade e coerência científica | `features_used[].contribution/stability`, `limitations[]` | Model Cards (Mitchell et al., 2019) |
| Uma tarefa por vez | Fases 1→5 sequenciais, cada uma só começa com a anterior "provada" | — (disciplina de execução, sem análogo externo direto) |
| Usar o mínimo de dados possível | SUSC-20F reusa rasters/API já existentes, não adquire dado novo | — |

## 6. Onde a API já cumpre isso, e o que ainda falta

Já cumprido: contrato no formato padrão da área (OGC-like), maturidade por região
(padrão Flood Hub), model card por resposta (padrão Mitchell et al.), reject option
granular (padrão selective prediction), teste de valor incremental do DINO com correção
estatística correta (padrão de fusão multimodal clínica, aplicado com mais rigor que a
média).

Falta, na ordem em que a literatura sugeriria priorizar:

1. **Model card estático agregado** (documento único, não por-resposta) — baixo custo, alto
   valor para defesa/revisão, usa dado que já existe.
2. **Registro machine-readable de modelo-por-região** (`{"recife": "v12", "curitiba": null,
   "petropolis": null}`) — já identificado no mapeamento de gates da Fase 2 como gate #8,
   ainda informal.
3. **Curitiba**: os Leads B/C (ANA, Global Flood Database) já deram corroboração
   hidrológica real — o próximo passo natural é replicar a extração de eventos reais
   (estilo v8/v9 do Recife) para permitir um Firth próprio ali. Só então `region_maturity`
   de Curitiba muda de `limited_evidence` para `available` — hoje a API já está correta em
   recusar.
4. **Petrópolis**: bloqueado por design (mistura enchente/deslizamento não resolvida) — sem
   ação de API até essa separação de fenômeno.
5. **Re-teste do DINO** se mais patches independentes existirem no futuro (ressalva já
   registrada na Fase 1) — a literatura clínica sugere, quando isso acontecer, triangular o
   LRT com decision curve analysis/IDI em vez de trocar de método, não substituí-lo.
6. **Conformidade OGC API - Processes**, só se/quando interoperabilidade com clientes GIS
   externos virar requisito real — não é gargalo do produto hoje.
7. Interface web + camada LLM de explicação — já sequenciado no `txtpragab.docx` como
   posterior ao contrato, permanece assim.

## 7. Papel do DINO, resumido

DINO não é o motor de decisão e não deveria virar um — nem o rascunho original, nem a
literatura da área, nem o teste A/B feito aqui apontam nessa direção. O papel correto,
confirmado por três fontes independentes (regra fixa do projeto, plano do colega, e o
padrão empírico da literatura de fusão multimodal aplicada a risco), é: evidência visual
explicável na interface, sujeita a novo teste se a amostra crescer, nunca promovida a
feature por decisão de conveniência.

---

## Referências

- [Flood Forecasting API — Google for Developers](https://developers.google.com/flood-forecasting)
- [A flood forecasting AI model, trained and evaluated globally — Google Research](https://research.google/blog/a-flood-forecasting-ai-model-trained-and-evaluated-globally/)
- [OGC API - Processes — OGC API Workshop](https://ogcapi-workshop.ogc.org/api-deep-dive/processes/)
- [SpatioTemporal Asset Catalog (STAC) Community Standard](https://docs.ogc.org/cs/25-004/25-004.html)
- [Flood susceptibility mapping: integrating machine learning and GIS — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590197424000302)
- [Enhancing Flood Susceptibility Mapping Through High-Resolution Earth Observation — MDPI Remote Sensing](https://doi.org/10.3390/rs18142418)
- [Model Cards for Model Reporting — Mitchell et al., arXiv:1810.03993](https://arxiv.org/pdf/1810.03993)
- [Automatic Generation of Model and Data Cards: A Step Towards Responsible AI — arXiv:2405.06258](https://arxiv.org/pdf/2405.06258)
- [Uncertainty-Driven Reliability: Selective Prediction and Trustworthy Deployment — arXiv:2508.07556](https://arxiv.org/pdf/2508.07556)
- [Selective Classification for Deep Neural Networks — arXiv:1705.08500](https://arxiv.org/pdf/1705.08500)
- [Flood-DamageSense: Multimodal Mamba for Building Flood Damage Assessment — arXiv:2506.06667](https://arxiv.org/abs/2506.06667)
