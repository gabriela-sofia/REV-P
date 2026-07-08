# SUSC-GT01 - Ground Reference Policy & Evidence State Taxonomy

Politica oficial e validavel de referencia observacional por patch do REV-P. Este
marco e puramente metodologico: nao busca internet, nao baixa raster, nao roda
SAR, nao cria embeddings, nao altera o score_v6, nao cria score_v7 e nao treina
modelo. Ele formaliza no codigo a taxonomia de estados de evidencia e as regras
que impedem transformar suscetibilidade, score, alerta, area de risco ou noticia
em rotulo de campo.

## 1. Por que o REV-P nao usa ground truth binario agora

O REV-P nao trata "ground truth" como rotulo binario absoluto de alagou/nao
alagou. Rotular patches como positivo/negativo exigiria geometria de ocorrencia
confiavel, separacao de fenomeno, janela temporal pre-evento e QA — que hoje nao
existem de forma consistente. Rotular sem isso produziria vazamento, negativos
falsos e overclaim. Enquanto essa base nao esta pronta, a referencia observacional
so pode ser usada como **evidencia review-only**, nunca como verdade supervisionada.

## 2. Quatro objetos distintos

A arquitetura correta separa quatro objetos, que nunca se confundem:

- **event_record** — registro historico/documental do evento: data, cidade, tipo
  de fenomeno, fonte, autoridade, precisao temporal.
- **footprint** — geometria observacional ou tecnica: poligono oficial, ponto
  oficial, bbox, footprint SAR, area de risco, alerta, contexto documental.
- **patch_link** — vinculo auditavel entre footprint/evento e patch: intersecao,
  bbox overlap, buffer, mesma regiao/periodo ou insuficiente.
- **score_evaluation** — comparacao review-only entre score/features do patch e a
  evidencia observacional. Nao treina modelo, nao altera o score oficial, nao cria
  score v7.

E as distincoes conceituais que a politica protege:

- **suscetibilidade** e uma propriedade estrutural do terreno; **risco** combina
  suscetibilidade com exposicao/vulnerabilidade; **evento observado** e um fato
  datado; **footprint** e geometria; **patch_link** e um vinculo auditavel;
  **reference evidence** e evidencia review-only de qualidade suficiente; e
  **ground truth supervisionado** e rotulo de treino — que este marco nao produz.

## 3. Estados de referencia observacional

| Estado | Default | Negativo | eval review-only | calib review-only | Campos exigidos |
| --- | --- | --- | --- | --- | --- |
| positive_strong | - | nao | true | true | event_date;geometry_type;source_authority;patch_link_quality;uncertainty_m;qa_status |
| positive_provisional | - | nao | false | true | event_date;source_authority;documentation_quality |
| unlabeled | padrao | nao | false | false | none |
| hard_negative_audited | - | sim | true | true | observability;exclusion_check;qa_status |
| no_data | - | nao | false | false | blocking_reason |
| rejected | - | nao | false | false | rejection_reason |

O detalhe completo esta em `susc_gt01_evidence_state_taxonomy.csv` e
`susc_gt01_ground_reference_policy.json`.

## 4. Por que o estado padrao e unlabeled

Todo patch nasce **unlabeled**. Ausencia de registro nao e negativo verdadeiro:
um patch sem evidencia documentada apenas nao tem evidencia suficiente — nao ha
prova de que ficou seco. Tratar o universo nao rotulado como negativo criaria
negativos falsos em massa. Por isso o universo sem evidencia e
`unlabeled_background`, nunca negative.

## 5. Por que hard negative e raro

**hard_negative_audited** so existe num caso raro e auditado: um patch observado
no mesmo evento, fora do footprint, sem exclusoes relevantes (nuvem, sombra,
ambiguidade), com comportamento seco e QA aceito. Exige `observability`,
`exclusion_check` e `qa_status`. Sem essa auditoria explicita, o patch permanece
unlabeled — nunca negativo por omissao.

## 6. Por que score_evaluation e review-only

A comparacao entre score/features e evidencia observacional e exploratoria e
review-only. Ela nao valida o modelo, nao vira benchmark, nao corrige o score_v6
e nao autoriza score_v7. `eligible_for_evaluation` so pode ser true para
positive_strong e hard_negative_audited (ou casos explicitamente aceitos por
politica); `eligible_for_calibration` so pode ser true em regime review-only.

## 7. Quais evidencias podem ou nao promover um patch

A tabela de decisao (`susc_gt01_review_only_decision_table.csv`) fixa o teto de
promocao por tipo de evidencia:

| Evidencia | Pode promover a | patch_link maximo | eval review-only | Decisao |
| --- | --- | --- | --- | --- |
| official_polygon | positive_strong | intersection_strong | true | promocao_forte_permitida_com_qa |
| official_point | positive_strong | buffer | false | provisorio_ou_forte_conforme_buffer_e_qa |
| official_bbox | positive_provisional | bbox_overlap | false | provisorio_sem_geometria_fina |
| sar_footprint | positive_provisional | bbox_overlap | false | tecnico_review_only_nunca_verdade_automatica |
| risk_area | unlabeled | insufficient | false | nao_promove_area_de_risco_nao_e_evento |
| alert_only | unlabeled | insufficient | false | nao_promove_alerta_nao_e_evento |
| news_report | positive_provisional | region_period | false | documental_fraco_nao_forte |
| municipality_affected | unlabeled | region_period | false | nao_promove_municipio_nao_e_patch |
| neighborhood_affected | unlabeled | region_period | false | nao_promove_bairro_nao_e_patch |
| documentary_context | positive_provisional | region_period | false | contexto_documental_review_only |
| post_event_footprint | no_data | none | false | nao_vira_feature_pre_evento |

Regras nao-negociaveis: **alerta** nao pode virar evento observado nem patch_link
forte; **area de risco** nao pode virar evento ocorrido; **noticia** nao vira
patch_link forte; **municipio/bairro atingido** nao vira patch atingido;
**footprint pos-evento** nao vira feature pre-evento; e **Sentinel-1/SAR** pode
gerar ou confirmar footprint, mas nao vira verdade automatica.

## 8. Quais bloqueios impedem treino supervisionado

Sao constantes deste marco, nunca derivadas: `eligible_for_training=false`,
`eligible_for_ground_truth=false`, `score_v7_candidate=false`, `trainable=false`,
`review_only=true`. Nenhuma linha de nenhum artefato pode habilitar treino,
ground truth supervisionado ou score_v7 oficial. As razoes estao catalogadas em
`susc_gt01_not_ground_truth_reasons.csv`.

## 9. Separacao de fenomeno em Petropolis

Petropolis exige separacao explicita entre `flood`, `flash_flood`, `landslide` e
`mixed_flood_landslide`. Enquanto o fenomeno estiver misto, nenhum patch da regiao
pode ser promovido: o caso permanece contextual/bloqueado.

## 10. Como isso prepara os proximos marcos

Esta base metodologica executavel (schemas + policy + validador + testes +
relatorio) deixa o projeto pronto para: **calibracao candidata** review-only;
**canarios SAR** observacionais; **QA humano** dos estados; a **matriz multimodal
escalavel**; e um **benchmark futuro** — que permanece nao autorizado enquanto a
amostra e a geometria oficial nao forem suficientes.

REV-P nao preve enchentes operacionalmente: produz analise estrutural review-only
com evidencia observacional auditavel.
