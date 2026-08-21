# Blocos para artigo - Metodologia (review-only)

## Unidade de analise por patch
A unidade de analise e o patch, um recorte espacial fixo. Sao 300
patches (100 por regiao em Recife, Curitiba e Petropolis). O patch e unidade de
feature_evidence e de score_evaluation; nunca e ocorrencia confirmada.

## Matriz multimodal
Cada patch recebe seis familias de features: fisica_topografica, hidrologica,
urbana/territorial, espectral_umidade, chuva_hidrometeorologica e documental. A
cobertura fisica, espectral e de chuva e completa; a territorial e parcial por
missingness de MapBiomas/solo exposto/agua/impermeabilizacao. Cobertura nao e score
de suscetibilidade.

## score_v6
O score_v6 candidato e robust_minmax de uma combinacao linear ponderada de cinco
componentes (topografia_hidrologia 0.4, chuva 0.25, urbano_espectral 0.2, mitigacao
por vegetacao -0.1, evidencia documental 0.05), conforme o contrato 17c35. E um
indice de suscetibilidade, nao uma previsao de evento.

## Evidencia observacional
A evidencia observacional e review-only: canarios documentais em Recife e overlays
tecnicos SAR em Curitiba. Separa-se explicitamente event_record, source_footprint,
derived_patch_link, feature_evidence e score_evaluation. O footprint SAR e
pos-evento e nao e feature pre-evento nem geometria oficial.

## Avaliacao review-only
Os patches observacionais sao comparados ao universo nao rotulado
(unlabeled_background) no score_v6 e nas features. O background nunca e negativo:
ausencia de evidencia documentada nao e evidencia de ausencia. As metricas de
hit-rate e enrichment sao exploratorias review-only e nunca sao benchmark.

## Controle de vazamento temporal
As features pre-evento nao incorporam o footprint pos-evento. A separacao entre
feature_evidence (pre-evento) e source_footprint (pos-evento) e mantida para evitar
vazamento temporal.

## Limitacoes
Amostra observacional minima (7 patches); missingness territorial;
ausencia de benchmark; ausencia de geometria oficial em Curitiba; fenomeno misto em
Petropolis. Com 7 patches nao ha conclusao estatistica forte.

## Proximos passos
Ampliar a amostra observacional com geometria oficial, preencher a cobertura
territorial (pacote MapBiomas/GEE), separar o fenomeno de Petropolis e consolidar a
geometria de Curitiba. Nenhum passo cria score_v7, benchmark, treino ou ground truth
agora.
