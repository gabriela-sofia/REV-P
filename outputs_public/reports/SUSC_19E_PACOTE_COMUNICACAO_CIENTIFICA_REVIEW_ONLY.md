# SUSC-19E - Pacote de comunicacao cientifica review-only

## Objetivo

Transformar os resultados de 18H, 19A, 19B, 19C e 19D em um pacote de comunicacao
claro para trabalho de conclusao, artigo e apresentacao, sem inflar claims. Nao cria
dado novo, score novo, benchmark, treino nem ground truth.

## Enquadramento cientifico

O REV-P e um framework multimodal auditavel de suscetibilidade urbana a enchentes,
com avaliacao observacional review-only. Nao e predicao operacional, nao e ground
truth patch-level, nao e modelo treinado, nao e benchmark supervisionado, nao e
score_v7 e nao e sistema de alerta.

## Sintese do estado atual

300 patches consolidados; cobertura fisica/espectral/chuva
completas e territorial parcial (33.3%). 7
patches observacionais review-only (Recife 5, Curitiba
2); background nao rotulado de 293 (nunca
negativo). score_v6 medio observado 0.596 contra 0.521;
0/7 no top-30 global e
3/7 no top-30 regional.

## Estado por regiao

| Regiao | Papel | Patch-links fortes | Bloqueio principal |
| --- | --- | --- | --- |
| recife | referencia_forte_review_only | 5 | amostra_de_uma_unica_regiao_e_um_unico_evento_forte |
| curitiba | segunda_regiao_tecnica_sar | 0 | ausencia_de_geometria_oficial_de_ocorrencia_18D_aguardando_resposta |
| petropolis | bloqueada_fenomeno_misto | 0 | fenomeno_misto_deslizamento_inundacao_sem_separacao_e_sem_geometria_forte |

## Cobertura multimodal

| Familia | Cobertura | Status | Principal lacuna |
| --- | --- | --- | --- |
| fisica_topografica | 100.0% | completa | nenhuma |
| espectral_umidade | 100.0% | completa | nenhuma |
| chuva_hidrometeorologica | 100.0% | completa | nenhuma |
| territorial | 33.3% | parcial | MapBiomas_class;exposed_soil_prop;water_prop;impervious_proxy (missingness herdado do 19B) |
| documental | 2.0% | rara | documental so em Recife e contexto misto em Petropolis |
| observacional | 2.7% | rara | evidencia so em Recife e SAR de Curitiba |

## Claims permitidos

- O REV-P estima suscetibilidade urbana a enchentes por patch (score_v6 candidato). (`permitido`)
- O framework integra features fisicas, hidrologicas, espectrais, territoriais, de chuva e evidencia documental. (`permitido`)
- A avaliacao observacional e review-only, sem treino e sem benchmark. (`permitido`)
- O diagnostico 19D identifica divergencias do score_v6 sem alterar o score. (`permitido`)
- O score_v7 permanece bloqueado ate ampliar amostra, preencher territorial e ter benchmark. (`permitido`)
- Areas com evidencia observacional review-only tendem a ter score_v6 maior que o universo nao rotulado. (`permitido_com_ressalva`)
- O sinal urbano e topografico e coerente com maior suscetibilidade nos observados. (`permitido_com_ressalva`)

## Claims proibidos

- O REV-P preve enchentes operacionalmente. (`proibido`)
- O REV-P possui ground truth patch-level. (`proibido`)
- O REV-P treina um modelo supervisionado. (`proibido`)
- O REV-P validou estatisticamente com amostra suficiente. (`proibido`)
- O REV-P tem um benchmark 17B. (`proibido`)
- A referencia SAR de Curitiba e geometria oficial de ocorrencia. (`proibido`)
- O universo sem evidencia documentada e um conjunto negativo. (`proibido`)

## Plano de figuras

| Figura | Titulo | Precisa criar | Prioridade |
| --- | --- | --- | --- |
| FIG_01 | Pipeline geral do REV-P | true | alta |
| FIG_02 | Matriz multimodal por patch | true | alta |
| FIG_03 | Separacao event_record/footprint/patch_link/score_evaluation | true | alta |
| FIG_04 | Mapa conceitual por regiao | false | media |
| FIG_05 | Barras de cobertura multimodal | true | alta |
| FIG_06 | Ranking score_v6 observacional | true | alta |
| FIG_07 | Contraste de features observacional | true | alta |
| FIG_08 | Bloqueios do score_v7 | false | media |

## Roteiro de slides

15 slides em `roteiro_slides_rev_p_19e.md`, do problema a conclusao, sempre com o
enquadramento review-only, incluindo limitacoes (slide 13) e proximos passos
(slide 14).

## Riscos de comunicacao

Oito riscos em `matriz_riscos_comunicacao_19e.csv` (overclaim de previsao, ground
truth falso, background chamado de negativo, SAR como geometria oficial, score_v7
implicito, estatistica forte com n pequeno, Petropolis promovido, missingness
omitido), cada um com correcao e regra de validacao.

## Por que nao ha ground truth, treino, benchmark nem score_v7

Nao ha verdade de campo por patch; nao ha treino supervisionado; nao ha benchmark
17B; o score_v7 permanece bloqueado por amostra, missingness territorial e ausencia
de benchmark. O score_v6 permanece intacto.

## Proximos passos

Ampliar amostra observacional com geometria oficial, preencher a cobertura
territorial (pacote MapBiomas/GEE), separar o fenomeno de Petropolis e consolidar a
geometria de Curitiba. Estado da comunicacao: `19E_COMUNICACAO_PRONTA_COM_RESSALVAS`;
estado de artigo/slides: `ARTIGO_SLIDES_PRONTOS_COM_RESSALVAS`.
