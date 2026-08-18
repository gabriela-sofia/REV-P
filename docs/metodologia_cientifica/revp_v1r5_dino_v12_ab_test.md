# v1r5 -- Teste A/B fisico vs fisico+DINO no v12 (n=278, dado mais forte)

## Objetivo

Executa a Fase 1 do PLANO_ACAO_produto_v1.md: decide se DINO agrega valor incremental ao modelo fisico usando o dataset mais forte disponivel (`dataset_v12_final.csv`, n=278, 154 pos/124 neg, LOO-AUC=0.6781 documentado), em vez do recorte antigo de 163 pontos usado em v1r4.

## Join espacial

Dos 278 pontos do v12, 109 caem dentro de um patch Sentinel com embedding DINO real (58 positivos, 51 negativos) -- restricao imposta pela cobertura de patches com embedding (23 patches: 10 positivos + 13 negativos-evidencia), nao pelo tamanho do v12 em si.

## Orcamento de EPV

Com minority=51 no subconjunto joined, o orcamento de EPV (>=10 na classe minoritaria) permitiu: Modelo A = rain_decay_index_api_chirps, twi_dinf, slope_deg (EPV=17.00); Modelo B = Modelo A + 2 componente(s) PCA do DINO (EPV=10.20). As 6 features fisicas completas NAO couberam no orcamento (EPV cairia abaixo de 10); reduzido para as mais fortes por evidencia real do proprio v12 (menor p-valor no Firth multivariado de n=278: rain_decay_index_api_chirps, twi_dinf, ...), nao por escolha arbitraria.

## Resultado -- razao de verossimilhanca (nao DeLong, nao delta-AUC bruto)

LRT: estatistica=10.6734, df=2, p=0.0048. Decisao: **DINO_LRT_SIGNIFICANT_BUT_CONFOUNDED_BY_PATCH_LEVEL_PSEUDOREPLICATION**. LOO-AUC descritivo: Modelo A=0.6734, Modelo B=0.6927, delta=+0.0193 (reportado apenas como evidencia complementar, nunca como criterio de decisao -- ver revalidacao cientifica anterior, secao 3.2).

## Limitacao critica: DINO e por patch, nao por ponto (pseudorreplicacao)

O embedding DINO (e, portanto, `dino_pca1`/`dino_pca2`) e calculado UMA VEZ POR PATCH Sentinel, nao uma vez por ponto. Os 109 pontos joined caem em apenas 23 patches unicos -- ate 10 pontos compartilham exatamente o mesmo vetor DINO. Isso e pseudorreplicacao: o tamanho amostral efetivamente independente para as dimensoes DINO e muito mais proximo de 23 do que de 109. As features fisicas NAO tem esse problema (sao calculadas por ponto, a partir de terreno/chuva reais na coordenada exata). Como ha patches com pontos de ambas as classes (positivo e negativo) e patches puros de uma so classe, parte do sinal que a LRT capta pode refletir qual PATCH um ponto caiu, nao uma caracteristica visual real de risco de enchente -- por isso a decisao acima foi marcada como CONFOUNDED, nao como um achado limpo. Uma reanalise por patch (1 linha por patch, nao por ponto) e o proximo passo real antes de aceitar este resultado como definitivo.

## Limitacoes explicitas

n=109 restrito a Recife e aos patches com embedding DINO real (nao e uma amostra aleatoria do v12); nenhum label foi criado; nenhum treino supervisionado novo; DINO permanece evidencia auxiliar independentemente do resultado -- a decisao de produto (DINO como feature vs. evidencia visual explicavel) e tomada a parte deste script, com base neste resultado mais o restante do plano de acao.

## Implementacao Firth

`firthlogist` exige Python <3.11 e nao esta disponivel neste ambiente (Python 3.12); a regressao logistica penalizada de Firth foi reimplementada localmente (scores modificados, Heinze & Schemper 2002) e validada antes de uso: reproduz os coeficientes padronizados publicados em `primaria_v12_firth_multivariate_coefs.csv` com 4 casas decimais de precisao e a log-verossimilhanca penalizada publicada em `all_reports_v12_primary.json` (-150.5935 vs -150.59347...). Os intervalos de confianca/p-valores aqui usam aproximacao de Wald (nao perfil de verossimilhanca penalizada como o `firthlogist` de referencia), o que pode gerar pequenas diferencas no terceiro/quarto digito de p-valor para coeficientes proximos da fronteira de significancia -- nao afeta os pontos estimados nem a log-verossimilhanca usada na LRT.
