# v1r6 -- Sensibilidade cluster-robusta ao achado DINO do v1r5

## Objetivo

v1r5 achou LRT significante (p=0.0048) para DINO no Modelo B, mas descobriu que dino_pca1/2 sao por-patch (23 unicos para 109 pontos, ate 10 pontos compartilhando o mesmo vetor). Este script reestima a incerteza dos coeficientes DINO com erro-padrao cluster-robusto (sandwich, clusterizado por patch_id, correcao de pequena amostra), que e o jeito estatisticamente correto de tratar observacoes nao-independentes -- em vez de assumir que os 109 pontos sao 109 informacoes DINO independentes.

## Resultado -- teste de Wald conjunto cluster-robusto para DINO

estatistica=3.4835, df=2, p=0.1752 (vs. LRT ingenuo de v1r5: p=0.0048). **Veredito: DINO_DOES_NOT_SURVIVE_CLUSTER_ROBUST_CHECK_CONSISTENT_WITH_PSEUDOREPLICATION_CONFOUND.**

## Correlacao descritiva por patch (n=23, so evidencia complementar)

Spearman dino_pca1 x fracao_positiva_no_patch: rho=0.0654, p=0.7669. Spearman dino_pca2 x fracao_positiva_no_patch: rho=-0.0260, p=0.9064. n=23 e pequeno demais para um teste formal robusto -- reportado so como leitura complementar, nao como prova.

## Interpretacao

O sinal de v1r5 NAO sobrevive ao controle por clustering de patch: uma vez que a nao-independencia entre pontos do mesmo patch e contabilizada corretamente, a evidencia de que DINO agrega valor incremental ao modelo fisico deixa de ser estatisticamente solida com o n atual. Isso confirma a suspeita de v1r5 -- o resultado ingenuo era, ao menos em parte, um artefato de pseudorreplicacao, nao evidencia real de conteudo visual preditivo do DINO. **Conclusao honesta para o produto**: com os dados disponiveis agora (23 patches unicos com embedding real em Recife), o teste A/B nao suporta promover DINO a feature do score -- ele continua evidencia visual auxiliar/explicavel na interface, nunca input do modelo, ate que mais patches independentes com embedding real estejam disponiveis para refazer este teste com poder estatistico adequado.

## Limitacoes explicitas

n_clusters=23 e pequeno para inferencia cluster-robusta assintotica (regra pratica usual pede G>=20-50 clusters; estamos na borda inferior) -- a correcao de pequena amostra aplicada (c=1.096) e um paliativo padrao, nao uma garantia. Nenhum label foi criado. DINO continua evidencia auxiliar independentemente do veredito deste script.
