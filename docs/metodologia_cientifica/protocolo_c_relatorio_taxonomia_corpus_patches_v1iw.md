# Relatorio v1iw - Reconciliacao 59 vs 128

## Resultado

A v1iw consolidou a taxonomia publica dos corpus em `datasets/patch_corpus_taxonomy_registry.csv` e confirmou que 59 e 128 pertencem a camadas diferentes.

## Decisao metodologica

O numero 59 deve ser tratado como corpus territorial consolidado: 18 Recife, 27 Petropolis e 14 Curitiba. Esse e o numero adequado para descrever o universo territorial do estudo.

O numero 128 deve ser tratado como manifesto Sentinel candidato: 37 Recife, 48 Petropolis e 43 Curitiba. Esse e o numero adequado para descrever a camada de assets Sentinel usados para planejamento e execucao do pipeline Sentinel-first.

Nao houve ajuste numerico para fazer uma tabela "bater". A diferenca foi preservada porque e metodologicamente correta: uma camada conta unidades territoriais, a outra conta referencias Sentinel candidatas.

## Implicacao para DINO

Quando o texto falar de DINO, o numero correto e o subset realmente executado na etapa citada. O manifesto Sentinel de 128 explica disponibilidade e planejamento. O subset operacional de embeddings explica o que foi efetivamente processado localmente.

## Implicacao para ground truth

Nenhuma camada da v1iw pode ser usada como ground truth. O corpus de 59, o manifesto de 128, o subset DINO, as unidades oficiais e os anchors oficiais permanecem sem criacao de label, sem target e sem treinamento.

## Texto recomendado para o TCC

"O REV-P distingue o corpus territorial consolidado do manifesto Sentinel-first. O corpus territorial e formado por 59 patches, distribuidos em 18 no Recife, 27 em Petropolis e 14 em Curitiba. Esse numero descreve as unidades territoriais/contextuais do estudo. Em paralelo, o manifesto Sentinel-first registra 128 assets candidatos, distribuidos em 37 no Recife, 48 em Petropolis e 43 em Curitiba. Esse segundo numero descreve referencias Sentinel disponiveis para o pipeline e pode incluir mais de uma referencia associada a contextos territoriais. Portanto, 59 e 128 nao sao contagens concorrentes: representam camadas metodologicas distintas. Nenhuma delas constitui ground truth operacional ou rotulo de treinamento."
