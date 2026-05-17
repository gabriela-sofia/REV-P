# Sumário de evidências científicas DINO Sentinel-first

## Escopo

Este documento resume a trilha DINO Sentinel-first de revisão apenas do REV-P. Registra o que foi tecnicamente demonstrado, quais produtos de evidência local foram gerados e quais afirmações científicas permanecem explicitamente proibidas.

A trilha DINO utiliza um encoder auto-supervisionado congelado para inspeção estrutural de patches Sentinel. Não cria rótulos, alvos, classificadores supervisionados, verdade de inundação, classes de suscetibilidade nem afirmações de desempenho preditivo.

## O que foi tecnicamente demonstrado

- Um manifest de entrada DINO Sentinel-first foi criado a partir dos manifests consolidados do repositório.
- O preflight local de assets confirmou quais referências Sentinel podem ser resolvidas no workspace privado.
- O DINOv2 com registros foi carregado como encoder congelado na execução local.
- Pixels Sentinel reais foram lidos apenas nos estágios de execução de embedding ou robustez explícitos.
- Os embeddings locais foram gerados apenas em `local_runs/`.
- Um corpus regional balanceado foi produzido para Curitiba, Petrópolis e Recife.
- Uma execução expandida do corpus local produziu 12 embeddings, 4 por região.
- Diagnósticos estruturais foram gerados para vizinhos, clusters, medoids, outliers, pontes de grafo, robustez a perturbações, estabilidade longitudinal, proveniência e triagem de revisão humana.

## Evidências locais produzidas

A evidência de execução local está armazenada em `local_runs/dino_embeddings/` e intencionalmente não é versionada:

- v1fw: scaffold de extração dry-run e esquema de saída.
- v1fx: execução smoke de embedding.
- v1fy: análise exploratória do corpus de embeddings.
- v1fz: corpus regional balanceado e análise estrutural.
- v1ga: análise de consistência estrutural.
- v1gb: revisão visual estrutural local.
- v1gc: diagnósticos geo-estruturais.
- v1gd: diagnósticos de robustez a perturbações.
- v1ge: corpus expandido de embeddings Sentinel.
- v1gf: índice de evidência estrutural.
- v1gg: pacote de revisão humana.
- v1gh: diagnósticos estruturais longitudinais.
- v1gi: rastreador de proveniência estrutural.
- v1gj: auditoria de prontidão multimodal com execução multimodal desabilitada.
- v1gk: auditoria de reprodutibilidade.

## O que não foi reivindicado

A trilha DINO não reivindica:

- rótulos de inundação observada;
- alvos binários;
- rótulos automáticos;
- classes de suscetibilidade;
- interpretação de cluster para classe;
- desempenho preditivo;
- acurácia de modelo supervisionado;
- inferência de vulnerabilidade;
- inferência de ocorrência de inundação;
- prontidão de fusão multimodal além da auditoria de bloqueadores.

`review_priority` é um campo de triagem de revisão humana apenas. Não é um rótulo científico, não é um alvo e não é um proxy para status de vulnerabilidade ou inundação.

## Status dos embeddings

Os embeddings reais foram gerados localmente usando o DINOv2 com registros como encoder congelado. O corpus local expandido contém 12 embeddings Sentinel com dimensão de embedding 768:

- Curitiba: 4
- Petrópolis: 4
- Recife: 4

Os arrays de embeddings permanecem apenas locais em `local_runs/` e não são destinados ao Git.

## Status da análise estrutural

A camada estrutural atual suporta:

- análise de vizinho mais próximo;
- verificações de pares recíprocos;
- coordenadas PCA/manifold;
- diagnósticos leves de clustering;
- revisão de medoids e casos extremos;
- diagnósticos de outliers estruturais;
- diagnósticos de grafo geo-estruturais;
- candidatos de ponte inter-região;
- verificações de robustez a perturbações;
- persistência longitudinal de diagnósticos;
- rastreabilidade de proveniência e revisão.

Todos os outputs são diagnósticos apenas de revisão. Apoiam inspeção manual e auditoria de método, não classificação.

## Status de robustez

O v1gd testou perturbações controladas para auditoria de sensibilidade local:

- ruído gaussiano leve;
- escalonamento de brilho;
- escalonamento de contraste;
- desfoque leve;
- jitter de recorte;
- dropout de banda controlado.

Essas perturbações não são aumentos de dados de treinamento. Não são usadas para treinar ou validar um modelo supervisionado.

## Status da revisão humana

O v1gg gerou um pacote local de revisão humana a partir dos diagnósticos estruturais. Inclui itens de revisão para candidatos de ponte, outliers, medoids, embeddings robustos ou instáveis e exemplos regionais.

As notas humanas permanecem vazias até a revisão manual. O pacote não copia rasters brutos e não versiona outputs visuais.

## Status multimodal

O multimodal permanece em espera.

- `multimodal_execution_enabled=false`
- `multimodal_training_enabled=false`
- `multimodal_hold=true`

O v1gj é apenas uma auditoria de prontidão. Prontidão não é execução, não é geração de stack, não é fusão e não é treinamento.

## Limitações atuais

- O corpus expandido ainda é um subconjunto local de auditoria, não a execução completa de 128 patches.
- Os embeddings DINO dependem da disponibilidade local do modelo e do acesso local a assets.
- Os diagnósticos de grafo estrutural e perturbação são sensíveis ao tamanho do corpus e às configurações de top-k.
- As comparações regionais são apenas descritivas e estruturais.
- A fusão multimodal permanece bloqueada por vínculos, geometria, CRS e questões de recuperação/prontidão.

## Próximos passos válidos

- Fazer commit dos scripts versionados e da documentação após a auditoria final.
- Executar o corpus completo de embeddings Sentinel de 128 patches localmente quando o tempo de computação for aceitável.
- Regenerar v1gf–v1gk a partir do corpus maior.
- Conduzir revisão manual dos itens do v1gg antes de escrever a interpretação científica.
- Manter o multimodal em espera até que os bloqueadores sejam eliminados por evidência explícita.
