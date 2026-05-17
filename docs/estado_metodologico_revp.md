# Estado metodológico do REV-P

## Escopo atual

O REV-P é um protocolo auditável de preparação de evidências físico-ambientais, geoespaciais e visuais sobre patches urbanos associados a suscetibilidade a inundação e alagamento. O projeto está em estágio de revisão e auditoria estrutural.

O pipeline não executa classificação supervisionada, não cria rótulos binários de enchente observada, não define alvos de treinamento e não emite afirmações preditivas sobre vulnerabilidade de patches.

## Ausência de verdade de campo observacional

Não existe, no estado atual do projeto, um conjunto validado de rótulos de inundação observada para os patches inventariados. Os registros de enchentes históricas disponíveis são evidência geoespacial qualitativa para revisão humana, não ground truth para treino supervisionado. Nenhum patch recebeu rótulo binário de suscetibilidade.

## Ausência de classificação supervisionada

O projeto não utiliza, em nenhum estágio atual, classificadores supervisionados, funções de perda, otimizadores, splits de treino/validação/teste, métricas de desempenho preditivo ou qualquer protocolo de validação cruzada. Resultados de clustering e análise de vizinhança são diagnósticos estruturais, não classes aprendidas.

## DINO como encoder visual congelado

DINOv2 com registros é utilizado exclusivamente como encoder visual pré-treinado e congelado para extração de representações estruturais de patches Sentinel. O encoder não é ajustado, não é retreinado e não é avaliado como classificador. Embeddings extraídos são usados para comparação de vizinhança, análise de outliers, clustering exploratório e triagem de revisão humana.

## GIS como baseline interpretável

O índice de vulnerabilidade multicritério GIS (v1gq) é um proxy estrutural e interpretável construído sobre indicadores físico-ambientais objetivos: distância ao rio, uso do solo, densidade populacional e densidade viária. O índice não é ground truth, não é rótulo supervisionado, não é alvo de treinamento e não é evidência de desempenho preditivo. A integração com embeddings DINO é exploratória e estrutural.

## Bloqueadores ativos

- Ausência de rótulos de inundação observada validados.
- Ausência de alvos supervisionados.
- `patch_bound_validated = 0/59`.
- `preflight_ready = 0/59`.
- Porta CRS permanece bloqueada.
- Problema de nomenclatura Recife ext/bg permanece sem resolução para vinculação canônica de TIFs.
- Trilha multimodal em espera até resolução do balanço/recuperação de Recife.

## Escopo permitido do DINO

DINO pode ser utilizado apenas como encoder auto-supervisionado congelado para:

- extração de embeddings;
- recuperação de vizinhos mais próximos;
- projeção PCA;
- clustering exploratório;
- detecção de outliers;
- suporte a revisão visual e manual.

O DINO não deve ser reportado como classificador supervisionado de suscetibilidade a inundação.

## Limitações documentadas

- O corpus balanceado é intencionalmente pequeno e exploratório.
- Comparações regionais são descritivas e estruturais, não inferenciais.
- O índice GIS é parcial: 2/4 indicadores disponíveis para Recife; Curitiba e Petrópolis com cobertura insuficiente.
- A cobertura de uso do solo FBDS não alcança os patches do dino-corpus em Petrópolis.
- Execução de embeddings depende de disponibilidade local de modelo ou download explicitamente autorizado.
- Assets multimodais permanecem excluídos do caminho ativo.

## Próximos passos válidos

- Expandir corpus de embeddings Sentinel em direção aos 128 patches do manifest v1fu.
- Repetir análise de consistência estrutural em execuções locais maiores.
- Revisão humana manual de medoids, outliers e vizinhos recíprocos.
- Ampliar cobertura de uso do solo via fontes externas (MapBiomas, IBGE LULC).
- Revisitar trilha multimodal após resolução dos bloqueadores de Recife.
