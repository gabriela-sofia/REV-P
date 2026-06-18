# Estado metodológico do REV-P

Última atualização: 2026-06-18.

## Escopo atual

O REV-P é um protocolo auditável de preparação de evidências físico-ambientais, geoespaciais e visuais sobre patches urbanos associados a suscetibilidade a inundação e alagamento em três regiões brasileiras: Recife, Petrópolis e Curitiba. O projeto está em estágio de revisão e auditoria estrutural, em modo review-only.

Corpus e artefatos consolidados:

- **59 patches territoriais/contextuais** (Recife 18, Petrópolis 27, Curitiba 14);
- **128 assets Sentinel candidatos** como inventário de entrada do pipeline visual;
- **12 embeddings DINOv2 reais** (4 por região, 768 dimensões, encoder congelado).

O pipeline não executa classificação supervisionada, não cria rótulos binários de enchente observada, não define alvos de treinamento e não emite afirmações preditivas sobre vulnerabilidade de patches.

## Ausência de ground truth operacional

Não existe, no estado atual do projeto, um conjunto validado de rótulos de inundação observada para os patches inventariados. Os registros de enchentes históricas disponíveis são evidência geoespacial qualitativa para revisão supervisora, não ground truth para treino supervisionado. Nenhum patch recebeu rótulo binário de suscetibilidade.

- `ground_truth_operational_status = ABSENT`;
- rótulos binários formais: ausentes;
- negativos formais: ausentes (ausência de registro e pseudo-ausência não constituem negativo);
- `training_ready = false`.

## Ausência de classificação supervisionada

O projeto não utiliza, em nenhum estágio atual, classificadores supervisionados, funções de perda, otimizadores, splits de treino/validação/teste, métricas de desempenho preditivo ou qualquer protocolo de validação cruzada. Resultados de clustering e análise de vizinhança são diagnósticos estruturais, não classes aprendidas.

## DINOv2 como encoder visual congelado

DINOv2 com registros (`facebook/dinov2-with-registers-base`) é utilizado exclusivamente como encoder visual pré-treinado e congelado para extração de representações estruturais de patches Sentinel. O encoder não é ajustado, não é retreinado e não é avaliado como classificador. Os 12 embeddings extraídos (768 dimensões, com SHA256 registrado) são usados apenas para comparação de vizinhança, análise de outliers, projeção PCA, clustering exploratório e triagem de revisão supervisora.

O DINOv2 não é classificador, não mede acurácia operacional de detecção e não valida evento observado.

## Protocolo C como cadeia de evidência

O Protocolo C organiza evidências externas candidatas por região (fontes oficiais, meteorológicas, cartográficas) e separa explicitamente os tipos de referência: evidência contextual, referência temporal, referência candidata e ground truth operacional. Ele é uma **cadeia de evidência para revisão**, não uma validação operacional.

Estado por região (referências validadas pelo protocolo, não ground truth):

- **Recife** — referência candidata validada (pontuação 0.76);
- **Curitiba** — referência temporal validada (pontuação 0.70);
- **Petrópolis** — referência contextual validada (pontuação 0.55).

Gate metodológico: `can_create_training_label = BLOCKED` e `can_train_supervised_model = BLOCKED`. A conclusão científica permanece em `C4_BLOCKED_NO_FORMAL_NEGATIVES`. O Protocolo C **não fecha ground truth**.

## Auditoria de continuidade da base original (v2dz–v2ff)

Uma base de trabalho candidata anterior (códigos internos `v2dz`–`v2ef`) não estava disponível localmente. As cadeias `v2es`–`v2ey` (recuperação controlada) e `v2ez`–`v2ff` (auditoria forense de recuperabilidade) documentam a rastreabilidade dessa perda.

Estado consolidado:

- `original_base_status = ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE`;
- 53 registros candidatos da base original: não recuperáveis automaticamente (`original_53_recoverable = false`);
- fallback de 38 linhas: indisponível (`fallback_38_available = false`);
- somente referências/pistas recuperáveis foram encontradas — referência não é conteúdo.

Esta é uma auditoria de continuidade e rastreabilidade. Ela **não recupera ground truth operacional, não fecha rótulo, não libera treino e não substitui a base original por fallback**. A recuperação efetiva, se ocorrer, depende de ação manual e revisão humana.

## Bloqueadores ativos

- Ausência de rótulos de inundação observada validados.
- Ausência de alvos supervisionados e de negativos formais.
- Geometria de evento observado ausente em Curitiba e Petrópolis: a sobreposição patch-evento não foi executada por falta de geometria oficial.
- Separação de fenômeno pendente em Petrópolis 2022 (inundação e deslizamento coexistem nas fontes).
- Porta CRS permanece bloqueada para vinculação canônica de geometria.
- Base original `v2dz`–`v2ef` requer restauração manual.
- Trilha multimodal em espera até resolução dos bloqueadores de Recife.

Bloqueado não significa erro: cada bloqueio é uma pré-condição metodológica documentada, rastreável e auditável.

## Escopo permitido do DINOv2

O DINOv2 pode ser utilizado apenas como encoder auto-supervisionado congelado para: extração de embeddings, recuperação de vizinhos mais próximos, projeção PCA, clustering exploratório, detecção de outliers e suporte a revisão visual e manual. Não deve ser reportado como classificador supervisionado de suscetibilidade a inundação.

## Limitações documentadas

- O corpus de embeddings é intencionalmente pequeno e exploratório: 12 vetores reais são suficientes para análise estrutural, não para validação estatística de desempenho.
- Comparações regionais são descritivas e estruturais, não inferenciais.
- O índice GIS multicritério (estágio `v1gq`) é um proxy estrutural interpretável, não ground truth nem alvo de treinamento; sua cobertura é parcial.
- A cobertura de uso do solo não alcança todos os patches do dino-corpus em algumas regiões.
- Execução de embeddings depende de disponibilidade local de modelo ou download explicitamente autorizado.
- Assets multimodais permanecem excluídos do caminho ativo.

## Próximos passos válidos

- Obter geometria oficial de evento em Recife (COMPDEC) e Petrópolis (DRM-RJ).
- Resolver separação de fenômeno em Petrópolis 2022 com produto oficial.
- Ampliar o corpus de embeddings Sentinel em direção aos 128 assets do manifesto `v1fu`.
- Executar sobreposição patch-evento assim que geometria oficial estiver disponível.
- Definir protocolo de rótulo supervisionado apenas após ground truth estabelecido em pelo menos uma região.

## Histórico de versões

Os valores abaixo são registros de estados anteriores, preservados para rastreabilidade. **Não representam o estado atual.**

- Snapshot de prontidão de uma versão anterior: `patch_bound_validated = 0/59` e `preflight_ready = 0/59` — contagens de validação de limite de patch e de prontidão de preflight de etapas iniciais.
- O índice GIS multicritério (`v1gq`) foi inicialmente reportado com 2/4 indicadores disponíveis para Recife e cobertura insuficiente em Curitiba e Petrópolis.
- Problema de nomenclatura Recife (ext/bg) registrado como pendência de vinculação canônica de TIFs em etapas anteriores.
