# REV-P — Narrativa científica consolidada

Documento-âncora para artigo, apresentação e defesa. Consolida, em português brasileiro, o problema, a metodologia, a contribuição e os limites do REV-P. Não introduz afirmações novas: reúne o que já está registrado nos artefatos públicos e nos relatórios de execução.

Última atualização: 2026-08-18.

---

## 1. Problema

Áreas urbanas brasileiras sujeitas a inundação concentram vulnerabilidade físico-ambiental difícil de caracterizar de forma reprodutível e cientificamente honesta. Modelos que aprendem padrões diretamente de imagem correm o risco de capturar correlação espúria em vez de mecanismo físico real. O REV-P parte do caminho oposto: usa relações físico-hidrológicas já conhecidas (acúmulo de água, capacidade de escoamento, proximidade a corpos hídricos, chuva antecedente) como base causal do modelo, e testa essas relações contra eventos reais de enchente registrados por fontes oficiais.

## 2. Motivação

Modelar suscetibilidade a enchente exige equilíbrio entre poder preditivo e interpretabilidade científica. Um modelo caixa-preta pode acertar sem explicar por quê; um modelo que ignora física conhecida corre o risco de "descobrir" a enchente a partir de artefato de dado, não de mecanismo real. O REV-P resolve essa tensão fixando a base causal em variáveis físico-hidrológicas interpretáveis e usando dado orbital (Sentinel) apenas como evidência auxiliar — nunca como substituto da física.

## 3. Escopo

O REV-P é um pipeline causal aplicado a três regiões brasileiras — Recife, Curitiba e Petrópolis — com uma frente externa de validação internacional (Reino Unido e áreas cobertas pelo Copernicus Emergency Management Service).

Resultados consolidados:

- **Recife**: modelo causal maduro. Firth penalizado (`v12`), n=278 eventos reais (154 positivos / 124 negativos), LOO-AUC = 0,68 (repetido 5-fold: 0,67 ± 0,01). Motor de inferência local e API de contrato entregues e auditados.
- **Curitiba**: modelo treinado (Firth, AUC embaralhado 0,65) que colapsa em holdout temporal real de 2026 (AUC 0,52). Sete diagnósticos independentes descartaram vazamento espacial, sazonalidade, ruído de amostra, deriva administrativa e correlação com El Niño/La Niña. Resultado negativo documentado, não escondido.
- **Petrópolis**: bloqueado — mistura enchente/deslizamento não separada nas fontes disponíveis.
- **Frente externa**: piloto Reino Unido (7.476 pontos, 201 eventos independentes, AUC agrupada 0,79) e multirregião Copernicus EMS (25.249 pontos, 119 áreas, transferência serra↔planície sem perda relevante de desempenho).
- **DINOv2**: 12 embeddings reais (4 por região, 768 dimensões, encoder congelado), testados como feature causal via comparação A/B contra o modelo físico de Recife e descartados — mantidos apenas como análise estrutural auxiliar (similaridade, k-NN, PCA, medoids, outliers).

Fora de escopo nesta entrega: modelo operacional para Petrópolis; generalização temporal validada para Curitiba; uso de DINOv2 como classificador ou preditor.

## 4. Contribuição do REV-P

Um pipeline causal reprodutível que modela suscetibilidade a enchente a partir de variáveis físico-hidrológicas interpretáveis, valida essas variáveis contra evento real por região, e reporta com o mesmo rigor tanto o resultado positivo (Recife) quanto o negativo (Curitiba) e o bloqueio (Petrópolis) — sem promover dado orbital ou representação aprendida a substituto de mecanismo físico conhecido. A contribuição inclui a demonstração explícita, via comparação A/B, de que uma representação auto-supervisionada (DINOv2) não substitui física conhecida quando testada com rigor.

## 5. Pipeline metodológico

1. **Aquisição de evento real** — eventos de enchente confirmados por Defesa Civil, ANA, Diário Oficial e bases internacionais (Global Flood Database), por região.
2. **Engenharia de features físico-hidrológicas** — HAND, TWI, proximidade a corpos hídricos, chuva antecedente (CHIRPS), entre outras, sempre com direção causal esperada fixada antes do ajuste.
3. **Modelagem causal** — regressão logística penalizada de Firth como rota primária interpretável; GBM monotônico causal como diagnóstico complementar de não linearidade, restrito a preservar a direção causal esperada.
4. **Validação estatística rigorosa** — LOO-CV e k-fold repetido, sempre com desvio-padrão reportado; checagem de coerência física (sinal e significância) como parte da validação, não como etapa opcional.
5. **Motor de inferência e API** — implementação e auditoria ponta a ponta do modelo de Recife como motor de inferência local e contrato de API.
6. **Frente externa de validação** — piloto Reino Unido e multirregião Copernicus EMS, com tabela de pontos harmonizada entre fontes, testando transferência do modelo entre contextos geográficos distintos.
7. **DINOv2 como evidência auxiliar** — extração de embeddings com encoder congelado, análise estrutural exploratória e teste formal A/B contra o modelo físico, com resultado documentado (descarte como feature causal).

## 6. Por que Firth, e por que GBM monotônico só como diagnóstico

A regressão logística penalizada de Firth foi escolhida por lidar bem com eventos raros (poucos positivos frente a negativos) e por manter coeficientes interpretáveis com significância estatística e sinal esperado — essencial para defender que o modelo captura mecanismo físico, não artefato de amostra. O GBM monotônico entra apenas como diagnóstico: verifica se existe não linearidade real no fenômeno sem permitir que o modelo viole a direção causal já estabelecida em cada feature. Ele nunca substitui a rota interpretável em produção.

## 7. DINOv2 e embeddings

DINOv2 com registros (`facebook/dinov2-with-registers-base`) é usado exclusivamente como **encoder visual pré-treinado e congelado**. Foram extraídos 12 embeddings reais (4 por região, 768 dimensões, com hash SHA256 registrado). O encoder não é ajustado nem retreinado. Os embeddings foram testados formalmente como feature adicional ao modelo físico de Recife (comparação A/B) e descartados por não melhorarem o modelo causal — o resultado do teste está documentado, não omitido. Os embeddings seguem no repositório apenas como análise estrutural exploratória: similaridade, vizinhança (k-NN), projeção PCA, medoids e outliers.

## 8. Curitiba: um resultado negativo documentado

O colapso do modelo de Curitiba em holdout temporal real (AUC 0,65 embaralhado → 0,52 real) foi investigado com sete diagnósticos independentes, descartando as explicações mais prováveis (vazamento espacial, sazonalidade, ruído amostral, deriva administrativa, El Niño/La Niña). Um GBM monotônico causal confirmou não linearidade real no fenômeno, mas não resolveu o problema de generalização temporal. A rota declarada continua sendo a linear/interpretável, e o resultado é reportado como achado negativo informativo — parte legítima do método científico, não uma falha a esconder.

## 9. O que pode ser afirmado

- O modelo causal de Recife é real, auditado ponta a ponta, e generaliza sob validação cruzada (LOO-AUC 0,68).
- O modelo de Curitiba não generaliza para o período temporal de 2026, e essa limitação foi investigada exaustivamente, não apenas constatada.
- Petrópolis não tem modelo nesta entrega, por ausência de separação de fenômeno nas fontes.
- A frente externa confirma que o método (Firth sobre variáveis físico-hidrológicas) transfere entre contextos geográficos distintos (Reino Unido, serra e planície cobertas por Copernicus EMS) com desempenho na faixa esperada (AUC 0,70–0,88).
- DINOv2 foi testado como feature causal com rigor e descartado — a decisão de mantê-lo fora do modelo é baseada em evidência, não em suposição.

## 10. O que não pode ser afirmado

- Que o modelo de Curitiba está pronto para uso operacional.
- Que existe modelo para Petrópolis nesta entrega.
- Que DINOv2 mede acurácia operacional de detecção de inundação ou substitui a base físico-hidrológica.
- Que a definição de negativo é neutra em relação à métrica — o achado da frente externa mostra que a definição de negativo afeta o AUC obtido mais do que o fenômeno em si.

## 11. Limitações

- Nenhuma das três regiões brasileiras tem negativo formal aceito por gate metodológico (`C4_BLOCKED_NO_FORMAL_NEGATIVES`); a ausência é documentada, não contornada por proxy.
- Corpus de embeddings DINOv2 intencionalmente pequeno (12 vetores reais) — suficiente para análise estrutural, não para validação estatística de desempenho.
- Fontes externas oficiais (COMPDEC, DRM-RJ, Defesa Civil, CPRM) têm solicitações formais pendentes de resposta.
- A frente externa Copernicus EMS multirregião roda apenas com variáveis de terreno (sem chuva), medindo suscetibilidade espacial ("onde inunda quando chove"), não deflagração do evento.

## 12. Próximos passos

- Obter geometria oficial de evento em Petrópolis (DRM-RJ) para destravar a região.
- Resolver a separação de fenômeno (enchente x deslizamento) em Petrópolis 2022 com produto oficial.
- Consolidar o resultado negativo de Curitiba como achado documentado no artigo, em vez de buscar diagnósticos adicionais.
- Ampliar a tabela de pontos harmonizada da frente externa como evidência complementar de transferência regional.
