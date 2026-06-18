# REV-P — Narrativa científica consolidada

Documento-âncora para artigo, apresentação e defesa. Consolida, em português brasileiro, o problema, a metodologia, a contribuição e os limites do REV-P. Não introduz afirmações novas: reúne o que já está registrado nos artefatos públicos e nos relatórios de execução.

Última atualização: 2026-06-18.

---

## 1. Problema

Áreas urbanas brasileiras sujeitas a inundação e alagamento concentram suscetibilidade físico-ambiental que é difícil de caracterizar de forma reprodutível. Faltam, com frequência, rótulos observacionais confiáveis em nível de recorte territorial (patch): os registros oficiais de eventos existem, mas raramente vêm com geometria precisa, data consistente e licença que permita reuso como referência de treino. Sem esse ground truth, qualquer tentativa de classificação supervisionada de suscetibilidade parte de premissas frágeis.

## 2. Motivação

Em vez de forçar um rótulo que não existe, o REV-P trata o problema como uma questão de **preparação e auditoria de evidência**: organizar corpus territorial, representações visuais e evidência externa de modo rastreável, separando com clareza o que é contexto, o que é referência candidata e o que seria ground truth operacional. Isso permite avançar de forma honesta — documentando bloqueios em vez de mascará-los — e deixa o terreno preparado para, no futuro, estabelecer ground truth com método.

## 3. Escopo

O REV-P é um pipeline **auditável, Sentinel-first e review-only** sobre três regiões: Recife, Petrópolis e Curitiba.

Corpus e artefatos consolidados:

- 59 patches territoriais/contextuais (Recife 18, Petrópolis 27, Curitiba 14);
- 128 assets Sentinel candidatos como inventário de entrada;
- 12 embeddings DINOv2 reais (4 por região, 768 dimensões, encoder congelado).

Fora de escopo nesta entrega: detector operacional, preditor operacional, classificador supervisionado operacional, ground truth operacional patch-level fechado, rótulo binário final, negativo formal e liberação de treino supervisionado.

## 4. Contribuição do REV-P

Um protocolo reprodutível que separa, com rastreabilidade auditável, **evidência contextual** de **referência candidata** e de **ground truth operacional** — e documenta explicitamente o que ainda não pode ser afirmado. A contribuição não é um modelo: é a disciplina metodológica (corpus, embeddings estruturais, Protocolo C, gates, manifests, travas) que mantém cada afirmação dentro do que a evidência local sustenta.

## 5. Pipeline metodológico

1. **Corpus territorial** — definição dos 59 patches por região com linhagem documentada.
2. **Pipeline Sentinel-first** — inventário dos 128 assets candidatos, preflight e QA de entrada.
3. **Embeddings DINOv2** — extração com encoder congelado e análise estrutural (similaridade, k-NN, PCA, medoids, outliers).
4. **Protocolo C** — organização de evidência externa por região e separação de tipos de referência, com adjudicação de gates.
5. **Busca por ground truth** — tentativa de geometria oficial de evento e sobreposição patch-evento, documentada como busca e bloqueio.
6. **Auditoria de continuidade** — rastreabilidade e recuperabilidade da base de trabalho candidata anterior (`v2dz`–`v2ef`).
7. **Auditoria e entrega** — figuras, tabelas, métricas, relatórios e manifests para revisão.

O mapeamento legível de cada código interno está em [`revp_indice_etapas.md`](revp_indice_etapas.md).

## 6. Protocolo C

O Protocolo C é uma **cadeia de evidência externa para revisão**, não uma validação operacional. Ele coleta e adjudica fontes oficiais, meteorológicas e cartográficas, classificando-as em camadas:

- **evidência contextual** — sustenta contexto, não confirma evento no patch;
- **referência temporal** — ancora o evento no tempo;
- **referência candidata** — referência mais forte, ainda sujeita a revisão humana;
- **ground truth operacional** — não estabelecido em nenhuma região.

Estado por região (referências validadas pelo protocolo, não ground truth): Recife — referência candidata (0.76); Curitiba — referência temporal (0.70); Petrópolis — referência contextual (0.55).

Gates: a progressão é controlada por portas C1–C4. A conclusão atual é `C4_BLOCKED_NO_FORMAL_NEGATIVES` — não há negativos formais (ausência de registro e pseudo-ausência não constituem negativo), portanto `can_create_training_label` e `can_train_supervised_model` permanecem bloqueados. O Protocolo C **não fecha ground truth** e **não valida evento em nível de patch**.

## 7. DINOv2 e embeddings

DINOv2 com registros (`facebook/dinov2-with-registers-base`) é usado exclusivamente como **encoder visual pré-treinado e congelado**. Foram extraídos 12 embeddings reais (4 por região, 768 dimensões, com SHA256 registrado). O encoder não é ajustado nem retreinado.

Os embeddings servem apenas a análise estrutural exploratória: similaridade, vizinhança (k-NN), projeção PCA, medoids, outliers e triagem de revisão. O DINOv2 **não é classificador**, **não mede acurácia operacional de detecção** e **não valida inundação observada**. A diferença entre representação auto-supervisionada e classificador supervisionado é deliberada e central ao projeto.

## 8. Busca por ground truth e bloqueios

A busca por referência observacional foi conduzida e documentada como tal — incluindo solicitações formais a instituições e tentativas de obter geometria oficial de evento. Os bloqueios ativos:

- geometria de evento observado ausente em Curitiba e Petrópolis (sobreposição patch-evento não executada);
- separação de fenômeno pendente em Petrópolis 2022 (inundação e deslizamento coexistem nas fontes);
- porta CRS bloqueada para vinculação canônica de geometria;
- fontes externas oficiais com solicitações pendentes de resposta.

Sobre a base de trabalho candidata anterior (`v2dz`–`v2ef`): a base de 53 registros **não foi recuperada diretamente**. A auditoria forense `v2ez`–`v2ff` registrou `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE`, com `original_53_recoverable = false` e `fallback_38_available = false`. Referência textual recuperável **não equivale** a conteúdo recuperado; fallback **não substitui** base original; recuperação de arquivo **não equivale** a ground truth.

Bloqueado não significa erro: cada bloqueio é uma pré-condição metodológica documentada, rastreável e auditável.

## 9. O que pode ser afirmado

- O corpus, os 128 assets candidatos e os 12 embeddings existem, são reprodutíveis localmente e estão inventariados com hash.
- A análise estrutural dos embeddings (similaridade, vizinhança, PCA, medoids, outliers) é um diagnóstico exploratório válido.
- O Protocolo C organiza evidência externa real e qualifica referências candidatas/temporais/contextuais por região.
- A ausência de ground truth, de rótulo e de negativo formal está documentada e é auditável.

## 10. O que não pode ser afirmado

- Que existe ground truth operacional patch-level em qualquer região (`ground_truth_operational_status = ABSENT`).
- Que o DINOv2 detecta ou prediz inundação, ou que mede acurácia operacional.
- Que evidência contextual confirma evento observado no patch.
- Que a base original foi recuperada, ou que um fallback a substitui.
- Que o pipeline está pronto para treino supervisionado (`training_ready = false`).

## 11. Limitações

- Corpus de embeddings intencionalmente pequeno: 12 vetores reais bastam para análise estrutural, não para validação estatística de desempenho.
- Comparações regionais são descritivas e estruturais, não inferenciais.
- O índice GIS multicritério (`v1gq`) é proxy estrutural interpretável, não ground truth nem alvo; cobertura parcial.
- Execução de embeddings depende de disponibilidade local de modelo ou download explicitamente autorizado.
- Assets multimodais permanecem fora do caminho ativo.

## 12. Próximos passos

- Obter geometria oficial de evento em Recife (COMPDEC) e Petrópolis (DRM-RJ).
- Resolver a separação de fenômeno em Petrópolis 2022 com produto oficial.
- Ampliar o corpus de embeddings Sentinel em direção aos 128 assets do manifesto `v1fu`.
- Executar a sobreposição patch-evento assim que houver geometria oficial.
- Definir protocolo de rótulo supervisionado apenas após ground truth estabelecido em pelo menos uma região.
