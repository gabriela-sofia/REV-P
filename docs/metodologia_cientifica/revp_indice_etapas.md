# REV-P — Índice público de estágios

Este documento mapeia os códigos internos do pipeline REV-P para nomes públicos legíveis. Os nomes internos (códigos `v1xx`, `v2xx`) são preservados em scripts, testes e manifests por compatibilidade. Este índice serve como camada de leitura humana.

Última atualização: 2026-06-18

---

## Como usar este índice

- **Código interno**: nome usado em scripts, testes e referências técnicas — não alterar.
- **Nome público**: descrição legível para documentação, apresentações e revisão.
- **Finalidade**: o que o estágio faz em uma frase.
- **Executável**: se pode ser executado agora com os dados locais disponíveis.
- **Dependência de revisão humana**: se requer decisão ou validação humana antes de avançar.
- **Status científico**: estado atual do estágio no projeto.

---

## Marcos científicos (linha do tempo)

Esta é a leitura de alto nível para a banca. As microversões `v1*`/`v2*` são rastreabilidade técnica; o que importa são os marcos abaixo. Cada marco indica função, artefatos principais, claim permitido, claim proibido e estado.

| Marco | Função | Artefatos principais | Claim permitido | Claim proibido | Estado |
|---|---|---|---|---|---|
| `v1f`–`v1g` | Corpus territorial, manifesto Sentinel e scaffold DINO/multimodal | Manifesto `v1fu` (128 assets), corpus de 59 patches | "inventário de entrada reprodutível" | "classe / rótulo / alvo" | Concluído |
| `v1g` (DINOv2) | Extração de embeddings reais com encoder congelado | 12 embeddings 768D, tabelas de similaridade/k-NN/PCA, 5 figuras | "representação auto-supervisionada exploratória" | "detector / preditor / acurácia operacional" | Concluído |
| `v1i`–`v2a` | Protocolo C — aquisição e adjudicação de evidência externa | Registries, scorecards, gates C1–C4 | "evidência candidata para revisão" | "ground truth / evento confirmado" | Bloqueado em `C4_BLOCKED_NO_FORMAL_NEGATIVES` |
| `v2ah`–`v2am` | Atlas de evidência e integração para manuscrito | Atlas, apêndices, bundles LaTeX/Markdown | "empacotamento para revisão humana" | "resultado operacional" | Concluído |
| `v2an`–`v2bm` | Validação regional de referência | Pontuações Recife 0.76 / Curitiba 0.70 / Petrópolis 0.55 | "referência candidata/temporal/contextual" | "ground truth / label" | Concluído (referências, não GT) |
| `v2bn`–`v2ca` | Dry-run de ground truth sem criação de label | Planos de protocolo, scaffolds, adjudicações | "modelagem de protocolo hipotético" | "label criado / treino liberado" | Bloqueado (poucos positivos / sem geometria) |
| `v2cb`–`v2ch` | Cadeia operacional Curitiba | Intake de evidência, QA geométrico, release | "aquisição auditável bloqueada" | "geometria de evento confirmada" | Bloqueado auditável |
| `v2ci`–`v2cj` | Auditoria e normalização terminológica PT-BR | Mapa de termos, renomeações seguras | "padronização de linguagem" | — | Concluído |
| `v2dz`–`v2ef` | Base de trabalho candidata anterior | Base de 53 registros (referenciada por etapas posteriores) | "base de processo intermediária" | "ground truth / base recuperada" | Artefatos não recuperados diretamente |
| `v2es`–`v2ey` | Tentativa de recuperação controlada | Inspeção de artefatos vizinhos, validação de candidatos | "auditoria de continuidade" | "recuperação concluída / fallback como base" | Bloqueado — sem fonte válida nem fallback |
| `v2ez`–`v2ff` | Busca forense, validação de candidatos e decisão de recuperabilidade | Índice forense, extrator diff/patch, inspetor Git/reflog, painel de perda | "auditoria forense somente para revisão" | "ground truth / label / treino / detecção / predição" | Concluído — `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE` |
| `curadoria/repositorio-publico-ptbr` | Reorganização pública em português | README, índices, narrativa consolidada, relatórios | "consolidação editorial em PT-BR" | — | Em andamento |

As seções abaixo detalham os códigos internos de cada marco.

---

## Estágios de corpus e linhagem de patches

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v1fu` | Manifesto de entrada DINOv2 | Inventaria assets Sentinel candidatos para extração de embeddings | Sim | Não | Concluído |
| `v1fv` | Preflight de assets locais | Verifica condições locais antes da extração de embeddings | Sim | Não | Concluído |
| `v1fw` | Contrato de extração de embeddings | Define estrutura esperada da extração antes de executar | Sim | Não | Concluído |
| `v1fx` | Extração experimental de embeddings | Executa extração de 1–4 embeddings por região como ensaio | Local apenas | Não | Concluído |
| `v1fz` | Corpus balanceado de embeddings | Verifica equilíbrio do corpus entre regiões | Sim | Não | Concluído |

---

## Estágios de análise estrutural DINOv2

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v1ge`–`v1gg` | Trilha de revisão DINOv2 | Gera pacote de revisão estrutural (similaridade, k-NN, PCA) | Local apenas | Não | Concluído |
| `v1gh`–`v1gj` | Proveniência longitudinal e prontidão | Audita rastreabilidade dos embeddings e estado de prontidão | Sim | Não | Concluído |
| `v1gk` | Fechamento de reprodutibilidade DINOv2 | Verifica que os resultados são reprodutíveis localmente | Local apenas | Não | Concluído |
| `v1gp` | Auditoria de prontidão para publicação | Verifica que os artefatos públicos estão prontos para entrega | Sim | Não | Concluído |

---

## Estágios de análise GIS e multicritério

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v1gq` | Linha de base multicritério de vulnerabilidade GIS | Cria matriz de vulnerabilidade física com dados GIS | Sim | Não | Concluído |
| `v1gr` | Auditoria de prontidão de uso do solo | Verifica disponibilidade e prontidão de dados de uso do solo | Sim | Não | Concluído |
| `v1gs` | Habilitação de geometria de uso do solo | Ativa uso de geometrias de cobertura do solo | Sim | Não | Concluído |
| `v1gt` | Expansão de cobertura GIS | Estende análise GIS para áreas de cobertura adicional | Sim | Não | Concluído |

---

## Estágios do Protocolo C — aquisição de evidência

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v1if` | Aquisição de vetores oficiais observados | Busca e audita vetores oficiais de eventos em repositórios públicos | Sim | Sim | Bloqueado — 0 vetores encontrados |
| `v1ih` | Descoberta de vetores em bases abertas | Valida candidatos de bases geoespaciais abertas | Sim | Sim | Bloqueado — 0 confirmados |
| `v1ii` | Mineração dirigida em repositórios oficiais | Audita 6 repositórios oficiais com APIs para vetores de evento | Sim | Sim | Bloqueado — 0 confirmados |
| `v1hq` | Referências observacionais candidatas | Organiza 9 eventos candidatos com G1–G3 documentados | Sim | Sim | Em andamento |
| `v1hr` | Preflight de ligação evento-patch | Prepara estrutura para patch-linking sem executá-lo | Sim | Sim | Aguardando geometria |
| `v1ib` | Consolidação de referências observacionais | Classifica eventos por nível de evidência acumulada (L0–L6) | Sim | Sim | Petrópolis 2022 em L5, Petrópolis 2024 em L6 |
| `v1ic` | Separação de fenômeno por localidade | Classifica fenômeno (inundação vs. deslizamento) por localidade | Sim | Sim | Bloqueado em Petrópolis 2022 |

---

## Estágios do Protocolo C — geometria e geocodificação

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v2at` | Vinculação evento-patch Recife | Registra vínculos entre eventos observados e patches Recife | Sim | Sim | Concluído (metadata) |
| `v2au` | Sobreposição de geometria de evento | Prepara sobreposição espacial evento-patch | Sim | Sim | Bloqueado por geometria ausente |
| `v2av` | Construção de limites de patch | Gera limites geométricos dos patches | Sim | Não | Concluído |
| `v2aw` | Entrada de fonte geométrica | Organiza intake de fontes geométricas externas | Sim | Sim | Concluído |
| `v2ax` | Ingestão de geometria Recife | Processa geometrias de eventos em Recife | Sim | Sim | Concluído (metadata) |
| `v2ay` | Reconciliação de escopo do evento candidato | Ajusta escopo e limites dos eventos candidatos | Sim | Sim | Concluído |
| `v2az` | Orquestrador de replay de ponto de virada | Coordena replay de eventos a partir de ponto de virada | Sim | Sim | Concluído |
| `v2ba` | Bancada mínima de aquisição real | Executa aquisição mínima real de geometria | Sim | Sim | Concluído |
| `v2bb` | Construtor de feed de recuperação geométrica pública | Monta feed de fontes geométricas públicas | Sim | Não | Concluído |
| `v2bc` | Bancada de digitalização GIS Recife | Suporte à digitalização manual em Recife | Manual | Sim | Em andamento |
| `v2bd` | Recuperação de footprint de patch Sentinel | Recupera limites espaciais dos patches Sentinel | Sim | Não | Concluído |
| `v2be` | Integração de limites de patch TP1 | Integra limites de patch da etapa TP1 ao corpus | Sim | Não | Concluído |
| `v2bf` | Polígono de evento observado Recife TP2 | Organiza candidatos de polígono de evento para TP2 em Recife | Manual | Sim | Aguardando produto oficial |
| `v2bg` | Mineração de produto Charter 758 para TP2 | Busca e processa produto cartográfico Charter 758 | Sim | Sim | Concluído |
| `v2bh` | Georreferenciamento e digitalização Charter 758 | Orienta digitalização do produto Charter 758 | Manual | Sim | Em andamento |

---

## Estágios da cadeia Curitiba

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v2ca` | Vinculação de eventos Curitiba | Registra vínculos evento-patch para Curitiba | Sim | Sim | Concluído |
| `v2cb` | Aquisição de evidência de evento Curitiba | Coleta e valida evidências de eventos em Curitiba | Sim | Sim | Concluído |
| `v2cc` | Entrada de evidência externa Curitiba | Organiza intake de fontes externas para Curitiba | Sim | Sim | Concluído |
| `v2cd` | Download de evidência externa Curitiba | Executa passada de download de fontes autorizadas | Local apenas | Sim | Concluído |
| `v2ce` | Varredura profunda de fontes oficiais Curitiba | Extrai dados estruturados de fontes oficiais | Sim | Sim | Concluído |
| `v2cf` | Monitor de entrada e QA geométrico | Acompanha entrada de evidências e constrói QA geométrico | Sim | Sim | Bloqueado por geometria de evento ausente |
| `v2cg` | Sobreposição espacial e auditoria de sensibilidade | Executa sobreposição e avalia sensibilidade | Sim | Sim | Bloqueado por geometria de evento ausente |

---

## Estágios de candidatos TP2 e evidência externa multiregional

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v2ci` | Inventário de candidatos TP2 | Inventaria 38 candidatos observacionais para etapa TP2 | Sim | Sim | Concluído — 23 candidatos, 15 bloqueados |
| `v2cj` | Priorização de candidatos TP2 | Prioriza candidatos TP2 por critério metodológico | Sim | Sim | Concluído |
| `v2ck` | Protocolo de digitalização manual | Guia para digitalização manual de geometrias de evento | Manual | Sim | Em andamento |
| `v2cl` | Validação de geometria observada | Valida geometrias de eventos observados registrados | Sim | Sim | Concluído (metadata) |
| `v2cm` | Replay de evento por patch | Simula replay de evento sobre patches sem promover rótulo | Sim | Sim | Concluído |
| `v2cn` | Matriz de lacunas de evidência externa | Mapeia lacunas de evidência geoespacial por região | Sim | Não | Concluído — 38 lacunas |
| `v2co` | Aquisição e auditoria de evidência externa | Adquire e audita fontes externas registradas | Sim | Sim | Concluído |
| `v2cp` | Manifesto público de evidência externa | Gera manifesto público de evidências externas (3 entradas) | Sim | Não | Concluído |
| `v2cq` | QA geoespacial externo | Verifica qualidade geoespacial das evidências (CRS, hash, limites) | Sim | Não | Concluído — 3 auditorias |
| `v2cr` | Pareamento de patch com evidência externa | Registra pares patch-evidência com critérios de validade | Sim | Sim | Concluído — 3 pares |
| `v2cs` | Semeadura de fontes externas reais | Registra metadados de fontes externas reais | Sim | Não | Concluído |
| `v2ct` | Triagem de licença de fontes | Classifica licença e redistribuição por fonte | Sim | Sim | Concluído |
| `v2cu` | Sync de registro de fontes externas | Sincroniza registro de fontes com manifesto público | Sim | Não | Concluído |
| `v2cv` | Checklist de descoberta de produtos externos | Organiza checklist de produtos externos identificados | Sim | Sim | Concluído |
| `v2cw` | Leitura regional de prontidão de evidência | Consolida prontidão de evidência por região | Sim | Não | Concluído |

---

## Estágios de auditoria de continuidade e recuperação da base original

Esta faixa de estágios trata da **rastreabilidade e recuperabilidade** de uma base de trabalho candidata anterior (códigos `v2dz`–`v2ef`), cujos artefatos não estavam disponíveis localmente. É uma **auditoria de continuidade**: documenta o que existe, o que se perdeu e o que é recuperável. Não recupera ground truth operacional, não fecha rótulo, não libera treino e não substitui a base original por um fallback.

| Código interno | Nome público | Finalidade | Executável | Revisão humana | Status científico |
|---|---|---|---|---|---|
| `v2dz`–`v2ef` | Base de trabalho candidata anterior | Conjunto de trabalho intermediário de etapa anterior, referenciado por etapas posteriores | — | Sim | Artefatos ausentes/não recuperados localmente |
| `v2es`–`v2ey` | Tentativa de recuperação controlada | Inspeciona artefatos vizinhos, valida candidatos e tenta restaurar a base por cópia controlada | Sim | Sim | Bloqueado — nenhuma fonte válida e nenhum fallback disponível |
| `v2ez`–`v2ff` | Auditoria forense de recuperabilidade | Índice de busca forense, extração por diff/patch, inspeção de objetos Git/reflog, plano de backups, validação de candidatos, registro de decisão e painel de perda | Sim | Sim | Concluído — decisão `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE` |
| `curadoria/repositorio-publico-ptbr` | Reorganização pública em português | Consolida a camada pública (README, índices, relatórios) em português brasileiro e integra a linha do tempo metodológica | Sim | Sim | Em andamento |

**Estado consolidado da base original** (saída de `v2ez`–`v2ff`):

- `original_base_status = ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE`;
- 53 registros candidatos da base original: **não recuperáveis** automaticamente (`original_53_recoverable = false`);
- fallback de 38 linhas: **indisponível** (`fallback_38_available = false`);
- somente referências/pistas recuperáveis foram encontradas — **referência não é conteúdo**;
- `ground_truth_operational_status = ABSENT`; `training_ready = false`.

Limite preservado: esta auditoria **não recupera ground truth operacional, não fecha rótulo, não libera treino e não substitui a base original por fallback**. A recuperação efetiva, se ocorrer, depende de ação manual (restauração por diff ou objeto Git) e de revisão humana.

---

## Notas metodológicas gerais

- **"Executável: Local apenas"** significa que o estágio requer dados pesados que existem apenas no workspace local (GeoTIFFs, embeddings `.npz`). Pode ser re-executado localmente, mas não em CI.
- **"Executável: Manual"** significa que o estágio requer intervenção humana (digitalização, geocodificação controlada, revisão de documento).
- **"Revisão humana: Sim"** significa que a saída do estágio requer validação por pesquisador antes de uso em qualquer etapa subsequente.
- **Bloqueado** não significa erro: significa que uma pré-condição metodológica (geometria, licença, fenômeno, ground truth) não está satisfeita. O bloqueio é documentado, rastreável e auditável.
