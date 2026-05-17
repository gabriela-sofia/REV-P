# Protocolo de embeddings DINO Sentinel-first

## Objetivo

Este documento registra o fluxo auditável DINO Sentinel-first do REV-P. A trilha DINO é um caminho de representação auto-supervisionado e apenas de revisão para inspeção estrutural de material elegível de patches Sentinel. Não é um classificador supervisionado de suscetibilidade a inundações, não cria rótulos ou alvos e não promove clusters a classes científicas.

## Decisões metodológicas

- `review_only=true`
- `supervised_training=false`
- `labels_created=false`
- `predictive_claims=false`
- `multimodal_hold=true`
- clusters são apenas diagnósticos estruturais
- Sentinel-first é o caminho ativo
- stacks multimodais permanecem em espera até que o bloqueio de balanço/recuperação de Recife seja resolvido

## Justificativa

DINO é usado como encoder visual congelado porque o projeto não dispõe atualmente de rótulos observados de inundação validados, alvos supervisionados ou uma porta de verificação CRS/preflight liberada. Embeddings auto-supervisionados permitem comparação em nível de material, revisão de vizinhos, revisão de outliers e clustering estrutural sem reivindicar desempenho preditivo.

DINOv2 com registros é o backbone preferencial porque os tokens de registro são projetados para melhorar a estabilidade de representação em espaços de features de vision transformer. DINOv2 sem registros permanece como caminho de controle. DINOv3 é registrado apenas como comparação futura, se disponível e explicitamente revisado.

## Sumário de versões

| Versão | Script | Objetivo | Entradas | Outputs locais | QA | Status |
| --- | --- | --- | --- | --- | --- | --- |
| v1fw | `scripts/dino/revp_v1fw_dino_embedding_extraction_scaffold.py` | Scaffold dry-run e esquema de execução para embeddings futuros | manifest v1fu, preflight v1fv opcional, config DINO | `local_runs/dino_embeddings/v1fw/` | verificações dry-run, sem leitura de modelo/pixel por padrão | implementado |
| v1fx | `scripts/dino/revp_v1fx_dino_smoke_embedding_execution.py` | Execução smoke explícita com leitura real de pixels Sentinel e embeddings locais | manifest v1fu, preflight v1fv, config DINO | `local_runs/dino_embeddings/v1fx/` | tentativas de modelo, metadados, falhas, resumo, QA | implementado |
| v1fy | `scripts/dino/revp_v1fy_dino_embedding_corpus_analysis.py` | Análise exploratória do corpus a partir de embeddings locais | manifest de embedding local e metadados do v1fx | `local_runs/dino_embeddings/v1fy/` | corpus, PCA, clustering, vizinhos, diagnósticos regionais | implementado |
| v1fz | `scripts/dino/revp_v1fz_dino_balanced_embedding_corpus.py` | Subconjunto balanceado de embeddings Sentinel por região | manifest v1fu, preflight v1fv, config DINO | `local_runs/dino_embeddings/v1fz/` | seleção balanceada, embeddings, PCA/clustering/vizinhos/regiões | implementado |
| v1ga | `scripts/dino/revp_v1ga_dino_embedding_structural_consistency_analysis.py` | Análise de consistência estrutural entre regiões, vizinhos, clusters e seeds | manifest local e embeddings do v1fz | `local_runs/dino_embeddings/v1ga/` | consistência, centroide, estabilidade de cluster, QA de outliers | implementado |
| v1gb | `scripts/dino/revp_v1gb_dino_embedding_local_visual_structural_review.py` | Revisão visual estrutural local de embeddings, medoids, vizinhos e outliers | manifest local e embeddings do v1fz | `local_runs/dino_embeddings/v1gb/` | painéis visuais, consistência espacial, verificações multiescala, medoids, taxonomia de outliers | implementado |
| v1gc | `scripts/dino/revp_v1gc_dino_embedding_geo_structural_diagnostics.py` | Diagnósticos geo-estruturais ligando geometria local de patches com vizinhanças de embedding | manifest local e embeddings do v1fz | `local_runs/dino_embeddings/v1gc/` | comparações distância-similaridade, topologia de grafo, pontes inter-região, candidatos de transição | implementado |
| v1gd | `scripts/dino/revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py` | Diagnósticos de robustez a perturbações para embeddings DINO Sentinel locais | manifest local e embeddings do v1fz | `local_runs/dino_embeddings/v1gd/` | perturbações controladas, deriva, persistência de vizinhos, robustez de grafo, robustez regional | implementado |
| v1ge | `scripts/dino/revp_v1ge_dino_expanded_sentinel_embedding_corpus.py` | Execução expandida do corpus de embeddings Sentinel com retomada e balanceamento regional | manifest v1fu, preflight v1fv, config DINO | `local_runs/dino_embeddings/v1ge/` | consistência de embeddings, hashes, falhas por região, auditoria de retomada/skip | implementado |
| v1gf | `scripts/dino/revp_v1gf_dino_structural_evidence_index.py` | Índice integrado de evidência estrutural para triagem de revisão manual | outputs locais do v1fz/v1ge e v1ga–v1gd | `local_runs/dino_embeddings/v1gf/` | guardrails, resumo de prioridade de revisão, QA sem rótulo/alvo | implementado |
| v1gg | `scripts/dino/revp_v1gg_dino_human_review_package.py` | Pacote de revisão humana apenas local para medoids, outliers, pontes e representantes | índice de evidência estrutural do v1gf e manifests visuais locais | `local_runs/dino_embeddings/v1gg/` | manifest de revisão, lotes, README local, guardrails de revisão humana | implementado |
| v1gh | `scripts/dino/revp_v1gh_dino_longitudinal_structural_diagnostics.py` | Comparação longitudinal de diagnósticos estruturais entre fases DINO | outputs locais do v1fz–v1gg | `local_runs/dino_embeddings/v1gh/` | estabilidade de vizinhos, outliers, medoids, pontes, prioridade de revisão e regionais | implementado |
| v1gi | `scripts/dino/revp_v1gi_dino_structural_provenance_tracker.py` | Rastreamento de proveniência patch → embedding → diagnóstico | outputs locais do v1fz–v1gg | `local_runs/dino_embeddings/v1gi/` | índice de proveniência, histórico de diagnósticos, rastreabilidade de revisão | implementado |
| v1gj | `scripts/dino/revp_v1gj_multimodal_readiness_audit.py` | Auditoria de prontidão multimodal sem execução multimodal | manifests Sentinel v1fu/v1fv e outputs DINO locais | `local_runs/dino_embeddings/v1gj/` | tabela de prontidão, bloqueadores, inventário de assets, guardrails multimodal-desabilitado | implementado |
| v1gk | `scripts/dino/revp_v1gk_dino_pipeline_reproducibility_audit.py` | Auditoria final de reprodutibilidade do pipeline DINO Sentinel-first | scripts, configs, docs versionados e presença de outputs locais | `local_runs/dino_embeddings/v1gk/` | existência de scripts/configs/docs, guardrails, política local-only, auditoria de artefatos proibidos | implementado |
| v1gn | `scripts/dino/revp_v1gn_dino_execution_health_monitor.py` | Monitor operacional de saúde para outputs DINO locais | manifests de embedding locais e outputs upstream | `local_runs/dino_embeddings/v1gn/` | embeddings ausentes/corrompidos, verificações de hash/dimensão, presença upstream, pontuação de saúde | implementado |
| v1go | `scripts/dino/revp_v1go_dino_pipeline_orchestrator.py` | Camada leve de orquestração e validação | registro de versões e dependências locais de estágios | `local_runs/dino_embeddings/v1go/` | grafo de dependências, dry-run, validate-only, aplicação do bloqueio multimodal | implementado |
| v1gp | `scripts/dino/revp_v1gp_dino_github_release_readiness_audit.py` | Auditoria de prontidão para release no GitHub para consolidação de commit/PR local | scripts, configs, docs versionados e guardrails locais | `local_runs/dino_embeddings/v1gp/` | verificação de artefatos proibidos, verificação de paths privados, matriz metodológica, cobertura de docs/operação, status de prontidão | implementado |
| v1gq | `scripts/dino/revp_v1gq_gis_multicriteria_vulnerability_baseline.py` | Baseline de vulnerabilidade multicritério GIS para patches Sentinel | manifest v1ge, índice de evidência v1gf, camadas GIS externas via `--gis-root` | `local_runs/dino_embeddings/v1gq/` | prontidão de indicadores, pontuações DR/DV por patch, índice de vulnerabilidade parcial (2/4 indicadores para Recife), crosswalk DINO | implementado |
| v1gr | `scripts/dino/revp_v1gr_gis_land_use_readiness_and_conversion_audit.py` | Auditoria de prontidão e conversão de uso do solo GIS para desbloquear o indicador land_use no v1gq | busca local no REV-P, `--gis-root` opcional apontando para dados GIS | `local_runs/dino_embeddings/v1gr/` | inventário de uso do solo, auditoria de dependências, leitura de DBF, mapeamento de classes, plano/resultados de conversão, cobertura regional, prontidão para v1gq | implementado |
| v1gs | `scripts/dino/revp_v1gs_gis_land_use_geometry_enablement.py` | Habilitação de geometria de uso do solo GIS e reexecução parcial do v1gq | `--gis-root` apontando para dados GIS; script v1gq para reexecução | `local_runs/dino_embeddings/v1gs/` | auditoria de dependências, auditoria de sidecar, leitura de geometria (pyogrio→geopandas→fiona), conversão para WGS84, schema/extensão/distribuição de classes, plano de reexecução do v1gq, resultados da reexecução | implementado |
| v1gt | `scripts/dino/revp_v1gt_gis_land_use_coverage_expansion_audit.py` | Auditoria de expansão de cobertura de uso do solo GIS no dino-corpus e no manifest completo v1fu | GeoJSON WGS84 do v1gs, manifest v1ge (dino-corpus) ou manifest v1fu (full-manifest), `--gis-root` opcional para resolução de TIFs | `local_runs/dino_embeddings/v1gt/` | cobertura por patch (COVERED/BBOX_OVERLAP_NO_CENTROID/UNCOVERED/NO_TIF), resumo por região, inventário de fontes, lacunas de cobertura, candidatos de expansão, QA | implementado |

## Entradas e outputs

Entradas primárias versionadas:

- `manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv`
- `configs/dino_embedding_extraction.example.yaml`
- scripts em `scripts/dino/`

Entradas privadas/locais:

- `local_runs/dino_asset_preflight/v1fv/dino_local_asset_preflight_v1fv.csv`
- referências raster Sentinel resolvidas dentro do workspace privado `PROJETO`

Outputs exclusivamente locais:

- produtos QA CSV/JSON em `local_runs/dino_embeddings/`
- arquivos `.npz` de embedding em `local_runs/dino_embeddings/*/embeddings/`

Nenhum arquivo `.npz`, `.npy`, raster, GeoTIFF, checkpoint ou output de `local_runs/` é destinado ao Git.

## Checklist de QA e auditoria

- embeddings `.npz` ficam apenas em `local_runs/`
- outputs pesados estão fora do Git
- `pytest` passa
- manifests locais são gerados
- arquivos QA CSV são gerados
- arquivos JSON de resumo são gerados
- o corpus balanceado do v1fz contém Curitiba, Petropolis/Petrópolis e Recife
- nenhum loop de treinamento, otimizador, rótulo ou alvo é criado
- nenhuma métrica preditiva ou afirmação supervisionada é emitida

## Escopo da análise exploratória

A camada de análise suporta:

- coordenadas PCA e variância explicada
- clustering estrutural leve
- verificações de vizinho mais próximo e vizinho recíproco
- diagnósticos de outliers
- centroides regionais, dispersão e similaridade intra/inter-região
- estabilidade de cluster entre seeds e valores de K
- painéis de revisão visual local para vizinhos mais próximos, medoids, casos de borda, exemplares regionais e outliers
- diagnósticos de consistência espacial local e verificações estruturais multiescala
- diagnósticos geo-estruturais comparando vizinhanças de embedding com centroides de patches, topologia de grafo local e candidatos de ponte inter-região
- diagnósticos de robustez a perturbações sob pequenas alterações controladas de imagem, sem criar aumentos de treinamento
- execução de corpus local expandida com suporte a retomada/skip-existing
- indexação integrada de evidência estrutural para triagem de revisão
- empacotamento de revisão humana apenas local para inspeção manual posterior
- verificações de estabilidade longitudinal entre fases de diagnóstico DINO
- rastreamento de proveniência de patch a embedding a pacote de revisão
- auditoria de prontidão multimodal enquanto a execução multimodal permanece desabilitada
- auditoria final de reprodutibilidade e registro de comandos para reexecuções locais
- monitoramento operacional de saúde para manutenção local
- orquestração leve para dry-run e validação seletivos

Esses outputs são diagnósticos estruturais para revisão. Não são classes semânticas, não são rótulos de inundação e não são evidência de desempenho de modelo.

## Revisão visual estrutural

v1gb adiciona painéis visuais apenas locais para suportar revisão humana posterior do comportamento estrutural dos embeddings. Os painéis visam tornar auditáveis as relações de vizinho mais próximo, pares recíprocos, medoids, casos de borda, embeddings isolados e exemplares regionais — sem converter clusters em rótulos.

As seleções de medoid e representativo são apenas conveniências estruturais. Indicam posições em um espaço de embedding para inspeção, não pertencimento a classe no mundo real, não ocorrência de inundação e não evidência de validação. A QA visual também é limitada pelo tamanho atual do corpus local, a disponibilidade de renderização de patches Sentinel e o fato de que os painéis de imagem derivam de leituras locais de tempo de execução em vez de dados brutos versionados.

## Diagnósticos geo-estruturais

v1gc compara relações de embedding com geometria espacial local quando coordenadas de centroide ou limites de metadados raster estão disponíveis. Produz tabelas de distância versus similaridade, métricas de sobreposição regional, resumos de compacidade, componentes de grafo, hubs, candidatos de ponte e candidatos de transição, além de verificações de continuidade de topologia.

A camada de grafo é apenas diagnóstica: nós são patches, arestas são relações de vizinho mais próximo em embedding, e pontes inter-região são candidatos para revisão humana. Não são classes, não são rótulos, não são alvos de validação e não são evidência de desempenho preditivo. As métricas de topologia dependem do tamanho atual do corpus, da disponibilidade de coordenadas e do top-k de vizinhança escolhido; devem ser tratadas como auxílios de auditoria para selecionar exemplos para inspeção manual posterior.

## Diagnósticos de robustez a perturbações

v1gd aplica pequenas perturbações reversíveis a renderizações Sentinel locais para medir se as relações de embedding DINO são estruturalmente estáveis. As perturbações incluem ruído Gaussiano leve, escalonamento de brilho, escalonamento de contraste, desfoque, jitter de recorte e dropout de banda opcional. São usadas apenas para auditoria de sensibilidade e não são salvas como conjunto de treinamento.

Os outputs de robustez comparam embeddings originais versus perturbados por meio de deriva cosseno, persistência de vizinho mais próximo, estabilidade de atribuição de cluster, persistência de medoid, persistência de aresta de grafo, persistência de ponte, estabilidade de hub e resumos de deriva regional. Esses diagnósticos suportam revisão manual de sensibilidade e não estabelecem confiabilidade preditiva, pertencimento a classe ou desempenho supervisionado.

## Corpus expandido e triagem de revisão humana

v1ge expande a execução de embeddings Sentinel-first além do corpus balanceado inicial pequeno quando a computação local e a disponibilidade do modelo permitem. A execução permanece apenas local e suporta `--limit`, `--per-region-limit`, `--resume` e `--skip-existing` para que a execução parcial possa ser auditada sem sobrescrever embeddings locais anteriores.

v1gf consolida diagnósticos estruturais em um único índice de evidência por patch. O campo `review_priority` é uma indicação determinística de triagem apenas para inspeção humana. Não é rótulo, não é classe, não é alvo e não é evidência de que um patch tem qualquer status de inundação ou suscetibilidade.

v1gg empacota referências locais para futura revisão humana de medoids, outliers, pontes, embeddings robustos/instáveis, exemplos de vizinhos recíprocos e representantes regionais. Não copia rasters brutos e não versiona imagens locais. Notas humanas ficam intencionalmente em branco até que a inspeção manual ocorra.

## Diagnósticos longitudinais e proveniência

v1gh compara sinais estruturais entre as fases DINO locais para verificar se relações de vizinhos, marcadores de outlier, papéis de medoid, papéis de ponte, resumos regionais e triagem de prioridade de revisão permanecem rastreáveis entre versões. O output é uma auditoria de persistência diagnóstica, não uma afirmação de mudança ambiental temporal.

v1gi registra a proveniência patch → embedding → diagnóstico. Rastreia quais versões tocaram cada patch, quais diagnósticos foram produzidos, quais arquivos QA passaram, quais visualizações locais existem e se cada patch aparece em papéis de medoid, ponte, outlier ou pacote de revisão humana.

## Hold multimodal

v1gj audita a prontidão estrutural para trabalho multimodal futuro sem ativar a execução multimodal. Registra disponibilidade Sentinel, status de preflight local, categorias de bloqueadores conhecidos, inventário de assets e guardrails. Prontidão não equivale a execução: `multimodal_execution_enabled=false` e `multimodal_training_enabled=false` permanecem restrições ativas.

## Encerramento de reprodutibilidade

v1gk audita os scripts versionados, configs obrigatórias, documentação, presença de outputs locais, proteção `.gitignore`, flags determinísticas e guardrails metodológicos. A auditoria é leve e não reexecuta extração de embeddings ou outras operações pesadas.

Os resumos finais de comandos e científico são versionados separadamente:

- `docs/dino_command_registry.md`
- `docs/dino_sentinel_scientific_evidence_summary.md`

## Manutenção operacional

v1gn monitora a saúde da execução local sem adicionar nova análise científica. Verifica disponibilidade de arquivos de embedding, consistência de manifest, legibilidade de hash, embeddings corrompidos, divergência de dimensões, presença de outputs upstream, consistência regional e hashes estruturais duplicados. A pontuação operacional é limitada a `HEALTHY`, `WARNING` ou `DEGRADED`.

v1go fornece uma camada leve de orquestração para manutenção local futura. Pode validar um único estágio ou todos os estágios, produzir um grafo de dependências e executar planos de comando dry-run. Não habilita execução multimodal e deve ser usado de forma conservadora antes de qualquer reexecução pesada.

Orientações de recuperação:

- usar `--validate-only` antes de reexecutar um estágio;
- usar `--dry-run` para inspecionar o plano exato de comandos;
- usar execução em nível de estágio apenas quando os outputs upstream já forem conhecidos;
- manter `local_runs/` não rastreado;
- reexecutar v1gn após qualquer limpeza manual ou execução retomada de embeddings.

## Auditoria de prontidão para release

v1gp audita o estado versionado do bloco DINO Sentinel-first antes de qualquer commit ou consolidação de PR. Verifica artefatos proibidos fora de `local_runs/`, caminhos absolutos privados em arquivos versionáveis, proteções metodológicas, documentação obrigatória e cobertura de scripts/testes para blocos recentes. A auditoria produz um status de prontidão local (`READY_FOR_LOCAL_COMMIT`, `READY_WITH_REVIEW_NOTES` ou `BLOCKED`) e outputs CSV/JSON estruturados sem realizar nenhuma operação git.

## Baseline de vulnerabilidade multicritério GIS

v1gq computa um baseline de vulnerabilidade multicritério para os 12 patches Sentinel auditados no v1ge. Usa até quatro indicadores: distância ao rio, uso do solo, densidade populacional e densidade viária. O índice de vulnerabilidade é um proxy estrutural e interpretável para comparação, priorização e revisão humana — não é verdade de campo, não é rótulo e não é alvo supervisionado. A integração com DINO é apenas exploratória e estrutural.

Indicadores disponíveis com dados GIS locais: distância ao rio (GeoJSON de hidrografia disponível para as três regiões), densidade viária (Recife apenas, via segmentos municipais de vias). Uso do solo é BLOCKED para todos os patches do dino-corpus — a camada FBDS de Petrópolis foi convertida para WGS84 (v1gs, 6861 feições), mas os 4 patches de Petrópolis do dino-corpus têm centroides em lat ≈ −22.598, fora da extensão da camada FBDS (lat −22.575 a −22.202); esta é uma lacuna de cobertura de dados, não um erro de processamento. Curitiba e Recife não têm fontes de uso do solo disponíveis. Densidade populacional é BLOCKED (sem dados censitários encontrados).

O índice parcial (2/4 indicadores) é computado apenas para patches de Recife. Patches de Curitiba e Petrópolis são BLOCKED por cobertura insuficiente de indicadores. O índice usa pesos iguais (0,25 cada) e escala ordinal de 1 a 3 por indicador.

O script aceita um argumento `--gis-root` apontando para um diretório local de dados GIS. Sem ele, o script opera em modo apenas auditoria e todos os indicadores são BLOCKED. Caminhos privados do GIS root aparecem apenas em outputs de `local_runs/` e nunca em arquivos versionáveis.

## Auditoria de prontidão e conversão de uso do solo GIS

v1gr inventaria arquivos de uso do solo no diretório local REV-P e em um GIS root externo opcional, audita dependências Python GIS (fiona, geopandas, pyogrio, shapely, rasterio, pandas), lê tabelas de atributos de shapefiles FBDS usando um leitor DBF puro Python (sem necessidade de fiona) e constrói um mapeamento candidato de classe para pontuação para integração futura com o v1gq.

Resultados atuais com gis-root do PROJETO: um arquivo FBDS de uso do solo encontrado para Petrópolis (`RJ_3303906_USO.shp`, 8,7 MB), DBF lido com sucesso, seis valores únicos de CLASSE_USO extraídos (formação florestal, formação não florestal, silvicultura, água, área antropizada, área edificada). Conversão de geometria bem-sucedida após instalar shapely, fiona, geopandas, pyogrio. Curitiba e Recife não têm fontes de uso do solo.

A tabela de mapeamento de classes é um candidato apenas para revisão humana. Atribui pontuações ordinais (1 = floresta/água, 2 = antropizado/agricultura/silvicultura, 3 = urbano/edificado) a strings de classe FBDS conhecidas. Não é uma classificação final, não é verdade de campo, não é rótulo e não é alvo supervisionado.

v1gr não modifica outputs do v1gq. Caminhos GIS privados aparecem apenas em outputs de `local_runs/` e nunca em arquivos-fonte versionáveis.

## Habilitação de geometria de uso do solo GIS

v1gs tenta a leitura de geometria do shapefile FBDS de Petrópolis em ordem de prioridade (pyogrio → geopandas → fiona), reprojeta de SIRGAS 2000 UTM Zona 23S (EPSG:31983) para WGS84 (EPSG:4326) e salva o resultado em `local_runs/dino_embeddings/v1gs/converted/petropolis_land_use_v1gs.geojson`. Em seguida, orquestra uma reexecução parcial do v1gq com `--land-use-geojson-petropolis` apontando para o arquivo convertido, escrevendo os outputs da reexecução em `local_runs/dino_embeddings/v1gq_rerun_v1gs/`.

Resultados atuais: geometria lida via pyogrio, 6861 feições, 6 classes CLASSE_USO. Reexecução do v1gq realizada. No entanto, os centroides dos patches Sentinel de Petrópolis (lat ≈ −22.598) ficam aproximadamente 2–3 km ao sul do limite sul da camada FBDS (lat −22.575). Testes ponto-em-polígono não retornam correspondência para nenhum dos quatro patches de Petrópolis. O indicador `land_use` permanece BLOCKED para todos os patches de Petrópolis. Esta é uma lacuna real de cobertura de dados entre as localizações de patches do corpus v1ge e a extensão mapeada pelo FBDS — não é um erro de processamento.

A integração do indicador land_use com o v1gq é arquiteturalmente conectada via `--land-use-geojson-petropolis`. Se dados de uso do solo cobrindo a extensão sul dos patches de Petrópolis se tornarem disponíveis no futuro, o v1gs pode ser reexecutado sem alterações adicionais no código.

## Auditoria de expansão de cobertura de uso do solo GIS

v1gt avalia a cobertura de fontes de uso do solo para cada patch Sentinel em dois escopos: o dino-corpus de 12 patches (manifest v1ge) e o manifest completo v1fu de 128 patches. Usa o GeoJSON WGS84 do FBDS de Petrópolis produzido pelo v1gs como única fonte conhecida e verifica cada patch por sobreposição de bbox e teste ponto-em-polígono de centroide (usando shapely). Os status de cobertura são `COVERED`, `BBOX_OVERLAP_NO_CENTROID`, `UNCOVERED` e `NO_TIF`.

**Resultados dino-corpus** (12 patches): 0 COVERED, 2 apenas BBOX (patches de Petrópolis com sobreposição de bbox mas centroides ~2–3 km ao sul do limite do FBDS — consistente com a lacuna de cobertura encontrada no v1gs), 10 UNCOVERED (Curitiba e Recife não têm fonte de uso do solo). Status geral: `BBOX_PARTIAL`.

**Resultados full-manifest** (128 patches): 33 COVERED, 13 apenas BBOX, 82 UNCOVERED. Os 33 patches cobertos são patches de Petrópolis no manifest v1fu cujos centroides estão dentro da extensão do FBDS. Isso confirma que a camada FBDS cobre porções da área Sentinel manifestada de Petrópolis — os patches de Petrópolis do dino-corpus ficam fora desse limite.

Cinco candidatos de expansão estão documentados (MapBiomas para Curitiba, Recife e Petrópolis; FBDS estendido para Petrópolis; grade IBGE LULC para todas as regiões) — todos `NOT_ACQUIRED`. Nenhum rótulo é criado. O status de cobertura indica apenas disponibilidade de dados, não verdade de campo ou classificação de vulnerabilidade.

## Limitações atuais

- O corpus balanceado é intencionalmente pequeno e exploratório.
- Os painéis de revisão visual são auxílios locais de QA e não devem ser interpretados como explicações semânticas de clusters.
- Os diagnósticos geo-estruturais de grafo são sensíveis ao tamanho do corpus local, à proveniência das coordenadas e às configurações de vizinhança.
- Os diagnósticos de perturbação são auditorias locais de sensibilidade; não são aumentos de treinamento e não validam robustez operacional.
- A execução do corpus expandido ainda é limitada pela disponibilidade local do modelo, velocidade de CPU/GPU e acesso a assets privados.
- `review_priority` e entradas do pacote de revisão humana são apenas auxílios de fluxo de auditoria.
- A estabilidade longitudinal é persistência diagnóstica entre outputs locais, não inferência de série temporal ambiental.
- A prontidão multimodal é uma camada de auditoria de bloqueadores e preparação de compatibilidade, não fusão, geração de stack ou treinamento multimodal.
- A auditoria de reprodutibilidade verifica estrutura do pipeline e disponibilidade de evidência local; não substitui a reexecução completa do fluxo local.
- Monitoramento de saúde e orquestração são ferramentas de manutenção, não novas camadas de evidência.
- Execução em CPU é aceitável para execuções smoke e de auditoria, mas execuções completas podem ser lentas.
- Embeddings DINO dependem de disponibilidade local ou download de modelo explicitamente permitido.
- Assets multimodais permanecem excluídos do caminho ativo.
- Comparações regionais são apenas descritivas e estruturais.

## Próximos passos válidos

- Expandir embeddings Sentinel em direção ao corpus completo de 128 patches.
- Repetir a análise de consistência v1ga em execuções locais maiores.
- Revisar manualmente medoids, outliers e vizinhos recíprocos.
- Adicionar painéis de QA visual apenas se mantidos locais ou explicitamente aprovados para documentação versionada.
- Revisitar stacks multimodais apenas após resolução dos bloqueadores de recuperação/balanço de Recife.
