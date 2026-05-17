# Linhagem e grounding territorial dos patches do REV-P

## O que são os patches

Patches são recortes geográficos delimitados sobre áreas urbanas associadas a histórico de inundação e alagamento. Cada patch corresponde a uma bounding box territorial previamente definida sobre Curitiba, Petrópolis ou Recife.

O conjunto canônico tem 59 patches: 14 em Curitiba (CUR\_01–14), 27 em Petrópolis (PET\_01–27) e 18 em Recife (REC\_01–18).

Os patches não foram criados pelo pipeline DINO. A geometria dos patches vem de uma base territorial pré-existente (raw geometry do v1d, registros de sidecar externos). O DINO opera sobre imagens Sentinel associadas a esses patches — não define, não redesenha e não requalifica os limites territoriais.

## Origem da geometria

A raw geometry de 52 patches está disponível como bounding boxes em graus decimais WGS84, extraídas de arquivos sidecar externos produzidos antes da fase DINO. Os 7 patches restantes (CUR\_08–14) são placeholders — a bounding box de origem ainda não foi localizada e vinculada.

O crosswalk entre IDs canônicos (CUR\_01, PET\_01, REC\_01…) e IDs brutos de geração (curitiba\_00038, petropolis\_00104…) foi identificado como ação localmente executável sem operações espaciais (SPF\_04). Nenhuma promoção canônica foi feita até o momento.

## Inventário Sentinel

128 GeoTIFF Sentinel foram inventariados no workspace privado: 43 para Curitiba, 48 para Petrópolis e 37 para Recife. Todos usam projeção WGS84 UTM (EPSG:32722/32723/32725). As fontes oficiais de referência usam SIRGAS 2000 UTM (EPSG:31982/31983/31985) — a discrepância CRS é um bloqueador ativo (B1) e não foi promovida.

O manifest Sentinel (v1fu) registra 128 entradas com referências relativas de caminho. Nenhum pixel foi lido na construção do manifest.

## Trilha de vinculação patch → Sentinel

A vinculação entre IDs canônicos de patch e TIFs Sentinel é um problema de metadados não completamente resolvido. O estado atual é:

- 20 patches com designação candidata de TIF (PATCH\_TO\_TIF\_DESIGNATION)
- 32 patches com designação não resolvida
- 7 patches placeholder sem geometria de origem
- 0 patches com vinculação confirmada por sobreposição espacial (PATCH\_TO\_TIF\_CONFIRMED: requer Gate 3, atualmente BLOQUEADO)

A confirmação espacial depende do desbloqueio da porta de preflight (B1 + B6 + B7). Nenhuma operação espacial foi executada.

## Pipeline DINO sobre os Sentinel

O encoder DINOv2 com registros foi usado como encoder visual congelado para extrair representações dos patches Sentinel. A sequência de execução local foi:

1. **v1fv** — preflight local de assets: verificação de quais referências Sentinel são acessíveis no workspace privado antes da extração
2. **v1fx** — execução smoke: leitura real de pixels para 5 patches, extração de embeddings locais
3. **v1fz** — corpus balanceado: 12 embeddings (4 por região) para análise estrutural inicial
4. **v1ge** — corpus expandido: 12 embeddings com suporte a retomada e auditoria de hash

O modelo não foi ajustado, não foi retreinado e não foi avaliado como classificador. Os embeddings extraídos têm dimensão 768 e ficam exclusivamente em `local_runs/`.

## Diagnósticos estruturais

Sobre o corpus de 12 embeddings locais, foram produzidos diagnósticos de:

- vizinhos mais próximos e pares recíprocos (v1ga, v1gc)
- outliers e medoids (v1ga, v1gf)
- PCA e clustering exploratório (v1fy, v1fz)
- robustez a perturbações controladas (v1gd)
- geo-estrutura comparando distância embedding com distância espacial (v1gc)
- estabilidade longitudinal entre fases (v1gh)
- proveniência patch → embedding → diagnóstico (v1gi)

Todos esses outputs são diagnósticos de revisão. Não são classes semânticas, não são rótulos de inundação e não são evidência de desempenho preditivo.

## Contexto GIS

O baseline multicritério GIS (v1gq) associa indicadores físico-ambientais aos 12 patches do corpus DINO:

- **Distância ao rio**: disponível para as três regiões
- **Densidade viária**: disponível para Recife
- **Uso do solo**: BLOQUEADO para todos os patches do dino-corpus — a camada FBDS de Petrópolis cobre lat −22.575 a −22.202; os 4 patches de Petrópolis do dino-corpus têm centroides em lat ≈ −22.598, fora dessa extensão
- **Densidade populacional**: BLOQUEADO — sem dados censitários locais

O índice parcial (2/4 indicadores) está disponível apenas para patches de Recife. O índice é um proxy interpretável para comparação — não é verdade de campo nem alvo supervisionado.

A auditoria de cobertura de uso do solo (v1gt) avaliou os 128 patches do manifest v1fu: 33 patches de Petrópolis dentro do FBDS têm cobertura disponível. Os 12 patches do dino-corpus não incluem esses 33 — a lacuna é de localização geográfica, não de processamento.

## O que os embeddings não são

Os embeddings DINO extraídos sobre patches Sentinel **não são**:

- rótulos de inundação
- classes de suscetibilidade
- alvos de treinamento supervisionado
- ground truth de qualquer tipo
- evidência de desempenho preditivo
- resultado de ajuste fino ou retreinamento

O campo `review_priority` no índice estrutural (v1gf) é um indicador de triagem determinístico para selecionar candidatos de inspeção humana. Não é rótulo, não é classe e não é proxy de vulnerabilidade.

## Bloqueadores ativos

| Bloqueador | Status | Consequência |
|---|---|---|
| B1 — CRS externo | Não resolvido | Operações espaciais proibidas |
| B6 — prontidão de patch | 0/59 LOCKED | Validação espacial bloqueada |
| B7 — revisão humana | Pendente | Abertura de portas depende de revisão |
| Rótulos observados | Não disponíveis | Treinamento supervisionado bloqueado |
| Multimodal | Em espera | Pendente resolução Recife balanço/recuperação |

Com esses bloqueadores ativos, o escopo válido do projeto é: extração local de embeddings, diagnósticos estruturais exploratórios, contexto GIS interpretável e triagem de revisão humana.

## Claims permitidos e proibidos

**Permitidos:**
- "Os embeddings DINO capturam características estruturais visuais dos patches Sentinel."
- "O índice GIS é um proxy interpretável para priorização de revisão."
- "Os diagnósticos estruturais suportam triagem humana de patches candidatos."

**Proibidos:**
- Qualquer afirmação de que DINO prediz suscetibilidade ou vulnerabilidade.
- Qualquer uso do índice GIS como ground truth ou alvo.
- Qualquer promoção de cluster ou embedding para classe real de inundação.
- Qualquer afirmação de validação preditiva sem rótulos observados formais.
- Qualquer claim de que a fusão multimodal está ativa ou concluída.
