# Datasets e artefatos de pesquisa do REV-P

## O que o projeto produziu

O REV-P produziu quatro tipos de artefatos com papéis metodológicos distintos:

**Manifests auditáveis** — tabelas CSV/JSON que descrevem corpora, assets e decisões
metodológicas. São commitados neste repositório porque documentam o método sem
depender de arquivos pesados. O manifest Sentinel v1fu registra 128 candidatos com
campos explícitos de `label_status=NO_LABEL`, `target_status=NO_TARGET` e
`claim_scope=REVIEW_ONLY_NO_PREDICTIVE_CLAIM`.

**Corpus de patches Sentinel** — 59 patches territoriais (14 Curitiba, 27 Petrópolis,
18 Recife) com bounding boxes derivadas de bases externas pré-DINO. Os GeoTIFFs
correspondentes existem no workspace privado, somam múltiplos gigabytes e não são
versionados. O que é público é a rastreabilidade: qual patch tem qual designação de
TIF, qual QA foi executado, qual estado de vinculação persiste.

**Embeddings DINO** — representações visuais extraídas pelo encoder DINOv2 com
registros (congelado) sobre patches Sentinel. O corpus operacional tem 12 embeddings
balanceados (4 por região). Arquivos `.npz` ficam em `local_runs/` e não são
versionados. São evidência de execução técnica, não dataset para treinamento.

**Evidências externas GIS** — fontes institucionais por região (PE3D/Recife,
SGB/Petrópolis, GeoCuritiba/Curitiba) indexadas no pacote de validação externa. Os
arquivos pesados ficam locais; o pacote público contém CSV/JSON de evidência e guardrails.

---

## Por que os patches Sentinel são uma contribuição científica

Os 59 patches não foram selecionados aleatoriamente. A seleção passou por:

1. Grounding territorial em áreas com histórico de inundação e alagamento documentado
   em fontes institucionais (Defesa Civil, SGB/CPRM, PE3D, GeoCuritiba)
2. Derivação de bounding boxes a partir de bases externas pré-existentes — o DINO
   não define os limites, opera sobre o que já estava territorialmente consolidado
3. Separação metodológica de três regiões com papéis distintos: Recife como caso de
   evidência forte, Petrópolis como caso de complexidade de processo, Curitiba como
   região de contraste metodológico
4. Auditoria de vinculação Sentinel por patch (v1fm–v1fo), com documentação explícita
   de cada estado de resolução — incluindo os estados não resolvidos

O corpus não é um dataset de treinamento. É um conjunto territorial auditável para
análise exploratória e revisão estrutural.

---

## Por que os manifests são auditáveis

Um manifest é auditável quando qualquer leitor pode verificar, a partir do arquivo
público, que uma decisão metodológica foi tomada e quais guardrails ela impõe.

O manifest v1fu satisfaz isso: cada linha tem `label_status`, `target_status`,
`pixel_read_status` e `claim_scope` preenchidos com valores fixos que proíbem
classificação supervisionada. O script que gerou o manifest tem um QA automatizado
que falha se qualquer desses campos for diferente do esperado. Isso é rastreabilidade
por código, não apenas por documentação.

O pacote de validação externa tem um CSV de guardrails (`external_validation_master_claims_guardrail_v1.csv`)
com claims `ALLOWED` e `FORBIDDEN` explicitamente separados. Qualquer redação futura
pode ser verificada contra essa lista.

---

## Por que os dados pesados ficam fora do GitHub

Os GeoTIFFs Sentinel têm resolução de 10 m, cobrem patches de ~1,6 km² e somam
entre 10 MB e 200 MB por arquivo. Os 128 candidatos totalizam múltiplos gigabytes.
Versionar esses arquivos:

- tornaria o repositório impraticável para clonar;
- não adicionaria valor científico: as imagens Sentinel são de acesso público via
  Copernicus Open Access Hub e podem ser reproduzidas a partir dos metadados dos
  manifests;
- exporia caminhos e estruturas do workspace privado.

O que o repositório versiona é suficiente para verificar a metodologia: manifests com
referências de caminho relativo, QA passado, configuração do encoder, guardrails
explícitos. Um revisor pode confirmar que o método é auditável sem acesso aos rasters.

---

## Como os datasets se conectam à metodologia

A sequência abaixo é a cadeia de rastreabilidade do projeto:

**Grounding territorial**
Seleção de três regiões com histórico documentado. Bounding boxes derivadas de bases
externas. O DINO não participou dessa etapa.

↓

**Sentinel-first**
128 GeoTIFFs Level-2A inventariados. Decisão de priorizar Sentinel sobre multimodal
documentada em v1ft com justificativa quantitativa: 1 stack Recife disponível contra
37 candidatos Sentinel — desequilíbrio que bloqueia multimodal até recuperação.

↓

**Auditabilidade dos patches**
Designação TIF por patch (v1fm): 20 patches com candidato; 32 sem resolução; 7
placeholder. Estado documentado, não encoberto. Reconciliação de naming Recife ext/bg
(v1fo): 18 patches com problema de nomenclatura documentado e não resolvido.

↓

**Manifests**
v1fu: 128 entradas Sentinel com `label_status=NO_LABEL`, `target_status=NO_TARGET`,
QA PASS. Nenhum pixel lido na construção do manifest.

↓

**QA**
18 guardrails auditados a zero antes de qualquer extração. Pacote de validação
externa: 11 QA checks passados, 0 falhas.

↓

**Embeddings DINO**
DINOv2 com registros, encoder congelado. Extração local em v1fx (5 patches, smoke)
e v1fz (12 patches, corpus balanceado). Embeddings em `local_runs/` — não versionados,
não labels, não targets. Diagnósticos estruturais: kNN, clustering, outliers,
robustez, proveniência, triagem de revisão humana.

↓

**Contextualização GIS**
Índice multicritério v1gq sobre os 12 patches do corpus: distância ao rio, uso do
solo, densidade viária. Não é ground truth. Não é alvo supervisionado. É proxy
interpretável para comparação estrutural com os embeddings.

↓

**Revisão humana**
Pacote de evidências externas por região: PE3D/Recife (evidência forte de terreno),
SGB/Petrópolis (terreno com separação de processo obrigatória), GeoCuritiba (contraste
metodológico). Estado: mastered for review — pacote pronto, revisão humana não executada.

↓

**Governança multimodal**
Multimodal explicitamente em hold. Condição de desbloqueio: recuperação do stack
Recife, balanceamento regional, aprovação de revisor. Decisão documentada em v1ft,
não inferida.

---

## O que o projeto não produziu

- Rótulos de inundação observada (não existem)
- Dataset de treinamento supervisionado (não existe)
- Ground truth de suscetibilidade (não existe)
- Métricas de desempenho preditivo (não existem)
- Classificador supervisionado (não existe)

O resultado atual é um corpus territorial auditável com representações estruturais
exploratórias e evidências externas documentadas por região. Isso é a contribuição
desta fase — não mais, não menos.

---

## Referências internas

- [`datasets/dataset_registry.csv`](../../datasets/dataset_registry.csv) — registro geral de datasets
- [`datasets/patch_corpus_registry.csv`](../../datasets/patch_corpus_registry.csv) — corpora de patches por estágio
- [`datasets/external_evidence_registry.csv`](../../datasets/external_evidence_registry.csv) — evidências externas por região
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](patch_lineage_and_grounding.md) — linhagem territorial dos patches
- [`docs/estado_metodologico_revp.md`](../estado_metodologico_revp.md) — estado e limitações metodológicas
- [`manifests/external_validation/`](../../manifests/external_validation/) — pacote de validação externa
