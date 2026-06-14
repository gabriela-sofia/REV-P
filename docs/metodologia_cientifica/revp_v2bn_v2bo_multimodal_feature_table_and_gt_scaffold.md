# v2bn / v2bo — Feature table multimodal e scaffold de ground truth

Versão: `v2bn`, `v2bo`
Modo: review-only. Não habilita treino, não cria label, não cria negativo.

Esta dupla de estágios retoma a trilha multimodal/ground truth no próximo
identificador livre do repositório (após `v2bm`), preparando o REV-P para a
próxima fase de IA sem forçar nenhum claim preditivo antes do ground truth.

## O que v2bn faz

`v2bn` constrói uma **feature table multimodal de prontidão para revisão**: uma
linha por entrada Sentinel, unindo

- o spine canônico de entrada (`v1fu`, 128 patches), que já carrega o
  `split_group` anti-vazamento (agrupado por região/asset, sem random split);
- o manifesto **real** de embeddings DINOv2 (`v1ge`, 12 vetores 768D
  congelados, 4 por região), referenciados por `dino_embedding_uri`,
  `dino_embedding_dim` e `dino_embedding_hash` — os vetores densos **nunca** são
  copiados para o CSV;
- flags de disponibilidade de GIS, evidência (registry v2at) e binding
  evento-patch (overlay v2au), tratadas apenas como disponibilidade, nunca
  promovidas a label.

`v2bn` **não habilita treino**. `allowed_for_training=False` para todas as
linhas; as colunas de ground truth ficam vazias/NA por desenho.

### Reconciliação histórica "0 vs 12 embeddings"

O estágio resolve, sem apagar nada, a divergência entre artefatos:

- `HISTORICAL_STALE_ZERO_EMBEDDINGS` — o registry fail-closed
  `dino_embedding_feature_store_registry_v1ph.csv` foi gravado vazio porque
  nenhum vetor denso foi parseado para sua tabela. Isso é um artefato de
  *escopo de parsing*, não uma afirmação de que embeddings não existem.
- `LOCAL_MANIFEST_AVAILABLE` / `PUBLIC_FINAL_REPORT_ONLY` — o manifesto local
  `v1ge` e os relatórios públicos finais registram 12 embeddings reais 768D.

Não há contradição: o "0" e o "12" descrevem escopos diferentes. A reconciliação
fica registrada em `multimodal_embedding_inventory_v2bn.csv` e no relatório.

## O que v2bo faz

`v2bo` prepara o **protocolo de label sem criar label**. Emite:

- `gt_patch_registry_scaffold_v2bo.csv` — uma linha por patch candidato, com
  todas as colunas de label vazias/NA e `human_review_required=True`;
- `gt_label_policy_v2bo.json` — o que pode (e como) virar label; embeddings,
  proxy GIS, coerência contextual e metadados **não** podem virar label;
- `gt_negative_policy_v2bo.json` — ausência de evidência não é negativo;
  pseudo-ausência, fundo aleatório e distância de âncora não são negativos
  formais; matched negatives só com critério formal e evidência comparável;
  unknown permanece unknown;
- `gt_training_gate_v2bo.json` — gate bloqueado.

`v2bo` **não cria labels**. Quando o gate for liberado por ground truth
auditável, os primeiros modelos devem ser baselines leves sobre embeddings
congelados (Logistic Regression, Random Forest, HistGradientBoosting/XGBoost,
MLP raso), validados com grupos/blocks — nunca com random split simples.

## Posição metodológica

- DINOv2 continua **frozen**; sem fine-tuning, sem early fusion, sem
  pixel-space fusion. A multimodalidade aqui é **prontidão em nível de
  feature**, não predição.
- O gargalo permanece ground truth, binding evento-patch, QA, negativos formais,
  anti-leakage e rastreabilidade — não o modelo.
- Não se treina modelo supervisionado enquanto `labels_created=false`,
  `formal_negative_count=0` e `can_train_supervised_model=false`.

## Outputs

- v2bn: `local_runs/multimodal/v2bn/`
- v2bo: `local_runs/ground_truth/v2bo/`

Todos os outputs são leves (`.csv`, `.json`, `.md`) e ficam apenas em
`local_runs/`. Nenhum dado bruto, embedding denso ou checkpoint é versionado.
