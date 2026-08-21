# REV-P v2fg — DINOv2 como camada de governança da API (não como variável preditiva)

**Status**: implementado e testado. Complementa `PLANO_ACAO_produto_v1.md`,
`revp_fase1_conclusao_dino_ab_test.md`, `revp_fase2_decisoes_design_contrato.md` e
`dino_sentinel_embedding_protocol.md`. **Não altera o modelo estatístico de nenhuma região.**

## 1. A regra que a camada existe para respeitar

O teste A/B da Fase 1 (`revp_fase1_conclusao_dino_ab_test.md`) já havia decidido, com LRT e
erro-padrão cluster-robusto por patch, que o embedding DINOv2 **não entra no modelo de Firth**.
O que faltava era um lugar legítimo para ele. Esta camada é esse lugar: governança —
validação de domínio, similaridade territorial e auditoria.

Invariantes verificados por teste (`tests/test_revp_v2fg_dinov2_governance_layer.py`):

* `score.value`, `score.confidence_interval` e `features_used` são **idênticos** com e sem
  evidência visual na requisição;
* `dino_governance.affects_score` é `False` por construção do schema (Pydantic recusa `True`);
* `ScoreResponse` ganhou **exatamente um** campo novo (`dino_governance`) — o bloco físico do
  contrato é bit-a-bit o mesmo.

## 2. Fluxo

```
POST /score
  │
  ├─ gates.py  ──────────────►  região, features físicas, status        (inalterado)
  ├─ engine_bridge ─────────►  Firth v12 + bootstrap → score, CI        (inalterado)
  │
  └─ dino_governance_bridge  →  Dinov2GovernanceEngine
         │                          │
         │  resolve o embedding     ├─ cosseno contra TODOS os medoids regionais
         │  de consulta:            ├─ medoid territorial mais próximo → suggested_region
         │   1. visual_patch_id     ├─ gate OOD: cos < limiar → out_of_domain
         │   2. bbox do patch       ├─ requested_region ≠ suggested_region → mismatch
         │      Sentinel (privado)  └─ bloco de auditoria (manifesto, contagens, base do limiar)
         │   3. nada → no_visual_evidence
         │
         └─► response.dino_governance   (nunca None, nunca silencioso)
```

Os cinco estados possíveis são todos explícitos e auditáveis: `in_domain`, `out_of_domain`,
`no_visual_evidence`, `invalid_embedding`, `governance_unavailable`. Quando o gate OOD dispara
ou há divergência territorial, a resposta **também** ganha uma linha em `limitations` —
o score continua sendo entregue, marcado para revisão.

## 3. Componentes

| Arquivo | Papel |
|---|---|
| `scripts/dino/revp_v2fg_dinov2_embedder.py` | `Dinov2Embedder`: `facebook/dinov2-with-registers-base`, 768D, L2, CPU/GPU. Fail-closed sem pesos locais. |
| `scripts/dino/revp_v2fg_dinov2_governance_engine.py` | `Dinov2GovernanceEngine`: cosseno, medoid territorial, gate OOD, auditoria. |
| `scripts/dino/revp_v2fg_build_dinov2_governance_corpus.py` | E2/E3: valida embeddings reais, calcula medoids, escreve manifesto. |
| `outputs_public/data/susc_20e_api_contrato_inferencia_recife/scripts/dino_governance_bridge.py` | Resolve o embedding da requisição e chama a engine. |
| `.../contract_schema.py` | `DinoGovernance`, `DinoMedoidSimilarity`, `DinoGovernanceAudit`. |

## 4. Corpus e medoids — de onde vêm os números

Construídos **só** a partir dos embeddings já persistidos em `datasets/dino_*embedding*.csv`
(gerados pelos executores v1qj/v1qw/v1qy/v1r0–v1r8, backbone `dinov2-with-registers-base`,
768D, L2-normalizados). Nada é gerado nesta etapa — apenas lido, validado e indexado.

Resultado real da execução (`datasets/dinov2_governance_summary_v2fg.csv`):

| | |
|---|---|
| fontes reais lidas | 8 |
| candidatos | 134 |
| válidos | 97 |
| bloqueados | 37 (todos `DUPLICATA_IDENTICA_DE_PATCH_JA_ACEITO`) |
| por região | RECIFE 52, CURITIBA 24, PET 21 |

Medoids (definição idêntica à de `outputs_public/tables/table_dino_medoids.csv`: maior
similaridade de cosseno média dentro do recorte):

| Recorte | Medoid | n | cos médio |
|---|---|---|---|
| CURITIBA | `CUR_00402` | 24 | 0,870973 |
| PET | `PET_00614` | 21 | 0,810092 |
| RECIFE | `REC_00292` | 52 | 0,636340 |
| CORPUS | `REC_00529` | 97 | 0,712836 |

## 5. Limiar OOD — derivado, não arbitrado

`threshold_default = 0,305554` = percentil 5% da similaridade de cada embedding válido do
corpus ao medoid regional mais próximo (n=97; min 0,2079 / p50 0,8422 / p95 0,9154 / max 1,0).
A base de cálculo fica escrita no próprio manifesto (`ood_gate.threshold_basis`) e aparece na
auditoria de toda resposta. Precedência de configuração: argumento do construtor →
`REVP_DINOV2_OOD_THRESHOLD` → manifesto.

## 6. O que a similaridade territorial vale (leitura honesta)

**Concordância territorial medida no próprio corpus: 65/97 = 0,6701.** Um terço dos patches
reais tem como medoid mais próximo o de outra região. Portanto:

> `suggested_region` é evidência estrutural fraca sobre território, **não** um classificador
> regional. `territorial_match = "mismatch"` é sinal para revisão humana, nunca veredito, e
> nunca invalida o score físico.

Esse número está gravado em `diagnostics.territorial_concordance` do manifesto e é exposto em
`dino_governance.audit.territorial_concordance_in_corpus` — quem lê a resposta vê o quanto o
sinal vale.

## 7. Reprodutibilidade contra as tabelas públicas (divergência real, registrada)

O manifesto compara o corpus atual com `outputs_public/tables/table_dino_similarity_matrix.csv`
(11 patches em comum, 110 pares): **max |Δ| = 0,11934, média |Δ| = 0,047523**. A divergência se
concentra nos patches de Recife. As tabelas públicas foram produzidas por uma rodada local
(`local_runs/`, não persistida no repositório), então os vetores hoje em `datasets/` não
reproduzem bit-a-bit aquela rodada. Isso está registrado em
`cross_check_published` — nenhuma tabela pública foi alterada e nenhum número foi ajustado
para casar.

Consequência prática: os medoids de v2fg (`CUR_00402`, `PET_00614`, `REC_00292`) **não** são os
mesmos de `table_dino_medoids.csv` (`CUR_00038`, `PET_00104`, `REC_00205`) — recortes diferentes
(97 patches vs. 12) e vetores de rodadas diferentes. Ambos ficam rastreáveis lado a lado.

## 8. Mock

`Dinov2Embedder` tem um backend determinístico derivado do SHA-256 do arquivo. Ele exige
opt-in explícito (`mock=True` ou `REVP_DINOV2_ALLOW_MOCK=true`), marca todo vetor com
`backend="mock"`, e **nunca entra no corpus** — o pipeline v2fg só lê CSVs reais e bloqueia
qualquer linha com termo de fixture/mock. Sem pesos locais e sem `REVP_DINO_ALLOW_DOWNLOAD`,
o extrator devolve `None`, não um vetor sintético silencioso.

## 9. Como reconstruir os artefatos

```bash
python scripts/dino/revp_v2fg_build_dinov2_governance_corpus.py --dry-run   # confere sem escrever
python scripts/dino/revp_v2fg_build_dinov2_governance_corpus.py            # escreve os 4 artefatos
python -m pytest tests/test_revp_v2fg_dinov2_governance_layer.py -q
```

## 10. Fronteira metodológica (repetida porque é o ponto)

* DINOv2 não entra no modelo físico de Firth nem em seus coeficientes.
* Nenhum embedding, similaridade ou medoid é rótulo, classe, alvo, predição ou confirmação de
  evento observado.
* Nenhum gate desta camada bloqueia inferência em silêncio.
* Nenhuma URL de auditoria é emitida — a auditoria aponta para caminhos de arquivo reais do
  repositório.
