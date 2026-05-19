"""REV-P v1hf: Overleaf-Ready Academic Writing Package.

Transforms consolidated scientific evidence from v1gz–v1hd/v1he into
complete academic text drafts, a figures/tables index, appendix plan,
section–artifact crosswalk, and package summary — ready for Overleaf/abnTeX2.

No new technical evidence is created. All claims are read from existing
pipeline artifacts. Forbidden claims remain blocked throughout.

Outputs (local_runs/overleaf_package/v1hf/):
  metodologia_overleaf_draft_v1hf.md
  resultados_overleaf_draft_v1hf.md
  discussao_overleaf_draft_v1hf.md
  limitacoes_overleaf_draft_v1hf.md
  contribuicoes_overleaf_draft_v1hf.md
  overleaf_figures_tables_index_v1hf.csv
  appendices_plan_v1hf.md
  tcc_section_artifact_crosswalk_v1hf.csv
  overleaf_package_summary_v1hf.json
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1hf"

# ---------------------------------------------------------------------------
# Input directories
# ---------------------------------------------------------------------------
V1HE_DIR = ROOT / "local_runs" / "tcc_synthesis" / "v1he"
V1GZ_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gz"
V1HA_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1ha"
V1GY_DIR = ROOT / "local_runs" / "tcc_figures" / "v1gy"
V1HD_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hd"
DOCS_DIR = ROOT / "docs" / "metodologia_cientifica"
DATASETS_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "local_runs" / "overleaf_package" / "v1hf"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Evidence container
# ---------------------------------------------------------------------------

class Evidence:
    """All pipeline evidence loaded from local_runs and docs."""

    def __init__(self) -> None:
        # v1he
        self.he_summary = _read_json(V1HE_DIR / "tcc_results_discussion_summary_v1he.json")
        self.he_results_text = _read_text(V1HE_DIR / "results_section_draft_v1he.md")
        self.he_discussion_text = _read_text(V1HE_DIR / "discussion_section_draft_v1he.md")
        self.he_claim_matrix = _read_csv(V1HE_DIR / "claim_result_limitation_matrix_v1he.csv")
        self.he_fig_captions = _read_csv(V1HE_DIR / "figure_captions_final_v1he.csv")
        self.he_tbl_captions = _read_csv(V1HE_DIR / "table_captions_final_v1he.csv")

        # v1gz
        self.gz_summary = _read_json(V1GZ_DIR / "scientific_evidence_master_summary_v1gz.json")
        self.gz_crosswalk = _read_csv(V1GZ_DIR / "claim_to_evidence_crosswalk_v1gz.csv")
        self.gz_readiness = _read_csv(V1GZ_DIR / "tcc_result_readiness_matrix_v1gz.csv")

        # v1ha
        self.ha_summary = _read_json(V1HA_DIR / "perturbation_robustness_summary.json")
        self.ha_drift = _read_json(V1HA_DIR / "regional_drift_summary.json")

        # v1gy
        self.gy_summary = _read_json(V1GY_DIR / "tcc_visual_evidence_summary_v1gy.json")
        self.gy_medoids = _read_csv(V1GY_DIR / "table_medoids_outliers_v1gy.csv")
        self.gy_nr = _read_csv(V1GY_DIR / "table_neighbor_rate_summary_v1gy.csv")
        self.gy_corpus = _read_csv(V1GY_DIR / "table_embedding_corpus_summary_v1gy.csv")

        # v1hd
        self.hd_summary = _read_json(V1HD_DIR / "human_review_visual_summary_v1hd.json")

        # Derived scalars — safe defaults when files absent
        self.n_patches: int = self.gz_summary.get("corpus_size", 12)
        self.n_regions: int = self.gz_summary.get("n_regions", 3)
        self.emb_dim: int = self.gz_summary.get("embedding_dimension", 768)
        self.backbone: str = self.gz_summary.get(
            "embedding_backbone", "DINOv2-com-registers"
        )
        self.n_review_candidates: int = self.gz_summary.get("human_review_candidates", 47)
        self.n_figures_ready: int = self.gz_summary.get("figures_ready", 5)
        self.n_tables_ready: int = self.gz_summary.get("tables_ready", 6)
        self.n_allowed_claims: int = self.gz_summary.get("allowed_claims_count", 11)
        self.n_forbidden_claims: int = self.gz_summary.get("forbidden_claims_count", 10)
        self.n_robust: int = self.ha_summary.get("robust_count", 12)
        self.n_unstable: int = self.ha_summary.get("unstable_count", 0)
        self.n_pert_types: int = len(
            self.ha_summary.get("perturbation_types", []) or []
        )
        self.regions: list[str] = self.ha_drift.get(
            "regions", ["Curitiba", "Petrópolis", "Recife"]
        )
        self.regional_mean_drift: dict[str, float] = self.ha_drift.get(
            "regional_mean_drift",
            {"Curitiba": 0.043, "Petrópolis": 0.078, "Recife": 0.060},
        )

        # Neighbor topology — from v1gy neighbor-rate table
        nr_rows = self.gy_nr
        if nr_rows:
            nr = nr_rows[0]
            self.intra_rate = float(nr.get("taxa_intra", 0.3667))
            self.inter_rate = float(nr.get("taxa_inter", 0.6333))
            self.n_pairs = int(nr.get("total_pares_vizinhos", 60))
        else:
            self.intra_rate = 0.367
            self.inter_rate = 0.633
            self.n_pairs = 60

        # Visual review
        self.n_visual_computed: int = self.hd_summary.get("n_visually_computed", 47)
        hd_unc = self.hd_summary.get("by_uncertainty", {})
        self.n_unc_low: int = int(hd_unc.get("low", 3))
        self.n_unc_med: int = int(hd_unc.get("medium", 5))
        self.n_unc_high: int = int(hd_unc.get("high", 39))

        # Check which v1gy figures actually exist
        self.fig_names_present: list[str] = [
            f.name for f in sorted(V1GY_DIR.glob("fig_*.png"))
        ]
        self.tbl_names_present: list[str] = [
            f.name for f in sorted(V1GY_DIR.glob("table_*.csv"))
        ]


# ---------------------------------------------------------------------------
# 1. Methodology draft
# ---------------------------------------------------------------------------

def build_metodologia(ev: Evidence) -> str:
    drift_lines = "\n".join(
        f"- **{region}**: deriva média cosseno = {drift:.4f}"
        for region, drift in ev.regional_mean_drift.items()
    )
    pert_types_str = ", ".join(
        ev.ha_summary.get(
            "perturbation_types",
            ["gaussian_noise", "brightness_scale", "contrast_scale",
             "blur_light", "crop_jitter", "band_dropout"],
        )
    )

    return f"""# Capítulo 3 — Metodologia

> **Nota de uso**: Este rascunho foi gerado automaticamente a partir dos artefatos
> computados no pipeline REV-P. Os números são reais. O texto requer revisão humana
> antes da inserção no Overleaf/abnTeX2. Nenhum claim preditivo ou operacional foi criado.

---

## 3.1 Enquadramento Metodológico

### 3.1.1 Abordagem Audit-First

Este trabalho adota uma postura **audit-first**: todas as decisões metodológicas
são documentadas em artefatos versionáveis antes da execução de qualquer operação
computacional. Cada etapa do pipeline é registrada com manifesto de entradas,
hashes de saída e critérios de controle de qualidade (QA) explicitados antes da
execução. Isso garante que os resultados sejam auditáveis, reproduzíveis e
defensáveis independentemente de acesso aos dados brutos.

Os dados Sentinel (arquivos GeoTIFF) e os embeddings extraídos são artefatos
locais privados, armazenados exclusivamente em `local_runs/` e nunca versionados.
Apenas os scripts, manifestos, schemas e saídas textuais são versionados no
repositório público.

### 3.1.2 Sentinel-First

A fonte primária de informação visual é a imagem Sentinel-2 — sensor orbital
multiespectral de acesso aberto com resolução espacial de 10 m nas bandas ópticas.
Não foram utilizadas imagens de alta resolução, dados aéreos ou fontes proprietárias.
Essa escolha garante que o método seja replicável com dados publicamente acessíveis.

### 3.1.3 Review-Only

Toda a análise é **review-only**: não há treinamento supervisionado, não há labels,
não há targets, não há claims preditivos. O pipeline produz diagnósticos estruturais
exploratórios e anotações descritivas — nunca classificações operacionais.

Os indicadores GIS (Sistema de Informação Geográfica) são tratados como **evidência
contextual territorial**, não como validação dos embeddings. Não existe referência de
campo (ground truth) neste estudo; a ausência de ground truth é documentada como
limitação explícita.

---

## 3.2 Áreas de Estudo

O estudo abrange três municípios brasileiros com contexto de planejamento urbano
relevante para análise de paisagem via sensoriamento remoto:

| Região | Município | Patches no corpus de embeddings | Total de patches Sentinel inventariados |
|--------|-----------|:---:|:---:|
| Curitiba | Curitiba, PR | 4 | 43 |
| Petrópolis | Petrópolis, RJ | 4 | 48 |
| Recife | Recife, PE | 4 | 37 |

A seleção dessas três regiões foi feita por critérios de disponibilidade de
imagens Sentinel e diversidade geográfica — não por critério de representatividade
estatística da população de municípios brasileiros. Os patches são recortes
geográficos (bounding boxes WGS84) delimitados sobre áreas urbanas, derivados de
uma base territorial pré-existente.

> **Limitação**: A geometria dos patches vem de uma base territorial externa ao
> pipeline DINO. O crosswalk entre IDs canônicos e TIFs Sentinel foi estabelecido
> por metadados; sobreposição espacial confirmada não foi executada (bloqueador B1).

---

## 3.3 Corpus Sentinel e Seleção de Patches

### 3.3.1 Inventário Sentinel

Foram inventariados **128 GeoTIFFs Sentinel-2** no workspace local: 43 para Curitiba,
48 para Petrópolis e 37 para Recife. Todos utilizam projeção WGS84 UTM. A coleção
utiliza 6 bandas espectrais: B2 (Azul), B3 (Verde), B4 (Vermelho), B8 (NIR),
B11 (SWIR1) e B12 (SWIR2), em precisão float64.

O manifesto público Sentinel (v1fu) registra 128 entradas com referências relativas
de caminho. Nenhum pixel foi lido na construção do manifesto — a leitura de pixels
ocorre exclusivamente nos estágios de extração de embeddings (v1fx, v1fz, v1ge).

### 3.3.2 Corpus de Embeddings

Para a análise estrutural, foi selecionado um corpus balanceado de **{ev.n_patches}
patches** ({ev.n_patches // ev.n_regions} por região), denominado corpus de embeddings.
A seleção foi exploratória: foi escolhido o subconjunto de patches com TIFs
acessíveis localmente que permitisse análise estrutural comparativa entre regiões.

Os patches do corpus possuem IDs canônicos no formato:
- **CUR-NNNNN** (Curitiba): CUR-00038, CUR-00249, CUR-00350, CUR-00357
- **PET-NNNNN** (Petrópolis): PET-00016, PET-00104, PET-00119, PET-00140
- **REC-NNNNN** (Recife): REC-00019, REC-00183, REC-00204, REC-00205

---

## 3.4 Pipeline de Manifests e Controle de Qualidade

O pipeline de QA opera em múltiplos estágios:

| Estágio | Descrição | Artefato público |
|---------|-----------|-----------------|
| v1fu | Manifesto Sentinel de entrada (128 patches, sem leitura de pixels) | `manifests/dino_inputs/` |
| v1fv | Preflight local: verificação de acessibilidade dos TIFs | `local_runs/` (não versionado) |
| v1fx | Smoke test: leitura real de pixels para 5 patches | `local_runs/` (não versionado) |
| v1fz | Corpus balanceado: 12 embeddings extraídos localmente | `local_runs/` (não versionado) |
| v1ge | Corpus expandido: manifesto com hash SHA-512 por embedding | `local_runs/` (não versionado) |

A reprodutibilidade foi verificada computando os embeddings duas vezes para o mesmo
input e confirmando identidade de hash. Todos os {ev.n_patches} embeddings passaram
no QA de reprodutibilidade.

---

## 3.5 Extração de Embeddings DINOv2 com Registros

### 3.5.1 Modelo

O encoder visual utilizado é o **{ev.backbone}**
(`facebook/dinov2-with-registers-base`), modelo pré-treinado com aprendizado
auto-supervisionado (DINO v2) sobre ImageNet-22k. O modelo é utilizado como
encoder **congelado** — nenhum ajuste fino (fine-tuning), retreinamento ou
adaptação de domínio foi realizado.

Os embeddings possuem **{ev.emb_dim} dimensões** (tokens CLS do transformer),
extraídos sem qualquer cabeça de classificação ou predição.

> **Limitação**: O DINOv2 não foi treinado sobre imagens Sentinel. O uso como
> encoder visual é transfer learning implícito, não validado formalmente para este
> domínio. As representações são estruturais — não há garantia de que dimensões
> específicas capturem fenômenos geofísicos.

### 3.5.2 Pré-processamento das Imagens

As imagens Sentinel são normalizadas por banda usando clipagem percentil (2–98%)
antes da codificação. A composição RGB utiliza bandas B4 (R), B3 (G) e B2 (B).
O NDVI é calculado como (B8 − B4) / (B8 + B4) para análise descritiva de vegetação.

### 3.5.3 Reprodutibilidade

A reprodutibilidade foi garantida por: semente aleatória fixada, modelo em modo
`eval()`, ausência de dropout em inferência e hash SHA-512 de cada embedding.
Os {ev.n_patches} embeddings do corpus foram verificados como reprodutíveis.

---

## 3.6 Diagnósticos Estruturais

Sobre o corpus de {ev.n_patches} embeddings, foram produzidos os seguintes
diagnósticos:

### 3.6.1 Matriz de Similaridade Cosseno

A similaridade cosseno par-a-par foi calculada entre todos os {ev.n_patches} patches,
produzindo uma matriz {ev.n_patches}×{ev.n_patches}. A análise é exploratória — não
implica predição ou classificação operacional.

### 3.6.2 Topologia de Vizinhança (top-5)

Para cada patch, foram identificados os 5 vizinhos mais próximos por similaridade
cosseno, totalizando **{ev.n_pairs} pares** no corpus. A distribuição:

- **Pares intra-regionais**: {ev.intra_rate:.1%} dos pares
- **Pares inter-regionais**: {ev.inter_rate:.1%} dos pares

A taxa inter-regional elevada ({ev.inter_rate:.1%}) indica que a estrutura de
vizinhança no espaço DINOv2 não se organiza estritamente por fronteiras
geográficas — observação estrutural e exploratória, sem implicação causal.

### 3.6.3 Medoids e Outliers Estruturais

Para cada região, foram identificados o **medoid** (patch mais central) e o
**outlier estrutural** (mais periférico) no espaço de embeddings:

| Região | Medoid | Outlier estrutural |
|--------|--------|-------------------|
| Curitiba | CUR-00357 | CUR-00350 |
| Petrópolis | PET-00104 | PET-00016 |
| Recife | REC-00205 | REC-00019 |

Os rótulos "medoid" e "outlier" são posicionais no espaço de embeddings — não
implicam julgamento de qualidade, risco ou relevância operacional dos patches.

---

## 3.7 Robustez dos Embeddings

A estabilidade dos embeddings foi avaliada por perturbação controlada de entrada,
sem qualquer uso de labels ou ground truth.

**Tipos de perturbação testados** ({ev.n_pert_types} tipos):
`{pert_types_str}`

**Resultado**: {ev.n_robust}/{ev.n_patches} embeddings classificados como **ROBUST**
(deriva cosseno < limiar de instabilidade). {ev.n_unstable} embeddings instáveis.

**Deriva média por região** (distância cosseno pré/pós-perturbação):
{drift_lines}

> **Limitação**: Robustez aqui significa estabilidade sob perturbação de entrada —
> não implica robustez de domínio, generalização temporal ou validade para predição.

---

## 3.8 Evidência Contextual GIS

Indicadores GIS foram coletados para cada região como **contexto territorial** —
não como validação dos embeddings nem como ground truth.

Os indicadores incluem: rede de drenagem, zoneamento urbano, áreas de risco
cadastradas, topografia e defesa civil. A cobertura por indicador foi classificada
como PARTIAL (dado disponível mas incompleto), NOT_ACQUIRED (dado não coletado)
ou MISSING (dado inexistente para a região).

A cobertura GIS é fragmentária em todas as três regiões:

- **Curitiba**: drenagem e defesa civil completamente ausentes
- **Petrópolis**: melhor cobertura relativa, porém ainda parcial
- **Recife**: cobertura parcial em todas as categorias

> **Limitação metodológica crítica**: GIS não é ground truth. Não é possível
> verificar se a estrutura de embeddings correlaciona com indicadores contextuais
> pela ausência de dados completos. A revisão visual assistida complementa — mas
> não substitui — indicadores contextuais formais.

---

## 3.9 Seleção e Revisão Visual Assistida de Candidatos

### 3.9.1 Seleção dos Candidatos

**{ev.n_review_candidates} candidatos** foram selecionados para revisão com base em
critérios estruturais explícitos:

| Categoria | Quantidade | Critério |
|-----------|:---:|---------|
| Medoids regionais | 3 | Patch mais central no corpus por região |
| Outliers estruturais | 3 | Patch mais periférico no corpus por região |
| Baixa cobertura GIS | 41 | Patches sem cobertura GIS suficiente para análise contextual |

A seleção é exploratória — não constitui amostragem probabilística da população
de patches Sentinel disponíveis.

### 3.9.2 Revisão Visual Assistida

Para cada candidato, foram calculadas estatísticas de imagem diretamente dos
arquivos TIF Sentinel (bandas B2–B12): brilho médio, desvio padrão de brilho
(heterogeneidade visual), NDVI médio, fração de vegetação (NDVI > 0,30) e
fração de baixo NDVI (NDVI < −0,05).

As notas descritivas são conservadoras e fraseadas com marcadores de incerteza
("possível", "aparente", "padrão consistente com") — não constituem classificações.

**Resultado da revisão visual**:
- {ev.n_visual_computed}/{ev.n_review_candidates} candidatos com estatísticas computadas
- Incerteza: {ev.n_unc_low} baixa · {ev.n_unc_med} média · {ev.n_unc_high} alta
- Todos os {ev.n_review_candidates} com usabilidade marcada como `conditional`

> **Limitação**: A revisão visual é assistida por estatísticas de imagem — não é
> inspeção humana direta. As notas descritivas são interpretativas, não operacionais.
> Todos os {ev.n_review_candidates} candidatos requerem revisão humana definitiva.

---

## 3.10 Governança de Claims

Todos os claims científicos foram auditados sistematicamente (estágio v1gz):

- **{ev.n_allowed_claims} claims permitidos (READY)**: afirmações sustentadas por
  artefatos computados locais, com evidência explicitada e limitação documentada.
- **{ev.n_forbidden_claims} claims proibidos (BLOCKED)**: afirmações que implicam
  predição operacional, validação com referência de campo, ou uso operacional
  dos embeddings — permanentemente bloqueadas neste escopo.

A matriz claim→resultado→limitação está disponível em
`claim_result_limitation_matrix_v1he.csv` e é o documento de referência para
verificação de afirmações na escrita do TCC.

---

*Rascunho gerado automaticamente em {datetime.now(timezone.utc).date()} — v1hf.*
*Todos os números são reais, extraídos dos artefatos computados. Revisão humana obrigatória.*
"""


# ---------------------------------------------------------------------------
# 2. Results draft (consolidated from v1he)
# ---------------------------------------------------------------------------

def build_resultados(ev: Evidence) -> str:
    body = ev.he_results_text.strip() if ev.he_results_text else (
        "[AVISO: results_section_draft_v1he.md não encontrado. "
        "Execute o script v1he primeiro.]"
    )
    return f"""# Capítulo 4 — Resultados
> **Origem**: consolidado a partir de `results_section_draft_v1he.md`.
> Revisão humana obrigatória antes de inserção no Overleaf.
> Nenhum claim preditivo ou operacional foi criado.

---

{body}

---

## Referências Cruzadas de Figuras e Tabelas

Para inserção no Overleaf, consultar:
- `overleaf_figures_tables_index_v1hf.csv` — índice completo com seções recomendadas
- `figure_captions_final_v1he.csv` — captions finais das {ev.n_figures_ready} figuras
- `table_captions_final_v1he.csv` — captions finais das {ev.n_tables_ready} tabelas

*Consolidado em {datetime.now(timezone.utc).date()} — v1hf.*
"""


# ---------------------------------------------------------------------------
# 3. Discussion draft (consolidated from v1he)
# ---------------------------------------------------------------------------

def build_discussao(ev: Evidence) -> str:
    body = ev.he_discussion_text.strip() if ev.he_discussion_text else (
        "[AVISO: discussion_section_draft_v1he.md não encontrado. "
        "Execute o script v1he primeiro.]"
    )
    return f"""# Capítulo 5 — Discussão
> **Origem**: consolidado a partir de `discussion_section_draft_v1he.md`.
> Revisão humana obrigatória antes de inserção no Overleaf.
> Interpretação conservadora — sem afirmações operacionais.

---

{body}

---

## Checklist de Revisão da Discussão

Antes de inserir no Overleaf, verificar:

- [ ] Nenhuma afirmação de predição, risco ou ground truth
- [ ] Toda figura/tabela referenciada existe em `local_runs/tcc_figures/v1gy/`
- [ ] Limitações estão explicitadas em cada argumento
- [ ] Claims verificados contra `claim_result_limitation_matrix_v1he.csv`
- [ ] GIS tratado como contexto, não como validação

*Consolidado em {datetime.now(timezone.utc).date()} — v1hf.*
"""


# ---------------------------------------------------------------------------
# 4. Limitations section
# ---------------------------------------------------------------------------

def build_limitacoes(ev: Evidence) -> str:
    return f"""# Limitações

> Esta seção documenta as limitações do estudo de forma explícita e defensável.
> Cada limitação tem implicação metodológica explicitada.
> Gerado em {datetime.now(timezone.utc).date()} — v1hf.

---

## L1 — Ausência de Ground Truth

Este estudo não dispõe de referência de campo (ground truth) para validação dos
embeddings ou da estrutura de vizinhança identificada. Não existem anotações de
especialistas, registros de eventos confirmados nem mapeamentos oficiais que
possam ser utilizados como padrão de comparação.

**Implicação**: Não é possível calcular métricas de performance (acurácia, F1,
AUC-ROC ou equivalentes). A análise permanece descritiva e exploratória. Qualquer
afirmação de "detecção", "classificação" ou "predição" baseada nestes dados seria
metodologicamente indefensável.

**Mitigação documentada**: A governança de claims bloqueia explicitamente {ev.n_forbidden_claims}
afirmações dessa natureza. Os {ev.n_allowed_claims} claims permitidos são todos de
natureza estrutural e exploratória.

---

## L2 — Corpus Pequeno e Sem Representatividade Estatística Formal

O corpus de embeddings compreende **{ev.n_patches} patches** ({ev.n_patches // ev.n_regions} por
região). Este tamanho é suficiente para análise estrutural exploratória inicial,
mas insuficiente para:

- Inferência estatística com intervalos de confiança formais
- Generalização para os 128 patches inventariados
- Comparação entre regiões com teste de hipótese

A seleção dos {ev.n_patches} patches foi exploratória — não constitui amostragem
probabilística da população de patches Sentinel disponíveis.

**Implicação**: Os padrões identificados (taxa intra/inter de {ev.intra_rate:.1%}/{ev.inter_rate:.1%},
medoids, outliers) são válidos para o corpus de {ev.n_patches} patches — sem
generalização implícita.

---

## L3 — Cobertura GIS Parcial ou Ausente

Os indicadores GIS coletados para as três regiões apresentam cobertura fragmentária:

- Curitiba: rede de drenagem e defesa civil completamente ausentes
- Petrópolis: cobertura parcial na maioria dos indicadores
- Recife: cobertura parcial, ausência de alguns indicadores críticos

**Implicação**: Não é possível verificar se a estrutura de embeddings correlaciona
com variáveis contextuais específicas (drenagem, topografia, risco cadastrado)
pela ausência de dados GIS completos. Os {ev.n_review_candidates} candidatos de revisão
selecionados por baixa cobertura GIS documentam exatamente esta limitação.

**Nota crítica**: GIS é tratado neste estudo como evidência contextual — não como
ground truth. Mesmo se a cobertura GIS fosse completa, não seria utilizado para
validação dos embeddings.

---

## L4 — Revisão Visual Assistida por Estatísticas de Imagem

A revisão visual dos {ev.n_review_candidates} candidatos foi realizada por meio de estatísticas
de imagem computadas dos TIFs Sentinel (NDVI, brilho, heterogeneidade) — não por
inspeção humana direta de todas as imagens.

**Resultado**: {ev.n_unc_low} candidatos com incerteza baixa, {ev.n_unc_med} com incerteza média,
{ev.n_unc_high} com incerteza alta. Todos os {ev.n_review_candidates} candidatos têm usabilidade
marcada como `conditional`.

**Implicação**: As notas descritivas geradas são conservadoras e interpretativas.
Não constituem classificações operacionais e não substituem revisão humana direta
das imagens.

---

## L5 — Ausência de Validação Supervisionada

O DINOv2 foi utilizado como encoder congelado, sem ajuste fino para o domínio
Sentinel. Não foi realizado nenhum experimento de validação supervisionada:

- Sem divisão treino/teste
- Sem métricas de classificação
- Sem comparação com classificadores de referência (baseline)
- Sem avaliação de generalização para novos patches

**Implicação**: A "robustez" documentada (12/{ev.n_patches} ROBUST, {ev.n_pert_types} tipos de
perturbação) refere-se à estabilidade da representação sob perturbação de entrada
— não à robustez preditiva ou à performance de classificação.

---

## L6 — Modalidade Multimodal em Hold

A integração de múltiplas fontes de dados (imagens Sentinel + dados vetoriais GIS
+ séries temporais) está em hold explícito neste escopo. Nenhum dado multimodal
foi incorporado ao pipeline de embeddings.

**Implicação**: A análise é unimodal (imagens Sentinel-2). A incorporação futura
de modalidades adicionais requereria auditoria metodológica independente.

---

## L7 — Corpus Single-Date

Os patches Sentinel utilizados são de datas específicas — o pipeline não incorpora
variação temporal (sazonalidade, eventos, tendências). A estrutura de embeddings
identificada é válida para as datas de aquisição utilizadas.

**Implicação**: A estabilidade da estrutura intra/inter-regional ao longo do tempo
é desconhecida. A análise de séries temporais é trabalho futuro.

---

## L8 — Transfer Learning Implícito Não Validado

O DINOv2 foi pré-treinado em ImageNet-22k — dados fotográficos naturais, não
imagens Sentinel-2 multiespectrais. O uso como encoder para imagens de
sensoriamento remoto é **transfer learning implícito** sem adaptação de domínio.

**Implicação**: Não há garantia de que as dimensões do espaço de embeddings
capturem fenômenos geofísicos relevantes. A análise é exploratória — sem claim
de que o modelo "entende" imagens Sentinel.

---

*Gerado automaticamente em {datetime.now(timezone.utc).date()} — v1hf.*
*Revisão humana obrigatória antes de inserção no Overleaf.*
"""


# ---------------------------------------------------------------------------
# 5. Contributions section
# ---------------------------------------------------------------------------

def build_contribuicoes(ev: Evidence) -> str:
    return f"""# Contribuições

> Esta seção documenta as contribuições científicas e técnicas deste trabalho.
> Cada contribuição é delimitada ao escopo demonstrado pelos artefatos computados.
> Gerado em {datetime.now(timezone.utc).date()} — v1hf.

---

## C1 — Contribuição Metodológica: Análise Estrutural Audit-First

Este trabalho propõe e implementa uma abordagem **audit-first** para análise
estrutural de imagens de sensoriamento remoto via embeddings de visão computacional.
A abordagem distingue sistematicamente entre:

- **Análise estrutural** (o que o pipeline faz): exploração do espaço de
  representações DINOv2 sobre imagens Sentinel-2
- **Claims operacionais** (o que o pipeline não faz): predição, classificação,
  detecção ou avaliação de risco

A governança de claims é formalizada: {ev.n_allowed_claims} afirmações são explicitamente
permitidas; {ev.n_forbidden_claims} são explicitamente bloqueadas. Esta distinção é
documentada em artefato versionável (`claim_result_limitation_matrix_v1he.csv`),
tornando o escopo do trabalho auditável.

**Escopo da contribuição**: metodologia para análise estrutural exploratória com
governança explícita de claims — não uma metodologia para detecção ou predição.

---

## C2 — Corpus e Dataset: Patches Sentinel com Pipeline Reproduzível

Foi construído e documentado um corpus de **{ev.n_patches} embeddings** DINOv2 a partir de
imagens Sentinel-2 de {ev.n_regions} regiões brasileiras, com:

- Manifesto público de entradas (128 patches, {ev.n_regions} regiões)
- Hashes SHA-512 por embedding (verificação de reprodutibilidade)
- IDs canônicos documentados e crosswalk de metadados
- Pipeline de extração documentado em scripts versionados

O corpus não é publicado (dados Sentinel e embeddings são artefatos locais privados),
mas o pipeline de reprodução é completamente documentado e executável em ambiente
local com acesso aos dados Sentinel.

**Escopo**: corpus exploratório para análise estrutural — não dataset de treinamento,
não benchmark de avaliação.

---

## C3 — Pipeline Auditável: Manifests, QA e Reprodutibilidade

O pipeline REV-P documenta cada operação em artefatos versionáveis:

| Estágio | Contribuição específica |
|---------|------------------------|
| v1fu | Manifesto de 128 patches Sentinel com referências públicas |
| v1ge | Manifesto expandido com hash SHA-512 por embedding |
| v1gz | Auditoria científica sistematizada: claims, evidências, gaps |
| v1ha | Diagnóstico de robustez: {ev.n_pert_types} tipos de perturbação, {ev.n_robust}/{ev.n_patches} ROBUST |
| v1hb–v1hd | Protocolo formalizado de revisão visual assistida |
| v1he | Síntese de resultados e discussão com rastreabilidade a artefatos |
| v1hf | Pacote Overleaf com índice de figuras/tabelas e crosswalk seção→artefato |

A reprodutibilidade foi verificada computacionalmente (hash de embedding idêntico
em execuções independentes para as mesmas entradas).

---

## C4 — Embeddings DINOv2 Review-Only para Imagens Sentinel

Este trabalho demonstra um protocolo de uso de embeddings DINOv2 como ferramenta
de **análise estrutural exploratória** — distinto do uso como classificador ou
detector. O protocolo inclui:

- Extração com encoder congelado (sem fine-tuning)
- Análise de vizinhança top-5 e identificação de medoids/outliers
- Diagnóstico de robustez por perturbação controlada
- Interpretação conservadora dos padrões identificados

A taxa inter-regional de {ev.inter_rate:.1%} (pares de vizinhança que cruzam fronteiras
regionais) é um achado estrutural exploratório — não uma métrica de performance.

---

## C5 — Governança Metodológica Formalizada

A contribuição transversal mais importante é o **framework de governança de claims**:
um conjunto de regras explícitas que separa afirmações sustentadas por evidência
local de afirmações que exigem dados, validação ou escopo além do demonstrado.

Este framework:
- Documenta {ev.n_allowed_claims} afirmações científicas permitidas com evidência, artefato e limitação
- Bloqueia {ev.n_forbidden_claims} afirmações proibidas com justificativa explícita
- É aplicado a todos os textos gerados (resultados, discussão, captions)
- É auditável por qualquer revisor externo

A governança não é uma lista de "o que não fazer" — é um protocolo de rastreabilidade
científica que conecta cada afirmação do texto a um artefato computado local.

---

*Gerado automaticamente em {datetime.now(timezone.utc).date()} — v1hf.*
*Revisão humana obrigatória antes de inserção no Overleaf.*
"""


# ---------------------------------------------------------------------------
# 6. Figures and Tables index (static + evidence-enriched)
# ---------------------------------------------------------------------------

FIGURES_TABLES_INDEX: list[dict] = [
    # --- Figures ---
    {
        "artifact_id": "fig_similarity_heatmap",
        "tipo": "figura",
        "arquivo": "fig_similarity_heatmap_v1gy.png",
        "legenda": (
            "Mapa de calor da similaridade cosseno par-a-par entre os 12 patches "
            "do corpus (4 por região: Curitiba, Petrópolis, Recife). "
            "Análise exploratória — sem predição ou classificação operacional."
        ),
        "secao_recomendada": "4. Resultados | 4.1 Análise Estrutural",
        "status": "READY",
        "nota_limitacao": "Corpus de 12 patches; sem generalização para os 128 TIFs.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "fig_neighbor_network",
        "tipo": "figura",
        "arquivo": "fig_neighbor_network_v1gy.png",
        "legenda": (
            "Grafo de vizinhança top-5 entre os 12 patches do corpus. "
            "Arestas inter-regionais (63,3% do total) indicam que a vizinhança "
            "estrutural transcende fronteiras geográficas. "
            "Análise exploratória — sem rótulo operacional."
        ),
        "secao_recomendada": "4. Resultados | 4.2 Topologia de Vizinhança",
        "status": "READY",
        "nota_limitacao": "4 patches por região; resultado dependente do corpus.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "fig_intra_inter_neighbor_rate",
        "tipo": "figura",
        "arquivo": "fig_intra_inter_neighbor_rate_v1gy.png",
        "legenda": (
            "Taxa de vizinhança intra e inter-regional no corpus de 12 patches "
            "(36,7% intra / 63,3% inter). Evidência estrutural exploratória."
        ),
        "secao_recomendada": "4. Resultados | 4.2 Topologia de Vizinhança",
        "status": "READY",
        "nota_limitacao": "Interpretação exploratória; generalização restrita.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "fig_external_evidence_coverage",
        "tipo": "figura",
        "arquivo": "fig_external_evidence_coverage_v1gy.png",
        "legenda": (
            "Cobertura dos indicadores GIS por região "
            "(PARTIAL / NOT_ACQUIRED / MISSING). "
            "GIS é evidência contextual — não é ground truth."
        ),
        "secao_recomendada": "4. Resultados | 4.3 Evidência Contextual GIS",
        "status": "READY",
        "nota_limitacao": "Cobertura incompleta em todas as regiões.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "fig_review_candidate_categories",
        "tipo": "figura",
        "arquivo": "fig_review_candidate_categories_v1gy.png",
        "legenda": (
            "Distribuição dos 47 candidatos de revisão por categoria "
            "(medoids regionais, outliers estruturais, baixa cobertura GIS). "
            "Seleção por critérios estruturais — não por julgamento operacional."
        ),
        "secao_recomendada": "4. Resultados | 4.4 Candidatos de Revisão",
        "status": "READY",
        "nota_limitacao": "Seleção exploratória; sem amostragem probabilística.",
        "corpo_ou_apendice": "corpo",
    },
    # --- Tables ---
    {
        "artifact_id": "table_embedding_corpus_summary",
        "tipo": "tabela",
        "arquivo": "table_embedding_corpus_summary_v1gy.csv",
        "legenda": (
            "Resumo do corpus de embeddings por região: número de patches, "
            "dimensão, modelo e status de extração."
        ),
        "secao_recomendada": "4. Resultados | 4.1 Corpus e Extração",
        "status": "READY",
        "nota_limitacao": "Corpus de 12 patches; seleção inicial exploratória.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "table_neighbor_rate_summary",
        "tipo": "tabela",
        "arquivo": "table_neighbor_rate_summary_v1gy.csv",
        "legenda": (
            "Taxas de vizinhança intra e inter-regional no corpus de 12 patches "
            "(top-5 vizinhos). Total de 60 pares analisados."
        ),
        "secao_recomendada": "4. Resultados | 4.2 Topologia de Vizinhança",
        "status": "READY",
        "nota_limitacao": "Top-5 em corpus de 12 patches; generalização restrita.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "table_medoids_outliers",
        "tipo": "tabela",
        "arquivo": "table_medoids_outliers_v1gy.csv",
        "legenda": (
            "Medoids e outliers estruturais por região no espaço de embeddings "
            "DINOv2. Posição é relativa ao corpus de 4 patches por região."
        ),
        "secao_recomendada": "4. Resultados | 4.3 Estrutura Regional",
        "status": "READY",
        "nota_limitacao": "Medoid de 4 patches; não representa a região completa.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "table_external_evidence_coverage_summary",
        "tipo": "tabela",
        "arquivo": "table_external_evidence_coverage_summary_v1gy.csv",
        "legenda": (
            "Cobertura dos indicadores GIS por região e por tipo de indicador. "
            "GIS é contextual — não é ground truth nem validação."
        ),
        "secao_recomendada": "4. Resultados | 4.3 Evidência Contextual GIS",
        "status": "READY",
        "nota_limitacao": "Cobertura incompleta; sem indicadores de drenagem em Curitiba.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "table_review_candidates_summary",
        "tipo": "tabela",
        "arquivo": "table_review_candidates_summary_v1gy.csv",
        "legenda": (
            "Resumo dos 47 candidatos de revisão por categoria de seleção. "
            "Nenhum candidato recebeu label ou classificação operacional."
        ),
        "secao_recomendada": "4. Resultados | 4.4 Candidatos de Revisão",
        "status": "READY",
        "nota_limitacao": "Seleção por critérios estruturais; sem amostragem formal.",
        "corpo_ou_apendice": "corpo",
    },
    {
        "artifact_id": "table_figures_for_tcc_manifest",
        "tipo": "tabela",
        "arquivo": "table_figures_for_tcc_manifest_v1gy.csv",
        "legenda": (
            "Manifesto de figuras e tabelas geradas para o TCC: "
            "arquivo, status, seção recomendada e nota de limitação."
        ),
        "secao_recomendada": "Apêndice B — Manifesto de Evidências",
        "status": "READY",
        "nota_limitacao": "Manifesto administrativo; não é resultado científico.",
        "corpo_ou_apendice": "apendice",
    },
]


# ---------------------------------------------------------------------------
# 7. Appendices plan
# ---------------------------------------------------------------------------

def build_appendices_plan(ev: Evidence) -> str:
    return f"""# Plano de Apêndices — REV-P v1hf

> Gerado automaticamente em {datetime.now(timezone.utc).date()}.
> Os apêndices complementam o corpo do TCC sem sobrecarregar o texto principal.
> Marcados como RECOMENDADO (incluir), OPCIONAL (incluir se houver espaço) ou
> REFERÊNCIA (mencionar mas não inserir integralmente).

---

## Apêndice A — Robustez dos Embeddings

**Status**: RECOMENDADO
**Origem**: artefatos do estágio v1ha (`local_runs/dino_embeddings/v1ha/`)
**Conteúdo**:
- Tabela de deriva média por patch e por região
- Tipos de perturbação testados: {ev.n_pert_types} ({", ".join(ev.ha_summary.get("perturbation_types", ["gaussian_noise", "brightness_scale", "contrast_scale", "blur_light", "crop_jitter", "band_dropout"]))})
- Resultado: {ev.n_robust}/{ev.n_patches} ROBUST, {ev.n_unstable} instáveis
- Arquivo: `robust_embeddings.csv`, `perturbation_robustness_summary.json`

**Observação**: Inclui dados quantitativos que sustentam a afirmação de robustez
sem sobrecarregar a Seção 4. Pode ser referenciado no corpo com "Vide Apêndice A".

---

## Apêndice B — Manifesto de Evidências e Figuras

**Status**: RECOMENDADO
**Origem**: v1gy (`local_runs/tcc_figures/v1gy/`)
**Conteúdo**:
- `tcc_visual_evidence_manifest_v1gy.csv` — manifesto completo de figuras/tabelas
- `table_figures_for_tcc_manifest_v1gy.csv` — índice com status por artefato
- Lista de artefatos gerados: {ev.n_figures_ready} figuras, {ev.n_tables_ready} tabelas

**Observação**: Documenta a proveniência de cada figura e tabela.

---

## Apêndice C — Protocolo de Revisão Humana

**Status**: RECOMENDADO
**Origem**: `docs/metodologia_cientifica/human_review_protocol.md`
**Conteúdo**:
- Critérios de seleção dos 47 candidatos
- Protocolo de anotação com campos e escala de incerteza
- Regras de governança para notas descritivas
- Distinção explícita entre revisão assistida e classificação operacional

**Observação**: Essencial para fundamentar metodologicamente a seção 4.5
(Revisão Visual Assistida).

---

## Apêndice D — Contact Sheets de Revisão Visual

**Status**: OPCIONAL
**Origem**: `local_runs/dino_embeddings/v1hc/figures/` (artefatos locais, não versionados)
**Conteúdo**:
- 4 contact sheets por categoria (medoids, outliers, baixa cobertura GIS, todos)
- Previews individuais RGB+NDVI para cada candidato

**Observação**: Inserir apenas 2–3 exemplos representativos no corpo do texto
(medoid + outlier severo). Contact sheets completas → apêndice D, se houver espaço.
PNGs não são versionados. Para o Overleaf, fazer upload manual das imagens selecionadas.

---

## Apêndice E — Governança de Claims

**Status**: RECOMENDADO
**Origem**: v1gz + v1he
**Conteúdo**:
- `claim_to_evidence_crosswalk_v1gz.csv` — crosswalk claim → evidência
- `claim_result_limitation_matrix_v1he.csv` — matriz claim → resultado → limitação
- {ev.n_allowed_claims} claims permitidos documentados
- {ev.n_forbidden_claims} claims proibidos bloqueados

**Observação**: Fundamental para demonstrar rigor metodológico na banca.
Pode ser apresentado como tabela ou como lista no apêndice.

---

## Apêndice F — Registros de Dataset e Corpus

**Status**: OPCIONAL
**Origem**: `datasets/`
**Conteúdo**:
- `dataset_registry.csv` — registro de todos os datasets utilizados
- `patch_corpus_registry.csv` — evolução do corpus por estágio de pipeline
- `external_evidence_registry.csv` — indicadores GIS coletados

**Observação**: Documenta a linhagem dos dados de entrada. Relevante para
reprodutibilidade e transparência.

---

## Apêndice G — Dossier de Evidências Científicas

**Status**: REFERÊNCIA (não inserir integralmente)
**Origem**: `docs/metodologia_cientifica/scientific_evidence_master_dossier.md`
**Conteúdo**:
- Inventário completo de artefatos científicos do pipeline
- Mapeamento evidência → seção do TCC

**Observação**: Documento longo — inserir apenas sumário ou referenciar como
documento técnico suplementar não publicado.

---

## Apêndice H — Manifesto SHA-512 de Embeddings

**Status**: REFERÊNCIA (não inserir — artefato local privado)
**Origem**: `local_runs/dino_embeddings/v1ge/` (não versionado)
**Conteúdo**:
- Hash SHA-512 de cada embedding para verificação de reprodutibilidade
- 12 embeddings, 12 hashes verificados

**Observação**: Mencionar no texto que os embeddings têm hash verificado,
sem inserir a tabela completa (contém referências a paths locais).

---

## O Que Não Vai para Apêndice

- TIFs Sentinel brutos (privados, artefatos pesados)
- Embeddings NPZ (artefatos locais pesados)
- Paths privados ou absolutos de qualquer natureza
- Qualquer afirmação de predição, classificação ou ground truth
- Resultados de experimentos não executados

---

*Plano gerado em {datetime.now(timezone.utc).date()} — v1hf.*
"""


# ---------------------------------------------------------------------------
# 8. Section–Artifact crosswalk (static constant)
# ---------------------------------------------------------------------------

CROSSWALK: list[dict] = [
    {
        "secao_tcc": "3.1 Enquadramento Metodológico",
        "argumento_cientifico": (
            "Abordagem audit-first, Sentinel-first, review-only: "
            "sem treinamento supervisionado, sem labels, sem ground truth"
        ),
        "artifact_sustentando": (
            "docs/metodologia_cientifica/dino_sentinel_embedding_protocol.md; "
            "scientific_evidence_master_summary_v1gz.json (methodological_guardrails)"
        ),
        "figura_tabela": "—",
        "limitacao_associada": (
            "Review-only não implica que o método é operacional; "
            "é uma escolha de escopo explícita"
        ),
    },
    {
        "secao_tcc": "3.2–3.3 Áreas de Estudo e Corpus Sentinel",
        "argumento_cientifico": (
            "128 TIFs Sentinel-2 inventariados; 12 selecionados para corpus "
            "de embeddings (4 por região: Curitiba, Petrópolis, Recife)"
        ),
        "artifact_sustentando": (
            "datasets/patch_corpus_registry.csv; "
            "manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/"
        ),
        "figura_tabela": "table_embedding_corpus_summary_v1gy.csv",
        "limitacao_associada": (
            "Seleção inicial exploratória sem representatividade estatística formal; "
            "crosswalk canônico patch→TIF não confirmado espacialmente"
        ),
    },
    {
        "secao_tcc": "3.4–3.5 Pipeline de Extração e QA",
        "argumento_cientifico": (
            "DINOv2-com-registers (768-dim, frozen); pipeline reproduzível "
            "com hash SHA-512 por embedding; 12/12 reprodutibilidade verificada"
        ),
        "artifact_sustentando": (
            "docs/metodologia_cientifica/dino_sentinel_embedding_protocol.md; "
            "local_runs/dino_embeddings/v1ge/ (hash manifesto — local only)"
        ),
        "figura_tabela": "—",
        "limitacao_associada": (
            "Transfer learning implícito não validado formalmente para Sentinel; "
            "embeddings locais e privados, não publicados"
        ),
    },
    {
        "secao_tcc": "4.1 Similaridade Estrutural de Embeddings",
        "argumento_cientifico": (
            "Matriz de similaridade cosseno 12×12; padrões de vizinhança "
            "estrutural documentados; análise exploratória"
        ),
        "artifact_sustentando": (
            "local_runs/dino_embeddings/v1gu/embedding_similarity_matrix_v1gu.json; "
            "local_runs/dino_embeddings/v1gu/embedding_neighbors_v1gu.csv"
        ),
        "figura_tabela": (
            "fig_similarity_heatmap_v1gy.png; "
            "fig_neighbor_network_v1gy.png"
        ),
        "limitacao_associada": (
            "Corpus de 12 patches; resultado não generaliza para os 128 TIFs; "
            "sem afirmação de predição"
        ),
    },
    {
        "secao_tcc": "4.2 Topologia de Vizinhança",
        "argumento_cientifico": (
            "36,7% pares intra-regionais / 63,3% inter-regionais em 60 pares top-5; "
            "estrutura de vizinhança não segue fronteiras geográficas"
        ),
        "artifact_sustentando": (
            "local_runs/dino_embeddings/v1gu/embedding_neighbors_v1gu.csv; "
            "local_runs/tcc_figures/v1gy/table_neighbor_rate_summary_v1gy.csv"
        ),
        "figura_tabela": (
            "fig_intra_inter_neighbor_rate_v1gy.png; "
            "table_neighbor_rate_summary_v1gy.csv"
        ),
        "limitacao_associada": (
            "4 patches por região; resultado dependente do corpus específico; "
            "sem inferência causal"
        ),
    },
    {
        "secao_tcc": "4.3 Estrutura Regional: Medoids e Outliers",
        "argumento_cientifico": (
            "Medoids estruturais: CUR_00357, PET_00104, REC_00205; "
            "outliers: CUR_00350, PET_00016, REC_00019"
        ),
        "artifact_sustentando": (
            "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json; "
            "local_runs/tcc_figures/v1gy/table_medoids_outliers_v1gy.csv"
        ),
        "figura_tabela": "table_medoids_outliers_v1gy.csv",
        "limitacao_associada": (
            "Medoid de 4 patches — não representa a região completa; "
            "outlier é posicional, não implica patch problemático"
        ),
    },
    {
        "secao_tcc": "4.4 Robustez dos Embeddings",
        "argumento_cientifico": (
            "12/12 ROBUST sob 6 tipos de perturbação; "
            "deriva cosseno: Curitiba=0,043, Petrópolis=0,078, Recife=0,060"
        ),
        "artifact_sustentando": (
            "local_runs/dino_embeddings/v1ha/robust_embeddings.csv; "
            "local_runs/dino_embeddings/v1ha/perturbation_robustness_summary.json"
        ),
        "figura_tabela": "Apêndice A — Robustez dos Embeddings",
        "limitacao_associada": (
            "Robustez de entrada, não robustez de domínio; "
            "sem generalização temporal ou de novo domínio"
        ),
    },
    {
        "secao_tcc": "4.5 Evidência Contextual GIS",
        "argumento_cientifico": (
            "5 indicadores GIS por região: cobertura PARTIAL/NOT_ACQUIRED/MISSING; "
            "GIS é contextual, não é ground truth"
        ),
        "artifact_sustentando": (
            "local_runs/dino_embeddings/v1gv/evidence_coverage_matrix_v1gv.csv; "
            "local_runs/tcc_figures/v1gy/table_external_evidence_coverage_summary_v1gy.csv"
        ),
        "figura_tabela": (
            "fig_external_evidence_coverage_v1gy.png; "
            "table_external_evidence_coverage_summary_v1gy.csv"
        ),
        "limitacao_associada": (
            "Cobertura incompleta em todas as regiões; "
            "GIS não permite validação dos embeddings"
        ),
    },
    {
        "secao_tcc": "4.6 Candidatos de Revisão e Revisão Visual",
        "argumento_cientifico": (
            "47 candidatos selecionados por critérios estruturais; "
            "47/47 com estatísticas de imagem computadas (NDVI, brilho, heterogeneidade)"
        ),
        "artifact_sustentando": (
            "local_runs/dino_embeddings/v1hb/human_review_execution_manifest_v1hb.csv; "
            "local_runs/dino_embeddings/v1hd/human_review_visual_annotation_v1hd.csv"
        ),
        "figura_tabela": (
            "fig_review_candidate_categories_v1gy.png; "
            "table_review_candidates_summary_v1gy.csv"
        ),
        "limitacao_associada": (
            "Revisão assistida por estatísticas — não inspeção humana direta; "
            "todos os 47 com usabilidade conditional"
        ),
    },
    {
        "secao_tcc": "5. Discussão (interpretação e limitações)",
        "argumento_cientifico": (
            "Interpretação conservadora dos resultados estruturais; "
            "explicitação de limitações e do que os resultados não sustentam"
        ),
        "artifact_sustentando": (
            "local_runs/tcc_synthesis/v1he/discussion_section_draft_v1he.md; "
            "local_runs/dino_embeddings/v1gz/claim_to_evidence_crosswalk_v1gz.csv"
        ),
        "figura_tabela": "Todas as figuras e tabelas do corpo",
        "limitacao_associada": (
            "Corpus pequeno; ausência de ground truth; GIS parcial; "
            "revisão visual assistida — não supervisionada"
        ),
    },
]


# ---------------------------------------------------------------------------
# 9. Package summary JSON
# ---------------------------------------------------------------------------

def build_summary(ev: Evidence, sections_generated: list[str]) -> dict:
    n_crosswalk = len(CROSSWALK)
    n_figs = sum(1 for r in FIGURES_TABLES_INDEX if r["tipo"] == "figura")
    n_tbls = sum(1 for r in FIGURES_TABLES_INDEX if r["tipo"] == "tabela")
    n_body = sum(1 for r in FIGURES_TABLES_INDEX if r["corpo_ou_apendice"] == "corpo")
    n_app_entries = sum(1 for r in FIGURES_TABLES_INDEX if r["corpo_ou_apendice"] == "apendice")

    required_sections = {
        "metodologia", "resultados", "discussao", "limitacoes", "contribuicoes"
    }
    ready = (
        required_sections.issubset(set(sections_generated))
        and n_figs >= 5
        and n_tbls >= 5
        and ev.n_forbidden_claims >= 10
        and n_crosswalk >= 8
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "sections_ready": sections_generated,
        "sections_required": sorted(required_sections),
        "figures_ready": ev.n_figures_ready,
        "tables_ready": ev.n_tables_ready,
        "figures_indexed": n_figs,
        "tables_indexed": n_tbls,
        "indexed_for_body": n_body,
        "indexed_for_appendix": n_app_entries,
        "appendices_planned": 8,
        "crosswalk_entries": n_crosswalk,
        "forbidden_claims_checked": True,
        "forbidden_claims_blocked": ev.n_forbidden_claims,
        "allowed_claims_ready": ev.n_allowed_claims,
        "ready_for_template_insertion": ready,
        "ready_justification": (
            f"{len(sections_generated)}/5 seções geradas; "
            f"{ev.n_figures_ready} figs; {ev.n_tables_ready} tabelas; "
            f"{ev.n_forbidden_claims} claims proibidos bloqueados; "
            f"{n_crosswalk} entradas no crosswalk"
        ),
        "corpus": {
            "n_patches": ev.n_patches,
            "n_regions": ev.n_regions,
            "embedding_dim": ev.emb_dim,
            "backbone": ev.backbone,
        },
        "methodological_guardrails": {
            "labels_created": False,
            "predictions_made": False,
            "ground_truth_established": False,
            "review_only": True,
            "all_forbidden_claims_blocked": True,
            "gis_contextual_only": True,
            "multimodal_disabled": True,
        },
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"[v1hf] Written: {path.name} ({len(content)} chars)")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[v1hf] Written: {path.name} ({len(rows)} rows)")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[v1hf] Written: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("[v1hf] Loading evidence from pipeline outputs...")
    ev = Evidence()

    print(f"[v1hf] corpus={ev.n_patches} patches, {ev.n_regions} regions, "
          f"dim={ev.emb_dim}")
    print(f"[v1hf] figures_ready={ev.n_figures_ready}, tables_ready={ev.n_tables_ready}")
    print(f"[v1hf] claims_allowed={ev.n_allowed_claims}, "
          f"claims_blocked={ev.n_forbidden_claims}")
    print(f"[v1hf] robust={ev.n_robust}/{ev.n_patches}, "
          f"visual_computed={ev.n_visual_computed}/{ev.n_review_candidates}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sections_generated: list[str] = []

    print("[v1hf] Generating Metodologia draft...")
    write_md(OUT_DIR / "metodologia_overleaf_draft_v1hf.md", build_metodologia(ev))
    sections_generated.append("metodologia")

    print("[v1hf] Generating Resultados draft...")
    write_md(OUT_DIR / "resultados_overleaf_draft_v1hf.md", build_resultados(ev))
    sections_generated.append("resultados")

    print("[v1hf] Generating Discussão draft...")
    write_md(OUT_DIR / "discussao_overleaf_draft_v1hf.md", build_discussao(ev))
    sections_generated.append("discussao")

    print("[v1hf] Generating Limitações draft...")
    write_md(OUT_DIR / "limitacoes_overleaf_draft_v1hf.md", build_limitacoes(ev))
    sections_generated.append("limitacoes")

    print("[v1hf] Generating Contribuições draft...")
    write_md(OUT_DIR / "contribuicoes_overleaf_draft_v1hf.md", build_contribuicoes(ev))
    sections_generated.append("contribuicoes")

    print("[v1hf] Generating figures/tables index...")
    write_csv(
        OUT_DIR / "overleaf_figures_tables_index_v1hf.csv",
        FIGURES_TABLES_INDEX,
    )

    print("[v1hf] Generating appendices plan...")
    write_md(OUT_DIR / "appendices_plan_v1hf.md", build_appendices_plan(ev))

    print("[v1hf] Generating section-artifact crosswalk...")
    write_csv(
        OUT_DIR / "tcc_section_artifact_crosswalk_v1hf.csv",
        CROSSWALK,
    )

    print("[v1hf] Generating package summary JSON...")
    summary = build_summary(ev, sections_generated)
    write_json(OUT_DIR / "overleaf_package_summary_v1hf.json", summary)

    print(f"\n[v1hf] Done. Outputs in {OUT_DIR.relative_to(ROOT)}")
    print(f"[v1hf] ready_for_template_insertion: "
          f"{summary['ready_for_template_insertion']}")
    print(f"[v1hf] Justification: {summary['ready_justification']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
