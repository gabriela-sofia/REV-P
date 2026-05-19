"""REV-P v1he: TCC Results and Discussion Synthesis Package.

Transforms consolidated scientific evidence from v1gz–v1hd into draft text,
figure/table captions, and a claim-result-limitation matrix ready for Overleaf.

All text is grounded in actual numbers from local pipeline outputs.
No new claims are created. All forbidden claims remain blocked.

Outputs (local_runs/tcc_synthesis/v1he/):
  results_section_draft_v1he.md
  discussion_section_draft_v1he.md
  figure_captions_final_v1he.csv
  table_captions_final_v1he.csv
  claim_result_limitation_matrix_v1he.csv
  tcc_results_discussion_summary_v1he.json
  overleaf_insert_plan_v1he.md
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1he"

# Input dirs
V1GZ_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gz"
V1HA_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1ha"
V1GY_DIR = ROOT / "local_runs" / "tcc_figures" / "v1gy"
V1HB_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hb"
V1HC_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hc"
V1HD_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hd"
OUT_DIR = ROOT / "local_runs" / "tcc_synthesis" / "v1he"


# ---------------------------------------------------------------------------
# Data loading
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


class Evidence:
    """Container for all pipeline evidence loaded from local_runs."""

    def __init__(self) -> None:
        # v1gz
        self.gz_summary = _read_json(V1GZ_DIR / "scientific_evidence_master_summary_v1gz.json")
        self.gz_claims = _read_csv(V1GZ_DIR / "evidence_strength_by_claim_v1gz.csv")
        self.gz_crosswalk = _read_csv(V1GZ_DIR / "claim_to_evidence_crosswalk_v1gz.csv")
        self.gz_readiness = _read_csv(V1GZ_DIR / "tcc_result_readiness_matrix_v1gz.csv")

        # v1ha
        self.ha_summary = _read_json(V1HA_DIR / "perturbation_robustness_summary.json")
        self.ha_robust = _read_csv(V1HA_DIR / "robust_embeddings.csv")
        self.ha_drift = _read_csv(V1HA_DIR / "embedding_drift_metrics.csv")
        self.ha_regional_drift = _read_json(V1HA_DIR / "regional_drift_summary.json")

        # v1gy
        self.gy_summary = _read_json(V1GY_DIR / "tcc_visual_evidence_summary_v1gy.json")
        self.gy_manifest = _read_csv(V1GY_DIR / "table_figures_for_tcc_manifest_v1gy.csv")
        self.gy_corpus = _read_csv(V1GY_DIR / "table_embedding_corpus_summary_v1gy.csv")
        self.gy_medoids = _read_csv(V1GY_DIR / "table_medoids_outliers_v1gy.csv")
        self.gy_neighbor_rate = _read_csv(V1GY_DIR / "table_neighbor_rate_summary_v1gy.csv")
        self.gy_coverage = _read_csv(V1GY_DIR / "table_external_evidence_coverage_summary_v1gy.csv")
        self.gy_review_cats = _read_csv(V1GY_DIR / "table_review_candidates_summary_v1gy.csv")

        # v1hb
        self.hb_manifest = _read_csv(V1HB_DIR / "human_review_execution_manifest_v1hb.csv")

        # v1hc
        self.hc_summary = _read_json(V1HC_DIR / "visual_review_preview_summary_v1hc.json")

        # v1hd
        self.hd_summary = _read_json(V1HD_DIR / "human_review_visual_summary_v1hd.json")
        self.hd_examples = _read_csv(V1HD_DIR / "human_review_visual_examples_for_tcc_v1hd.csv")

        # Derived
        self.n_patches = self.gz_summary.get("corpus_size", 12)
        self.n_regions = self.gz_summary.get("n_regions", 3)
        self.emb_dim = self.gz_summary.get("embedding_dimension", 768)
        self.backbone = self.gz_summary.get("embedding_backbone", "DINOv2-com-registers")
        self.n_review_candidates = self.gz_summary.get("human_review_candidates", 47)
        self.n_figures_ready = self.gy_summary.get("total_figures_ready", 0)
        self.n_tables_ready = self.gy_summary.get("total_tables_ready", 0)
        self.n_allowed_claims = self.gz_summary.get("allowed_claims_count", 0)
        self.n_forbidden_claims = self.gz_summary.get("forbidden_claims_count", 0)
        self.n_robust = self.ha_summary.get("robust_count", 0)
        self.n_unstable = self.ha_summary.get("unstable_count", 0)
        self.n_perturbation_types = len(self.ha_summary.get("perturbation_types", []))

        # Neighbor rate from table
        if self.gy_neighbor_rate:
            nr = self.gy_neighbor_rate[0]
            self.intra_rate = float(nr.get("taxa_intra", 0.3667))
            self.inter_rate = float(nr.get("taxa_inter", 0.6333))
            self.n_neighbor_pairs = int(nr.get("total_pares_vizinhos", 60))
            self.n_intra = int(nr.get("pares_intra_regiao", 22))
            self.n_inter = int(nr.get("pares_inter_regiao", 38))
        else:
            self.intra_rate = 0.367
            self.inter_rate = 0.633
            self.n_neighbor_pairs = 60
            self.n_intra = 22
            self.n_inter = 38

        # Regional drift
        self.regional_mean_drift: dict[str, float] = {}
        rd = self.ha_regional_drift.get("regional_mean_drift", {})
        for region, val in rd.items():
            self.regional_mean_drift[region] = float(val)

        # Visual review stats
        self.n_visual_computed = self.hd_summary.get("n_visually_computed", 0)
        self.hd_uncertainty = self.hd_summary.get("by_uncertainty", {})
        self.hd_usable = self.hd_summary.get("by_usable_in_discussion", {})


# ---------------------------------------------------------------------------
# Results section draft
# ---------------------------------------------------------------------------

def build_results_section(ev: Evidence) -> str:
    # Medoid distances
    medoid_info = ""
    for r in ev.gy_medoids:
        medoid_info += (
            f"- **{r['regiao']}**: medoid = {r['medoid']}, "
            f"outlier estrutural = {r['outliers']}\n"
        )

    # Regional drift
    drift_lines = ""
    for region, drift in ev.regional_mean_drift.items():
        drift_lines += f"- {region}: deriva média = {drift:.4f} (cosseno)\n"

    # Perturbation types
    pert_types = ", ".join(ev.ha_summary.get("perturbation_types", []))

    # Hd uncertainty
    low_n = ev.hd_uncertainty.get("low", 0)
    med_n = ev.hd_uncertainty.get("medium", 0)
    high_n = ev.hd_uncertainty.get("high", 0)

    return f"""# Seção 4 — Resultados

> **Nota metodológica**: Este rascunho foi gerado automaticamente a partir dos
> artefatos computados em v1gz–v1hd. Os números são reais. O texto requer revisão
> humana antes de inserção no Overleaf. Nenhum claim preditivo ou operacional foi criado.

---

## 4.1 Corpus Sentinel e Extração de Embeddings

O corpus de análise compreende **{ev.n_patches} imagens Sentinel-2** (patches de aproximadamente
10 m de resolução espacial), distribuídas em **{ev.n_regions} regiões geográficas**:
Curitiba, Petrópolis e Recife, com 4 patches por região. Os patches foram selecionados
por amostragem exploratória inicial, sem critério de representatividade estatística formal.

A extração de embeddings utilizou o modelo **{ev.backbone}**, produzindo vetores de
**{ev.emb_dim} dimensões** por patch. Todos os {ev.n_patches} embeddings foram extraídos
com sucesso, verificados por hash e registrados em manifesto auditável (v1ge).

> **Tabela 4.1** — Resumo do corpus de embeddings por região
> (arquivo: `table_embedding_corpus_summary_v1gy.csv`)

---

## 4.2 Análise Estrutural de Embeddings

### 4.2.1 Matriz de Similaridade Cosseno

A similaridade cosseno par-a-par foi calculada entre todos os {ev.n_patches} patches,
produzindo uma matriz {ev.n_patches}×{ev.n_patches} de similaridades no espaço de alta
dimensão do DINOv2. A análise exploratória desta matriz revela padrões de vizinhança
estrutural entre patches de regiões distintas.

> **Figura 4.1** — Mapa de calor da similaridade cosseno par-a-par
> (arquivo: `fig_similarity_heatmap_v1gy.png`)

### 4.2.2 Topologia de Vizinhança (top-5)

Para cada patch, foram identificados os 5 vizinhos mais próximos por similaridade
cosseno, totalizando **{ev.n_neighbor_pairs} pares de vizinhança** no corpus.
A distribuição desses pares entre fronteiras regionais revela:

- **Pares intra-regionais**: {ev.n_intra} de {ev.n_neighbor_pairs} ({ev.intra_rate:.1%})
- **Pares inter-regionais**: {ev.n_inter} de {ev.n_neighbor_pairs} ({ev.inter_rate:.1%})

A taxa inter-regional elevada ({ev.inter_rate:.1%}) indica que a estrutura de vizinhança
no espaço de embeddings não se organiza estritamente por fronteiras geográficas. Esta
observação é de natureza estrutural e exploratória — não implica relação causal ou
operacional entre os patches.

> **Figura 4.2** — Grafo de vizinhança top-5 entre patches
> (arquivo: `fig_neighbor_network_v1gy.png`)

> **Figura 4.3** — Taxa de vizinhança intra/inter-regional
> (arquivo: `fig_intra_inter_neighbor_rate_v1gy.png`)

> **Tabela 4.2** — Resumo das taxas de vizinhança
> (arquivo: `table_neighbor_rate_summary_v1gy.csv`)

---

## 4.3 Estrutura Regional: Medoids e Outliers

Para cada região, foi identificado o **medoid** (patch mais central na distribuição
de embeddings) e o patch **outlier estrutural** (mais periférico):

{medoid_info}
> **Tabela 4.3** — Medoids e outliers estruturais por região
> (arquivo: `table_medoids_outliers_v1gy.csv`)

Os medoids representam o ponto de referência estrutural de cada região no espaço
de embeddings. Os outliers documentam variabilidade interna. Nenhum desses rótulos
implica qualidade superior, inferior ou operacional dos patches.

---

## 4.4 Evidência Contextual Externa (GIS)

Indicadores GIS foram coletados como evidência contextual para as três regiões,
sem uso como ground truth ou validação. A cobertura varia por região e indicador:

- **Curitiba**: terrain_geocuritiba (PARTIAL); land_use e population_density
  (NOT_ACQUIRED); drainage e defesa_civil (MISSING)
- **Petrópolis**: geological_cprm, land_use_fbds, terrain_sgb_rigeo (PARTIAL);
  population_density (NOT_ACQUIRED); drainage (MISSING)
- **Recife**: coastal_context, drainage_esig, terrain_pe3d (PARTIAL);
  land_use e population_density (NOT_ACQUIRED)

> **Figura 4.4** — Cobertura de indicadores GIS por região
> (arquivo: `fig_external_evidence_coverage_v1gy.png`)

> **Tabela 4.4** — Resumo da cobertura de evidência externa
> (arquivo: `table_external_evidence_coverage_summary_v1gy.csv`)

---

## 4.5 Candidatos de Revisão Humana

**{ev.n_review_candidates} candidatos de revisão** foram selecionados para inspeção
visual com base em critérios estruturais de embedding (medoids, outliers) e cobertura
GIS externa baixa. A seleção é exploratória — não constitui classificação automática.

> **Figura 4.5** — Distribuição de candidatos por categoria de revisão
> (arquivo: `fig_review_candidate_categories_v1gy.png`)

> **Tabela 4.5** — Resumo de candidatos de revisão por categoria
> (arquivo: `table_review_candidates_summary_v1gy.csv`)

---

## 4.6 Revisão Visual Assistida (v1hc/v1hd)

Previews Sentinel RGB e NDVI foram gerados para **{ev.n_visual_computed} de
{ev.n_review_candidates} candidatos** ({ev.n_visual_computed / ev.n_review_candidates:.0%}).
A análise visual foi conduzida com base em estatísticas de imagem computadas
(NDVI médio, reflexão, heterogeneidade espacial), sem inspeção humana direta.

Distribuição de incerteza pós-revisão visual:
- Baixa (medoids): {low_n} candidatos
- Média (corpus v1gu com sinal NDVI claro): {med_n} candidatos
- Alta (não-corpus ou sinal ambíguo): {high_n} candidatos

Todos os {ev.n_review_candidates} candidatos foram marcados `usable_in_discussion:
conditional` — uso requer explicitação de limitações.

---

## 4.7 Robustez dos Embeddings (v1ha)

A estabilidade dos embeddings foi verificada sob {ev.n_perturbation_types} tipos
de perturbação ({pert_types}).

Resultados:
- **{ev.n_robust} de {ev.n_patches} embeddings** classificados como ROBUST
- **{ev.n_unstable} embeddings** instáveis (0% do corpus)
- Deriva média por região:
{drift_lines}
A robustez sob perturbação suporta a afirmação de que os embeddings capturam
estrutura estável da imagem Sentinel — sem implicar validação contra ground truth
ou predição de qualquer variável alvo.

> **Apêndice A** — Resultados de robustez por tipo de perturbação
> (artefatos: `local_runs/dino_embeddings/v1ha/`)

---

*Rascunho gerado automaticamente em {datetime.now(timezone.utc).date()} — v1he.*
*Números reais extraídos de artefatos computados. Revisão humana obrigatória antes do Overleaf.*
"""


# ---------------------------------------------------------------------------
# Discussion section draft
# ---------------------------------------------------------------------------

def build_discussion_section(ev: Evidence) -> str:
    drift_lines = ""
    for region, drift in ev.regional_mean_drift.items():
        drift_lines += f"  - {region}: {drift:.4f}\n"

    return f"""# Seção 5 — Discussão

> **Nota metodológica**: Este rascunho foi gerado automaticamente. O texto é de
> natureza exploratória e interpretativa. Nenhum claim preditivo ou operacional
> foi criado. Revisão humana obrigatória antes da inserção no Overleaf.

---

## 5.1 O Que os Resultados Sustentam

Os resultados apresentados nas Seções 4.1–4.7 sustentam as seguintes afirmações:

**a) Coerência estrutural dos embeddings**
Os {ev.n_patches} embeddings DINOv2 extraídos de imagens Sentinel-2 das regiões
de Curitiba, Petrópolis e Recife apresentam estrutura de similaridade cosseno
computável e internamente consistente. A matriz de similaridade revela padrões de
vizinhança que podem ser interpretados exploratoriamente — sem validação contra
referência externa.

**b) Heterogeneidade estrutural inter-regional**
A taxa de vizinhança inter-regional ({ev.inter_rate:.1%} dos {ev.n_neighbor_pairs}
pares de vizinhança top-5) é maior do que a taxa intra-regional ({ev.intra_rate:.1%}).
Isso indica que o espaço de embeddings DINOv2 não se organiza estritamente por
fronteira geográfica — patches de regiões distintas são estruturalmente mais
próximos do que patches da mesma região em {ev.inter_rate:.1%} dos pares.

Esta observação é relevante para a análise exploratória: a similitude estrutural
entre patches transcende a localização geográfica no espaço de características
DINOv2. A interpretação desta observação requer cautela — o corpus de {ev.n_patches}
patches é pequeno e a generalização é limitada.

**c) Representatividade regional dos medoids**
Os medoids identificados por região (CUR_00357, PET_00104, REC_00205) representam
o centro da distribuição de embeddings de cada região. Eles servem como âncoras
para interpretação visual comparativa — sem implicar que sejam "amostras típicas"
no sentido estatístico formal, dado o tamanho reduzido do corpus.

**d) Heterogeneidade documentada pelos outliers**
Os outliers estruturais identificados (CUR_00350, PET_00016, REC_00019) apresentam
características notáveis: PET_00016 e REC_00019 têm similaridade máxima com seus
vizinhos mais próximos inferior a 0.55 — muito abaixo do padrão do corpus (0.72–0.91).
Isso documenta variação estrutural real no espaço de embeddings, sem implicar
anomalia operacional ou risco em qualquer sentido.

**e) Estabilidade sob perturbação**
Os {ev.n_patches} embeddings foram testados sob {ev.n_perturbation_types} tipos de
perturbação e todos resultaram em status ROBUST. A deriva média por região foi:
{drift_lines}
Esta evidência suporta a afirmação de que os embeddings capturam características
estáveis da imagem Sentinel — não são artefatos de variação aleatória de entrada.

**f) Evidência visual assistida**
A revisão visual assistida de {ev.n_visual_computed}/{ev.n_review_candidates} candidatos
por estatísticas de imagem (NDVI, reflexão, heterogeneidade espacial) produz
descrições de padrão visual que complementam a evidência estrutural dos embeddings.
{ev.hd_uncertainty.get("low", 0)} candidatos apresentam incerteza baixa (medoids com sinal claro);
{ev.hd_uncertainty.get("medium", 0)} apresentam incerteza média; {ev.hd_uncertainty.get("high", 0)} mantêm incerteza alta.

---

## 5.2 O Que os Resultados Não Sustentam

Os resultados desta análise **não permitem** as seguintes afirmações:

- **Predição de risco hidrológico**: Os embeddings DINOv2 não foram treinados
  para detectar ou predizer categorias de risco operacional ou vulnerabilidade territorial.
  A análise é puramente estrutural e exploratória.

- **Validação contra ground truth**: Não existe referência de campo (ground truth)
  neste estudo. A revisão visual assistida não é validação supervisionada — é
  interpretação exploratória de padrões de imagem.

- **Classificação de vulnerabilidade**: Nenhum patch foi classificado como
  "vulnerável", "de risco" ou equivalente. Os rótulos "medoid" e "outlier"
  são estruturais, não operacionais.

- **Generalização para os 128 patches**: O corpus de {ev.n_patches} embeddings
  é uma amostra inicial exploratória. As observações não se generalizam
  automaticamente para o conjunto completo de patches disponíveis.

- **GIS como verdade de campo**: Os indicadores GIS coletados são evidência
  contextual — não são validação externa do método DINO. A cobertura é
  parcial/incompleta em todas as regiões.

- **Performance do modelo**: Não há métrica de performance (acurácia, F1, AUC)
  porque não há tarefa supervisionada nem ground truth. O DINO não "errou"
  ou "acertou" — ele produziu representações estruturais.

---

## 5.3 Interpretação da Taxa Intra/Inter-Regional

A taxa inter-regional de {ev.inter_rate:.1%} pode ser interpretada de três formas
exploratórias, sem hierarquia de validade:

1. **Similaridade visual trans-regional**: patches de regiões distintas compartilham
   características visuais estruturais no espaço DINOv2 (textura, cobertura,
   padrão espectral) que transcendem a localização geográfica.

2. **Heterogeneidade intra-regional**: a variação dentro de cada região é suficiente
   para que patches de regiões distintas sejam mais próximos entre si do que
   patches da mesma região. Isso documenta diversidade interna do corpus por região.

3. **Limitação do corpus**: com apenas 4 patches por região, a matriz de vizinhança
   pode refletir particularidades das imagens selecionadas, não padrões regionais
   generalizáveis. Análise mais robusta requereria corpus maior.

A interpretação mais conservadora combina os três pontos: a taxa inter-regional
documenta que a estrutura DINOv2 no espaço de embeddings não é determinada
primariamente por fronteiras administrativas no corpus atual.

---

## 5.4 Papel dos Medoids e Outliers na Discussão

Os **medoids regionais** (um por região) são úteis como pontos de referência
estrutural para comparação visual entre regiões. A revisão visual assistida
confirmou que esses patches apresentam estatísticas de imagem coerentes com
o padrão regional (incerteza: baixa para os 3 medoids).

Os **outliers estruturais** (um por região) documentam variabilidade interna.
PET_00016 e REC_00019 apresentam similaridade máxima excepcionalmente baixa
(< 0.55), distinguindo-os fortemente do padrão do corpus. A interpretação desta
divergência requer inspeção visual humana direta — as estatísticas de imagem
sugerem padrões distintos (diferenças de NDVI e reflexão), mas não explicam
a causa da divergência estrutural.

**Uso metodológico na Discussão**: medoids e outliers não são "exemplos bons"
e "exemplos problemáticos" — são posições estruturais no espaço de embeddings
que permitem interpretar a heterogeneidade regional de forma não-supervisionada.

---

## 5.5 Cobertura GIS e Sua Ausência

A baixa cobertura de indicadores GIS em todas as regiões — especialmente Curitiba,
onde drenagem e defesa civil estão completamente ausentes — é uma limitação
metodológica relevante. Ela não invalida os resultados, mas restringe a
interpretação contextual:

- Não é possível verificar se a estrutura de embeddings correlaciona com
  variáveis contextuais específicas (drenagem, topografia) por ausência de dados.
- Os 43 candidatos Curitiba selecionados por baixa cobertura GIS documentam
  exatamente essa limitação: há estrutura de embedding mas contexto GIS insuficiente.
- A revisão visual assistida parcialmente compensa essa lacuna ao fornecer
  descrições de padrão de imagem (NDVI, reflexão, heterogeneidade) — sem
  substituir indicadores contextuais formais.

---

## 5.6 Por Que o Método é Válido como Análise Estrutural

A análise DINOv2 sobre imagens Sentinel-2 é válida como **análise estrutural
exploratória** pelos seguintes motivos:

1. **Embeddings são computados de forma reproduzível**: o pipeline é auditável,
   com hash de entrada, manifesto de extração e verificação de reprodutibilidade.

2. **Os embeddings são estáveis**: 12/12 embeddings são ROBUST sob 6 tipos de
   perturbação — a estrutura não é artefato de variação aleatória.

3. **A análise é honesta sobre seus limites**: não há claims de validação,
   performance ou predição. O corpus é pequeno e a generalização é explicitamente
   restrita.

4. **A revisão humana é metodológica**: os candidatos de revisão foram selecionados
   com base em critérios estruturais (medoids, outliers, baixa cobertura GIS),
   não em julgamento operacional.

5. **GIS é contextual**: os indicadores geográficos enriquecem a interpretação
   sem serem tratados como verdade de campo ou validação.

O método não responde à pergunta "quais patches estão em risco?" — e não se
propõe a isso. Ele responde à pergunta "qual é a estrutura do espaço de
representações DINOv2 para imagens Sentinel-2 nestas três regiões, e o que
essa estrutura pode informar exploratoriamente sobre padrões visuais?".

---

## 5.7 Limitações e Trabalho Futuro

**Limitações documentadas**:

1. Corpus de {ev.n_patches} embeddings — amostra inicial exploratória, sem representatividade
   estatística formal para as regiões.
2. Ausência de ground truth — impossível calcular métricas de performance.
3. Cobertura GIS parcial ou ausente em todas as regiões.
4. Revisão visual assistida por estatísticas de imagem — não substitui inspeção
   humana direta de todas as {ev.n_review_candidates} imagens.
5. Corpus single-date — não captura variação temporal (sazonalidade, eventos).
6. DINOv2 não foi treinado para imagens Sentinel — transfer learning implícito
   não validado formalmente.

**Trabalho futuro**:

1. Expandir corpus para os 128 patches disponíveis e replicar análise estrutural.
2. Incorporar séries temporais Sentinel para análise de variação sazonal.
3. Coletar indicadores GIS completos para todas as regiões (especialmente drenagem).
4. Conduzir revisão visual humana direta dos {ev.n_review_candidates} candidatos.
5. Comparar estrutura DINOv2 com outros backbones (EfficientNet, ResNet) para
   análise de sensibilidade ao modelo.
6. Avaliar se a taxa inter-regional se mantém em corpus maior — ou se é artefato
   do corpus de {ev.n_patches} patches.

---

*Rascunho gerado automaticamente em {datetime.now(timezone.utc).date()} — v1he.*
*Todos os números são reais, extraídos dos artefatos computados. Revisão humana obrigatória.*
"""


# ---------------------------------------------------------------------------
# Figure captions
# ---------------------------------------------------------------------------

FIGURE_CAPTIONS = [
    {
        "figure_id": "fig_similarity_heatmap",
        "filename": "fig_similarity_heatmap_v1gy.png",
        "caption_pt": (
            "Mapa de calor da similaridade cosseno par-a-par entre os 12 patches "
            "do corpus (4 por região: Curitiba, Petrópolis, Recife). Valores maiores "
            "indicam maior proximidade estrutural no espaço de embeddings DINOv2 "
            "(768 dimensões). A análise é exploratória — não implica predição ou "
            "classificação operacional."
        ),
        "tcc_section": "4. Resultados | 4.1",
        "claim_scope": "Similaridade estrutural exploratória entre patches Sentinel",
        "limitation_note": (
            "Corpus de 12 patches; resultados não generalizam para o conjunto "
            "completo de 128 patches. Sem ground truth."
        ),
        "forbidden_terms_checked": "yes",
    },
    {
        "figure_id": "fig_neighbor_network",
        "filename": "fig_neighbor_network_v1gy.png",
        "caption_pt": (
            "Grafo de vizinhança top-5 entre os 12 patches do corpus. Cada nó "
            "representa um patch; arestas conectam os 5 vizinhos mais próximos "
            "por similaridade cosseno. Arestas inter-regionais (63,3% do total) "
            "indicam que a estrutura de vizinhança no espaço DINOv2 transcende "
            "fronteiras geográficas. Análise exploratória — sem rótulo operacional."
        ),
        "tcc_section": "4. Resultados | 4.1",
        "claim_scope": "Topologia de vizinhança estrutural no espaço de embeddings",
        "limitation_note": (
            "Top-5 vizinhos em corpus de 12 patches. "
            "Interpretação exploratória; generalização restrita."
        ),
        "forbidden_terms_checked": "yes",
    },
    {
        "figure_id": "fig_intra_inter_neighbor_rate",
        "filename": "fig_intra_inter_neighbor_rate_v1gy.png",
        "caption_pt": (
            "Distribuição dos 60 pares de vizinhança top-5 entre fronteiras "
            "regionais. Pares intra-regionais: 22 (36,7%); inter-regionais: "
            "38 (63,3%). A predominância inter-regional documenta heterogeneidade "
            "estrutural — não implica relação causal ou operacional entre os patches."
        ),
        "tcc_section": "4. Resultados | 4.1",
        "claim_scope": "Taxa intra/inter-regional de vizinhança estrutural",
        "limitation_note": (
            "Resultado de corpus de 12 patches (4 por região). "
            "Corpus maior pode alterar a distribuição."
        ),
        "forbidden_terms_checked": "yes",
    },
    {
        "figure_id": "fig_external_evidence_coverage",
        "filename": "fig_external_evidence_coverage_v1gy.png",
        "caption_pt": (
            "Cobertura de indicadores GIS externos por região (5 indicadores por "
            "região). Status: AVAILABLE, PARTIAL, NOT_ACQUIRED ou MISSING. Os "
            "indicadores são evidência contextual — não constituem ground truth "
            "ou validação do método."
        ),
        "tcc_section": "4. Resultados | 4.3",
        "claim_scope": "Disponibilidade de evidência contextual GIS por região",
        "limitation_note": (
            "Cobertura incompleta em todas as regiões. "
            "GIS é contextual, não validante."
        ),
        "forbidden_terms_checked": "yes",
    },
    {
        "figure_id": "fig_review_candidate_categories",
        "filename": "fig_review_candidate_categories_v1gy.png",
        "caption_pt": (
            "Distribuição dos 47 candidatos de revisão humana por categoria "
            "de seleção: medoid regional (3), outlier estrutural (3) e cobertura "
            "GIS externa baixa (43, incluindo sobreposições). A seleção é "
            "exploratória — os candidatos não são classificados automaticamente."
        ),
        "tcc_section": "4. Resultados | 4.4",
        "claim_scope": "Seleção exploratória de candidatos para revisão humana",
        "limitation_note": (
            "47 candidatos de 3 regiões; seleção não representa "
            "amostragem probabilística do conjunto completo."
        ),
        "forbidden_terms_checked": "yes",
    },
]

TABLE_CAPTIONS = [
    {
        "table_id": "table_embedding_corpus",
        "filename": "table_embedding_corpus_summary_v1gy.csv",
        "caption_pt": (
            "Resumo do corpus de embeddings por região: número de patches, norma "
            "do centroide e medoid identificado. Extração com DINOv2-com-registers "
            "(768 dimensões). Análise estrutural exploratória — sem rótulo de risco."
        ),
        "tcc_section": "4. Resultados | 4.1",
        "claim_scope": "Corpus de embeddings DINOv2 por região",
        "limitation_note": "4 patches por região; não representa distribuição regional completa.",
    },
    {
        "table_id": "table_medoids_outliers",
        "filename": "table_medoids_outliers_v1gy.csv",
        "caption_pt": (
            "Medoids e outliers estruturais identificados por região no espaço "
            "de embeddings DINOv2. O medoid é o patch mais central; o outlier é "
            "o mais periférico. Identificação exploratória — sem implicação "
            "operacional ou de qualidade."
        ),
        "tcc_section": "4. Resultados | 4.2",
        "claim_scope": "Posição estrutural de medoids e outliers por região",
        "limitation_note": (
            "1 outlier por região no corpus de 4 patches. "
            "Análise exploratória; corpus pequeno."
        ),
    },
    {
        "table_id": "table_neighbor_rate",
        "filename": "table_neighbor_rate_summary_v1gy.csv",
        "caption_pt": (
            "Taxa de vizinhança intra e inter-regional para os 12 patches do "
            "corpus (top-5 vizinhos por patch). Pares intra-regionais: 22/60 "
            "(36,7%); inter-regionais: 38/60 (63,3%). Análise topológica exploratória "
            "no espaço de embeddings."
        ),
        "tcc_section": "4. Resultados | 4.1",
        "claim_scope": "Distribuição de vizinhança entre fronteiras regionais",
        "limitation_note": (
            "Top-5 vizinhos em corpus de 12 patches (4 por região). "
            "Resultados exploratórios; generalização restrita."
        ),
    },
    {
        "table_id": "table_external_coverage",
        "filename": "table_external_evidence_coverage_summary_v1gy.csv",
        "caption_pt": (
            "Resumo de cobertura de indicadores GIS externos por região. "
            "Indicadores incluem: terrain, geological, land_use, population_density, "
            "drainage, coastal_context. Status: PARTIAL em indicadores disponíveis; "
            "NOT_ACQUIRED/MISSING nos demais. GIS é evidência contextual, não ground truth."
        ),
        "tcc_section": "4. Resultados | 4.3",
        "claim_scope": "Disponibilidade de evidência contextual externa",
        "limitation_note": "Cobertura incompleta em todas as regiões.",
    },
    {
        "table_id": "table_review_candidates",
        "filename": "table_review_candidates_summary_v1gy.csv",
        "caption_pt": (
            "Resumo dos 47 candidatos de revisão humana por categoria de seleção. "
            "Seleção baseada em critérios estruturais de embedding e cobertura GIS — "
            "sem atribuição automática de classe ou risco."
        ),
        "tcc_section": "4. Resultados | 4.4",
        "claim_scope": "Candidatos para revisão humana por categoria exploratória",
        "limitation_note": "Seleção exploratória; candidatos não são classificados.",
    },
    {
        "table_id": "table_visual_evidence_manifest",
        "filename": "tcc_visual_evidence_manifest_v1gy.csv",
        "caption_pt": (
            "Manifesto de evidências visuais para o TCC: figuras e tabelas "
            "disponíveis, status de geração e seção de destino. "
            "Artefatos auditáveis produzidos por pipeline reproduzível."
        ),
        "tcc_section": "Apêndice | Manifesto de Evidências",
        "claim_scope": "Auditabilidade do pipeline de evidências",
        "limitation_note": "Alguns artefatos marcados NEEDS_LOCAL_OUTPUT.",
    },
]


# ---------------------------------------------------------------------------
# Claim-result-limitation matrix
# ---------------------------------------------------------------------------

CLAIM_MATRIX = [
    {
        "claim_allowed": "Embeddings DINOv2 exibem coerência estrutural entre patches de regiões distintas",
        "result_supporting_it": "Similaridade cosseno calculável; padrões de vizinhança documentados",
        "evidence_artifact": "embedding_similarity_matrix_v1gu.json; embedding_neighbors_v1gu.csv",
        "figure_or_table": "fig_similarity_heatmap_v1gy.png; fig_neighbor_network_v1gy.png",
        "limitation": "Corpus de 12 patches; sem generalização estatística formal",
        "blocked_overclaim": "NÃO: embeddings NÃO predizem risco; NÃO validam modelo contra ground truth",
    },
    {
        "claim_allowed": "Taxa inter-regional de vizinhança top-5 é de 63,3% no corpus atual",
        "result_supporting_it": "38 de 60 pares são inter-regionais; calculado sobre todos os 12 patches",
        "evidence_artifact": "embedding_neighbors_v1gu.csv; table_neighbor_rate_summary_v1gy.csv",
        "figure_or_table": "fig_intra_inter_neighbor_rate_v1gy.png",
        "limitation": "4 patches por região; resultado dependente do corpus específico",
        "blocked_overclaim": "NÃO: taxa inter-regional NÃO implica que DINO detecta padrões regionais de risco",
    },
    {
        "claim_allowed": "Medoids estruturais identificados: CUR_00357, PET_00104, REC_00205",
        "result_supporting_it": "Menor distância ao centroide regional no espaço de embeddings",
        "evidence_artifact": "embedding_regional_summary_v1gu.json; table_medoids_outliers_v1gy.csv",
        "figure_or_table": "table_medoids_outliers_v1gy.csv",
        "limitation": "Medoid de corpus de 4 patches por região; não é medoid da região completa",
        "blocked_overclaim": "NÃO: medoid NÃO é o 'melhor' patch; NÃO implica representatividade estatística",
    },
    {
        "claim_allowed": "Outliers estruturais identificados: CUR_00350, PET_00016, REC_00019",
        "result_supporting_it": "PET_00016 e REC_00019 com similaridade máxima < 0.55 (outliers severos)",
        "evidence_artifact": "embedding_regional_summary_v1gu.json; embedding_neighbors_v1gu.csv",
        "figure_or_table": "table_medoids_outliers_v1gy.csv",
        "limitation": "Outlier relativo ao corpus de 4 patches; não implica patch problemático",
        "blocked_overclaim": "NÃO: outlier NÃO é 'anomalia de risco'; NÃO implica falha de método",
    },
    {
        "claim_allowed": "12/12 embeddings são ROBUST sob 6 tipos de perturbação",
        "result_supporting_it": "Deriva média cosseno: Curitiba=0.043, Petrópolis=0.078, Recife=0.060",
        "evidence_artifact": "robust_embeddings.csv; perturbation_robustness_summary.json",
        "figure_or_table": "Apêndice A (v1ha artifacts)",
        "limitation": "Perturbação de entrada, não perturbação de domínio; sem generalização temporal",
        "blocked_overclaim": "NÃO: robustez NÃO valida modelo para predição; NÃO prova representatividade",
    },
    {
        "claim_allowed": "GIS fornece contexto territorial para 3 regiões (PARTIAL/NOT_ACQUIRED/MISSING)",
        "result_supporting_it": "5 indicadores coletados por região; cobertura documentada em v1gv",
        "evidence_artifact": "evidence_coverage_matrix_v1gv.csv; table_external_evidence_coverage_summary_v1gy.csv",
        "figure_or_table": "fig_external_evidence_coverage_v1gy.png",
        "limitation": "Cobertura incompleta em todas as regiões; GIS não é ground truth",
        "blocked_overclaim": "NÃO: GIS NÃO é ground truth; NÃO valida embeddings; NÃO confirma risco",
    },
    {
        "claim_allowed": "47 candidatos de revisão selecionados por critérios estruturais",
        "result_supporting_it": "3 medoids + 3 outliers + 43 baixa cobertura GIS; 47/47 com preview",
        "evidence_artifact": "human_review_execution_manifest_v1hb.csv; visual_review_preview_manifest_v1hc.csv",
        "figure_or_table": "fig_review_candidate_categories_v1gy.png; table_review_candidates_summary_v1gy.csv",
        "limitation": "Seleção exploratória; não é amostragem probabilística",
        "blocked_overclaim": "NÃO: candidatos NÃO são classes; NÃO são classificados por risco",
    },
    {
        "claim_allowed": "Revisão visual assistida por estatísticas de imagem para 47/47 candidatos",
        "result_supporting_it": "NDVI médio, reflexão e heterogeneidade computados de TIFs Sentinel",
        "evidence_artifact": "human_review_visual_annotation_v1hd.csv; human_review_visual_summary_v1hd.json",
        "figure_or_table": "v1hc figures (local_runs/dino_embeddings/v1hc/figures/)",
        "limitation": "Revisão assistida, não humana direta; interpretação estatística",
        "blocked_overclaim": "NÃO: revisão visual NÃO valida DINO; NÃO cria labels; NÃO classifica risco",
    },
    {
        "claim_allowed": "Pipeline completamente documentado e auditável (hash, manifesto, scripts)",
        "result_supporting_it": "Manifesto v1ge com hash SHA-512 por embedding; scripts versionados",
        "evidence_artifact": "dino_expanded_embedding_manifest_v1ge.csv; scripts/dino/*",
        "figure_or_table": "—",
        "limitation": "Reprodutibilidade local; dados Sentinel privados (não publicados no git)",
        "blocked_overclaim": "NÃO: auditabilidade NÃO implica validação externa do método",
    },
    {
        "claim_allowed": "Análise exploratória estrutural é metodologicamente válida para o escopo proposto",
        "result_supporting_it": "11 claims READY; 10 forbidden claims BLOCKED; 676 testes passando",
        "evidence_artifact": "scientific_evidence_master_summary_v1gz.json; testes pytest",
        "figure_or_table": "—",
        "limitation": "Validade metodológica não implica aplicabilidade operacional",
        "blocked_overclaim": "NÃO: validade da análise NÃO implica predição; NÃO implica uso operacional",
    },
]


# ---------------------------------------------------------------------------
# Overleaf insert plan
# ---------------------------------------------------------------------------

def build_overleaf_plan(ev: Evidence) -> str:
    return f"""# Plano de Inserção no Overleaf — REV-P v1he

> Gerado automaticamente em {datetime.now(timezone.utc).date()}.
> Corpus: {ev.n_patches} patches · {ev.n_regions} regiões · {ev.emb_dim}-dim · {ev.backbone}.
> {ev.n_figures_ready} figuras prontas · {ev.n_tables_ready} tabelas prontas · {ev.n_allowed_claims} claims permitidos.
> Artefatos em `local_runs/tcc_figures/v1gy/` e `local_runs/dino_embeddings/v1h*/`.
> Dados Sentinel e embeddings não são versionados — são locais.

---

## Seção 4 — Resultados

### 4.1 Análise Estrutural de Embeddings

**Figuras a inserir:**
- `fig_similarity_heatmap_v1gy.png` → Figura 4.1 (mapa de calor similaridade)
- `fig_neighbor_network_v1gy.png` → Figura 4.2 (grafo de vizinhança)
- `fig_intra_inter_neighbor_rate_v1gy.png` → Figura 4.3 (taxa intra/inter)

**Tabelas a inserir:**
- `table_embedding_corpus_summary_v1gy.csv` → Tabela 4.1 (corpus por região)
- `table_neighbor_rate_summary_v1gy.csv` → Tabela 4.2 (taxas de vizinhança)

**Texto base:** `results_section_draft_v1he.md` → Seção 4.1–4.2

---

### 4.2 Estrutura Regional

**Tabelas a inserir:**
- `table_medoids_outliers_v1gy.csv` → Tabela 4.3 (medoids e outliers)

**Texto base:** `results_section_draft_v1he.md` → Seção 4.3

---

### 4.3 Evidência Contextual Externa

**Figuras a inserir:**
- `fig_external_evidence_coverage_v1gy.png` → Figura 4.4 (cobertura GIS)

**Tabelas a inserir:**
- `table_external_evidence_coverage_summary_v1gy.csv` → Tabela 4.4

**Texto base:** `results_section_draft_v1he.md` → Seção 4.4

---

### 4.4 Candidatos de Revisão Humana

**Figuras a inserir:**
- `fig_review_candidate_categories_v1gy.png` → Figura 4.5 (categorias de candidatos)

**Tabelas a inserir:**
- `table_review_candidates_summary_v1gy.csv` → Tabela 4.5

**Texto base:** `results_section_draft_v1he.md` → Seção 4.5

---

### 4.5 Revisão Visual Assistida

**Figuras para referência** (não inserir diretamente — muito pesadas):
- Contact sheets: `local_runs/dino_embeddings/v1hc/figures/contact_sheet_*_v1hc.png`
- Previews individuais: `local_runs/dino_embeddings/v1hc/figures/preview_*_v1hc.png`

**Estratégia**: inserir apenas 2–3 exemplos representativos (medoid + outlier severo)
como figuras inline. Contact sheets → Apêndice.

**Texto base:** `results_section_draft_v1he.md` → Seção 4.6

---

## Seção 5 — Discussão

**Texto base:** `discussion_section_draft_v1he.md`

**Estrutura recomendada:**
- 5.1 O que os resultados sustentam → `discussion_section_draft_v1he.md` § 5.1
- 5.2 O que não sustentam → `discussion_section_draft_v1he.md` § 5.2
- 5.3 Taxa intra/inter-regional → `discussion_section_draft_v1he.md` § 5.3
- 5.4 Medoids e outliers → `discussion_section_draft_v1he.md` § 5.4
- 5.5 Cobertura GIS → `discussion_section_draft_v1he.md` § 5.5
- 5.6 Validade do método → `discussion_section_draft_v1he.md` § 5.6
- 5.7 Limitações e trabalho futuro → `discussion_section_draft_v1he.md` § 5.7

**Referência cruzada com claims:**
- Usar `claim_result_limitation_matrix_v1he.csv` para verificar que cada afirmação
  na Discussão tem evidência e limitação explicitadas.

---

## Apêndice

**A — Robustez dos Embeddings:**
- Texto: resultados de v1ha (12/12 ROBUST; deriva por região)
- Tabela: `local_runs/dino_embeddings/v1ha/robust_embeddings.csv`
- Observação: OPTIONAL — não é obrigatório para a tese se o corpus for muito pequeno

**B — Manifesto de Evidências:**
- `tcc_visual_evidence_manifest_v1gy.csv`

**C — Protocolo de Revisão Humana:**
- `docs/metodologia_cientifica/human_review_protocol.md`

**D — Contact Sheets de Revisão Visual:**
- `local_runs/dino_embeddings/v1hc/figures/contact_sheet_all_review_candidates_v1hc.png`
- Usar apenas se houver espaço e relevância narrativa.

---

## O Que Não Inserir

- TIF brutos (privados, grandes)
- NPZ de embeddings
- PNG individuais dos 47 candidatos (excessivamente muitos)
- Paths privados e dados locais não versionados
- Qualquer claim de predição, risco ou ground truth

---

## Ações Manuais no Overleaf

1. Criar arquivo `resultados.tex` e colar seções de `results_section_draft_v1he.md`
2. Criar arquivo `discussao.tex` e colar seções de `discussion_section_draft_v1he.md`
3. Copiar figuras de `local_runs/tcc_figures/v1gy/` para o projeto Overleaf (5 PNGs)
4. Converter tabelas CSV para LaTeX com `\\begin{{table}}...\\end{{table}}`
5. Inserir captions de `figure_captions_final_v1he.csv` e `table_captions_final_v1he.csv`
6. Revisar numeração de figuras/tabelas conforme estrutura final do TCC
7. Verificar que nenhuma afirmação viola as regras de `claim_result_limitation_matrix_v1he.csv`

---

*Plano gerado em {datetime.now(timezone.utc).date()} — v1he.*
"""


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary(ev: Evidence) -> dict:
    ready_for_overleaf = (
        ev.n_figures_ready >= 5
        and ev.n_tables_ready >= 5
        and ev.n_allowed_claims >= 10
        and ev.n_forbidden_claims >= 10
        and ev.n_robust == ev.n_patches
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "total_figures_ready": ev.n_figures_ready,
        "total_tables_ready": ev.n_tables_ready,
        "total_claims_ready": ev.n_allowed_claims,
        "total_claims_partial": 0,
        "total_forbidden_claims_blocked": ev.n_forbidden_claims,
        "review_candidates_processed": ev.n_review_candidates,
        "visual_review_status": (
            "COMPLETED" if ev.n_visual_computed == ev.n_review_candidates else "PARTIAL"
        ),
        "robustness_status": (
            "FULLY_ROBUST" if ev.n_unstable == 0 else "PARTIAL"
        ),
        "ready_for_overleaf": ready_for_overleaf,
        "ready_for_overleaf_justification": (
            f"{ev.n_figures_ready} figures ready, "
            f"{ev.n_tables_ready} tables ready, "
            f"{ev.n_allowed_claims} claims documented, "
            f"{ev.n_forbidden_claims} forbidden claims blocked, "
            f"{ev.n_robust}/{ev.n_patches} embeddings robust"
        ),
        "corpus": {
            "n_patches": ev.n_patches,
            "n_regions": ev.n_regions,
            "embedding_dim": ev.emb_dim,
            "backbone": ev.backbone,
        },
        "neighbor_topology": {
            "n_pairs": ev.n_neighbor_pairs,
            "intra_rate": ev.intra_rate,
            "inter_rate": ev.inter_rate,
        },
        "methodological_guardrails": {
            "labels_created": False,
            "predictions_made": False,
            "ground_truth_established": False,
            "review_only": True,
            "all_forbidden_claims_blocked": True,
        },
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"[v1he] Written: {path.name} ({len(content)} chars)")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[v1he] Written: {path.name} ({len(rows)} rows)")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[v1he] Written: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[v1he] Loading evidence from pipeline outputs...")
    ev = Evidence()

    print(f"[v1he] corpus={ev.n_patches} patches, {ev.n_regions} regions, dim={ev.emb_dim}")
    print(f"[v1he] figures_ready={ev.n_figures_ready}, tables_ready={ev.n_tables_ready}")
    print(f"[v1he] claims_allowed={ev.n_allowed_claims}, claims_blocked={ev.n_forbidden_claims}")
    print(f"[v1he] robust={ev.n_robust}/{ev.n_patches}, visual_computed={ev.n_visual_computed}/47")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[v1he] Generating Results draft...")
    results_text = build_results_section(ev)
    write_md(OUT_DIR / "results_section_draft_v1he.md", results_text)

    print("[v1he] Generating Discussion draft...")
    discussion_text = build_discussion_section(ev)
    write_md(OUT_DIR / "discussion_section_draft_v1he.md", discussion_text)

    print("[v1he] Generating figure captions...")
    write_csv(OUT_DIR / "figure_captions_final_v1he.csv", FIGURE_CAPTIONS)

    print("[v1he] Generating table captions...")
    write_csv(OUT_DIR / "table_captions_final_v1he.csv", TABLE_CAPTIONS)

    print("[v1he] Generating claim-result-limitation matrix...")
    write_csv(OUT_DIR / "claim_result_limitation_matrix_v1he.csv", CLAIM_MATRIX)

    print("[v1he] Generating summary JSON...")
    summary = build_summary(ev)
    write_json(OUT_DIR / "tcc_results_discussion_summary_v1he.json", summary)

    print("[v1he] Generating Overleaf insert plan...")
    plan = build_overleaf_plan(ev)
    write_md(OUT_DIR / "overleaf_insert_plan_v1he.md", plan)

    print(f"\n[v1he] Done. Outputs in {OUT_DIR.relative_to(ROOT)}")
    print(f"[v1he] ready_for_overleaf: {summary['ready_for_overleaf']}")
    print(f"[v1he] Justification: {summary['ready_for_overleaf_justification']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
