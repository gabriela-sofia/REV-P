"""REV-P v1hb: Human Review Execution Package.

Transforms v1gw candidates into an executable, auditable review package.
Organizes 47 candidates, generates annotation templates, and prepares
discussion inputs — all review-only, no labels or classes created.

Review is interpretative: visual inspection + structural evidence
interpretation, NOT validation or classification.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1hb"
PHASE_NAME = "HUMAN_REVIEW_EXECUTION_PACKAGE"

DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / PHASE

# Input directories
V1GW_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gw"
V1GU_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gu"
V1GV_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gv"
V1GE_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1ge"

# Forbidden terms in review context
FORBIDDEN_REVIEW_TERMS = {
    "prediction", "predictive", "detect", "detection", "classify", "classification",
    "class", "label", "risk", "vulnerability", "ground truth", "ground-truth",
    "validate", "validation", "accuracy", "performance", "train", "supervised",
    "target", "ground truth", "causal", "proven", "proves", "validate",
}

# Allowed review interpretations
ALLOWED_REVIEW_SCOPE = [
    "Visual pattern observation",
    "Structural coherence with DINO neighbors",
    "GIS contextual consistency",
    "Data quality assessment",
    "Regional heterogeneity interpretation",
    "Embedding space topology understanding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1hb: Human review execution package."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_candidates() -> list[dict[str, str]]:
    """Load 47 candidates from v1gw."""
    return read_csv(V1GW_DIR / "review_candidates_v1gw.csv")


def load_embedding_evidence() -> dict[str, Any]:
    """Load v1gu evidence (medoids, outliers, neighbors)."""
    summary = read_json(V1GU_DIR / "embedding_regional_summary_v1gu.json")
    return {
        "medoids_outliers": summary.get("medoids_and_outliers", {}),
        "neighbors": read_csv(V1GU_DIR / "embedding_neighbors_v1gu.csv"),
        "intra_inter": summary.get("intra_inter_region_analysis", {}),
    }


def load_gis_evidence() -> dict[str, str]:
    """Load v1gv GIS coverage matrix."""
    matrix = read_csv(V1GV_DIR / "evidence_coverage_matrix_v1gv.csv")
    return {row["canonical_patch_id"]: row for row in matrix if "canonical_patch_id" in row}


def build_execution_manifest(
    candidates: list[dict[str, str]],
    embedding_evidence: dict[str, Any],
    gis_evidence: dict[str, str],
) -> list[dict[str, object]]:
    """Build detailed execution manifest per candidate."""
    rows: list[dict[str, object]] = []

    for idx, candidate in enumerate(candidates, 1):
        patch_id = candidate.get("canonical_patch_id", "")
        region = candidate.get("region", "")
        categories = candidate.get("categories", "").split(";")
        categories = [c.strip() for c in categories if c.strip()]

        # Embedding evidence
        embedding_source = "v1gu"
        if "medoid_regional" in categories or "outlier_structural" in categories:
            medoid_info = embedding_evidence.get("medoids_outliers", {}).get(region, {})
            embedding_evidence_source = f"Medoid: {medoid_info.get('medoid', '')}, Outliers: {medoid_info.get('outliers', [])}"
        else:
            embedding_evidence_source = "Intra/inter-region neighborhood analysis"

        # GIS evidence
        gis_row = gis_evidence.get(patch_id, {})
        gis_indicators = [k for k, v in gis_row.items() if v and k != "canonical_patch_id" and k != "region"]
        gis_evidence_source = f"Coverage: {len(gis_indicators)}/13 indicators"

        rows.append({
            "review_item_id": f"HRE{idx:03d}",
            "canonical_patch_id": patch_id,
            "region": region,
            "candidate_category": "; ".join(categories),
            "selection_reason": candidate.get("selection_basis", ""),
            "embedding_evidence_source": embedding_source,
            "gis_evidence_source": gis_evidence_source,
            "suggested_visual_check": "Sentinel RGB + NDVI; Land use, water, infrastructure",
            "reviewer_status": "PENDING",
            "reviewer_observation": "",
            "methodological_interpretation": "Structural coherence + contextual observations",
            "allowed_claim_scope": "Visual pattern; Regional heterogeneity; Contextual consistency",
            "forbidden_claim_warning": "NO prediction, detection, classification, or vulnerability labeling",
        })

    return rows


def build_annotation_template() -> list[dict[str, str]]:
    """Build annotation template for manual review."""
    # Template has placeholder rows for each review item
    # Reviewer fills in the blanks
    template = []

    annotation_fields = [
        "review_item_id",
        "canonical_patch_id",
        "reviewer_name_or_initials",
        "review_date",
        "visual_pattern_notes",
        "surrounding_context_notes",
        "external_evidence_notes",
        "uncertainty_level",
        "usable_in_discussion",
        "discussion_note",
        "no_label_created_confirmed",
        "no_prediction_claim_confirmed",
    ]

    # Create empty template structure
    for i in range(1, 48):  # 47 items
        template.append({
            "review_item_id": f"HRE{i:03d}",
            "canonical_patch_id": "",
            "reviewer_name_or_initials": "",
            "review_date": "",
            "visual_pattern_notes": "",
            "surrounding_context_notes": "",
            "external_evidence_notes": "",
            "uncertainty_level": "low | medium | high",
            "usable_in_discussion": "yes | no | conditional",
            "discussion_note": "",
            "no_label_created_confirmed": "yes | no",
            "no_prediction_claim_confirmed": "yes | no",
        })

    return template, annotation_fields


def build_category_summary(candidates: list[dict[str, str]]) -> list[dict[str, object]]:
    """Summarize candidates by category."""
    category_counts: dict[str, int] = defaultdict(int)
    category_patches: dict[str, list[str]] = defaultdict(list)
    category_regions: dict[str, set[str]] = defaultdict(set)

    for candidate in candidates:
        patch_id = candidate.get("canonical_patch_id", "")
        region = candidate.get("region", "")
        categories = candidate.get("categories", "").split(";")

        for cat in categories:
            cat = cat.strip()
            if cat:
                category_counts[cat] += 1
                category_patches[cat].append(patch_id)
                category_regions[cat].add(region)

    rows = []
    for category in sorted(category_counts.keys()):
        rows.append({
            "category": category,
            "n_patches": category_counts[category],
            "regions": "; ".join(sorted(category_regions[category])),
            "interpretation_allowed": "Regional heterogeneity; structural variation; contextual observation",
            "interpretation_forbidden": "Risk classification; vulnerability prediction; ground truth",
            "review_focus": get_category_focus(category),
        })

    return rows


def get_category_focus(category: str) -> str:
    """Get suggested review focus per category."""
    focus_map = {
        "medoid_regional": "Central representative: how typical is this for the region?",
        "outlier_structural": "Structural outlier: what makes it structurally different?",
        "coverage_external_low": "Low GIS data: visual clues visible despite data gaps?",
        "coherence_embedding_gis_high": "Embedding-GIS match: do structures align visually?",
        "conflict_embedding_gis": "Embedding-GIS divergence: why might signals differ?",
        "bridge_inter_regional": "Inter-regional link: do visual features suggest connection?",
        "geometry_complete": "Complete geometry: use as reference or comparison point?",
        "geometry_incomplete": "Incomplete geometry: data quality assessment",
    }
    return focus_map.get(category, "General visual and contextual assessment")


def build_discussion_inputs(
    candidates: list[dict[str, str]],
    embedding_evidence: dict[str, Any],
) -> list[dict[str, str]]:
    """Build table of findings ready for TCC Discussion."""
    rows: list[dict[str, str]] = []

    # Group by region and category
    for region in ["Curitiba", "Petrópolis", "Recife"]:
        region_candidates = [c for c in candidates if c.get("region") == region]

        for category in ["medoid_regional", "outlier_structural", "coverage_external_low"]:
            category_candidates = [
                c for c in region_candidates
                if category in c.get("categories", "")
            ]

            if category_candidates:
                rows.append({
                    "finding_id": f"DISC_{region[:3]}_{category[:3]}",
                    "evidence_type": f"{category.replace('_', ' ').title()} ({region})",
                    "finding_summary": f"{len(category_candidates)} patches in {category.replace('_', ' ')} category",
                    "supporting_artifact": f"v1gu summary + v1gy figures",
                    "interpretation_for_discussion": f"Structural analysis reveals {category.replace('_', ' ')} patterns in {region}",
                    "limitation": "Based on 4 patches per region; exploratory scope",
                    "claim_allowed": "yes",
                    "claim_blocked": "no_prediction | no_classification | no_ground_truth",
                })

    return rows


def build_protocol_document() -> str:
    """Build human review protocol documentation."""
    return """# Human Review Protocol — REV-P v1hb

## Propósito

A revisão humana é uma etapa de interpretação estrutural exploratória. Reviewers inspecionam
visualmente os patches candidatos para compreender:

- Padrões visuais coerentes com vizinhança estrutural (DINO);
- Consistência entre estrutura de embedding e evidência GIS;
- Variação regional na representação de paisagem;
- Qualidade de dados e artefatos de aquisição.

**NÃO é**: Validação de modelo, classificação de risco, atribuição de label, predição, detecção.

## O Que o Reviewer Pode Afirmar

✓ Observação de padrão visual
✓ Coerência estrutural com vizinhos DINO
✓ Consistência contextual (GIS ou visual)
✓ Qualidade de dados (nuvens, sombras, artifacts)
✓ Heterogeneidade regional
✓ Incerteza e limitações observadas

## O Que o Reviewer NÃO Pode Afirmar

✗ Predição de enchente ou inundação
✗ Detecção de risco ou vulnerabilidade
✗ Classificação de tipo ou categoria operacional
✗ Validação de DINO contra ground truth
✗ Performance ou acurácia de modelo
✗ Causalidade ou determinismo
✗ Generalização além da amostra (12/128 patches)

## Categorias de Revisão

### Medoid Regional
Patch mais central na distribuição de embeddings por região.

**Foco**: Quão representativo é este patch? Que características tipificam a região?

**Não é**: Rótulo de "melhor patch" ou "patch de referência operacional".

### Outlier Estrutural
Patch periférico na distribuição de embeddings regionais.

**Foco**: O que torna este patch estruturalmente distinto? Há variação visual correspondente?

**Não é**: Patch "anômalo" ou "problemático".

### Cobertura GIS Baixa
Patches com poucos indicadores GIS disponíveis.

**Foco**: Como a interpretação muda sem dados contextuais? Há padrões visuais compensatórios?

**Não é**: Patches de qualidade inferior ou inúteis.

### Coherência Embedding-GIS
Patches com sinais concordantes entre estrutura DINO e cobertura GIS.

**Foco**: Estrutura de embedding alinha com estrutura visual-territorial observável?

**Não é**: Validação de DINO; nem GIS é "ground truth".

### Conflito Embedding-GIS
Patches com sinais divergentes entre DINO e GIS.

**Foco**: Por que a divergência? Dados GIS incompletos? Heterogeneidade visual real?

**Não é**: Erro de modelo ou falha de método.

## Protocolo de Anotação

Para cada patch:

1. **Inspeção Visual**
   - Abrir Sentinel RGB + NDVI
   - Observar: tipo de uso de solo, água, infraestrutura, qualidade de dados
   - Registrar padrões observáveis

2. **Contexto Geográfico**
   - Proximity a corpos d'água (qualitativa)
   - Contexto topográfico se disponível
   - Contexto administrativo/regional

3. **Sinais Estruturais**
   - DINO vizinhos: visualmente similares?
   - Cobertura GIS: faz sentido dado o padrão visual?
   - Heterogeneidade regional observada?

4. **Qualidade de Dados**
   - Cobertura de nuvem
   - Sombras ou artifacts
   - Completude de feature

5. **Incerteza**
   - Como reviewers avaliam confiança nas observações?
   - Que fatores limitam interpretação?

## Campos de Anotação

- **reviewer_name_or_initials**: Identificação anônima (ex: REV-01)
- **review_date**: Data da anotação
- **visual_pattern_notes**: O que reviewer observa no RGB/NDVI
- **surrounding_context_notes**: Contexto geográfico observado
- **external_evidence_notes**: Consistência com indicadores GIS
- **uncertainty_level**: low | medium | high
- **usable_in_discussion**: yes | no | conditional (patch pode informar TCC?)
- **discussion_note**: Como este patch informa a Discussão?
- **no_label_created_confirmed**: Confirmação: nenhum label foi criado
- **no_prediction_claim_confirmed**: Confirmação: nenhuma claim preditiva foi feita

## Síntese para a Discussão do TCC

Revisão humana alimenta a Seção 5 (Discussão) com:

1. Observações de padrão visual por categoria
2. Coerência estrutural entre DINO e paisagem visual
3. Variação regional documentada
4. Limitações e incertezas
5. **Sem claims preditivas ou classificatórias**

## Garantias Metodológicas

- Revisão humana permanece **interpretativa**, não operacional
- Nenhum **label, classe ou target** é criado
- Nenhuma **validação supervisionada** é executada
- Nenhum **ground truth** é estabelecido
- **Toda incerteza é documentada**
- **Corpus permanece pequeno** (12/128); generalização não é afirmada

---

**Última atualização**: 2026-05-18
**Fase**: v1hb
**Status**: Review-only; interpretative; audit-ready
"""


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        prepare_output_dir(output_dir, args.force)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load all data
    candidates = load_candidates()
    embedding_evidence = load_embedding_evidence()
    gis_evidence = load_gis_evidence()

    # Build outputs
    execution_manifest = build_execution_manifest(candidates, embedding_evidence, gis_evidence)
    annotation_template, template_fields = build_annotation_template()
    category_summary = build_category_summary(candidates)
    discussion_inputs = build_discussion_inputs(candidates, embedding_evidence)
    protocol_doc = build_protocol_document()

    # Write outputs
    write_csv(
        output_dir / "human_review_execution_manifest_v1hb.csv",
        execution_manifest,
        [
            "review_item_id", "canonical_patch_id", "region", "candidate_category",
            "selection_reason", "embedding_evidence_source", "gis_evidence_source",
            "suggested_visual_check", "reviewer_status", "reviewer_observation",
            "methodological_interpretation", "allowed_claim_scope", "forbidden_claim_warning",
        ],
    )

    write_csv(
        output_dir / "human_review_annotation_template_v1hb.csv",
        annotation_template,
        template_fields,
    )

    write_csv(
        output_dir / "human_review_category_summary_v1hb.csv",
        category_summary,
        ["category", "n_patches", "regions", "interpretation_allowed", "interpretation_forbidden", "review_focus"],
    )

    write_csv(
        output_dir / "human_review_discussion_inputs_v1hb.csv",
        discussion_inputs,
        ["finding_id", "evidence_type", "finding_summary", "supporting_artifact", "interpretation_for_discussion", "limitation", "claim_allowed", "claim_blocked"],
    )

    write_md(
        output_dir / "human_review_protocol_v1hb.md",
        protocol_doc,
    )

    print(f"[OK] Human review execution package complete: {output_dir}")
    print(f"  - Execution manifest: {len(execution_manifest)} items")
    print(f"  - Annotation template: {len(annotation_template)} rows")
    print(f"  - Category summary: {len(category_summary)} categories")
    print(f"  - Discussion inputs: {len(discussion_inputs)} findings")
    print(f"  - Protocol document: v1hb-specific guidance")

    return 0


if __name__ == "__main__":
    sys.exit(main())
