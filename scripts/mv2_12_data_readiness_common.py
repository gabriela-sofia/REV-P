"""Núcleo compartilhado do marco MV2-12 — Data Intake, Evidence Backlog & Download Readiness.

Este módulo NÃO treina modelo, NÃO cria label, NÃO cria silver e NÃO inicia sandbox.
Ele consolida o que já foi diagnosticado pela auditoria MV2-10 (presente em
``outputs_public/audits/mv2_10_gap_audit/``) e confirmado pela programação MV2-11
(branch ``marco/mv2-11-rebalanceamento-representacional-regional``) numa frente
objetiva de dados: o que existe, o que falta, o que precisa ser baixado, recuperado
localmente, digitalizado ou solicitado por portal, e qual dia/gate cada dado destrava.

Princípios fail-closed preservados:
- ausência de evidência NUNCA é tratada como negativo;
- candidato de IA / candidato regional NUNCA é promovido a silver formal ou label;
- PNG/JPG/NPZ/renderização visual NUNCA é tratado como raster Sentinel espectral;
- dado bruto pesado NUNCA é publicado em ``outputs_public``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MV2_10_AUDIT_DIR = PROJECT_ROOT / "outputs_public" / "audits" / "mv2_10_gap_audit"
MV2_10_EXTERNAL_BACKLOG = MV2_10_AUDIT_DIR / "mv2_10_external_data_backlog.csv"
MV2_10_GAP_MATRIX = MV2_10_AUDIT_DIR / "mv2_10_gap_matrix.csv"
MV2_10_SCHEDULE = MV2_10_AUDIT_DIR / "mv2_10_schedule_alignment.csv"

EVENT_GEOJSON_DIR = (
    PROJECT_ROOT
    / "docs"
    / "protocolo_c"
    / "v2aq_event_geometry_patch_link"
    / "geojson_candidates"
)

CHARTER_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "external_sources"
    / "recife_minimal_tp"
    / "event_polygon_REC_2022_05_24_30"
    / "charter758"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_readiness"

# Diretório local-only (git-ignored) para onde QUALQUER bruto baixado deve ir.
# Nunca apontar download de bruto para outputs_public.
QUARANTINE_DIR_REL = "data_quarantine/mv2_12_intake"
EXTERNAL_LOCAL_DIR_REL = "external_data_local/mv2_12"

# Extensões consideradas dado bruto pesado — proibidas em outputs_public.
HEAVY_RAW_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".geotiff",
    ".jp2",
    ".safe",
    ".img",
    ".cog",
    ".npz",
    ".npy",
    ".nc",
    ".hdf",
    ".he5",
    ".zip",
    ".gpkg",
    ".shp",
}

# Extensões de renderização visual — NUNCA confundir com raster espectral nativo.
VISUAL_RENDER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Extensões de raster espectral nativo Sentinel.
NATIVE_SPECTRAL_EXTENSIONS = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog"}

# ---------------------------------------------------------------------------
# Taxonomias obrigatórias (MV2-12)
# ---------------------------------------------------------------------------

STATUS_VALUES = {
    "PRESENTE_VALIDADO",
    "PRESENTE_SEM_VINCULO_FORTE",
    "PRESENTE_SEM_METADATA",
    "FALTANTE_RECUPERAR_LOCAL",
    "FALTANTE_BAIXAR_EXTERNO",
    "FALTANTE_DIGITALIZAR",
    "FALTANTE_PORTAL_MANUAL",
    "FALTANTE_INDISPONIVEL",
    "DESCONHECIDO",
    "NAO_NECESSARIO_AGORA",
}

PURPOSE_VALUES = {
    "TEMPORALIDADE",
    "CLOUD_COVER",
    "SCENE_LINEAGE",
    "RASTER_SENTINEL_ESPECTRAL",
    "GEOMETRIA_EVENTO",
    "OVERLAY_PATCH_EVENTO",
    "REBALANCEAMENTO_REGIONAL",
    "EVIDENCIA_OBSERVACIONAL",
    "NEGATIVO_FORMAL",
    "SPLIT_ANTI_LEAKAGE",
    "REPRODUTIBILIDADE",
    "CONTEXTO_TERRITORIAL",
}

# ---------------------------------------------------------------------------
# Fatos consolidados das auditorias anteriores (proveniência documentada).
# Fonte: MV2-10 (sumário/gap_matrix/schedule presentes localmente) e MV2-11
# (mv2_11_regional_rebalancing_summary.json, regional_representation_coverage.csv).
# Estes números são CONTAGENS review-only; não criam label, silver nem treino.
# ---------------------------------------------------------------------------

# Cobertura por cidade pós-MV2-11 (regional_representation_coverage.csv).
REGIONAL_COVERAGE = {
    # cidade: (total_contract_assets, physical_found, canonical_visual, dinov2_total)
    "Curitiba": (43, 43, 43, 47),
    "Recife": (37, 2, 2, 6),
    "Petropolis": (48, 3, 3, 7),
}

# Mínimo de canônicos por cidade para considerar o corpus balanceado (review-only).
BALANCED_MIN_CANONICAL_PER_CITY = 37

# Estado pós-MV2-11 (mv2_11_regional_rebalancing_summary.json).
MV2_11_FACTS = {
    "total_contract_assets": 128,
    "dinov2_total_after_mv2_11": 60,
    "canonical_visual_assets_total": 48,
    "regional_asset_candidates_found": 131,
    "regional_asset_candidates_strong": 0,
    "coverage_bias_risk": "ALTO",
    "city_confounder_risk": "ALTO",
    "labels_created": 0,
    "can_train": False,
    "sandbox_status": "bloqueado",
    "ground_truth_operacional_status": "ausente",
}

# Status de cada dia do cronograma após MV2-11 (cronograma_after_regional_rebalancing.csv).
SCHEDULE_DAY_STATUS = {
    "DIA_08": "PARCIAL_SUBSET_BALANCEADO_MAS_FULL_ENVIESADO",
    "DIA_10": "BLOQUEADO_SEM_BASELINE_ESPECTRAL_SENTINEL",
    "DIA_18": "BLOQUEADO_EVIDENCIA_OBSERVACIONAL",
    "DIA_19": "BLOQUEADO_SILVER_FORMAL_ZERO",
    "DIA_21": "BLOQUEADO_SPLITS_E_NEGATIVOS_FORMAIS",
    "DIA_22": "BLOQUEADO_SANDBOX",
}

# Fenômeno por região — Petrópolis é cohort SEPARADO de mass_movement (não misturar
# com flood). Mantido explícito para impedir cross-cohort em qualquer derivação.
REGION_PHENOMENON = {
    "Recife": "flood",
    "Curitiba": "urban_flooding",
    "Petropolis": "mass_movement",
}

# ---------------------------------------------------------------------------
# Helpers de IO
# ---------------------------------------------------------------------------


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, columns: list[str], rows: Sequence[Mapping[str, object]]
) -> None:
    """Escreve CSV leve com newline final único e quoting mínimo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_status(value: str) -> str:
    if value not in STATUS_VALUES:
        raise ValueError(f"current_status inválido fora da taxonomia MV2-12: {value!r}")
    return value


def validate_purpose(value: str) -> str:
    for part in value.split("|"):
        if part and part not in PURPOSE_VALUES:
            raise ValueError(
                f"scientific_purpose inválido fora da taxonomia MV2-12: {part!r}"
            )
    return value


def infer_region_from_text(text: str) -> str:
    low = text.lower()
    if "recife" in low or "/rec" in low or "_rec" in low or "rec_" in low:
        return "Recife"
    if "petropolis" in low or "petrópolis" in low or "/pet" in low or "pet_" in low:
        return "Petropolis"
    if "curitiba" in low or "/ctb" in low or "ctb_" in low or "/cur" in low:
        return "Curitiba"
    return "DESCONHECIDO"
