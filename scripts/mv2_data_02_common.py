"""Núcleo compartilhado do marco MV2-DATA-02 — Sentinel API & Raster Access Harness.

Infraestrutura real de acesso a API/raster Sentinel (GEE / CDSE STAC / CDSE OData) sobre os
128 targets do MV2-DATA-01, com guardrails fail-closed FORTES:
- metadados só são consultados se a config permitir explicitamente (allow_network +
  allow_metadata_calls) e houver credencial em variável de ambiente;
- download de raster só existe em modo canary/local-only/opt-in, jamais em lote;
- nenhum dos 128 targets é baixado sem scene_id/datetime/binding forte;
- nenhum raster vai para outputs_public; raster só em diretório privado git-ignored;
- nenhum segredo/token é gravado em arquivo público (credencial só por env var);
- Dia 10 não é desbloqueado sem raster privado validado.

Reusa IO e classificadores do MV2-13 (uma só fonte de verdade).
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (  # noqa: F401  (reuso/reexport intencional)
    PROJECT_ROOT,
    S2_SCENE_RE,
    classify_datetime,
    clean,
    crs_status,
    read_csv_rows,
    write_csv,
    write_json,
)

PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_api_raster_harness"
DATA01_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_gee_lineage_targets"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

# Diretórios privados/local-only aceitos para raster canary (todos git-ignored).
PRIVATE_DIR_ALLOWED_PREFIXES = ("local_only", "data_local", "private_outputs")

OFFICIAL_ANCHOR_REGISTRY = PROJECT_ROOT / "datasets" / "official_anchor_sentinel_patch_registry.csv"

DATA01_TARGETS = DATA01_DIR / "mv2_data_01_lineage_targets.csv"
DATA01_STAC_PLAN = DATA01_DIR / "mv2_data_01_stac_odata_metadata_plan.csv"
DATA01_GEE_PLAN = DATA01_DIR / "mv2_data_01_gee_metadata_query_plan.csv"

BRANCH_EXECUTADA = "analysis/temporal-asset-readiness-mv1"
BRANCH_ESPERADA = "marco/mv2-data-vertente-b-desbloqueio-dados"

DEFAULT_S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# Bandas mínimas para um crop canary Sentinel-2.
CANARY_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
CANARY_SCL = "SCL"

# Metadados esperados de uma consulta GEE metadata-only.
GEE_METADATA_FIELDS = [
    "PRODUCT_ID",
    "MGRS_TILE",
    "system:time_start",
    "CLOUDY_PIXEL_PERCENTAGE",
    "CLOUD_COVERAGE_ASSESSMENT",
    "NODATA_PIXEL_PERCENTAGE",
    "DATATAKE_IDENTIFIER",
    "GENERATION_TIME",
    "SPACECRAFT_NAME",
    "SENSING_ORBIT_NUMBER",
    "PROCESSING_BASELINE",
]

PROVIDERS = ["GEE", "CDSE_STAC", "CDSE_ODATA"]

API_LINEAGE_STATUSES = {
    "API_LINEAGE_RESOLVED_STRONG",
    "API_LINEAGE_RESOLVED_PARTIAL",
    "API_LINEAGE_REVIEW_REQUIRED",
    "API_LINEAGE_CONFLICT",
    "API_LINEAGE_NOT_EXECUTED",
    "API_LINEAGE_NOT_FOUND",
    "API_LINEAGE_INVALID",
}

RISK_TYPES = [
    "RISK_API_WITHOUT_AUTH",
    "RISK_NETWORK_CALL_UNCONTROLLED",
    "RISK_STAC_WITHOUT_LINEAGE",
    "RISK_ODATA_DOWNLOAD_WITHOUT_REVIEW",
    "RISK_RASTER_IN_OUTPUTS_PUBLIC",
    "RISK_BULK_DOWNLOAD_TOO_EARLY",
    "RISK_CANARY_WITHOUT_STRONG_LINEAGE",
    "RISK_CROP_WITHOUT_CRS",
    "RISK_MISSING_SCL",
    "RISK_PNG_AS_RASTER",
    "RISK_DAY10_FALSE_UNLOCK",
    "RISK_SECRET_LEAK",
    "RISK_PRIVATE_PATH_LEAK",
    "RISK_COST_QUOTA_RATE_LIMIT",
]


def ensure_public_dir() -> Path:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    return PUBLIC_DIR


def load_targets() -> list[dict[str, str]]:
    """Lê os 128 targets do MV2-DATA-01 (entrada obrigatória)."""
    return read_csv_rows(DATA01_TARGETS)


def is_private_path(path: str) -> bool:
    """True se o caminho começa por um prefixo privado/local-only aceito."""
    norm = clean(path).replace("\\", "/").lstrip("./")
    return any(
        norm == pref or norm.startswith(pref + "/")
        for pref in PRIVATE_DIR_ALLOWED_PREFIXES
    )


def load_official_strong_anchors() -> list[dict[str, str]]:
    """Âncoras oficiais com lineage Sentinel FORTE e explícito (scene_id documentado).

    Estas vêm do registry oficial (não dos 128 targets) e têm scene_id/datetime/tile/
    coleção/CRS documentados — fonte legítima de lineage forte para um canary, distinto
    de qualquer auto-join por tile/zona.
    """
    rows = read_csv_rows(OFFICIAL_ANCHOR_REGISTRY)
    strong: list[dict[str, str]] = []
    for r in rows:
        scene = clean(r.get("scene_id_sanitized"))
        scene_date = clean(r.get("scene_date"))
        crs = clean(r.get("crs"))
        collection = clean(r.get("gee_collection"))
        # Forte só com scene_id no formato granule S2 + data válida não-futura + CRS.
        if not S2_SCENE_RE.search(scene):
            continue
        if classify_datetime(scene_date[:10])[1] != "VALID_SINGLE":
            continue
        if crs_status(crs) == "ABSENT" or not collection:
            continue
        strong.append(r)
    return strong
