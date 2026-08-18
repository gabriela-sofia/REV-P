"""Núcleo compartilhado do marco MV2-DATA-04 — Corpus Metadata Probe.

Sai do canary técnico Protocolo-C (DATA-03) e começa a mirar o CORPUS oficial de 128 targets,
ainda em modo metadata-only: sem raster, sem crop, sem download, sem feature espectral. O
ponto central é PROVAR que, sem janela temporal por target, toda consulta GEE/STAC é
bloqueada de forma limpa (nunca query aberta ampla), e que OData bloqueia sem product_id.

Fail-closed:
- sem config local privada -> nenhuma chamada de rede (herdado do DATA-03);
- sem janela temporal -> GEE/STAC bloqueiam (BLOCKED_NO_TEMPORAL_WINDOW), nunca query aberta;
- sem product_id -> OData bloqueia (BLOCKED_NO_PRODUCT_ID);
- canary oficial Protocolo-C nunca entra no batch do corpus;
- cidade/tile/região nunca é vínculo forte; T23KPR nunca auto-vinculado ao corpus;
- Dia 10/sandbox/treino permanecem bloqueados.

Reusa config/env do DATA-03 e IO/classificadores do MV2-13 (uma só fonte de verdade).
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_03_common import (  # noqa: F401  (reuso/reexport intencional)
    CREDENTIAL_ENV_VARS,
    DEFAULT_S2_COLLECTION,
    ENV_ALLOW_METADATA_CALLS,
    ENV_ALLOW_NETWORK,
    ENV_ALLOW_RASTER_CANARY,
    ENV_ALLOW_RASTER_DOWNLOAD,
    GEE_METADATA_FIELDS,
    OPTIONAL_LIBS,
    PROJECT_ROOT,
    classify_datetime,
    clean,
    crs_status,
    effective_flags,
    env_present,
    ensure_public_dir as _ensure_data03_dir,
    lib_available,
    load_official_strong_anchors,
    metadata_blocked,
    read_csv_rows,
    write_csv,
    write_json,
)

PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_corpus_metadata_probe"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

DATA01_TARGETS = PROJECT_ROOT / "outputs_public" / "mv2_data_gee_lineage_targets" / "mv2_data_01_lineage_targets.csv"

BRANCH_EXECUTADA = "analysis/temporal-asset-readiness-mv1"

# Batch default: no máximo 15, balanceado por região (até 5 cada).
MAX_BATCH = 15
MAX_PER_REGION = 5
BATCH_REGIONS = ["Recife", "Petropolis", "Curitiba"]

CORPUS_LINEAGE_STATUSES = {
    "CORPUS_LINEAGE_RESOLVED_STRONG",
    "CORPUS_LINEAGE_RESOLVED_PARTIAL",
    "CORPUS_LINEAGE_REVIEW_REQUIRED",
    "CORPUS_LINEAGE_BLOCKED_NO_TEMPORAL_WINDOW",
    "CORPUS_LINEAGE_BLOCKED_NO_PRODUCT_ID",
    "CORPUS_LINEAGE_NOT_EXECUTED",
    "CORPUS_LINEAGE_CONFLICT",
    "CORPUS_LINEAGE_INVALID",
}

RISK_TYPES = [
    "RISK_METADATA_PROBE_WITHOUT_TEMPORAL_WINDOW",
    "RISK_OPEN_ENDED_STAC_QUERY",
    "RISK_CITY_TILE_MATCH",
    "RISK_CANARY_AS_CORPUS",
    "RISK_ODATA_WITHOUT_PRODUCT_ID",
    "RISK_SECRET_LEAK",
    "RISK_PRIVATE_PATH_LEAK",
    "RISK_RASTER_DOWNLOAD_TOO_EARLY",
    "RISK_DAY10_FALSE_UNLOCK",
    "RISK_SANDBOX_FALSE_UNLOCK",
]


def ensure_public_dir() -> Path:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    return PUBLIC_DIR


def load_corpus_targets() -> list[dict[str, str]]:
    """Os 128 targets do corpus (DATA-01)."""
    return read_csv_rows(DATA01_TARGETS)


def official_canary_ids() -> set[str]:
    """patch/refpatch ids das âncoras oficiais Protocolo-C (nunca entram no corpus batch)."""
    ids: set[str] = set()
    for a in load_official_strong_anchors():
        rp = clean(a.get("reference_patch_id"))
        an = clean(a.get("anchor_id"))
        if rp:
            ids.add(rp)
        if an:
            ids.add(an)
    return ids


def load_batch() -> list[dict[str, str]]:
    """Lê o batch selecionado (saída da Tarefa 2)."""
    return read_csv_rows(PUBLIC_DIR / "mv2_data_04_corpus_metadata_batch.csv")


def filled_temporal_windows() -> dict[str, tuple[str, str]]:
    """Janelas temporais preenchidas no template (de execução futura), por target_id.

    Só conta janelas com início e fim presentes e não-futuros (nunca inventadas). No
    primeiro run o template ainda não existe -> dicionário vazio -> probes bloqueiam.
    """
    out: dict[str, tuple[str, str]] = {}
    for r in read_csv_rows(PUBLIC_DIR / "mv2_data_04_temporal_window_template.csv"):
        tid = clean(r.get("target_id"))
        start = clean(r.get("temporal_window_start"))
        end = clean(r.get("temporal_window_end"))
        if not (tid and start and end):
            continue
        # nunca aceitar janela futura
        if classify_datetime(start[:10])[1] == "INVALID_FUTURE" or classify_datetime(end[:10])[1] == "INVALID_FUTURE":
            continue
        out[tid] = (start, end)
    return out
