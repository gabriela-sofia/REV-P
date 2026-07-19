"""Núcleo compartilhado do marco MV2-DATA-03 — Live Metadata Probe & Private Raster Canary.

Sai do harness fail-closed do MV2-DATA-02 e testa, de forma controlada, a camada REAL de
API/metadados Sentinel e, opcionalmente, no máximo 1 raster canary privado/local-only — usando
exclusivamente as 2 âncoras OFICIAIS (Protocolo-C, namespace REFPATCH_*), nunca os 128 patch
do corpus. Mesmo com canary bem-sucedido, isso é prova técnica do pipeline, não baseline do
corpus, e NÃO desbloqueia o Dia 10.

Fail-closed:
- sem config local privada -> CONFIG_MISSING, nenhuma chamada de rede;
- credencial só por variável de ambiente; valores nunca são impressos/gravados;
- allow_network/metadata/raster_download exigem config local presente E flag explícita;
- canary exige REV_P_ALLOW_RASTER_CANARY=YES;
- raster só em diretório privado git-ignored; nunca em outputs_public;
- canary oficial nunca vira patch do corpus; T23KPR nunca auto-vinculado a PET_* do corpus.

Reusa IO/classificadores do MV2-13 e utilitários do MV2-DATA-02 (uma só fonte de verdade).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mv2_data_02_common import (  # noqa: F401  (reuso/reexport intencional)
    CANARY_BANDS,
    CANARY_SCL,
    DEFAULT_S2_COLLECTION,
    GEE_METADATA_FIELDS,
    PROJECT_ROOT,
    S2_SCENE_RE,
    classify_datetime,
    clean,
    crs_status,
    is_private_path,
    load_official_strong_anchors,
    read_csv_rows,
    write_csv,
    write_json,
)

PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_live_api_canary"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"
DEFAULT_PRIVATE_DIR = "local_only/mv2_data_live_api_canary"

DATA01_TARGETS = PROJECT_ROOT / "outputs_public" / "mv2_data_gee_lineage_targets" / "mv2_data_01_lineage_targets.csv"
OFFICIAL_ANCHOR_REGISTRY = PROJECT_ROOT / "datasets" / "official_anchor_sentinel_patch_registry.csv"

BRANCH_EXECUTADA = "analysis/temporal-asset-readiness-mv1"

MAX_CANARY_CANDIDATES = 2
MAX_CANARY_EXECUTIONS = 1

# Caminhos aceitos para a config local PRIVADA (procurados, nunca criados com segredo).
LOCAL_CONFIG_RELPATHS = [
    "local_only/mv2_data_api_raster_harness/api_config.local.json",
    "data_local/mv2_data_api_raster_harness/api_config.local.json",
    "private_outputs/mv2_data_api_raster_harness/api_config.local.json",
]

# Variáveis de ambiente permitidas (presença detectada; valores NUNCA impressos).
CREDENTIAL_ENV_VARS = [
    "CDSE_TOKEN",
    "CDSE_USERNAME",
    "CDSE_PASSWORD",
    "GEE_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
]
ENV_ALLOW_NETWORK = "REV_P_ALLOW_NETWORK"
ENV_ALLOW_METADATA_CALLS = "REV_P_ALLOW_METADATA_CALLS"
ENV_ALLOW_RASTER_DOWNLOAD = "REV_P_ALLOW_RASTER_DOWNLOAD"
ENV_ALLOW_RASTER_CANARY = "REV_P_ALLOW_RASTER_CANARY"
CANARY_CONFIRM_VALUE = "YES"

OPTIONAL_LIBS = ["requests", "rasterio", "pystac_client", "ee"]

CONSENSUS_STATUSES = {
    "API_CONFIRMED_STRONG",
    "REGISTRY_ONLY_STRONG",
    "PARTIAL_CONFIRMATION",
    "CONFLICT",
    "NOT_EXECUTED",
    "INVALID",
}

RISK_TYPES = [
    "RISK_SECRET_LEAK",
    "RISK_PRIVATE_PATH_LEAK",
    "RISK_RASTER_PUBLIC_LEAK",
    "RISK_CANARY_AS_CORPUS_UNLOCK",
    "RISK_BULK_DOWNLOAD_ACCIDENT",
    "RISK_STAC_ASSET_DOWNLOAD_TOO_EARLY",
    "RISK_GEE_EXPORT_TOO_EARLY",
    "RISK_TILE_AUTO_JOIN",
    "RISK_CREDENTIAL_LOGGING",
    "RISK_COST_QUOTA_RATE_LIMIT",
    "RISK_DAY10_FALSE_UNLOCK",
    "RISK_SANDBOX_FALSE_UNLOCK",
]


def ensure_public_dir() -> Path:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    return PUBLIC_DIR


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def env_present(name: str) -> bool:
    """True se a env var existe e é não-vazia. NUNCA retorna/loga o valor."""
    return bool(os.environ.get(name, "").strip())


def find_local_config() -> tuple[Path | None, dict[str, object]]:
    """Localiza e carrega a config local privada (sem segredo). (path|None, dict)."""
    for rel in LOCAL_CONFIG_RELPATHS:
        p = PROJECT_ROOT / rel
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            return p, (raw if isinstance(raw, dict) else {})
    return None, {}


def lib_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def effective_flags() -> dict[str, object]:
    """Resolve as flags efetivas combinando config local + env vars, fail-closed.

    Sem config local presente, NADA de rede é permitido (CONFIG_MISSING), mesmo que env
    vars estejam setadas.
    """
    path, cfg = find_local_config()
    found = path is not None

    private_dir = clean(cfg.get("private_output_dir")) or DEFAULT_PRIVATE_DIR
    try:
        max_mb = int(str(cfg.get("max_download_mb", 50)))
    except (TypeError, ValueError):
        max_mb = 50

    if not found:
        allow_network = allow_metadata = allow_raster_download = allow_canary = False
    else:
        allow_network = (cfg.get("allow_network") is True) or _truthy_env(ENV_ALLOW_NETWORK)
        allow_metadata = (cfg.get("allow_metadata_calls") is True) or _truthy_env(ENV_ALLOW_METADATA_CALLS)
        allow_raster_download = (cfg.get("allow_raster_download") is True) or _truthy_env(ENV_ALLOW_RASTER_DOWNLOAD)
        allow_canary = os.environ.get(ENV_ALLOW_RASTER_CANARY, "").strip() == CANARY_CONFIRM_VALUE

    return {
        "config_found": found,
        "config_path": path.relative_to(PROJECT_ROOT).as_posix() if path else "",
        "gee_enabled": cfg.get("gee_enabled") is True,
        "cdse_stac_enabled": cfg.get("cdse_stac_enabled") is True,
        "cdse_odata_enabled": cfg.get("cdse_odata_enabled") is True,
        "allow_network": allow_network,
        "allow_metadata_calls": allow_metadata,
        "allow_raster_download": allow_raster_download,
        "allow_canary": allow_canary,
        "max_download_mb": max_mb,
        "private_output_dir": private_dir,
    }


def metadata_blocked(flags: dict[str, object]) -> bool:
    return not (flags.get("allow_network") and flags.get("allow_metadata_calls"))


def corpus_patch_ids() -> set[str]:
    """patch_ids dos 128 targets do corpus (DATA-01) — canary nunca pode coincidir."""
    return {clean(r.get("patch_id")) for r in read_csv_rows(DATA01_TARGETS) if clean(r.get("patch_id"))}
