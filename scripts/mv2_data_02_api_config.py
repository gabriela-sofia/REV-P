"""MV2-DATA-02 — Leitor de config do harness de API/raster (fail-closed).

A config controla se a rede pode ser usada, se metadados podem ser consultados e se um
raster canary pode ser baixado. Os defaults são SEMPRE fail-closed:
- allow_network=false;
- allow_metadata_calls=false;
- allow_raster_download=false;
- allow_canary_download=false.

Nenhum token/segredo é lido de arquivo público: credenciais só por variável de ambiente
(cdse_token_env_var / gee_service_account_env_var). O arquivo de config (se existir) carrega
apenas FLAGS e nomes de env vars — nunca os segredos em si.

Resolução do arquivo de config (opcional, privado):
1. variável de ambiente REV_P_MV2_DATA_02_CONFIG (caminho para um JSON);
2. <private_output_dir>/mv2_data_02_api_config.json (default em local_only/...);
3. caso nenhum exista, usa os defaults fail-closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mv2_data_02_common import PROJECT_ROOT, clean

CONFIG_ENV_VAR = "REV_P_MV2_DATA_02_CONFIG"
CANARY_CONFIRM_ENV_VAR = "REV_P_ALLOW_RASTER_CANARY"
CANARY_CONFIRM_VALUE = "YES"

# Chaves que NUNCA podem aparecer no arquivo de config (seriam segredo em claro).
FORBIDDEN_SECRET_KEYS = {
    "cdse_token", "cdse_access_token", "token", "password", "secret",
    "gee_service_account_json", "service_account_key", "private_key",
}

DEFAULT_CONFIG: dict[str, object] = {
    "gee_enabled": False,
    "cdse_stac_enabled": False,
    "cdse_odata_enabled": False,
    "allow_network": False,
    "allow_metadata_calls": False,
    "allow_raster_download": False,
    "allow_canary_download": False,
    "max_download_mb": 50,
    "private_output_dir": "local_only/mv2_data_api_raster_harness",
    "cdse_token_env_var": "CDSE_ACCESS_TOKEN",
    "gee_project": "",
    "gee_service_account_env_var": "GEE_SERVICE_ACCOUNT_JSON",
    "notes": "fail-closed por padrao; edite uma copia privada em local_only/ para habilitar",
}

_BOOL_KEYS = {
    "gee_enabled", "cdse_stac_enabled", "cdse_odata_enabled", "allow_network",
    "allow_metadata_calls", "allow_raster_download", "allow_canary_download",
}


def _resolve_config_path() -> Path | None:
    env_path = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    default_priv = PROJECT_ROOT / str(DEFAULT_CONFIG["private_output_dir"]) / "mv2_data_02_api_config.json"
    if default_priv.exists():
        return default_priv
    return None


def load_config() -> dict[str, object]:
    """Carrega a config fail-closed, opcionalmente sobreposta por um JSON privado."""
    cfg: dict[str, object] = dict(DEFAULT_CONFIG)
    path = _resolve_config_path()
    if path is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raw = {}
        if isinstance(raw, dict):
            for k, val in raw.items():
                low = str(k).lower()
                # Nunca aceitar segredo em claro no arquivo de config.
                if low in FORBIDDEN_SECRET_KEYS:
                    continue
                if k in cfg:
                    cfg[k] = val
    # Coerção fail-closed: qualquer valor não-True vira False nas flags booleanas.
    for k in _BOOL_KEYS:
        cfg[k] = cfg.get(k) is True
    try:
        cfg["max_download_mb"] = int(str(cfg.get("max_download_mb", 50)))
    except (TypeError, ValueError):
        cfg["max_download_mb"] = 50
    return cfg


def credential_present(env_var_name: str) -> bool:
    return bool(clean(os.environ.get(clean(env_var_name), "")))


def canary_env_confirmed() -> bool:
    return os.environ.get(CANARY_CONFIRM_ENV_VAR, "").strip() == CANARY_CONFIRM_VALUE


def config_blocks_metadata(cfg: dict[str, object]) -> bool:
    return not (cfg.get("allow_network") and cfg.get("allow_metadata_calls"))


def write_example_config() -> Path:
    """Escreve o arquivo de config de EXEMPLO público (sem segredo)."""
    from mv2_data_02_common import ensure_public_dir, write_json

    out = ensure_public_dir() / "mv2_data_02_api_config.example.json"
    write_json(out, DEFAULT_CONFIG)
    return out


def main() -> None:
    out = write_example_config()
    cfg = load_config()
    print(f"[mv2_data_02_config] exemplo: {out.relative_to(PROJECT_ROOT).as_posix()}")
    print(
        f"[mv2_data_02_config] default fail-closed: allow_network={cfg['allow_network']} "
        f"allow_metadata_calls={cfg['allow_metadata_calls']} "
        f"allow_raster_download={cfg['allow_raster_download']} "
        f"allow_canary_download={cfg['allow_canary_download']}"
    )


if __name__ == "__main__":
    main()
