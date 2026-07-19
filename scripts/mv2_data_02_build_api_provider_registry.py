"""MV2-DATA-02 Tarefa 1 — Registry de providers de API Sentinel.

Documenta GEE, CDSE STAC e CDSE OData: propósito, suporte a metadata/raster, autenticação,
env vars de credencial, se estão habilitados pela config e se a rede é permitida. Nenhum
segredo é gravado — apenas o NOME da env var de credencial.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_api_provider_registry.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_api_config import load_config
from mv2_data_02_common import PROJECT_ROOT, PUBLIC_DIR, ensure_public_dir, write_csv

COLUMNS = [
    "provider_id",
    "provider_name",
    "purpose",
    "supports_metadata",
    "supports_raster_download",
    "requires_auth",
    "auth_method",
    "credential_env_vars",
    "enabled_by_config",
    "network_allowed",
    "default_mode",
    "public_output_allowed",
    "private_output_required",
    "notes",
]


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    network = "true" if cfg.get("allow_network") else "false"
    cdse_env = str(cfg.get("cdse_token_env_var"))
    gee_env = str(cfg.get("gee_service_account_env_var"))

    specs: list[dict[str, object]] = [
        {
            "provider_id": "GEE",
            "provider_name": "Google Earth Engine",
            "purpose": "consulta de metadados COPERNICUS/S2_SR_HARMONIZED (metadata-only)",
            "supports_metadata": "true",
            "supports_raster_download": "false",
            "requires_auth": "true",
            "auth_method": "service_account_or_user_oauth",
            "credential_env_vars": gee_env,
            "enabled_by_config": "true" if cfg.get("gee_enabled") else "false",
            "default_mode": "metadata_only",
        },
        {
            "provider_id": "CDSE_STAC",
            "provider_name": "Copernicus Data Space Ecosystem - STAC",
            "purpose": "busca de itens Sentinel-2 por bbox+datetime (metadata-only)",
            "supports_metadata": "true",
            "supports_raster_download": "false",
            "requires_auth": "false",
            "auth_method": "none_for_search_token_for_download",
            "credential_env_vars": cdse_env,
            "enabled_by_config": "true" if cfg.get("cdse_stac_enabled") else "false",
            "default_mode": "metadata_only",
        },
        {
            "provider_id": "CDSE_ODATA",
            "provider_name": "Copernicus Data Space Ecosystem - OData",
            "purpose": "resolução de produto e endpoint de download (canary/local-only)",
            "supports_metadata": "true",
            "supports_raster_download": "true",
            "requires_auth": "true",
            "auth_method": "bearer_token_env_var",
            "credential_env_vars": cdse_env,
            "enabled_by_config": "true" if cfg.get("cdse_odata_enabled") else "false",
            "default_mode": "metadata_only_download_opt_in",
        },
    ]

    rows: list[dict[str, object]] = []
    for s in specs:
        s.update(
            {
                "network_allowed": network,
                "public_output_allowed": "metadata_only_no_raster",
                "private_output_required": "true" if s["supports_raster_download"] == "true" else "false",
                "notes": "credencial so por env var; nenhum segredo em arquivo publico",
            }
        )
        rows.append(s)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_api_provider_registry.csv"
    write_csv(out, COLUMNS, rows)
    print(
        f"[mv2_data_02_providers] {len(rows)} providers | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
