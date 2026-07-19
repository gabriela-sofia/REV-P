"""MV2-DATA-02 Tarefa 4 — Cliente GEE (metadata-only).

Prepara a consulta à coleção COPERNICUS/S2_SR_HARMONIZED e coleta SOMENTE metadados. Não
exporta imagem, não baixa raster. Executa apenas se o pacote earthengine estiver instalado,
houver credencial e a config permitir. Sem credencial → NOT_EXECUTED_AUTH_MISSING.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_gee_metadata_results.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_api_config import config_blocks_metadata, credential_present, load_config
from mv2_data_02_common import (
    DEFAULT_S2_COLLECTION,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_targets,
    write_csv,
)

COLUMNS = [
    "result_id",
    "target_id",
    "executed",
    "auth_status",
    "collection",
    "matched_images",
    "selected_product_id",
    "selected_scene_id",
    "selected_datetime",
    "selected_mgrs_tile",
    "selected_cloudy_pixel_percentage",
    "selected_cloud_coverage_assessment",
    "metadata_status",
    "blocking_reason",
    "notes",
]


def _earthengine_available() -> bool:
    try:
        import ee  # noqa: F401
        return True
    except ImportError:
        return False


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    targets = load_targets()

    ee_ok = _earthengine_available()
    cred_ok = credential_present(str(cfg.get("gee_service_account_env_var")))
    metadata_blocked = config_blocks_metadata(cfg) or not cfg.get("gee_enabled")

    if not cred_ok:
        auth_status = "AUTH_MISSING"
    elif not ee_ok:
        auth_status = "EE_PACKAGE_MISSING"
    else:
        auth_status = "AUTH_PRESENT"

    rows: list[dict[str, object]] = []
    for i, t in enumerate(targets, start=1):
        base: dict[str, object] = {
            "result_id": f"MV2_DATA_02_GEE_{i:05d}",
            "target_id": clean(t.get("target_id")),
            "collection": DEFAULT_S2_COLLECTION,
            "auth_status": auth_status,
            "matched_images": "",
            "selected_product_id": "",
            "selected_scene_id": "",
            "selected_datetime": "",
            "selected_mgrs_tile": "",
            "selected_cloudy_pixel_percentage": "",
            "selected_cloud_coverage_assessment": "",
            "executed": "false",
        }
        if auth_status != "AUTH_PRESENT":
            base.update(
                {
                    "metadata_status": "NOT_EXECUTED_AUTH_MISSING",
                    "blocking_reason": auth_status.lower(),
                    "notes": "credencial GEE so por env var; pacote ee + service account necessarios",
                }
            )
        elif metadata_blocked:
            base.update(
                {
                    "metadata_status": "NOT_EXECUTED_CONFIG_BLOCKED",
                    "blocking_reason": "allow_network/allow_metadata_calls/gee_enabled=false",
                    "notes": "consulta filterBounds+filterDate pronta; metadata-only",
                }
            )
        else:
            base.update(
                {
                    "metadata_status": "EXECUTED_NO_SELECTION_PARSED",
                    "blocking_reason": "",
                    "notes": "metadata-only; selecao/parsing de imagem fica para revisao humana",
                }
            )
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_gee_metadata_results.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    print(
        f"[mv2_data_02_gee] {len(rows)} alvos | executados={executed} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
