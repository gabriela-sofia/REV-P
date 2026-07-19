"""MV2-DATA-03 Tarefa 3 — Live GEE metadata probe (metadata-only).

Executa consulta GEE metadata-only para os canary candidates oficiais SÓ se allow_network +
allow_metadata_calls + gee_enabled + pacote ee + credencial. Nunca exporta imagem, nunca
baixa raster. Caso contrário, NOT_EXECUTED.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_live_gee_metadata_probe.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_03_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    effective_flags,
    ensure_public_dir,
    env_present,
    lib_available,
    metadata_blocked,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "probe_id",
    "canary_candidate_id",
    "executed",
    "auth_status",
    "collection",
    "query_mode",
    "product_id",
    "scene_id",
    "matched_images",
    "selected_product_id",
    "selected_datetime",
    "selected_mgrs_tile",
    "selected_cloudy_pixel_percentage",
    "selected_cloud_coverage_assessment",
    "metadata_match_status",
    "blocking_reason",
    "notes",
]


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    candidates = read_csv_rows(PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv")

    cred_ok = env_present("GOOGLE_APPLICATION_CREDENTIALS") and env_present("GEE_PROJECT")
    ee_ok = lib_available("ee")
    blocked = metadata_blocked(flags) or not flags.get("gee_enabled")
    auth_status = "AUTH_PRESENT" if cred_ok else "AUTH_MISSING"
    if cred_ok and not ee_ok:
        auth_status = "EE_PACKAGE_MISSING"

    rows: list[dict[str, object]] = []
    for i, c in enumerate(candidates, start=1):
        base: dict[str, object] = {
            "probe_id": f"MV2_DATA_03_GEEPROBE_{i:02d}",
            "canary_candidate_id": clean(c.get("canary_candidate_id")),
            "auth_status": auth_status,
            "collection": clean(c.get("gee_collection")),
            "query_mode": "metadata_only_no_export",
            "product_id": clean(c.get("product_id")),
            "scene_id": clean(c.get("scene_id")),
            "matched_images": "",
            "selected_product_id": "",
            "selected_datetime": "",
            "selected_mgrs_tile": "",
            "selected_cloudy_pixel_percentage": "",
            "selected_cloud_coverage_assessment": "",
            "executed": "false",
        }
        if blocked:
            base.update({
                "metadata_match_status": "NOT_EXECUTED",
                "blocking_reason": "config_local_ausente_ou_flags_gee_metadata_desabilitadas",
                "notes": "consulta filterBounds+filterDate pronta; metadata-only",
            })
        elif auth_status != "AUTH_PRESENT":
            base.update({
                "metadata_match_status": "NOT_EXECUTED",
                "blocking_reason": auth_status.lower(),
                "notes": "credencial GEE so por env var (GEE_PROJECT + GOOGLE_APPLICATION_CREDENTIALS)",
            })
        else:
            base.update({
                "metadata_match_status": "EXECUTED_NO_SELECTION_PARSED",
                "blocking_reason": "",
                "notes": "metadata-only; parsing/selecao de imagem fica para revisao humana",
            })
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_live_gee_metadata_probe.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    print(
        f"[mv2_data_03_gee_probe] {len(rows)} probes | executados={executed} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
