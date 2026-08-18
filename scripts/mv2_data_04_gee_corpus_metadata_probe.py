"""MV2-DATA-04 Tarefa 3 — GEE corpus metadata probe (metadata-only).

Prepara consulta COPERNICUS/S2_SR_HARMONIZED para os targets do batch. Metadata-only: nunca
exporta imagem, nunca baixa raster. BLOQUEIA se não houver janela temporal (nunca faz query
aberta) e se a config não permitir.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_gee_corpus_metadata_probe.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    DEFAULT_S2_COLLECTION,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    effective_flags,
    ensure_public_dir,
    env_present,
    filled_temporal_windows,
    lib_available,
    load_batch,
    metadata_blocked,
    write_csv,
)

COLUMNS = [
    "probe_id",
    "target_id",
    "executed",
    "auth_status",
    "collection",
    "bbox",
    "crs",
    "datetime_window",
    "temporal_window_status",
    "matched_images",
    "selected_product_id",
    "selected_datetime",
    "selected_mgrs_tile",
    "selected_cloudy_pixel_percentage",
    "selected_cloud_coverage_assessment",
    "metadata_status",
    "blocking_reason",
    "notes",
]


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    batch = load_batch()
    windows = filled_temporal_windows()

    cred_ok = env_present("GOOGLE_APPLICATION_CREDENTIALS") and env_present("GEE_PROJECT")
    ee_ok = lib_available("ee")
    config_block = metadata_blocked(flags) or not flags.get("gee_enabled")
    auth_status = "AUTH_PRESENT" if cred_ok else "AUTH_MISSING"
    if cred_ok and not ee_ok:
        auth_status = "EE_PACKAGE_MISSING"

    rows: list[dict[str, object]] = []
    for i, t in enumerate(batch, start=1):
        tid = clean(t.get("target_id"))
        window = windows.get(tid)
        has_window = window is not None
        datetime_window = f"{window[0]}/{window[1]}" if window else ""
        base: dict[str, object] = {
            "probe_id": f"MV2_DATA_04_GEEC_{i:02d}",
            "target_id": tid,
            "collection": DEFAULT_S2_COLLECTION,
            "bbox": clean(t.get("bbox")),
            "crs": clean(t.get("crs")),
            "datetime_window": datetime_window,
            "temporal_window_status": "PRESENT" if has_window else "MISSING",
            "auth_status": auth_status,
            "matched_images": "",
            "selected_product_id": "",
            "selected_datetime": "",
            "selected_mgrs_tile": "",
            "selected_cloudy_pixel_percentage": "",
            "selected_cloud_coverage_assessment": "",
            "executed": "false",
        }
        # Ordem dos bloqueios: janela temporal é o gargalo crítico do corpus.
        if not has_window:
            base.update({
                "metadata_status": "BLOCKED_NO_TEMPORAL_WINDOW",
                "blocking_reason": "sem_janela_temporal_nunca_fazer_query_aberta",
                "notes": "filterBounds pronto; falta filterDate (janela temporal por target)",
            })
        elif config_block:
            base.update({
                "metadata_status": "NOT_EXECUTED_CONFIG_BLOCKED",
                "blocking_reason": "config_ausente_ou_flags_gee_metadata_desabilitadas",
                "notes": "janela presente; consulta metadata-only pronta",
            })
        elif auth_status != "AUTH_PRESENT":
            base.update({
                "metadata_status": "NOT_EXECUTED_AUTH_MISSING",
                "blocking_reason": auth_status.lower(),
                "notes": "credencial GEE so por env var",
            })
        else:
            base.update({
                "metadata_status": "EXECUTED_NO_SELECTION_PARSED",
                "blocking_reason": "",
                "notes": "metadata-only com filterDate; selecao fica para revisao humana",
            })
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_gee_corpus_metadata_probe.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    blocked = sum(1 for r in rows if r["metadata_status"] == "BLOCKED_NO_TEMPORAL_WINDOW")
    print(
        f"[mv2_data_04_gee_probe] {len(rows)} probes | executados={executed} "
        f"blocked_no_temporal_window={blocked} | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
