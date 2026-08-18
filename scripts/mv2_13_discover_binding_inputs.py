"""MV2-13 Tarefa 1 — Descoberta e normalização de entradas de binding.

Localiza as entradas relevantes (MV2-10/11/12 Data Readiness/12 Spectral Reconstruction
e inventários de lineage/patch), resolvendo cada uma local ou em outro worktree (sem
checkout). Registra ausências sem falhar e gera índice leve.

Saída: outputs_public/mv2_sentinel_binding/mv2_13_input_discovery.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    INPUT_RELPATHS,
    INPUT_STAGE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    ensure_output_dir,
    resolve_input,
    write_csv,
)

COLUMNS = [
    "input_id", "source_stage", "path", "exists", "file_type", "size_bytes",
    "role", "read_status", "notes",
]

ROLES = {
    "anchors": "fonte_primaria_anchors_281",
    "spectral_summary": "estado_mv2_12_spectral",
    "stac_resolution_plan": "plano_stac_mv2_12",
    "lineage_manifest": "inventario_asset",
    "raster_metadata_index": "inventario_raster_metadata",
    "patch_corpus_registry": "inventario_patch_corpus",
    "data_readiness_matrix": "matriz_dados_faltantes_mv2_12",
    "raster_native_backlog": "backlog_raster_nativo_mv2_12",
    "regional_coverage": "cobertura_regional_mv2_11",
    "gap_matrix": "gap_matrix_mv2_10",
}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, key in enumerate(INPUT_RELPATHS, start=1):
        path, origin = resolve_input(key)
        rel = INPUT_RELPATHS[key]
        exists = path is not None
        size = path.stat().st_size if (path and path.exists()) else 0
        ext = Path(rel).suffix.lower().lstrip(".") or "unknown"
        rows.append(
            {
                "input_id": f"MV2_13_IN_{i:02d}",
                "source_stage": INPUT_STAGE[key],
                # Apenas caminho relativo — nunca caminho absoluto privado.
                "path": rel,
                "exists": str(exists).lower(),
                "file_type": ext,
                "size_bytes": size,
                "role": ROLES.get(key, ""),
                "read_status": origin,
                "notes": "ausente; registrado sem falhar" if not exists else "",
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_13_input_discovery.csv"
    write_csv(out, COLUMNS, rows)
    found = sum(1 for r in rows if r["exists"] == "true")
    other = sum(1 for r in rows if str(r["read_status"]).startswith("READ_ONLY"))
    print(f"[mv2_13_discovery] entradas: {len(rows)} | encontradas: {found} | de outra branch: {other}")
    print(f"[mv2_13_discovery] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
