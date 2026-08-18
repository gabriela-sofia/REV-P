"""MV2-DATA-02 Tarefa 8 — Validador de intake de raster privado.

Valida o manifesto de execução canary (se existir): que o raster está FORA de
outputs_public, que tem as bandas mínimas, SCL, checksum, e CRS/geotransform se
rasterio/pyproj estiverem disponíveis. Se nada foi executado, retorna OK com
NO_RASTER_EXECUTED para cada candidato.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_private_raster_validation.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_common import (
    CANARY_BANDS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    is_private_path,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "validation_id",
    "canary_id",
    "raster_executed",
    "private_path_registered",
    "public_leak_detected",
    "file_exists_private",
    "size_bytes",
    "sha256",
    "bands_ok",
    "scl_ok",
    "crs_ok",
    "geotransform_ok",
    "validation_status",
    "notes",
]


def _public_raster_leak() -> bool:
    raster_ext = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog", ".nc", ".hdf"}
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in raster_ext:
            return True
    return False


def _raster_checks(private_path: str) -> dict[str, object]:
    """Checa CRS/geotransform se libs existirem; senão marca UNKNOWN (não falha)."""
    out: dict[str, object] = {"crs_ok": "UNKNOWN", "geotransform_ok": "UNKNOWN"}
    abs_path = PROJECT_ROOT / private_path
    if not abs_path.exists():
        return out
    try:
        import rasterio  # noqa: F401
    except ImportError:
        out["notes_lib"] = "rasterio_indisponivel"
        return out
    try:
        import rasterio

        with rasterio.open(abs_path) as ds:
            out["crs_ok"] = "true" if ds.crs else "false"
            out["geotransform_ok"] = "true" if ds.transform else "false"
    except Exception as exc:
        out["crs_ok"] = "false"
        out["geotransform_ok"] = "false"
        out["notes_lib"] = f"erro_rasterio:{type(exc).__name__}"
    return out


def build_rows() -> list[dict[str, object]]:
    manifest_path = PUBLIC_DIR / "mv2_data_02_raster_canary_execution_manifest.csv"
    manifest = read_csv_rows(manifest_path)
    leak = _public_raster_leak()

    rows: list[dict[str, object]] = []
    for i, m in enumerate(manifest, start=1):
        cid = clean(m.get("canary_id"))
        executed = clean(m.get("executed")) == "true"
        base: dict[str, object] = {
            "validation_id": f"MV2_DATA_02_PRV_{i:03d}",
            "canary_id": cid,
            "raster_executed": "true" if executed else "false",
            "public_leak_detected": "true" if leak else "false",
            "private_path_registered": "",
            "file_exists_private": "false",
            "size_bytes": clean(m.get("size_bytes")),
            "sha256": clean(m.get("sha256")),
            "bands_ok": "NA",
            "scl_ok": "NA",
            "crs_ok": "NA",
            "geotransform_ok": "NA",
        }
        if not executed:
            base.update(
                {
                    "validation_status": "NO_RASTER_EXECUTED",
                    "notes": "nenhum raster executado; fail-closed preservado",
                }
            )
            rows.append(base)
            continue

        priv = clean(m.get("private_raster_path"))
        registered_private = is_private_path(priv)
        exists = (PROJECT_ROOT / priv).exists() if priv else False
        bands = clean(m.get("bands_present"))
        bands_ok = all(b in bands for b in CANARY_BANDS)
        scl_ok = clean(m.get("scl_present")) == "true"
        checks = _raster_checks(priv) if exists else {}

        status = "PRIVATE_RASTER_VALID"
        if leak or not registered_private:
            status = "INVALID_PUBLIC_OR_NONPRIVATE_PATH"
        elif not exists:
            status = "INVALID_FILE_MISSING"
        elif not bands_ok or not scl_ok:
            status = "INVALID_BANDS_OR_SCL"

        base.update(
            {
                "private_path_registered": "true" if registered_private else "false",
                "file_exists_private": "true" if exists else "false",
                "bands_ok": "true" if bands_ok else "false",
                "scl_ok": "true" if scl_ok else "false",
                "crs_ok": checks.get("crs_ok", "UNKNOWN"),
                "geotransform_ok": checks.get("geotransform_ok", "UNKNOWN"),
                "validation_status": status,
                "notes": clean(str(checks.get("notes_lib", ""))) or "raster privado validado",
            }
        )
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_private_raster_validation.csv"
    write_csv(out, COLUMNS, rows)
    leaks = sum(1 for r in rows if r["public_leak_detected"] == "true")
    print(
        f"[mv2_data_02_private_val] {len(rows)} entradas | public_leaks={leaks} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
