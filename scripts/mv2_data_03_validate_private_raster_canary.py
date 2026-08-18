"""MV2-DATA-03 Tarefa 8 — Validar raster canary privado.

Se não houve download, retorna NO_RASTER_EXECUTED_OK. Se houve, valida que o arquivo está
em local privado (fora de outputs_public), tamanho, SHA256, bandas, SCL e CRS/geotransform
(se rasterio existir), registrando limitações quando bibliotecas faltarem.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_private_raster_canary_validation.csv
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from mv2_data_03_common import (
    CANARY_BANDS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    is_private_path,
    lib_available,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "validation_id",
    "execution_id",
    "raster_executed",
    "public_leak_detected",
    "private_file_exists",
    "size_ok",
    "sha256_ok",
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


def _raster_checks(abs_path: Path) -> dict[str, object]:
    out: dict[str, object] = {"crs_ok": "UNKNOWN", "geotransform_ok": "UNKNOWN", "lib_note": ""}
    if not lib_available("rasterio"):
        out["lib_note"] = "rasterio_indisponivel_crs_geotransform_nao_verificados"
        return out
    try:
        import rasterio

        with rasterio.open(abs_path) as ds:
            out["crs_ok"] = "true" if ds.crs else "false"
            out["geotransform_ok"] = "true" if ds.transform else "false"
    except Exception as exc:
        out["crs_ok"] = "false"
        out["geotransform_ok"] = "false"
        out["lib_note"] = f"erro_rasterio:{type(exc).__name__}"
    return out


def build_rows() -> list[dict[str, object]]:
    manifest = read_csv_rows(PUBLIC_DIR / "mv2_data_03_private_raster_canary_execution_manifest.csv")
    leak = _public_raster_leak()

    rows: list[dict[str, object]] = []
    for i, m in enumerate(manifest, start=1):
        executed = clean(m.get("executed")) == "true"
        base: dict[str, object] = {
            "validation_id": f"MV2_DATA_03_VAL_{i:02d}",
            "execution_id": clean(m.get("execution_id")),
            "raster_executed": "true" if executed else "false",
            "public_leak_detected": "true" if leak else "false",
            "private_file_exists": "false",
            "size_ok": "NA",
            "sha256_ok": "NA",
            "bands_ok": "NA",
            "scl_ok": "NA",
            "crs_ok": "NA",
            "geotransform_ok": "NA",
        }
        if not executed:
            base.update({
                "validation_status": "NO_RASTER_EXECUTED_OK",
                "notes": "nenhum raster executado; fail-closed preservado",
            })
            rows.append(base)
            continue

        priv = clean(m.get("private_output_ref"))
        registered_private = is_private_path(priv)
        abs_path = (PROJECT_ROOT / priv) if priv else None
        exists = bool(abs_path is not None and abs_path.exists())
        size_ok = False
        sha_ok = False
        checks: dict[str, object] = {}
        if exists and abs_path is not None:
            size_ok = abs_path.stat().st_size > 0
            h = hashlib.sha256()
            with abs_path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            sha_ok = h.hexdigest() == clean(m.get("sha256"))
            checks = _raster_checks(abs_path)
        bands = clean(m.get("bands_requested"))
        bands_ok = all(b in bands for b in CANARY_BANDS)
        scl_ok = clean(m.get("scl_requested")) == "SCL"

        status = "PRIVATE_RASTER_VALID"
        if leak or not registered_private:
            status = "INVALID_PUBLIC_OR_NONPRIVATE_PATH"
        elif not exists:
            status = "INVALID_FILE_MISSING"
        elif not (size_ok and sha_ok and bands_ok and scl_ok):
            status = "INVALID_SIZE_SHA_BANDS_OR_SCL"

        base.update({
            "private_file_exists": "true" if exists else "false",
            "size_ok": "true" if size_ok else "false",
            "sha256_ok": "true" if sha_ok else "false",
            "bands_ok": "true" if bands_ok else "false",
            "scl_ok": "true" if scl_ok else "false",
            "crs_ok": checks.get("crs_ok", "UNKNOWN"),
            "geotransform_ok": checks.get("geotransform_ok", "UNKNOWN"),
            "validation_status": status,
            "notes": clean(str(checks.get("lib_note", ""))) or "raster privado validado",
        })
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_private_raster_canary_validation.csv"
    write_csv(out, COLUMNS, rows)
    leaks = sum(1 for r in rows if r["public_leak_detected"] == "true")
    print(
        f"[mv2_data_03_private_val] {len(rows)} entrada(s) | public_leaks={leaks} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
