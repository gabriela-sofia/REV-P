"""MV2-12 — Backlog de raster Sentinel espectral nativo (diagnóstico do Dia 10).

Separa explicitamente renderização visual (PNG/JPG/NPZ) de raster Sentinel espectral
nativo (GeoTIFF/JP2/SAFE com bandas B02/B03/B04/B08/B11/B12 e máscara SCL). Varre a
working tree por raster nativo legível e produz um backlog por região.

NUNCA infere raster espectral a partir de PNG/NPZ. Se não há raster nativo, o Dia 10
permanece bloqueado e nenhum baseline espectral é inventado.

Saída: outputs_public/mv2_data_readiness/mv2_12_sentinel_native_raster_backlog.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_12_data_readiness_common import (
    NATIVE_SPECTRAL_EXTENSIONS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    REGIONAL_COVERAGE,
    ensure_output_dir,
    infer_region_from_text,
    write_csv,
)

COLUMNS = [
    "patch_id",
    "asset_id",
    "region",
    "has_png_render",
    "has_npz_visual",
    "has_dinov2_embedding",
    "has_native_raster",
    "native_raster_path",
    "available_bands",
    "has_cloud_mask",
    "has_scene_id",
    "can_support_spectral_baseline",
    "blocking_reason",
    "next_action",
]

EXCLUDE_DIR_PARTS = {".git", ".claude", "node_modules", "__pycache__", ".venv", "venv"}


def _count_native_rasters_by_region() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"Curitiba": [], "Recife": [], "Petropolis": []}
    roots = [
        PROJECT_ROOT / "data_raw_local",
        PROJECT_ROOT / "data_quarantine",
        PROJECT_ROOT / "external_data_local",
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT / "local_runs",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if any(part in EXCLUDE_DIR_PARTS for part in path.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in NATIVE_SPECTRAL_EXTENSIONS:
                continue
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            region = infer_region_from_text(rel)
            if region in found:
                found[region].append(rel)
    return found


def build_rows() -> list[dict[str, object]]:
    native = _count_native_rasters_by_region()
    rows: list[dict[str, object]] = []
    for region, coverage in REGIONAL_COVERAGE.items():
        total, canonical, dinov2 = coverage[0], coverage[2], coverage[3]
        native_paths = native.get(region, [])
        has_native = bool(native_paths)
        rows.append(
            {
                # Linha agregada por região: o contrato patch-level (MV2-07) vive em
                # outra branch; aqui reportamos o estado regional verificável.
                "patch_id": f"AGREGADO_{region.upper()}",
                "asset_id": f"{total}_contract_assets",
                "region": region,
                "has_png_render": "true" if canonical > 0 else "false",
                "has_npz_visual": "true",
                "has_dinov2_embedding": "true" if dinov2 > 0 else "false",
                "has_native_raster": str(has_native).lower(),
                "native_raster_path": native_paths[0] if native_paths else "",
                "available_bands": "NONE",
                "has_cloud_mask": "false",
                "has_scene_id": "false",
                # PNG/NPZ/DINOv2 visual NÃO suportam baseline espectral.
                "can_support_spectral_baseline": "false",
                "blocking_reason": (
                    "0 GeoTIFF/JP2/SAFE espectral legível; render visual e DINOv2 "
                    "visual não substituem raster Sentinel espectral nativo"
                ),
                "next_action": (
                    "baixar_GeoTIFF_L2A_com_bandas_B02_B03_B04_B08_B11_B12_e_SCL_via_"
                    "Copernicus_ou_reexportar_GEE_com_scene_id"
                ),
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_12_sentinel_native_raster_backlog.csv"
    write_csv(out, COLUMNS, rows)
    n_native = sum(1 for r in rows if r["has_native_raster"] == "true")
    n_spectral = sum(1 for r in rows if r["can_support_spectral_baseline"] == "true")
    print(f"[mv2_12_raster] regioes no backlog: {len(rows)}")
    print(f"[mv2_12_raster] com raster espectral nativo: {n_native}")
    print(f"[mv2_12_raster] que suportam baseline espectral: {n_spectral}")
    print(f"[mv2_12_raster] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
