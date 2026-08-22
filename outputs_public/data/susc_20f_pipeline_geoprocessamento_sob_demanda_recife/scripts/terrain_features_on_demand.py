"""SUSC-20F -- amostragem de terreno sob demanda (elevacao, declividade, HAND-Dinf,"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import rasterio
from pyproj import Transformer

RASTER_FILES = {
    "elevation_m": "dem_filled.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "hand_m_dinf": "hand_dinf.tif",
    "twi_dinf": "twi_dinf.tif",
}


def _raster_dir() -> Path | None:
    raw = os.environ.get("REVP_SUSC20F_TERRAIN_RASTER_DIR", "")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


_transformer_to_31985 = Transformer.from_crs("EPSG:4326", "EPSG:31985", always_xy=True)


def sample_terrain_features(lat: float, lon: float) -> Optional[dict]:
    raster_dir = _raster_dir()
    if raster_dir is None:
        return None

    x, y = _transformer_to_31985.transform(lon, lat)

    out: dict[str, float] = {}
    for feat_name, fname in RASTER_FILES.items():
        path = raster_dir / fname
        if not path.exists():
            return None
        with rasterio.open(path) as ds:
            row, col = ds.index(x, y)
            if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
                return None
            val = next(ds.sample([(x, y)]))[0]
            if ds.nodata is not None and val == ds.nodata:
                return None
            out[feat_name] = float(val)
    return out


def coverage_bbox_wgs84() -> tuple[float, float, float, float]:
    return (-35.032950343243996, -8.167561668035802, -34.84255396733535, -7.9156195118262955)
