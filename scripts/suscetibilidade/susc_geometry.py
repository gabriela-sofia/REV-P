"""
SUSC shared geometry helpers (pure Python, optional shapely).

bbox parsing, point-in-bbox, bbox intersection, haversine distance, simple
GeoJSON coordinate extraction, Brazil/region coordinate validation. Uses shapely
only if available; never requires it.
"""

from __future__ import annotations

import importlib.util
import math

# shapely is optional and never required; detect without importing.
HAS_SHAPELY = importlib.util.find_spec("shapely") is not None

BRAZIL_BOUNDS = (-75.0, -30.0, -35.0, 6.0)  # lon_min, lon_max, lat_min, lat_max
REGION_BOUNDS = {
    "recife": (-35.20, -34.80, -8.50, -7.85),
    "petropolis": (-43.45, -43.00, -22.75, -22.20),
    "curitiba": (-49.55, -49.00, -25.75, -25.20),
}


def parse_bbox(s):
    """Parse 'xmin;ymin;xmax;ymax' (or comma) into [xmin,ymin,xmax,ymax] or None."""
    if not s:
        return None
    sep = ";" if ";" in s else ","
    try:
        xs = [float(v) for v in str(s).split(sep)]
        return xs if len(xs) == 4 else None
    except (ValueError, AttributeError):
        return None


def point_in_bbox(lon, lat, bbox) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def bbox_intersect(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_of_points(points):
    if not points:
        return None
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return [min(lons), min(lats), max(lons), max(lats)]


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def in_brazil(lon, lat) -> bool:
    lo, hi, la, ha = BRAZIL_BOUNDS
    return lo <= lon <= hi and la <= lat <= ha


def region_of_coord(lon, lat) -> str:
    for reg, (lo, hi, la, ha) in REGION_BOUNDS.items():
        if lo <= lon <= hi and la <= lat <= ha:
            return reg
    return "unknown"


def geojson_points(obj):
    """Extract all (lon, lat) pairs and the geometry types from a GeoJSON object."""
    geoms = []
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        geoms = [ft.get("geometry") for ft in obj.get("features", [])]
    elif isinstance(obj, dict) and obj.get("type") == "Feature":
        geoms = [obj.get("geometry")]
    elif isinstance(obj, dict) and "coordinates" in obj:
        geoms = [obj]
    pts = []
    types = []

    def walk(c):
        if isinstance(c, list):
            if c and isinstance(c[0], (int, float)) and len(c) >= 2:
                pts.append((float(c[0]), float(c[1])))
            else:
                for x in c:
                    walk(x)

    for g in geoms:
        if g:
            types.append(g.get("type", "unknown"))
            walk(g.get("coordinates", []))
    return pts, types
