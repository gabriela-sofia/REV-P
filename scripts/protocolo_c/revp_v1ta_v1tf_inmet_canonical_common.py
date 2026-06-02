"""Shared helpers — REV-P Protocol C v1ta-v1tf INMET canonical hydromet QA.

Review-only. No labels, targets, operational ground truth, formal negatives.
"""
from __future__ import annotations

import csv, hashlib, json, math, re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Re-export from downstream commons so callers need one import.
from revp_v1sg_v1sz_official_download_common import (  # noqa: F401
    DATASETS, DOCS, SCHEMAS, _p, raw_root,
    write_json_safe, write_doc,
    write_schema_for as write_schema,
    safe_relpath, hash_short, forbidden_guardrail_scan,
)
from revp_v1sr_v1sz_hydromet_context_common import (  # noqa: F401
    haversine_km, parse_date_safe, normalize_date, normalize_region,
    station_region_distances, nearest_region_and_distance,
    REGION_CENTROIDS, PROXIMITY_THRESHOLDS_KM,
    build_window, rolling_window_summary, guardrail_row, scan_guardrails,
    ABS_PATH_RE, FORBIDDEN_TRUE,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# CSV I/O (always writes header even on empty)
# ---------------------------------------------------------------------------

def read_csv_safe(path: Path | str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def write_csv_with_header(path: Path | str, rows: list[dict[str, Any]],
                          fields: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _viol = scan_guardrails(rows, p.name)
    if _viol:
        raise ValueError(f"Guardrail violation writing {p.name}: {_viol[:2]}")
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({f: row.get(f, "") for f in fields})


# ---------------------------------------------------------------------------
# Numeric / coordinate helpers
# ---------------------------------------------------------------------------

def parse_decimal_comma_float(text: str, default: float = 0.0) -> float:
    """Parse Brazilian decimal-comma notation (e.g. '-22,75777777' → -22.758)."""
    s = str(text or "").strip().replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def detect_coordinate_anomaly(lat: float, lon: float) -> str:
    """Return anomaly code or 'OK'."""
    if lat == 0.0 and lon == 0.0:
        return "ZERO_COORDS"
    # Brazil bounding box (rough): lat -33.8 to 5.3, lon -73.9 to -28.6
    if not (-34.0 <= lat <= 6.0):
        return "LAT_OUT_OF_BRAZIL"
    if not (-74.0 <= lon <= -28.0):
        return "LON_OUT_OF_BRAZIL"
    # Detect inverted lat/lon (lat looks like lon or vice-versa)
    if abs(lat) > 50 or abs(lon) < 20:
        return "POSSIBLE_LAT_LON_SWAP"
    return "OK"


def station_coordinate_quality_status(lat: float, lon: float,
                                       raw_lat_text: str, raw_lon_text: str) -> str:
    anomaly = detect_coordinate_anomaly(lat, lon)
    if anomaly != "OK":
        return f"COORD_ANOMALY_{anomaly}"
    # Check that the raw value actually contained a comma (proof of correct parse)
    if "," in str(raw_lat_text) or "," in str(raw_lon_text):
        return "CANONICAL_DECIMAL_COMMA_PARSED"
    return "CANONICAL_DECIMAL_POINT_PARSED"


def normalize_station_code(code: str) -> str:
    return str(code or "").strip().upper()


def normalize_uf(uf: str) -> str:
    return str(uf or "").strip().upper()[:2]


def compare_station_records(
    v1si_lat: str, v1si_lon: str, canon_lat: str, canon_lon: str
) -> dict[str, Any]:
    """Compare v1si vs canonical coordinates; return discrepancy dict."""
    si_lat = parse_decimal_comma_float(v1si_lat, 9999.0)
    si_lon = parse_decimal_comma_float(v1si_lon, 9999.0)
    ca_lat = parse_decimal_comma_float(canon_lat, 9999.0)
    ca_lon = parse_decimal_comma_float(canon_lon, 9999.0)

    if si_lat == 9999.0 or ca_lat == 9999.0:
        return {"delta_km": "", "discrepancy_type": "PARSE_FAILED"}

    if detect_coordinate_anomaly(si_lat, si_lon) != "OK":
        dtype = "V1SI_COORD_ANOMALY"
        delta = ""
    elif si_lat == ca_lat and si_lon == ca_lon:
        dtype = "NO_DISCREPANCY"
        delta = "0.00"
    else:
        try:
            d = haversine_km(si_lat, si_lon, ca_lat, ca_lon)
            delta = f"{d:.2f}"
            dtype = "DECIMAL_COMMA_CORRECTION" if d > 1.0 else "MINOR_ROUNDING"
        except Exception:
            delta = ""
            dtype = "CALCULATION_ERROR"

    return {"delta_km": delta, "discrepancy_type": dtype}
