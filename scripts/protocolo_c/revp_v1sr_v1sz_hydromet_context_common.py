"""Shared helpers for REV-P Protocol C v1sr-v1sz hydrometeorological context layer.

Review-only. Never creates labels, targets, operational ground truth or formal
negatives. Precipitation proximity or temporal overlap does not validate events.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Re-export from acquisition common so callers only need one import.
from revp_v1sg_v1sz_official_download_common import (  # noqa: F401
    DATASETS, DOCS, SCHEMAS, _p, raw_root,
    read_csv_safe, write_csv_with_header, write_json_safe, write_doc,
    write_schema_for as write_schema,
    safe_relpath, hash_short, forbidden_guardrail_scan,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Target region centroids (WGS-84 decimal degrees)
# ---------------------------------------------------------------------------

REGION_CENTROIDS: dict[str, tuple[float, float]] = {
    "RECIFE":   (-8.0539,  -34.8813),
    "PET":      (-22.5058, -43.1773),
    "CURITIBA": (-25.4284, -49.2733),
}

PROXIMITY_THRESHOLDS_KM = (25.0, 50.0, 100.0)

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def parse_float_safe(text: str, default: float = 0.0) -> float:
    try:
        return float(str(text or "").strip().replace(",", "."))
    except (ValueError, TypeError):
        return default


def station_region_distances(lat: float, lon: float) -> dict[str, float]:
    """Return distance in km from (lat, lon) to each target region centroid."""
    if lat == 0.0 and lon == 0.0:
        return {r: float("inf") for r in REGION_CENTROIDS}
    return {r: haversine_km(lat, lon, clat, clon) for r, (clat, clon) in REGION_CENTROIDS.items()}


def nearest_region_and_distance(lat: float, lon: float) -> tuple[str, float]:
    dists = station_region_distances(lat, lon)
    nearest = min(dists, key=lambda r: dists[r])
    return nearest, dists[nearest]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",   # YYYY-MM-DD or YYYY/MM/DD
    r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",   # DD/MM/YYYY or DD-MM-YYYY
]


def parse_date_safe(text: str) -> date | None:
    s = str(text or "").strip()
    if not s:
        return None
    for pat in _DATE_PATTERNS:
        m = re.search(pat, s)
        if m:
            g = m.groups()
            try:
                if len(g[0]) == 4:
                    return date(int(g[0]), int(g[1]), int(g[2]))
                else:
                    return date(int(g[2]), int(g[1]), int(g[0]))
            except (ValueError, OverflowError):
                continue
    return None


def normalize_date(text: str) -> str:
    """Return ISO YYYY-MM-DD or empty string."""
    d = parse_date_safe(text)
    return d.isoformat() if d else ""


def build_window(event_date: date, before_days: int = 7, after_days: int = 1) -> tuple[date, date]:
    return event_date - timedelta(days=before_days), event_date + timedelta(days=after_days)


def dates_in_window(start: date, end: date) -> list[date]:
    result = []
    d = start
    while d <= end:
        result.append(d)
        d += timedelta(days=1)
    return result


def window_position(event_date: date, d: date) -> str:
    delta = (d - event_date).days
    if delta < -5:
        return "EARLY_ANTECEDENT"
    if delta < 0:
        return "ANTECEDENT"
    if delta == 0:
        return "EVENT_DAY"
    return "POST_EVENT"


# ---------------------------------------------------------------------------
# Region / normalize helpers
# ---------------------------------------------------------------------------

def normalize_region(text: str) -> str:
    lo = str(text or "").upper().strip()
    if any(k in lo for k in ("PET", "PETROPO", "PETRÓPOLIS", "PETROPOLIS")): return "PET"
    if any(k in lo for k in ("RECIFE", "PERNAMBUCO", "PE")): return "RECIFE"
    if any(k in lo for k in ("CURITIBA", "PARANA", "PARANÁ", "PR")): return "CURITIBA"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Precipitation helpers
# ---------------------------------------------------------------------------

def daily_precip_aggregate(records: list[dict[str, str]]) -> dict[str, float]:
    """Aggregate hourly/sub-daily records to daily totals. {ISO_date: mm}"""
    totals: dict[str, float] = {}
    for r in records:
        d = normalize_date(r.get("date", ""))
        if not d:
            continue
        mm = parse_float_safe(r.get("precipitation_mm", "0"), 0.0)
        if mm < 0:
            mm = 0.0
        totals[d] = totals.get(d, 0.0) + mm
    return totals


def rolling_window_summary(
    daily: dict[str, float], anchor_date: date, windows: tuple[int, ...] = (1, 3, 7)
) -> dict[str, float]:
    """Compute rolling sums ending on anchor_date for each window length."""
    result: dict[str, float] = {}
    for w in windows:
        total = 0.0
        for delta in range(w):
            d = (anchor_date - timedelta(days=delta)).isoformat()
            total += daily.get(d, 0.0)
        result[f"rain_{w}d"] = round(total, 2)
    return result


# ---------------------------------------------------------------------------
# Guardrail helpers
# ---------------------------------------------------------------------------

GUARDRAIL_FIELDS = [
    "review_only", "does_not_validate_event", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "formal_negative", "dino_validates_event", "absence_as_negative",
]

FORBIDDEN_TRUE = [
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "dino_validates_event",
    "absence_as_negative",
]

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
_FORBIDDEN_LITERAL = "local" + "_runs"


def guardrail_row() -> dict[str, str]:
    return {
        "review_only": "true",
        "does_not_validate_event": "true",
        "can_create_operational_label": "false",
        "can_train_model": "false",
        "target_created": "false",
        "ground_truth_operational": "false",
        "formal_negative": "false",
        "dino_validates_event": "false",
        "absence_as_negative": "false",
    }


def scan_guardrails(rows: list[dict[str, Any]], label: str) -> list[str]:
    """Return list of violation strings (empty = clean)."""
    issues: list[str] = []
    for i, row in enumerate(rows):
        for f in FORBIDDEN_TRUE:
            if str(row.get(f, "false")).strip().lower() == "true":
                issues.append(f"{label}[{i}].{f}=true")
        for k, v in row.items():
            sv = str(v)
            if ABS_PATH_RE.search(sv):
                issues.append(f"{label}[{i}].{k}=abs_path")
            if _FORBIDDEN_LITERAL in sv.lower():
                issues.append(f"{label}[{i}].{k}=local_runs_exposure")
    return issues
