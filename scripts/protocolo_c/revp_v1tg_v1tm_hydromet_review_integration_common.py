"""Shared helpers — REV-P Protocol C v1tg-v1tm hydromet review integration.

Review-only. Never creates labels, targets, operational ground truth, formal
negatives, or uses precipitation as validation/negative evidence.
"""
from __future__ import annotations

import csv, re
from pathlib import Path
from typing import Any

from revp_v1ta_v1tf_inmet_canonical_common import (  # noqa: F401
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_json_safe,
    write_schema, write_doc,
    safe_relpath, hash_short,
    guardrail_row, scan_guardrails, ABS_PATH_RE,
    parse_decimal_comma_float as parse_float_safe,
    normalize_region,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# New guardrail fields specific to this layer
# ---------------------------------------------------------------------------

EXTRA_GUARDRAIL_FIELDS = ["hydromet_validates_event", "hydromet_is_negative_evidence"]

EXTRA_FORBIDDEN_TRUE = ["hydromet_validates_event", "hydromet_is_negative_evidence"]


def guardrail_row_extended() -> dict[str, str]:
    """Standard guardrail fields plus hydromet-specific ones."""
    row = guardrail_row()
    row.update({
        "hydromet_validates_event":      "false",
        "hydromet_is_negative_evidence": "false",
    })
    return row


def scan_guardrails_extended(rows: list[dict[str, Any]], label: str) -> list[str]:
    """scan_guardrails + hydromet_validates_event + hydromet_is_negative_evidence."""
    issues = scan_guardrails(rows, label)
    for i, row in enumerate(rows):
        for f in EXTRA_FORBIDDEN_TRUE:
            if str(row.get(f, "false")).strip().lower() == "true":
                issues.append(f"{label}[{i}].{f}=true")
    return issues


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_event_id(text: str) -> str:
    return str(text or "").strip().upper().replace(" ", "_")[:120]


def normalize_patch_id(text: str) -> str:
    return str(text or "").strip().upper()[:40]


def normalize_review_sample_id(text: str) -> str:
    return str(text or "").strip().upper()[:40]


def normalize_decision(text: str) -> str:
    lo = str(text or "").strip().upper()
    if lo in ("C1", "C2", "C3", "C4"):
        return lo
    return "PENDING"


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_station_coverage(distance_km: float) -> str:
    if distance_km <= 0:
        return "NO_STATION"
    if distance_km <= 25:
        return "CLOSE_STATION_WITHIN_25KM"
    if distance_km <= 50:
        return "NEAR_STATION_WITHIN_50KM"
    if distance_km <= 100:
        return "DISTANT_STATION_WITHIN_100KM"
    return "VERY_DISTANT_STATION_BEYOND_100KM"


def classify_precipitation_context(rain_7d: float, rain_1d: float) -> str:
    """Returns context label. Never 'validated' or 'negative'."""
    if rain_7d < 0 or rain_1d < 0:
        return "PRECIPITATION_DATA_MISSING"
    if rain_7d >= 150 or rain_1d >= 50:
        return "HIGH_RAINFALL_CONTEXT_REVIEW_ONLY"
    if rain_7d >= 50 or rain_1d >= 20:
        return "MODERATE_RAINFALL_CONTEXT_REVIEW_ONLY"
    if rain_7d > 0 or rain_1d > 0:
        return "LOW_RAINFALL_CONTEXT_REVIEW_ONLY"
    return "ZERO_RAINFALL_CONTEXT_REVIEW_ONLY"


def classify_hydromet_support_level(
    distance_km: float, rain_7d: float, has_window: bool
) -> str:
    """Classify contextual support only — never 'validated'."""
    if not has_window:
        return "HYDROMET_CONTEXT_WAITING_EVENT_WINDOW"
    if distance_km <= 0:
        return "HYDROMET_CONTEXT_MISSING_STATION"
    if rain_7d < 0:
        return "HYDROMET_CONTEXT_MISSING_PRECIPITATION"
    if distance_km > 100:
        return "HYDROMET_CONTEXT_LIMITED_STATION_DISTANCE"
    return "HYDROMET_CONTEXT_AVAILABLE"


def build_hydromet_question_set(
    event_candidate_id: str,
    region: str,
    rain_7d: str,
    nearest_station: str,
    distance_km: str,
) -> list[dict[str, str]]:
    """Return review questions with empty answer slots. Never implies validation."""
    return [
        {
            "question_key":   "hydromet_precipitation_present",
            "question_text":  (
                f"Há precipitação oficial INMET na janela do evento "
                f"({event_candidate_id}, {region})? "
                f"rain_7d={rain_7d}mm. (sim/nao/incerto)"
            ),
        },
        {
            "question_key":   "hydromet_station_proximity_adequate",
            "question_text":  (
                f"A estação {nearest_station} (distância ~{distance_km}km) "
                f"está suficientemente próxima para contexto regional? (sim/nao/incerto)"
            ),
        },
        {
            "question_key":   "hydromet_temporal_compatibility",
            "question_text":  "A fonte INMET é temporalmente compatível com o evento? (sim/nao/incerto)",
        },
        {
            "question_key":   "hydromet_spatial_coverage_limitation",
            "question_text":  "Há limitação de cobertura espacial da estação para esta região? (sim/nao/incerto)",
        },
        {
            "question_key":   "hydromet_contextual_only",
            "question_text":  "A evidência INMET deve ser tratada apenas como contexto hidrometeorológico? (sim/nao/incerto)",
        },
        {
            "question_key":   "hydromet_overclaim_risk",
            "question_text":  "Existe risco de overclaim se a precipitação INMET for usada como validação? (sim/nao/incerto)",
        },
        {
            "question_key":   "hydromet_requires_independent_source",
            "question_text":  "O caso ainda requer fonte observacional independente além do INMET? (sim/nao/incerto)",
        },
    ]
