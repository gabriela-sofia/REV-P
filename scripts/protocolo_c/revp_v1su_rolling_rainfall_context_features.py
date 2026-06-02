"""REV-P v1su — Rolling rainfall context features (review-only).

Computes 1-day, 3-day, 7-day rolling precipitation sums per station per
event window. Features are strictly contextual; they cannot become targets
or training labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_schema, write_doc,
    guardrail_row, scan_guardrails, read_csv_safe,
    parse_float_safe, parse_date_safe, rolling_window_summary,
    hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_FEAT = _p("REVP_V1SU_OUT_FEAT", DATASETS / "protocol_c_rolling_rainfall_context_features_v1su.csv")
OUT_SUM  = _p("REVP_V1SU_OUT_SUM",  DATASETS / "protocol_c_rolling_rainfall_context_features_summary_v1su.csv")
SCHEMA_F = _p("REVP_V1SU_SCHEMA_F", SCHEMAS  / "protocol_c_rolling_rainfall_context_features_v1su_schema.csv")
SCHEMA_S = _p("REVP_V1SU_SCHEMA_S", SCHEMAS  / "protocol_c_rolling_rainfall_context_features_summary_v1su_schema.csv")
DOC      = _p("REVP_V1SU_DOC",      DOCS     / "revp_v1su_rolling_rainfall_context_features.md")

FEAT_FIELDS = [
    "feature_id", "event_window_id", "region", "station_code",
    "anchor_date", "rain_1d", "rain_3d", "rain_7d",
    "max_1d_in_window", "station_count", "nearest_station_distance_km",
    "feature_status",
    "review_only", "does_not_validate_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_MAX_ROWS = 3000


def run() -> dict[str, Any]:
    ctx_rows = read_csv_safe(DATASETS / "protocol_c_inmet_precipitation_event_window_context_v1st.csv")
    win_rows = read_csv_safe(DATASETS / "protocol_c_event_date_windows_v1ss.csv")
    prox_rows = read_csv_safe(DATASETS / "protocol_c_inmet_station_region_proximity_v1sr.csv")

    # Build station → distance lookup
    station_dist: dict[str, str] = {r["station_code"]: r.get("distance_km", "") for r in prox_rows}

    # Build per-station daily series from context table
    # {(event_window_id, station_code): {iso_date: mm}}
    from collections import defaultdict
    series: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in ctx_rows:
        mm_str = r.get("precipitation_mm", "")
        if not mm_str:
            continue
        mm = parse_float_safe(mm_str, -1.0)
        if mm < 0:
            continue
        key = (r.get("event_window_id", ""), r.get("station_code", ""))
        d = r.get("date", "")
        if d:
            series[key][d] = mm

    rows: list[dict[str, Any]] = []

    for w in win_rows:
        if w.get("blocked_reason"):
            continue
        ev_d = parse_date_safe(w.get("parsed_date", ""))
        if not ev_d:
            continue
        wid = w["event_window_id"]
        region = w.get("region", "")

        # Collect all stations for this window
        station_keys = {k[1] for k in series if k[0] == wid}
        if not station_keys:
            continue

        # Per station: compute rolling features anchored on event date
        for code in sorted(station_keys):
            if len(rows) >= _MAX_ROWS:
                break
            daily = series.get((wid, code), {})
            roll = rolling_window_summary(daily, ev_d, windows=(1, 3, 7))
            max_1d = max((v for v in daily.values()), default=0.0)
            dist_km = station_dist.get(code, "")

            row: dict[str, Any] = {
                "feature_id":     f"V1SU_F{hash_short(wid+code,8)}",
                "event_window_id": wid,
                "region":          region,
                "station_code":    code,
                "anchor_date":     ev_d.isoformat(),
                "rain_1d":         str(roll.get("rain_1d", 0.0)),
                "rain_3d":         str(roll.get("rain_3d", 0.0)),
                "rain_7d":         str(roll.get("rain_7d", 0.0)),
                "max_1d_in_window": f"{max_1d:.2f}",
                "station_count":   str(len(station_keys)),
                "nearest_station_distance_km": dist_km,
                "feature_status":  "ROLLING_CONTEXT_REVIEW_ONLY",
                "notes":           "",
            }
            row.update(guardrail_row())
            rows.append(row)
        if len(rows) >= _MAX_ROWS:
            break

    if not rows:
        rows = [{
            "feature_id": "FAIL_CLOSED_NO_FEATURES",
            "event_window_id": "", "region": "", "station_code": "",
            "anchor_date": "", "rain_1d": "", "rain_3d": "", "rain_7d": "",
            "max_1d_in_window": "", "station_count": "0",
            "nearest_station_distance_km": "",
            "feature_status": "FAIL_CLOSED_NO_CONTEXT_DATA",
            "notes": "", **guardrail_row(),
        }]

    violations = scan_guardrails(rows, "v1su_feat")
    if violations:
        raise ValueError(f"Guardrail violations in v1su: {violations[:3]}")

    write_csv_with_header(OUT_FEAT, rows, FEAT_FIELDS)
    write_schema(SCHEMA_F, FEAT_FIELDS, "v1su_features")

    real_rows = [r for r in rows if r["feature_status"] == "ROLLING_CONTEXT_REVIEW_ONLY"]
    windows_covered = len({r["event_window_id"] for r in real_rows})
    summary = [
        {"stat_key": "feature_rows",         "stat_value": str(len(real_rows))},
        {"stat_key": "event_windows_covered", "stat_value": str(windows_covered)},
        {"stat_key": "can_be_target",         "stat_value": "false"},
        {"stat_key": "can_be_label",          "stat_value": "false"},
        {"stat_key": "stage",                 "stat_value": "v1su"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1su_summary")

    write_doc(DOC, "v1su — Rolling Rainfall Context Features", [
        "## Objetivo",
        "Calcular features contextuais de precipitacao (1d/3d/7d) por estacao "
        "e janela de evento. Nao podem virar target ou label.",
        f"## Resultado\nFeatures: {len(real_rows)}. Janelas cobertas: {windows_covered}.",
        "## Limitacao",
        "Features de precipitacao sao descritivas; nao validam evento nem criam "
        "ground truth operacional.",
    ])
    print(f"[v1su] features={len(real_rows)} windows_covered={windows_covered}")
    return {"features": len(real_rows), "windows": windows_covered}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1su rolling rainfall context").parse_args()
    run()
