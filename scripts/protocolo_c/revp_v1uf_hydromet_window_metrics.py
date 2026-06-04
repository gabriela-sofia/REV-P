#!/usr/bin/env python3
"""
v1uf — Hydromet Window Metrics

For each extracted INMET station series, parses the CSV (BR encoding/separator/
decimal), normalizes datetime + precipitation, filters by v1ue temporal windows,
and computes precipitation metrics. Precipitation anchors temporal plausibility,
NEVER ground reference / label.
"""

import argparse
import csv
import os
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1uf"

METRIC_COLUMNS = [
    "metric_id", "event_id", "station_candidate_id", "source_id",
    "window_type", "window_start", "window_end", "observed_variable",
    "precipitation_total_mm", "precipitation_max_hourly_mm",
    "precipitation_max_daily_mm", "valid_observation_count",
    "missing_observation_count", "coverage_ratio", "temporal_overlap_status",
    "metric_status", "evidence_role", "can_support_temporal_gate",
    "can_create_ground_reference", "can_create_training_label", "limitations",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_local_asset(asset: dict, staging_root: str) -> str:
    """Locate the extracted CSV file in staging using event_id + station_name basename."""
    event_id = asset.get("event_id", "")
    fname = asset.get("station_name", "")
    candidate_dir = os.path.join(staging_root, "evidence_staging", "v1uf", "inmet", event_id)
    if not os.path.isdir(candidate_dir):
        return ""
    if fname:
        direct = os.path.join(candidate_dir, fname)
        if os.path.exists(direct):
            return direct
    # fallback: first CSV containing the station_code
    code = asset.get("station_code", "")
    for f in os.listdir(candidate_dir):
        if code and code.upper() in f.upper():
            return os.path.join(candidate_dir, f)
    return ""


def detect_encoding_read(path: str, encodings: list) -> list:
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read().splitlines()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        return f.read().splitlines()


def detect_separator(line: str, candidates: list) -> str:
    best = ";"
    best_count = -1
    for sep in candidates:
        c = line.count(sep)
        if c > best_count:
            best_count = c
            best = sep
    return best


def find_header_index(lines: list, datetime_hints: list, precip_hints: list) -> int:
    for i, line in enumerate(lines[:15]):
        low = line.lower()
        if any(h in low for h in datetime_hints) and (
            any(h in low for h in precip_hints) or "hora" in low
        ):
            return i
    return 0


def match_column(header_cells: list, hints: list) -> int:
    for idx, cell in enumerate(header_cells):
        low = cell.strip().lower()
        for h in hints:
            if h in low:
                return idx
    return -1


def parse_value_br(raw: str, na_tokens: list) -> float | None:
    s = (raw or "").strip().strip('"')
    if s in na_tokens or s == "":
        return None
    s = s.replace(",", ".")
    try:
        v = float(s)
        if v <= -9999:
            return None
        return v
    except ValueError:
        return None


def parse_datetime_cell(date_str: str, hour_str: str) -> datetime | None:
    date_str = (date_str or "").strip().strip('"')
    hour_str = (hour_str or "").strip().strip('"')
    if not date_str:
        return None
    # normalize hour like "0000 UTC" or "00:00"
    h = hour_str.replace("UTC", "").strip()
    if h.isdigit() and len(h) in (3, 4):
        h = h.zfill(4)
        h = f"{h[:2]}:{h[2:]}"
    for dfmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            d = datetime.strptime(date_str, dfmt).date()
            if h:
                for hfmt in ("%H:%M", "%H:%M:%S", "%H"):
                    try:
                        t = datetime.strptime(h, hfmt).time()
                        return datetime.combine(d, t)
                    except ValueError:
                        continue
            return datetime.combine(d, datetime.min.time())
        except ValueError:
            continue
    return None


def parse_series(path: str, policy: dict) -> list:
    """Returns list of (datetime, precip_mm) tuples."""
    parsing = policy.get("parsing", {})
    encodings = parsing.get("candidate_encodings", ["latin-1", "utf-8"])
    separators = parsing.get("candidate_separators", [";", ","])
    na_tokens = parsing.get("na_tokens", ["", "-9999"])
    col = policy.get("column_detection", {})
    dt_hints = [h.lower() for h in col.get("datetime_hints", ["data", "hora"])]
    precip_hints = [h.lower() for h in col.get("precipitation_hints", ["precipita", "chuva"])]

    lines = detect_encoding_read(path, encodings)
    if not lines:
        return []
    header_idx = find_header_index(lines, dt_hints, precip_hints)
    sep = detect_separator(lines[header_idx], separators)
    header_cells = [c.strip().lower() for c in lines[header_idx].split(sep)]

    date_idx = match_column(header_cells, ["data"])
    hour_idx = match_column(header_cells, ["hora"])
    precip_idx = match_column(header_cells, precip_hints)

    if date_idx < 0 or precip_idx < 0:
        return []

    series = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        cells = line.split(sep)
        if len(cells) <= max(date_idx, precip_idx, hour_idx):
            continue
        dt = parse_datetime_cell(cells[date_idx], cells[hour_idx] if hour_idx >= 0 else "")
        if dt is None:
            continue
        precip = parse_value_br(cells[precip_idx], na_tokens)
        series.append((dt, precip))
    return series


def compute_window_metrics(series: list, win_start: datetime, win_end: datetime) -> dict:
    in_window = [(dt, p) for dt, p in series if win_start <= dt <= win_end]
    valid = [(dt, p) for dt, p in in_window if p is not None]
    missing = len(in_window) - len(valid)
    expected_hours = int((win_end - win_start).total_seconds() // 3600) + 1

    if not valid:
        return {
            "total": "", "max_hourly": "", "max_daily": "",
            "valid_count": str(len(valid)), "missing_count": str(missing),
            "coverage": "0.0", "first": "", "last": "",
            "status": "INSUFFICIENT_COVERAGE" if expected_hours > 0 else "NO_DATA",
        }

    total = sum(p for _, p in valid)
    max_hourly = max(p for _, p in valid)
    daily = {}
    for dt, p in valid:
        daily.setdefault(dt.date(), 0.0)
        daily[dt.date()] += p
    max_daily = max(daily.values()) if daily else 0.0
    coverage = round(len(valid) / expected_hours, 3) if expected_hours > 0 else 0.0
    valid_sorted = sorted(valid, key=lambda x: x[0])

    return {
        "total": f"{total:.1f}", "max_hourly": f"{max_hourly:.1f}",
        "max_daily": f"{max_daily:.1f}", "valid_count": str(len(valid)),
        "missing_count": str(missing), "coverage": str(coverage),
        "first": valid_sorted[0][0].isoformat(), "last": valid_sorted[-1][0].isoformat(),
        "status": "COMPUTED" if coverage >= 0.5 else "INSUFFICIENT_COVERAGE",
    }


def main():
    parser = argparse.ArgumentParser(description="v1uf — Hydromet Window Metrics")
    parser.add_argument("--assets", default="datasets/protocolo_c/v1uf_station_series_asset_registry.csv")
    parser.add_argument("--windows", default="datasets/protocolo_c/v1ue_event_temporal_window_registry.csv")
    parser.add_argument("--policy", default="configs/protocolo_c/v1uf_hydromet_metrics_policy.yaml")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c")
    args = parser.parse_args()

    assets = load_csv(args.assets)
    windows = load_csv(args.windows)
    policy = load_yaml(args.policy)

    metric_windows = ["event_core_window", "pre_event_window_3d",
                      "pre_event_window_7d", "post_event_window_3d"]

    rows = []
    seq = 0
    extracted_assets = [a for a in assets if a.get("extraction_status") == "EXTRACTED"]

    for asset in extracted_assets:
        event_id = asset.get("event_id", "")
        sc_id = asset.get("station_candidate_id", "")
        source_id = asset.get("source_id", "")

        local_path = find_local_asset(asset, args.local_only_dir)
        series = parse_series(local_path, policy) if local_path else []

        ev_windows = [w for w in windows
                      if w["event_id"] == event_id and w["window_type"] in metric_windows]

        for w in ev_windows:
            try:
                ws = datetime.fromisoformat(w["window_start"])
                we = datetime.fromisoformat(w["window_end"]).replace(hour=23, minute=59, second=59)
            except ValueError:
                continue

            if not series:
                m = {"total": "", "max_hourly": "", "max_daily": "", "valid_count": "0",
                     "missing_count": "0", "coverage": "0.0", "first": "", "last": "",
                     "status": "NO_SERIES_DATA"}
                overlap = "NO_DATA"
            else:
                m = compute_window_metrics(series, ws, we)
                overlap = "COVERED" if m["status"] == "COMPUTED" else "PARTIAL_OR_NONE"

            rows.append({
                "metric_id": f"MET_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id,
                "station_candidate_id": sc_id,
                "source_id": source_id,
                "window_type": w["window_type"],
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                "observed_variable": "precipitation",
                "precipitation_total_mm": m["total"],
                "precipitation_max_hourly_mm": m["max_hourly"],
                "precipitation_max_daily_mm": m["max_daily"],
                "valid_observation_count": m["valid_count"],
                "missing_observation_count": m["missing_count"],
                "coverage_ratio": m["coverage"],
                "temporal_overlap_status": overlap,
                "metric_status": m["status"],
                "evidence_role": "temporal_anchor",
                "can_support_temporal_gate": "true" if m["status"] == "COMPUTED" else "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "limitations": "Precipitation anchors temporal plausibility; NOT patch-level truth",
            })
            seq += 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "v1uf_hydromet_window_metrics_registry.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for r in rows:
        statuses[r["metric_status"]] = statuses.get(r["metric_status"], 0) + 1
    print(f"[Hydromet Window Metrics v1uf] {len(rows)} window metrics from {len(extracted_assets)} assets")
    for s, c in sorted(statuses.items()):
        print(f"  {s}: {c}")
    print(f"  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {out_path}")


if __name__ == "__main__":
    main()
