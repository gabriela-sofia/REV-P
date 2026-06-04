#!/usr/bin/env python3
"""
v1ue — Temporal Window Builder

Builds temporal search windows per event. Windows anchor time/plausibility,
NEVER flood geometry. No overlay executed.
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ue"

WINDOW_COLUMNS = [
    "window_id", "event_id", "region", "city", "start_date", "end_date",
    "window_type", "window_start", "window_end", "purpose",
    "can_support_temporal_gate", "can_create_label",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_date(s: str):
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def build_window(event: dict, wdef: dict) -> dict:
    start = parse_date(event["start_date"])
    end = parse_date(event["end_date"])
    anchor = wdef.get("anchor", "declared_event")
    off_start = wdef.get("offset_start_days", 0)
    off_end = wdef.get("offset_end_days", 0)

    if anchor == "declared_event":
        win_start = start
        win_end = end
    elif anchor == "start_date":
        win_start = start + timedelta(days=off_start)
        win_end = start + timedelta(days=off_end)
    elif anchor == "end_date":
        win_start = end + timedelta(days=off_start)
        win_end = end + timedelta(days=off_end)
    elif anchor == "event_span":
        win_start = start + timedelta(days=off_start)
        win_end = end + timedelta(days=off_end)
    else:
        win_start = start
        win_end = end

    return {
        "window_start": win_start.isoformat(),
        "window_end": win_end.isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="v1ue — Temporal Window Builder")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--windows-config", default="configs/protocolo_c/v1ue_event_temporal_windows.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ue_event_temporal_window_registry.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.windows_config)
    events = load_csv(args.events)
    window_defs = config.get("window_definitions", [])

    rows = []
    seq = 0
    for event in events:
        for wdef in window_defs:
            win = build_window(event, wdef)
            rows.append({
                "window_id": f"WIN_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event["event_id"],
                "region": event.get("region", ""),
                "city": event.get("city", ""),
                "start_date": event["start_date"],
                "end_date": event["end_date"],
                "window_type": wdef["window_type"],
                "window_start": win["window_start"],
                "window_end": win["window_end"],
                "purpose": wdef.get("purpose", ""),
                "can_support_temporal_gate": str(wdef.get("can_support_temporal_gate", False)).lower(),
                "can_create_label": "false",
            })
            seq += 1

    if args.dry_run:
        print(f"[Temporal Window Builder v1ue] DRY RUN — would create {len(rows)} windows")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Temporal Window Builder v1ue] {len(rows)} windows for {len(events)} events")
    wtypes = {}
    for r in rows:
        wtypes[r["window_type"]] = wtypes.get(r["window_type"], 0) + 1
    for wt, c in sorted(wtypes.items()):
        print(f"  {wt}: {c}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
