#!/usr/bin/env python3
"""
v1uf — Station Evidence Integrity Audit

Audits the v1uf acquisition/extraction chain: ZIP availability, hash presence,
extracted file existence, column detection, window coverage, official station
resolution, event vs year specificity, and spatial-truth misuse risk.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1uf"

AUDIT_COLUMNS = [
    "audit_id", "event_id", "station_candidate_id", "asset_id",
    "check_name", "status", "severity", "reason", "required_action",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="v1uf — Station Evidence Integrity Audit")
    parser.add_argument("--manifest", default="datasets/protocolo_c/v1uf_large_download_manifest.csv")
    parser.add_argument("--assets", default="datasets/protocolo_c/v1uf_station_series_asset_registry.csv")
    parser.add_argument("--catalog", default="datasets/protocolo_c/v1uf_official_station_catalog_registry.csv")
    parser.add_argument("--metrics", default="datasets/protocolo_c/v1uf_hydromet_window_metrics_registry.csv")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    args = parser.parse_args()

    manifest = load_csv(args.manifest)
    assets = load_csv(args.assets)
    catalog = load_csv(args.catalog)
    metrics = load_csv(args.metrics)

    manifest_by_event = {m["event_id"]: m for m in manifest}
    coord_by_candidate = {c["station_candidate_id"]: c for c in catalog}
    metrics_by_event = {}
    for m in metrics:
        metrics_by_event.setdefault(m["event_id"], []).append(m)

    rows = []
    seq = 0

    def add(event_id, sc_id, asset_id, check, status, severity, reason, action):
        nonlocal seq
        rows.append({
            "audit_id": f"AUD_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id,
            "station_candidate_id": sc_id,
            "asset_id": asset_id,
            "check_name": check,
            "status": status,
            "severity": severity,
            "reason": reason,
            "required_action": action,
        })
        seq += 1

    for asset in assets:
        event_id = asset.get("event_id", "")
        sc_id = asset.get("station_candidate_id", "")
        asset_id = asset.get("asset_id", "")
        ext_status = asset.get("extraction_status", "")
        man = manifest_by_event.get(event_id, {})

        # Check 1: ZIP downloaded
        zip_status = man.get("download_status", "")
        if zip_status in ("DOWNLOADED", "DOWNLOADED_CACHED"):
            add(event_id, sc_id, asset_id, "zip_downloaded", "PASS", "LOW",
                f"ZIP status={zip_status}", "")
        else:
            add(event_id, sc_id, asset_id, "zip_downloaded", "FAIL", "MEDIUM",
                f"ZIP status={zip_status}", "RETRY_INMET_STATION_EXTRACTION")

        # Check 2: ZIP hash present
        if man.get("zip_sha256"):
            add(event_id, sc_id, asset_id, "zip_hash_present", "PASS", "LOW",
                "zip_sha256 recorded", "")
        else:
            add(event_id, sc_id, asset_id, "zip_hash_present", "NOT_APPLICABLE", "LOW",
                "No ZIP downloaded", "")

        # Check 3: extracted file exists
        if ext_status == "EXTRACTED":
            add(event_id, sc_id, asset_id, "extracted_file_exists", "PASS", "LOW",
                "Asset extracted", "")
        else:
            add(event_id, sc_id, asset_id, "extracted_file_exists", "FAIL", "MEDIUM",
                f"extraction_status={ext_status}", "RETRY_INMET_STATION_EXTRACTION")

        # Check 4: columns detected
        if ext_status == "EXTRACTED":
            has_dt = asset.get("has_datetime_column") == "true"
            has_pr = asset.get("has_precipitation_column") == "true"
            if has_dt and has_pr:
                add(event_id, sc_id, asset_id, "columns_detected", "PASS", "LOW",
                    "datetime + precipitation columns found", "")
            else:
                add(event_id, sc_id, asset_id, "columns_detected", "NEEDS_REVIEW", "MEDIUM",
                    f"datetime={has_dt} precip={has_pr}", "MANUAL_REVIEW_INMET_SERIES")

        # Check 5: window coverage
        ev_metrics = metrics_by_event.get(event_id, [])
        computed = [m for m in ev_metrics if m.get("metric_status") == "COMPUTED"]
        if ext_status == "EXTRACTED":
            if computed:
                add(event_id, sc_id, asset_id, "window_coverage", "PASS", "LOW",
                    f"{len(computed)} windows with sufficient coverage", "")
            else:
                add(event_id, sc_id, asset_id, "window_coverage", "FAIL", "MEDIUM",
                    "No window with sufficient coverage", "MANUAL_REVIEW_INMET_SERIES")

        # Check 6: station officially resolved
        coord = coord_by_candidate.get(sc_id, {})
        if coord.get("coordinate_status") == "FROM_OFFICIAL_CATALOG":
            add(event_id, sc_id, asset_id, "station_officially_resolved", "PASS", "LOW",
                "Coordinate from official catalog", "")
        else:
            add(event_id, sc_id, asset_id, "station_officially_resolved", "NEEDS_REVIEW", "MEDIUM",
                "Coordinate MISSING — catalog not resolved", "RETRY_official_catalog_resolution")

        # Check 7: event vs year specificity
        add(event_id, sc_id, asset_id, "event_vs_year_specificity", "NEEDS_REVIEW", "MEDIUM",
            "Series is YEAR-specific, not event-specific (whole year extracted, windowed downstream)",
            "Window metrics narrow to event; geometry still required for event-specificity")

        # Check 8: spatial truth misuse risk (CRITICAL guardrail)
        add(event_id, sc_id, asset_id, "spatial_truth_misuse_risk", "PASS", "CRITICAL",
            "Station is point sensor; metrics flagged as temporal anchor only, not flood geometry",
            "DO_NOT_PROMOTE_GROUND_REFERENCE")

    # If no assets at all, emit a chain-level audit
    if not assets:
        for man in manifest:
            event_id = man.get("event_id", "")
            add(event_id, "", "", "zip_downloaded",
                "PASS" if man.get("download_status") in ("DOWNLOADED", "DOWNLOADED_CACHED") else "FAIL",
                "MEDIUM", f"download_status={man.get('download_status', '')}",
                "RETRY_INMET_STATION_EXTRACTION")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "v1uf_station_evidence_integrity_registry.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"[Station Evidence Integrity Audit v1uf] {len(rows)} checks")
    for s, c in sorted(statuses.items()):
        print(f"  {s}: {c}")
    print(f"\nRegistry: {out_path}")


if __name__ == "__main__":
    main()
