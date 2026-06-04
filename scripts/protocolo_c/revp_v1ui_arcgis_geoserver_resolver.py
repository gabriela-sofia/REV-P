#!/usr/bin/env python3
"""
v1ui — ArcGIS / GeoServer Resolver

Detects and registers metadata from ArcGIS REST and GeoServer/WFS services.
Does not download full feature sets — only registers layer metadata.
"""

import argparse
import csv
import json
import os

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ui"

LAYER_COLUMNS = [
    "service_id", "event_id", "source_id", "service_url", "service_type",
    "layer_id", "layer_name", "geometry_type", "spatial_reference",
    "extent", "fields", "feature_count_estimate",
    "event_relevance_score", "can_query_features", "sample_downloaded",
    "candidate_status", "notes",
]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_yaml(path):
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fetch_json(url, timeout=30):
    if not HAS_URLLIB:
        return None
    try:
        if "?" in url:
            url += "&f=json"
        else:
            url += "?f=json"
        req = urllib.request.Request(url,
            headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read(500_000).decode("utf-8"))
    except Exception:
        return None


def resolve_arcgis(url, event_id, source_id, seq, allow_web, timeout):
    rows = []
    if not allow_web:
        rows.append({
            "service_id": f"SVC_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id, "source_id": source_id,
            "service_url": url, "service_type": "arcgis_rest",
            "layer_id": "", "layer_name": "",
            "geometry_type": "", "spatial_reference": "",
            "extent": "", "fields": "", "feature_count_estimate": "",
            "event_relevance_score": "0", "can_query_features": "false",
            "sample_downloaded": "false",
            "candidate_status": "DRY_RUN", "notes": "",
        })
        return rows, seq + 1

    data = fetch_json(url, timeout)
    if not data:
        rows.append({
            "service_id": f"SVC_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id, "source_id": source_id,
            "service_url": url, "service_type": "arcgis_rest",
            "layer_id": "", "layer_name": "",
            "geometry_type": "", "spatial_reference": "",
            "extent": "", "fields": "", "feature_count_estimate": "",
            "event_relevance_score": "0", "can_query_features": "false",
            "sample_downloaded": "false",
            "candidate_status": "FETCH_FAILED", "notes": "",
        })
        return rows, seq + 1

    layers = data.get("layers", []) or data.get("services", [])
    if not layers and data.get("name"):
        layers = [data]

    for layer in layers[:50]:
        lid = str(layer.get("id", ""))
        lname = layer.get("name", "")
        gtype = layer.get("geometryType", "")
        sr = ""
        extent_info = layer.get("extent", {})
        if isinstance(extent_info, dict):
            sr_info = extent_info.get("spatialReference", {})
            sr = str(sr_info.get("wkid", "")) if sr_info else ""
        fields = [f.get("name", "") for f in layer.get("fields", [])[:20]]

        rows.append({
            "service_id": f"SVC_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id, "source_id": source_id,
            "service_url": url, "service_type": "arcgis_rest",
            "layer_id": lid, "layer_name": lname,
            "geometry_type": gtype, "spatial_reference": sr,
            "extent": json.dumps(extent_info)[:200] if extent_info else "",
            "fields": "|".join(fields),
            "feature_count_estimate": "",
            "event_relevance_score": "0",
            "can_query_features": str(bool(gtype)).lower(),
            "sample_downloaded": "false",
            "candidate_status": "REGISTERED", "notes": "",
        })
        seq += 1

    return rows, seq


def main():
    parser = argparse.ArgumentParser(description="v1ui — ArcGIS / GeoServer Resolver")
    parser.add_argument("--discovery", default="datasets/protocolo_c/v1ui_public_discovery_registry.csv")
    parser.add_argument("--crawl-manifest", default="datasets/protocolo_c/v1ui_public_artifact_download_manifest.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_arcgis_geoserver_layer_registry.csv")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    discoveries = load_csv(args.discovery)
    crawl = load_csv(args.crawl_manifest)

    service_urls = set()
    entries = []
    for d in discoveries:
        if d.get("candidate_class") in ("ARCGIS_REST_CANDIDATE", "GEOSERVER_WFS_CANDIDATE"):
            entries.append(d)
            service_urls.add(d.get("candidate_url", ""))
    for c in crawl:
        stype = c.get("detected_service_type", "")
        if stype.startswith("arcgis") or stype.startswith("geoserver"):
            url = c.get("discovered_url", "")
            if url not in service_urls:
                entries.append(c)
                service_urls.add(url)

    rows = []
    seq = 0
    for entry in entries:
        url = entry.get("candidate_url", "") or entry.get("discovered_url", "")
        event_id = entry.get("event_id", "")
        source_id = entry.get("source_id", "")
        new_rows, seq = resolve_arcgis(url, event_id, source_id, seq,
                                        args.allow_web, args.timeout)
        rows.extend(new_rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LAYER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    registered = sum(1 for r in rows if r["candidate_status"] == "REGISTERED")
    print(f"[ArcGIS/GeoServer Resolver v1ui] {len(rows)} entries | registered={registered}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
