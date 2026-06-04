#!/usr/bin/env python3
"""
v1uf — Official Station Catalog Resolver

Resolves official station coordinates WITHOUT inventing anything.
Coordinates are filled ONLY from an official catalog downloaded this session
(saved to local_only with hash). Otherwise coordinate_status stays MISSING.

NEVER geocodes by city name. NEVER uses municipal centroid as station coordinate.
A station is NEVER flood geometry. Also writes the station binding registry.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None

try:
    import requests
except ImportError:
    requests = None

PROTOCOL_VERSION = "v1uf"
USER_AGENT = (
    "REV-P-AcademicResearch/1.0 "
    "(TCC ground reference evidence acquisition; contact: academic use only)"
)

CATALOG_COLUMNS = [
    "station_candidate_id", "source_id", "official_catalog_id", "station_code",
    "station_name", "municipality", "uf", "latitude", "longitude",
    "altitude_m", "coordinate_status", "coordinate_source_url",
    "coordinate_source_sha256", "catalog_acquisition_status",
    "match_confidence", "match_method", "limitations",
    "can_anchor_temporal_evidence", "can_anchor_spatial_context",
    "can_create_ground_reference",
]

BINDING_COLUMNS = [
    "station_binding_id", "event_id", "station_candidate_id", "source_id",
    "station_code", "station_name", "station_type", "city", "uf",
    "event_start", "event_end", "coordinate_status", "distance_to_city_km",
    "distance_status", "binding_method", "binding_confidence",
    "can_support_temporal_gate", "can_support_hydromet_plausibility",
    "can_support_spatial_context", "cannot_support_patch_truth_reason", "notes",
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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel_path(path: str) -> str:
    try:
        return os.path.relpath(path, start=".").replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def fetch_inmet_catalog(url: str, dest_dir: str, timeout: int) -> dict:
    """Fetch INMET station catalog (JSON). Returns parsed catalog + provenance."""
    if requests is None:
        return {"status": "DEPENDENCY_MISSING", "stations": [], "sha256": "", "saved_path": ""}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        raw = resp.content
        sha = sha256_bytes(raw)
        os.makedirs(dest_dir, exist_ok=True)
        saved = os.path.join(dest_dir, "inmet_station_catalog.json")
        with open(saved, "wb") as f:
            f.write(raw)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return {"status": "PARSE_ERROR", "stations": [], "sha256": sha, "saved_path": rel_path(saved)}
        stations = data if isinstance(data, list) else data.get("estacoes", [])
        return {"status": "FETCHED", "stations": stations, "sha256": sha, "saved_path": rel_path(saved)}
    except requests.exceptions.RequestException as e:
        return {"status": f"NETWORK_ERROR:{type(e).__name__}", "stations": [], "sha256": "", "saved_path": ""}


def match_inmet_station(code: str, catalog_stations: list) -> dict:
    """Find a station by code in the INMET catalog. Returns coordinate fields or empty."""
    for st in catalog_stations:
        cd = str(st.get("CD_ESTACAO", st.get("codigo", ""))).strip().upper()
        if cd == code.upper():
            lat = st.get("VL_LATITUDE", st.get("latitude", ""))
            lon = st.get("VL_LONGITUDE", st.get("longitude", ""))
            alt = st.get("VL_ALTITUDE", st.get("altitude", ""))
            name = st.get("DC_NOME", st.get("nome", ""))
            muni = st.get("DC_NOME", st.get("municipio", ""))
            return {
                "found": True, "latitude": str(lat), "longitude": str(lon),
                "altitude": str(alt), "name": str(name), "municipality": str(muni),
            }
    return {"found": False}


def main():
    parser = argparse.ArgumentParser(description="v1uf — Official Station Catalog Resolver")
    parser.add_argument("--stations", default="datasets/protocolo_c/v1ue_station_candidate_registry.csv")
    parser.add_argument("--catalog-config", default="configs/protocolo_c/v1uf_station_catalog_sources.yaml")
    parser.add_argument("--binding-config", default="configs/protocolo_c/v1uf_station_target_binding.yaml")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stations = load_csv(args.stations)
    catalog_config = load_yaml(args.catalog_config)
    binding_config = load_yaml(args.binding_config)
    events = {e["event_id"]: e for e in load_csv(args.events)}

    # Target station codes per source
    target_codes = {}  # source_id -> {code: {city, uf, type}}
    for t in catalog_config.get("target_station_codes", []):
        target_codes.setdefault(t["source_id"], {})[t["station_code"]] = t

    # Fetch INMET catalog once (if allowed)
    inmet_catalog = {"status": "NOT_ATTEMPTED", "stations": [], "sha256": "", "saved_path": ""}
    inmet_url = ""
    if args.allow_web and not args.dry_run:
        for cat in catalog_config.get("catalog_sources", []):
            if cat["catalog_id"] == "INMET_STATION_CATALOG":
                for u in cat.get("base_urls", []):
                    if "apitempo" in u:
                        inmet_url = u
                        break
                if inmet_url:
                    dest = os.path.join(args.local_only_dir, "evidence_raw", "v1uf", "INMET_BDMEP", "_catalog")
                    inmet_catalog = fetch_inmet_catalog(inmet_url, dest, args.timeout)
                break

    print(f"[Station Catalog Resolver v1uf] INMET catalog status: {inmet_catalog['status']}")
    if inmet_catalog["stations"]:
        print(f"  Catalog stations loaded: {len(inmet_catalog['stations'])}")

    catalog_rows = []
    seen = set()
    for st in stations:
        sc_id = st.get("station_candidate_id", "")
        source_id = st.get("source_id", "")
        # Determine candidate code from station_search policy code or known target
        st_code = st.get("station_code", "")
        if st_code in ("UNKNOWN_NEEDS_RESOLUTION", ""):
            # try to find code from target_codes by city
            city = st.get("city", "")
            match_code = ""
            for code, info in target_codes.get(source_id, {}).items():
                if info.get("city") == city:
                    match_code = code
                    break
            st_code = match_code or st_code

        key = (sc_id, st_code)
        if key in seen:
            continue
        seen.add(key)

        lat = lon = alt = ""
        coord_status = "MISSING"
        coord_url = ""
        coord_sha = ""
        match_conf = "0.0"
        match_method = "NONE"
        muni = st.get("city", "")
        st_name = st.get("station_name", "")

        if (source_id == "INMET_BDMEP" and st_code and st_code != "UNKNOWN_NEEDS_RESOLUTION"
                and inmet_catalog["stations"]):
            m = match_inmet_station(st_code, inmet_catalog["stations"])
            if m.get("found"):
                lat = m["latitude"]
                lon = m["longitude"]
                alt = m["altitude"]
                st_name = m["name"] or st_name
                muni = m["municipality"] or muni
                coord_status = "FROM_OFFICIAL_CATALOG"
                coord_url = inmet_url
                coord_sha = inmet_catalog["sha256"]
                match_conf = "1.0"
                match_method = "INMET_CODE_EXACT_MATCH"

        catalog_rows.append({
            "station_candidate_id": sc_id,
            "source_id": source_id,
            "official_catalog_id": "INMET_STATION_CATALOG" if source_id == "INMET_BDMEP" else "",
            "station_code": st_code,
            "station_name": st_name,
            "municipality": muni,
            "uf": st.get("uf", ""),
            "latitude": lat,
            "longitude": lon,
            "altitude_m": alt,
            "coordinate_status": coord_status,
            "coordinate_source_url": coord_url,
            "coordinate_source_sha256": coord_sha,
            "catalog_acquisition_status": inmet_catalog["status"] if source_id == "INMET_BDMEP" else "NOT_ATTEMPTED",
            "match_confidence": match_conf,
            "match_method": match_method,
            "limitations": "Station anchors time/plausibility; NOT flood geometry",
            "can_anchor_temporal_evidence": "true",
            "can_anchor_spatial_context": "false",
            "can_create_ground_reference": "false",
        })

    # Station binding registry
    coord_by_candidate = {r["station_candidate_id"]: r for r in catalog_rows}
    binding_rows = []
    bseq = 0
    for b in binding_config.get("bindings", []):
        event_id = b["event_id"]
        ev = events.get(event_id, {})
        # find station candidate(s) for this event/source
        ev_stations = [s for s in stations
                       if s.get("event_id") == event_id and s.get("source_id") == b.get("source_id")]
        for st in ev_stations:
            sc_id = st.get("station_candidate_id", "")
            cat = coord_by_candidate.get(sc_id, {})
            coord_status = cat.get("coordinate_status", "MISSING")
            distance_status = "NOT_COMPUTED" if coord_status == "MISSING" else "OFFICIAL_COORD_AVAILABLE_NO_SAFE_CITY_REF"
            binding_rows.append({
                "station_binding_id": f"BND_{PROTOCOL_VERSION}_{bseq:04d}",
                "event_id": event_id,
                "station_candidate_id": sc_id,
                "source_id": b.get("source_id", ""),
                "station_code": cat.get("station_code", st.get("station_code", "")),
                "station_name": cat.get("station_name", st.get("station_name", "")),
                "station_type": st.get("station_type", ""),
                "city": b.get("city", ""),
                "uf": b.get("uf", ""),
                "event_start": ev.get("start_date", ""),
                "event_end": ev.get("end_date", ""),
                "coordinate_status": coord_status,
                "distance_to_city_km": "",
                "distance_status": distance_status,
                "binding_method": b.get("binding_method", ""),
                "binding_confidence": "1.0" if coord_status == "FROM_OFFICIAL_CATALOG" else "0.5",
                "can_support_temporal_gate": "true",
                "can_support_hydromet_plausibility": "true",
                "can_support_spatial_context": "false",
                "cannot_support_patch_truth_reason": "Station is point sensor, not flood extent geometry",
                "notes": b.get("note", ""),
            })
            bseq += 1

    os.makedirs(args.out_dir, exist_ok=True)
    cat_path = os.path.join(args.out_dir, "v1uf_official_station_catalog_registry.csv")
    with open(cat_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(catalog_rows)

    bind_path = os.path.join(args.out_dir, "v1uf_station_binding_registry.csv")
    with open(bind_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BINDING_COLUMNS)
        writer.writeheader()
        writer.writerows(binding_rows)

    resolved = sum(1 for r in catalog_rows if r["coordinate_status"] == "FROM_OFFICIAL_CATALOG")
    missing = sum(1 for r in catalog_rows if r["coordinate_status"] == "MISSING")
    print(f"\n[Results] Catalog rows: {len(catalog_rows)} | Bindings: {len(binding_rows)}")
    print(f"  coordinate FROM_OFFICIAL_CATALOG: {resolved}")
    print(f"  coordinate MISSING: {missing}")
    print(f"  can_anchor_spatial_context=false (all — stations not flood geometry)")
    print(f"\n  Catalog: {cat_path}")
    print(f"  Binding: {bind_path}")


if __name__ == "__main__":
    main()
