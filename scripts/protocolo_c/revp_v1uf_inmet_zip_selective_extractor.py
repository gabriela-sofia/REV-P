#!/usr/bin/env python3
"""
v1uf — INMET ZIP Selective Extractor

Opens downloaded INMET year ZIPs safely, lists internal files, and extracts
ONLY files matching target stations (code / name / UF / municipality / year).
Never extracts everything indiscriminately. Records each asset with hashes.
"""

import argparse
import csv
import hashlib
import os
import sys
import zipfile

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1uf"

ASSET_COLUMNS = [
    "asset_id", "event_id", "source_id", "year", "station_candidate_id",
    "station_code", "station_name", "zip_sha256", "internal_zip_path",
    "extracted_local_path_hash", "file_sha256", "file_size_bytes",
    "extraction_status", "reason", "has_datetime_column",
    "has_precipitation_column", "has_temperature_column",
    "has_quality_flags", "notes",
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


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rel_path(path: str) -> str:
    try:
        return os.path.relpath(path, start=".").replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def matches_station(internal_name: str, codes: list, name_hints: list, uf: str, year: str) -> bool:
    upper = internal_name.upper()
    if year and year not in internal_name:
        return False
    code_hit = any(c.upper() in upper for c in codes) if codes else False
    name_hit = any(h.upper() in upper for h in name_hints) if name_hints else False
    return code_hit or name_hit


def detect_columns(header_line: str) -> dict:
    lower = header_line.lower()
    return {
        "has_datetime": any(k in lower for k in ["data", "hora"]),
        "has_precipitation": any(k in lower for k in ["precipita", "chuva"]),
        "has_temperature": "temperatura" in lower or "temp" in lower,
        "has_quality_flags": "qualidade" in lower or "flag" in lower,
    }


def peek_header(zf: zipfile.ZipFile, internal_name: str) -> str:
    try:
        with zf.open(internal_name) as fh:
            raw = fh.read(4096)
        for enc in ("latin-1", "utf-8", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("latin-1", errors="replace")
        # INMET files have a metadata block (REGIAO/UF/DATA DE FUNDACAO...) before the
        # real column header. The column header has a date field AND (precip OR hora);
        # a bare "DATA DE FUNDACAO" line must not be mistaken for it.
        lines = text.splitlines()
        for line in lines[:15]:
            low = line.lower()
            if "data" in low and ("precipita" in low or "hora" in low):
                return line
        return lines[0] if lines else ""
    except Exception:
        return ""


def safe_extract_member(zf: zipfile.ZipFile, internal_name: str, dest_dir: str) -> str:
    normalized = os.path.normpath(internal_name)
    if normalized.startswith("..") or os.path.isabs(normalized):
        return ""
    os.makedirs(dest_dir, exist_ok=True)
    flat_name = os.path.basename(internal_name)
    dest_path = os.path.join(dest_dir, flat_name)
    with zf.open(internal_name) as src, open(dest_path, "wb") as out:
        out.write(src.read())
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="v1uf — INMET ZIP Selective Extractor")
    parser.add_argument("--manifest", default="datasets/protocolo_c/v1uf_large_download_manifest.csv")
    parser.add_argument("--binding", default="configs/protocolo_c/v1uf_station_target_binding.yaml")
    parser.add_argument("--stations", default="datasets/protocolo_c/v1ue_station_candidate_registry.csv")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c")
    args = parser.parse_args()

    manifest = load_csv(args.manifest)
    binding_config = load_yaml(args.binding)
    stations = load_csv(args.stations)

    bindings = {b["event_id"]: b for b in binding_config.get("bindings", [])}

    # station_candidate_id lookup by (source, event)
    station_id_map = {}
    for s in stations:
        key = f"{s.get('source_id', '')}_{s.get('event_id', '')}"
        station_id_map.setdefault(key, s.get("station_candidate_id", ""))

    rows = []
    seq = 0
    for entry in manifest:
        event_id = entry["event_id"]
        source_id = entry["source_id"]
        zip_local = entry.get("zip_local_path", "")
        zip_sha = entry.get("zip_sha256", "")
        binding = bindings.get(event_id, {})
        year = binding.get("year", "")
        codes = binding.get("target_station_codes", [])
        name_hints = binding.get("target_station_name_hints", [])
        uf = binding.get("uf", "")
        station_id = station_id_map.get(f"{source_id}_{event_id}", "")

        if entry.get("download_status") not in ("DOWNLOADED", "DOWNLOADED_CACHED") or not zip_local:
            rows.append(_asset_row(seq, event_id, source_id, year, station_id,
                                   codes, name_hints, zip_sha, "", "",
                                   "ZIP_NOT_AVAILABLE",
                                   f"download_status={entry.get('download_status', '')}"))
            seq += 1
            continue

        if not os.path.exists(zip_local):
            rows.append(_asset_row(seq, event_id, source_id, year, station_id,
                                   codes, name_hints, zip_sha, "", "",
                                   "ZIP_FILE_MISSING", f"Path not found: {zip_local}"))
            seq += 1
            continue

        try:
            with zipfile.ZipFile(zip_local, "r") as zf:
                names = zf.namelist()
                matched = [n for n in names if not n.endswith("/") and
                           matches_station(n, codes, name_hints, uf, year)]

                if not matched:
                    rows.append(_asset_row(seq, event_id, source_id, year, station_id,
                                           codes, name_hints, zip_sha, "", "",
                                           "NO_STATION_MATCH",
                                           f"{len(names)} files in ZIP, 0 matched station"))
                    seq += 1
                    continue

                staging = os.path.join(args.local_only_dir, "evidence_staging", "v1uf", "inmet", event_id)
                for internal in matched:
                    header = peek_header(zf, internal)
                    cols = detect_columns(header)
                    extracted = safe_extract_member(zf, internal, staging)
                    if not extracted:
                        rows.append(_asset_row(seq, event_id, source_id, year, station_id,
                                               codes, name_hints, zip_sha, internal, "",
                                               "UNSAFE_PATH_SKIPPED", "Path traversal blocked"))
                        seq += 1
                        continue
                    fsha = sha256_file(extracted)
                    fsize = os.path.getsize(extracted)
                    # station code/name from internal filename
                    st_code = next((c for c in codes if c.upper() in internal.upper()), "")
                    rows.append({
                        "asset_id": f"AST_{PROTOCOL_VERSION}_{seq:04d}",
                        "event_id": event_id,
                        "source_id": source_id,
                        "year": year,
                        "station_candidate_id": station_id,
                        "station_code": st_code,
                        "station_name": os.path.basename(internal)[:120],
                        "zip_sha256": zip_sha,
                        "internal_zip_path": internal,
                        "extracted_local_path_hash": hash_str(rel_path(extracted)),
                        "file_sha256": fsha,
                        "file_size_bytes": str(fsize),
                        "extraction_status": "EXTRACTED",
                        "reason": "Matched target station",
                        "has_datetime_column": str(cols["has_datetime"]).lower(),
                        "has_precipitation_column": str(cols["has_precipitation"]).lower(),
                        "has_temperature_column": str(cols["has_temperature"]).lower(),
                        "has_quality_flags": str(cols["has_quality_flags"]).lower(),
                        "notes": "",
                    })
                    seq += 1
        except zipfile.BadZipFile:
            rows.append(_asset_row(seq, event_id, source_id, year, station_id,
                                   codes, name_hints, zip_sha, "", "",
                                   "ZIP_CORRUPT", "BadZipFile"))
            seq += 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "v1uf_station_series_asset_registry.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ASSET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for r in rows:
        statuses[r["extraction_status"]] = statuses.get(r["extraction_status"], 0) + 1
    print(f"[INMET ZIP Selective Extractor v1uf] {len(rows)} asset records")
    for s, c in sorted(statuses.items()):
        print(f"  {s}: {c}")
    print(f"\nRegistry: {out_path}")


def _asset_row(seq, event_id, source_id, year, station_id, codes, name_hints,
               zip_sha, internal, fsha, status, reason):
    return {
        "asset_id": f"AST_{PROTOCOL_VERSION}_{seq:04d}",
        "event_id": event_id,
        "source_id": source_id,
        "year": year,
        "station_candidate_id": station_id,
        "station_code": codes[0] if codes else "",
        "station_name": "",
        "zip_sha256": zip_sha,
        "internal_zip_path": internal,
        "extracted_local_path_hash": "",
        "file_sha256": fsha,
        "file_size_bytes": "",
        "extraction_status": status,
        "reason": reason,
        "has_datetime_column": "false",
        "has_precipitation_column": "false",
        "has_temperature_column": "false",
        "has_quality_flags": "false",
        "notes": "",
    }


if __name__ == "__main__":
    main()
