#!/usr/bin/env python3
"""
v1uh — Event Field Mapper

Maps received fields to canonical Protocol C fields.
Does not geocode, does not convert coordinates without CRS,
does not infer hazard if field is ambiguous.
"""

import argparse
import csv
import hashlib
import os

PROTOCOL_VERSION = "v1uh"

MAPPING_COLUMNS = [
    "mapping_id", "candidate_id", "event_id", "asset_id",
    "canonical_field", "source_field", "source_field_confidence",
    "value_sample_hash", "mapping_status", "ambiguity_reason",
    "requires_human_review", "notes",
]

CANONICAL_FIELDS = [
    "event_date", "event_time", "hazard_type", "phenomenon",
    "locality", "address", "municipality", "uf",
    "latitude", "longitude", "x", "y",
    "geometry", "crs", "source", "confidence", "observation_type",
]

FIELD_SYNONYMS = {
    "event_date": [
        "data", "Data", "DATA", "dt_ocorrencia", "data_ocorrencia",
        "timestamp", "date", "data_evento", "dt_evento", "data_registro",
    ],
    "event_time": [
        "hora", "time", "horario", "hr_ocorrencia",
    ],
    "hazard_type": [
        "tipo", "fenomeno", "hazard", "classe", "ocorrencia", "tipo_evento",
        "type", "hazard_type", "event_type", "natureza",
    ],
    "phenomenon": [
        "fenomeno", "phenomenon", "tipo_fenomeno", "descricao_fenomeno",
    ],
    "locality": [
        "bairro", "localidade", "neighborhood", "locality",
        "setor", "area", "regiao",
    ],
    "address": [
        "rua", "endereco", "logradouro", "address", "street",
    ],
    "municipality": [
        "municipio", "cidade", "city", "municipality", "mun",
    ],
    "uf": [
        "uf", "estado", "state", "UF",
    ],
    "latitude": [
        "latitude", "lat", "Latitude", "LAT", "y_coord",
    ],
    "longitude": [
        "longitude", "lon", "lng", "Longitude", "LON", "LNG", "x_coord",
    ],
    "x": ["x", "X", "coord_x", "easting"],
    "y": ["y", "Y", "coord_y", "northing"],
    "geometry": ["geometry", "geom", "wkt", "the_geom", "GEOMETRY"],
    "crs": ["crs", "CRS", "srid", "SRID", "epsg", "projection"],
    "source": ["fonte", "origem", "instituicao", "source", "provider"],
    "confidence": ["confianca", "confidence", "qualidade", "quality"],
    "observation_type": [
        "tipo_observacao", "observation_type", "obs_type", "metodo",
    ],
}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hash_value(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()[:12]


def map_fields(_candidate: dict, asset: dict) -> list[dict]:
    columns_str = asset.get("columns_detected", "")
    if not columns_str:
        return []

    source_columns = [c.strip() for c in columns_str.split("|") if c.strip()]
    source_lower = {c.lower(): c for c in source_columns}

    mappings = []
    for canonical, synonyms in FIELD_SYNONYMS.items():
        matched = None
        confidence = "NONE"
        ambiguity = ""

        exact_matches = []
        seen = set()
        for syn in synonyms:
            key = syn.lower()
            if key in source_lower and source_lower[key] not in seen:
                exact_matches.append(source_lower[key])
                seen.add(source_lower[key])

        if len(exact_matches) == 1:
            matched = exact_matches[0]
            confidence = "HIGH"
        elif len(exact_matches) > 1:
            matched = exact_matches[0]
            confidence = "AMBIGUOUS"
            ambiguity = f"Multiple matches: {', '.join(exact_matches)}"

        if matched:
            mappings.append({
                "canonical_field": canonical,
                "source_field": matched,
                "source_field_confidence": confidence,
                "value_sample_hash": "",
                "mapping_status": "MAPPED" if confidence == "HIGH" else "NEEDS_REVIEW",
                "ambiguity_reason": ambiguity,
                "requires_human_review": str(confidence != "HIGH").lower(),
            })

    return mappings


def main():
    parser = argparse.ArgumentParser(description="v1uh — Event Field Mapper")
    parser.add_argument("--candidates",
                        default="datasets/protocolo_c/v1uh_observed_geometry_candidate_registry.csv")
    parser.add_argument("--assets",
                        default="datasets/protocolo_c/v1uh_response_asset_inventory.csv")
    parser.add_argument("--out",
                        default="datasets/protocolo_c/v1uh_event_field_mapping_registry.csv")
    args = parser.parse_args()

    candidates = load_csv(args.candidates)
    assets_by_id = {a["asset_id"]: a for a in load_csv(args.assets)}

    rows = []
    seq = 0
    for cand in candidates:
        asset = assets_by_id.get(cand.get("asset_id", ""), {})
        mappings = map_fields(cand, asset)
        for m in mappings:
            rows.append({
                "mapping_id": f"MAP_{PROTOCOL_VERSION}_{seq:04d}",
                "candidate_id": cand.get("candidate_id", ""),
                "event_id": cand.get("event_id", ""),
                "asset_id": cand.get("asset_id", ""),
                **m,
                "notes": "",
            })
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Event Field Mapper v1uh] {len(rows)} field mappings")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
