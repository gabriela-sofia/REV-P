#!/usr/bin/env python3
"""
v1uj — GeoSGB ArcGIS REST Resolver

A v1ui falhou em path generico GeoSGB. Aqui testamos bases ArcGIS REST
reais/provaveis, consultamos ?f=pjson, listamos services e layers e
registramos metadata (geometry type, fields, extent, spatialReference)
SEM baixar features massivamente.

Modelagem/suscetibilidade e classificada como CONTEXTO, nunca ocorrencia
observada. Sem login, sem bypass. DRY_RUN sem --allow-web.
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

PROTOCOL_VERSION = "v1uj"

GEOSGB_COLUMNS = [
    "geosgb_record_id", "event_id", "service_url", "layer_id", "layer_name",
    "geometry_type", "spatial_reference", "fields", "extent",
    "relevance_score", "is_event_specific", "is_observed_occurrence_candidate",
    "is_contextual_layer", "blocking_reason",
]


def load_yaml(path):
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fetch_pjson(url, timeout=30):
    if not HAS_URLLIB:
        return None
    try:
        sep = "&" if "?" in url else "?"
        full = f"{url}{sep}f=pjson"
        req = urllib.request.Request(
            full, headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read(1_000_000).decode("utf-8"))
    except Exception:
        return None


def classify_layer(layer_name, cfg, event_terms_by_event):
    """Pontua e classifica uma layer. Retorna dict. Funcao pura."""
    lower = (layer_name or "").lower()
    rel = cfg.get("relevance_terms", {})
    observed_terms = [t.lower() for t in rel.get("observed_occurrence", [])]
    context_terms = [t.lower() for t in rel.get("contextual_only", [])]
    inst_terms = [t.lower() for t in cfg.get("institution_terms", [])]

    is_contextual = any(t in lower for t in context_terms)
    is_observed = (not is_contextual) and any(t in lower for t in observed_terms)

    score = 0
    if is_observed:
        score += 20
    if is_contextual:
        score += 5
    if any(t in lower for t in inst_terms):
        score += 5

    matched_event = ""
    for ev_id, terms in event_terms_by_event.items():
        if any(str(t).lower() in lower for t in terms):
            matched_event = ev_id
            score += 10
            break

    return {
        "is_observed": is_observed,
        "is_contextual": is_contextual,
        "score": score,
        "event_id": matched_event,
    }


def extract_layer_meta(layer):
    lid = str(layer.get("id", ""))
    lname = layer.get("name", "")
    gtype = layer.get("geometryType", "")
    extent = layer.get("extent", {})
    sr = ""
    if isinstance(extent, dict):
        sr_info = extent.get("spatialReference", {})
        sr = str(sr_info.get("wkid", "")) if sr_info else ""
    fields = [f.get("name", "") for f in layer.get("fields", [])[:25]]
    return lid, lname, gtype, sr, "|".join(fields), json.dumps(extent)[:200] if extent else ""


def main():
    parser = argparse.ArgumentParser(description="v1uj — GeoSGB ArcGIS REST Resolver")
    parser.add_argument("--config", default="configs/protocolo_c/v1uj_geosgb_service_targets.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_geosgb_layer_registry.csv")
    parser.add_argument("--services-fixture", default="",
                        help="JSON local de services (testes offline)")
    parser.add_argument("--layers-fixture", default="",
                        help="JSON local de layers (testes offline)")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    bases = cfg.get("rest_base_urls", [])
    suffixes = cfg.get("service_suffixes", ["MapServer", "FeatureServer"])
    event_terms = cfg.get("event_terms", {})
    max_services = cfg.get("max_services_listed", 80)
    max_layers = cfg.get("max_layers_per_service", 60)

    services_fixture = None
    if args.services_fixture and os.path.exists(args.services_fixture):
        with open(args.services_fixture, "r", encoding="utf-8") as f:
            services_fixture = json.load(f)
    layers_fixture = None
    if args.layers_fixture and os.path.exists(args.layers_fixture):
        with open(args.layers_fixture, "r", encoding="utf-8") as f:
            layers_fixture = json.load(f)

    rows = []
    seq = 0

    for base in bases:
        services_doc = services_fixture
        if services_doc is None and args.allow_web:
            services_doc = fetch_pjson(base, args.timeout)

        if not services_doc:
            rows.append({
                "geosgb_record_id": f"GSGB_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": "", "service_url": base, "layer_id": "",
                "layer_name": "", "geometry_type": "", "spatial_reference": "",
                "fields": "", "extent": "", "relevance_score": "0",
                "is_event_specific": "false",
                "is_observed_occurrence_candidate": "false",
                "is_contextual_layer": "false",
                "blocking_reason": "DRY_RUN" if not args.allow_web else "ENDPOINT_UNREACHABLE",
            })
            seq += 1
            continue

        services = services_doc.get("services", [])[:max_services]
        for svc in services:
            sname = svc.get("name", "")
            stype = svc.get("type", "MapServer")
            if stype not in suffixes:
                continue
            service_url = f"{base.rstrip('/')}/{sname}/{stype}"

            layers_doc = layers_fixture
            if layers_doc is None and args.allow_web:
                layers_doc = fetch_pjson(service_url, args.timeout)
            if not layers_doc:
                rows.append({
                    "geosgb_record_id": f"GSGB_{PROTOCOL_VERSION}_{seq:04d}",
                    "event_id": "", "service_url": service_url, "layer_id": "",
                    "layer_name": sname, "geometry_type": "", "spatial_reference": "",
                    "fields": "", "extent": "", "relevance_score": "0",
                    "is_event_specific": "false",
                    "is_observed_occurrence_candidate": "false",
                    "is_contextual_layer": "false",
                    "blocking_reason": "NO_LAYERS",
                })
                seq += 1
                continue

            for layer in layers_doc.get("layers", [])[:max_layers]:
                lid, lname, gtype, sr, fields, extent = extract_layer_meta(layer)
                cl = classify_layer(lname, cfg, event_terms)
                blocking = ""
                if cl["is_contextual"]:
                    blocking = "susceptibility_is_context_not_occurrence"
                rows.append({
                    "geosgb_record_id": f"GSGB_{PROTOCOL_VERSION}_{seq:04d}",
                    "event_id": cl["event_id"], "service_url": service_url,
                    "layer_id": lid, "layer_name": lname,
                    "geometry_type": gtype, "spatial_reference": sr,
                    "fields": fields, "extent": extent,
                    "relevance_score": str(cl["score"]),
                    "is_event_specific": str(bool(cl["event_id"])).lower(),
                    "is_observed_occurrence_candidate": str(cl["is_observed"]).lower(),
                    "is_contextual_layer": str(cl["is_contextual"]).lower(),
                    "blocking_reason": blocking,
                })
                seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GEOSGB_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    observed = sum(1 for r in rows if r["is_observed_occurrence_candidate"] == "true")
    print(f"[GeoSGB REST Resolver v1uj] {len(rows)} layer records | observed_candidates={observed}")
    print(f"  susceptibility_is_not_observed_occurrence=true | download_features=false")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
