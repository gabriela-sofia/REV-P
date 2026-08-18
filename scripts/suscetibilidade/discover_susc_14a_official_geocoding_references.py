"""
SUSC-14A FASE 2 - discover official spatial-reference sources per city.

Builds a deterministic registry of official/traceable spatial-reference
endpoints (street axes, neighborhoods, address points, critical flood points,
drainage) for Recife, Curitiba and Petropolis, and also registers
already-acquired local official files (SUSC-13C raw drop) that can serve as
geocoding references. With SUSC_13B_NETWORK=1 each endpoint is additionally
probed live; offline every probe records ``network_disabled`` and nothing is
fabricated.

Writes:
  - manifests/suscetibilidade/susc_14a_official_geocoding_reference_registry_v1.csv
  - outputs_public/suscetibilidade/SUSC_14A_official_reference_discovery_report.md
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import rel, write_csv, write_markdown  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_14a_official_geocoding_reference_registry_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_official_reference_discovery_report.md"

FIELDS = [
    "reference_id", "region", "institution", "source_name", "source_url",
    "source_type", "format", "access_method", "contains_street_geometry",
    "contains_address_points", "contains_neighborhood_geometry",
    "contains_flood_points", "contains_flood_polygons", "contains_drainage",
    "download_candidate", "priority", "review_only", "notes",
]

# local files classifiable as reference layers (address/street/neighborhood/flood)
LOCAL_REFERENCE_HINTS = {
    "coordenada": dict(address=1, street=1, neighborhood=1),
    "logradouro": dict(street=1, address=1),
    "bairro": dict(neighborhood=1),
    "endereco": dict(address=1, street=1),
    "ponto": dict(flood_points=1),
    "alagamento": dict(flood_points=1),
    "drenagem": dict(drainage=1),
    "diamante_brasil": dict(neighborhood=1, address=1),
}
LOCAL_REFERENCE_EXTS = {".geojson", ".json", ".kml", ".kmz", ".gpkg", ".csv"}


def _b(flag: dict, key: str) -> str:
    return "true" if flag.get(key) else "false"


def _endpoint_rows() -> list[dict]:
    rows: list[dict] = []
    net = g14.network_enabled()
    for region, inst, name, url, stype, method, flag in g14.REFERENCE_ENDPOINTS:
        has_geom = any(flag.get(k) for k in ("street", "address", "neighborhood",
                                             "flood_points", "flood_polygons", "drainage"))
        # priority: occurrence-grade address/street geometry on official domain ranks high
        if flag.get("flood_polygons"):
            prio = "high"
        elif flag.get("address") or flag.get("street"):
            prio = "high"
        elif flag.get("flood_points"):
            prio = "medium"
        else:
            prio = "medium_low"
        # live probe is opt-in; offline => network_disabled, no invented resource
        probe_note = ""
        if net and method in ("ckan_api", "arcgis_rest", "wfs_getcapabilities"):
            res = g14.http_get(url if "?" not in url else url, accept="application/json")
            probe_note = f"probe={res.get('status','')}"
        else:
            probe_note = "probe=network_disabled" if not net else "probe=html_root_not_fetched"
        rows.append({
            "region": region, "institution": inst, "source_name": name,
            "source_url": url, "source_type": stype, "format": method,
            "access_method": method,
            "contains_street_geometry": _b(flag, "street"),
            "contains_address_points": _b(flag, "address"),
            "contains_neighborhood_geometry": _b(flag, "neighborhood"),
            "contains_flood_points": _b(flag, "flood_points"),
            "contains_flood_polygons": _b(flag, "flood_polygons"),
            "contains_drainage": _b(flag, "drainage"),
            "download_candidate": "true" if (has_geom and method in ("ckan_api", "arcgis_rest", "wfs_getcapabilities")) else "false",
            "priority": prio, "review_only": "true",
            "notes": f"endpoint institucional oficial; nenhum URL de download inventado; {probe_note}",
        })
    return rows


def _local_rows() -> list[dict]:
    rows: list[dict] = []
    base = g14.LOCAL_OFFICIAL_DIR
    if not base.exists():
        return rows
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in LOCAL_REFERENCE_EXTS:
            continue
        low = g14.normalize_text(p.name)
        flag: dict = {}
        for hint, caps in LOCAL_REFERENCE_HINTS.items():
            if hint in low:
                flag.update(caps)
        if not flag:
            continue
        rows.append({
            "region": "recife" if "recife" in low or "regiao" in low else "unknown",
            "institution": "Prefeitura/Dados Abertos Recife (local 13C)",
            "source_name": f"Camada local reutilizada: {p.name}",
            "source_url": "", "source_type": "official", "format": p.suffix.lower().lstrip("."),
            "access_method": "reused_local_official",
            "contains_street_geometry": _b(flag, "street"),
            "contains_address_points": _b(flag, "address"),
            "contains_neighborhood_geometry": _b(flag, "neighborhood"),
            "contains_flood_points": _b(flag, "flood_points"),
            "contains_flood_polygons": _b(flag, "flood_polygons"),
            "contains_drainage": _b(flag, "drainage"),
            "download_candidate": "false",
            "priority": "high" if flag.get("address") else "medium",
            "review_only": "true",
            "notes": f"arquivo oficial ja adquirido localmente (13C); caminho={rel(p)}",
        })
    return rows


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Official Geocoding Reference Discovery")
    print("=" * 60)
    print(f"network enabled: {g14.network_enabled()} (set {g14.NETWORK_ENV}=1 to probe live)")
    rows = _endpoint_rows() + _local_rows()
    for i, r in enumerate(rows):
        r["reference_id"] = f"S14AREF_{i:04d}"
    rows = [{k: r.get(k, "") for k in FIELDS} for r in rows]
    write_csv(REGISTRY, rows, FIELDS)

    by_region = Counter(r["region"] for r in rows)
    addr = sum(1 for r in rows if r["contains_address_points"] == "true")
    street = sum(1 for r in rows if r["contains_street_geometry"] == "true")
    flood = sum(1 for r in rows if r["contains_flood_points"] == "true" or r["contains_flood_polygons"] == "true")
    local = sum(1 for r in rows if r["access_method"] == "reused_local_official")
    dl_cand = sum(1 for r in rows if r["download_candidate"] == "true")

    def tbl(c):
        return "\n".join(f"| {k} | {v} |" for k, v in sorted(c.items())) or "| nenhum | 0 |"

    md = f"""# SUSC-14A - Descoberta de referencias espaciais oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Objetivo
Localizar fontes oficiais/rastreaveis capazes de fornecer geometria de referencia
(logradouros, eixos de via, bairros, enderecos, pontos criticos de alagamento,
drenagem, mapas de risco/relatorios) para tentar resgatar o vinculo espacial de
ocorrencias oficiais de cheia registradas sem lat/lon (gargalo do SUSC-13C).

## 2. Politica de rede
- Ativacao live: `SUSC_13B_NETWORK=1`; rede nesta execucao: **{"Sim" if g14.network_enabled() else "Nao"}**.
- Sem Google Maps, sem geocoding generico, sem chave de API, sem raster.
- OSM/Nominatim nao e consultado; apenas camadas oficiais/rastreaveis.

## 3. Referencias registradas
Total: **{len(rows)}** (endpoints institucionais + camadas locais ja adquiridas).

| regiao | referencias |
|---|---|
{tbl(by_region)}

- Com pontos de endereco: **{addr}**
- Com eixos/logradouros: **{street}**
- Com pontos/poligonos de cheia: **{flood}**
- Camadas locais reutilizadas (13C): **{local}**
- Candidatas a download live: **{dl_cand}**

## 4. Limites e governanca
- Endpoints sao raizes institucionais; nenhum URL de download direto e inventado.
- Centroide de municipio/bairro nunca vira evento patch-level.
- Mapas de risco/suscetibilidade nunca viram ocorrencia observada.
- Tudo permanece review-only; nada cria ground truth, treino ou score v7.
"""
    write_markdown(REPORT, md)
    print(f"registry -> {rel(REGISTRY)} ({len(rows)} refs; {local} locais; {dl_cand} download-cand)")
    print("review-only. No invented direct URL. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
