"""
SUSC-13C-LIVE real ArcGIS REST probe.

Walks ArcGIS REST roots (?f=pjson), lists folders/services, detects Map/Feature
servers, inspects layers, and flags flood/inundation/Defesa-Civil layers. For a
candidate layer it attempts a bounded GeoJSON query and saves the raw response
(<=250MB, no raster). Requires SUSC_13B_NETWORK=1. Never invents data.

Writes:
  - outputs_public/suscetibilidade/SUSC_13C_live_arcgis_layers.csv
  - datasets/suscetibilidade/observed_event_sources_susc13c_live/arcgis/  (raw geojson)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import ensure_dir, rel, write_csv  # noqa: E402

LAYERS_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_live_arcgis_layers.csv"
RAW_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13c_live" / "arcgis"

ARCGIS_ROOTS = [
    ("recife", "https://www.recife.pe.gov.br/arcgis/rest/services"),
    ("recife", "https://geoportal.sgb.gov.br/server/rest/services"),
    ("petropolis", "https://inea.maps.arcgis.com/sharing/rest"),
    ("curitiba", "https://geocuritiba.ippuc.org.br/server/rest/services"),
]

FIELDS = [
    "arcgis_root", "region", "service_name", "service_type", "layer_id",
    "layer_name", "matched_keywords", "service_url", "query_status",
    "geojson_saved", "feature_count_hint", "is_download_candidate",
    "review_only", "notes",
]

MAX_SERVICES = 60
MAX_LAYERS_PER_SERVICE = 40
QUERY_RECORD_LIMIT = 1000


def _pjson(url: str) -> dict | None:
    res = wd.http_get(f"{url}?f=pjson", accept="application/json")
    if res["status"] != "ok":
        return None
    try:
        return json.loads(res["bytes"].decode("utf-8", "ignore"))  # type: ignore[union-attr]
    except Exception:
        return None


def _list_services(root: str) -> tuple[list, str]:
    data = _pjson(root)
    if data is None:
        return [], "root_unreachable"
    services = list(data.get("services", []))
    # one level of folders
    for folder in data.get("folders", [])[:20]:
        fdata = _pjson(f"{root}/{folder}")
        if fdata:
            services.extend(fdata.get("services", []))
        wd.polite_sleep()
    return services[:MAX_SERVICES], "ok"


def main() -> int:
    print("=" * 60)
    print("SUSC-13C-LIVE ArcGIS Probe")
    print("=" * 60)
    rows: list[dict] = []
    root_status: Counter = Counter()
    saved = 0

    for region, root in ARCGIS_ROOTS:
        services, status = _list_services(root)
        root_status[status] += 1
        if status != "ok":
            rows.append({
                "arcgis_root": root, "region": region, "service_name": "", "service_type": "",
                "layer_id": "", "layer_name": "", "matched_keywords": "", "service_url": "",
                "query_status": status, "geojson_saved": "false", "feature_count_hint": "",
                "is_download_candidate": "false", "review_only": "true",
                "notes": "raiz ArcGIS inacessivel",
            })
            print(f"  root {root}: {status}")
            continue
        print(f"  root {root}: {len(services)} services")
        for svc in services:
            name = svc.get("name", "")
            stype = svc.get("type", "MapServer")
            svc_url = f"{root}/{name}/{stype}"
            # service-name keyword OR inspect layers
            svc_data = _pjson(svc_url)
            wd.polite_sleep()
            if not svc_data:
                continue
            layers = (svc_data.get("layers") or [])[:MAX_LAYERS_PER_SERVICE]
            for lyr in layers:
                lname = lyr.get("name", "")
                ltext = f"{name} {lname}"
                matched = wd.matched_keywords(ltext)
                if not matched:
                    continue
                lid = lyr.get("id", "")
                qurl = (f"{svc_url}/{lid}/query?where=1%3D1&outFields=*&returnGeometry=true"
                        f"&resultRecordCount={QUERY_RECORD_LIMIT}&f=geojson")
                qres = wd.http_get(qurl, max_bytes=wd.MAX_BYTES, accept="application/json")
                geojson_saved = "false"
                fcount = ""
                qstatus = qres["status"]
                if qres["status"] == "ok":
                    body = qres["bytes"]
                    try:
                        gj = json.loads(body.decode("utf-8", "ignore"))  # type: ignore[union-attr]
                        feats = gj.get("features", []) if isinstance(gj, dict) else []
                        fcount = str(len(feats))
                        if feats:
                            ensure_dir(RAW_DIR)
                            safe = f"{region}_{name.replace('/','_')}_{stype}_{lid}.geojson".replace(" ", "_")
                            out = RAW_DIR / safe[:180]
                            out.write_bytes(body)  # type: ignore[arg-type]
                            geojson_saved = "true"
                            saved += 1
                    except Exception:
                        qstatus = "parse_error"
                rows.append({
                    "arcgis_root": root, "region": region, "service_name": name,
                    "service_type": stype, "layer_id": str(lid), "layer_name": lname,
                    "matched_keywords": "|".join(matched), "service_url": qurl,
                    "query_status": qstatus, "geojson_saved": geojson_saved,
                    "feature_count_hint": fcount,
                    "is_download_candidate": "true" if geojson_saved == "true" else "false",
                    "review_only": "true",
                    "notes": "layer ArcGIS com keyword; geometria revisada review-only" if geojson_saved == "true" else "layer matched; sem GeoJSON salvo",
                })
                wd.polite_sleep()

    write_csv(LAYERS_CSV, rows, FIELDS)
    cands = sum(1 for r in rows if r["is_download_candidate"] == "true")
    print(f"layers -> {rel(LAYERS_CSV)} ({len(rows)} rows, {cands} candidates, {saved} geojson saved)")
    print("root status:", dict(root_status))
    print("review-only. No raster. No invented data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
