"""
SUSC-13C-LIVE real WFS/GeoServer probe.

Calls GetCapabilities on WFS endpoints, lists FeatureTypes, filters flood/
inundation/occurrence layers, and attempts a bounded GetFeature (GeoJSON) saving
small raw responses. Requires SUSC_13B_NETWORK=1. Never invents data.

Writes:
  - outputs_public/suscetibilidade/SUSC_13C_live_wfs_layers.csv
  - datasets/suscetibilidade/observed_event_sources_susc13c_live/wfs/  (raw geojson/gml)
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import ensure_dir, rel, write_csv  # noqa: E402

LAYERS_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_live_wfs_layers.csv"
RAW_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13c_live" / "wfs"

WFS_ENDPOINTS = [
    ("curitiba", "https://geoportal.iat.pr.gov.br/geoserver/ows"),
    ("curitiba", "https://geocuritiba.ippuc.org.br/geoserver/ows"),
    ("petropolis", "https://geoserver.inea.rj.gov.br/geoserver/ows"),
]

FIELDS = [
    "wfs_endpoint", "region", "feature_type", "matched_keywords",
    "getcapabilities_status", "getfeature_status", "geojson_saved",
    "is_download_candidate", "review_only", "notes",
]

MAX_FEATURETYPES = 60
GETFEATURE_COUNT = 500


def _capabilities(endpoint: str) -> tuple[list[str], str]:
    url = f"{endpoint}?service=WFS&version=2.0.0&request=GetCapabilities"
    res = wd.http_get(url, max_bytes=8 * 1024 * 1024, accept="application/xml")
    if res["status"] != "ok":
        return [], res["status"]
    try:
        text = res["bytes"].decode("utf-8", "ignore")  # type: ignore[union-attr]
    except Exception:
        return [], "decode_error"
    names = re.findall(r"<(?:[\w]+:)?Name>([^<]+)</(?:[\w]+:)?Name>", text)
    return names[:MAX_FEATURETYPES], "ok"


def main() -> int:
    print("=" * 60)
    print("SUSC-13C-LIVE WFS Probe")
    print("=" * 60)
    rows: list[dict] = []
    ep_status: Counter = Counter()
    saved = 0

    for region, endpoint in WFS_ENDPOINTS:
        names, status = _capabilities(endpoint)
        ep_status[status] += 1
        if status != "ok":
            rows.append({
                "wfs_endpoint": endpoint, "region": region, "feature_type": "",
                "matched_keywords": "", "getcapabilities_status": status,
                "getfeature_status": "", "geojson_saved": "false",
                "is_download_candidate": "false", "review_only": "true",
                "notes": "GetCapabilities falhou",
            })
            print(f"  endpoint {endpoint}: {status}")
            continue
        matched_types = [n for n in names if wd.matched_keywords(n)]
        print(f"  endpoint {endpoint}: {len(names)} feature types, {len(matched_types)} matched")
        if not matched_types:
            rows.append({
                "wfs_endpoint": endpoint, "region": region, "feature_type": "",
                "matched_keywords": "", "getcapabilities_status": "ok",
                "getfeature_status": "no_matching_featuretype", "geojson_saved": "false",
                "is_download_candidate": "false", "review_only": "true",
                "notes": f"{len(names)} feature types, nenhum com keyword de alagamento/inundacao",
            })
            continue
        for ft in matched_types:
            gurl = (f"{endpoint}?service=WFS&version=2.0.0&request=GetFeature&typeNames={ft}"
                    f"&outputFormat=application/json&count={GETFEATURE_COUNT}")
            gres = wd.http_get(gurl, max_bytes=wd.MAX_BYTES, accept="application/json")
            geojson_saved = "false"
            if gres["status"] == "ok" and gres["bytes"]:
                ensure_dir(RAW_DIR)
                safe = f"{region}_{ft.replace(':','_').replace('/','_')}.geojson"
                out = RAW_DIR / safe[:180]
                out.write_bytes(gres["bytes"])  # type: ignore[arg-type]
                geojson_saved = "true"
                saved += 1
            rows.append({
                "wfs_endpoint": endpoint, "region": region, "feature_type": ft,
                "matched_keywords": "|".join(wd.matched_keywords(ft)),
                "getcapabilities_status": "ok", "getfeature_status": gres["status"],
                "geojson_saved": geojson_saved,
                "is_download_candidate": "true" if geojson_saved == "true" else "false",
                "review_only": "true",
                "notes": "FeatureType com keyword; GetFeature review-only" if geojson_saved == "true" else "matched mas GetFeature nao salvou",
            })
            wd.polite_sleep()

    write_csv(LAYERS_CSV, rows, FIELDS)
    cands = sum(1 for r in rows if r["is_download_candidate"] == "true")
    print(f"layers -> {rel(LAYERS_CSV)} ({len(rows)} rows, {cands} candidates, {saved} saved)")
    print("endpoint status:", dict(ep_status))
    print("review-only. No invented data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
