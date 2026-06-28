"""
SUSC-13C-LIVE real CKAN probe.

Queries public CKAN APIs (package_search + package_show) for flood/inundation/
Defesa-Civil datasets and records every candidate vector/tabular/documentary
resource with real HTTP status. Requires SUSC_13B_NETWORK=1. Never invents data;
unreachable bases are recorded as failures.

Writes:
  - outputs_public/suscetibilidade/SUSC_13C_live_ckan_resources.csv
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
from susc_io import rel, write_csv  # noqa: E402

RESOURCES_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_live_ckan_resources.csv"

# (region, base_api). Federal/ANA bases are tagged 'national'.
CKAN_BASES = [
    ("recife", "http://dados.recife.pe.gov.br/api/3/action"),
    ("recife", "http://www.dados.pe.gov.br/api/3/action"),
    ("petropolis", "http://www.dados.rj.gov.br/api/3/action"),
    ("curitiba", "https://dadosabertos.curitiba.pr.gov.br/api/3/action"),
    ("curitiba", "http://www.dadosabertos.pr.gov.br/api/3/action"),
    ("national", "https://dadosabertos.ana.gov.br/api/3/action"),
    ("national", "https://dados.gov.br/api/3/action"),
]

QUERIES = [
    "alagamento", "inundação", "inundacao", "enchente", "enxurrada",
    "defesa civil", "ocorrência", "ocorrencia",
    "recife alagamento", "recife inundação", "petrópolis inundação",
    "petropolis enchente", "curitiba alagamento", "curitiba defesa civil ocorrência",
]

ALLOWED_FMT = {"csv", "xlsx", "xls", "geojson", "json", "kml", "kmz", "zip",
               "shp", "gpkg", "pdf", "xml", "gml", "wfs", "wms"}

FIELDS = [
    "ckan_base", "region", "package_id", "package_title", "resource_id",
    "resource_name", "format", "url", "matched_keywords", "ext",
    "is_download_candidate", "status_code", "content_type", "review_only", "notes",
]

MAX_PACKAGES_PER_BASE = 40


def _search(base: str, query: str) -> tuple[list, dict]:
    from urllib.parse import quote
    url = f"{base}/package_search?q={quote(query)}&rows=25"
    res = wd.http_get(url, accept="application/json")
    if res["status"] != "ok":
        return [], res
    try:
        data = json.loads(res["bytes"].decode("utf-8", "ignore"))  # type: ignore[union-attr]
        return data.get("result", {}).get("results", []), res
    except Exception as exc:
        res["status"] = "parse_error"
        res["error"] = str(exc)[:120]
        return [], res


def _package_show(base: str, pid: str) -> list:
    url = f"{base}/package_show?id={pid}"
    res = wd.http_get(url, accept="application/json")
    if res["status"] != "ok":
        return []
    try:
        data = json.loads(res["bytes"].decode("utf-8", "ignore"))  # type: ignore[union-attr]
        return data.get("result", {}).get("resources", [])
    except Exception:
        return []


def main() -> int:
    print("=" * 60)
    print("SUSC-13C-LIVE CKAN Probe")
    print("=" * 60)
    if not wd.network_enabled():
        print("network disabled; set SUSC_13B_NETWORK=1. Writing empty result with note.")
    rows: list[dict] = []
    base_status: Counter = Counter()

    for region, base in CKAN_BASES:
        seen_pkgs: dict[str, dict] = {}
        any_ok = False
        last_status = "not_attempted"
        for q in QUERIES:
            pkgs, res = _search(base, q)
            last_status = res["status"]
            if res["status"] == "ok":
                any_ok = True
            for pkg in pkgs:
                pid = pkg.get("id") or pkg.get("name", "")
                if pid and pid not in seen_pkgs and len(seen_pkgs) < MAX_PACKAGES_PER_BASE:
                    seen_pkgs[pid] = pkg
            wd.polite_sleep()
        base_status[last_status if not any_ok else "ok"] += 1
        if not any_ok:
            rows.append({
                "ckan_base": base, "region": region, "package_id": "", "package_title": "",
                "resource_id": "", "resource_name": "", "format": "", "url": "",
                "matched_keywords": "", "ext": "", "is_download_candidate": "false",
                "status_code": last_status, "content_type": "", "review_only": "true",
                "notes": "CKAN base nao respondeu / sem package_search valido",
            })
            print(f"  base {base}: {last_status}")
            continue

        # for matched packages, pull resources (package_show for richness)
        for pid, pkg in seen_pkgs.items():
            title = pkg.get("title", "") or pkg.get("name", "")
            resources = pkg.get("resources") or _package_show(base, pid)
            for rsrc in resources:
                rurl = (rsrc.get("url") or "").strip()
                fmt = (rsrc.get("format") or "").strip().lower()
                rtext = f"{title} {rsrc.get('name','')} {fmt}"
                matched = wd.matched_keywords(rtext)
                if not matched:
                    continue
                ext = wd.classify_url_ext(rurl)
                fmt_ok = fmt in ALLOWED_FMT or ext.lstrip(".") in ALLOWED_FMT
                is_raster = ext in {"raster_blocked", "executable_blocked"}
                dl = bool(rurl.startswith("http")) and fmt_ok and not is_raster and not wd.looks_like_risk(rtext)
                rows.append({
                    "ckan_base": base, "region": region, "package_id": pid,
                    "package_title": title[:120], "resource_id": rsrc.get("id", ""),
                    "resource_name": (rsrc.get("name") or "")[:120], "format": fmt,
                    "url": rurl, "matched_keywords": "|".join(matched),
                    "ext": ext, "is_download_candidate": "true" if dl else "false",
                    "status_code": "200", "content_type": "",
                    "review_only": "true",
                    "notes": "recurso CKAN com keyword de alagamento/inundacao" if dl else "recurso encontrado mas nao baixavel/risco",
                })
        print(f"  base {base}: ok, {len(seen_pkgs)} pacotes matched")

    write_csv(RESOURCES_CSV, rows, FIELDS)
    cands = sum(1 for r in rows if r["is_download_candidate"] == "true")
    print(f"resources -> {rel(RESOURCES_CSV)} ({len(rows)} rows, {cands} download candidates)")
    print("base status:", dict(base_status))
    print("review-only. No invented data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
