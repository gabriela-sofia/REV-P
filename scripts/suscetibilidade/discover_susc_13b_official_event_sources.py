"""
SUSC-13B-AUTO automatic discovery of official flood/inundation event sources.

Builds a deterministic query plan and a registry of candidate official endpoints
(CKAN, ArcGIS REST, WFS/GeoServer, open-data portals). When SUSC_13B_NETWORK=1
each endpoint is probed for flood-relevant resources/layers; otherwise every
probe records ``network_disabled`` and nothing is fabricated.

Writes:
  - manifests/suscetibilidade/susc_13b_auto_discovery_query_manifest_v1.csv
  - manifests/suscetibilidade/susc_13b_auto_discovered_sources_v1.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_discovery_report.md
  - outputs_public/suscetibilidade/SUSC_13B_auto_discovery_debug_log.csv
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import rel, write_csv, write_markdown  # noqa: E402

QUERY_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13b_auto_discovery_query_manifest_v1.csv"
DISCOVERED = ROOT / "manifests" / "suscetibilidade" / "susc_13b_auto_discovered_sources_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_discovery_report.md"
DEBUG_LOG = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_discovery_debug_log.csv"

QUERY_FIELDS = [
    "query_id", "region", "municipality", "query_string", "keyword_template",
    "discovery_method", "target_institution", "target_url", "target_domain",
    "executed", "network_enabled", "status", "review_only", "notes",
]

DISCOVERY_FIELDS = [
    "discovery_id", "region", "municipality", "institution", "source_title",
    "source_url", "source_domain", "discovery_method", "matched_keywords",
    "source_format", "is_direct_download", "is_api_endpoint", "is_official_domain",
    "priority_score", "priority_tier", "expected_geometry", "expected_temporal_fields",
    "download_candidate", "manual_review_reason", "review_only", "notes",
]

DEBUG_FIELDS = [
    "probe_id", "region", "institution", "discovery_method", "url",
    "network_enabled", "status", "http_code", "content_type", "n_resources_found",
    "review_only", "error_or_note",
]


def _build_query_manifest() -> list[dict]:
    rows: list[dict] = []
    net = "true" if wd.network_enabled() else "false"
    for region, meta in wd.REGIONS.items():
        city = meta["municipality"]
        # generic keyword queries (executed against the discovery layer as intent)
        for tmpl in wd.KEYWORD_TEMPLATES:
            q = tmpl.format(city=city)
            method = "data_portal_html"
            if "FeatureServer" in tmpl or "MapServer" in tmpl:
                method = "arcgis_rest"
            elif "WFS" in tmpl or "GetCapabilities" in tmpl:
                method = "wfs_getcapabilities"
            elif "geojson" in tmpl or "csv" in tmpl or "xlsx" in tmpl or "shapefile" in tmpl:
                method = "ckan_api"
            rows.append({
                "region": region, "municipality": city, "query_string": q,
                "keyword_template": tmpl, "discovery_method": method,
                "target_institution": "multiple_official", "target_url": "",
                "target_domain": "", "executed": "false", "network_enabled": net,
                "status": "planned_offline" if net == "false" else "planned",
                "review_only": "true",
                "notes": "consulta de descoberta planejada; execucao live so com SUSC_13B_NETWORK=1",
            })
        # site-scoped intents
        for q in wd.SITE_TEMPLATES.get(region, []):
            rows.append({
                "region": region, "municipality": city, "query_string": q,
                "keyword_template": "site_scoped", "discovery_method": "data_portal_html",
                "target_institution": "official_domain_scope", "target_url": "",
                "target_domain": "", "executed": "false", "network_enabled": net,
                "status": "documented_intent",
                "review_only": "true",
                "notes": "intent documentado; sem scraping aberto/agressivo",
            })
    for i, r in enumerate(rows):
        r["query_id"] = f"S13BQ_{i:04d}"
    return rows


# ---------------------------------------------------------------------------
# Live probes (only run when network is enabled; offline -> network_disabled)
# ---------------------------------------------------------------------------

def _probe_ckan(url: str, region: str) -> tuple[list[dict], dict]:
    """Probe a CKAN package_search endpoint for flood-relevant resources."""
    found: list[dict] = []
    city = wd.REGIONS[region]["municipality"]
    query = f"{url}?q=alagamento%20OR%20inundacao%20OR%20enchente%20{city}&rows=50"
    res = wd.http_get(query, accept="application/json")
    if res["status"] != "ok":
        return found, res
    try:
        data = json.loads(res["bytes"].decode("utf-8", "ignore"))
        packages = data.get("result", {}).get("results", [])
    except Exception as exc:  # malformed JSON
        res["status"] = "parse_error"
        res["error"] = str(exc)[:120]
        return found, res
    for pkg in packages:
        title = pkg.get("title", "") or pkg.get("name", "")
        for rsrc in pkg.get("resources", []):
            rurl = rsrc.get("url", "") or ""
            rtext = f"{title} {rsrc.get('name','')} {rsrc.get('format','')}"
            if not wd.matched_keywords(rtext):
                continue
            ext = wd.classify_url_ext(rurl)
            if ext in {"raster_blocked", "executable_blocked", "unknown_ext"}:
                continue
            found.append({
                "title": f"{title} :: {rsrc.get('name','')}".strip(" :"),
                "url": rurl, "format": rsrc.get("format", "") or ext,
                "matched": rtext, "method": "ckan_api",
                "is_direct_download": "true", "is_api_endpoint": "false",
            })
    return found, res


def _probe_arcgis(url: str, _region: str) -> tuple[list[dict], dict]:
    """List ArcGIS REST services/layers and flag flood-relevant layers."""
    found: list[dict] = []
    res = wd.http_get(f"{url}?f=pjson", accept="application/json")
    if res["status"] != "ok":
        return found, res
    try:
        data = json.loads(res["bytes"].decode("utf-8", "ignore"))
    except Exception as exc:
        res["status"] = "parse_error"
        res["error"] = str(exc)[:120]
        return found, res
    services = data.get("services", [])
    for svc in services:
        name = svc.get("name", "")
        if not wd.matched_keywords(name):
            continue
        stype = svc.get("type", "MapServer")
        svc_url = f"{url.rsplit('/rest/services', 1)[0]}/rest/services/{name}/{stype}"
        found.append({
            "title": f"ArcGIS {stype}: {name}", "url": svc_url, "format": "arcgis_service",
            "matched": name, "method": "arcgis_rest",
            "is_direct_download": "false", "is_api_endpoint": "true",
        })
    return found, res


def _probe_wfs(url: str, _region: str) -> tuple[list[dict], dict]:
    """WFS GetCapabilities -> flood-relevant FeatureTypes."""
    found: list[dict] = []
    cap = f"{url}?service=WFS&version=2.0.0&request=GetCapabilities"
    res = wd.http_get(cap, accept="application/xml")
    if res["status"] != "ok":
        return found, res
    try:
        text = res["bytes"].decode("utf-8", "ignore")
    except Exception:
        res["status"] = "decode_error"
        return found, res
    names = re.findall(r"<(?:[\w]+:)?Name>([^<]+)</(?:[\w]+:)?Name>", text)
    for name in names:
        if not wd.matched_keywords(name):
            continue
        gf = (f"{url}?service=WFS&version=2.0.0&request=GetFeature&typeNames="
              f"{name}&outputFormat=application/json&count=1000")
        found.append({
            "title": f"WFS FeatureType: {name}", "url": gf, "format": "wfs_geojson",
            "matched": name, "method": "wfs_getcapabilities",
            "is_direct_download": "true", "is_api_endpoint": "true",
        })
    return found, res


def _probe_portal(url: str, _region: str) -> tuple[list[dict], dict]:
    """Light, robots-respecting check of an official portal home page. Extracts
    direct data-file links whose text/URL matches flood keywords. depth=0 only."""
    found: list[dict] = []
    allowed, why = wd.robots_allows(url, urlparse_path(url))
    if not allowed:
        res = {"status": "robots_blocked" if wd.network_enabled() else "network_disabled",
               "http_code": "", "content_type": "", "error": why}
        return found, res
    wd.polite_sleep()
    res = wd.http_get(url, max_bytes=2 * 1024 * 1024, accept="text/html")
    if res["status"] != "ok":
        return found, res
    try:
        html = res["bytes"].decode("utf-8", "ignore")
    except Exception:
        res["status"] = "decode_error"
        return found, res
    for m in re.finditer(r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
        href, anchor = m.group(1), re.sub(r"<[^>]+>", " ", m.group(2))
        ext = wd.classify_url_ext(href)
        if ext in {"raster_blocked", "executable_blocked", "unknown_ext"}:
            continue
        text = f"{anchor} {href}"
        if not wd.matched_keywords(text):
            continue
        full = href if href.startswith("http") else f"{url.rstrip('/')}/{href.lstrip('/')}"
        found.append({
            "title": anchor.strip()[:120] or full, "url": full, "format": ext,
            "matched": text, "method": "data_portal_html",
            "is_direct_download": "true", "is_api_endpoint": "false",
        })
    return found, res


def urlparse_path(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).path or "/"


PROBES = {
    "ckan_api": _probe_ckan,
    "arcgis_rest": _probe_arcgis,
    "wfs_getcapabilities": _probe_wfs,
    "data_portal_html": _probe_portal,
}


def _discover() -> tuple[list[dict], list[dict]]:
    sources: list[dict] = []
    debug: list[dict] = []
    net = "true" if wd.network_enabled() else "false"
    sidx = 0
    for pidx, (region, institution, url, method) in enumerate(wd.ENDPOINTS):
        city = wd.REGIONS[region]["municipality"]
        is_official = wd.is_official_domain(url)
        # Always register the endpoint itself as a candidate (manual-search root).
        score, tier = wd.priority_score(
            is_official=is_official, has_geometry_hint=method in {"arcgis_rest", "wfs_getcapabilities"},
            has_temporal_hint=False, is_occurrence=method == "ckan_api",
            is_risk_or_alert=False, is_news=False)
        sources.append({
            "discovery_id": f"S13BSRC_{sidx:04d}", "region": region, "municipality": city,
            "institution": institution, "source_title": f"{institution} ({method})",
            "source_url": url, "source_domain": wd.domain_of(url), "discovery_method": method,
            "matched_keywords": "", "source_format": method,
            "is_direct_download": "false",
            "is_api_endpoint": "true" if method in {"ckan_api", "arcgis_rest", "wfs_getcapabilities"} else "false",
            "is_official_domain": "true" if is_official else "false",
            "priority_score": score, "priority_tier": tier,
            "expected_geometry": "polygon|point|bbox" if method in {"arcgis_rest", "wfs_getcapabilities"} else "csv|geojson|point",
            "expected_temporal_fields": "data|date|dt_ocorrencia|inicio|fim",
            "download_candidate": "false",
            "manual_review_reason": "endpoint institucional raiz; requer probe live ou busca manual",
            "review_only": "true",
            "notes": "endpoint registrado; nenhum download direto inventado",
        })
        sidx += 1

        probe = PROBES.get(method)
        found: list[dict] = []
        res: dict = {"status": "network_disabled" if net == "false" else "not_attempted",
                     "http_code": "", "content_type": "", "error": ""}
        if probe is not None:
            found, res = probe(url, region)
        debug.append({
            "probe_id": f"S13BPROBE_{pidx:04d}", "region": region, "institution": institution,
            "discovery_method": method, "url": url, "network_enabled": net,
            "status": str(res.get("status", "")), "http_code": str(res.get("http_code", "")),
            "content_type": str(res.get("content_type", "")), "n_resources_found": len(found),
            "review_only": "true", "error_or_note": str(res.get("error", ""))[:160],
        })
        for fobj in found:
            ftext = fobj.get("matched", "")
            is_risk = wd.looks_like_risk(ftext)
            ext = wd.classify_url_ext(fobj["url"])
            has_geom_hint = fobj["format"] in {"wfs_geojson", "arcgis_service"} or ext in {".geojson", ".json", ".kml", ".kmz", ".gpkg", ".zip"}
            fscore, ftier = wd.priority_score(
                is_official=is_official, has_geometry_hint=has_geom_hint,
                has_temporal_hint=False, is_occurrence=True,
                is_risk_or_alert=is_risk, is_news=False)
            dl_ok = (fobj.get("is_direct_download") == "true"
                     and ext not in {"raster_blocked", "executable_blocked", "unknown_ext"}
                     and not is_risk)
            sources.append({
                "discovery_id": f"S13BSRC_{sidx:04d}", "region": region, "municipality": city,
                "institution": institution, "source_title": fobj["title"][:160],
                "source_url": fobj["url"], "source_domain": wd.domain_of(fobj["url"]),
                "discovery_method": fobj["method"],
                "matched_keywords": "|".join(wd.matched_keywords(ftext)),
                "source_format": fobj["format"], "is_direct_download": fobj.get("is_direct_download", "false"),
                "is_api_endpoint": fobj.get("is_api_endpoint", "false"),
                "is_official_domain": "true" if wd.is_official_domain(fobj["url"]) else "false",
                "priority_score": fscore, "priority_tier": "low" if is_risk else ftier,
                "expected_geometry": "polygon|point|bbox" if has_geom_hint else "unknown",
                "expected_temporal_fields": "data|date|dt_ocorrencia",
                "download_candidate": "true" if dl_ok else "false",
                "manual_review_reason": "recurso descoberto via probe live" if not is_risk else "rebaixado para risco/alerta",
                "review_only": "true",
                "notes": "descoberto por probe; classificacao definitiva ocorre no parser",
            })
            sidx += 1
    return sources, debug


def _write_report(queries, sources, debug) -> None:
    net = wd.network_enabled()
    by_region = Counter(s["region"] for s in sources)
    by_method = Counter(s["discovery_method"] for s in sources)
    dl_cands = sum(1 for s in sources if s["download_candidate"] == "true")
    probe_status = Counter(d["status"] for d in debug)
    blocked = "Sim" if not net else "Nao"
    md = f"""# SUSC-13B-AUTO - Relatorio de descoberta automatica

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Politica de rede
- Variavel de ativacao: `SUSC_13B_NETWORK=1`.
- Rede habilitada nesta execucao: **{"Sim" if net else "Nao"}**.
- Bloqueio de rede registrado (offline determinístico): **{blocked}**.
- robots.txt respeitado antes de qualquer crawling HTML; sem chave de API; sem
  Google Maps; sem geocoding generico; sem raster; limite de 250MB por arquivo.

## 2. Plano de consultas
Total de consultas planejadas: **{len(queries)}** (Recife/Petropolis/Curitiba x
templates de palavra-chave + intents site-scoped). Nenhuma consulta executa
scraping aberto; intents site-scoped sao documentados, nao raspados.

## 3. Endpoints e fontes candidatas
Total de fontes candidatas registradas: **{len(sources)}**.

| regiao | fontes |
|---|---|
{_tbl(by_region)}

| metodo de descoberta | fontes |
|---|---|
{_tbl(by_method)}

## 4. Probes executados
Total de probes: **{len(debug)}**.

| status do probe | n |
|---|---|
{_tbl(probe_status)}

## 5. Candidatos a download
Fontes com `download_candidate=true`: **{dl_cands}**.
{"Offline: nenhum recurso direto foi confirmado; aquisicao registra network_disabled." if not net else "Recursos diretos confirmados via probe live; aquisicao tentara download leve e controlado."}

## 6. Limites e governanca
- Endpoints sao raizes institucionais conhecidas; nenhum URL de download direto
  e inventado.
- Risco, alerta, previsao e suscetibilidade sao rebaixados, nunca evento observado.
- Tudo permanece review-only; nada cria ground truth, treino ou score v7.
"""
    write_markdown(REPORT, md)


def _tbl(counter: Counter) -> str:
    if not counter:
        return "| nenhum | 0 |"
    return "\n".join(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Official Event Source Discovery")
    print("=" * 60)
    print(f"network enabled: {wd.network_enabled()} (set {wd.NETWORK_ENV}=1 to probe live)")
    queries = _build_query_manifest()
    sources, debug = _discover()
    write_csv(QUERY_MANIFEST, queries, QUERY_FIELDS)
    write_csv(DISCOVERED, sources, DISCOVERY_FIELDS)
    write_csv(DEBUG_LOG, debug, DEBUG_FIELDS)
    _write_report(queries, sources, debug)
    print(f"queries -> {rel(QUERY_MANIFEST)} ({len(queries)})")
    print(f"sources -> {rel(DISCOVERED)} ({len(sources)})")
    print(f"debug   -> {rel(DEBUG_LOG)} ({len(debug)})")
    print("download candidates:", sum(1 for s in sources if s["download_candidate"] == "true"))
    print("review-only. No invented direct URL. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
