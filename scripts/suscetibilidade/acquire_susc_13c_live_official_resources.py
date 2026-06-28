"""
SUSC-13C-LIVE acquisition of live candidate resources.

Aggregates the live discovery (CKAN/ArcGIS/WFS/HTML probes + 13B live discovered
sources) into a single live source registry, then really downloads the candidate
vector/tabular/documentary files (<=250MB, no raster/Sentinel). ArcGIS/WFS GeoJSON
already saved by their probes is registered, not re-fetched. Requires
SUSC_13B_NETWORK=1.

Writes:
  - outputs_public/suscetibilidade/SUSC_13C_live_discovered_sources.csv
  - outputs_public/suscetibilidade/SUSC_13C_live_discovery_report.md
  - manifests/suscetibilidade/susc_13c_live_download_manifest_v1.csv
  - datasets/suscetibilidade/observed_event_sources_susc13c_live/  (raw downloads)
  - outputs_public/suscetibilidade/SUSC_13C_live_acquisition_report.md
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import ensure_dir, read_csv, rel, sha256_bytes, write_csv, write_markdown  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
DROP = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13c_live"

CKAN_CSV = OUT / "SUSC_13C_live_ckan_resources.csv"
ARCGIS_CSV = OUT / "SUSC_13C_live_arcgis_layers.csv"
WFS_CSV = OUT / "SUSC_13C_live_wfs_layers.csv"
HTML_CSV = OUT / "SUSC_13C_live_html_file_links.csv"
B13_SOURCES = MAN / "susc_13b_auto_discovered_sources_v1.csv"

LIVE_SOURCES = OUT / "SUSC_13C_live_discovered_sources.csv"
LIVE_DISCOVERY_REPORT = OUT / "SUSC_13C_live_discovery_report.md"
DOWNLOAD_MANIFEST = MAN / "susc_13c_live_download_manifest_v1.csv"
ACQ_REPORT = OUT / "SUSC_13C_live_acquisition_report.md"

LIVE_SOURCE_FIELDS = [
    "source_id", "region", "institution", "source_url", "source_domain",
    "discovery_method", "status_code", "content_type", "matched_keywords",
    "is_direct_download", "is_api_endpoint", "is_ckan", "is_arcgis", "is_wfs",
    "is_download_candidate", "review_only", "reason",
]

DL_FIELDS = [
    "download_id", "source_probe", "region", "institution", "source_url",
    "download_status", "http_status", "content_type", "content_length",
    "file_path", "file_size_bytes", "sha256", "content_ext", "blocked_reason",
    "requires_manual_review", "can_be_ground_truth", "allowed_for_training",
    "review_only", "notes",
]

RASTER_CT = ("image/tiff", "image/geotiff", "application/x-netcdf", "image/jp2")
MAX_DOWNLOADS = 80
# HTML links on official portals frequently match only the host token (e.g.
# "defesa-civil" in defesacivil.pr.gov.br) and point to institutional PDFs
# (decrees, logos). Only fetch HTML links that carry a genuine flood term.
HTML_FLOOD_TERMS = ("alagament", "inundac", "inundaç", "enchente", "enxurrada",
                    "transbordament", "mancha", "alagad", "flood")


def _html_link_is_flood(row: dict) -> bool:
    text = " ".join([row.get("matched_keywords", ""), row.get("anchor_text", ""),
                     row.get("file_url", "")]).lower()
    return any(t in text for t in HTML_FLOOD_TERMS)


def _inst_from(url: str) -> str:
    d = wd.domain_of(url)
    if "recife" in d:
        return "Prefeitura/Dados Abertos Recife"
    if "apac" in d:
        return "APAC-PE"
    if "petropolis" in d:
        return "Prefeitura de Petropolis"
    if "inea" in d:
        return "INEA-RJ"
    if "curitiba" in d or "ippuc" in d:
        return "Prefeitura/GeoCuritiba"
    if "defesacivil.pr" in d:
        return "Defesa Civil PR"
    if "sgb" in d or "cprm" in d:
        return "CPRM/SGB"
    if "ana.gov" in d:
        return "ANA"
    if "dados.gov" in d or "mdr" in d:
        return "Governo Federal"
    return d or "desconhecido"


def _build_live_sources() -> list[dict]:
    rows: list[dict] = []
    sid = 0

    def add(region, url, method, status, ct, matched, is_dd, is_api, is_ckan, is_arcgis, is_wfs, dl, reason):
        nonlocal sid
        rows.append({
            "source_id": f"S13CLIVE_{sid:04d}", "region": region, "institution": _inst_from(url),
            "source_url": url, "source_domain": wd.domain_of(url), "discovery_method": method,
            "status_code": status, "content_type": ct, "matched_keywords": matched,
            "is_direct_download": is_dd, "is_api_endpoint": is_api, "is_ckan": is_ckan,
            "is_arcgis": is_arcgis, "is_wfs": is_wfs, "is_download_candidate": dl,
            "review_only": "true", "reason": reason,
        })
        sid += 1

    for r in (read_csv(CKAN_CSV) if CKAN_CSV.exists() else []):
        if not r.get("url"):
            add(r["region"], r["ckan_base"], "ckan_api", r.get("status_code", ""), "", "",
                "false", "true", "true", "false", "false", "false", r.get("notes", ""))
            continue
        add(r["region"], r["url"], "ckan_api", r.get("status_code", ""), r.get("format", ""),
            r.get("matched_keywords", ""), "true", "false", "true", "false", "false",
            r.get("is_download_candidate", "false"), "recurso CKAN")
    for r in (read_csv(ARCGIS_CSV) if ARCGIS_CSV.exists() else []):
        url = r.get("service_url") or r.get("arcgis_root", "")
        add(r["region"], url, "arcgis_rest", r.get("query_status", ""), "geojson",
            r.get("matched_keywords", ""), "true", "true", "false", "true", "false",
            r.get("is_download_candidate", "false"), r.get("notes", ""))
    for r in (read_csv(WFS_CSV) if WFS_CSV.exists() else []):
        add(r["region"], r.get("wfs_endpoint", ""), "wfs_getcapabilities",
            r.get("getfeature_status") or r.get("getcapabilities_status", ""), "geojson",
            r.get("matched_keywords", ""), "true", "true", "false", "false", "true",
            r.get("is_download_candidate", "false"), r.get("notes", ""))
    for r in (read_csv(HTML_CSV) if HTML_CSV.exists() else []):
        if not r.get("file_url"):
            continue
        add(r["region"], r["file_url"], "data_portal_html", "", r.get("ext", ""),
            r.get("matched_keywords", ""), "true", "false", "false", "false", "false",
            r.get("is_download_candidate", "false"), "link HTML oficial")
    return rows


def _is_raster_ct(ct: str) -> bool:
    return any(ct.lower().startswith(x) for x in RASTER_CT)


def _download(region, url, source_probe) -> dict:
    inst = _inst_from(url)
    ext = wd.classify_url_ext(url)
    base = {
        "download_id": "", "source_probe": source_probe, "region": region,
        "institution": inst, "source_url": url, "requires_manual_review": "true",
        "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
    }
    if ext in {"raster_blocked", "executable_blocked"}:
        return {**base, "download_status": "blocked", "http_status": "", "content_type": "",
                "content_length": "", "file_path": "", "file_size_bytes": "", "sha256": "",
                "content_ext": ext, "blocked_reason": ext, "notes": "extensao bloqueada"}
    res = wd.http_get(url, max_bytes=wd.MAX_BYTES)
    ct = str(res.get("content_type", ""))
    if res["status"] != "ok":
        return {**base, "download_status": res["status"], "http_status": str(res.get("http_code", "")),
                "content_type": ct, "content_length": str(res.get("content_length", "")),
                "file_path": "", "file_size_bytes": "", "sha256": "", "content_ext": ext,
                "blocked_reason": str(res.get("error", ""))[:80], "notes": "download nao concluido"}
    if _is_raster_ct(ct):
        return {**base, "download_status": "blocked", "http_status": "200", "content_type": ct,
                "content_length": str(res.get("content_length", "")), "file_path": "",
                "file_size_bytes": "", "sha256": "", "content_ext": ext,
                "blocked_reason": "raster_content_type", "notes": "content-type raster bloqueado"}
    data = res["bytes"]
    sub = "ckan" if source_probe == "ckan" else ("html" if source_probe == "html" else "misc")
    d = ensure_dir(DROP / sub)
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download"
    if "." not in name:
        name = f"{sha256_bytes(url.encode())[:16]}{ext if ext.startswith('.') else '.bin'}"
    out = d / name[:180]
    out.write_bytes(data)  # type: ignore[arg-type]
    return {**base, "download_status": "downloaded", "http_status": str(res.get("http_code", "200")),
            "content_type": ct, "content_length": str(res.get("content_length", "")),
            "file_path": rel(out), "file_size_bytes": len(data), "sha256": res.get("sha256", ""),
            "content_ext": ext, "blocked_reason": "", "notes": "download leve controlado"}


def main() -> int:
    print("=" * 60)
    print("SUSC-13C-LIVE Acquisition")
    print("=" * 60)
    print(f"network enabled: {wd.network_enabled()}")
    ensure_dir(DROP)

    sources = _build_live_sources()
    write_csv(LIVE_SOURCES, sources, LIVE_SOURCE_FIELDS)

    # download manifest
    rows: list[dict] = []
    # CKAN + HTML direct file downloads
    ckan_cands = [r for r in (read_csv(CKAN_CSV) if CKAN_CSV.exists() else []) if r.get("is_download_candidate") == "true" and r.get("url")]
    html_cands = [r for r in (read_csv(HTML_CSV) if HTML_CSV.exists() else [])
                  if r.get("is_download_candidate") == "true" and r.get("file_url") and _html_link_is_flood(r)]
    n = 0
    for r in ckan_cands:
        if n >= MAX_DOWNLOADS:
            break
        rows.append(_download(r["region"], r["url"], "ckan"))
        n += 1
        wd.polite_sleep()
    for r in html_cands:
        if n >= MAX_DOWNLOADS:
            break
        rows.append(_download(r["region"], r["file_url"], "html"))
        n += 1
        wd.polite_sleep()

    # register ArcGIS/WFS raw geojson already saved by the probes
    for sub in ("arcgis", "wfs"):
        d = DROP / sub
        if d.exists():
            for p in sorted(d.glob("*")):
                if p.is_file():
                    rows.append({
                        "download_id": "", "source_probe": sub, "region": "unknown",
                        "institution": "ArcGIS/WFS probe", "source_url": rel(p),
                        "download_status": "downloaded_by_probe", "http_status": "200",
                        "content_type": "application/geo+json", "content_length": str(p.stat().st_size),
                        "file_path": rel(p), "file_size_bytes": p.stat().st_size,
                        "sha256": sha256_bytes(p.read_bytes()), "content_ext": p.suffix.lower(),
                        "blocked_reason": "", "requires_manual_review": "true",
                        "can_be_ground_truth": "false", "allowed_for_training": "false",
                        "review_only": "true", "notes": "GeoJSON salvo pelo probe ArcGIS/WFS",
                    })

    for i, r in enumerate(rows):
        r["download_id"] = f"S13CDL_{i:04d}"
    write_csv(DOWNLOAD_MANIFEST, rows, DL_FIELDS)

    # discovery report
    by_method = Counter(s["discovery_method"] for s in sources)
    dl_status = Counter(r["download_status"] for r in rows)
    downloaded = sum(1 for r in rows if r["download_status"] in {"downloaded", "downloaded_by_probe"})
    blocked = sum(1 for r in rows if r["download_status"] == "blocked")
    write_markdown(LIVE_DISCOVERY_REPORT, f"""# SUSC-13C-LIVE - Relatorio de descoberta live

Status: **review-only**

Fontes live consolidadas: **{len(sources)}** (CKAN/ArcGIS/WFS/HTML).

| metodo | n |
|---|---|
{_tbl(by_method)}

Candidatos a download: **{sum(1 for s in sources if s['is_download_candidate'] == 'true')}**.
Detalhes por tecnologia em SUSC_13C_live_ckan_resources.csv, _arcgis_layers.csv,
_wfs_layers.csv, _html_file_links.csv.
""")

    write_markdown(ACQ_REPORT, f"""# SUSC-13C-LIVE - Relatorio de aquisicao

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

- Tentativas de download: **{len(rows)}**
- Downloads concluidos: **{downloaded}**
- Bloqueados: **{blocked}**

| status | n |
|---|---|
{_tbl(dl_status)}

Limites: <=250MB, sem raster/Sentinel bruto, sem chave de API, content-type
verificado. SHA256, status HTTP, content-type e tamanho registrados no manifesto.
""")

    print(f"live sources: {len(sources)} | download attempts: {len(rows)} | downloaded: {downloaded} | blocked: {blocked}")
    print("status:", dict(dl_status))
    print("review-only. No raster. No invented data.")
    return 0


def _tbl(counter: Counter) -> str:
    if not counter:
        return "| nenhum | 0 |"
    return "\n".join(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))))


if __name__ == "__main__":
    raise SystemExit(main())
