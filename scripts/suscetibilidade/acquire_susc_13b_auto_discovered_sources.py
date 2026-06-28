"""
SUSC-13B-AUTO controlled acquisition of auto-discovered sources.

Downloads only sources flagged ``download_candidate=true`` by the discovery
stage, and only when network is enabled (SUSC_13B_NETWORK=1). Every attempt is
recorded with content-type, content-length, SHA256 and a status. Raster,
Sentinel raw, executables, files over 250MB and unknown types are blocked.
Manual files dropped into the acquisition directory are registered with hashes.

Writes:
  - datasets/suscetibilidade/observed_event_sources_susc13b_auto/  (raw + README)
  - manifests/suscetibilidade/susc_13b_auto_download_manifest_v1.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_acquisition_report.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import ensure_dir, read_csv, rel, sha256_file, write_csv, write_markdown  # noqa: E402

DISCOVERED = ROOT / "manifests" / "suscetibilidade" / "susc_13b_auto_discovered_sources_v1.csv"
DROP_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13b_auto"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13b_auto_download_manifest_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_acquisition_report.csv"

FIELDS = [
    "download_id", "discovery_id", "region", "institution", "source_title",
    "discovery_method", "url_or_local_path", "download_status", "http_code_or_error",
    "content_type", "content_length", "file_path", "file_size_bytes", "sha256",
    "content_ext", "geometry_detected", "coordinate_detected", "requires_manual_review",
    "can_be_ground_truth", "allowed_for_training", "review_only", "notes",
]

README = """# observed_event_sources_susc13b_auto

Diretorio de aquisicao automatica/controlada do SUSC-13B-AUTO.

Conteudo:
- Subpastas por metodo (`ckan/`, `arcgis/`, `wfs/`, `portal/`) recebem arquivos
  baixados automaticamente quando `SUSC_13B_NETWORK=1` e a fonte foi marcada
  como `download_candidate=true` na descoberta.
- Voce tambem pode colocar manualmente arquivos oficiais/pequenos aqui (CSV,
  XLSX, GeoJSON, JSON, KML/KMZ, WKT, TXT, XML/GML, GPKG, ZIP vetorial, PDF
  pequeno) com data, tipo de evento e coordenada/poligono.

Bloqueado: raster, Sentinel bruto, executavel, Google Maps, geocoding generico,
chave de API e arquivos acima de 250MB. Todos os artefatos permanecem
`review_only=true`, `can_be_ground_truth=false`, `allowed_for_training=false`.
"""

METHOD_SUBDIR = {
    "ckan_api": "ckan", "arcgis_rest": "arcgis", "wfs_getcapabilities": "wfs",
    "data_portal_html": "portal",
}


def _filename_for(url: str, ext: str) -> str:
    base = url.split("?")[0].rstrip("/").split("/")[-1]
    if not base or "." not in base:
        from susc_io import sha256_bytes
        base = sha256_bytes(url.encode())[:16] + (ext if ext.startswith(".") else ".bin")
    return base


def _attempt_download(src: dict) -> dict:
    url = (src.get("source_url") or "").strip()
    method = src.get("discovery_method", "")
    ext = wd.classify_url_ext(url)
    base = {
        "download_id": "", "discovery_id": src.get("discovery_id", ""),
        "region": src.get("region", ""), "institution": src.get("institution", ""),
        "source_title": src.get("source_title", ""), "discovery_method": method,
        "url_or_local_path": url, "requires_manual_review": "true",
        "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
    }
    if ext in {"raster_blocked", "executable_blocked"}:
        return {**base, "download_status": f"blocked_{ext}", "http_code_or_error": "",
                "content_type": "", "content_length": "", "file_path": "",
                "file_size_bytes": "", "sha256": "", "content_ext": ext,
                "geometry_detected": "false", "coordinate_detected": "false",
                "notes": "extensao bloqueada por politica"}
    if ext == "unknown_ext" and method not in {"arcgis_rest", "wfs_getcapabilities"}:
        return {**base, "download_status": "blocked_unknown_extension", "http_code_or_error": "",
                "content_type": "", "content_length": "", "file_path": "",
                "file_size_bytes": "", "sha256": "", "content_ext": ext,
                "geometry_detected": "false", "coordinate_detected": "false",
                "notes": "tipo nao identificavel; nao baixado"}

    res = wd.http_get(url, max_bytes=wd.MAX_BYTES)
    status = res["status"]
    if status != "ok":
        return {**base, "download_status": status if status != "network_disabled" else "not_attempted_network_disabled",
                "http_code_or_error": str(res.get("error", "")) or str(res.get("status", "")),
                "content_type": str(res.get("content_type", "")),
                "content_length": str(res.get("content_length", "")), "file_path": "",
                "file_size_bytes": "", "sha256": "", "content_ext": ext,
                "geometry_detected": "false", "coordinate_detected": "false",
                "notes": "sem download (offline ou falha controlada)"}
    data = res["bytes"]
    subdir = ensure_dir(DROP_DIR / METHOD_SUBDIR.get(method, "misc"))
    out = subdir / _filename_for(url, ext if ext.startswith(".") else ".json")
    out.write_bytes(data)  # type: ignore[arg-type]
    return {**base, "download_status": "downloaded", "http_code_or_error": str(res.get("http_code", "200")),
            "content_type": str(res.get("content_type", "")),
            "content_length": str(res.get("content_length", "")), "file_path": rel(out),
            "file_size_bytes": len(data), "sha256": res.get("sha256", ""),  # type: ignore[arg-type]
            "content_ext": ext, "geometry_detected": "unknown_pending_parse",
            "coordinate_detected": "unknown_pending_parse",
            "notes": "download leve controlado; parser classifica"}


def _manual_files() -> list[Path]:
    if not DROP_DIR.exists():
        return []
    return sorted(p for p in DROP_DIR.rglob("*") if p.is_file() and p.name != "README.md")


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Controlled Acquisition")
    print("=" * 60)
    print(f"network enabled: {wd.network_enabled()}")
    ensure_dir(DROP_DIR)
    write_markdown(DROP_DIR / "README.md", README)

    if not DISCOVERED.exists():
        print(f"STOP: discovered sources not found: {rel(DISCOVERED)}")
        return 1
    sources = read_csv(DISCOVERED)
    rows: list[dict] = []
    for src in sources:
        if str(src.get("download_candidate", "")).strip().lower() != "true":
            continue
        rows.append(_attempt_download(src))

    # Register downloaded + manually placed files (avoid double-counting README).
    already = {r.get("file_path") for r in rows if r.get("file_path")}
    for p in _manual_files():
        if rel(p) in already:
            continue
        rows.append({
            "download_id": "", "discovery_id": "MANUAL", "region": "unknown",
            "institution": "manual_placement", "source_title": p.name,
            "discovery_method": "manual_drop", "url_or_local_path": rel(p),
            "download_status": "present_local_manual", "http_code_or_error": "",
            "content_type": "", "content_length": "", "file_path": rel(p),
            "file_size_bytes": p.stat().st_size, "sha256": sha256_file(p),
            "content_ext": p.suffix.lower(), "geometry_detected": "unknown_pending_parse",
            "coordinate_detected": "unknown_pending_parse", "requires_manual_review": "true",
            "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
            "notes": "arquivo colocado manualmente; parser classifica conservadoramente",
        })

    for i, r in enumerate(rows):
        r["download_id"] = f"S13BDL_{i:04d}"
    write_csv(MANIFEST, rows, FIELDS)

    report_rows = [{
        "download_id": r["download_id"], "region": r["region"], "institution": r["institution"],
        "source_title": r["source_title"], "discovery_method": r["discovery_method"],
        "download_status": r["download_status"], "content_ext": r["content_ext"],
        "file_size_bytes": r["file_size_bytes"], "requires_manual_review": r["requires_manual_review"],
        "can_be_ground_truth": r["can_be_ground_truth"], "allowed_for_training": r["allowed_for_training"],
        "review_only": r["review_only"],
    } for r in rows]
    write_csv(REPORT, report_rows, [
        "download_id", "region", "institution", "source_title", "discovery_method",
        "download_status", "content_ext", "file_size_bytes", "requires_manual_review",
        "can_be_ground_truth", "allowed_for_training", "review_only"])

    print(f"download candidates processed: {sum(1 for r in rows if r['discovery_id'] != 'MANUAL')}")
    print(f"manual files: {sum(1 for r in rows if r['discovery_id'] == 'MANUAL')}")
    print("status:", dict(Counter(r["download_status"] for r in rows)))
    print("No raster. No >250MB. No API key. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
