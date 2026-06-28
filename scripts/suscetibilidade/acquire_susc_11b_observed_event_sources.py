"""
SUSC-11B Observed Event Acquisition (controlled, offline-safe)

Reads the observed-event source registry and attempts a light, safe download only
when a direct URL is recorded (url_is_direct_download=true). Uses the shared safe
downloader (blocks raster, >100MB, non-allowlisted extensions, no API key). Records
every attempt with url/status/size/sha256/reason. Offline or no-direct-url ->
not_attempted_no_direct_url_or_no_network (does not fail the milestone). Any file
manually placed in the acquisition dir is also registered with sha256.

Writes:
  - datasets/suscetibilidade/observed_event_sources_susc11b/  (dir, downloads if any)
  - manifests/suscetibilidade/susc_11b_observed_event_download_manifest_v1.csv
  - outputs_public/suscetibilidade/SUSC_11B_observed_event_acquisition_report.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, ensure_dir, sha256_file, rel  # noqa: E402
from susc_downloads import safe_download  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_source_registry_v1.csv"
DL_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc11b"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_download_manifest_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11B_observed_event_acquisition_report.csv"

FIELDS = ["download_id", "source_id", "region", "institution", "source_name",
          "url_or_local_path", "download_status", "http_status_or_error", "file_path",
          "file_size_bytes", "sha256", "content_ext", "geometry_detected",
          "coordinate_detected", "requires_manual_review", "review_only", "notes"]

README = (
    "# observed_event_sources_susc11b\n\n"
    "Destino de downloads leves opt-in do SUSC-11B (CSV/GeoJSON/JSON/KML/KMZ/WKT/TXT/"
    "PDF pequeno/ZIP vetorial <=100MB) de URL direta oficial rastreavel. Vazio por "
    "padrao (sem URL direta no registry / offline). Nenhum raster, nenhuma imagem "
    "Sentinel, nenhuma chave de API, nenhum Google Maps, nenhum geocoding generico.\n"
)


def main() -> int:
    print("=" * 60)
    print("SUSC-11B Observed Event Acquisition (offline-safe)")
    print("=" * 60)
    ensure_dir(DL_DIR)
    readme = DL_DIR / "README.md"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")

    if not REGISTRY.exists():
        print(f"STOP: registry not found: {REGISTRY}")
        return 1
    registry = read_csv(REGISTRY)

    rows = []
    for i, src in enumerate(registry):
        is_direct = str(src.get("url_is_direct_download", "")).strip().lower() == "true"
        allowed = str(src.get("download_allowed", "")).strip().lower() == "true"
        url = (src.get("url") or "").strip()
        base = {
            "download_id": f"OBSDL_{i:03d}", "source_id": src.get("source_id", ""),
            "region": src.get("region", ""), "institution": src.get("institution", ""),
            "source_name": src.get("source_name", ""),
            "requires_manual_review": "true", "review_only": "true",
        }
        if not (is_direct and allowed):
            rows.append({**base, "url_or_local_path": url,
                         "download_status": "not_attempted_no_direct_url_or_no_network",
                         "http_status_or_error": "no_direct_url", "file_path": "",
                         "file_size_bytes": "", "sha256": "", "content_ext": "",
                         "geometry_detected": "false", "coordinate_detected": "false",
                         "notes": "official source documented; manual acquisition pending"})
            continue
        res = safe_download(url, DL_DIR)
        fp = res.get("file_path", "")
        sha = res.get("sha256", "")
        if fp and not sha:
            try:
                sha = sha256_file(fp)
            except OSError:
                sha = ""
        rows.append({**base, "url_or_local_path": url,
                     "download_status": res["download_status"],
                     "http_status_or_error": res["http_status_or_error"],
                     "file_path": rel(fp) if fp else "",
                     "file_size_bytes": res.get("file_size_bytes", ""), "sha256": sha,
                     "content_ext": res.get("content_ext", ""),
                     "geometry_detected": "unknown_pending_parse",
                     "coordinate_detected": "unknown_pending_parse",
                     "notes": res.get("notes", "")})

    # register any manually placed files in the acquisition dir
    for p in sorted(DL_DIR.rglob("*")):
        if p.is_file() and p.name != "README.md":
            rows.append({
                "download_id": f"OBSLOCAL_{p.stem[:16]}", "source_id": "MANUAL",
                "region": "unknown", "institution": "manual_placement", "source_name": p.name,
                "url_or_local_path": rel(p), "download_status": "present_local_manual",
                "http_status_or_error": "", "file_path": rel(p),
                "file_size_bytes": p.stat().st_size, "sha256": sha256_file(p),
                "content_ext": p.suffix.lower(), "geometry_detected": "unknown_pending_parse",
                "coordinate_detected": "unknown_pending_parse", "requires_manual_review": "true",
                "review_only": "true", "notes": "manually placed traceable source"})

    write_csv(MANIFEST, rows, FIELDS)

    # public acquisition report: compact view per source with outcome
    status_counts = Counter(r["download_status"] for r in rows)
    report_rows = [{
        "download_id": r["download_id"], "region": r["region"],
        "institution": r["institution"], "source_name": r["source_name"],
        "download_status": r["download_status"], "file_size_bytes": r["file_size_bytes"],
        "content_ext": r["content_ext"], "review_only": "true",
    } for r in rows]
    write_csv(REPORT, report_rows,
              ["download_id", "region", "institution", "source_name", "download_status",
               "file_size_bytes", "content_ext", "review_only"])

    print(f"\nregistry sources: {len(registry)} | manifest rows: {len(rows)}")
    print("status:", dict(status_counts))
    print("No raster. No >100MB. No API key. No Google Maps. Offline-safe. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
