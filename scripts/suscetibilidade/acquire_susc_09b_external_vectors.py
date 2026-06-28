"""
SUSC-09B External Vector Acquisition (controlled, offline-safe)

Reads the external source registry and attempts a light, safe download only when a
direct URL is present. Uses the shared safe downloader (blocks raster, >100MB, non
allowlisted extensions). Records every attempt with url/status/size/sha256/reason.
Offline or no-direct-url -> not_attempted (does not fail the milestone).

Writes:
  - datasets/suscetibilidade/external_event_geometry_sources_susc09b/  (dir, downloads if any)
  - manifests/suscetibilidade/susc_09b_external_download_manifest_v1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, ensure_dir, sha256_file, rel  # noqa: E402
from susc_downloads import safe_download  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_source_registry_v1.csv"
DL_DIR = ROOT / "datasets" / "suscetibilidade" / "external_event_geometry_sources_susc09b"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_download_manifest_v1.csv"

FIELDS = ["download_id", "source_id", "region", "target_institution", "url_or_local_path",
          "download_status", "http_status_or_error", "file_path", "file_size_bytes",
          "sha256", "content_ext", "geometry_detected", "coordinate_detected",
          "requires_manual_review", "review_only", "notes"]


def main() -> int:
    print("=" * 60)
    print("SUSC-09B External Vector Acquisition (offline-safe)")
    print("=" * 60)
    ensure_dir(DL_DIR)
    readme = DL_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# external_event_geometry_sources_susc09b\n\n"
            "Destino de downloads leves opt-in do SUSC-09B (CSV/GeoJSON/KML/WKT/PDF/ZIP vetorial "
            "<=100MB) de URL direta oficial. Vazio por padrao. Nenhum raster, nenhuma imagem "
            "Sentinel, nenhuma chave de API, nenhum geocoding.\n", encoding="utf-8")

    if not REGISTRY.exists():
        print(f"STOP: registry not found: {REGISTRY}")
        return 1
    registry = read_csv(REGISTRY)

    rows = []
    for i, src in enumerate(registry):
        direct = (src.get("direct_url") or "").strip()
        if not direct:
            rows.append({
                "download_id": f"DL_{i:03d}", "source_id": src["source_id"],
                "region": src["region"], "target_institution": src["target_institution"],
                "url_or_local_path": src.get("reference_url", ""),
                "download_status": "not_attempted_no_direct_url_or_no_network",
                "http_status_or_error": "no_direct_url", "file_path": "", "file_size_bytes": "",
                "sha256": "", "content_ext": "", "geometry_detected": "false",
                "coordinate_detected": "false", "requires_manual_review": "true",
                "review_only": "true",
                "notes": "manual official acquisition pending (offline plan)",
            })
            continue
        res = safe_download(direct, DL_DIR)
        fp = res.get("file_path", "")
        sha = res.get("sha256", "")
        if fp and not sha:
            try:
                sha = sha256_file(fp)
            except OSError:
                sha = ""
        rows.append({
            "download_id": f"DL_{i:03d}", "source_id": src["source_id"],
            "region": src["region"], "target_institution": src["target_institution"],
            "url_or_local_path": direct, "download_status": res["download_status"],
            "http_status_or_error": res["http_status_or_error"],
            "file_path": rel(fp) if fp else "", "file_size_bytes": res.get("file_size_bytes", ""),
            "sha256": sha, "content_ext": res.get("content_ext", ""),
            "geometry_detected": "unknown_pending_parse", "coordinate_detected": "unknown_pending_parse",
            "requires_manual_review": "true", "review_only": "true",
            "notes": res.get("notes", ""),
        })

    # also register any manually placed files
    for p in sorted(DL_DIR.rglob("*")):
        if p.is_file() and p.name != "README.md":
            rows.append({
                "download_id": f"LOCAL_{p.stem[:18]}", "source_id": "MANUAL", "region": "unknown",
                "target_institution": "manual_placement",
                "url_or_local_path": rel(p), "download_status": "present_local_manual",
                "http_status_or_error": "", "file_path": rel(p),
                "file_size_bytes": p.stat().st_size, "sha256": sha256_file(p),
                "content_ext": p.suffix.lower(), "geometry_detected": "unknown_pending_parse",
                "coordinate_detected": "unknown_pending_parse", "requires_manual_review": "true",
                "review_only": "true", "notes": "manually placed traceable source",
            })

    write_csv(MANIFEST, rows, FIELDS)
    from collections import Counter
    print(f"\nregistry sources: {len(registry)} | manifest rows: {len(rows)}")
    print("status:", dict(Counter(r["download_status"] for r in rows)))
    print("No raster. No >100MB. No API key. Offline-safe. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
