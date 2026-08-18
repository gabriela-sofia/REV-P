"""
SUSC-07B Controlled External Coordinate Acquisition (optional, offline-safe)

Reads the external acquisition queue and attempts ONLY light, safe downloads when
a direct URL is already known in the repo/manifest AND the network is available.
It NEVER: downloads heavy rasters/zips, uses API keys, scrapes aggressively, uses
generic geocoding, or queries map providers. If there is no direct URL or no
network, it records the item as not attempted (the marker does not fail the
milestone).

Outputs:
  - datasets/suscetibilidade/external_event_geometry_sources/   (dir, downloads if any)
  - manifests/suscetibilidade/susc_07b_external_coordinate_acquisition_manifest_v1.csv
  - outputs_public/suscetibilidade/SUSC_07B_external_coordinate_acquisition_report.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

QUEUE = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_external_coordinate_acquisition_queue.csv"
)
DL_DIR = ROOT / "datasets" / "suscetibilidade" / "external_event_geometry_sources"
MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_07b_external_coordinate_acquisition_manifest_v1.csv"
)
REPORT = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_external_coordinate_acquisition_report.csv"
)

LIGHT_EXTS = {".csv", ".geojson", ".json", ".kml", ".txt", ".pdf"}
MANIFEST_FIELDS = [
    "source_id", "evidence_id", "url_or_local_path", "download_status",
    "http_status_or_error", "file_path", "file_size_bytes", "sha256",
    "source_type", "geometry_detected", "coordinate_detected",
    "requires_manual_review", "notes",
]


def main() -> int:
    print("=" * 60)
    print("SUSC-07B Controlled External Acquisition (offline-safe)")
    print("=" * 60)

    DL_DIR.mkdir(parents=True, exist_ok=True)
    readme = DL_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# external_event_geometry_sources\n\n"
            "Destino de downloads leves opt-in do SUSC-07B (CSV/GeoJSON/KML/PDF pequeno) "
            "obtidos de URLs diretas rastreaveis. Vazio por padrao (aquisicao offline = "
            "not_attempted). Nenhuma coordenada e inventada; nada de raster pesado/ZIP/API.\n",
            encoding="utf-8",
        )

    if not QUEUE.exists():
        print(f"STOP: acquisition queue not found: {QUEUE}")
        return 1

    with QUEUE.open("r", encoding="utf-8", newline="") as f:
        queue = list(csv.DictReader(f))

    # No direct download URLs exist in the queue (offline plan). Optionally, a
    # human can add a 'direct_url' column later; here we honor offline/no-URL.
    manifest_rows = []
    for i, q in enumerate(queue):
        direct_url = (q.get("direct_url") or "").strip()
        if not direct_url:
            status = "not_attempted_no_direct_url_or_no_network"
            err = "no_direct_url_in_queue"
        else:
            # Even with a URL, downloads are opt-in and disabled by default here.
            status = "skipped_download_disabled_by_default"
            err = "download_allowed=false"
        manifest_rows.append({
            "source_id": f"SRC_{i:04d}",
            "evidence_id": q.get("evidence_id", ""),
            "url_or_local_path": direct_url or q.get("target_dataset_or_document", ""),
            "download_status": status,
            "http_status_or_error": err,
            "file_path": "",
            "file_size_bytes": "",
            "sha256": "",
            "source_type": q.get("target_institution", ""),
            "geometry_detected": "false",
            "coordinate_detected": "false",
            "requires_manual_review": "true",
            "notes": "offline acquisition; manual official-source retrieval pending",
        })

    # If any files were manually placed in DL_DIR, register them (real, traceable).
    for p in sorted(DL_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in LIGHT_EXTS and p.name != "README.md":
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            manifest_rows.append({
                "source_id": f"LOCAL_{p.stem[:20]}",
                "evidence_id": "MANUAL_PLACEMENT",
                "url_or_local_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "download_status": "present_local_manual",
                "http_status_or_error": "",
                "file_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "file_size_bytes": p.stat().st_size,
                "sha256": h.hexdigest(),
                "source_type": "manual_placement",
                "geometry_detected": "unknown_pending_extraction",
                "coordinate_detected": "unknown_pending_extraction",
                "requires_manual_review": "true",
                "notes": "manually placed traceable source; extraction will parse it",
            })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows)

    # report = compact status rollup
    from collections import Counter
    counts = Counter(r["download_status"] for r in manifest_rows)
    with REPORT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["download_status", "count"])
        for k, v in sorted(counts.items()):
            w.writerow([k, v])

    print(f"\nqueue rows: {len(queue)} | manifest rows: {len(manifest_rows)}")
    print("download status:", dict(counts))
    print("\nNo heavy download. No API. No geocoding. Offline-safe. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
