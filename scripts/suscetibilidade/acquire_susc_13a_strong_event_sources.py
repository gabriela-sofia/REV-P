"""
SUSC-13A controlled acquisition of strong observed-event sources.

Attempts downloads only when the registry explicitly marks a direct/light URL as
downloadable. Manual files placed in the SUSC-13A drop directory are registered
with hashes. Raster, Sentinel raw, API-key flows, aggressive scraping and files
over 100MB are blocked by the shared safe downloader and by registry policy.

Writes:
  - manifests/suscetibilidade/susc_13a_strong_event_download_manifest_v1.csv
  - outputs_public/suscetibilidade/SUSC_13A_strong_event_acquisition_report.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_downloads import safe_download  # noqa: E402
from susc_io import ensure_dir, read_csv, rel, sha256_file, write_csv, write_markdown  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_source_registry_v1.csv"
DROP_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13a"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_download_manifest_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_acquisition_report.csv"

FIELDS = [
    "download_id",
    "source_id",
    "region",
    "institution",
    "source_name",
    "url_or_local_path",
    "download_status",
    "http_status_or_error",
    "file_path",
    "file_size_bytes",
    "sha256",
    "content_ext",
    "geometry_detected",
    "coordinate_detected",
    "requires_manual_review",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
    "notes",
]

README = """# observed_event_sources_susc13a

Diretorio de entrada manual para o SUSC-13A.

Coloque aqui apenas arquivos oficiais/tecnicos pequenos e rastreaveis que possam
fortalecer evidencias observacionais reais de alagamento/inundacao:

- CSV, TSV ou XLSX com data, tipo de evento e lat/lon.
- GeoJSON, KML, KMZ, GPKG ou ZIP de SHP vetorial com CRS documentado.
- WKT ou TXT pequeno contendo geometria explicita.
- PDF pequeno quando trouxer tabela/texto rastreavel de ocorrencia, data e local.

Preferir arquivos que contenham: data ou periodo do evento, municipio/regiao,
coordenada ou poligono/bbox, tipo de evento, nome da fonte, instituicao e URL ou
referencia de origem. Nao colocar raster, Sentinel bruto, Google Maps, geocoding
generico, chaves de API ou arquivos acima de 100MB.

Todos os arquivos permanecem `review_only=true`, `can_be_ground_truth=false` e
`allowed_for_training=false`.
"""


def _write_readme() -> None:
    ensure_dir(DROP_DIR)
    readme = DROP_DIR / "README.md"
    write_markdown(readme, README)


def _manual_files() -> list[Path]:
    if not DROP_DIR.exists():
        return []
    return sorted(p for p in DROP_DIR.rglob("*") if p.is_file() and p.name != "README.md")


def main() -> int:
    print("=" * 60)
    print("SUSC-13A Strong Event Acquisition (controlled)")
    print("=" * 60)
    _write_readme()
    if not REGISTRY.exists():
        print(f"STOP: registry not found: {rel(REGISTRY)}")
        return 1

    registry = read_csv(REGISTRY)
    rows: list[dict] = []
    for i, src in enumerate(registry):
        is_direct = str(src.get("url_is_direct_download", "")).strip().lower() == "true"
        allowed = str(src.get("download_allowed", "")).strip().lower() == "true"
        url = (src.get("url") or "").strip()
        base = {
            "download_id": f"S13ADL_{i:03d}",
            "source_id": src.get("source_id", ""),
            "region": src.get("region", ""),
            "institution": src.get("institution", ""),
            "source_name": src.get("source_name", ""),
            "requires_manual_review": "true",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        }
        if not (is_direct and allowed):
            rows.append(
                {
                    **base,
                    "url_or_local_path": url,
                    "download_status": "not_attempted_no_direct_url_or_no_network",
                    "http_status_or_error": "no_direct_download_allowed",
                    "file_path": "",
                    "file_size_bytes": "",
                    "sha256": "",
                    "content_ext": "",
                    "geometry_detected": "false",
                    "coordinate_detected": "false",
                    "notes": "institutional source registered; manual acquisition required",
                }
            )
            continue
        res = safe_download(url, DROP_DIR)
        fp = res.get("file_path", "")
        rows.append(
            {
                **base,
                "url_or_local_path": url,
                "download_status": str(res.get("download_status", "")),
                "http_status_or_error": str(res.get("http_status_or_error", "")),
                "file_path": rel(fp) if fp else "",
                "file_size_bytes": res.get("file_size_bytes", ""),
                "sha256": res.get("sha256", ""),
                "content_ext": res.get("content_ext", ""),
                "geometry_detected": "unknown_pending_parse" if fp else "false",
                "coordinate_detected": "unknown_pending_parse" if fp else "false",
                "notes": str(res.get("notes", "")),
            }
        )

    for p in _manual_files():
        rows.append(
            {
                "download_id": f"S13ALOCAL_{len(rows):03d}",
                "source_id": "MANUAL",
                "region": "unknown",
                "institution": "manual_placement",
                "source_name": p.name,
                "url_or_local_path": rel(p),
                "download_status": "present_local_manual",
                "http_status_or_error": "",
                "file_path": rel(p),
                "file_size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "content_ext": p.suffix.lower(),
                "geometry_detected": "unknown_pending_parse",
                "coordinate_detected": "unknown_pending_parse",
                "requires_manual_review": "true",
                "can_be_ground_truth": "false",
                "allowed_for_training": "false",
                "review_only": "true",
                "notes": "manually placed source; parser must classify conservatively",
            }
        )

    write_csv(MANIFEST, rows, FIELDS)
    report_rows = [
        {
            "download_id": r["download_id"],
            "region": r["region"],
            "institution": r["institution"],
            "source_name": r["source_name"],
            "download_status": r["download_status"],
            "file_size_bytes": r["file_size_bytes"],
            "content_ext": r["content_ext"],
            "requires_manual_review": r["requires_manual_review"],
            "can_be_ground_truth": r["can_be_ground_truth"],
            "allowed_for_training": r["allowed_for_training"],
            "review_only": r["review_only"],
        }
        for r in rows
    ]
    write_csv(
        REPORT,
        report_rows,
        [
            "download_id",
            "region",
            "institution",
            "source_name",
            "download_status",
            "file_size_bytes",
            "content_ext",
            "requires_manual_review",
            "can_be_ground_truth",
            "allowed_for_training",
            "review_only",
        ],
    )

    print(f"registry sources: {len(registry)} | manifest rows: {len(rows)}")
    print("status:", dict(Counter(r["download_status"] for r in rows)))
    print("No raster. No >100MB. No API key. No aggressive scraping. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
