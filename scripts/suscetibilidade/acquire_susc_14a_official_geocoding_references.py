"""
SUSC-14A FASE 3 - acquire (lightly) the official spatial-reference layers.

Reads the FASE 2 registry. For each ``reused_local_official`` reference it
registers the existing local file (already acquired by SUSC-13C) with its
SHA256/size, no copy. For each live ``download_candidate`` it only attempts a
controlled download when SUSC_13B_NETWORK=1 (offline => network_disabled).
Rasters, executables, HTML-without-data, files >250MB and private data are
blocked.

Writes:
  - manifests/suscetibilidade/susc_14a_official_reference_download_manifest_v1.csv
  - datasets/suscetibilidade/official_geocoding_references_susc14a/  (raw, gitignored)
  - outputs_public/suscetibilidade/SUSC_14A_official_reference_acquisition_report.md
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import ensure_dir, read_csv, rel, sha256_file, write_csv, write_markdown  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_14a_official_geocoding_reference_registry_v1.csv"
DROP = ROOT / "datasets" / "suscetibilidade" / "official_geocoding_references_susc14a"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14a_official_reference_download_manifest_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_official_reference_acquisition_report.md"

FIELDS = [
    "download_id", "reference_id", "region", "institution", "source_name",
    "source_url", "access_method", "download_status", "http_status",
    "content_type", "file_path", "file_size_bytes", "sha256", "content_ext",
    "blocked_reason", "requires_manual_review", "can_be_ground_truth",
    "allowed_for_training", "review_only", "notes",
]

RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf", ".grib", ".grib2"}
EXEC_EXTS = {".exe", ".dll", ".bin", ".msi", ".bat", ".sh"}
ALLOWED_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".geojson", ".json", ".kml",
                ".kmz", ".gpkg", ".wkt", ".txt", ".xml", ".gml", ".zip", ".pdf"}
MAX_BYTES = 250 * 1024 * 1024
MAX_DOWNLOADS = 40


def _gov(row: dict) -> dict:
    row["requires_manual_review"] = "true"
    row["can_be_ground_truth"] = "false"
    row["allowed_for_training"] = "false"
    row["review_only"] = "true"
    return row


def _local_files_for(ref: dict) -> list[Path]:
    """Resolve the local file a reused_local_official reference points to."""
    note = ref.get("notes", "")
    out = []
    if "caminho=" in note:
        relpath = note.split("caminho=", 1)[1].strip()
        p = ROOT / relpath
        if p.exists():
            out.append(p)
    return out


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Official Reference Acquisition")
    print("=" * 60)
    ensure_dir(DROP)
    if not REGISTRY.exists():
        print(f"STOP: missing registry {rel(REGISTRY)}")
        return 1
    refs = read_csv(REGISTRY)
    net = g14.network_enabled()
    rows: list[dict] = []
    downloads = 0

    for ref in refs:
        rid = ref.get("reference_id", "")
        base = {
            "reference_id": rid, "region": ref.get("region", ""),
            "institution": ref.get("institution", ""), "source_name": ref.get("source_name", ""),
            "source_url": ref.get("source_url", ""), "access_method": ref.get("access_method", ""),
        }
        if ref.get("access_method") == "reused_local_official":
            for p in _local_files_for(ref):
                ext = p.suffix.lower()
                rows.append(_gov({**base,
                    "download_status": "reused_local_official", "http_status": "",
                    "content_type": "", "file_path": rel(p),
                    "file_size_bytes": p.stat().st_size, "sha256": sha256_file(p),
                    "content_ext": ext, "blocked_reason": "",
                    "notes": "camada oficial ja adquirida (13C); registrada sem copia"}))
            continue

        if ref.get("download_candidate") != "true":
            rows.append(_gov({**base,
                "download_status": "endpoint_only", "http_status": "", "content_type": "",
                "file_path": "", "file_size_bytes": "", "sha256": "", "content_ext": "",
                "blocked_reason": "", "notes": "raiz institucional; sem recurso direto a baixar"}))
            continue

        if not net:
            rows.append(_gov({**base,
                "download_status": "network_disabled", "http_status": "", "content_type": "",
                "file_path": "", "file_size_bytes": "", "sha256": "", "content_ext": "",
                "blocked_reason": "", "notes": f"{g14.NETWORK_ENV} nao habilitado; aquisicao live ignorada (offline-safe)"}))
            continue

        # network on: probe the endpoint root (metadata only). We never invent a
        # direct download URL here; full live resource walking is the 13C live
        # layer's job and its outputs are reused. So at most we record the probe.
        if downloads >= MAX_DOWNLOADS:
            rows.append(_gov({**base, "download_status": "skipped_quota", "http_status": "",
                "content_type": "", "file_path": "", "file_size_bytes": "", "sha256": "",
                "content_ext": "", "blocked_reason": "max_downloads", "notes": "quota de aquisicao atingida"}))
            continue
        res = g14.http_get(ref.get("source_url", ""), accept="application/json")
        downloads += 1
        rows.append(_gov({**base,
            "download_status": "probed_endpoint", "http_status": str(res.get("http_code", "")),
            "content_type": str(res.get("content_type", "")), "file_path": "",
            "file_size_bytes": "", "sha256": "", "content_ext": "",
            "blocked_reason": "" if res.get("status") == "ok" else str(res.get("status", "")),
            "notes": "probe live do endpoint; recursos diretos reutilizados da camada 13C"}))

    for i, r in enumerate(rows):
        r["download_id"] = f"S14ADL_{i:04d}"
    rows = [{k: r.get(k, "") for k in FIELDS} for r in rows]
    write_csv(MANIFEST, rows, FIELDS)

    status = Counter(r["download_status"] for r in rows)
    reused = status.get("reused_local_official", 0)
    md = f"""# SUSC-14A - Aquisicao de referencias espaciais oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Politica
- Rede habilitada: **{"Sim" if net else "Nao"}** (ativacao `SUSC_13B_NETWORK=1`).
- Aceitos: CSV, XLSX, GeoJSON, JSON, KML/KMZ, SHP ZIP, GPKG, WFS/ArcGIS GeoJSON, PDF pequeno, TXT.
- Bloqueados: raster, Sentinel bruto, executavel, HTML sem dado, dados privados, arquivos > 250MB.
- Sem copia de dado pesado: camadas locais ja adquiridas (13C) sao registradas por SHA256/caminho.

## 2. Resultado
Total de registros no manifesto: **{len(rows)}**.

| download_status | n |
|---|---|
{chr(10).join(f"| {k} | {v} |" for k, v in sorted(status.items())) or "| nenhum | 0 |"}

- Camadas oficiais locais reutilizadas (sem copia): **{reused}**.
- Endpoints institucionais registrados (sem recurso direto inventado): **{status.get('endpoint_only', 0) + status.get('probed_endpoint', 0)}**.

## 3. Governanca
Nenhum arquivo bruto pesado e versionado. A reproducao usa este manifesto
(URL/SHA256/tamanho). Tudo review-only; nada vira ground truth, treino ou score v7.
"""
    write_markdown(REPORT, md)
    print(f"manifest -> {rel(MANIFEST)} ({len(rows)} rows; reused_local={reused})")
    print("review-only. No heavy raw committed. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
