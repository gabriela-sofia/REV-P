"""MV2 DATA-06/07 real acquisition queue builder.

When no real local inputs exist yet, this builds *operational* public queues that
tell a human reviewer exactly what to acquire to unblock the metadata-only front:
real temporal windows (with a traceable official source) and real sensor lineage
(asset_ref -> source_asset_ref linking to Sentinel-2). It never resolves a window
by bbox, never infers a sensor by filename, never marks Sentinel-2 without a
source and never promotes DINO/PNG/NPZ to a raster. Reports are PT-BR.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_06_09_real_acquisition_queue"
TEMPORAL_CSV = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_promotion" / "mv2_data_06_temporal_window_promotion.csv"
LINEAGE_CSV = PROJECT_ROOT / "outputs_public" / "mv2_data_source_sensor_lineage_promotion" / "mv2_data_07_sensor_lineage_promotion.csv"
GEOMETRY_REGISTRY = PROJECT_ROOT / "datasets" / "v2av_patch_boundary_geometry_registry.csv"

CITY_BY_PREFIX = {"REC": "Recife", "PET": "Petropolis", "CUR": "Curitiba"}

ACCEPTED_TEMPORAL_SOURCES = [
    "CEMADEN",
    "Defesa Civil",
    "Copernicus EMS/CEMS",
    "SGB/CPRM",
    "ANA",
    "boletim municipal oficial",
    "artigo/publicacao cientifica",
    "registro interno auditavel com referencia",
]

ACCEPTED_LINEAGE_SOURCES = [
    "historico/export GEE",
    "manifest de asset original",
    "script de export",
    "metadata oficial Sentinel",
    "registro interno auditavel ligando asset_ref a source_asset_ref",
]

SUGGESTED_TEMPORAL_BY_CITY = {
    "Recife": "Copernicus EMS / APAC / Defesa Civil PE / CEMADEN",
    "Petropolis": "Copernicus EMS / Defesa Civil RJ / SGB-CPRM / CEMADEN",
    "Curitiba": "Defesa Civil PR / SIMEPAR / CEMADEN",
}

QUERY_HINT_BY_CITY = {
    "Recife": "Identificar a janela temporal do evento de alagamento/inundacao em Recife (ex.: maio/2022) e registrar inicio/fim com boletim oficial rastreavel.",
    "Petropolis": "Identificar a janela temporal do evento de deslizamento/chuva intensa em Petropolis (ex.: fev/2022) e registrar inicio/fim com boletim oficial rastreavel.",
    "Curitiba": "Identificar a janela temporal do evento de alagamento urbano em Curitiba e registrar inicio/fim com boletim oficial rastreavel.",
}

DATA06_FIELDS = [
    "target_rank",
    "patch_id",
    "asset_id",
    "city",
    "bbox",
    "crs",
    "needed_field",
    "accepted_source_family",
    "suggested_source",
    "query_hint_ptbr",
    "source_ref_required",
    "reviewer_action",
    "status",
]

DATA07_FIELDS = [
    "target_rank",
    "patch_id",
    "asset_id",
    "slot_id",
    "evidence_id",
    "needed_field",
    "accepted_source_family",
    "source_asset_ref_required",
    "sensor_source_ref_required",
    "reviewer_action",
    "status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def city_for_patch(patch_id: str) -> str:
    return CITY_BY_PREFIX.get((patch_id or "")[:3], "")


def _geometry_lookup(registry_path: Path = GEOMETRY_REGISTRY) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in read_csv(registry_path):
        lookup[row.get("patch_id", "")] = row
    return lookup


def _bbox_crs_for(patch_id: str, geometry: dict[str, dict[str, str]]) -> tuple[str, str]:
    """Return (bbox, crs) only from real local geometry; never invented."""
    row = geometry.get(patch_id, {})
    if (row.get("is_valid_geometry") or "").lower() == "true":
        minx, miny, maxx, maxy = (row.get("bbox_minx", ""), row.get("bbox_miny", ""), row.get("bbox_maxx", ""), row.get("bbox_maxy", ""))
        if all([minx, miny, maxx, maxy]):
            return f"{minx},{miny},{maxx},{maxy}", row.get("crs", "")
    return "", ""


def build_data06_queue(max_targets: int = 10) -> list[dict[str, Any]]:
    temporal = read_csv(TEMPORAL_CSV)
    geometry = _geometry_lookup()
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(temporal[: max(0, max_targets)], 1):
        patch_id = row.get("patch_id", "")
        city = city_for_patch(patch_id)
        bbox, crs = _bbox_crs_for(patch_id, geometry)
        rows.append(
            {
                "target_rank": rank,
                "patch_id": patch_id,
                "asset_id": row.get("asset_id", ""),
                "city": city,
                "bbox": bbox,
                "crs": crs,
                "needed_field": "temporal_window_start+temporal_window_end+temporal_window_source+source_ref",
                "accepted_source_family": " | ".join(ACCEPTED_TEMPORAL_SOURCES),
                "suggested_source": SUGGESTED_TEMPORAL_BY_CITY.get(city, "fonte oficial rastreavel"),
                "query_hint_ptbr": QUERY_HINT_BY_CITY.get(city, "Identificar janela temporal do evento com fonte oficial rastreavel."),
                "source_ref_required": "sim",
                "reviewer_action": "Preencher inputs_local/data_06_temporal_windows/ com janela temporal + fonte rastreavel (sem inventar data).",
                "status": "PENDING_REAL_ACQUISITION",
            }
        )
    return rows


def build_data07_queue(max_targets: int = 10) -> list[dict[str, Any]]:
    lineage = read_csv(LINEAGE_CSV)
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(lineage[: max(0, max_targets)], 1):
        rows.append(
            {
                "target_rank": rank,
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "slot_id": row.get("slot_id", ""),
                "evidence_id": row.get("evidence_id", ""),
                "needed_field": "sensor_family=SENTINEL_2+sensor_source_ref",
                "accepted_source_family": " | ".join(ACCEPTED_LINEAGE_SOURCES),
                "source_asset_ref_required": "sim",
                "sensor_source_ref_required": "sim",
                "reviewer_action": "Preencher inputs_local/data_07_sensor_lineage/ ligando asset_ref a source_asset_ref Sentinel-2 (sem inferir por nome/visual).",
                "status": "PENDING_REAL_ACQUISITION",
            }
        )
    return rows


def write_data08_checklist(out_dir: Path = OUT_DIR) -> None:
    write_text(
        out_dir / "mv2_data_08_metadata_config_checklist.md",
        """# DATA-08 - Checklist de aquisicao de config metadata-only

Crie LOCALMENTE (nunca versione) `configs/api_config.local.json`:

```json
{
  "allow_network": true,
  "allow_metadata_calls": true,
  "allow_raster_download": false,
  "allow_canary_download": false,
  "providers": {
    "GEE": {"enabled": false, "project_id_env": "REV_P_GEE_PROJECT_ID"},
    "CDSE_STAC": {"enabled": false},
    "CDSE_ODATA": {"enabled": false}
  }
}
```

## Regras
- [ ] `allow_network=true` e `allow_metadata_calls=true`.
- [ ] `allow_raster_download=false` e `allow_canary_download=false` (frente metadata-only).
- [ ] Segredos somente em variaveis de ambiente; nunca no arquivo versionado.
- [ ] Nunca versionar `configs/api_config.local.json`, `.env`, token ou credencial.

Enquanto este arquivo nao existir localmente, DATA-08 fica `BLOCKED_NO_CONFIG`
e o motor permanece em replay-only.
""",
    )


def build_summary(d06: list[dict[str, Any]], d07: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "DATA-06/07/08/09 real acquisition queue",
        "data_06_targets": len(d06),
        "data_07_targets": len(d07),
        "data_06_with_local_bbox": sum(1 for row in d06 if row.get("bbox")),
        "accepted_temporal_sources": ACCEPTED_TEMPORAL_SOURCES,
        "accepted_lineage_sources": ACCEPTED_LINEAGE_SOURCES,
        "auto_resolution_by_bbox": False,
        "auto_resolution_by_filename": False,
        "live_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_outputs(out_dir: Path = OUT_DIR, max_targets: int = 10) -> dict[str, Any]:
    d06 = build_data06_queue(max_targets)
    d07 = build_data07_queue(max_targets)
    write_csv(out_dir / "mv2_data_06_temporal_window_acquisition_queue.csv", DATA06_FIELDS, d06)
    write_csv(out_dir / "mv2_data_07_sensor_lineage_acquisition_queue.csv", DATA07_FIELDS, d07)
    write_data08_checklist(out_dir)
    summary = build_summary(d06, d07)
    write_json(out_dir / "mv2_data_06_09_real_acquisition_summary.json", summary)
    write_text(
        out_dir / "mv2_data_06_09_real_acquisition_report.md",
        f"""# Fila de aquisicao real DATA-06/07/08/09

## Estado
- targets DATA-06 (janela temporal): {summary['data_06_targets']}
- targets DATA-07 (sensor lineage): {summary['data_07_targets']}
- targets DATA-06 com bbox local disponivel: {summary['data_06_with_local_bbox']}

## DATA-06 - Janela temporal
Fontes aceitas: {', '.join(ACCEPTED_TEMPORAL_SOURCES)}.
Acao: preencher `inputs_local/data_06_temporal_windows/` com janela inicio/fim e
fonte oficial rastreavel. Nunca inventar data; nunca resolver por bbox.

## DATA-07 - Sensor lineage
Fontes aceitas: {', '.join(ACCEPTED_LINEAGE_SOURCES)}.
Acao: preencher `inputs_local/data_07_sensor_lineage/` ligando `asset_ref` a
`source_asset_ref` Sentinel-2. Nunca inferir sensor por nome/visual; nunca marcar
Sentinel-2 sem fonte.

## DATA-08 - Config metadata-only
Ver `mv2_data_08_metadata_config_checklist.md`. Criar `configs/api_config.local.json`
local (nunca versionar) habilitando apenas network + metadata calls.

## Garantias
- nenhuma resolucao automatica por bbox ou filename;
- DINO/PNG/NPZ nao sao promovidos a raster;
- chamadas/downloads/rasters/crops: 0/0/0/0.
""",
    )
    write_text(out_dir / "commands.txt", "python scripts/mv2_data_06_09_real_acquisition_queue.py")
    return summary


def write_schema(project_root: Path = PROJECT_ROOT) -> None:
    write_json(
        project_root / "datasets" / "schemas" / "schema_mv2_data_06_09_real_acquisition_queue.json",
        {
            "schema_id": "schema_mv2_data_06_09_real_acquisition_queue",
            "data_06_fields": DATA06_FIELDS,
            "data_07_fields": DATA07_FIELDS,
            "accepted_temporal_sources": ACCEPTED_TEMPORAL_SOURCES,
            "accepted_lineage_sources": ACCEPTED_LINEAGE_SOURCES,
            "rules": {
                "no_auto_resolution_by_bbox": True,
                "no_auto_resolution_by_filename": True,
                "no_sentinel_2_without_source": True,
                "no_dino_png_npz_as_raster": True,
            },
            "side_effects": {"live_calls": 0, "downloads": 0, "rasters": 0, "crops": 0},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-targets", type=int, default=10)
    args = parser.parse_args(argv)
    write_schema(PROJECT_ROOT)
    summary = write_outputs(OUT_DIR, args.max_targets)
    print(
        "[mv2_data_06_09_real_acquisition_queue] "
        f"DATA-06 fila={summary['data_06_targets']} DATA-07 fila={summary['data_07_targets']} "
        "calls/downloads/rasters/crops=0/0/0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
