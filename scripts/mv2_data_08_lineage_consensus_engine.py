"""MV2 DATA-08 Sentinel-2 lineage consensus engine.

Aggregates per-provider :class:`MetadataProviderResult` rows into a single
:class:`MetadataConsensusRecord` per target, following strict, auditable rules.
The engine never downloads, never opens a raster and never invents a product id:
it only reconciles metadata that providers (or replay fixtures) returned.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mv2_data_08_metadata_provider_contracts import (
    NON_CALL_STATUSES,
    OFFICIAL_PROVIDERS,
    MetadataConsensusRecord,
    MetadataProviderResult,
)

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_08_lineage_consensus"

CONSENSUS_STATUSES = [
    "STRONG",
    "MEDIUM_REVIEW",
    "WEAK_BLOCKED",
    "CONFLICT",
    "NO_MATCH",
    "NO_CALL",
]

CONSENSUS_FIELDS = [
    "patch_id",
    "asset_id",
    "consensus_status",
    "product_id",
    "datetime_utc",
    "mgrs_tile",
    "collection",
    "providers_considered",
    "providers_agreeing",
    "conflict_reason",
    "evidence_count",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _has_geometry(result: MetadataProviderResult) -> bool:
    return bool(result.geometry) or bool(result.bbox) or bool(result.odata_geofootprint)


def _date_only(value: str) -> str:
    return (value or "").strip()[:10]


def compute_consensus(
    patch_id: str, asset_id: str, results: Iterable[MetadataProviderResult]
) -> MetadataConsensusRecord:
    """Reconcile one target's provider results into a single consensus record."""
    considered = list(results)
    providers_considered = sorted({r.provider for r in considered})

    executed = [r for r in considered if r.status not in NON_CALL_STATUSES]
    if not executed:
        return MetadataConsensusRecord(
            patch_id=patch_id,
            asset_id=asset_id,
            consensus_status="NO_CALL",
            providers_considered=providers_considered,
            evidence_count=0,
        )

    # A provider may already flag a per-result conflict (e.g. out-of-window).
    if any(r.status == "CONFLICT" for r in executed):
        return MetadataConsensusRecord(
            patch_id=patch_id,
            asset_id=asset_id,
            consensus_status="CONFLICT",
            providers_considered=providers_considered,
            conflict_reason="provider_reported_conflict",
            evidence_count=len(executed),
        )

    matched = [r for r in executed if (r.product_id or "").strip()]
    if not matched:
        return MetadataConsensusRecord(
            patch_id=patch_id,
            asset_id=asset_id,
            consensus_status="NO_MATCH",
            providers_considered=providers_considered,
            evidence_count=0,
        )

    product_ids = {r.product_id.strip() for r in matched}
    datetimes = {_date_only(r.datetime_utc) for r in matched if _date_only(r.datetime_utc)}
    geometries_present = [r for r in matched if _has_geometry(r)]

    # Conflicts: divergent product ids, incompatible datetimes.
    if len(product_ids) > 1:
        return MetadataConsensusRecord(
            patch_id=patch_id,
            asset_id=asset_id,
            consensus_status="CONFLICT",
            providers_considered=providers_considered,
            conflict_reason="divergent_product_id",
            evidence_count=len(matched),
        )
    if len(datetimes) > 1:
        return MetadataConsensusRecord(
            patch_id=patch_id,
            asset_id=asset_id,
            consensus_status="CONFLICT",
            providers_considered=providers_considered,
            conflict_reason="incompatible_datetime",
            evidence_count=len(matched),
        )

    single_pid = next(iter(product_ids))
    official_sources = {r.provider for r in matched if r.provider in OFFICIAL_PROVIDERS}
    has_datetime = bool(datetimes)
    has_geometry = bool(geometries_present)
    collections = {(r.collection or "").strip() for r in matched if (r.collection or "").strip()}
    mgrs = {(r.mgrs_tile or "").strip() for r in matched if (r.mgrs_tile or "").strip()}
    agreeing = sorted({r.provider for r in matched})

    chosen = matched[0]
    base = dict(
        patch_id=patch_id,
        asset_id=asset_id,
        product_id=single_pid,
        datetime_utc=_date_only(chosen.datetime_utc),
        mgrs_tile=(chosen.mgrs_tile or "").strip(),
        collection=(chosen.collection or "").strip(),
        providers_considered=providers_considered,
        providers_agreeing=agreeing,
        evidence_count=len(matched),
    )

    # STRONG: same product id across 2+ official sources, OR a single official
    # source with product id + datetime + geometry.
    if len(official_sources) >= 2 and len(product_ids) == 1:
        return MetadataConsensusRecord(consensus_status="STRONG", **base)
    if len(official_sources) >= 1 and single_pid and has_datetime and has_geometry:
        return MetadataConsensusRecord(consensus_status="STRONG", **base)

    # MEDIUM_REVIEW: datetime + intersects (geometry) + collection + mgrs tile.
    if has_datetime and has_geometry and collections and mgrs:
        return MetadataConsensusRecord(consensus_status="MEDIUM_REVIEW", **base)

    # WEAK_BLOCKED: tile/filename only, name without official source, no geometry.
    return MetadataConsensusRecord(
        consensus_status="WEAK_BLOCKED",
        conflict_reason="tile_or_name_only_or_no_geometry",
        **base,
    )


def consensus_for_targets(
    results_by_target: dict[tuple[str, str], list[MetadataProviderResult]]
) -> list[MetadataConsensusRecord]:
    records: list[MetadataConsensusRecord] = []
    for (patch_id, asset_id), results in results_by_target.items():
        records.append(compute_consensus(patch_id, asset_id, results))
    return records


def summarize(records: list[MetadataConsensusRecord]) -> dict[str, Any]:
    counts = {status: 0 for status in CONSENSUS_STATUSES}
    for record in records:
        counts[record.consensus_status] = counts.get(record.consensus_status, 0) + 1
    return {
        "stage": "DATA-08 lineage consensus",
        "targets": len(records),
        "counts": counts,
        "confirmed_lineage": counts.get("STRONG", 0),
        "conflicts": counts.get("CONFLICT", 0),
        "live_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_outputs(records: list[MetadataConsensusRecord], out_dir: Path = OUT_DIR) -> dict[str, Path]:
    rows = [record.to_row() for record in records]
    csv_path = out_dir / "lineage_consensus.csv"
    summary_path = out_dir / "lineage_consensus_summary.json"
    conflict_path = out_dir / "lineage_conflict_report.md"
    write_csv(csv_path, CONSENSUS_FIELDS, rows)
    summary = summarize(records)
    write_json(summary_path, summary)
    conflicts = [r for r in records if r.consensus_status == "CONFLICT"]
    weak = [r for r in records if r.consensus_status == "WEAK_BLOCKED"]
    lines = ["# Relatorio de conflitos de lineage Sentinel-2", ""]
    lines.append(f"- targets avaliados: {len(records)}")
    lines.append(f"- conflitos: {len(conflicts)}")
    lines.append(f"- weak/blocked: {len(weak)}")
    lines.append("")
    if not conflicts and not weak:
        lines.append("Nenhum conflito ou match fraco registrado (estado limpo / NO_CALL).")
    for record in conflicts:
        lines.append(
            f"- CONFLICT {record.patch_id}/{record.asset_id}: {record.conflict_reason} "
            f"(providers={';'.join(record.providers_considered)})"
        )
    for record in weak:
        lines.append(
            f"- WEAK_BLOCKED {record.patch_id}/{record.asset_id}: {record.conflict_reason}"
        )
    write_text(conflict_path, "\n".join(lines))
    write_text(out_dir / "commands.txt", "python scripts/mv2_data_08_lineage_consensus_engine.py")
    return {"csv": csv_path, "summary": summary_path, "conflict": conflict_path}


def write_schema(project_root: Path = PROJECT_ROOT) -> None:
    write_json(
        project_root / "datasets" / "schemas" / "schema_mv2_data_08_lineage_consensus.json",
        {
            "schema_id": "schema_mv2_data_08_lineage_consensus",
            "required_fields": CONSENSUS_FIELDS,
            "allowed_consensus_status": CONSENSUS_STATUSES,
            "rules": {
                "STRONG": "product_id exato em 2+ fontes oficiais OU fonte oficial unica com product_id+datetime+geometry",
                "MEDIUM_REVIEW": "datetime+intersects+collection+mgrs_tile compativeis",
                "WEAK_BLOCKED": "tile/filename apenas, nome sem fonte oficial ou match sem geometria",
                "CONFLICT": "product_id divergente, datetime incompativel, geometria incompativel ou produto fora da janela",
                "NO_MATCH": "chamadas validas, nenhum item",
                "NO_CALL": "nenhuma chamada executada",
            },
            "side_effects": {"live_calls": 0, "downloads": 0, "rasters": 0, "crops": 0},
        },
    )


def main(argv: list[str] | None = None) -> int:
    write_schema()
    records: list[MetadataConsensusRecord] = []
    write_outputs(records)
    print("[mv2_data_08_lineage_consensus_engine] schema + saidas vazias (NO_CALL)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
