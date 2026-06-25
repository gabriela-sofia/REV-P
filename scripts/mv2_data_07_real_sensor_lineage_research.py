"""MV2 DATA-07 real sensor lineage research.

Resolves ``asset_ref -> source_asset_ref -> sensor_family`` for the same queue
targets, reading committed internal registries only (no network, no raster read):

  - ``v2at_event_patch_package_registry.csv`` (sentinel_sensor_family, obs date)
  - ``dino_sentinel_input_manifest_v1fu.csv`` (source_asset_type, asset path)
  - ``v2bd_patch_asset_lineage_registry.csv`` (asset_sensor, asset_type)

Rules enforced: only ``SENTINEL_2`` may be ``spectral_eligible``; ``SENTINEL_1``
is ``support_only``; DINO/PNG/NPZ/UNKNOWN block; the sensor is never inferred from
a visual name; absolute local paths are redacted; without proof the target stays
``UNKNOWN_BLOCKED``. A local candidate is written only if a STRONG complete
lineage (family + explicit source ref) is found.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_07_real_sensor_lineage_research"
QUEUE_CSV = PROJECT_ROOT / "outputs_public" / "mv2_data_06_09_real_acquisition_queue" / "mv2_data_07_sensor_lineage_acquisition_queue.csv"
PACKAGE_REGISTRY = PROJECT_ROOT / "datasets" / "v2at_event_patch_package_registry.csv"
V1FU_MANIFEST = PROJECT_ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
V2BD_LINEAGE = PROJECT_ROOT / "datasets" / "v2bd_patch_asset_lineage_registry.csv"
LOCAL_CANDIDATE = PROJECT_ROOT / "inputs_local" / "data_07_sensor_lineage" / "data_07_sensor_lineage_real_candidate.csv"

CANDIDATE_FIELDS = [
    "target_rank",
    "patch_id",
    "asset_id",
    "asset_ref",
    "source_asset_ref",
    "sensor_family",
    "sensor_source_ref_public_or_redacted",
    "source_type",
    "spectral_eligible",
    "support_only",
    "evidence_strength",
    "review_status",
    "blocked_reason",
]

QUERY_PACK_FIELDS = [
    "target_rank",
    "patch_id",
    "asset_id",
    "objective",
    "accepted_source_family",
    "query_string",
    "what_to_confirm",
]

LOCAL_FIELDS = ["patch_id", "asset_id", "sensor_family", "sensor_source_ref", "review_status"]


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


def normalize_sensor_family(raw: str) -> str:
    """Map committed registry sensor labels to canonical families. Never visual."""
    value = (raw or "").strip().upper()
    if value in {"SENTINEL2_MSI", "SENTINEL_2", "SENTINEL-2", "S2", "S2_MSI"}:
        return "SENTINEL_2"
    if value in {"SENTINEL1", "SENTINEL_1", "SENTINEL-1", "S1", "S1_SAR"}:
        return "SENTINEL_1"
    return "UNKNOWN"


def _lookup(path: Path, key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        out.setdefault(row.get(key, ""), row)
    return out


def _redact_asset_path(path_ref: str) -> str:
    """Keep only the repo-relative sentinel asset reference, never an absolute path."""
    value = (path_ref or "").strip().replace("\\", "/")
    if not value:
        return ""
    if "data/sentinel/" in value:
        return "<repo_local>/data/sentinel/" + value.split("data/sentinel/")[-1]
    return "<redacted_asset_path>"


def research_targets(
    queue_path: Path = QUEUE_CSV,
    package_path: Path = PACKAGE_REGISTRY,
    v1fu_path: Path = V1FU_MANIFEST,
    v2bd_path: Path = V2BD_LINEAGE,
) -> list[dict[str, Any]]:
    queue = read_csv(queue_path)
    packages = _lookup(package_path, "patch_id")
    v1fu = _lookup(v1fu_path, "source_asset_id")
    v2bd = _lookup(v2bd_path, "candidate_asset_id")
    rows: list[dict[str, Any]] = []
    for entry in queue:
        patch_id = entry.get("patch_id", "")
        asset_id = entry.get("asset_id", "")
        pkg = packages.get(patch_id, {})
        manifest = v1fu.get(asset_id, {})
        bd = v2bd.get(asset_id, {})

        family = normalize_sensor_family(pkg.get("sentinel_sensor_family", ""))
        source_type = manifest.get("source_asset_type", "") or bd.get("asset_type", "")
        asset_path = _redact_asset_path(manifest.get("asset_path_reference", "") or bd.get("asset_file", ""))
        manifest_id = manifest.get("dino_input_id", "")
        obs_date = pkg.get("sentinel_observation_date", "")

        # Explicit S2 product/scene id is the gold standard sensor_source_ref.
        explicit_product = ""  # v2at sentinel_asset_id is UNKNOWN for these targets

        if family == "SENTINEL_2" and source_type == "SENTINEL_TIF_ASSET":
            # Family proven by committed registry; explicit product id still absent.
            sensor_source_ref = f"v2at:SENTINEL2_MSI(obs {obs_date}); manifest:{manifest_id}; asset:{asset_path}"
            spectral_eligible, support_only = "true", "false"
            if explicit_product:
                strength, review, blocked = "STRONG", "REAL_SENSOR_LINEAGE_CANDIDATE", ""
            else:
                strength, review, blocked = "MEDIUM", "NEEDS_REVIEW", "EXPLICIT_S2_PRODUCT_ID_ABSENT_NEEDS_HUMAN_CONFIRM"
        elif family == "SENTINEL_1":
            sensor_source_ref = f"manifest:{manifest_id}; asset:{asset_path}"
            spectral_eligible, support_only = "false", "true"
            strength, review, blocked = "MEDIUM", "NEEDS_REVIEW", "SENTINEL_1_SUPPORT_ONLY_NOT_SPECTRAL"
        else:
            # SENTINEL_TIF_ASSET without an S2 declaration is NOT inferred as S2.
            sensor_source_ref = (f"manifest:{manifest_id}; asset:{asset_path}" if manifest_id else "")
            spectral_eligible, support_only = "false", "false"
            strength, review, blocked = "WEAK", "UNKNOWN_BLOCKED", "NO_SENTINEL_2_DECLARATION"

        rows.append(
            {
                "target_rank": entry.get("target_rank", ""),
                "patch_id": patch_id,
                "asset_id": asset_id,
                "asset_ref": asset_id,
                "source_asset_ref": asset_path,
                "sensor_family": family,
                "sensor_source_ref_public_or_redacted": sensor_source_ref,
                "source_type": source_type,
                "spectral_eligible": spectral_eligible,
                "support_only": support_only,
                "evidence_strength": strength,
                "review_status": review,
                "blocked_reason": blocked,
            }
        )
    return rows


def build_query_pack(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pack: list[dict[str, Any]] = []
    for row in candidates:
        pack.append(
            {
                "target_rank": row["target_rank"],
                "patch_id": row["patch_id"],
                "asset_id": row["asset_id"],
                "objective": "Resolver product_id/scene_id Sentinel-2 explicito (sensor_source_ref) do asset",
                "accepted_source_family": "historico/export GEE | manifest de asset original | script de export | metadata oficial Sentinel | registro interno auditavel",
                "query_string": f"asset {row['asset_id']} patch {row['patch_id']} export GEE COPERNICUS/S2_SR_HARMONIZED scene_id product_id",
                "what_to_confirm": "product_id/scene_id Sentinel-2 explicito + sensor_family confirmado (sem inferir por nome visual)",
            }
        )
    return pack


def write_local_candidate_if_strong(candidates: list[dict[str, Any]], dest: Path = LOCAL_CANDIDATE) -> bool:
    strong = [row for row in candidates if row["evidence_strength"] == "STRONG" and row["sensor_family"] == "SENTINEL_2"]
    if not strong:
        return False
    local_rows = [
        {
            "patch_id": row["patch_id"],
            "asset_id": row["asset_id"],
            "sensor_family": row["sensor_family"],
            "sensor_source_ref": row["sensor_source_ref_public_or_redacted"],
            "review_status": "REAL_SENSOR_LINEAGE_CANDIDATE",
        }
        for row in strong
    ]
    write_csv(dest, LOCAL_FIELDS, local_rows)
    return True


def summarize(candidates: list[dict[str, Any]], local_created: bool) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_review: dict[str, int] = {}
    for row in candidates:
        by_family[row["sensor_family"]] = by_family.get(row["sensor_family"], 0) + 1
        by_review[row["review_status"]] = by_review.get(row["review_status"], 0) + 1
    return {
        "stage": "DATA-07 real sensor lineage research",
        "targets": len(candidates),
        "sentinel_2_family_documented": by_family.get("SENTINEL_2", 0),
        "by_sensor_family": by_family,
        "by_review_status": by_review,
        "strong_complete_lineage": sum(1 for r in candidates if r["evidence_strength"] == "STRONG"),
        "local_candidate_created": local_created,
        "live_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_outputs(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    candidates = research_targets()
    query_pack = build_query_pack(candidates)
    local_created = write_local_candidate_if_strong(candidates)
    write_csv(out_dir / "mv2_data_07_real_sensor_lineage_candidates.csv", CANDIDATE_FIELDS, candidates)
    write_csv(out_dir / "mv2_data_07_real_sensor_lineage_query_pack.csv", QUERY_PACK_FIELDS, query_pack)
    summary = summarize(candidates, local_created)
    write_json(out_dir / "mv2_data_07_real_sensor_lineage_summary.json", summary)
    write_text(
        out_dir / "mv2_data_07_real_sensor_lineage_report.md",
        f"""# DATA-07 - pesquisa de lineage sensorial real

## Estado
- targets investigados: {summary['targets']}
- familia Sentinel-2 documentada (committed): {summary['sentinel_2_family_documented']}
- distribuicao por familia: {json.dumps(summary['by_sensor_family'], ensure_ascii=True)}
- distribuicao por review: {json.dumps(summary['by_review_status'], ensure_ascii=True)}
- lineage completo forte: {summary['strong_complete_lineage']}
- input local criado (nao versionado): {summary['local_candidate_created']}

## Evidencia
- REC (5): v2at declara sentinel_sensor_family=SENTINEL2_MSI (obs 2022-05-24) e v1fu marca
  SENTINEL_TIF_ASSET -> familia Sentinel-2 documentada por registries committed. Falta o
  product_id/scene_id Sentinel-2 explicito (sensor_source_ref) -> MEDIUM / NEEDS_REVIEW.
- PET (5): v1fu marca SENTINEL_TIF_ASSET, mas v2at registra sensor UNKNOWN. Sem declaracao
  Sentinel-2 explicita -> UNKNOWN_BLOCKED (nao inferido por nome de asset).

## Regras respeitadas
- so SENTINEL_2 pode ser spectral_eligible; SENTINEL_1 seria support_only;
- DINO/PNG/NPZ/UNKNOWN bloqueiam; sensor nunca inferido por nome visual;
- paths locais absolutos redigidos; render nunca usado como raster.

## Side effects
- chamadas/downloads/rasters/crops: 0/0/0/0.
""",
    )
    write_text(out_dir / "commands.txt", "python scripts/mv2_data_07_real_sensor_lineage_research.py")
    return summary


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    summary = write_outputs(OUT_DIR)
    print(
        "[mv2_data_07_real_sensor_lineage_research] "
        f"targets={summary['targets']} s2_family={summary['sentinel_2_family_documented']} "
        f"strong_complete={summary['strong_complete_lineage']} "
        f"local_input={summary['local_candidate_created']} "
        "calls/downloads/rasters/crops=0/0/0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
