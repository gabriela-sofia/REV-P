from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPAIR_CANDIDATES = Path("outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv")
REPAIR_METRICS = Path("outputs_public/metrics/revp_temporal_metadata_repair_candidates_mv1.json")
BACKFILL_TABLE = Path("outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv")
BACKFILL_METRICS = Path("outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json")
READINESS_TABLE = Path("outputs_public/tables/revp_temporal_asset_readiness_mv1.csv")
READINESS_METRICS = Path("outputs_public/metrics/revp_temporal_asset_readiness_mv1.json")

ASSET_OUTPUT = Path("outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv")
COVERAGE_OUTPUT = Path("outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv")
METRICS_OUTPUT = Path("outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json")
REPORT_OUTPUT = Path("outputs_public/execution_reports/revp_normalized_temporal_asset_manifest_mv1.md")

ASSET_COLUMNS = [
    "normalized_asset_id",
    "canonical_patch_id",
    "asset_id",
    "region",
    "asset_type",
    "acquisition_date",
    "cloud_cover",
    "cloud_cover_required",
    "cloud_cover_available",
    "patch_asset_link_status",
    "temporal_usability_status",
    "is_clean_temporal_asset",
    "source_files",
    "applied_repair_fields",
    "applied_repair_rules",
    "applied_repair_confidences",
    "rejected_repair_count",
    "lineage_status",
    "guardrail_status",
]

COVERAGE_COLUMNS = [
    "canonical_patch_id",
    "region",
    "normalized_asset_count",
    "clean_temporal_asset_count",
    "unique_acquisition_dates_count",
    "unique_acquisition_dates",
    "s2_clean_asset_count",
    "s1_clean_asset_count",
    "dinov2_clean_asset_count",
    "has_1_date",
    "has_2_dates",
    "has_3plus_dates",
    "eligible_for_temporal_embedding_drift",
    "temporal_coverage_status",
    "remaining_blockers",
    "next_required_action",
]

ALLOWED_CONFIDENCE = {
    "HIGH_EXPLICIT_FIELD",
    "HIGH_UNIQUE_REGISTRY_MATCH",
    "MEDIUM_UNAMBIGUOUS_FILENAME_PATTERN",
}
MISSING_VALUES = {"", "unknown", "missing", "ausente", "desconhecido", "none", "null", "nan", "n/a", "na"}
TEMPORAL_ASSET_TYPES = {"sentinel_2_optical", "sentinel_1_sar", "dinov2_embedding"}
NON_TEMPORAL_ASSET_TYPES = {"patch_manifest", "external_evidence"}

GUARDRAILS = [
    "manifesto derivado em modo shadow",
    "sem alterar manifestos originais",
    "sem download automatico",
    "sem STAC ou API",
    "sem calculo de drift",
    "sem treino",
    "sem rotulo",
    "sem positivos ou negativos formais",
    "sem classe-zero",
    "sem GT operacional",
    "sem tabela multimodal de atributos",
]


@dataclass
class AssetAccumulator:
    canonical_patch_id: str
    asset_id: str
    values: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    source_files: set[str] = field(default_factory=set)
    applied_fields: set[str] = field(default_factory=set)
    applied_rules: set[str] = field(default_factory=set)
    applied_confidences: set[str] = field(default_factory=set)
    rejected_count: int = 0
    applied_count: int = 0
    lineage: list[str] = field(default_factory=list)
    ambiguous_fields: set[str] = field(default_factory=set)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Build derived normalized temporal asset manifest for REV-P MV1.")
    parser.add_argument("--repo-root", default=".", help="REV-P repository root.")
    return parser


def is_present(value: Any) -> bool:
    return str(value).strip().lower() not in MISSING_VALUES


def parse_float(value: Any) -> float | None:
    if not is_present(value):
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def candidate_is_applied(row: dict[str, str]) -> bool:
    return (
        row.get("is_applicable") == "true"
        and row.get("is_deterministic") == "true"
        and row.get("repair_confidence") in ALLOWED_CONFIDENCE
        and is_present(row.get("normalized_value", ""))
        and row.get("rejection_reason") in {"", "NONE"}
        and not row.get("repair_rule", "").startswith("REJECT")
    )


def normalized_asset_key(row: dict[str, str]) -> tuple[str, str]:
    patch_id = row.get("canonical_patch_id", "").strip()
    asset_id = row.get("asset_id", "").strip() or "unknown"
    return patch_id, asset_id


def accumulate_candidates(rows: list[dict[str, str]]) -> dict[tuple[str, str], AssetAccumulator]:
    assets: dict[tuple[str, str], AssetAccumulator] = {}
    for row in rows:
        key = normalized_asset_key(row)
        patch_id, asset_id = key
        if not is_present(patch_id):
            patch_id = "unknown"
        if not is_present(asset_id):
            asset_id = "unknown"
        asset = assets.setdefault(key, AssetAccumulator(patch_id, asset_id))
        if is_present(row.get("source_file", "")):
            asset.source_files.add(row["source_file"])
        if not candidate_is_applied(row):
            asset.rejected_count += 1
            if row.get("rejection_reason") == "AMBIGUOUS_CONFLICT":
                asset.ambiguous_fields.add(row.get("repair_field", "unknown"))
            continue
        field = row["repair_field"]
        value = row["normalized_value"].strip()
        asset.values[field].add(value)
        asset.applied_fields.add(field)
        asset.applied_rules.add(row.get("repair_rule", "unknown"))
        asset.applied_confidences.add(row.get("repair_confidence", "unknown"))
        asset.applied_count += 1
        asset.lineage.append(
            f"{row.get('source_file', 'unknown')}::{row.get('source_field', 'unknown')}::{field}::{row.get('repair_rule', 'unknown')}"
        )
    for asset in assets.values():
        for field, values in asset.values.items():
            comparable = {value for value in values if is_present(value)}
            if field in {"region", "asset_type", "acquisition_date", "cloud_cover"} and len(comparable) > 1:
                asset.ambiguous_fields.add(field)
    return assets


def single_value(asset: AssetAccumulator, field: str) -> str:
    values = sorted(value for value in asset.values.get(field, set()) if is_present(value))
    if len(values) == 1:
        return values[0]
    return "unknown"


def link_is_valid(asset: AssetAccumulator) -> bool:
    if not is_present(asset.canonical_patch_id) or not is_present(asset.asset_id):
        return False
    if asset.asset_id == "unknown":
        return False
    links = asset.values.get("patch_asset_link", set())
    return any(is_present(link) and asset.canonical_patch_id in link and asset.asset_id in link for link in links)


def classify_usability(asset: AssetAccumulator) -> tuple[str, bool]:
    if asset.ambiguous_fields:
        return "BLOCKED_AMBIGUOUS_REPAIR", False
    if not is_present(asset.canonical_patch_id):
        return "BLOCKED_MISSING_PATCH_ID", False
    region = single_value(asset, "region")
    if not is_present(region):
        return "BLOCKED_MISSING_REGION", False
    asset_type = single_value(asset, "asset_type")
    if not is_present(asset_type):
        return "BLOCKED_MISSING_ASSET_TYPE", False
    if asset_type == "unknown_asset_type":
        return "BLOCKED_UNKNOWN_ASSET_TYPE", False
    if asset_type in NON_TEMPORAL_ASSET_TYPES:
        return "NOT_TEMPORAL_ASSET", False
    if not link_is_valid(asset):
        return "BLOCKED_MISSING_PATCH_ID", False
    if not is_present(single_value(asset, "acquisition_date")):
        return "BLOCKED_MISSING_ACQUISITION_DATE", False
    if asset_type == "sentinel_2_optical" and parse_float(single_value(asset, "cloud_cover")) is None:
        return "BLOCKED_MISSING_S2_CLOUD_COVER", False
    if asset_type in TEMPORAL_ASSET_TYPES:
        return "CLEAN_TEMPORAL_ASSET", True
    return "NOT_TEMPORAL_ASSET", False


def build_asset_rows(assets: dict[tuple[str, str], AssetAccumulator]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, asset in sorted(assets.items(), key=lambda item: (item[0][0], item[0][1])):
        region = single_value(asset, "region")
        asset_type = single_value(asset, "asset_type")
        acquisition_date = single_value(asset, "acquisition_date")
        cloud_cover = single_value(asset, "cloud_cover")
        cloud_required = asset_type == "sentinel_2_optical"
        cloud_available = parse_float(cloud_cover) is not None
        usability, clean = classify_usability(asset)
        rows.append(
            {
                "normalized_asset_id": f"{asset.canonical_patch_id}::{asset.asset_id}",
                "canonical_patch_id": asset.canonical_patch_id,
                "asset_id": asset.asset_id,
                "region": region,
                "asset_type": asset_type,
                "acquisition_date": acquisition_date,
                "cloud_cover": cloud_cover if cloud_available else "missing",
                "cloud_cover_required": str(cloud_required).lower(),
                "cloud_cover_available": str(cloud_available).lower(),
                "patch_asset_link_status": "LINKED" if link_is_valid(asset) else "MISSING_OR_INVALID",
                "temporal_usability_status": usability,
                "is_clean_temporal_asset": str(clean).lower(),
                "source_files": "|".join(sorted(asset.source_files)) if asset.source_files else "unknown",
                "applied_repair_fields": "|".join(sorted(asset.applied_fields)) if asset.applied_fields else "none",
                "applied_repair_rules": "|".join(sorted(asset.applied_rules)) if asset.applied_rules else "none",
                "applied_repair_confidences": "|".join(sorted(asset.applied_confidences)) if asset.applied_confidences else "none",
                "rejected_repair_count": str(asset.rejected_count),
                "lineage_status": "APPLIED_DETERMINISTIC_SHADOW" if asset.applied_count else "NO_APPLIED_REPAIR",
                "guardrail_status": "GUARDRAILS_PRESERVED_SHADOW_MANIFEST_ONLY",
            }
        )
    return rows


def coverage_status(clean_count: int, date_count: int, asset_count: int, blockers: set[str]) -> tuple[str, str]:
    if date_count >= 3:
        return "READY_FOR_TEMPORAL_DRIFT", "NO_DRIFT_ACTION_IN_THIS_STEP"
    if date_count == 2:
        return "NEAR_READY_NEEDS_1_DATE", "ADD_ONE_CLEAN_TEMPORAL_ACQUISITION"
    if date_count == 1:
        return "PARTIAL_NEEDS_2_DATES", "ADD_TWO_CLEAN_TEMPORAL_ACQUISITIONS"
    if asset_count and clean_count == 0:
        return "HAS_ASSETS_BUT_NO_CLEAN_DATE", "REPAIR_ACQUISITION_DATE_AND_REQUIRED_METADATA"
    if blockers:
        return "METADATA_REPAIR_STILL_REQUIRED", "CONTINUE_DETERMINISTIC_METADATA_REPAIR"
    return "MANUAL_REVIEW_REQUIRED", "MANUAL_REVIEW_REQUIRED"


def build_coverage_rows(asset_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in asset_rows:
        grouped[row["canonical_patch_id"]].append(row)
    coverage_rows: list[dict[str, str]] = []
    for patch_id in sorted(grouped):
        rows = grouped[patch_id]
        clean_rows = [row for row in rows if row["is_clean_temporal_asset"] == "true"]
        dates = sorted({row["acquisition_date"] for row in clean_rows if is_present(row["acquisition_date"])})
        regions = sorted({row["region"] for row in rows if is_present(row["region"])})
        blockers = {row["temporal_usability_status"] for row in rows if row["temporal_usability_status"] != "CLEAN_TEMPORAL_ASSET"}
        date_count = len(dates)
        status, action = coverage_status(len(clean_rows), date_count, len(rows), blockers)
        coverage_rows.append(
            {
                "canonical_patch_id": patch_id,
                "region": "|".join(regions) if regions else "unknown",
                "normalized_asset_count": str(len(rows)),
                "clean_temporal_asset_count": str(len(clean_rows)),
                "unique_acquisition_dates_count": str(date_count),
                "unique_acquisition_dates": ";".join(dates) if dates else "unknown",
                "s2_clean_asset_count": str(sum(1 for row in clean_rows if row["asset_type"] == "sentinel_2_optical")),
                "s1_clean_asset_count": str(sum(1 for row in clean_rows if row["asset_type"] == "sentinel_1_sar")),
                "dinov2_clean_asset_count": str(sum(1 for row in clean_rows if row["asset_type"] == "dinov2_embedding")),
                "has_1_date": str(date_count == 1).lower(),
                "has_2_dates": str(date_count == 2).lower(),
                "has_3plus_dates": str(date_count >= 3).lower(),
                "eligible_for_temporal_embedding_drift": str(date_count >= 3).lower(),
                "temporal_coverage_status": status,
                "remaining_blockers": "|".join(sorted(blockers)) if blockers else "NONE",
                "next_required_action": action,
            }
        )
    return coverage_rows


def normalized_status(coverage_rows: list[dict[str, str]]) -> str:
    eligible = sum(1 for row in coverage_rows if row["eligible_for_temporal_embedding_drift"] == "true")
    one_or_two = sum(1 for row in coverage_rows if row["has_1_date"] == "true" or row["has_2_dates"] == "true")
    if eligible >= 30:
        return "STEP_A_NORMALIZED_MANIFEST_READY_FOR_TEMPORAL_DRIFT"
    if 20 <= eligible <= 29:
        return "STEP_A_NORMALIZED_MANIFEST_READY_FOR_PILOT_DRIFT"
    if 1 <= eligible <= 19:
        return "STEP_A_NORMALIZED_MANIFEST_LIMITED_PILOT_ONLY"
    if eligible == 0 and one_or_two:
        return "STEP_A_NORMALIZED_MANIFEST_REQUIRES_ADDITIONAL_ACQUISITIONS"
    return "STEP_A_NORMALIZED_MANIFEST_BLOCKED_BY_METADATA"


def build_metrics(asset_rows: list[dict[str, str]], coverage_rows: list[dict[str, str]]) -> dict[str, Any]:
    blockers = Counter(
        blocker
        for row in coverage_rows
        for blocker in row["remaining_blockers"].split("|")
        if blocker and blocker != "NONE"
    )
    return {
        "total_normalized_assets": len(asset_rows),
        "total_clean_temporal_assets": sum(1 for row in asset_rows if row["is_clean_temporal_asset"] == "true"),
        "total_patches_in_coverage": len(coverage_rows),
        "patches_with_1_date": sum(1 for row in coverage_rows if row["has_1_date"] == "true"),
        "patches_with_2_dates": sum(1 for row in coverage_rows if row["has_2_dates"] == "true"),
        "patches_with_3plus_dates": sum(1 for row in coverage_rows if row["has_3plus_dates"] == "true"),
        "patches_eligible_for_drift": sum(
            1 for row in coverage_rows if row["eligible_for_temporal_embedding_drift"] == "true"
        ),
        "counts_by_region": dict(sorted(Counter(row["region"] for row in coverage_rows).items())),
        "counts_by_asset_type": dict(sorted(Counter(row["asset_type"] for row in asset_rows).items())),
        "counts_by_temporal_usability_status": dict(
            sorted(Counter(row["temporal_usability_status"] for row in asset_rows).items())
        ),
        "counts_by_temporal_coverage_status": dict(
            sorted(Counter(row["temporal_coverage_status"] for row in coverage_rows).items())
        ),
        "remaining_blockers": dict(blockers.most_common()),
        "step_a_normalized_manifest_status": normalized_status(coverage_rows),
        "guardrails_preserved": GUARDRAILS,
        "input_files": [
            READINESS_TABLE.as_posix(),
            READINESS_METRICS.as_posix(),
            BACKFILL_TABLE.as_posix(),
            BACKFILL_METRICS.as_posix(),
            REPAIR_CANDIDATES.as_posix(),
            REPAIR_METRICS.as_posix(),
        ],
    }


def branch_name(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("ref: refs/heads/"):
        return text.removeprefix("ref: refs/heads/")
    return text or "unknown"


def build_report(repo_root: Path, metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# revp_normalized_temporal_asset_manifest_mv1",
            "",
            "## 1. Escopo",
            "Manifesto temporal normalizado derivado para a Vertente A, em modo shadow.",
            "",
            "## 2. Relacao com as tres etapas anteriores",
            "- Usa readiness audit, backfill queue e repair candidates como entradas.",
            "",
            "## 3. Por que o manifesto e derivado/shadow",
            "- O manifesto aplica candidatos somente na camada derivada e nao altera manifestos originais.",
            "",
            "## 4. Arquivos de entrada",
            *[f"- `{item}`" for item in metrics["input_files"]],
            "",
            "## 5. Regras de aplicacao de candidatos",
            "- Apenas `is_applicable=true`, `is_deterministic=true` e confianca alta/media permitida.",
            "",
            "## 6. Regras de rejeicao",
            "- Rejeitados, ambiguos, contexto fraco e valores unknown/missing nao sao aplicados.",
            "",
            "## 7. Definicao de asset temporal limpo",
            "- Exige patch, regiao, tipo temporal, data, vinculo patch/asset e cloud cover numerico para Sentinel-2 optico.",
            "",
            "## 8. Cobertura temporal por patch",
            *[f"- {status}: {count}" for status, count in metrics["counts_by_temporal_coverage_status"].items()],
            "",
            "## 9. Contagem de patches com 1, 2 e 3+ datas",
            f"- 1 data: {metrics['patches_with_1_date']}",
            f"- 2 datas: {metrics['patches_with_2_dates']}",
            f"- 3+ datas: {metrics['patches_with_3plus_dates']}",
            "",
            "## 10. Status da Vertente A",
            f"- `step_a_normalized_manifest_status`: `{metrics['step_a_normalized_manifest_status']}`",
            "",
            "## 11. Blockers restantes",
            *[f"- {blocker}: {count}" for blocker, count in metrics["remaining_blockers"].items()],
            "",
            "## 12. Proximos passos permitidos dentro da Vertente A",
            "- Curar aquisicoes adicionais e reparar metadados determinísticos restantes.",
            "",
            "## 13. Limitacoes",
            "- Esta etapa nao calcula drift temporal e nao mede acuracia.",
            "",
            "## 14. Guardrails preservados",
            f"- Branch auditada: `{branch_name(repo_root)}`",
            *[f"- {item}" for item in GUARDRAILS],
            "",
        ]
    )


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    rows = read_csv(root / REPAIR_CANDIDATES)
    assets = accumulate_candidates(rows)
    asset_rows = build_asset_rows(assets)
    coverage_rows = build_coverage_rows(asset_rows)
    metrics = build_metrics(asset_rows, coverage_rows)

    write_csv(root / ASSET_OUTPUT, asset_rows, ASSET_COLUMNS)
    write_csv(root / COVERAGE_OUTPUT, coverage_rows, COVERAGE_COLUMNS)
    write_json(root / METRICS_OUTPUT, metrics)
    (root / REPORT_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_OUTPUT).write_text(build_report(root, metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    execute(Path(args.repo_root))


if __name__ == "__main__":
    main()
