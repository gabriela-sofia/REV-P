from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any


ASSET_MANIFEST = Path("outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv")
PATCH_COVERAGE = Path("outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv")
NORMALIZED_METRICS = Path("outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json")
NORMALIZED_REPORT = Path("outputs_public/execution_reports/revp_normalized_temporal_asset_manifest_mv1.md")

GAP_PLAN_OUTPUT = Path("outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv")
REQUIREMENTS_OUTPUT = Path("outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv")
METRICS_OUTPUT = Path("outputs_public/metrics/revp_temporal_acquisition_gap_plan_mv1.json")
REPORT_OUTPUT = Path("outputs_public/execution_reports/revp_temporal_acquisition_gap_plan_mv1.md")

GAP_PLAN_COLUMNS = [
    "canonical_patch_id",
    "region",
    "current_clean_dates_count",
    "current_clean_dates",
    "missing_dates_to_3",
    "current_clean_temporal_asset_count",
    "s2_clean_asset_count",
    "s1_clean_asset_count",
    "dinov2_clean_asset_count",
    "eligible_for_temporal_embedding_drift_now",
    "acquisition_gap_status",
    "acquisition_priority",
    "is_actionable_for_temporal_backfill",
    "requires_metadata_repair_before_acquisition",
    "remaining_blockers",
    "minimum_new_acquisitions_required",
    "recommended_sensor_family",
    "recommended_processing_contract",
    "next_required_action",
    "guardrail_status",
]

REQUIREMENTS_COLUMNS = [
    "canonical_patch_id",
    "region",
    "required_slot_index",
    "existing_dates",
    "required_new_date_status",
    "required_sensor_family",
    "required_minimum_metadata",
    "requires_same_patch_geometry",
    "requires_same_preprocessing_contract",
    "requires_dinov2_frozen_inference_later",
    "requires_cloud_cover_if_s2",
    "requires_no_label_creation",
    "requires_no_event_assumption",
    "requirement_status",
    "blocking_reason",
]

GUARDRAILS = [
    "sem download de assets",
    "sem consulta STAC API ou GEE",
    "sem gerar URLs de download",
    "sem calcular embedding drift",
    "sem gerar embeddings novos",
    "sem treino",
    "sem rotulo",
    "sem positivos ou negativos formais",
    "sem classe-zero",
    "sem GT operacional",
    "sem tabela multimodal de atributos",
    "sem alterar manifestos originais",
]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Build REV-P MV1 temporal acquisition gap plan.")
    parser.add_argument("--repo-root", default=".", help="REV-P repository root.")
    return parser


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


def parse_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        return 0


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def clean_dates(row: dict[str, str]) -> list[str]:
    raw = str(row.get("unique_acquisition_dates", "")).strip()
    if raw.lower() in {"", "unknown", "missing", "none", "null", "n/a"}:
        return []
    return sorted({item.strip() for item in raw.split(";") if item.strip()})


def build_asset_context(asset_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    context: dict[str, set[str]] = {}
    for row in asset_rows:
        patch_id = row.get("canonical_patch_id", "").strip()
        if not patch_id:
            continue
        context.setdefault(patch_id, set()).add(row.get("asset_type", "unknown_asset_type"))
    return context


def recommended_sensor(row: dict[str, str], asset_context: dict[str, set[str]]) -> str:
    patch_id = row["canonical_patch_id"]
    if parse_int(row.get("s2_clean_asset_count")) > 0:
        return "same_family_as_existing_clean_asset"
    if parse_int(row.get("s1_clean_asset_count")) > 0:
        return "sentinel_1_sar_optional_support"
    if "sentinel_2_optical" in asset_context.get(patch_id, set()):
        return "sentinel_2_optical_preferred"
    return "requires_manual_sensor_selection"


def status_and_priority(row: dict[str, str]) -> tuple[str, str, bool, bool]:
    date_count = parse_int(row.get("unique_acquisition_dates_count"))
    blockers = row.get("remaining_blockers", "")
    region = row.get("region", "unknown").lower()
    clean_assets = parse_int(row.get("clean_temporal_asset_count"))

    if date_count >= 3:
        return "READY_NO_GAP", "P0_NEEDS_1_DATE", False, False
    if date_count == 2:
        return "NEEDS_1_MORE_DATE", "P0_NEEDS_1_DATE", True, False
    if date_count == 1:
        return "NEEDS_2_MORE_DATES", "P1_NEEDS_2_DATES", True, False
    if "BLOCKED_MISSING_PATCH_ID" in blockers:
        return "BLOCKED_BY_MISSING_PATCH_LINKAGE", "P3_REPAIR_METADATA_FIRST", False, True
    if region == "unknown" or "BLOCKED_MISSING_REGION" in blockers:
        return "BLOCKED_BY_UNKNOWN_REGION", "P3_REPAIR_METADATA_FIRST", False, True
    if "BLOCKED_AMBIGUOUS_REPAIR" in blockers:
        return "MANUAL_REVIEW_REQUIRED", "P4_MANUAL_REVIEW", False, True
    if "BLOCKED_MISSING_ASSET_TYPE" in blockers:
        return "BLOCKED_BY_METADATA_REPAIR", "P3_REPAIR_METADATA_FIRST", False, True
    if date_count == 0 and clean_assets > 0:
        return "NEEDS_3_DATES", "P2_NEEDS_3_DATES_WITH_CLEAN_PATCH_CONTEXT", True, False
    if "NOT_TEMPORAL_ASSET" in blockers:
        return "MANUAL_REVIEW_REQUIRED", "P4_MANUAL_REVIEW", False, True
    return "BLOCKED_BY_METADATA_REPAIR", "P3_REPAIR_METADATA_FIRST", False, True


def next_action(status: str) -> str:
    mapping = {
        "READY_NO_GAP": "NO_ACQUISITION_GAP_FOR_THIS_PATCH",
        "NEEDS_1_MORE_DATE": "PLAN_ONE_REAL_TEMPORAL_ACQUISITION_SLOT",
        "NEEDS_2_MORE_DATES": "PLAN_TWO_REAL_TEMPORAL_ACQUISITION_SLOTS",
        "NEEDS_3_DATES": "PLAN_THREE_REAL_TEMPORAL_ACQUISITION_SLOTS",
        "BLOCKED_BY_METADATA_REPAIR": "REPAIR_METADATA_BEFORE_TEMPORAL_ACQUISITION",
        "BLOCKED_BY_MISSING_PATCH_LINKAGE": "REPAIR_PATCH_ASSET_LINKAGE_BEFORE_ACQUISITION",
        "BLOCKED_BY_UNKNOWN_REGION": "REPAIR_REGION_BEFORE_ACQUISITION",
        "MANUAL_REVIEW_REQUIRED": "MANUAL_REVIEW_REQUIRED",
    }
    return mapping[status]


def processing_contract() -> str:
    return "SAME_PATCH_GEOMETRY_SAME_PREPROCESSING_DINOV2_FROZEN_LATER_NO_LABELS"


def build_gap_plan(coverage_rows: list[dict[str, str]], asset_context: dict[str, set[str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in sorted(coverage_rows, key=lambda item: item["canonical_patch_id"]):
        date_count = parse_int(row.get("unique_acquisition_dates_count"))
        missing = max(0, 3 - date_count)
        status, priority, actionable, repair_first = status_and_priority(row)
        if status.startswith("BLOCKED") or status == "MANUAL_REVIEW_REQUIRED":
            minimum = "0"
        else:
            minimum = str(missing)
        output.append(
            {
                "canonical_patch_id": row["canonical_patch_id"],
                "region": row.get("region", "unknown") or "unknown",
                "current_clean_dates_count": str(date_count),
                "current_clean_dates": ";".join(clean_dates(row)) if clean_dates(row) else "unknown",
                "missing_dates_to_3": str(missing),
                "current_clean_temporal_asset_count": row.get("clean_temporal_asset_count", "0"),
                "s2_clean_asset_count": row.get("s2_clean_asset_count", "0"),
                "s1_clean_asset_count": row.get("s1_clean_asset_count", "0"),
                "dinov2_clean_asset_count": row.get("dinov2_clean_asset_count", "0"),
                "eligible_for_temporal_embedding_drift_now": row.get("eligible_for_temporal_embedding_drift", "false"),
                "acquisition_gap_status": status,
                "acquisition_priority": priority,
                "is_actionable_for_temporal_backfill": str(actionable).lower(),
                "requires_metadata_repair_before_acquisition": str(repair_first).lower(),
                "remaining_blockers": row.get("remaining_blockers", "NONE") or "NONE",
                "minimum_new_acquisitions_required": minimum,
                "recommended_sensor_family": recommended_sensor(row, asset_context),
                "recommended_processing_contract": processing_contract(),
                "next_required_action": next_action(status),
                "guardrail_status": "GUARDRAILS_PRESERVED_ACQUISITION_PLAN_ONLY",
            }
        )
    return output


def build_requirements(plan_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    requirements: list[dict[str, str]] = []
    for row in plan_rows:
        slots = parse_int(row["missing_dates_to_3"])
        sensor = row["recommended_sensor_family"]
        status = "ACTIONABLE_REQUIREMENT" if row["is_actionable_for_temporal_backfill"] == "true" else "BLOCKED_REQUIREMENT"
        for slot in range(1, slots + 1):
            requirements.append(
                {
                    "canonical_patch_id": row["canonical_patch_id"],
                    "region": row["region"],
                    "required_slot_index": str(slot),
                    "existing_dates": row["current_clean_dates"],
                    "required_new_date_status": "UNSPECIFIED_REAL_ACQUISITION_NEEDED",
                    "required_sensor_family": sensor,
                    "required_minimum_metadata": "canonical_patch_id|region|asset_type|acquisition_date|patch_asset_link|s2_cloud_cover_when_required",
                    "requires_same_patch_geometry": "true",
                    "requires_same_preprocessing_contract": "true",
                    "requires_dinov2_frozen_inference_later": "true",
                    "requires_cloud_cover_if_s2": str(sensor in {"sentinel_2_optical_preferred", "same_family_as_existing_clean_asset"}).lower(),
                    "requires_no_label_creation": "true",
                    "requires_no_event_assumption": "true",
                    "requirement_status": status,
                    "blocking_reason": "NONE" if status == "ACTIONABLE_REQUIREMENT" else row["remaining_blockers"],
                }
            )
    return requirements


def minimum_slots_for_targets(plan_rows: list[dict[str, str]]) -> dict[str, Any]:
    actionable_slots = sorted(
        parse_int(row["minimum_new_acquisitions_required"])
        for row in plan_rows
        if row["is_actionable_for_temporal_backfill"] == "true"
        and parse_int(row["minimum_new_acquisitions_required"]) > 0
    )

    def estimate(target: int) -> Any:
        if len(actionable_slots) < target:
            return "insufficient_actionable_patches"
        return sum(actionable_slots[:target])

    return {
        "minimum_new_acquisitions_for_1_eligible_patch": estimate(1),
        "minimum_new_acquisitions_for_20_eligible_patches": estimate(20),
        "minimum_new_acquisitions_for_30_eligible_patches": estimate(30),
    }


def acquisition_status(plan_rows: list[dict[str, str]]) -> str:
    if any(row["acquisition_gap_status"] == "NEEDS_1_MORE_DATE" for row in plan_rows):
        return "STEP_A_ACQUISITION_PLAN_READY_FOR_TARGETED_BACKFILL"
    if any(row["acquisition_gap_status"] == "NEEDS_2_MORE_DATES" for row in plan_rows):
        return "STEP_A_ACQUISITION_PLAN_REQUIRES_MULTI_DATE_BACKFILL"
    repair_first = sum(1 for row in plan_rows if row["requires_metadata_repair_before_acquisition"] == "true")
    if plan_rows and repair_first >= len(plan_rows) / 2:
        return "STEP_A_ACQUISITION_PLAN_REQUIRES_METADATA_REPAIR_FIRST"
    if not any(row["is_actionable_for_temporal_backfill"] == "true" for row in plan_rows):
        return "STEP_A_ACQUISITION_PLAN_BLOCKED_MANUAL_REVIEW_REQUIRED"
    return "STEP_A_ACQUISITION_PLAN_REQUIRES_METADATA_REPAIR_FIRST"


def build_metrics(plan_rows: list[dict[str, str]], requirements: list[dict[str, str]]) -> dict[str, Any]:
    date_counts = Counter(row["current_clean_dates_count"] for row in plan_rows)
    slots_by_region: Counter[str] = Counter()
    for req in requirements:
        slots_by_region[req["region"]] += 1
    estimates = minimum_slots_for_targets(plan_rows)
    return {
        "total_patches_evaluated": len(plan_rows),
        "patches_with_0_dates": date_counts.get("0", 0),
        "patches_with_1_date": date_counts.get("1", 0),
        "patches_with_2_dates": date_counts.get("2", 0),
        "patches_with_3plus_dates": sum(count for value, count in date_counts.items() if parse_int(value) >= 3),
        "total_missing_acquisition_slots": len(requirements),
        "missing_slots_by_region": dict(sorted(slots_by_region.items())),
        "counts_by_acquisition_gap_status": dict(sorted(Counter(row["acquisition_gap_status"] for row in plan_rows).items())),
        "counts_by_acquisition_priority": dict(sorted(Counter(row["acquisition_priority"] for row in plan_rows).items())),
        "patches_actionable_for_temporal_backfill": sum(
            1 for row in plan_rows if row["is_actionable_for_temporal_backfill"] == "true"
        ),
        "patches_requiring_repair_before_acquisition": sum(
            1 for row in plan_rows if row["requires_metadata_repair_before_acquisition"] == "true"
        ),
        **estimates,
        "step_a_acquisition_plan_status": acquisition_status(plan_rows),
        "guardrails_preserved": GUARDRAILS,
        "input_files": [
            ASSET_MANIFEST.as_posix(),
            PATCH_COVERAGE.as_posix(),
            NORMALIZED_METRICS.as_posix(),
            NORMALIZED_REPORT.as_posix(),
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
            "# revp_temporal_acquisition_gap_plan_mv1",
            "",
            "## 1. Escopo",
            "Plano publico e auditavel de lacunas de aquisicao temporal para a Vertente A.",
            "",
            "## 2. Relacao com o manifesto temporal normalizado",
            f"- Entrada principal: `{PATCH_COVERAGE.as_posix()}`",
            "",
            "## 3. Por que ainda nao se calcula drift",
            "- Nenhum patch possui 3 ou mais datas limpas no manifesto normalizado.",
            "",
            "## 4. Arquivos de entrada",
            *[f"- `{item}`" for item in metrics["input_files"]],
            "",
            "## 5. Regra de lacuna temporal",
            "- `missing_dates_to_3 = max(0, 3 - datas_limpas_atuais)`.",
            "",
            "## 6. Patches com 0, 1, 2 e 3+ datas",
            f"- 0 datas: {metrics['patches_with_0_dates']}",
            f"- 1 data: {metrics['patches_with_1_date']}",
            f"- 2 datas: {metrics['patches_with_2_dates']}",
            f"- 3+ datas: {metrics['patches_with_3plus_dates']}",
            "",
            "## 7. Slots minimos de aquisicao necessarios",
            f"- Total de slots faltantes: {metrics['total_missing_acquisition_slots']}",
            f"- Para 1 patch elegivel: {metrics['minimum_new_acquisitions_for_1_eligible_patch']}",
            f"- Para 20 patches elegiveis: {metrics['minimum_new_acquisitions_for_20_eligible_patches']}",
            f"- Para 30 patches elegiveis: {metrics['minimum_new_acquisitions_for_30_eligible_patches']}",
            "",
            "## 8. Prioridades de aquisicao",
            *[f"- {key}: {value}" for key, value in metrics["counts_by_acquisition_priority"].items()],
            "",
            "## 9. Requisitos tecnicos por nova aquisicao",
            "- Data real nao especificada ainda.",
            "- Mesma geometria de patch, mesmo contrato de pre-processamento e DINOv2 frozen em etapa posterior.",
            "- Sem criacao de rotulos e sem assumir evento.",
            "",
            "## 10. Estimativa minima para viabilizar piloto de drift",
            f"- `step_a_acquisition_plan_status`: `{metrics['step_a_acquisition_plan_status']}`",
            "",
            "## 11. Blockers restantes",
            *[f"- {key}: {value}" for key, value in metrics["counts_by_acquisition_gap_status"].items()],
            "",
            "## 12. Proximos passos permitidos dentro da Vertente A",
            "- Reparar metadados determinísticos restantes e planejar aquisicoes reais sem download automatico.",
            "",
            "## 13. Limitacoes",
            "- Esta etapa nao inventa datas, nao baixa assets, nao consulta API e nao calcula drift.",
            "",
            "## 14. Guardrails preservados",
            f"- Branch auditada: `{branch_name(repo_root)}`",
            *[f"- {item}" for item in GUARDRAILS],
            "",
        ]
    )


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    coverage_rows = read_csv(root / PATCH_COVERAGE)
    asset_rows = read_csv(root / ASSET_MANIFEST)
    asset_context = build_asset_context(asset_rows)
    plan_rows = build_gap_plan(coverage_rows, asset_context)
    requirements = build_requirements(plan_rows)
    metrics = build_metrics(plan_rows, requirements)

    write_csv(root / GAP_PLAN_OUTPUT, plan_rows, GAP_PLAN_COLUMNS)
    write_csv(root / REQUIREMENTS_OUTPUT, requirements, REQUIREMENTS_COLUMNS)
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
