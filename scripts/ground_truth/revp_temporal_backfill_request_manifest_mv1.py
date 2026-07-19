from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GAP_PLAN_INPUT = Path("outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv")
REQUIREMENTS_INPUT = Path("outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv")
GAP_METRICS_INPUT = Path("outputs_public/metrics/revp_temporal_acquisition_gap_plan_mv1.json")

REQUEST_MANIFEST_OUTPUT = Path("outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv")
BATCH_TARGETS_OUTPUT = Path("outputs_public/tables/revp_temporal_backfill_batch_targets_mv1.csv")
METRICS_OUTPUT = Path("outputs_public/metrics/revp_temporal_backfill_request_manifest_mv1.json")
REPORT_OUTPUT = Path("outputs_public/execution_reports/revp_temporal_backfill_request_manifest_mv1.md")

REQUEST_COLUMNS = [
    "request_id",
    "canonical_patch_id",
    "region",
    "required_slot_index",
    "current_clean_dates_count",
    "existing_dates",
    "missing_dates_to_3",
    "required_sensor_family",
    "request_mode",
    "request_status",
    "is_actionable_request",
    "blocked_before_request",
    "blocking_reason",
    "minimum_required_metadata",
    "requires_same_patch_geometry",
    "requires_same_preprocessing_contract",
    "requires_dinov2_frozen_inference_later",
    "requires_cloud_cover_if_s2",
    "requires_no_label_creation",
    "requires_no_event_assumption",
    "future_search_backend",
    "future_search_contract_status",
    "guardrail_status",
]

BATCH_COLUMNS = [
    "batch_id",
    "batch_goal",
    "target_patch_count",
    "canonical_patch_id",
    "region",
    "included_request_ids",
    "required_new_acquisitions_for_patch",
    "current_clean_dates_count",
    "projected_clean_dates_after_batch",
    "would_be_eligible_after_batch",
    "batch_priority_rank",
    "batch_inclusion_reason",
    "batch_blocking_reason",
    "guardrail_status",
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

GUARDRAIL_STATUS = "GUARDRAILS_PRESERVED_DRY_RUN_REQUEST_ONLY"

BATCH_DEFINITIONS = [
    (
        "BATCH_MIN_1_PATCH_TEMPORAL_DRIFT_PILOT",
        "single_patch_minimum_temporal_drift_pilot",
        1,
    ),
    (
        "BATCH_MIN_20_PATCHES_TEMPORAL_DRIFT_PILOT",
        "twenty_patch_temporal_drift_pilot",
        20,
    ),
    (
        "BATCH_MIN_30_PATCHES_TEMPORAL_DRIFT_READY",
        "thirty_patch_temporal_drift_ready",
        30,
    ),
]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Build REV-P MV1 temporal backfill dry-run request manifest.")
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def valid_region(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {"", "unknown", "multiple_or_none", "none", "null", "missing"}


def has_patch_linkage(row: dict[str, str]) -> bool:
    patch_id = row.get("canonical_patch_id", "").strip()
    blockers = row.get("remaining_blockers", "")
    return bool(patch_id) and "MISSING_PATCH_LINKAGE" not in blockers and "BLOCKED_MISSING_PATCH_ID" not in blockers


def requirement_complete(req: dict[str, str], gap_row: dict[str, str]) -> bool:
    return (
        bool(req.get("required_slot_index", "").strip())
        and parse_int(gap_row.get("missing_dates_to_3")) > 0
        and bool(req.get("required_minimum_metadata", "").strip())
        and req.get("required_new_date_status") == "UNSPECIFIED_REAL_ACQUISITION_NEEDED"
    )


def is_actionable_requirement(req: dict[str, str], gap_row: dict[str, str]) -> bool:
    sensor = req.get("required_sensor_family", "")
    return (
        req.get("requirement_status") == "ACTIONABLE_REQUIREMENT"
        and is_true(gap_row.get("is_actionable_for_temporal_backfill"))
        and not is_true(gap_row.get("requires_metadata_repair_before_acquisition"))
        and valid_region(req.get("region", ""))
        and has_patch_linkage(gap_row)
        and requirement_complete(req, gap_row)
        and sensor != "requires_manual_sensor_selection"
        and req.get("blocking_reason", "NONE") == "NONE"
        and all(
            is_true(req.get(field))
            for field in [
                "requires_same_patch_geometry",
                "requires_same_preprocessing_contract",
                "requires_dinov2_frozen_inference_later",
                "requires_no_label_creation",
                "requires_no_event_assumption",
            ]
        )
    )


def blocked_status(req: dict[str, str], gap_row: dict[str, str]) -> tuple[str, str, str, str]:
    gap_status = gap_row.get("acquisition_gap_status", "")
    blockers = ";".join(
        item
        for item in [
            gap_row.get("remaining_blockers", ""),
            req.get("blocking_reason", ""),
            gap_status,
        ]
        if item and item != "NONE"
    )
    sensor = req.get("required_sensor_family", "")
    if "UNKNOWN_REGION" in gap_status or "MISSING_REGION" in blockers or not valid_region(req.get("region", "")):
        return (
            "BLOCKED_METADATA_REPAIR_REQUIRED",
            "BLOCKED_BY_UNKNOWN_REGION",
            "UNDEFINED_BACKEND_BLOCKED",
            blockers or "BLOCKED_BY_UNKNOWN_REGION",
        )
    if "MISSING_PATCH_LINKAGE" in gap_status or "MISSING_PATCH_LINKAGE" in blockers or "BLOCKED_MISSING_PATCH_ID" in blockers:
        return (
            "BLOCKED_METADATA_REPAIR_REQUIRED",
            "BLOCKED_BY_MISSING_PATCH_LINKAGE",
            "UNDEFINED_BACKEND_BLOCKED",
            blockers or "BLOCKED_BY_MISSING_PATCH_LINKAGE",
        )
    if gap_status == "MANUAL_REVIEW_REQUIRED" or "AMBIGUOUS" in blockers or "MANUAL" in blockers:
        return (
            "BLOCKED_MANUAL_REVIEW_REQUIRED",
            "BLOCKED_BY_MANUAL_REVIEW",
            "MANUAL_SOURCE_REVIEW_NOT_EXECUTED",
            blockers or "MANUAL_REVIEW_REQUIRED",
        )
    if sensor == "requires_manual_sensor_selection":
        return (
            "BLOCKED_METADATA_REPAIR_REQUIRED",
            "BLOCKED_BY_UNSUPPORTED_SENSOR_SELECTION",
            "UNDEFINED_BACKEND_BLOCKED",
            blockers or "requires_manual_sensor_selection",
        )
    return (
        "BLOCKED_METADATA_REPAIR_REQUIRED",
        "BLOCKED_BY_METADATA_REPAIR",
        "UNDEFINED_BACKEND_BLOCKED",
        blockers or "BLOCKED_BY_METADATA_REPAIR",
    )


def request_contract(req: dict[str, str], gap_row: dict[str, str]) -> dict[str, str]:
    actionable = is_actionable_requirement(req, gap_row)
    if actionable:
        mode = "DRY_RUN_BACKFILL_REQUEST"
        status = "READY_FOR_FUTURE_SEARCH"
        backend = "STAC_OR_GEE_FUTURE_SEARCH_NOT_EXECUTED"
        contract_status = "DRY_RUN_CONTRACT_READY_NOT_EXECUTED"
        blocker = "NONE"
    else:
        mode, status, backend, blocker = blocked_status(req, gap_row)
        contract_status = "BLOCKED_BEFORE_FUTURE_SEARCH_CONTRACT"

    return {
        "request_mode": mode,
        "request_status": status,
        "is_actionable_request": str(actionable).lower(),
        "blocked_before_request": str(not actionable).lower(),
        "blocking_reason": blocker,
        "future_search_backend": backend,
        "future_search_contract_status": contract_status,
    }


def build_requests(requirement_rows: list[dict[str, str]], gap_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    gap_by_patch = {row["canonical_patch_id"]: row for row in gap_rows}
    rows: list[dict[str, str]] = []
    sorted_requirements = sorted(
        requirement_rows,
        key=lambda row: (
            row.get("canonical_patch_id", ""),
            parse_int(row.get("required_slot_index")),
            row.get("region", ""),
        ),
    )
    for index, req in enumerate(sorted_requirements, start=1):
        gap_row = gap_by_patch.get(req.get("canonical_patch_id", ""), {})
        contract = request_contract(req, gap_row)
        rows.append(
            {
                "request_id": f"REQ_MV1_{index:05d}",
                "canonical_patch_id": req.get("canonical_patch_id", ""),
                "region": req.get("region", ""),
                "required_slot_index": req.get("required_slot_index", ""),
                "current_clean_dates_count": gap_row.get("current_clean_dates_count", "0"),
                "existing_dates": req.get("existing_dates", gap_row.get("current_clean_dates", "unknown")),
                "missing_dates_to_3": gap_row.get("missing_dates_to_3", "0"),
                "required_sensor_family": req.get("required_sensor_family", ""),
                **contract,
                "minimum_required_metadata": req.get("required_minimum_metadata", ""),
                "requires_same_patch_geometry": req.get("requires_same_patch_geometry", "false"),
                "requires_same_preprocessing_contract": req.get("requires_same_preprocessing_contract", "false"),
                "requires_dinov2_frozen_inference_later": req.get("requires_dinov2_frozen_inference_later", "false"),
                "requires_cloud_cover_if_s2": req.get("requires_cloud_cover_if_s2", "false"),
                "requires_no_label_creation": req.get("requires_no_label_creation", "true"),
                "requires_no_event_assumption": req.get("requires_no_event_assumption", "true"),
                "guardrail_status": GUARDRAIL_STATUS,
            }
        )
    return rows


def actionable_patch_groups(request_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_patch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in request_rows:
        if row["is_actionable_request"] == "true":
            by_patch[row["canonical_patch_id"]].append(row)

    groups: list[dict[str, Any]] = []
    for patch_id, rows in by_patch.items():
        missing = parse_int(rows[0]["missing_dates_to_3"])
        if len(rows) != missing or missing <= 0:
            continue
        groups.append(
            {
                "canonical_patch_id": patch_id,
                "region": rows[0]["region"],
                "request_ids": [row["request_id"] for row in sorted(rows, key=lambda item: parse_int(item["required_slot_index"]))],
                "required_new_acquisitions": missing,
                "current_clean_dates_count": parse_int(rows[0]["current_clean_dates_count"]),
            }
        )
    return sorted(
        groups,
        key=lambda item: (
            -item["current_clean_dates_count"],
            item["required_new_acquisitions"],
            item["region"],
            item["canonical_patch_id"],
        ),
    )


def build_batch_targets(request_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = actionable_patch_groups(request_rows)
    rows: list[dict[str, str]] = []
    for batch_id, batch_goal, target_count in BATCH_DEFINITIONS:
        selected = groups[:target_count]
        if len(selected) < target_count:
            rows.append(
                {
                    "batch_id": batch_id,
                    "batch_goal": batch_goal,
                    "target_patch_count": str(target_count),
                    "canonical_patch_id": "insufficient_actionable_patches",
                    "region": "not_applicable",
                    "included_request_ids": "none",
                    "required_new_acquisitions_for_patch": "0",
                    "current_clean_dates_count": "0",
                    "projected_clean_dates_after_batch": "0",
                    "would_be_eligible_after_batch": "false",
                    "batch_priority_rank": "0",
                    "batch_inclusion_reason": "BATCH_TARGET_NOT_EXECUTABLE",
                    "batch_blocking_reason": "INSUFFICIENT_ACTIONABLE_PATCHES",
                    "guardrail_status": GUARDRAIL_STATUS,
                }
            )
            continue
        for rank, group in enumerate(selected, start=1):
            projected = group["current_clean_dates_count"] + group["required_new_acquisitions"]
            rows.append(
                {
                    "batch_id": batch_id,
                    "batch_goal": batch_goal,
                    "target_patch_count": str(target_count),
                    "canonical_patch_id": group["canonical_patch_id"],
                    "region": group["region"],
                    "included_request_ids": "|".join(group["request_ids"]),
                    "required_new_acquisitions_for_patch": str(group["required_new_acquisitions"]),
                    "current_clean_dates_count": str(group["current_clean_dates_count"]),
                    "projected_clean_dates_after_batch": str(projected),
                    "would_be_eligible_after_batch": str(projected >= 3).lower(),
                    "batch_priority_rank": str(rank),
                    "batch_inclusion_reason": "ACTIONABLE_PATCH_CLOSEST_TO_THREE_CLEAN_DATES",
                    "batch_blocking_reason": "NONE",
                    "guardrail_status": GUARDRAIL_STATUS,
                }
            )
    return rows


def step_status(actionable_patch_count: int, total_requests: int, blocked_requests: int) -> str:
    if actionable_patch_count >= 30:
        return "STEP_A_BACKFILL_REQUESTS_READY_FOR_30_PATCH_DRY_RUN"
    if actionable_patch_count >= 20:
        return "STEP_A_BACKFILL_REQUESTS_READY_FOR_20_PATCH_PILOT_DRY_RUN"
    if actionable_patch_count >= 1:
        return "STEP_A_BACKFILL_REQUESTS_READY_FOR_SINGLE_PATCH_DRY_RUN"
    if total_requests and blocked_requests > total_requests / 2:
        return "STEP_A_BACKFILL_REQUESTS_REQUIRE_METADATA_REPAIR_FIRST"
    return "STEP_A_BACKFILL_REQUESTS_BLOCKED"


def batch_metrics(batch_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    payload: dict[str, dict[str, int]] = {}
    for batch_id, _, target_count in BATCH_DEFINITIONS:
        rows = [row for row in batch_rows if row["batch_id"] == batch_id and row["batch_blocking_reason"] == "NONE"]
        payload[batch_id] = {
            "target_patch_count": target_count,
            "selected_patch_count": len(rows),
            "required_new_acquisitions": sum(parse_int(row["required_new_acquisitions_for_patch"]) for row in rows),
        }
    return payload


def build_metrics(
    request_rows: list[dict[str, str]],
    batch_rows: list[dict[str, str]],
    gap_rows: list[dict[str, str]],
    requirement_rows: list[dict[str, str]],
    gap_metrics: dict[str, Any],
) -> dict[str, Any]:
    actionable_groups = actionable_patch_groups(request_rows)
    actionable_patch_ids = {group["canonical_patch_id"] for group in actionable_groups}
    blocked_patch_ids = {
        row["canonical_patch_id"]
        for row in request_rows
        if row["canonical_patch_id"] and row["canonical_patch_id"] not in actionable_patch_ids
    }
    total_requests = len(request_rows)
    actionable_requests = sum(1 for row in request_rows if row["is_actionable_request"] == "true")
    blocked_requests = total_requests - actionable_requests
    source_patch_count_marked_actionable = sum(
        1 for row in gap_rows if row.get("is_actionable_for_temporal_backfill") == "true"
    )
    source_requirement_count_marked_actionable = sum(
        1 for row in requirement_rows if row.get("requirement_status") == "ACTIONABLE_REQUIREMENT"
    )
    return {
        "total_requests": total_requests,
        "actionable_requests": actionable_requests,
        "blocked_requests": blocked_requests,
        "source_patch_count_marked_actionable_for_temporal_backfill": source_patch_count_marked_actionable,
        "source_actionable_requirement_count": source_requirement_count_marked_actionable,
        "request_actionability_recheck": "FAIL_CLOSED_REQUIRES_SUPPORTED_SENSOR_FAMILY_AND_COMPLETE_METADATA",
        "counts_by_request_status": dict(sorted(Counter(row["request_status"] for row in request_rows).items())),
        "counts_by_region": dict(sorted(Counter(row["region"] for row in request_rows).items())),
        "counts_by_sensor_family": dict(sorted(Counter(row["required_sensor_family"] for row in request_rows).items())),
        "actionable_patch_count": len(actionable_patch_ids),
        "blocked_patch_count": len(blocked_patch_ids),
        "total_gap_plan_patches": len(gap_rows),
        "source_total_missing_acquisition_slots": gap_metrics.get("total_missing_acquisition_slots", total_requests),
        "batch_targets": batch_metrics(batch_rows),
        "step_a_backfill_request_status": step_status(len(actionable_patch_ids), total_requests, blocked_requests),
        "guardrails_preserved": GUARDRAILS,
        "input_files": [
            GAP_PLAN_INPUT.as_posix(),
            REQUIREMENTS_INPUT.as_posix(),
            GAP_METRICS_INPUT.as_posix(),
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
    batch_lines = []
    for batch_id, data in metrics["batch_targets"].items():
        batch_lines.append(
            f"- `{batch_id}`: {data['selected_patch_count']} patches, "
            f"{data['required_new_acquisitions']} novas aquisicoes planejadas em dry-run"
        )
    return "\n".join(
        [
            "# revp_temporal_backfill_request_manifest_mv1",
            "",
            "## 1. Escopo",
            "Manifesto publico e auditavel de requisicoes futuras de backfill temporal para a Vertente A.",
            "",
            "## 2. Entradas usadas",
            *[f"- `{item}`" for item in metrics["input_files"]],
            "",
            "## 3. Regra de conversao slot -> request",
            "- Cada slot faltante do arquivo de requisitos vira uma linha de requisicao dry-run.",
            "- A requisicao permanece bloqueada quando metadados, regiao, vinculo de patch ou familia de sensor ainda impedem busca futura.",
            "",
            "## 4. Totais de requests",
            f"- Requests totais: {metrics['total_requests']}",
            f"- Requests acionaveis: {metrics['actionable_requests']}",
            f"- Requests bloqueados antes da busca: {metrics['blocked_requests']}",
            f"- Patches marcados como acionaveis na entrada anterior: {metrics['source_patch_count_marked_actionable_for_temporal_backfill']}",
            f"- Requisitos marcados como acionaveis na entrada anterior: {metrics['source_actionable_requirement_count']}",
            f"- Recheck aplicado: `{metrics['request_actionability_recheck']}`",
            "",
            "## 5. Contagem por status",
            *[f"- {key}: {value}" for key, value in metrics["counts_by_request_status"].items()],
            "",
            "## 6. Contagem por regiao",
            *[f"- {key}: {value}" for key, value in metrics["counts_by_region"].items()],
            "",
            "## 7. Contagem por familia de sensor",
            *[f"- {key}: {value}" for key, value in metrics["counts_by_sensor_family"].items()],
            "",
            "## 8. Batches minimos de dry-run",
            *batch_lines,
            "",
            "## 9. Status da etapa",
            f"- `step_a_backfill_request_status`: `{metrics['step_a_backfill_request_status']}`",
            "",
            "## 10. Limitacoes",
            "- Nenhuma data futura foi inventada.",
            "- Nenhuma busca STAC, API, GEE, download ou execucao externa foi realizada.",
            "- Nenhum embedding, drift, treino, rotulo ou evidencia de ausencia foi criado.",
            "",
            "## 11. Proximo passo permitido",
            "- Apenas revisao tecnica do manifesto e, futuramente, execucao controlada por operador externo com contrato de metadados completo.",
            "",
            "## 12. Guardrails preservados",
            f"- Branch auditada: `{branch_name(repo_root)}`",
            *[f"- {item}" for item in GUARDRAILS],
            "",
        ]
    )


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    gap_rows = read_csv(root / GAP_PLAN_INPUT)
    requirement_rows = read_csv(root / REQUIREMENTS_INPUT)
    gap_metrics = read_json(root / GAP_METRICS_INPUT)
    request_rows = build_requests(requirement_rows, gap_rows)
    batch_rows = build_batch_targets(request_rows)
    metrics = build_metrics(request_rows, batch_rows, gap_rows, requirement_rows, gap_metrics)

    write_csv(root / REQUEST_MANIFEST_OUTPUT, request_rows, REQUEST_COLUMNS)
    write_csv(root / BATCH_TARGETS_OUTPUT, batch_rows, BATCH_COLUMNS)
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
