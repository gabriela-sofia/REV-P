from __future__ import annotations

import csv
import hashlib
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GAP_PLAN_INPUT = Path("outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv")
REQUIREMENTS_INPUT = Path("outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv")
REQUEST_MANIFEST_INPUT = Path("outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv")
ASSET_MANIFEST_INPUT = Path("outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv")
PATCH_COVERAGE_INPUT = Path("outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv")
REQUEST_METRICS_INPUT = Path("outputs_public/metrics/revp_temporal_backfill_request_manifest_mv1.json")
NORMALIZED_METRICS_INPUT = Path("outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json")

SENSOR_RESOLUTION_OUTPUT = Path("outputs_public/tables/revp_temporal_sensor_family_resolution_mv1.csv")
RESOLVED_REQUIREMENTS_OUTPUT = Path("outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv")
METRICS_OUTPUT = Path("outputs_public/metrics/revp_temporal_sensor_family_resolution_mv1.json")
REPORT_OUTPUT = Path("outputs_public/execution_reports/revp_temporal_sensor_family_resolution_mv1.md")

SENSOR_RESOLUTION_COLUMNS = [
    "canonical_patch_id",
    "region",
    "required_slot_index",
    "original_required_sensor_family",
    "resolved_required_sensor_family",
    "sensor_resolution_status",
    "sensor_resolution_rule",
    "sensor_resolution_confidence",
    "evidence_asset_ids",
    "evidence_asset_types",
    "has_clean_s2_asset",
    "has_clean_s1_asset",
    "has_dinov2_embedding_only",
    "has_unknown_asset_type",
    "is_sensor_resolved",
    "is_request_potentially_actionable_after_resolution",
    "blocking_reason",
    "guardrail_status",
]

RESOLVED_REQUIREMENT_EXTRA_COLUMNS = [
    "original_required_sensor_family",
    "resolved_required_sensor_family",
    "sensor_resolution_status",
    "sensor_resolution_confidence",
    "is_sensor_resolved",
    "is_requirement_actionable_after_sensor_resolution",
    "remaining_requirement_blocker",
    "resolution_lineage",
]

GUARDRAILS = [
    "sem download de assets",
    "sem consulta STAC API ou GEE",
    "sem gerar URLs de download",
    "sem calcular embedding drift",
    "sem gerar embeddings novos",
    "sem treino",
    "sem rotulo",
    "sem criar evidencia de ausencia supervisionada",
    "sem classe-zero",
    "sem GT operacional",
    "sem tabela multimodal de atributos",
    "sem alterar manifestos originais",
]

GUARDRAIL_STATUS = "GUARDRAILS_PRESERVED_SENSOR_FAMILY_RESOLUTION_ONLY"

BATCH_TARGETS = {
    "recoverable_1_patch": 1,
    "recoverable_20_patches": 20,
    "recoverable_30_patches": 30,
}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Resolve REV-P MV1 temporal backfill sensor family in dry-run mode.")
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
    return value.strip().lower() not in {"", "unknown", "multiple_or_none", "none", "null", "missing"}


def public_asset_ref(row: dict[str, str]) -> str:
    raw = "|".join(
        [
            row.get("canonical_patch_id", ""),
            row.get("asset_id", ""),
            row.get("normalized_asset_id", ""),
            row.get("asset_type", ""),
        ]
    )
    return f"asset_ref_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def build_asset_context(asset_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    for row in asset_rows:
        patch_id = row.get("canonical_patch_id", "").strip()
        if not patch_id:
            continue
        item = context.setdefault(
            patch_id,
            {
                "asset_types": Counter(),
                "clean_types": Counter(),
                "clean_s2": [],
                "clean_s1": [],
                "clean_dinov2": [],
                "unknown": [],
                "non_temporal": [],
            },
        )
        asset_type = row.get("asset_type", "unknown") or "unknown"
        item["asset_types"][asset_type] += 1
        if row.get("temporal_usability_status") == "NOT_TEMPORAL_ASSET":
            item["non_temporal"].append(row)
        if asset_type == "unknown":
            item["unknown"].append(row)
        if is_true(row.get("is_clean_temporal_asset")):
            item["clean_types"][asset_type] += 1
            if asset_type == "sentinel_2_optical":
                item["clean_s2"].append(row)
            elif asset_type == "sentinel_1_sar":
                item["clean_s1"].append(row)
            elif asset_type == "dinov2_embedding":
                item["clean_dinov2"].append(row)
    return context


def evidence_rows(context: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in ["clean_s2", "clean_s1", "clean_dinov2", "unknown", "non_temporal"]:
        rows.extend(context.get(key, []))
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        unique[public_asset_ref(row)] = row
    return [unique[key] for key in sorted(unique)]


def evidence_asset_ids(context: dict[str, Any]) -> str:
    rows = evidence_rows(context)
    if not rows:
        return "none"
    return "|".join(public_asset_ref(row) for row in rows)


def evidence_asset_types(context: dict[str, Any]) -> str:
    rows = evidence_rows(context)
    if not rows:
        return "none"
    return "|".join(sorted({row.get("asset_type", "unknown") or "unknown" for row in rows}))


def missing_patch_context(req: dict[str, str], gap_row: dict[str, str] | None) -> bool:
    if not req.get("canonical_patch_id", "").strip():
        return True
    if not valid_region(req.get("region", "")):
        return True
    if gap_row is None:
        return True
    blockers = f"{gap_row.get('remaining_blockers', '')};{gap_row.get('acquisition_gap_status', '')}"
    return "MISSING_PATCH_LINKAGE" in blockers or "BLOCKED_MISSING_PATCH_ID" in blockers


def resolve_sensor(req: dict[str, str], gap_row: dict[str, str] | None, context: dict[str, Any]) -> dict[str, str]:
    clean_s2 = context.get("clean_s2", [])
    clean_s1 = context.get("clean_s1", [])
    clean_dino = context.get("clean_dinov2", [])
    unknown = context.get("unknown", [])
    clean_acquisition_families = sum(1 for rows in [clean_s2, clean_s1] if rows)
    has_dino_without_source_sensor = bool(clean_dino) and not clean_s2 and not clean_s1

    if missing_patch_context(req, gap_row):
        status = "BLOCKED_MISSING_PATCH_CONTEXT"
        resolved = "requires_manual_sensor_selection"
        rule = "MISSING_PATCH_REGION_OR_LINKAGE_CONTEXT"
        confidence = "REJECTED_MISSING_CONTEXT"
        resolved_flag = False
        blocker = "MISSING_PATCH_CONTEXT"
    elif clean_acquisition_families > 1:
        status = "BLOCKED_AMBIGUOUS_SENSOR_FAMILY"
        resolved = "blocked_ambiguous_sensor_family"
        rule = "MULTIPLE_CLEAN_ACQUISITION_SENSOR_FAMILIES_WITHOUT_TIE_RULE"
        confidence = "REJECTED_AMBIGUOUS"
        resolved_flag = False
        blocker = "AMBIGUOUS_SENSOR_FAMILY"
    elif clean_s2:
        status = "RESOLVED_SENTINEL_2_OPTICAL"
        resolved = "sentinel_2_optical_preferred"
        rule = "EXISTING_CLEAN_SENTINEL_2_OPTICAL_ASSET_ON_PATCH"
        confidence = "HIGH_EXISTING_CLEAN_S2_ASSET"
        resolved_flag = True
        blocker = "NONE"
    elif clean_s1:
        status = "RESOLVED_SENTINEL_1_SUPPORT_ONLY"
        resolved = "sentinel_1_sar_optional_support"
        rule = "EXISTING_CLEAN_SENTINEL_1_SAR_AS_SUPPORT_ONLY"
        confidence = "MEDIUM_EXISTING_S1_SUPPORT_ONLY"
        resolved_flag = True
        blocker = "SENTINEL_1_IS_SUPPORT_ONLY_NOT_SENTINEL_2_BACKFILL"
    elif has_dino_without_source_sensor:
        status = "BLOCKED_DERIVED_EMBEDDING_WITHOUT_SOURCE_SENSOR"
        resolved = "blocked_derived_embedding_without_source_sensor"
        rule = "DINOV2_EMBEDDING_IS_DERIVED_NOT_ACQUISITION_SENSOR"
        confidence = "REJECTED_DERIVED_EMBEDDING_ONLY"
        resolved_flag = False
        blocker = "DERIVED_EMBEDDING_WITHOUT_SOURCE_SENSOR"
    elif unknown:
        status = "BLOCKED_UNKNOWN_ASSET_TYPE"
        resolved = "blocked_unknown_asset_type"
        rule = "ONLY_UNKNOWN_OR_UNUSABLE_ASSET_TYPE_AVAILABLE"
        confidence = "REJECTED_UNKNOWN_ASSET_TYPE"
        resolved_flag = False
        blocker = "UNKNOWN_ASSET_TYPE"
    else:
        status = "BLOCKED_MANUAL_REVIEW_REQUIRED"
        resolved = "requires_manual_sensor_selection"
        rule = "NO_DETERMINISTIC_ACQUISITION_SENSOR_EVIDENCE"
        confidence = "REJECTED_MISSING_CONTEXT"
        resolved_flag = False
        blocker = "MANUAL_REVIEW_REQUIRED"

    return {
        "resolved_required_sensor_family": resolved,
        "sensor_resolution_status": status,
        "sensor_resolution_rule": rule,
        "sensor_resolution_confidence": confidence,
        "has_clean_s2_asset": str(bool(clean_s2)).lower(),
        "has_clean_s1_asset": str(bool(clean_s1)).lower(),
        "has_dinov2_embedding_only": str(has_dino_without_source_sensor).lower(),
        "has_unknown_asset_type": str(bool(unknown)).lower(),
        "is_sensor_resolved": str(resolved_flag).lower(),
        "blocking_reason": blocker,
    }


def requirement_actionable_after_resolution(
    req: dict[str, str],
    gap_row: dict[str, str] | None,
    resolution: dict[str, str],
) -> tuple[bool, str]:
    blockers: list[str] = []
    if req.get("required_new_date_status") != "UNSPECIFIED_REAL_ACQUISITION_NEEDED":
        blockers.append("DATE_STATUS_NOT_UNSPECIFIED_REAL_ACQUISITION_NEEDED")
    if not req.get("canonical_patch_id", "").strip():
        blockers.append("MISSING_PATCH_ID")
    if not valid_region(req.get("region", "")):
        blockers.append("MISSING_REGION")
    if gap_row is None:
        blockers.append("MISSING_GAP_PLAN_ROW")
    else:
        gap_blockers = f"{gap_row.get('remaining_blockers', '')};{gap_row.get('acquisition_gap_status', '')}"
        if "MISSING_PATCH_LINKAGE" in gap_blockers or "BLOCKED_MISSING_PATCH_ID" in gap_blockers:
            blockers.append("MISSING_PATCH_LINKAGE")
        if is_true(gap_row.get("requires_metadata_repair_before_acquisition")):
            blockers.append("METADATA_REPAIR_REQUIRED_BEFORE_REQUEST")
        if gap_row.get("remaining_blockers", "NONE") not in {"", "NONE"}:
            blockers.append("PREVIOUS_METADATA_BLOCKER_PRESENT")
    if resolution["resolved_required_sensor_family"] != "sentinel_2_optical_preferred":
        blockers.append("SENSOR_NOT_RESOLVED_TO_SENTINEL_2_OPTICAL")
    if req.get("blocking_reason", "NONE") != "NONE":
        blockers.append("REQUIREMENT_ALREADY_BLOCKED")
    actionable = not blockers
    return actionable, "NONE" if actionable else "|".join(blockers)


def build_resolution_rows(
    requirement_rows: list[dict[str, str]],
    gap_rows: list[dict[str, str]],
    asset_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    gap_by_patch = {row["canonical_patch_id"]: row for row in gap_rows}
    asset_context = build_asset_context(asset_rows)
    rows: list[dict[str, str]] = []
    for req in sorted(
        requirement_rows,
        key=lambda row: (row.get("canonical_patch_id", ""), parse_int(row.get("required_slot_index")), row.get("region", "")),
    ):
        patch_id = req.get("canonical_patch_id", "")
        gap_row = gap_by_patch.get(patch_id)
        context = asset_context.get(patch_id, {})
        resolution = resolve_sensor(req, gap_row, context)
        actionable, remaining = requirement_actionable_after_resolution(req, gap_row, resolution)
        rows.append(
            {
                "canonical_patch_id": patch_id,
                "region": req.get("region", ""),
                "required_slot_index": req.get("required_slot_index", ""),
                "original_required_sensor_family": req.get("required_sensor_family", ""),
                "resolved_required_sensor_family": resolution["resolved_required_sensor_family"],
                "sensor_resolution_status": resolution["sensor_resolution_status"],
                "sensor_resolution_rule": resolution["sensor_resolution_rule"],
                "sensor_resolution_confidence": resolution["sensor_resolution_confidence"],
                "evidence_asset_ids": evidence_asset_ids(context),
                "evidence_asset_types": evidence_asset_types(context),
                "has_clean_s2_asset": resolution["has_clean_s2_asset"],
                "has_clean_s1_asset": resolution["has_clean_s1_asset"],
                "has_dinov2_embedding_only": resolution["has_dinov2_embedding_only"],
                "has_unknown_asset_type": resolution["has_unknown_asset_type"],
                "is_sensor_resolved": resolution["is_sensor_resolved"],
                "is_request_potentially_actionable_after_resolution": str(actionable).lower(),
                "blocking_reason": remaining if not actionable else "NONE",
                "guardrail_status": GUARDRAIL_STATUS,
            }
        )
    return rows


def build_resolved_requirements(
    requirement_rows: list[dict[str, str]],
    resolution_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    resolution_by_key = {
        (row["canonical_patch_id"], row["required_slot_index"], row["region"]): row for row in resolution_rows
    }
    output: list[dict[str, str]] = []
    for req in sorted(
        requirement_rows,
        key=lambda row: (row.get("canonical_patch_id", ""), parse_int(row.get("required_slot_index")), row.get("region", "")),
    ):
        key = (req.get("canonical_patch_id", ""), req.get("required_slot_index", ""), req.get("region", ""))
        resolution = resolution_by_key[key]
        row = dict(req)
        row["original_required_sensor_family"] = req.get("required_sensor_family", "")
        row["required_sensor_family"] = resolution["resolved_required_sensor_family"]
        row["resolved_required_sensor_family"] = resolution["resolved_required_sensor_family"]
        row["sensor_resolution_status"] = resolution["sensor_resolution_status"]
        row["sensor_resolution_confidence"] = resolution["sensor_resolution_confidence"]
        row["is_sensor_resolved"] = resolution["is_sensor_resolved"]
        row["is_requirement_actionable_after_sensor_resolution"] = resolution[
            "is_request_potentially_actionable_after_resolution"
        ]
        row["remaining_requirement_blocker"] = resolution["blocking_reason"]
        row["resolution_lineage"] = "DERIVED_FROM_TEMPORAL_REQUIREMENTS_AND_NORMALIZED_ASSET_MANIFEST_MV1"
        output.append(row)

    columns = list(requirement_rows[0].keys()) if requirement_rows else []
    for column in RESOLVED_REQUIREMENT_EXTRA_COLUMNS:
        if column not in columns:
            columns.append(column)
    return output, columns


def actionable_patch_groups(rows: list[dict[str, str]], gap_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    gap_by_patch = {row["canonical_patch_id"]: row for row in gap_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["is_request_potentially_actionable_after_resolution"] == "true":
            grouped[row["canonical_patch_id"]].append(row)

    groups: list[dict[str, Any]] = []
    for patch_id, patch_rows in grouped.items():
        gap_row = gap_by_patch.get(patch_id, {})
        missing = parse_int(gap_row.get("missing_dates_to_3", len(patch_rows)))
        if missing > 0 and len(patch_rows) == missing:
            groups.append(
                {
                    "canonical_patch_id": patch_id,
                    "region": patch_rows[0]["region"],
                    "current_clean_dates_count": parse_int(gap_row.get("current_clean_dates_count")),
                    "required_new_acquisitions": missing,
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


def recoverable_batches(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key, target in BATCH_TARGETS.items():
        selected = groups[:target]
        output[key] = {
            "target_patch_count": target,
            "recoverable": len(selected) >= target,
            "selected_patch_count": len(selected) if len(selected) >= target else 0,
            "required_new_acquisitions": sum(item["required_new_acquisitions"] for item in selected)
            if len(selected) >= target
            else 0,
        }
    return output


def step_status(groups: list[dict[str, Any]], resolution_rows: list[dict[str, str]]) -> str:
    count = len(groups)
    if count >= 30:
        return "STEP_A_SENSOR_RESOLUTION_READY_FOR_30_PATCH_REQUEST_REBUILD"
    if count >= 20:
        return "STEP_A_SENSOR_RESOLUTION_READY_FOR_20_PATCH_REQUEST_REBUILD"
    if count >= 1:
        return "STEP_A_SENSOR_RESOLUTION_READY_FOR_SINGLE_PATCH_REQUEST_REBUILD"
    if any(
        row["sensor_resolution_status"] in {"RESOLVED_SENTINEL_2_OPTICAL", "RESOLVED_SENTINEL_1_SUPPORT_ONLY"}
        for row in resolution_rows
    ):
        return "STEP_A_SENSOR_RESOLUTION_PARTIAL_REPAIR_REQUIRED"
    return "STEP_A_SENSOR_RESOLUTION_BLOCKED"


def build_metrics(
    resolution_rows: list[dict[str, str]],
    gap_rows: list[dict[str, str]],
    request_metrics: dict[str, Any],
    normalized_metrics: dict[str, Any],
) -> dict[str, Any]:
    groups = actionable_patch_groups(resolution_rows, gap_rows)
    status_counts = Counter(row["sensor_resolution_status"] for row in resolution_rows)
    return {
        "total_slots_evaluated": len(resolution_rows),
        "slots_originally_requires_manual_sensor_selection": sum(
            1 for row in resolution_rows if row["original_required_sensor_family"] == "requires_manual_sensor_selection"
        ),
        "slots_resolved_to_sentinel_2": status_counts["RESOLVED_SENTINEL_2_OPTICAL"],
        "slots_marked_sentinel_1_support": status_counts["RESOLVED_SENTINEL_1_SUPPORT_ONLY"],
        "slots_blocked_by_derived_dinov2_without_source_sensor": status_counts[
            "BLOCKED_DERIVED_EMBEDDING_WITHOUT_SOURCE_SENSOR"
        ],
        "slots_blocked_by_unknown_asset_type": status_counts["BLOCKED_UNKNOWN_ASSET_TYPE"],
        "slots_blocked_by_ambiguous_sensor_family": status_counts["BLOCKED_AMBIGUOUS_SENSOR_FAMILY"],
        "slots_potentially_actionable_after_resolution": sum(
            1 for row in resolution_rows if row["is_request_potentially_actionable_after_resolution"] == "true"
        ),
        "patches_potentially_actionable_after_resolution": len(groups),
        "batches_potentially_recoverable": recoverable_batches(groups),
        "counts_by_sensor_resolution_status": dict(sorted(status_counts.items())),
        "source_request_manifest_status": request_metrics.get("step_a_backfill_request_status", "unknown"),
        "source_normalized_manifest_status": normalized_metrics.get("step_a_normalized_manifest_status", "unknown"),
        "step_a_sensor_resolution_status": step_status(groups, resolution_rows),
        "guardrails_preserved": GUARDRAILS,
        "input_files": [
            GAP_PLAN_INPUT.as_posix(),
            REQUIREMENTS_INPUT.as_posix(),
            REQUEST_MANIFEST_INPUT.as_posix(),
            ASSET_MANIFEST_INPUT.as_posix(),
            PATCH_COVERAGE_INPUT.as_posix(),
            REQUEST_METRICS_INPUT.as_posix(),
            NORMALIZED_METRICS_INPUT.as_posix(),
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
    batches = metrics["batches_potentially_recoverable"]
    return "\n".join(
        [
            "# revp_temporal_sensor_family_resolution_mv1",
            "",
            "## 1. Escopo",
            "Camada publica e auditavel de resolucao deterministica de familia de sensor para slots de backfill temporal.",
            "",
            "## 2. Relacao com o fail-closed do request manifest",
            f"- Status anterior: `{metrics['source_request_manifest_status']}`.",
            "- Esta etapa explica se o bloqueio por selecao manual de sensor pode ser reparado sem busca externa.",
            "",
            "## 3. Por que DINOv2 nao e sensor de aquisicao",
            "- DINOv2 aparece aqui como embedding derivado. Ele nao identifica, por si so, a familia fisica de aquisicao do asset fonte.",
            "- Por isso, embedding derivado sem sensor source preservado permanece bloqueado para backfill temporal.",
            "",
            "## 4. Arquivos de entrada",
            *[f"- `{item}`" for item in metrics["input_files"]],
            "",
            "## 5. Regras de resolucao",
            "- Asset limpo `sentinel_2_optical` resolve para `sentinel_2_optical_preferred` quando nao ha conflito.",
            "- Asset limpo `sentinel_1_sar` vira `sentinel_1_sar_optional_support`, sem substituir Sentinel-2.",
            "- Evidencia DINOv2 derivada isolada nao resolve sensor de aquisicao.",
            "",
            "## 6. Regras de bloqueio",
            "- Contexto ausente de patch, regiao ou vinculo preservavel bloqueia.",
            "- Asset unknown bloqueia.",
            "- Familias de aquisicao conflitantes bloqueiam.",
            "- Metadata blocker anterior bloqueia acionabilidade posterior.",
            "",
            "## 7. Slots resolvidos",
            f"- Sentinel-2: {metrics['slots_resolved_to_sentinel_2']}",
            f"- Sentinel-1 suporte: {metrics['slots_marked_sentinel_1_support']}",
            "",
            "## 8. Slots bloqueados",
            f"- DINOv2 derivado sem sensor source: {metrics['slots_blocked_by_derived_dinov2_without_source_sensor']}",
            f"- Unknown: {metrics['slots_blocked_by_unknown_asset_type']}",
            f"- Ambiguidade: {metrics['slots_blocked_by_ambiguous_sensor_family']}",
            "",
            "## 9. Impacto potencial sobre requests acionaveis",
            f"- Slots potencialmente acionaveis apos resolucao: {metrics['slots_potentially_actionable_after_resolution']}",
            f"- Patches potencialmente acionaveis apos resolucao: {metrics['patches_potentially_actionable_after_resolution']}",
            f"- `step_a_sensor_resolution_status`: `{metrics['step_a_sensor_resolution_status']}`",
            "",
            "## 10. Batches que poderiam ser reconstruidos",
            f"- 1 patch: {batches['recoverable_1_patch']['recoverable']}, aquisicoes: {batches['recoverable_1_patch']['required_new_acquisitions']}",
            f"- 20 patches: {batches['recoverable_20_patches']['recoverable']}, aquisicoes: {batches['recoverable_20_patches']['required_new_acquisitions']}",
            f"- 30 patches: {batches['recoverable_30_patches']['recoverable']}, aquisicoes: {batches['recoverable_30_patches']['required_new_acquisitions']}",
            "",
            "## 11. Proximos passos permitidos dentro da Vertente A",
            "- Revisar manualmente a origem dos embeddings derivados e recuperar sensor source apenas com evidencia explicita.",
            "- Reemitir manifesto dry-run acionavel somente se uma etapa posterior produzir requisitos resolvidos para Sentinel-2 sem blockers.",
            "",
            "## 12. Limitacoes",
            "- Nenhuma data nova foi criada.",
            "- Nenhum backend externo foi consultado.",
            "- Nenhum manifesto anterior foi alterado.",
            "",
            "## 13. Guardrails preservados",
            f"- Branch auditada: `{branch_name(repo_root)}`",
            *[f"- {item}" for item in GUARDRAILS],
            "",
        ]
    )


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    gap_rows = read_csv(root / GAP_PLAN_INPUT)
    requirement_rows = read_csv(root / REQUIREMENTS_INPUT)
    read_csv(root / REQUEST_MANIFEST_INPUT)
    asset_rows = read_csv(root / ASSET_MANIFEST_INPUT)
    read_csv(root / PATCH_COVERAGE_INPUT)
    request_metrics = read_json(root / REQUEST_METRICS_INPUT)
    normalized_metrics = read_json(root / NORMALIZED_METRICS_INPUT)

    resolution_rows = build_resolution_rows(requirement_rows, gap_rows, asset_rows)
    resolved_requirement_rows, resolved_requirement_columns = build_resolved_requirements(requirement_rows, resolution_rows)
    metrics = build_metrics(resolution_rows, gap_rows, request_metrics, normalized_metrics)

    write_csv(root / SENSOR_RESOLUTION_OUTPUT, resolution_rows, SENSOR_RESOLUTION_COLUMNS)
    write_csv(root / RESOLVED_REQUIREMENTS_OUTPUT, resolved_requirement_rows, resolved_requirement_columns)
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
