from __future__ import annotations

import csv
import json
import re
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


BACKFILL_TABLE = Path("outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv")
BACKFILL_METRICS = Path("outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json")
BACKFILL_REPORT = Path("outputs_public/execution_reports/revp_temporal_asset_backfill_queue_mv1.md")
READINESS_TABLE = Path("outputs_public/tables/revp_temporal_asset_readiness_mv1.csv")
READINESS_METRICS = Path("outputs_public/metrics/revp_temporal_asset_readiness_mv1.json")

OUTPUT_TABLE = Path("outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv")
OUTPUT_METRICS = Path("outputs_public/metrics/revp_temporal_metadata_repair_candidates_mv1.json")
OUTPUT_REPORT = Path("outputs_public/execution_reports/revp_temporal_metadata_repair_candidates_mv1.md")

CSV_COLUMNS = [
    "canonical_patch_id",
    "asset_id",
    "source_file",
    "source_field",
    "repair_field",
    "original_value",
    "candidate_value",
    "normalized_value",
    "repair_rule",
    "repair_confidence",
    "is_deterministic",
    "is_applicable",
    "rejection_reason",
    "asset_type_before",
    "asset_type_after",
    "region_before",
    "region_after",
    "acquisition_date_before",
    "acquisition_date_after",
    "cloud_cover_before",
    "cloud_cover_after",
    "would_increase_clean_asset_count",
    "would_increase_clean_date_count",
    "would_reduce_blocker",
    "guardrail_status",
]

SCAN_ROOTS = [
    Path("manifests"),
    Path("outputs_public"),
    Path("datasets"),
    Path("configs"),
    Path("local_runs"),
]
READABLE_SUFFIXES = {".csv", ".json", ".jsonl", ".md"}
STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl"}
MAX_READ_BYTES = 5_000_000
SCAN_KEYWORDS = (
    "sentinel",
    "dino",
    "asset",
    "manifest",
    "inventory",
    "lineage",
    "temporal",
    "embedding",
    "patch",
    "binding",
    "evidence",
)
OUT_OF_SCOPE_PATH_MARKERS = (
    "formal_negative",
    "negative_label",
    "label_readiness",
    "training_readiness",
    "supervised",
    "trainable",
)
MISSING_VALUES = {"", "unknown", "missing", "ausente", "desconhecido", "none", "null", "nan", "n/a", "na"}

PATCH_FIELDS = [
    "canonical_patch_id",
    "patch_id",
    "id_patch_canonico",
    "reference_patch_id",
    "patch_candidate_id",
    "patch_id_detected",
    "reference_patch",
]
ASSET_FIELDS = [
    "asset_id",
    "source_asset_id",
    "dino_input_id",
    "scene_id",
    "scene_id_sanitized",
    "source_path",
    "source_file",
    "asset_path_reference",
    "vector_sha256",
    "quality_id",
    "binding_id",
]
REGION_FIELDS = ["region", "regiao", "region_detected", "city", "cidade"]
DATE_FIELDS = [
    "acquisition_date",
    "acquisition_datetime",
    "datetime",
    "scene_date",
    "date_detected",
    "selected_sentinel_date",
    "candidate_dates",
    "extracted_date",
    "recovered_date",
    "preview_sentinel_date",
    "pre_scene_date",
    "post_scene_date",
]
CLOUD_FIELDS = [
    "cloud_cover",
    "cloud_cover_metadata",
    "local_cloud_fraction",
    "cloud_metadata_global",
    "cloud_cover_min",
    "cloud_cover_max",
    "cloud_cover_mean",
]
TYPE_FIELDS = ["asset_type", "source_asset_type", "asset_kind", "sensor", "platform", "modality", "type"]

STEP_READY_NORMALIZED_MANIFEST = "STEP_A_REPAIR_CANDIDATES_READY_FOR_NORMALIZED_TEMPORAL_MANIFEST"
STEP_REQUIRE_LINKAGE_REVIEW = "STEP_A_REPAIR_CANDIDATES_REQUIRE_LINKAGE_REVIEW"
STEP_REQUIRE_MANUAL_REVIEW = "STEP_A_REPAIR_CANDIDATES_REQUIRE_MANUAL_REVIEW"
STEP_BLOCKED_INSUFFICIENT_EVIDENCE = "STEP_A_REPAIR_BLOCKED_INSUFFICIENT_DETERMINISTIC_EVIDENCE"

GUARDRAILS = [
    "sem aplicar reparo nos manifestos originais",
    "sem download automatico",
    "sem STAC ou API",
    "sem calculo de drift",
    "sem treino",
    "sem rotulo",
    "sem positivos ou negativos formais",
    "sem classe-zero",
    "sem GT operacional",
    "sem tabela multimodal de atributos",
    "evento externo permanece evidencia contextual",
]


@dataclass
class PatchState:
    region: str = "unknown"
    current_dates: set[str] = field(default_factory=set)
    clean_assets: int = 0
    asset_count: int = 0
    asset_type_before: str = "unknown_asset_type"
    cloud_cover_before: str = "missing"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Build REV-P MV1 temporal metadata repair candidates.")
    parser.add_argument("--repo-root", default=".", help="REV-P repository root.")
    return parser


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def is_present(value: Any) -> bool:
    return str(value).strip().lower() not in MISSING_VALUES


def field_value(row: dict[str, Any], aliases: Iterable[str]) -> tuple[str, str]:
    normalized = {normalize_key(key): key for key in row}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key is not None and is_present(row.get(key)):
            return key, str(row.get(key)).strip()
    return "", ""


def all_field_values(row: dict[str, Any], aliases: Iterable[str]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    normalized = {normalize_key(key): key for key in row}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key is not None and is_present(row.get(key)):
            values.append((key, str(row.get(key)).strip()))
    return values


def parse_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        return 0


def parse_float(value: Any) -> float | None:
    if not is_present(value):
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def normalize_region(value: Any) -> str:
    text = str(value).strip()
    mapping = {
        "cur": "Curitiba",
        "curitiba": "Curitiba",
        "pet": "Petropolis",
        "petropolis": "Petropolis",
        "petrópolis": "Petropolis",
        "rec": "Recife",
        "recife": "Recife",
    }
    return mapping.get(text.lower(), text if is_present(text) else "")


def normalize_date(value: Any) -> str:
    text = str(value).strip()
    if not is_present(text):
        return ""
    match = re.search(r"\b(20\d{2})[-_/]?([01]\d)[-_/]?([0-3]\d)(?!\d)", text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def filename_date(value: Any) -> str:
    text = str(value).strip()
    if not is_present(text):
        return ""
    matches = re.findall(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", text)
    normalized = {f"{year}-{month}-{day}" for year, month, day in matches}
    if len(normalized) == 1:
        return next(iter(normalized))
    return ""


def normalize_asset_type(value: str, source_file: str) -> str:
    text = f"{source_file} {value}".lower()
    if "dinov2" in text or "dino" in text or "embedding" in text or "vector_sha256" in text:
        return "dinov2_embedding"
    if "sentinel-1" in text or "sentinel1" in text or re.search(r"\bs1\b", text) or " sar" in text:
        return "sentinel_1_sar"
    if (
        "sentinel-2" in text
        or "sentinel2" in text
        or re.search(r"\bs2\b", text)
        or "optical" in text
        or ("sentinel" in text and ("cloud" in text or "tile" in text or "mgrs" in text))
    ):
        return "sentinel_2_optical"
    if "external_evidence" in text or "evidencia_externa" in text or "official" in text or "gazette" in text:
        return "external_evidence"
    if "patch" in text and "manifest" in text:
        return "patch_manifest"
    return "unknown_asset_type"


def candidate_files(repo_root: Path) -> list[Path]:
    excluded = {
        OUTPUT_TABLE,
        OUTPUT_METRICS,
        OUTPUT_REPORT,
        BACKFILL_TABLE,
        BACKFILL_METRICS,
        BACKFILL_REPORT,
        READINESS_TABLE,
        READINESS_METRICS,
    }
    files: list[Path] = []
    for root in SCAN_ROOTS:
        absolute = repo_root / root
        if not absolute.exists():
            continue
        for path in absolute.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
                continue
            rel = path.relative_to(repo_root)
            text = rel.as_posix().lower()
            if rel in excluded:
                continue
            if any(marker in text for marker in OUT_OF_SCOPE_PATH_MARKERS):
                continue
            if any(keyword in text for keyword in SCAN_KEYWORDS):
                files.append(rel)
    return sorted(files, key=lambda item: item.as_posix())


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def flatten_json_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        scalar = {key: val for key, val in value.items() if not isinstance(val, (dict, list))}
        if scalar:
            records.append(scalar)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                records.extend(flatten_json_records(nested))
    elif isinstance(value, list):
        for item in value:
            records.extend(flatten_json_records(item))
    return records


def read_json(path: Path) -> list[dict[str, Any]]:
    return flatten_json_records(json.loads(path.read_text(encoding="utf-8-sig")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.extend(flatten_json_records(json.loads(text)))
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".json":
        return read_json(path)
    if suffix == ".jsonl":
        return read_jsonl(path)
    return []


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_patch_state(repo_root: Path) -> dict[str, PatchState]:
    states: dict[str, PatchState] = {}
    for row in read_csv(repo_root / BACKFILL_TABLE):
        dates = {
            normalize_date(item)
            for item in str(row.get("current_acquisition_dates", "")).split(";")
            if normalize_date(item)
        }
        patch_id = str(row.get("canonical_patch_id", "")).strip()
        states[patch_id] = PatchState(
            region=str(row.get("region", "unknown")).strip() or "unknown",
            current_dates=dates,
            clean_assets=parse_int(row.get("clean_asset_count")),
            asset_count=parse_int(row.get("asset_count_total")),
            asset_type_before="unknown_asset_type"
            if parse_int(row.get("unknown_asset_count")) > 0
            else "classified_or_mixed",
            cloud_cover_before="missing"
            if row.get("missing_numeric_cloud_cover_s2_only") == "true"
            else "not_required_or_present",
        )
    return states


def base_candidate(
    *,
    patch_id: str,
    asset_id: str,
    source_file: str,
    source_field: str,
    repair_field: str,
    original_value: str,
    candidate_value: str,
    normalized_value: str,
    repair_rule: str,
    repair_confidence: str,
    is_deterministic: bool,
    is_applicable: bool,
    rejection_reason: str,
    state: PatchState,
    asset_type_after: str = "",
    region_after: str = "",
    acquisition_date_after: str = "",
    cloud_cover_after: str = "",
) -> dict[str, str]:
    would_increase_date = repair_field == "acquisition_date" and is_applicable and normalized_value not in state.current_dates
    return {
        "canonical_patch_id": patch_id,
        "asset_id": asset_id or "unknown",
        "source_file": source_file,
        "source_field": source_field or "unknown",
        "repair_field": repair_field,
        "original_value": original_value if is_present(original_value) else "missing",
        "candidate_value": candidate_value if is_present(candidate_value) else "missing",
        "normalized_value": normalized_value if is_present(normalized_value) else "missing",
        "repair_rule": repair_rule,
        "repair_confidence": repair_confidence,
        "is_deterministic": str(is_deterministic).lower(),
        "is_applicable": str(is_applicable).lower(),
        "rejection_reason": rejection_reason,
        "asset_type_before": state.asset_type_before,
        "asset_type_after": asset_type_after or state.asset_type_before,
        "region_before": state.region,
        "region_after": region_after or state.region,
        "acquisition_date_before": ";".join(sorted(state.current_dates)) if state.current_dates else "unknown",
        "acquisition_date_after": acquisition_date_after or (";".join(sorted(state.current_dates)) if state.current_dates else "unknown"),
        "cloud_cover_before": state.cloud_cover_before,
        "cloud_cover_after": cloud_cover_after or state.cloud_cover_before,
        "would_increase_clean_asset_count": str(repair_field in {"acquisition_date", "patch_asset_link"} and is_applicable).lower(),
        "would_increase_clean_date_count": str(would_increase_date).lower(),
        "would_reduce_blocker": str(is_applicable).lower(),
        "guardrail_status": "GUARDRAILS_PRESERVED_REPAIR_CANDIDATE_ONLY",
    }


def conflict_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row["is_applicable"] == "true":
            grouped[(row["canonical_patch_id"], row["repair_field"])].add(row["normalized_value"])
    conflicts = {key for key, values in grouped.items() if len(values) > 1 and key[1] in {"region", "cloud_cover"}}
    output: list[dict[str, str]] = []
    for row in rows:
        if (row["canonical_patch_id"], row["repair_field"]) in conflicts:
            row = dict(row)
            row["repair_rule"] = "REJECT_CONFLICTING_VALUES"
            row["repair_confidence"] = "REJECTED_AMBIGUOUS"
            row["is_deterministic"] = "false"
            row["is_applicable"] = "false"
            row["rejection_reason"] = "AMBIGUOUS_CONFLICT"
            row["would_increase_clean_asset_count"] = "false"
            row["would_increase_clean_date_count"] = "false"
            row["would_reduce_blocker"] = "false"
        output.append(row)
    return output


def collect_candidates(repo_root: Path, patch_states: dict[str, PatchState]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidates: list[dict[str, str]] = []
    input_files: list[str] = []
    skipped_large_files: list[str] = []
    unreadable_files: dict[str, str] = {}
    fields_evaluated: set[str] = set()

    for rel in candidate_files(repo_root):
        path = repo_root / rel
        source = rel.as_posix()
        if path.stat().st_size > MAX_READ_BYTES:
            skipped_large_files.append(source)
            continue
        input_files.append(source)
        if path.suffix.lower() == ".md":
            path.read_text(encoding="utf-8", errors="replace")
            continue
        try:
            rows = read_rows(path)
        except (csv.Error, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            unreadable_files[source] = f"{type(exc).__name__}: {exc}"
            continue
        for row in rows:
            patch_field, patch_id = field_value(row, PATCH_FIELDS)
            if not patch_id or patch_id not in patch_states:
                continue
            state = patch_states[patch_id]
            asset_field, asset_id = field_value(row, ASSET_FIELDS)

            if asset_id:
                fields_evaluated.add("patch_asset_link")
                candidates.append(
                    base_candidate(
                        patch_id=patch_id,
                        asset_id=asset_id,
                        source_file=source,
                        source_field=f"{patch_field}+{asset_field}",
                        repair_field="patch_asset_link",
                        original_value=asset_id,
                        candidate_value=asset_id,
                        normalized_value=f"{patch_id}:{asset_id}",
                        repair_rule="PATCH_ASSET_LINK_FROM_MANIFEST",
                        repair_confidence="HIGH_EXPLICIT_FIELD",
                        is_deterministic=True,
                        is_applicable=True,
                        rejection_reason="NONE",
                        state=state,
                    )
                )

            for region_field, original in all_field_values(row, REGION_FIELDS):
                fields_evaluated.add("region")
                normalized = normalize_region(original)
                applicable = bool(normalized)
                candidates.append(
                    base_candidate(
                        patch_id=patch_id,
                        asset_id=asset_id,
                        source_file=source,
                        source_field=region_field,
                        repair_field="region",
                        original_value=original,
                        candidate_value=original,
                        normalized_value=normalized,
                        repair_rule="EXPLICIT_FIELD_NORMALIZATION" if applicable else "REJECT_MISSING_OR_UNKNOWN",
                        repair_confidence="HIGH_EXPLICIT_FIELD" if applicable else "REJECTED_MISSING_SOURCE",
                        is_deterministic=applicable,
                        is_applicable=applicable,
                        rejection_reason="NONE" if applicable else "MISSING_OR_UNKNOWN",
                        state=state,
                        region_after=normalized,
                    )
                )

            for type_field, original in all_field_values(row, TYPE_FIELDS + ASSET_FIELDS):
                fields_evaluated.add("asset_type")
                normalized = normalize_asset_type(original, source)
                applicable = normalized != "unknown_asset_type"
                candidates.append(
                    base_candidate(
                        patch_id=patch_id,
                        asset_id=asset_id,
                        source_file=source,
                        source_field=type_field,
                        repair_field="asset_type",
                        original_value=original,
                        candidate_value=original,
                        normalized_value=normalized,
                        repair_rule="UNAMBIGUOUS_ASSET_TYPE_PATTERN" if applicable else "REJECT_CONTEXT_ONLY_INFERENCE",
                        repair_confidence="HIGH_EXPLICIT_FIELD" if applicable else "LOW_CONTEXT_ONLY_NOT_APPLICABLE",
                        is_deterministic=applicable,
                        is_applicable=applicable,
                        rejection_reason="NONE" if applicable else "CONTEXT_ONLY_INFERENCE",
                        state=state,
                        asset_type_after=normalized,
                    )
                )

            for date_field, original in all_field_values(row, DATE_FIELDS):
                fields_evaluated.add("acquisition_date")
                normalized = normalize_date(original)
                applicable = bool(normalized)
                candidates.append(
                    base_candidate(
                        patch_id=patch_id,
                        asset_id=asset_id,
                        source_file=source,
                        source_field=date_field,
                        repair_field="acquisition_date",
                        original_value=original,
                        candidate_value=original,
                        normalized_value=normalized,
                        repair_rule="EXPLICIT_FIELD_NORMALIZATION" if applicable else "REJECT_MISSING_OR_UNKNOWN",
                        repair_confidence="HIGH_EXPLICIT_FIELD" if applicable else "REJECTED_MISSING_SOURCE",
                        is_deterministic=applicable,
                        is_applicable=applicable,
                        rejection_reason="NONE" if applicable else "MISSING_OR_UNKNOWN",
                        state=state,
                        acquisition_date_after=normalized,
                    )
                )

            path_values = all_field_values(row, ASSET_FIELDS)
            for path_field, original in path_values:
                fields_evaluated.add("acquisition_date")
                normalized = filename_date(original)
                if not normalized:
                    continue
                candidates.append(
                    base_candidate(
                        patch_id=patch_id,
                        asset_id=asset_id,
                        source_file=source,
                        source_field=path_field,
                        repair_field="acquisition_date",
                        original_value=original,
                        candidate_value=original,
                        normalized_value=normalized,
                        repair_rule="UNAMBIGUOUS_DATE_FROM_FILENAME",
                        repair_confidence="MEDIUM_UNAMBIGUOUS_FILENAME_PATTERN",
                        is_deterministic=True,
                        is_applicable=True,
                        rejection_reason="NONE",
                        state=state,
                        acquisition_date_after=normalized,
                    )
                )

            row_asset_type = normalize_asset_type(" ".join([value for _, value in all_field_values(row, TYPE_FIELDS + ASSET_FIELDS)]), source)
            for cloud_field, original in all_field_values(row, CLOUD_FIELDS):
                fields_evaluated.add("cloud_cover")
                parsed = parse_float(original)
                if row_asset_type != "sentinel_2_optical":
                    candidates.append(
                        base_candidate(
                            patch_id=patch_id,
                            asset_id=asset_id,
                            source_file=source,
                            source_field=cloud_field,
                            repair_field="cloud_cover",
                            original_value=original,
                            candidate_value=original,
                            normalized_value=str(parsed) if parsed is not None else "",
                            repair_rule="REJECT_CONTEXT_ONLY_INFERENCE",
                            repair_confidence="LOW_CONTEXT_ONLY_NOT_APPLICABLE",
                            is_deterministic=False,
                            is_applicable=False,
                            rejection_reason="NOT_REQUIRED_FOR_NON_S2_ASSET",
                            state=state,
                            asset_type_after=row_asset_type,
                        )
                    )
                    continue
                applicable = parsed is not None
                candidates.append(
                    base_candidate(
                        patch_id=patch_id,
                        asset_id=asset_id,
                        source_file=source,
                        source_field=cloud_field,
                        repair_field="cloud_cover",
                        original_value=original,
                        candidate_value=original,
                        normalized_value=str(parsed) if parsed is not None else "",
                        repair_rule="S2_CLOUD_COVER_NUMERIC_FIELD" if applicable else "REJECT_MISSING_OR_UNKNOWN",
                        repair_confidence="HIGH_EXPLICIT_FIELD" if applicable else "REJECTED_MISSING_SOURCE",
                        is_deterministic=applicable,
                        is_applicable=applicable,
                        rejection_reason="NONE" if applicable else "MISSING_OR_UNKNOWN",
                        state=state,
                        asset_type_after=row_asset_type,
                        cloud_cover_after=str(parsed) if parsed is not None else "missing",
                    )
                )

    if not candidates:
        for patch_id, state in patch_states.items():
            candidates.append(
                base_candidate(
                    patch_id=patch_id,
                    asset_id="unknown",
                    source_file="not_found",
                    source_field="not_found",
                    repair_field="acquisition_date",
                    original_value="missing",
                    candidate_value="missing",
                    normalized_value="missing",
                    repair_rule="REJECT_MISSING_OR_UNKNOWN",
                    repair_confidence="REJECTED_MISSING_SOURCE",
                    is_deterministic=False,
                    is_applicable=False,
                    rejection_reason="NO_SOURCE_CANDIDATE_FOUND",
                    state=state,
                )
            )

    meta = {
        "input_files_found": input_files,
        "skipped_large_files": skipped_large_files,
        "unreadable_files": unreadable_files,
        "fields_evaluated": sorted(fields_evaluated),
    }
    return conflict_candidates(candidates), meta


def estimate_impact(rows: list[dict[str, str]], patch_states: dict[str, PatchState]) -> dict[str, Any]:
    candidate_dates_by_patch: dict[str, set[str]] = defaultdict(set)
    patches_with_clean_asset_gain: set[str] = set()
    for row in rows:
        if row["is_applicable"] != "true":
            continue
        patch_id = row["canonical_patch_id"]
        if row["repair_field"] == "acquisition_date":
            candidate_dates_by_patch[patch_id].add(row["normalized_value"])
        if row["would_increase_clean_asset_count"] == "true":
            patches_with_clean_asset_gain.add(patch_id)

    migrate_1 = migrate_2 = migrate_3 = eligible_after = 0
    for patch_id, state in patch_states.items():
        repaired_dates = set(state.current_dates)
        repaired_dates.update(date for date in candidate_dates_by_patch.get(patch_id, set()) if is_present(date))
        total = len(repaired_dates)
        if total >= 3:
            eligible_after += 1
            migrate_3 += 1
        elif total == 2:
            migrate_2 += 1
        elif total == 1:
            migrate_1 += 1

    return {
        "patches_that_could_gain_clean_date": len(candidate_dates_by_patch),
        "patches_that_could_gain_clean_asset": len(patches_with_clean_asset_gain),
        "patches_migrating_to_1_clean_date": migrate_1,
        "patches_migrating_to_2_clean_dates": migrate_2,
        "patches_migrating_to_3plus_clean_dates": migrate_3,
        "estimated_eligible_patches_after_applicable_repair": eligible_after,
    }


def decide_status(rows: list[dict[str, str]], impact: dict[str, Any]) -> str:
    applicable = sum(1 for row in rows if row["is_applicable"] == "true")
    linkage = sum(1 for row in rows if row["repair_field"] == "patch_asset_link" and row["is_applicable"] == "true")
    filename = sum(1 for row in rows if row["repair_rule"] == "UNAMBIGUOUS_DATE_FROM_FILENAME")
    if applicable >= 100 and impact["patches_that_could_gain_clean_date"] >= 20:
        return STEP_READY_NORMALIZED_MANIFEST
    if applicable and linkage < max(1, applicable // 10):
        return STEP_REQUIRE_LINKAGE_REVIEW
    if applicable and filename >= max(1, applicable // 2):
        return STEP_REQUIRE_MANUAL_REVIEW
    if applicable:
        return STEP_REQUIRE_LINKAGE_REVIEW
    return STEP_BLOCKED_INSUFFICIENT_EVIDENCE


def build_metrics(rows: list[dict[str, str]], meta: dict[str, Any], patch_states: dict[str, PatchState]) -> dict[str, Any]:
    impact = estimate_impact(rows, patch_states)
    status = decide_status(rows, impact)
    applicable = sum(1 for row in rows if row["is_applicable"] == "true")
    rejected = len(rows) - applicable
    blockers = Counter(row["rejection_reason"] for row in rows if row["is_applicable"] != "true")
    return {
        "total_candidates_evaluated": len(rows),
        "total_applicable": applicable,
        "total_rejected": rejected,
        "counts_by_repair_field": dict(sorted(Counter(row["repair_field"] for row in rows).items())),
        "counts_by_applicable_repair_field": dict(
            sorted(Counter(row["repair_field"] for row in rows if row["is_applicable"] == "true").items())
        ),
        "counts_by_repair_confidence": dict(sorted(Counter(row["repair_confidence"] for row in rows).items())),
        "counts_by_repair_rule": dict(sorted(Counter(row["repair_rule"] for row in rows).items())),
        "counts_by_region": dict(sorted(Counter(row["region_after"] for row in rows if row["region_after"]).items())),
        "patches_that_could_gain_clean_date": impact["patches_that_could_gain_clean_date"],
        "patches_migrating_to_1_clean_date": impact["patches_migrating_to_1_clean_date"],
        "patches_migrating_to_2_clean_dates": impact["patches_migrating_to_2_clean_dates"],
        "patches_migrating_to_3plus_clean_dates": impact["patches_migrating_to_3plus_clean_dates"],
        "estimated_eligible_patches_after_applicable_repair": impact["estimated_eligible_patches_after_applicable_repair"],
        "remaining_blockers": dict(blockers.most_common()),
        "step_a_repair_status": status,
        "guardrails_preserved": GUARDRAILS,
        "input_files_found": [
            READINESS_TABLE.as_posix(),
            READINESS_METRICS.as_posix(),
            BACKFILL_TABLE.as_posix(),
            BACKFILL_METRICS.as_posix(),
            BACKFILL_REPORT.as_posix(),
            *meta["input_files_found"],
        ],
        "fields_evaluated": meta["fields_evaluated"],
        "skipped_large_files": meta["skipped_large_files"],
        "unreadable_files": meta["unreadable_files"],
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
    lines = [
        "# revp_temporal_metadata_repair_candidates_mv1",
        "",
        "## 1. Escopo",
        "Camada publica e deterministica de candidatos de reparo de metadados temporais para a Vertente A.",
        "",
        "## 2. Relacao com readiness audit e backfill queue",
        f"- Readiness: `{READINESS_TABLE.as_posix()}`",
        f"- Backfill: `{BACKFILL_TABLE.as_posix()}`",
        "",
        "## 3. Por que esta etapa nao aplica reparos ainda",
        "- Esta saida registra candidatos e rejeicoes. Nenhum manifesto original e alterado.",
        "",
        "## 4. Arquivos de entrada encontrados",
        *[f"- `{item}`" for item in metrics["input_files_found"]],
        "",
        "## 5. Campos avaliados",
        *[f"- {field}" for field in metrics["fields_evaluated"]],
        "",
        "## 6. Regras de extracao",
        "- Datas apenas de campo temporal explicito ou filename/path com uma unica data inequivoca.",
        "- Regiao apenas de campo explicito ou correspondencia unica.",
        "- Tipo de asset apenas por campo, manifesto ou padrao inequivoco.",
        "- Cloud cover numerico apenas para Sentinel-2 optico.",
        "",
        "## 7. Regras de rejeicao",
        "- Valores unknown/missing/vazios nao viram valor valido.",
        "- Conflitos multiplos viram AMBIGUOUS_CONFLICT.",
        "- Contexto fraco nao vira reparo aplicavel.",
        "",
        "## 8. Candidatos aplicaveis por campo",
        *[f"- {field}: {count}" for field, count in sorted(metrics["counts_by_applicable_repair_field"].items())],
        "",
        "## 9. Candidatos rejeitados por motivo",
        *[f"- {reason}: {count}" for reason, count in sorted(metrics["remaining_blockers"].items())],
        "",
        "## 10. Impacto potencial na Vertente A",
        f"- Patches que poderiam ganhar data limpa: {metrics['patches_that_could_gain_clean_date']}",
        f"- Migrariam para 1 data limpa: {metrics['patches_migrating_to_1_clean_date']}",
        f"- Migrariam para 2 datas limpas: {metrics['patches_migrating_to_2_clean_dates']}",
        f"- Migrariam para 3+ datas limpas: {metrics['patches_migrating_to_3plus_clean_dates']}",
        f"- Estimativa elegivel apos reparo aplicavel: {metrics['estimated_eligible_patches_after_applicable_repair']}",
        "",
        "## 11. Decisao sobre proxima etapa",
        f"- `step_a_repair_status`: `{metrics['step_a_repair_status']}`",
        "",
        "## 12. Limitacoes",
        "- A estimativa nao substitui um manifesto temporal normalizado.",
        "- Candidatos conflitantes permanecem bloqueados.",
        "",
        "## 13. Guardrails preservados",
        f"- Branch auditada: `{branch_name(repo_root)}`",
        *[f"- {item}" for item in GUARDRAILS],
        "",
    ]
    return "\n".join(lines)


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    patch_states = load_patch_state(root)
    rows, meta = collect_candidates(root, patch_states)
    rows = sorted(rows, key=lambda row: (row["canonical_patch_id"], row["repair_field"], row["source_file"], row["source_field"], row["normalized_value"]))
    metrics = build_metrics(rows, meta, patch_states)

    write_csv(root / OUTPUT_TABLE, rows)
    write_json(root / OUTPUT_METRICS, metrics)
    (root / OUTPUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUT_REPORT).write_text(build_report(root, metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    execute(Path(args.repo_root))


if __name__ == "__main__":
    main()
