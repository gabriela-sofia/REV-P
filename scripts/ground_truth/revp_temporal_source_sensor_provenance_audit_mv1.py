from __future__ import annotations

import csv
import hashlib
import json
import re
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SENSOR_RESOLUTION_INPUT = Path("outputs_public/tables/revp_temporal_sensor_family_resolution_mv1.csv")
RESOLVED_REQUIREMENTS_INPUT = Path("outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv")
REQUEST_MANIFEST_INPUT = Path("outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv")
ASSET_MANIFEST_INPUT = Path("outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv")
PATCH_COVERAGE_INPUT = Path("outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv")
REPAIR_CANDIDATES_INPUT = Path("outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv")

PROVENANCE_OUTPUT = Path("outputs_public/tables/revp_temporal_source_sensor_provenance_mv1.csv")
BLOCKERS_OUTPUT = Path("outputs_public/tables/revp_temporal_source_sensor_blockers_mv1.csv")
METRICS_OUTPUT = Path("outputs_public/metrics/revp_temporal_source_sensor_provenance_mv1.json")
REPORT_OUTPUT = Path("outputs_public/execution_reports/revp_temporal_source_sensor_provenance_mv1.md")

PROVENANCE_COLUMNS = [
    "canonical_patch_id",
    "region",
    "asset_ref_id",
    "asset_type",
    "derived_asset_type",
    "source_asset_ref_id",
    "source_sensor_family",
    "source_acquisition_date",
    "source_cloud_cover",
    "source_link_status",
    "source_link_rule",
    "source_link_confidence",
    "source_file",
    "source_field",
    "evidence_value_sanitized",
    "has_explicit_source_asset",
    "has_explicit_source_sensor",
    "has_explicit_source_date",
    "is_dinov2_derived_asset",
    "is_source_sensor_resolved",
    "would_unblock_sensor_family_resolution",
    "would_unblock_backfill_request",
    "blocking_reason",
    "guardrail_status",
]

BLOCKER_COLUMNS = [
    "canonical_patch_id",
    "region",
    "asset_ref_id",
    "asset_type",
    "blocking_category",
    "blocking_reason",
    "required_evidence_to_unblock",
    "affected_requirement_count",
    "affected_request_count",
    "can_be_repaired_from_existing_manifest",
    "requires_manual_review",
    "requires_external_search_later",
    "guardrail_status",
]

GUARDRAIL_STATUS = "GUARDRAILS_PRESERVED_SOURCE_SENSOR_PROVENANCE_AUDIT_ONLY"

GUARDRAILS = [
    "sem download de assets",
    "sem consulta STAC API ou GEE",
    "sem gerar URLs de download",
    "sem criar data real ou sintetica",
    "sem calcular embedding drift",
    "sem gerar embeddings novos",
    "sem treino",
    "sem rotulo",
    "sem criar evidencia de ausencia supervisionada",
    "sem classe-zero",
    "sem GT operacional",
    "sem tabela multimodal de atributos",
    "sem alterar manifestos originais",
    "sem alterar outputs anteriores",
]

SCAN_ROOTS = [
    Path("manifests"),
    Path("configs"),
]

EXCLUDED_SCAN_PARTS = {
    "revp_temporal_source_sensor_provenance_mv1.csv",
    "revp_temporal_source_sensor_blockers_mv1.csv",
    "revp_temporal_source_sensor_provenance_mv1.json",
    "revp_temporal_source_sensor_provenance_mv1.md",
}

SOURCE_ASSET_FIELDS = [
    "source_asset_id",
    "input_asset_id",
    "source_product_id",
    "sentinel_product_id",
    "product_id",
    "source_scene_id",
    "sentinel_scene_id",
    "scene_id",
]

SOURCE_SENSOR_FIELDS = [
    "source_sensor_family",
    "source_sensor",
    "sensor_family",
    "sensor",
    "source_asset_type",
    "source_modality",
    "modality",
    "collection",
    "platform",
    "asset_path_reference",
    "asset_path",
    "source_product_id",
    "sentinel_product_id",
    "product_id",
    "source_scene_id",
    "sentinel_scene_id",
    "scene_id",
]

SOURCE_DATE_FIELDS = [
    "source_acquisition_date",
    "acquisition_date",
    "source_datetime",
    "datetime",
    "selected_datetime",
    "date",
]

SOURCE_CLOUD_FIELDS = [
    "source_cloud_cover",
    "cloud_cover",
    "selected_cloudy_pixel_percentage",
    "selected_cloud_coverage_assessment",
]

BATCH_TARGETS = {
    "recoverable_1_patch": 1,
    "recoverable_20_patches": 20,
    "recoverable_30_patches": 30,
}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Audit REV-P MV1 temporal source sensor provenance offline.")
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


def stable_ref(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def public_asset_ref(row: dict[str, str]) -> str:
    raw = "|".join(
        [
            row.get("canonical_patch_id", ""),
            row.get("asset_id", ""),
            row.get("normalized_asset_id", ""),
            row.get("asset_type", ""),
        ]
    )
    return stable_ref("asset_ref", raw)


def source_ref(value: str) -> str:
    if not value or value.strip().lower() in {"unknown", "missing", "none", "null", "n/a"}:
        return "unknown_source_ref"
    return stable_ref("source_ref", value)


def file_ref(path: Path) -> str:
    return stable_ref("file_ref", path.as_posix())


def sanitize_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "missing"
    if re.search(r"[A-Za-z]:\\|\\\\|/Users/|/home/", text):
        return stable_ref("value_ref", text)
    if len(text) > 80:
        return stable_ref("value_ref", text)
    safe = re.sub(r"[^A-Za-z0-9_.:/=-]+", "_", text)
    if "label" in safe.lower() or "negative" in safe.lower() or "ground_truth" in safe.lower():
        return stable_ref("value_ref", text)
    return safe or "missing"


def normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "missing", "none", "null", "n/a"}:
        return "unknown"
    match = re.search(r"(20\d{2}|19\d{2})[-_/]([01]\d)[-_/]([0-3]\d)", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return "unknown"


def infer_sensor_family(values: list[str]) -> set[str]:
    sensors: set[str] = set()
    for value in values:
        lower = value.lower()
        if re.search(r"\bs2[ab]?_|sentinel[-_ ]?2|copernicus/s2|msil[12]c|sentinel-2", lower):
            sensors.add("sentinel_2_optical")
        if re.search(r"\bs1[ab]?_|sentinel[-_ ]?1|sentinel-1|\bsar\b|\biw_grd\b", lower):
            sensors.add("sentinel_1_sar")
        if "dinov2" in lower or "embedding" in lower or "external_evidence" in lower or "patch_manifest" in lower:
            sensors.add("non_temporal_derived_asset")
        if "sentinel" in lower and "sentinel_1" not in sensors and "sentinel_2" not in sensors:
            sensors.add("ambiguous_source_sensor")
    return sensors


def row_values(row: dict[str, str], fields: list[str]) -> list[str]:
    return [row.get(field, "") for field in fields if row.get(field, "").strip()]


def evidence_matches_asset(asset: dict[str, str], evidence: dict[str, str]) -> bool:
    patch_id = asset.get("canonical_patch_id", "")
    if patch_id and patch_id not in row_values(
        evidence,
        ["canonical_patch_id", "candidate_id", "patch_id", "target_id", "patch"],
    ):
        return False
    asset_ids = {
        asset.get("asset_id", ""),
        asset.get("normalized_asset_id", ""),
        public_asset_ref(asset),
    }
    asset_ids = {item for item in asset_ids if item}
    evidence_ids = set(
        row_values(
            evidence,
            [
                "asset_id",
                "source_asset_id",
                "input_asset_id",
                "derived_asset_id",
                "embedding_asset_id",
                "normalized_asset_id",
                "target_asset_id",
                "source_product_id",
                "sentinel_product_id",
                "product_id",
            ],
        )
    )
    return bool(asset_ids & evidence_ids)


def candidate_evidence_from_asset_row(asset: dict[str, str]) -> list[dict[str, str]]:
    values = row_values(asset, SOURCE_ASSET_FIELDS + SOURCE_SENSOR_FIELDS + SOURCE_DATE_FIELDS)
    if not values:
        return []
    direct: dict[str, str] = dict(asset)
    direct["_source_file_ref"] = "normalized_asset_manifest_row"
    direct["_source_field"] = "direct_asset_row"
    direct["_evidence_value"] = "|".join(values)
    return [direct]


def referenced_paths(repo_root: Path, asset_rows: list[dict[str, str]]) -> set[Path]:
    paths: set[Path] = set()
    for row in asset_rows:
        for raw in row.get("source_files", "").split("|"):
            item = raw.strip()
            if not item or re.match(r"[A-Za-z]:\\|\\\\", item):
                continue
            path = (repo_root / item).resolve()
            try:
                path.relative_to(repo_root)
            except ValueError:
                continue
            if path.exists() and path.suffix.lower() in {".csv", ".json"}:
                paths.add(path)
    return paths


def discover_scan_files(repo_root: Path, asset_rows: list[dict[str, str]]) -> list[Path]:
    files = referenced_paths(repo_root, asset_rows)
    for root_rel in SCAN_ROOTS:
        root = repo_root / root_rel
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name in EXCLUDED_SCAN_PARTS:
                continue
            if path.suffix.lower() not in {".csv", ".json"}:
                continue
            try:
                if path.stat().st_size > 8_000_000:
                    continue
            except OSError:
                continue
            files.add(path.resolve())
    return sorted(files, key=lambda item: item.as_posix())


def load_csv_evidence(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            useful_fields = {
                *SOURCE_ASSET_FIELDS,
                *SOURCE_SENSOR_FIELDS,
                *SOURCE_DATE_FIELDS,
                "canonical_patch_id",
                "candidate_id",
                "patch_id",
                "target_id",
                "asset_id",
                "derived_asset_id",
                "embedding_asset_id",
                "normalized_asset_id",
            }
            if not useful_fields.intersection(set(reader.fieldnames)):
                return []
            for row in reader:
                row["_source_file_ref"] = file_ref(path)
                row["_source_field"] = "csv_row"
                row["_evidence_value"] = "|".join(
                    row.get(field, "") for field in reader.fieldnames if any(token in field.lower() for token in ["source", "sensor", "sentinel", "asset", "scene", "product", "date", "time", "cloud"])
                )
                rows.append({key: str(value or "") for key, value in row.items()})
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    return rows


def flatten_json(item: Any, prefix: str = "") -> list[dict[str, str]]:
    if isinstance(item, dict):
        flat: dict[str, str] = {}
        nested_rows: list[dict[str, str]] = []
        for key, value in item.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                nested_rows.extend(flatten_json(value, name))
            else:
                flat[name] = str(value or "")
                flat[str(key)] = str(value or "")
        return [flat, *nested_rows]
    if isinstance(item, list):
        rows: list[dict[str, str]] = []
        for value in item:
            rows.extend(flatten_json(value, prefix))
        return rows
    return []


def load_json_evidence(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    rows = flatten_json(payload)
    output: list[dict[str, str]] = []
    for row in rows:
        values = row_values(row, SOURCE_ASSET_FIELDS + SOURCE_SENSOR_FIELDS + SOURCE_DATE_FIELDS)
        if not values:
            continue
        row["_source_file_ref"] = file_ref(path)
        row["_source_field"] = "json_node"
        row["_evidence_value"] = "|".join(values)
        output.append(row)
    return output


def load_evidence_rows(repo_root: Path, asset_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in discover_scan_files(repo_root, asset_rows):
        if path.suffix.lower() == ".csv":
            rows.extend(load_csv_evidence(path))
        elif path.suffix.lower() == ".json":
            rows.extend(load_json_evidence(path))
    return rows


def build_evidence_index(evidence_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    fields = [
        "asset_id",
        "source_asset_id",
        "input_asset_id",
        "derived_asset_id",
        "embedding_asset_id",
        "normalized_asset_id",
        "target_asset_id",
        "source_product_id",
        "sentinel_product_id",
        "product_id",
    ]
    for row in evidence_rows:
        for value in row_values(row, fields):
            index[value].append(row)
    return index


def choose_source_evidence(asset: dict[str, str], evidence_index: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    matches = candidate_evidence_from_asset_row(asset)
    candidates: list[dict[str, str]] = []
    for value in [asset.get("asset_id", ""), asset.get("normalized_asset_id", ""), public_asset_ref(asset)]:
        if value:
            candidates.extend(evidence_index.get(value, []))
    for evidence in candidates:
        if evidence_matches_asset(asset, evidence):
            matches.append(evidence)
    unique: dict[str, dict[str, str]] = {}
    for row in matches:
        key = "|".join(
            [
                row.get("_source_file_ref", ""),
                row.get("source_asset_id", ""),
                row.get("input_asset_id", ""),
                row.get("asset_id", ""),
                row.get("source_sensor_family", ""),
                row.get("source_asset_type", ""),
                row.get("asset_path_reference", ""),
                row.get("asset_path", ""),
            ]
        )
        unique[key] = row
    return list(unique.values())


def best_source_asset_id(matches: list[dict[str, str]]) -> str:
    for row in matches:
        for field in SOURCE_ASSET_FIELDS:
            value = row.get(field, "").strip()
            if value:
                return value
    return ""


def best_date(matches: list[dict[str, str]]) -> str:
    for row in matches:
        for field in SOURCE_DATE_FIELDS:
            date = normalize_date(row.get(field, ""))
            if date != "unknown":
                return date
    return "unknown"


def best_cloud(matches: list[dict[str, str]]) -> str:
    for row in matches:
        for field in SOURCE_CLOUD_FIELDS:
            value = str(row.get(field, "")).strip()
            if value and value.lower() not in {"unknown", "missing", "none", "null", "n/a"}:
                return sanitize_value(value)
    return "unknown"


def source_field_and_value(matches: list[dict[str, str]]) -> tuple[str, str, str]:
    for row in matches:
        for field in [*SOURCE_SENSOR_FIELDS, *SOURCE_ASSET_FIELDS, *SOURCE_DATE_FIELDS]:
            value = row.get(field, "").strip()
            if value:
                return row.get("_source_file_ref", "unknown_file_ref"), field, sanitize_value(value)
    return "unknown_file_ref", "none", "missing"


def resolve_provenance(asset: dict[str, str], matches: list[dict[str, str]]) -> dict[str, str]:
    asset_type = asset.get("asset_type", "unknown") or "unknown"
    is_dino = asset_type == "dinov2_embedding"
    values: list[str] = []
    for row in matches:
        values.extend(row_values(row, SOURCE_SENSOR_FIELDS + SOURCE_ASSET_FIELDS))
    sensors = infer_sensor_family(values)
    explicit_source_asset = bool(best_source_asset_id(matches))
    explicit_date = best_date(matches) != "unknown"
    explicit_sensor = bool(sensors - {"ambiguous_source_sensor", "non_temporal_derived_asset"})

    if "sentinel_1_sar" in sensors and "sentinel_2_optical" in sensors:
        family = "ambiguous_source_sensor"
        status = "BLOCKED_AMBIGUOUS_SOURCE_SENSOR"
        confidence = "REJECTED_AMBIGUOUS"
        rule = "CONFLICTING_SENTINEL_1_AND_SENTINEL_2_SOURCE_EVIDENCE"
        blocker = "CONFLICTING_SOURCE_SENSOR"
    elif "sentinel_2_optical" in sensors:
        family = "sentinel_2_optical"
        if explicit_date:
            status = "SOURCE_CONFIRMED_SENTINEL_2"
            confidence = "HIGH_EXPLICIT_SOURCE_FIELD" if explicit_source_asset else "MEDIUM_UNAMBIGUOUS_SENTINEL_PATTERN"
            rule = "EXPLICIT_SOURCE_SENSOR_AND_DATE_FOR_SENTINEL_2"
            blocker = "NONE"
        else:
            status = "BLOCKED_MISSING_SOURCE_LINK"
            confidence = "REJECTED_CONTEXT_ONLY"
            rule = "SOURCE_SENSOR_PRESENT_BUT_SOURCE_DATE_MISSING"
            blocker = "MISSING_SOURCE_DATE"
    elif "sentinel_1_sar" in sensors:
        family = "sentinel_1_sar"
        if explicit_date:
            status = "SOURCE_CONFIRMED_SENTINEL_1"
            confidence = "HIGH_EXPLICIT_SOURCE_FIELD" if explicit_source_asset else "MEDIUM_UNAMBIGUOUS_SENTINEL_PATTERN"
            rule = "EXPLICIT_SOURCE_SENSOR_AND_DATE_FOR_SENTINEL_1"
            blocker = "SENTINEL_1_SUPPORT_ONLY"
        else:
            status = "BLOCKED_MISSING_SOURCE_LINK"
            confidence = "REJECTED_CONTEXT_ONLY"
            rule = "SOURCE_SENSOR_PRESENT_BUT_SOURCE_DATE_MISSING"
            blocker = "MISSING_SOURCE_DATE"
    elif "ambiguous_source_sensor" in sensors:
        family = "ambiguous_source_sensor"
        status = "BLOCKED_AMBIGUOUS_SOURCE_SENSOR"
        confidence = "REJECTED_AMBIGUOUS"
        rule = "GENERIC_SENTINEL_SOURCE_WITHOUT_S1_OR_S2_FAMILY"
        blocker = "SOURCE_SENSOR_FAMILY_NOT_EXPLICIT"
    elif asset_type == "unknown":
        family = "unknown_source_sensor"
        status = "BLOCKED_UNKNOWN_ASSET_TYPE"
        confidence = "REJECTED_UNKNOWN"
        rule = "UNKNOWN_ASSET_TYPE_WITHOUT_EXPLICIT_SOURCE_SENSOR"
        blocker = "UNKNOWN_ASSET_TYPE"
    elif is_dino:
        family = "unknown_source_sensor"
        status = "BLOCKED_DINO_WITHOUT_EXPLICIT_SOURCE"
        confidence = "REJECTED_DERIVED_WITHOUT_SOURCE"
        rule = "DINOV2_DERIVED_ASSET_WITHOUT_EXPLICIT_SENTINEL_SOURCE_SENSOR"
        blocker = "DINO_DERIVED_WITHOUT_EXPLICIT_SOURCE_SENSOR"
    elif "non_temporal_derived_asset" in sensors or asset_type in {"patch_manifest", "external_evidence"}:
        family = "non_temporal_derived_asset"
        status = "SOURCE_CONFIRMED_NON_TEMPORAL_DERIVED"
        confidence = "REJECTED_CONTEXT_ONLY"
        rule = "NON_TEMPORAL_DERIVED_OR_METADATA_ASSET"
        blocker = "NON_TEMPORAL_DERIVED_ASSET"
    else:
        family = "unknown_source_sensor"
        status = "BLOCKED_MISSING_SOURCE_LINK"
        confidence = "REJECTED_CONTEXT_ONLY"
        rule = "NO_EXPLICIT_SOURCE_SENSOR_LINK_FOUND"
        blocker = "MISSING_SOURCE_LINK"

    source_file, source_field, evidence_value = source_field_and_value(matches)
    source_date = best_date(matches)
    source_cloud = best_cloud(matches)
    has_explicit_source_asset = str(explicit_source_asset).lower()
    has_explicit_source_sensor = str(explicit_sensor).lower()
    has_explicit_source_date = str(source_date != "unknown").lower()
    is_source_resolved = str(status in {"SOURCE_CONFIRMED_SENTINEL_2", "SOURCE_CONFIRMED_SENTINEL_1"}).lower()
    would_unblock_resolution = str(status == "SOURCE_CONFIRMED_SENTINEL_2").lower()

    return {
        "source_asset_ref_id": source_ref(best_source_asset_id(matches)),
        "source_sensor_family": family,
        "source_acquisition_date": source_date,
        "source_cloud_cover": source_cloud,
        "source_link_status": status,
        "source_link_rule": rule,
        "source_link_confidence": confidence,
        "source_file": source_file,
        "source_field": source_field,
        "evidence_value_sanitized": evidence_value,
        "has_explicit_source_asset": has_explicit_source_asset,
        "has_explicit_source_sensor": has_explicit_source_sensor,
        "has_explicit_source_date": has_explicit_source_date,
        "is_dinov2_derived_asset": str(is_dino).lower(),
        "is_source_sensor_resolved": is_source_resolved,
        "would_unblock_sensor_family_resolution": would_unblock_resolution,
        "blocking_reason": blocker,
    }


def build_count_by_patch(rows: list[dict[str, str]], actionable_field: str | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if actionable_field and row.get(actionable_field) != "true":
            continue
        patch_id = row.get("canonical_patch_id", "")
        if patch_id:
            counts[patch_id] += 1
    return counts


def build_provenance_rows(
    asset_rows: list[dict[str, str]],
    evidence_index: dict[str, list[dict[str, str]]],
    requirement_counts: Counter[str],
    request_counts: Counter[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for asset in sorted(asset_rows, key=lambda row: (row.get("canonical_patch_id", ""), row.get("asset_type", ""), row.get("asset_id", ""))):
        matches = choose_source_evidence(asset, evidence_index)
        provenance = resolve_provenance(asset, matches)
        patch_id = asset.get("canonical_patch_id", "")
        would_unblock_request = (
            provenance["would_unblock_sensor_family_resolution"] == "true"
            and requirement_counts.get(patch_id, 0) > 0
            and request_counts.get(patch_id, 0) > 0
        )
        rows.append(
            {
                "canonical_patch_id": patch_id,
                "region": asset.get("region", "unknown") or "unknown",
                "asset_ref_id": public_asset_ref(asset),
                "asset_type": asset.get("asset_type", "unknown") or "unknown",
                "derived_asset_type": "dinov2_derived_embedding" if asset.get("asset_type") == "dinov2_embedding" else asset.get("asset_type", "unknown"),
                **provenance,
                "would_unblock_backfill_request": str(would_unblock_request).lower(),
                "guardrail_status": GUARDRAIL_STATUS,
            }
        )
    return rows


def blocker_category(row: dict[str, str]) -> str:
    mapping = {
        "BLOCKED_DINO_WITHOUT_EXPLICIT_SOURCE": "DINO_WITHOUT_EXPLICIT_SOURCE",
        "BLOCKED_UNKNOWN_ASSET_TYPE": "UNKNOWN_ASSET_TYPE",
        "BLOCKED_AMBIGUOUS_SOURCE_SENSOR": "AMBIGUOUS_SOURCE_SENSOR",
        "BLOCKED_MISSING_SOURCE_LINK": "MISSING_SOURCE_LINK",
        "BLOCKED_CONTEXT_ONLY_INFERENCE": "CONTEXT_ONLY_INFERENCE",
        "SOURCE_CONFIRMED_NON_TEMPORAL_DERIVED": "NON_TEMPORAL_DERIVED_ASSET",
    }
    return mapping.get(row["source_link_status"], "SOURCE_SENSOR_NOT_USABLE_FOR_BACKFILL")


def required_evidence(row: dict[str, str]) -> str:
    if row["source_link_status"] == "BLOCKED_DINO_WITHOUT_EXPLICIT_SOURCE":
        return "explicit_source_asset|explicit_source_sensor_family|explicit_source_acquisition_date"
    if row["source_link_status"] == "BLOCKED_UNKNOWN_ASSET_TYPE":
        return "explicit_asset_type|explicit_source_sensor_family|explicit_source_acquisition_date"
    if row["source_link_status"] == "BLOCKED_AMBIGUOUS_SOURCE_SENSOR":
        return "unique_sentinel_1_or_sentinel_2_source_family|source_acquisition_date"
    if row["source_link_status"] == "BLOCKED_MISSING_SOURCE_LINK":
        return "explicit_source_asset|source_sensor_family|source_acquisition_date"
    return "manual_review_to_confirm_temporal_source_sensor"


def build_blocker_rows(
    provenance_rows: list[dict[str, str]],
    requirement_counts: Counter[str],
    request_counts: Counter[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in provenance_rows:
        if row["would_unblock_backfill_request"] == "true":
            continue
        patch_id = row["canonical_patch_id"]
        status = row["source_link_status"]
        rows.append(
            {
                "canonical_patch_id": patch_id,
                "region": row["region"],
                "asset_ref_id": row["asset_ref_id"],
                "asset_type": row["asset_type"],
                "blocking_category": blocker_category(row),
                "blocking_reason": row["blocking_reason"],
                "required_evidence_to_unblock": required_evidence(row),
                "affected_requirement_count": str(requirement_counts.get(patch_id, 0)),
                "affected_request_count": str(request_counts.get(patch_id, 0)),
                "can_be_repaired_from_existing_manifest": str(status in {"BLOCKED_AMBIGUOUS_SOURCE_SENSOR", "BLOCKED_MISSING_SOURCE_LINK"} and row["has_explicit_source_asset"] == "true").lower(),
                "requires_manual_review": str(status != "SOURCE_CONFIRMED_SENTINEL_2").lower(),
                "requires_external_search_later": str(status in {"BLOCKED_DINO_WITHOUT_EXPLICIT_SOURCE", "BLOCKED_UNKNOWN_ASSET_TYPE", "BLOCKED_MISSING_SOURCE_LINK"}).lower(),
                "guardrail_status": GUARDRAIL_STATUS,
            }
        )
    return rows


def actionable_patch_groups(
    provenance_rows: list[dict[str, str]],
    requirement_counts: Counter[str],
) -> list[dict[str, Any]]:
    by_patch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in provenance_rows:
        if row["would_unblock_sensor_family_resolution"] == "true":
            by_patch[row["canonical_patch_id"]].append(row)
    groups: list[dict[str, Any]] = []
    for patch_id, rows in by_patch.items():
        required = requirement_counts.get(patch_id, 0)
        if required > 0 and len(rows) >= required:
            groups.append({"canonical_patch_id": patch_id, "required_slots": required})
    return sorted(groups, key=lambda item: item["canonical_patch_id"])


def recoverable_batches(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key, target in BATCH_TARGETS.items():
        selected = groups[:target]
        recoverable = len(selected) >= target
        output[key] = {
            "target_patch_count": target,
            "recoverable": recoverable,
            "selected_patch_count": len(selected) if recoverable else 0,
            "affected_requirement_count": sum(item["required_slots"] for item in selected) if recoverable else 0,
        }
    return output


def step_status(groups: list[dict[str, Any]], provenance_rows: list[dict[str, str]]) -> str:
    if len(groups) >= 30:
        return "STEP_A_SOURCE_PROVENANCE_READY_FOR_30_PATCH_SENSOR_REBUILD"
    if len(groups) >= 20:
        return "STEP_A_SOURCE_PROVENANCE_READY_FOR_20_PATCH_SENSOR_REBUILD"
    if len(groups) >= 1:
        return "STEP_A_SOURCE_PROVENANCE_READY_FOR_SINGLE_PATCH_SENSOR_REBUILD"
    if any(row["has_explicit_source_asset"] == "true" for row in provenance_rows):
        return "STEP_A_SOURCE_PROVENANCE_PARTIAL_ONLY"
    return "STEP_A_SOURCE_PROVENANCE_BLOCKED"


def build_metrics(
    provenance_rows: list[dict[str, str]],
    blocker_rows: list[dict[str, str]],
    requirement_counts: Counter[str],
) -> dict[str, Any]:
    groups = actionable_patch_groups(provenance_rows, requirement_counts)
    status_counts = Counter(row["source_link_status"] for row in provenance_rows)
    dino_rows = [row for row in provenance_rows if row["is_dinov2_derived_asset"] == "true"]
    unknown_rows = [row for row in provenance_rows if row["asset_type"] == "unknown"]
    return {
        "total_assets_evaluated": len(provenance_rows),
        "total_with_source_sensor_resolved": sum(row["is_source_sensor_resolved"] == "true" for row in provenance_rows),
        "total_source_sentinel_2_resolved": status_counts["SOURCE_CONFIRMED_SENTINEL_2"],
        "total_source_sentinel_1_resolved": status_counts["SOURCE_CONFIRMED_SENTINEL_1"],
        "dinov2_with_explicit_source": sum(row["has_explicit_source_asset"] == "true" for row in dino_rows),
        "dinov2_without_explicit_source": sum(row["has_explicit_source_asset"] != "true" for row in dino_rows),
        "unknowns_resolved": sum(row["is_source_sensor_resolved"] == "true" for row in unknown_rows),
        "unknowns_blocked": sum(row["is_source_sensor_resolved"] != "true" for row in unknown_rows),
        "blockers_by_category": dict(sorted(Counter(row["blocking_category"] for row in blocker_rows).items())),
        "slots_that_could_be_unblocked": sum(
            requirement_counts.get(row["canonical_patch_id"], 0)
            for row in provenance_rows
            if row["would_unblock_sensor_family_resolution"] == "true"
        ),
        "patches_that_could_become_actionable": len(groups),
        "batches_potentially_recoverable": recoverable_batches(groups),
        "counts_by_source_link_status": dict(sorted(status_counts.items())),
        "step_a_source_sensor_provenance_status": step_status(groups, provenance_rows),
        "guardrails_preserved": GUARDRAILS,
        "input_files": [
            SENSOR_RESOLUTION_INPUT.as_posix(),
            RESOLVED_REQUIREMENTS_INPUT.as_posix(),
            REQUEST_MANIFEST_INPUT.as_posix(),
            ASSET_MANIFEST_INPUT.as_posix(),
            PATCH_COVERAGE_INPUT.as_posix(),
            REPAIR_CANDIDATES_INPUT.as_posix(),
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
            "# revp_temporal_source_sensor_provenance_mv1",
            "",
            "## 1. Escopo",
            "Auditoria publica, offline e deterministica de proveniencia sensor-fonte para assets temporais, DINOv2 e unknown.",
            "",
            "## 2. Relacao com o bloqueio da resolucao de sensor",
            "- A etapa anterior ficou bloqueada porque nenhum slot tinha familia Sentinel-2 resolvida.",
            f"- Status desta auditoria: `{metrics['step_a_source_sensor_provenance_status']}`.",
            "",
            "## 3. Por que DINOv2 nao e sensor de aquisicao",
            "- DINOv2 e derivado. Ele so pode carregar proveniencia Sentinel quando um manifesto explicito conecta o embedding ao asset fonte.",
            "- Mesmo com link explicito, o embedding nao vira sensor fisico de aquisicao.",
            "",
            "## 4. Arquivos de entrada",
            *[f"- `{item}`" for item in metrics["input_files"]],
            "",
            "## 5. Regras de proveniencia aceitas",
            "- Campo direto de source/input asset.",
            "- Campo direto de produto, cena, tile ou familia Sentinel.",
            "- Manifesto DINO que declare input Sentinel com patch e source asset unicos.",
            "- Padrao Sentinel inequivoco quando nao ha conflito.",
            "",
            "## 6. Regras de rejeicao",
            "- Contexto fraco ou diretorio generico nao resolve sensor.",
            "- Sensor ausente, data fonte ausente, conflito S1/S2 ou unknown permanecem bloqueados.",
            "- DINOv2 sem fonte Sentinel explicita permanece bloqueado.",
            "",
            "## 7. Source sensors resolvidos",
            f"- Total resolvido: {metrics['total_with_source_sensor_resolved']}",
            f"- Sentinel-2: {metrics['total_source_sentinel_2_resolved']}",
            f"- Sentinel-1: {metrics['total_source_sentinel_1_resolved']}",
            "",
            "## 8. DINO/unknowns bloqueados",
            f"- DINOv2 com source explicito: {metrics['dinov2_with_explicit_source']}",
            f"- DINOv2 sem source explicito: {metrics['dinov2_without_explicit_source']}",
            f"- Unknowns resolvidos: {metrics['unknowns_resolved']}",
            f"- Unknowns bloqueados: {metrics['unknowns_blocked']}",
            "",
            "## 9. Impacto potencial sobre requirements/requests",
            f"- Slots que poderiam ser desbloqueados: {metrics['slots_that_could_be_unblocked']}",
            f"- Patches que poderiam virar acionaveis: {metrics['patches_that_could_become_actionable']}",
            "",
            "## 10. Batches recuperaveis ou inexistentes",
            f"- 1 patch: {batches['recoverable_1_patch']['recoverable']}",
            f"- 20 patches: {batches['recoverable_20_patches']['recoverable']}",
            f"- 30 patches: {batches['recoverable_30_patches']['recoverable']}",
            "",
            "## 11. Proximos passos permitidos dentro da Vertente A",
            "- Recuperar manualmente source sensor family e acquisition date somente a partir de evidencia explicita.",
            "- Reconstruir resolucao de sensor apenas se os vinculos fonte ficarem unicos e auditaveis.",
            "",
            "## 12. Limitacoes",
            "- Esta auditoria nao consulta backend externo e nao baixa dados.",
            "- Esta auditoria nao cria datas e nao corrige requirements.",
            "",
            "## 13. Guardrails preservados",
            f"- Branch auditada: `{branch_name(repo_root)}`",
            *[f"- {item}" for item in GUARDRAILS],
            "",
        ]
    )


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    read_csv(root / SENSOR_RESOLUTION_INPUT)
    resolved_requirements = read_csv(root / RESOLVED_REQUIREMENTS_INPUT)
    request_manifest = read_csv(root / REQUEST_MANIFEST_INPUT)
    asset_rows = read_csv(root / ASSET_MANIFEST_INPUT)
    read_csv(root / PATCH_COVERAGE_INPUT)
    read_csv(root / REPAIR_CANDIDATES_INPUT)

    requirement_counts = build_count_by_patch(
        resolved_requirements,
        "is_requirement_actionable_after_sensor_resolution",
    )
    if not requirement_counts:
        requirement_counts = build_count_by_patch(resolved_requirements)
    request_counts = build_count_by_patch(request_manifest)
    evidence_rows = load_evidence_rows(root, asset_rows)
    evidence_index = build_evidence_index(evidence_rows)
    provenance_rows = build_provenance_rows(asset_rows, evidence_index, requirement_counts, request_counts)
    blocker_rows = build_blocker_rows(provenance_rows, requirement_counts, request_counts)
    metrics = build_metrics(provenance_rows, blocker_rows, requirement_counts)

    write_csv(root / PROVENANCE_OUTPUT, provenance_rows, PROVENANCE_COLUMNS)
    write_csv(root / BLOCKERS_OUTPUT, blocker_rows, BLOCKER_COLUMNS)
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
