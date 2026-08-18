from __future__ import annotations

import csv
import json
import re
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


READINESS_TABLE = Path("outputs_public/tables/revp_temporal_asset_readiness_mv1.csv")
READINESS_METRICS = Path("outputs_public/metrics/revp_temporal_asset_readiness_mv1.json")
READINESS_REPORT = Path("outputs_public/execution_reports/revp_temporal_asset_readiness_mv1.md")

OUTPUT_TABLE = Path("outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv")
OUTPUT_METRICS = Path("outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json")
OUTPUT_REPORT = Path("outputs_public/execution_reports/revp_temporal_asset_backfill_queue_mv1.md")

SCAN_ROOTS = [
    Path("manifests"),
    Path("outputs_public"),
    Path("datasets"),
    Path("configs"),
    Path("local_runs"),
]

CSV_COLUMNS = [
    "canonical_patch_id",
    "region",
    "current_clean_dates_count",
    "current_acquisition_dates",
    "needed_dates_to_reach_3",
    "asset_count_total",
    "clean_asset_count",
    "sentinel_2_asset_count",
    "sentinel_1_asset_count",
    "dinov2_asset_count",
    "unknown_asset_count",
    "requires_s2_cloud_cover_review",
    "missing_acquisition_date",
    "missing_region",
    "missing_patch_asset_link",
    "missing_numeric_cloud_cover_s2_only",
    "backfill_priority",
    "backfill_action",
    "blocking_reason",
    "eligible_after_backfill_if_completed",
]

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

STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl"}
READABLE_SUFFIXES = STRUCTURED_SUFFIXES | {".md"}
MAX_READ_BYTES = 5_000_000
MISSING_VALUES = {"", "unknown", "missing", "ausente", "desconhecido", "none", "null", "nan", "n/a", "na"}
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

STEP_A_BACKFILL_READY = "STEP_A_BACKFILL_QUEUE_READY"
STEP_A_METADATA_REPAIR_REQUIRED = "STEP_A_METADATA_REPAIR_REQUIRED"
STEP_A_ASSET_LINKAGE_REPAIR_REQUIRED = "STEP_A_ASSET_LINKAGE_REPAIR_REQUIRED"
STEP_A_MANUAL_TEMPORAL_CURATION_REQUIRED = "STEP_A_MANUAL_TEMPORAL_CURATION_REQUIRED"

GUARDRAILS = [
    "sem rotulo final",
    "sem positivo formal",
    "sem negativos formais",
    "sem classe-zero",
    "sem GT operacional",
    "sem treino supervisionado",
    "DINOv2 permanece frozen",
    "fila e label-free e review-only",
    "sem download automatico de assets",
    "sem calculo de drift temporal nesta etapa",
]


@dataclass
class AssetStats:
    sentinel_2_assets: set[str] = field(default_factory=set)
    sentinel_1_assets: set[str] = field(default_factory=set)
    dinov2_assets: set[str] = field(default_factory=set)
    patch_manifest_assets: set[str] = field(default_factory=set)
    external_evidence_assets: set[str] = field(default_factory=set)
    unknown_assets: set[str] = field(default_factory=set)
    rows_with_patch_asset_link: int = 0
    rows_without_patch_asset_link: int = 0
    s2_rows_without_numeric_cloud: int = 0
    rows_without_acquisition_date: int = 0

    def add(self, *, asset_key: str, asset_type: str, has_link: bool, has_date: bool, has_numeric_cloud: bool) -> None:
        if has_link:
            self.rows_with_patch_asset_link += 1
        else:
            self.rows_without_patch_asset_link += 1
        if not has_date:
            self.rows_without_acquisition_date += 1
        if asset_type == "sentinel_2_optical":
            self.sentinel_2_assets.add(asset_key)
            if not has_numeric_cloud:
                self.s2_rows_without_numeric_cloud += 1
        elif asset_type == "sentinel_1_sar":
            self.sentinel_1_assets.add(asset_key)
        elif asset_type == "dinov2_embedding":
            self.dinov2_assets.add(asset_key)
        elif asset_type == "patch_manifest":
            self.patch_manifest_assets.add(asset_key)
        elif asset_type == "external_evidence":
            self.external_evidence_assets.add(asset_key)
        else:
            self.unknown_assets.add(asset_key)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Build REV-P MV1 temporal asset backfill queue.")
    parser.add_argument("--repo-root", default=".", help="REV-P repository root.")
    return parser


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def is_present(value: Any) -> bool:
    return str(value).strip().lower() not in MISSING_VALUES


def field_value(row: dict[str, Any], aliases: Iterable[str]) -> str:
    normalized = {normalize_key(key): key for key in row}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key is not None and is_present(row.get(key)):
            return str(row.get(key)).strip()
    return ""


def present_fields(row: dict[str, Any], aliases: Iterable[str]) -> list[str]:
    normalized = {normalize_key(key): key for key in row}
    return sorted({normalized[normalize_key(alias)] for alias in aliases if normalize_key(alias) in normalized})


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "sim", "yes", "1"}


def parse_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        return 0


def parse_float(value: Any) -> float | None:
    if not is_present(value):
        return None
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def normalize_date(value: Any) -> str:
    text = str(value).strip()
    if not is_present(text):
        return ""
    match = re.search(r"\b(20\d{2})[-_/]?([01]\d)[-_/]?([0-3]\d)(?!\d)", text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def row_has_date(row: dict[str, Any]) -> bool:
    for field in present_fields(row, DATE_FIELDS):
        values = re.split(r"[|;,]", str(row.get(field, "")))
        if any(normalize_date(value) for value in values):
            return True
    return False


def row_has_numeric_cloud(row: dict[str, Any]) -> bool:
    return any(parse_float(row.get(field)) is not None for field in present_fields(row, CLOUD_FIELDS))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def candidate_files(repo_root: Path) -> list[Path]:
    excluded = {OUTPUT_TABLE, OUTPUT_METRICS, OUTPUT_REPORT}
    files: list[Path] = []
    for root in SCAN_ROOTS:
        absolute = repo_root / root
        if not absolute.exists():
            continue
        for path in absolute.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
                continue
            rel = path.relative_to(repo_root)
            if rel in excluded:
                continue
            text = rel.as_posix().lower()
            if any(marker in text for marker in OUT_OF_SCOPE_PATH_MARKERS):
                continue
            if any(keyword in text for keyword in SCAN_KEYWORDS):
                files.append(rel)
    return sorted(files, key=lambda item: item.as_posix())


def row_text(row: dict[str, Any], source: str) -> str:
    joined = " ".join([source, *[str(key) for key in row.keys()], *[str(value) for value in row.values()]])
    return joined.lower()


def classify_asset(row: dict[str, Any], source: str) -> str:
    text = row_text(row, source)
    if "dinov2" in text or "dino" in text or "embedding" in text or "vector_sha256" in text:
        return "dinov2_embedding"
    if "sentinel-1" in text or "sentinel1" in text or re.search(r"\bs1\b", text) or " sar" in text:
        return "sentinel_1_sar"
    if (
        "sentinel-2" in text
        or "sentinel2" in text
        or re.search(r"\bs2\b", text)
        or "optical" in text
        or ("sentinel" in text and ("cloud_cover" in text or "tile_id" in text or "mgrs" in text))
    ):
        return "sentinel_2_optical"
    if "external_evidence" in text or "evidencia_externa" in text or "official" in text or "gazette" in text:
        return "external_evidence"
    if "patch" in text and ("manifest" in text or not field_value(row, ASSET_FIELDS)):
        return "patch_manifest"
    return "unknown_asset_type"


def collect_asset_stats(repo_root: Path) -> tuple[dict[str, AssetStats], dict[str, Any]]:
    stats: dict[str, AssetStats] = {}
    input_files: list[str] = []
    skipped_large_files: list[str] = []
    unreadable_files: dict[str, str] = {}

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
        for row_index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                continue
            patch_id = field_value(row, PATCH_FIELDS)
            if not patch_id:
                continue
            asset_id = field_value(row, ASSET_FIELDS)
            has_link = bool(asset_id)
            asset_key = f"{source}:{asset_id or 'row_' + str(row_index)}"
            stats.setdefault(patch_id, AssetStats()).add(
                asset_key=asset_key,
                asset_type=classify_asset(row, source),
                has_link=has_link,
                has_date=row_has_date(row),
                has_numeric_cloud=row_has_numeric_cloud(row),
            )

    meta = {
        "input_files_used": input_files,
        "skipped_large_files": skipped_large_files,
        "unreadable_files": unreadable_files,
    }
    return stats, meta


def load_readiness_rows(repo_root: Path) -> list[dict[str, Any]]:
    return read_csv(repo_root / READINESS_TABLE)


def needed_dates(current_count: int) -> int:
    return max(0, 3 - current_count)


def priority_for(row: dict[str, Any], stats: AssetStats, unknown_count: int) -> str:
    current_count = parse_int(row.get("acquisition_dates_count"))
    asset_count = parse_int(row.get("asset_count_total"))
    if current_count == 2:
        return "P0_NEAR_READY"
    if current_count == 1:
        return "P1_PARTIAL_TEMPORAL"
    if asset_count > 0 and current_count == 0 and unknown_count < asset_count:
        return "P2_HAS_ASSETS_NO_CLEAN_DATES"
    if asset_count > 0 or unknown_count > 0 or stats.rows_without_patch_asset_link > 0:
        return "P3_METADATA_ONLY_OR_UNKNOWN"
    return "P4_NOT_ACTIONABLE_WITH_CURRENT_METADATA"


def action_for(
    *,
    current_count: int,
    missing_region: bool,
    missing_patch_asset_link: bool,
    missing_s2_cloud: bool,
    unknown_count: int,
    asset_count: int,
) -> str:
    if missing_patch_asset_link:
        return "REPAIR_PATCH_ASSET_LINKAGE"
    if unknown_count > 0 and unknown_count >= asset_count:
        return "CLASSIFY_ASSET_TYPE_BEFORE_TEMPORAL_USE"
    if missing_region:
        return "RECOVER_REGION_FROM_CANONICAL_PATCH_REGISTRY"
    if missing_s2_cloud:
        return "RECOVER_S2_CLOUD_COVER_METADATA"
    if current_count in {1, 2}:
        return "ADD_TEMPORAL_ACQUISITION_CANDIDATE"
    if current_count == 0 and asset_count > 0:
        return "RECOVER_ACQUISITION_DATE_FROM_MANIFEST_OR_FILENAME_ONLY_IF_UNAMBIGUOUS"
    return "MANUAL_REVIEW_REQUIRED"


def build_queue_rows(readiness_rows: list[dict[str, Any]], asset_stats: dict[str, AssetStats]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in sorted(readiness_rows, key=lambda row: str(row.get("canonical_patch_id", ""))):
        patch_id = str(item.get("canonical_patch_id", "")).strip()
        stats = asset_stats.get(patch_id, AssetStats())
        current_count = parse_int(item.get("acquisition_dates_count"))
        asset_count = parse_int(item.get("asset_count_total"))
        clean_asset_count = parse_int(item.get("clean_asset_count"))
        known_count = len(stats.sentinel_2_assets) + len(stats.sentinel_1_assets) + len(stats.dinov2_assets)
        scanned_unknown_count = len(stats.unknown_assets)
        unknown_count = max(scanned_unknown_count, max(0, asset_count - known_count - len(stats.external_evidence_assets)))
        missing_region = str(item.get("region", "")).strip().lower() == "unknown"
        missing_patch_asset_link = asset_count == 0 and stats.rows_with_patch_asset_link == 0
        missing_acquisition_date = current_count < 3
        missing_s2_cloud = len(stats.sentinel_2_assets) > 0 and stats.s2_rows_without_numeric_cloud > 0
        priority = priority_for(item, stats, unknown_count)
        action = action_for(
            current_count=current_count,
            missing_region=missing_region,
            missing_patch_asset_link=missing_patch_asset_link,
            missing_s2_cloud=missing_s2_cloud,
            unknown_count=unknown_count,
            asset_count=asset_count,
        )
        blockers = []
        if missing_acquisition_date:
            blockers.append("MISSING_ACQUISITION_DATE")
        if missing_region:
            blockers.append("MISSING_REGION")
        if missing_patch_asset_link:
            blockers.append("MISSING_PATCH_ASSET_LINK")
        if missing_s2_cloud:
            blockers.append("MISSING_NUMERIC_CLOUD_COVER_S2_ONLY")
        if unknown_count > 0:
            blockers.append("UNKNOWN_ASSET_TYPE")

        rows.append(
            {
                "canonical_patch_id": patch_id,
                "region": str(item.get("region", "unknown")).strip() or "unknown",
                "current_clean_dates_count": str(current_count),
                "current_acquisition_dates": str(item.get("acquisition_dates", "unknown")).strip() or "unknown",
                "needed_dates_to_reach_3": str(needed_dates(current_count)),
                "asset_count_total": str(asset_count),
                "clean_asset_count": str(clean_asset_count),
                "sentinel_2_asset_count": str(len(stats.sentinel_2_assets)),
                "sentinel_1_asset_count": str(len(stats.sentinel_1_assets)),
                "dinov2_asset_count": str(len(stats.dinov2_assets)),
                "unknown_asset_count": str(unknown_count),
                "requires_s2_cloud_cover_review": str(missing_s2_cloud).lower(),
                "missing_acquisition_date": str(missing_acquisition_date).lower(),
                "missing_region": str(missing_region).lower(),
                "missing_patch_asset_link": str(missing_patch_asset_link).lower(),
                "missing_numeric_cloud_cover_s2_only": str(missing_s2_cloud).lower(),
                "backfill_priority": priority,
                "backfill_action": action,
                "blocking_reason": ";".join(blockers) if blockers else "NONE",
                "eligible_after_backfill_if_completed": str(priority in {"P0_NEAR_READY", "P1_PARTIAL_TEMPORAL", "P2_HAS_ASSETS_NO_CLEAN_DATES"}).lower(),
            }
        )
    return rows


def decide_next_status(rows: list[dict[str, str]]) -> str:
    total = len(rows)
    p0_p1 = sum(1 for row in rows if row["backfill_priority"] in {"P0_NEAR_READY", "P1_PARTIAL_TEMPORAL"})
    actionable_metadata = sum(
        1
        for row in rows
        if row["backfill_action"]
        in {
            "RECOVER_ACQUISITION_DATE_FROM_MANIFEST_OR_FILENAME_ONLY_IF_UNAMBIGUOUS",
            "RECOVER_REGION_FROM_CANONICAL_PATCH_REGISTRY",
            "RECOVER_S2_CLOUD_COVER_METADATA",
            "ADD_TEMPORAL_ACQUISITION_CANDIDATE",
        }
    )
    unknown_or_linkage = sum(
        1
        for row in rows
        if row["unknown_asset_count"] != "0" or row["missing_patch_asset_link"] == "true"
    )
    if p0_p1 >= 20:
        return STEP_A_BACKFILL_READY
    if actionable_metadata > 0:
        return STEP_A_METADATA_REPAIR_REQUIRED
    if total and unknown_or_linkage / total >= 0.5:
        return STEP_A_ASSET_LINKAGE_REPAIR_REQUIRED
    return STEP_A_MANUAL_TEMPORAL_CURATION_REQUIRED


def build_metrics(rows: list[dict[str, str]], meta: dict[str, Any], step_a_next_status: str) -> dict[str, Any]:
    priority_counts = Counter(row["backfill_priority"] for row in rows)
    region_counts = Counter(row["region"] for row in rows)
    blockers = Counter(
        blocker
        for row in rows
        for blocker in row["blocking_reason"].split(";")
        if blocker and blocker != "NONE"
    )
    asset_type_counts = {
        "sentinel_2_optical": sum(int(row["sentinel_2_asset_count"]) for row in rows),
        "sentinel_1_sar": sum(int(row["sentinel_1_asset_count"]) for row in rows),
        "dinov2_embedding": sum(int(row["dinov2_asset_count"]) for row in rows),
        "unknown_asset_type": sum(int(row["unknown_asset_count"]) for row in rows),
    }
    p0_p1 = priority_counts["P0_NEAR_READY"] + priority_counts["P1_PARTIAL_TEMPORAL"]
    return {
        "total_patches_in_queue": len(rows),
        "counts_by_priority": dict(sorted(priority_counts.items())),
        "counts_by_region": dict(sorted(region_counts.items())),
        "counts_by_asset_type": asset_type_counts,
        "p0_p1_patch_count": p0_p1,
        "potentially_eligible_after_backfill_count": sum(
            1 for row in rows if row["eligible_after_backfill_if_completed"] == "true"
        ),
        "global_blockers": dict(blockers.most_common()),
        "step_a_next_status": step_a_next_status,
        "guardrails_preserved": GUARDRAILS,
        "input_files_used": [
            READINESS_TABLE.as_posix(),
            READINESS_METRICS.as_posix(),
            READINESS_REPORT.as_posix(),
            *meta["input_files_used"],
        ],
        "skipped_large_files": meta["skipped_large_files"],
        "unreadable_files": meta["unreadable_files"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def branch_name(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("ref: refs/heads/"):
        return text.removeprefix("ref: refs/heads/")
    return text or "unknown"


def build_report(repo_root: Path, rows: list[dict[str, str]], metrics: dict[str, Any]) -> str:
    near_ready = [row for row in rows if row["backfill_priority"] in {"P0_NEAR_READY", "P1_PARTIAL_TEMPORAL"}]
    lines = [
        "# revp_temporal_asset_backfill_queue_mv1",
        "",
        "## 1. Escopo da etapa",
        "Fila offline e auditavel de backfill/curadoria temporal para preparar a Vertente A para um futuro Temporal Embedding Drift label-free.",
        "",
        "## 2. Relacao com a auditoria anterior",
        f"- Entrada principal: `{READINESS_TABLE.as_posix()}`",
        f"- Status anterior: `{metrics.get('step_a_next_status')}` como etapa de reparo/curadoria, derivada da ausencia de patches elegiveis para drift temporal.",
        "",
        "## 3. Por que o drift temporal ainda nao deve ser executado",
        "- A fila nao possui volume suficiente de patches com tres datas limpas.",
        "- Esta etapa nao baixa assets, nao calcula embeddings e nao mede mudanca temporal.",
        "",
        "## 4. Arquivos de entrada usados",
        *[f"- `{item}`" for item in metrics["input_files_used"]],
        "",
        "## 5. Classificacao de tipos de asset",
        *[f"- {asset_type}: {count}" for asset_type, count in sorted(metrics["counts_by_asset_type"].items())],
        "",
        "## 6. Regra corrigida para cloud cover S2-only",
        "- Cloud cover numerico e obrigatorio apenas para candidatos classificados como `sentinel_2_optical`.",
        "- Ausencia de cloud cover em Sentinel-1, DINOv2, manifesto auxiliar ou asset unknown nao vira blocker universal.",
        "",
        "## 7. Contagem por prioridade de backfill",
        *[f"- {priority}: {count}" for priority, count in sorted(metrics["counts_by_priority"].items())],
        "",
        "## 8. Contagem por regiao",
        *[f"- {region}: {count}" for region, count in sorted(metrics["counts_by_region"].items())],
        "",
        "## 9. Acoes recomendadas",
        "- Recuperar datas de aquisicao somente de manifesto ou filename quando nao houver ambiguidade.",
        "- Recuperar regiao por registro canonico de patch.",
        "- Recuperar cloud cover numerico apenas para Sentinel-2 optico.",
        "- Classificar asset unknown antes de qualquer uso temporal.",
        "- Reparar vinculo patch/asset quando ausente.",
        "",
        "## 10. Patches mais proximos de elegibilidade",
        *[
            f"- `{row['canonical_patch_id']}`: {row['backfill_priority']}, precisa de {row['needed_dates_to_reach_3']} datas"
            for row in near_ready[:50]
        ],
        "",
        "## 11. Blockers restantes",
        *[f"- {blocker}: {count}" for blocker, count in sorted(metrics["global_blockers"].items())],
        "",
        "## 12. Proximos passos permitidos dentro da Vertente A",
        "- Curadoria temporal manual/offline.",
        "- Reparos de metadados e vinculos patch/asset.",
        "- Reexecucao da auditoria de readiness apos backfill.",
        "",
        "## 13. Guardrails preservados",
        f"- Branch auditada: `{branch_name(repo_root)}`",
        *[f"- {item}" for item in GUARDRAILS],
        "",
    ]
    return "\n".join(lines)


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    readiness_rows = load_readiness_rows(root)
    asset_stats, meta = collect_asset_stats(root)
    rows = build_queue_rows(readiness_rows, asset_stats)
    step_a_next_status = decide_next_status(rows)
    metrics = build_metrics(rows, meta, step_a_next_status)

    write_csv(root / OUTPUT_TABLE, rows)
    write_json(root / OUTPUT_METRICS, metrics)
    (root / OUTPUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUT_REPORT).write_text(build_report(root, rows, metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    execute(Path(args.repo_root))


if __name__ == "__main__":
    main()
