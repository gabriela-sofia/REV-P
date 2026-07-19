from __future__ import annotations

import csv
import json
import re
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


CSV_COLUMNS = [
    "canonical_patch_id",
    "region",
    "asset_count_total",
    "clean_asset_count",
    "acquisition_dates_count",
    "acquisition_dates",
    "cloud_cover_min",
    "cloud_cover_max",
    "cloud_cover_mean",
    "temporal_bucket_coverage",
    "has_1_date",
    "has_2_dates",
    "has_3plus_dates",
    "eligible_for_temporal_drift",
    "step_a_status",
    "blocking_reason",
]

OUTPUT_TABLE = Path("outputs_public/tables/revp_temporal_asset_readiness_mv1.csv")
OUTPUT_REPORT = Path("outputs_public/execution_reports/revp_temporal_asset_readiness_mv1.md")
OUTPUT_METRICS = Path("outputs_public/metrics/revp_temporal_asset_readiness_mv1.json")

SCAN_ROOTS = [
    Path("manifests"),
    Path("outputs_public"),
    Path("datasets"),
    Path("configs"),
    Path("local_runs"),
]

READABLE_SUFFIXES = {".csv", ".json", ".jsonl", ".md"}
STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl"}
KEYWORDS = (
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
)
MAX_READ_BYTES = 5_000_000
MISSING_VALUES = {"", "unknown", "missing", "ausente", "desconhecido", "none", "null", "nan", "n/a", "na"}

STEP_A_READY = "STEP_A_READY_TEMPORAL_EMBEDDING_DRIFT"
STEP_A_PILOT_READY = "STEP_A_PILOT_READY_TEMPORAL_EMBEDDING_DRIFT"
STEP_A_NEEDS_BACKFILL = "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL"
STEP_A_INSUFFICIENT_METADATA = "STEP_A_BLOCKED_INSUFFICIENT_TEMPORAL_METADATA"

GUARDRAILS = [
    "unknown nunca vira negativo",
    "ausencia de evento nunca vira classe 0",
    "evento documentado nao vira label patch-level",
    "restauracao v2dz-v2ef nao vira ground truth operacional",
    "DINOv2 permanece frozen",
    "etapa atual e label-free e review-only",
    "nao criar feature table multimodal",
    "nao iniciar baseline supervisionado",
    "nao criar positivos ou negativos formais",
    "readiness temporal nao e evidencia de acuracia",
    "cloud cover ausente vira blocker, nao reprovacao inventada",
    "patch nao e excluido por falta de evento",
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
REGION_FIELDS = ["region", "regiao", "region_detected", "city", "cidade"]
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


@dataclass
class PatchAudit:
    canonical_patch_id: str
    regions: set[str] = field(default_factory=set)
    assets: set[str] = field(default_factory=set)
    clean_assets: set[str] = field(default_factory=set)
    acquisition_dates: set[str] = field(default_factory=set)
    cloud_cover_values: list[float] = field(default_factory=list)
    source_files: set[str] = field(default_factory=set)

    def add_observation(
        self,
        *,
        source_file: str,
        asset_id: str,
        region: str,
        dates: Iterable[str],
        cloud_values: Iterable[float],
    ) -> None:
        source = source_file.replace("\\", "/")
        asset = asset_id.strip() or f"{source}:row"
        self.source_files.add(source)
        self.assets.add(f"{source}:{asset}")
        if is_present(region):
            self.regions.add(normalize_region(region))
        valid_dates = sorted({date for date in dates if is_present(date)})
        if valid_dates:
            self.acquisition_dates.update(valid_dates)
            self.clean_assets.add(f"{source}:{asset}")
        self.cloud_cover_values.extend(cloud_values)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Audit Sentinel/DINO temporal asset readiness for REV-P MV1 Step A.")
    parser.add_argument("--repo-root", default=".", help="REV-P repository root.")
    return parser


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def is_present(value: Any) -> bool:
    return str(value).strip().lower() not in MISSING_VALUES


def normalize_region(value: str) -> str:
    text = str(value).strip()
    mapping = {
        "cur": "Curitiba",
        "curitiba": "Curitiba",
        "pet": "Petropolis",
        "petropolis": "Petropolis",
        "petropolis/rj": "Petropolis",
        "rec": "Recife",
        "recife": "Recife",
    }
    return mapping.get(text.lower(), text or "unknown")


def field_value(row: dict[str, Any], aliases: Iterable[str]) -> str:
    normalized = {normalize_key(key): key for key in row}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key is None:
            continue
        value = row.get(key)
        if is_present(value):
            return str(value).strip()
    return ""


def present_aliases(row: dict[str, Any], aliases: Iterable[str]) -> list[str]:
    normalized = {normalize_key(key): key for key in row}
    return sorted({normalized[normalize_key(alias)] for alias in aliases if normalize_key(alias) in normalized})


def missing_aliases(row: dict[str, Any], aliases: Iterable[str]) -> list[str]:
    if not row:
        return list(aliases)
    if present_aliases(row, aliases):
        return []
    return list(aliases)


def normalize_patch_id(value: str) -> str:
    text = str(value).strip()
    if not is_present(text):
        return ""
    return text


def normalize_date(value: str) -> str:
    text = str(value).strip()
    if not is_present(text):
        return ""
    match = re.search(r"\b(20\d{2})[-_/]?([01]\d)[-_/]?([0-3]\d)(?!\d)", text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def split_date_candidates(value: str) -> list[str]:
    if not is_present(value):
        return []
    parts = re.split(r"[|;,]", str(value))
    dates = [normalize_date(part) for part in parts]
    if not any(dates):
        dates = [normalize_date(str(value))]
    return sorted({date for date in dates if date})


def row_dates(row: dict[str, Any]) -> list[str]:
    dates: set[str] = set()
    for field in present_aliases(row, DATE_FIELDS):
        dates.update(split_date_candidates(str(row.get(field, ""))))
    return sorted(dates)


def parse_float(value: Any) -> float | None:
    if not is_present(value):
        return None
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def row_cloud_values(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for field in present_aliases(row, CLOUD_FIELDS):
        parsed = parse_float(row.get(field))
        if parsed is not None:
            values.append(parsed)
    return values


def candidate_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    excluded = {OUTPUT_TABLE, OUTPUT_REPORT, OUTPUT_METRICS}
    for root in SCAN_ROOTS:
        absolute_root = repo_root / root
        if not absolute_root.exists():
            continue
        for path in absolute_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
                continue
            rel = path.relative_to(repo_root)
            if rel in excluded:
                continue
            normalized_name = rel.as_posix().lower()
            if any(keyword in normalized_name for keyword in KEYWORDS):
                files.append(rel)
    return sorted(files, key=lambda item: item.as_posix())


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def flatten_json_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if any(isinstance(v, (dict, list)) for v in value.values()):
            scalar = {key: val for key, val in value.items() if not isinstance(val, (dict, list))}
            if scalar:
                records.append(scalar)
            for nested in value.values():
                records.extend(flatten_json_records(nested))
        else:
            records.append(value)
    elif isinstance(value, list):
        for item in value:
            records.extend(flatten_json_records(item))
    return records


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return flatten_json_records(data)


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.extend(flatten_json_records(json.loads(text)))
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_rows(path)
    if suffix == ".json":
        return read_json_rows(path)
    if suffix == ".jsonl":
        return read_jsonl_rows(path)
    return []


def update_field_inventory(
    *,
    row: dict[str, Any],
    rel_path: Path,
    fields_used: dict[str, set[str]],
    missing_fields: dict[str, set[str]],
) -> None:
    rel = rel_path.as_posix()
    for group in (PATCH_FIELDS, REGION_FIELDS, ASSET_FIELDS, DATE_FIELDS, CLOUD_FIELDS):
        fields_used.setdefault(rel, set()).update(present_aliases(row, group))
    for label, group in {
        "canonical_patch_id": PATCH_FIELDS,
        "region": REGION_FIELDS,
        "asset": ASSET_FIELDS,
        "acquisition_date": DATE_FIELDS,
        "cloud_cover": CLOUD_FIELDS,
    }.items():
        if missing_aliases(row, group):
            missing_fields.setdefault(rel, set()).add(label)


def collect_data(repo_root: Path) -> tuple[dict[str, PatchAudit], dict[str, Any]]:
    patches: dict[str, PatchAudit] = {}
    input_files_found: list[str] = []
    skipped_large_files: list[str] = []
    unreadable_files: dict[str, str] = {}
    md_files_read: list[str] = []
    fields_used: dict[str, set[str]] = {}
    missing_fields: dict[str, set[str]] = {}
    rows_with_patch_asset_link = 0
    rows_with_temporal_field = 0
    rows_with_clean_date = 0

    for rel_path in candidate_files(repo_root):
        path = repo_root / rel_path
        rel = rel_path.as_posix()
        if path.stat().st_size > MAX_READ_BYTES:
            skipped_large_files.append(rel)
            continue
        input_files_found.append(rel)
        if path.suffix.lower() == ".md":
            path.read_text(encoding="utf-8", errors="replace")
            md_files_read.append(rel)
            continue
        if path.suffix.lower() not in STRUCTURED_SUFFIXES:
            continue
        try:
            rows = read_rows(path)
        except (csv.Error, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            unreadable_files[rel] = f"{type(exc).__name__}: {exc}"
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            update_field_inventory(row=row, rel_path=rel_path, fields_used=fields_used, missing_fields=missing_fields)
            patch_id = normalize_patch_id(field_value(row, PATCH_FIELDS))
            asset_id = field_value(row, ASSET_FIELDS)
            if patch_id and asset_id:
                rows_with_patch_asset_link += 1
            if present_aliases(row, DATE_FIELDS):
                rows_with_temporal_field += 1
            dates = row_dates(row)
            if dates:
                rows_with_clean_date += 1
            if not patch_id:
                continue
            if patch_id not in patches:
                patches[patch_id] = PatchAudit(patch_id)
            patches[patch_id].add_observation(
                source_file=rel,
                asset_id=asset_id,
                region=field_value(row, REGION_FIELDS),
                dates=dates,
                cloud_values=row_cloud_values(row),
            )

    meta = {
        "input_files_found": input_files_found,
        "skipped_large_files": skipped_large_files,
        "unreadable_files": unreadable_files,
        "md_files_read": md_files_read,
        "fields_used": {key: sorted(values) for key, values in sorted(fields_used.items()) if values},
        "missing_fields": {key: sorted(values) for key, values in sorted(missing_fields.items()) if values},
        "rows_with_patch_asset_link": rows_with_patch_asset_link,
        "rows_with_temporal_field": rows_with_temporal_field,
        "rows_with_clean_date": rows_with_clean_date,
    }
    return patches, meta


def temporal_bucket(date_count: int) -> str:
    if date_count <= 0:
        return "0_dates"
    if date_count == 1:
        return "1_date"
    if date_count == 2:
        return "2_dates"
    return "3plus_dates"


def is_patch_eligible(patch: PatchAudit) -> bool:
    return len(patch.acquisition_dates) >= 3


def patch_blocking_reason(patch: PatchAudit) -> str:
    blockers: list[str] = []
    if not patch.acquisition_dates:
        blockers.append("MISSING_ACQUISITION_DATE")
    elif len(patch.acquisition_dates) < 3:
        blockers.append("FEWER_THAN_3_CLEAN_DATES")
    if not patch.cloud_cover_values:
        blockers.append("MISSING_NUMERIC_CLOUD_COVER")
    if not patch.regions:
        blockers.append("MISSING_REGION")
    return ";".join(blockers) if blockers else "NONE"


def decide_global_status(patches: dict[str, PatchAudit], meta: dict[str, Any]) -> str:
    eligible = sum(1 for patch in patches.values() if is_patch_eligible(patch))
    if eligible >= 30:
        return STEP_A_READY
    if 20 <= eligible <= 29:
        return STEP_A_PILOT_READY
    if patches and (meta["rows_with_patch_asset_link"] > 0 or meta["rows_with_temporal_field"] > 0):
        return STEP_A_NEEDS_BACKFILL
    return STEP_A_INSUFFICIENT_METADATA


def fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def fmt_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_rows(patches: dict[str, PatchAudit], global_status: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for patch_id in sorted(patches):
        patch = patches[patch_id]
        dates = sorted(patch.acquisition_dates)
        clouds = patch.cloud_cover_values
        date_count = len(dates)
        rows.append(
            {
                "canonical_patch_id": patch_id,
                "region": ";".join(sorted(patch.regions)) if patch.regions else "unknown",
                "asset_count_total": str(len(patch.assets)),
                "clean_asset_count": str(len(patch.clean_assets)),
                "acquisition_dates_count": str(date_count),
                "acquisition_dates": ";".join(dates) if dates else "unknown",
                "cloud_cover_min": fmt_number(min(clouds)) if clouds else "missing",
                "cloud_cover_max": fmt_number(max(clouds)) if clouds else "missing",
                "cloud_cover_mean": fmt_number(mean(clouds)) if clouds else "missing",
                "temporal_bucket_coverage": temporal_bucket(date_count),
                "has_1_date": fmt_bool(date_count == 1),
                "has_2_dates": fmt_bool(date_count == 2),
                "has_3plus_dates": fmt_bool(date_count >= 3),
                "eligible_for_temporal_drift": fmt_bool(is_patch_eligible(patch)),
                "step_a_status": global_status,
                "blocking_reason": patch_blocking_reason(patch),
            }
        )
    return rows


def build_metrics(rows: list[dict[str, str]], meta: dict[str, Any], global_status: str) -> dict[str, Any]:
    region_counts = Counter(row["region"] for row in rows)
    blockers = Counter(
        blocker
        for row in rows
        for blocker in row["blocking_reason"].split(";")
        if blocker and blocker != "NONE"
    )
    metrics = {
        "global_counts": {
            "patches_total": len(rows),
            "assets_total": sum(int(row["asset_count_total"]) for row in rows),
            "clean_assets_total": sum(int(row["clean_asset_count"]) for row in rows),
            "input_files_found_total": len(meta["input_files_found"]),
            "rows_with_patch_asset_link": meta["rows_with_patch_asset_link"],
            "rows_with_temporal_field": meta["rows_with_temporal_field"],
            "rows_with_clean_date": meta["rows_with_clean_date"],
        },
        "counts_by_region": dict(sorted(region_counts.items())),
        "patches_with_1_date": sum(1 for row in rows if row["has_1_date"] == "true"),
        "patches_with_2_dates": sum(1 for row in rows if row["has_2_dates"] == "true"),
        "patches_with_3plus_dates": sum(1 for row in rows if row["has_3plus_dates"] == "true"),
        "eligible_patch_count_for_temporal_drift": sum(
            1 for row in rows if row["eligible_for_temporal_drift"] == "true"
        ),
        "step_a_global_status": global_status,
        "blockers_summary": dict(blockers.most_common()),
        "input_files_found": meta["input_files_found"],
        "fields_missing": meta["missing_fields"],
        "fields_used": meta["fields_used"],
        "skipped_large_files": meta["skipped_large_files"],
        "unreadable_files": meta["unreadable_files"],
        "guardrails_preserved": GUARDRAILS,
    }
    return metrics


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
    blockers = metrics["blockers_summary"]
    eligible = [row["canonical_patch_id"] for row in rows if row["eligible_for_temporal_drift"] == "true"]
    report_lines = [
        "# revp_temporal_asset_readiness_audit_mv1",
        "",
        "## 1. Escopo da Vertente A",
        "Auditoria offline e review-only de manifestos/assets Sentinel/DINO existentes para medir prontidao tecnica para Temporal Embedding Drift label-free. A tarefa permanece na Vertente A e nao redireciona o projeto para Cross-City/AEC.",
        "",
        "## 2. Arquivos de entrada encontrados",
        *[f"- `{item}`" for item in metrics["input_files_found"]],
        "",
        "## 3. Campos usados",
        *[
            f"- `{source}`: {', '.join(fields)}"
            for source, fields in sorted(metrics["fields_used"].items())
        ],
        "",
        "## 4. Campos ausentes",
        *[
            f"- `{source}`: {', '.join(fields)}"
            for source, fields in sorted(metrics["fields_missing"].items())
        ],
        "",
        "## 5. Regra de limpeza temporal",
        "- `clean_asset_count` conta somente asset com `canonical_patch_id` valido e pelo menos uma data de aquisicao valida extraida de campo temporal explicito.",
        "- `unknown` e `missing` nunca contam como data limpa.",
        "- Datas, regiao, cloud cover, asset e vinculo patch/asset nao sao inferidos a partir de nomes ambiguos.",
        "",
        "## 6. Contagem de patches por numero de datas",
        f"- 1 data: {metrics['patches_with_1_date']}",
        f"- 2 datas: {metrics['patches_with_2_dates']}",
        f"- 3+ datas: {metrics['patches_with_3plus_dates']}",
        "",
        "## 7. Contagem por regiao",
        *[f"- {region}: {count}" for region, count in sorted(metrics["counts_by_region"].items())],
        "",
        "## 8. Patches elegiveis para drift temporal",
        f"- Total elegivel: {metrics['eligible_patch_count_for_temporal_drift']}",
        *[f"- `{patch_id}`" for patch_id in eligible[:50]],
        "",
        "## 9. Decisao operacional da Vertente A",
        f"- Branch auditada: `{branch_name(repo_root)}`",
        f"- `step_a_global_status`: `{metrics['step_a_global_status']}`",
        "",
        "## 10. Blockers para iniciar Temporal Embedding Drift",
        *[f"- {blocker}: {count}" for blocker, count in sorted(blockers.items())],
        "",
        "## 11. Proximos passos permitidos dentro da Vertente A",
        "- Backfill/curadoria temporal de datas de aquisicao por patch/asset.",
        "- Curadoria de cloud cover numerico quando existir em metadados rastreaveis.",
        "- Reexecucao desta auditoria offline apos completar metadados temporais.",
        "",
        "## 12. Limitacoes",
        "- A auditoria nao baixa assets, nao abre raster, nao le pixel e nao calcula embeddings.",
        "- Cloud cover ausente e blocker informativo, nao reprovacao inventada.",
        "- Prontidao temporal nao mede acuracia e nao confirma evento.",
        "",
        "## 13. Guardrails preservados",
        *[f"- {item}" for item in GUARDRAILS],
        "",
    ]
    return "\n".join(report_lines)


def execute(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    patches, meta = collect_data(root)
    global_status = decide_global_status(patches, meta)
    rows = build_rows(patches, global_status)
    metrics = build_metrics(rows, meta, global_status)

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
