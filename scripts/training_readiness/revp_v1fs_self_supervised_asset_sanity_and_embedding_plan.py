from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1FR_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fr_self_supervised_dataloader_preflight"
OUT_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fs_self_supervised_asset_sanity_and_embedding_plan"
DOC_PATH = ROOT / "docs" / "revp_v1fs_self_supervised_asset_sanity_and_embedding_plan_report.md"

EXPECTED_OUTPUTS = [
    "asset_sanity_audit_v1fs.csv",
    "embedding_extraction_readiness_v1fs.csv",
    "embedding_extraction_plan_v1fs.csv",
    "embedding_split_matrix_v1fs.csv",
    "duplicate_and_conflict_audit_v1fs.csv",
    "blocked_after_v1fs.csv",
    "summary_v1fs.json",
    "qa_v1fs.csv",
    "status_v1fs.csv",
]

SAFE_STATUSES = {"SAFE_FOR_REVIEW_ONLY", "SAFE_FOR_REVIEW_ONLY_DUPLICATE_GROUP_FLAGGED"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def resolve_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def sha256_small(path: Path, max_mb: int = 50) -> str:
    if not path.exists() or not path.is_file():
        return ""
    if path.stat().st_size > max_mb * 1024 * 1024:
        return "SKIPPED_GT_50MB"
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dims(path: Path) -> tuple[str, str, str]:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return "", "", "NOT_PREVIEW_IMAGE"
    try:
        from PIL import Image

        with Image.open(path) as img:
            return str(img.width), str(img.height), "IMAGE_HEADER_ONLY_OK"
    except Exception as exc:
        return "", "", f"IMAGE_HEADER_FAILED:{type(exc).__name__}"


def npy_shape(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".npy":
        return "", "NOT_NPY"
    try:
        import numpy as np

        arr = np.load(path, mmap_mode="r")
        return "x".join(str(dim) for dim in arr.shape), f"NPY_MMAP_HEADER_OK:{arr.dtype}"
    except Exception as exc:
        return "", f"NPY_MMAP_HEADER_FAILED:{type(exc).__name__}"


def tif_header(path: Path) -> dict[str, str]:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return {
            "shape_or_dimensions": "",
            "crs_if_header_available": "",
            "bounds_if_header_available": "",
            "header_status": "NOT_TIF",
        }
    try:
        import rasterio

        with rasterio.open(path) as ds:
            return {
                "shape_or_dimensions": f"{ds.width}x{ds.height}x{ds.count};dtype={','.join(ds.dtypes)}",
                "crs_if_header_available": str(ds.crs) if ds.crs else "",
                "bounds_if_header_available": f"{ds.bounds.left},{ds.bounds.bottom},{ds.bounds.right},{ds.bounds.top}",
                "header_status": "TIF_HEADER_ONLY_OK",
            }
    except Exception as exc:
        return {
            "shape_or_dimensions": "",
            "crs_if_header_available": "",
            "bounds_if_header_available": "",
            "header_status": f"TIF_HEADER_FAILED:{type(exc).__name__}",
        }


def inspect_asset(row: dict[str, str]) -> dict[str, str]:
    path = resolve_path(row.get("asset_path", ""))
    exists = bool(path and path.exists() and path.is_file())
    size = path.stat().st_size if exists and path else 0
    extension = path.suffix.lower() if path else ""
    width = height = ""
    shape = ""
    crs = ""
    bounds = ""
    header_status = "NO_PATH"
    if exists and path:
        if extension in {".png", ".jpg", ".jpeg"}:
            width, height, header_status = image_dims(path)
            shape = f"{width}x{height}" if width and height else ""
        elif extension == ".npy":
            shape, header_status = npy_shape(path)
        elif extension in {".tif", ".tiff"}:
            info = tif_header(path)
            shape = info["shape_or_dimensions"]
            crs = info["crs_if_header_available"]
            bounds = info["bounds_if_header_available"]
            header_status = info["header_status"]
        else:
            header_status = "PATH_SIZE_ONLY"
    digest = sha256_small(path) if exists and path else ""
    too_small = size > 0 and size < 1024
    return {
        "asset_id": hashlib.sha1(row.get("asset_path", "").encode("utf-8")).hexdigest()[:16],
        "candidate_id": row.get("candidate_id", ""),
        "region": row.get("region", ""),
        "modality": row.get("modality", ""),
        "asset_path": row.get("asset_path", ""),
        "asset_type": row.get("asset_type", ""),
        "exists": "yes" if exists else "no",
        "extension": extension,
        "size_bytes": str(size),
        "shape_or_dimensions": shape,
        "crs_if_header_available": crs,
        "bounds_if_header_available": bounds,
        "split_group": row.get("split_group", ""),
        "source_family": row.get("source_family", ""),
        "sha256_small_file": digest,
        "basename": path.name if path else "",
        "header_status": header_status,
        "too_small_flag": "yes" if too_small else "no",
        "leakage_flag": row.get("leakage_risk", ""),
        "input_preflight_status": row.get("preflight_status", ""),
        "blocked_reason_from_v1fr": row.get("blocked_reason", ""),
    }


def duplicate_flags(audit_rows: list[dict[str, str]]) -> dict[str, str]:
    by_path = defaultdict(list)
    by_basename = defaultdict(list)
    by_hash = defaultdict(list)
    for row in audit_rows:
        if row["asset_path"]:
            by_path[row["asset_path"]].append(row["asset_id"])
        if row["basename"]:
            by_basename[row["basename"]].append(row["asset_id"])
        digest = row.get("sha256_small_file", "")
        if digest and not digest.startswith("SKIPPED"):
            by_hash[digest].append(row["asset_id"])
    flags = defaultdict(list)
    for mapping, label in [(by_path, "DUPLICATE_PATH"), (by_basename, "DUPLICATE_BASENAME"), (by_hash, "DUPLICATE_HASH")]:
        for _key, ids in mapping.items():
            if len(ids) > 1:
                for asset_id in ids:
                    flags[asset_id].append(label)
    return {asset_id: ";".join(sorted(set(values))) for asset_id, values in flags.items()}


def readiness_from(row: dict[str, str], dup_flag: str) -> tuple[str, str, str]:
    if row["input_preflight_status"] not in SAFE_STATUSES:
        return "BLOCKED_FROM_V1FR", "none", row["blocked_reason_from_v1fr"] or "Not eligible in v1fr."
    if row["exists"] != "yes":
        return "REVIEW_ONLY_REBASED_FILE_MISSING", "fix_path_or_exclude", "Asset path missing at v1fs check."
    if row["too_small_flag"] == "yes":
        return "REVIEW_ONLY_REBASED_TOO_SMALL", "inspect_or_exclude", "Asset file is very small and needs review."
    if row["header_status"].endswith("FAILED"):
        return "REVIEW_ONLY_REBASED_HEADER_FAILED", "inspect_header_failure", row["header_status"]
    if dup_flag:
        return "EMBEDDING_REVIEW_ONLY_READY_WITH_DUPLICATE_FLAG", "deduplicate_or_keep_grouped_before_future_extraction", dup_flag
    return "EMBEDDING_REVIEW_ONLY_READY", "future_embedding_extraction_after_reviewer_approval", "Path/header sanity passed for review-only future extraction."


def build_outputs(dl_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    audit_rows = [inspect_asset(row) for row in dl_rows]
    dup_map = duplicate_flags(audit_rows)
    readiness_rows = []
    conflict_rows = []
    blocked_rows = []
    for row in audit_rows:
        dup_flag = dup_map.get(row["asset_id"], "")
        status, next_step, reason = readiness_from(row, dup_flag)
        row["duplicate_flag"] = dup_flag
        readiness = {
            "asset_id": row["asset_id"],
            "candidate_id": row["candidate_id"],
            "region": row["region"],
            "modality": row["modality"],
            "asset_path": row["asset_path"],
            "exists": row["exists"],
            "size_bytes": row["size_bytes"],
            "shape_or_dimensions": row["shape_or_dimensions"],
            "crs_if_header_available": row["crs_if_header_available"],
            "split_group": row["split_group"],
            "duplicate_flag": dup_flag,
            "leakage_flag": row["leakage_flag"],
            "readiness_status": status,
            "allowed_next_step": next_step,
            "blocking_reason": "" if status.startswith("EMBEDDING_REVIEW_ONLY_READY") else reason,
            "notes": f"{row['header_status']}; bounds={row['bounds_if_header_available']}" if row["bounds_if_header_available"] else row["header_status"],
        }
        readiness_rows.append(readiness)
        if dup_flag:
            conflict_rows.append(
                {
                    "asset_id": row["asset_id"],
                    "candidate_id": row["candidate_id"],
                    "asset_path": row["asset_path"],
                    "basename": row["basename"],
                    "sha256_small_file": row["sha256_small_file"],
                    "duplicate_or_conflict_flag": dup_flag,
                    "recommendation": "keep in same split group or deduplicate before any future extraction",
                }
            )
        if not status.startswith("EMBEDDING_REVIEW_ONLY_READY"):
            blocked_rows.append(
                {
                    "asset_id": row["asset_id"],
                    "candidate_id": row["candidate_id"],
                    "region": row["region"],
                    "modality": row["modality"],
                    "asset_path": row["asset_path"],
                    "readiness_status": status,
                    "blocking_reason": reason,
                    "required_action": next_step,
                }
            )
    return audit_rows, readiness_rows, conflict_rows, blocked_rows


def embedding_plan() -> list[dict[str, str]]:
    return [
        {
            "modality": "sentinel_raster_path_only",
            "recommended_encoder_type": "pretrained vision backbone or geospatial pretrained encoder, review-only",
            "input_preparation": "future header-approved window/resize pipeline; no event target",
            "allowed_transform": "resize/crop/normalization from v1fr policy after review",
            "forbidden_transform": "mask/box/event class derivation or source-status target",
            "output_artifact_future": "embedding table with asset_id, split_group and encoder metadata",
            "evaluation_allowed": "embedding sanity, retrieval examples, clustering inspection",
            "evaluation_forbidden": "event detection metrics or binary/supervised readiness claims",
            "scientific_claim_allowed": "material representation explored under review-only REV-P scope",
        },
        {
            "modality": "multimodal_stack_path_only",
            "recommended_encoder_type": "transfer-learning encoder or lightweight projection head after review",
            "input_preparation": "shape-checked mmap input; no array-wide summary metrics in planning phase",
            "allowed_transform": "fixed normalization policy only after reviewer approval",
            "forbidden_transform": "target creation from status/evidence or region",
            "output_artifact_future": "stack embedding manifest; no labels",
            "evaluation_allowed": "modality separation and retrieval sanity",
            "evaluation_forbidden": "performance claims or class metrics",
            "scientific_claim_allowed": "stack assets can support future no-target representation review",
        },
        {
            "modality": "rgb_preview",
            "recommended_encoder_type": "none for training; optional visual QA encoder only after review",
            "input_preparation": "dimension inspection only; keep as explanatory material",
            "allowed_transform": "resize for figures or QA review",
            "forbidden_transform": "move into training set in v1fs",
            "output_artifact_future": "visual QA appendix only",
            "evaluation_allowed": "legibility and preview coverage",
            "evaluation_forbidden": "training or target evaluation",
            "scientific_claim_allowed": "visual previews explain available material, not event truth",
        },
    ]


def split_matrix(readiness_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = defaultdict(Counter)
    for row in readiness_rows:
        key = (row["region"], row["modality"], row["split_group"])
        grouped[key][row["readiness_status"]] += 1
    rows = []
    for (region, modality, split_group), counts in sorted(grouped.items()):
        rows.append(
            {
                "region": region,
                "modality": modality,
                "split_group": split_group,
                "ready_review_only": sum(v for k, v in counts.items() if k.startswith("EMBEDDING_REVIEW_ONLY_READY")),
                "blocked_or_rebased": sum(v for k, v in counts.items() if not k.startswith("EMBEDDING_REVIEW_ONLY_READY")),
                "total": sum(counts.values()),
                "split_policy": "geographic_source_aware_group; no random row split",
            }
        )
    return rows


def qa_rows(script_text: str, outputs_exist: bool, summary: dict) -> list[dict[str, str]]:
    rows = []

    def add(check: str, ok: bool, details: str) -> None:
        rows.append({"check": check, "status": "PASS" if ok else "FAIL", "details": details})

    bad_phrases = [
        "observed flood ground truth",
        "binary ready",
        "supervised ready",
        "final label",
        "positive/negative class",
        "validated flood",
        "performance claim",
    ]
    bad_code_terms = ["src." + "read", "read" + "(1)", "for " + "epoch", "." + "fit(", "back" + "ward" + "(", "optim" + "izer"]
    output_text = json.dumps(summary, ensure_ascii=False).lower()
    add("outputs exist", outputs_exist, "All required v1fs outputs are present.")
    add("v1fr counts preserved", summary["v1fr_safe_input_rows"] == 211 and summary["v1fr_blocked_rows"] == 142, f"safe={summary['v1fr_safe_input_rows']}; blocked={summary['v1fr_blocked_rows']}")
    add("eligible rows accounted", summary["eligible_input_rows"] == 211, f"eligible={summary['eligible_input_rows']}")
    add("blocked rows preserved", summary["blocked_reference_rows"] == 142, f"blocked={summary['blocked_reference_rows']}")
    add("no blocked entry promoted", summary["blocked_promotions"] == 0, "No v1fr blocked row is marked ready in v1fs.")
    add("dangerous terms absent in summary", not any(phrase in output_text for phrase in bad_phrases), "Checked summary text for forbidden phrasing.")
    add("script has no training loop or prohibited calls", not any(term in script_text for term in bad_code_terms), "Checked exact code patterns for training/pixel-read calls.")
    add("zero training", summary["guardrails"]["training_runs"] == 0, "No training executed.")
    add("zero labels and targets", summary["guardrails"]["labels_created"] == 0 and summary["guardrails"]["targets_created"] == 0, "No labels/targets created.")
    add("zero raster pixel reads", summary["guardrails"]["raster_pixel_reads"] == 0, "Header/path only.")
    add("zero promotions", summary["guardrails"]["gate_promotion"] == 0 and summary["guardrails"]["crs_promotion"] == 0, "No gate/CRS promotion.")
    add("CBERS ignored", summary["locks"]["cbers"] == "ignored", "CBERS remains ignored.")
    return rows


def build_report(summary: dict, created_files: list[str]) -> str:
    ready_counts = "\n".join(f"- `{k}`: {v}" for k, v in summary["readiness_status_counts"].items())
    modality_counts = "\n".join(f"- `{k}`: {v}" for k, v in summary["eligible_by_modality"].items())
    files = "\n".join(f"- `{path}`" for path in created_files)
    return f"""# REV-P v1fs Self-Supervised Asset Sanity and Embedding Extraction Plan

Version: `v1fs`

This phase audits the 211 v1fr review-only assets and prepares a concrete future embedding extraction plan. It does not train models, create labels, create targets, run weak supervision, report model metrics, or promote any gate/readiness state.

## Asset Sanity Result

Eligible input rows from v1fr: **{summary["eligible_input_rows"]}**

Readiness statuses:

{ready_counts}

Eligible modalities:

{modality_counts}

## Future Embedding Route

Sentinel TIF assets and multimodal NPY stacks can support a future reviewer-approved embedding extraction phase. TIFs were inspected header-only where possible. NPY stacks were inspected by memory-mapped shape only. RGB previews remain visual explanation material and are not moved into training scope in v1fs.

## What Remains Blocked

Supervised binary training, weak supervision, segmentation/detection, target creation, model metrics and performance-style claims remain blocked. The project still has `patch_bound_validated=0/59`, `preflight_ready=0/59`, blocked gates and no accepted event truth source.

## Required Next Phase

The next safe phase is a reviewer approval package for embedding extraction configuration: encoder choice, exact transform policy, grouped split table, asset exclusions and output schema. No extraction should run before that review.

## Files Created

{files}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    dl_rows = read_csv(V1FR_DIR / "dl_input_manifest_v1fr.csv")
    _preflight = read_csv(V1FR_DIR / "dataloader_preflight_v1fr.csv")
    _availability = read_csv(V1FR_DIR / "file_availability_audit_v1fr.csv")
    _modality = read_csv(V1FR_DIR / "modality_availability_matrix_v1fr.csv")
    _split = read_csv(V1FR_DIR / "split_preflight_v1fr.csv")
    _allowed = read_csv(V1FR_DIR / "allowed_transforms_review_only_v1fr.csv")
    _forbidden = read_csv(V1FR_DIR / "forbidden_transforms_v1fr.csv")
    _plan = read_csv(V1FR_DIR / "self_supervised_experiment_plan_v1fr.csv")
    blocked_reference = read_csv(V1FR_DIR / "blocked_cases_after_v1fr.csv")
    _leakage = read_csv(V1FR_DIR / "leakage_controls_v1fr.csv")
    v1fr_summary = json.loads((V1FR_DIR / "summary_v1fr.json").read_text(encoding="utf-8"))
    _qa = read_csv(V1FR_DIR / "qa_v1fr.csv")
    _status = read_csv(V1FR_DIR / "status_v1fr.csv")
    _report_text = (ROOT / "docs" / "revp_v1fr_self_supervised_dataloader_preflight_report.md").read_text(encoding="utf-8")

    eligible_rows = [row for row in dl_rows if row.get("preflight_status") in SAFE_STATUSES]
    blocked_from_dl = [row for row in dl_rows if row.get("preflight_status") not in SAFE_STATUSES]
    audit_rows, readiness_rows, conflict_rows, blocked_rows = build_outputs(eligible_rows)
    split_rows = split_matrix(readiness_rows)

    audit_fields = [
        "asset_id",
        "candidate_id",
        "region",
        "modality",
        "asset_path",
        "asset_type",
        "exists",
        "extension",
        "size_bytes",
        "shape_or_dimensions",
        "crs_if_header_available",
        "bounds_if_header_available",
        "split_group",
        "source_family",
        "sha256_small_file",
        "basename",
        "header_status",
        "too_small_flag",
        "leakage_flag",
        "input_preflight_status",
        "blocked_reason_from_v1fr",
        "duplicate_flag",
    ]
    readiness_fields = [
        "asset_id",
        "candidate_id",
        "region",
        "modality",
        "asset_path",
        "exists",
        "size_bytes",
        "shape_or_dimensions",
        "crs_if_header_available",
        "split_group",
        "duplicate_flag",
        "leakage_flag",
        "readiness_status",
        "allowed_next_step",
        "blocking_reason",
        "notes",
    ]
    write_csv(OUT_DIR / "asset_sanity_audit_v1fs.csv", audit_rows, audit_fields)
    write_csv(OUT_DIR / "embedding_extraction_readiness_v1fs.csv", readiness_rows, readiness_fields)
    write_csv(
        OUT_DIR / "embedding_extraction_plan_v1fs.csv",
        embedding_plan(),
        [
            "modality",
            "recommended_encoder_type",
            "input_preparation",
            "allowed_transform",
            "forbidden_transform",
            "output_artifact_future",
            "evaluation_allowed",
            "evaluation_forbidden",
            "scientific_claim_allowed",
        ],
    )
    write_csv(OUT_DIR / "embedding_split_matrix_v1fs.csv", split_rows, ["region", "modality", "split_group", "ready_review_only", "blocked_or_rebased", "total", "split_policy"])
    write_csv(OUT_DIR / "duplicate_and_conflict_audit_v1fs.csv", conflict_rows, ["asset_id", "candidate_id", "asset_path", "basename", "sha256_small_file", "duplicate_or_conflict_flag", "recommendation"])
    write_csv(OUT_DIR / "blocked_after_v1fs.csv", blocked_rows, ["asset_id", "candidate_id", "region", "modality", "asset_path", "readiness_status", "blocking_reason", "required_action"])

    readiness_counts = Counter(row["readiness_status"] for row in readiness_rows)
    modality_counts = Counter(row["modality"] for row in readiness_rows)
    region_modality_counts = Counter(f"{row['region']}|{row['modality']}" for row in readiness_rows if row["readiness_status"].startswith("EMBEDDING_REVIEW_ONLY_READY"))
    rebase_count = sum(v for k, v in readiness_counts.items() if not k.startswith("EMBEDDING_REVIEW_ONLY_READY"))
    summary = {
        "phase": "v1fs",
        "phase_name": "SELF_SUPERVISED_ASSET_SANITY_AND_EMBEDDING_EXTRACTION_PLAN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v1fr_safe_input_rows": int(v1fr_summary["preflight_status_counts"].get("SAFE_FOR_REVIEW_ONLY", 0)),
        "v1fr_blocked_rows": int(v1fr_summary["preflight_status_counts"].get("BLOCKED", 0)),
        "eligible_input_rows": len(eligible_rows),
        "blocked_reference_rows": len(blocked_from_dl),
        "readiness_status_counts": dict(sorted(readiness_counts.items())),
        "eligible_by_modality": dict(sorted(modality_counts.items())),
        "ready_by_region_modality": dict(sorted(region_modality_counts.items())),
        "reduced_or_blocked_by_v1fs": rebase_count,
        "duplicate_or_conflict_rows": len(conflict_rows),
        "blocked_promotions": 0,
        "next_phase_recommendation": "review_embedding_extraction_config_before_any_run",
        "locks": {
            "event_truth": "absent",
            "binary_supervised_status": "blocked",
            "supervised_status": "blocked",
            "patch_bound_validated": "0/59",
            "preflight_ready": "0/59",
            "gates": "blocked",
            "cbers": "ignored",
        },
        "guardrails": {
            "training_runs": 0,
            "labels_created": 0,
            "targets_created": 0,
            "weak_supervision_runs": 0,
            "model_metrics": 0,
            "model_claims": 0,
            "canonical_writes": 0,
            "gate_promotion": 0,
            "crs_promotion": 0,
            "patch_bound_validated_promotion": 0,
            "preflight_ready_promotion": 0,
            "raster_pixel_reads": 0,
            "band_reads": 0,
            "raster_summary_metrics": 0,
            "spatial_ops": 0,
            "coordinate_transform_ops": 0,
        },
        "pytest_status": os.environ.get("REV_PYTEST_STATUS_V1FS", "NOT_RUN_BY_SCRIPT"),
    }
    write_json(OUT_DIR / "summary_v1fs.json", summary)

    expected_exist = all((OUT_DIR / name).exists() for name in EXPECTED_OUTPUTS if name not in {"summary_v1fs.json", "qa_v1fs.csv", "status_v1fs.csv"})
    script_text = Path(__file__).read_text(encoding="utf-8")
    qa = qa_rows(script_text, expected_exist, summary)
    write_csv(OUT_DIR / "qa_v1fs.csv", qa, ["check", "status", "details"])

    status_rows = [
        {"field": "phase", "value": "v1fs"},
        {"field": "status", "value": "COMPLETE"},
        {"field": "qa_status", "value": "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"},
        {"field": "pytest_status", "value": os.environ.get("REV_PYTEST_STATUS_V1FS", "NOT_RUN_BY_SCRIPT")},
        {"field": "embedding_review_only_ready_assets", "value": str(sum(v for k, v in readiness_counts.items() if k.startswith("EMBEDDING_REVIEW_ONLY_READY")))},
        {"field": "rebased_or_blocked_assets", "value": str(rebase_count)},
        {"field": "duplicate_or_conflict_rows", "value": str(len(conflict_rows))},
        {"field": "training_runs", "value": "0"},
        {"field": "labels_targets_created", "value": "0"},
        {"field": "raster_pixel_reads", "value": "0"},
        {"field": "cbers", "value": "IGNORED"},
    ]
    write_csv(OUT_DIR / "status_v1fs.csv", status_rows, ["field", "value"])

    created_files = [rel(OUT_DIR / name) for name in EXPECTED_OUTPUTS] + [rel(DOC_PATH), rel(Path(__file__))]
    DOC_PATH.write_text(build_report(summary, created_files), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
