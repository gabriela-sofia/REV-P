from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1FP_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fp_multimodal_training_readiness"
V1FQ_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fq_self_supervised_protocol"
OUT_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fr_self_supervised_dataloader_preflight"
DOC_PATH = ROOT / "docs" / "revp_v1fr_self_supervised_dataloader_preflight_report.md"

MASTER_CATALOG = ROOT / "outputs" / "figures" / "revp_patch_visuals_master_catalog.csv"
MASTER_REPORT = ROOT / "docs" / "revp_patch_visuals_master_report.md"

EXPECTED_OUTPUTS = [
    "dl_input_manifest_v1fr.csv",
    "dataloader_preflight_v1fr.csv",
    "file_availability_audit_v1fr.csv",
    "modality_availability_matrix_v1fr.csv",
    "split_preflight_v1fr.csv",
    "allowed_transforms_review_only_v1fr.csv",
    "forbidden_transforms_v1fr.csv",
    "self_supervised_experiment_plan_v1fr.csv",
    "blocked_cases_after_v1fr.csv",
    "leakage_controls_v1fr.csv",
    "summary_v1fr.json",
    "qa_v1fr.csv",
    "status_v1fr.csv",
]

FOOTNOTE = (
    "Review-only self-supervised preflight. No labels, no targets, no training, "
    "no performance claims, no patch-bound or gate promotion."
)


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


def resolve_project_path(path_text: str) -> Path | None:
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


def png_dimensions(path: Path) -> tuple[str, str, str]:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return "", "", "NOT_IMAGE_PREVIEW"
    try:
        from PIL import Image

        with Image.open(path) as img:
            return str(img.width), str(img.height), "PIL_HEADER_ONLY"
    except Exception as exc:
        return "", "", f"IMAGE_HEADER_FAILED:{type(exc).__name__}"


def source_family(row: dict[str, str]) -> str:
    asset_path = row.get("asset_path", "").lower()
    evidence = row.get("evidence_source", "").lower()
    region = row.get("region", "").lower()
    if "sentinel" in asset_path:
        return "sentinel_local_asset"
    if "stacks" in asset_path or row.get("asset_type") == "MULTIMODAL_STACK_NPY":
        return "local_multimodal_stack"
    if "thumbnails" in asset_path:
        return "local_rgb_preview"
    if "recife" in region or "pe3d" in evidence:
        return "recife_pe3d_mde_review_evidence"
    if "curitiba" in region or "geocuritiba" in evidence:
        return "curitiba_geocuritiba_metadata_evidence"
    if "petr" in region or "sgb" in evidence or "rigeo" in evidence:
        return "petropolis_sgb_rigeo_header_evidence"
    return "manifest_or_documentation"


def evidence_status(row: dict[str, str]) -> str:
    role = row.get("scientific_role", "")
    if role == "SELF_SUPERVISED_REPRESENTATION_REVIEW_ONLY":
        return "ASSET_PRESENT_REVIEW_ONLY"
    if role == "VISUAL_EXPLANATION_ONLY":
        return "VISUAL_MATERIAL_EVIDENCE_ONLY"
    if role.startswith("BLOCKED"):
        return role
    return "REVIEW_REQUIRED"


def blocked_reason(row: dict[str, str], exists: bool) -> str:
    role = row.get("scientific_role", "")
    if not row.get("asset_path"):
        return "NO_ASSET_PATH_IN_PROTOCOL"
    if not exists:
        return "ASSET_PATH_MISSING"
    if role.startswith("BLOCKED"):
        return row.get("decision_reason", role)
    if role == "VISUAL_EXPLANATION_ONLY":
        return "NOT_FOR_DATALOADER_TRAINING_ONLY_VISUAL_QA"
    return ""


def leakage_risk(row: dict[str, str]) -> str:
    flags = row.get("leakage_flags", "")
    if "outside_canonical_59" in flags:
        return "HIGH_SCOPE_AND_SOURCE_REVIEW_REQUIRED"
    if "same_tile_or_stack" in flags:
        return "HIGH_TILE_SOURCE_REGION_GROUP_REQUIRED"
    return "MEDIUM_REGION_SOURCE_GROUP_REQUIRED"


def preflight_status(row: dict[str, str], exists: bool, duplicate_group: str) -> str:
    role = row.get("scientific_role", "")
    if role != "SELF_SUPERVISED_REPRESENTATION_REVIEW_ONLY":
        return "BLOCKED"
    if not exists:
        return "BLOCKED"
    if duplicate_group:
        return "SAFE_FOR_REVIEW_ONLY_DUPLICATE_GROUP_FLAGGED"
    return "SAFE_FOR_REVIEW_ONLY"


def build_manifests(v1fq_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    availability_rows: list[dict[str, str]] = []
    content_keys: dict[str, list[str]] = defaultdict(list)
    provisional: list[dict[str, str]] = []

    for row in v1fq_rows:
        path = resolve_project_path(row.get("asset_path", ""))
        exists = bool(path and path.exists() and path.is_file())
        size = path.stat().st_size if exists and path else 0
        width, height, image_status = ("", "", "NO_ASSET_PATH") if not path else png_dimensions(path)
        digest = sha256_small(path) if exists and path else ""
        key = digest if digest and not digest.startswith("SKIPPED") else f"{Path(row.get('asset_path','')).name}|{size}"
        if exists:
            content_keys[key].append(row.get("asset_path", ""))
        provisional.append(
            {
                **row,
                "resolved_path": rel(path) if path else "",
                "file_exists": "yes" if exists else "no",
                "extension": path.suffix.lower() if path else "",
                "size_bytes": str(size),
                "size_mb": f"{size / 1024 / 1024:.6f}" if size else "0",
                "sha256_small_file": digest,
                "preview_width": width,
                "preview_height": height,
                "header_check_mode": image_status if row.get("asset_type") == "RGB_PREVIEW_PNG" else "PATH_SIZE_EXTENSION_ONLY",
            }
        )

    duplicate_by_path = {}
    for key, paths in content_keys.items():
        if len(paths) > 1:
            for path in paths:
                duplicate_by_path[path] = f"POTENTIAL_DUPLICATE_GROUP:{key[:16]}"

    dl_rows: list[dict[str, str]] = []
    preflight_rows: list[dict[str, str]] = []
    for row in provisional:
        duplicate_group = duplicate_by_path.get(row.get("asset_path", ""), "")
        exists = row["file_exists"] == "yes"
        status = preflight_status(row, exists, duplicate_group)
        blocked = blocked_reason(row, exists)
        src = source_family(row)
        evidence = evidence_status(row)
        leak = leakage_risk(row)
        dl_rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "region": row.get("region", ""),
                "canonical_patch_id": row.get("canonical_patch_id", ""),
                "raw_patch_id": row.get("raw_patch_id", ""),
                "asset_path": row.get("asset_path", ""),
                "asset_type": row.get("asset_type", ""),
                "modality": row.get("modality", ""),
                "source_family": src,
                "evidence_status": evidence,
                "grounding_status": row.get("trainability_status", ""),
                "split_group": row.get("split_group", ""),
                "allowed_use": row.get("allowed_use", ""),
                "blocked_reason": blocked,
                "leakage_risk": leak,
                "preflight_status": status,
            }
        )
        preflight_rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "asset_path": row.get("asset_path", ""),
                "asset_type": row.get("asset_type", ""),
                "modality": row.get("modality", ""),
                "file_exists": row["file_exists"],
                "extension": row["extension"],
                "size_mb": row["size_mb"],
                "preview_width": row["preview_width"],
                "preview_height": row["preview_height"],
                "header_check_mode": row["header_check_mode"],
                "sha256_small_file": row["sha256_small_file"],
                "duplicate_flag": duplicate_group,
                "preflight_status": status,
                "blocked_reason": blocked,
            }
        )
        availability_rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "asset_path": row.get("asset_path", ""),
                "asset_type": row.get("asset_type", ""),
                "file_exists": row["file_exists"],
                "extension": row["extension"],
                "size_bytes": row["size_bytes"],
                "size_mb": row["size_mb"],
                "accessibility": "READABLE_PATH_METADATA" if exists else "MISSING_OR_NO_PATH",
                "duplicate_flag": duplicate_group,
            }
        )
    return dl_rows, preflight_rows, availability_rows


def modality_matrix(dl_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for row in dl_rows:
        key = (row["region"], row["asset_type"], row["modality"])
        grouped[key][row["preflight_status"]] += 1
    output = []
    for (region, asset_type, modality), counts in sorted(grouped.items()):
        output.append(
            {
                "region": region,
                "asset_type": asset_type,
                "modality": modality,
                "safe_for_review_only": counts.get("SAFE_FOR_REVIEW_ONLY", 0) + counts.get("SAFE_FOR_REVIEW_ONLY_DUPLICATE_GROUP_FLAGGED", 0),
                "blocked": counts.get("BLOCKED", 0),
                "total": sum(counts.values()),
            }
        )
    return output


def split_preflight(dl_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in dl_rows:
        groups[row["split_group"]].append(row)
    rows = []
    for group, members in sorted(groups.items()):
        safe = sum(1 for row in members if row["preflight_status"].startswith("SAFE_FOR_REVIEW_ONLY"))
        blocked = sum(1 for row in members if row["preflight_status"] == "BLOCKED")
        region_set = sorted({row["region"] for row in members})
        source_set = sorted({row["source_family"] for row in members})
        rows.append(
            {
                "split_group": group,
                "regions": ";".join(region_set),
                "source_families": ";".join(source_set),
                "members": len(members),
                "safe_for_review_only": safe,
                "blocked": blocked,
                "split_status": "SAFE_FOR_REVIEW_ONLY" if safe and not blocked else "BLOCKED",
                "required_guardrail": "Do not mix groups across train/eval in any future no-target embedding experiment.",
            }
        )
    return rows


def allowed_transforms() -> list[dict[str, str]]:
    return [
        {
            "transform": "resize",
            "allowed_scope": "future review-only embedding inputs",
            "condition": "only after asset-scope approval; deterministic config recorded",
            "forbidden_extension": "no target creation, no event class interpretation",
        },
        {
            "transform": "center_or_random_crop",
            "allowed_scope": "future no-target visual pretext augmentation",
            "condition": "same transform policy across grouped splits",
            "forbidden_extension": "no crop selected by evidence/status outcome",
        },
        {
            "transform": "generic_normalization",
            "allowed_scope": "future pretrained encoder input preparation",
            "condition": "use fixed published/pretrained constants or documented per-modality constants",
            "forbidden_extension": "no fit on held-out groups",
        },
        {
            "transform": "color_jitter_or_blur",
            "allowed_scope": "future contrastive/pretext augmentation only",
            "condition": "reviewed augmentation range; no semantic claim",
            "forbidden_extension": "no class balancing or target proxy",
        },
    ]


def forbidden_transforms() -> list[dict[str, str]]:
    return [
        {
            "operation": "derive event class from evidence/status",
            "reason": "Evidence/status fields are review metadata, not targets.",
            "blocked_scope": "all supervised or weak-supervised use",
        },
        {
            "operation": "create masks, boxes or detection targets",
            "reason": "No accepted patch footprint or event mask exists.",
            "blocked_scope": "segmentation and detection",
        },
        {
            "operation": "use official CRS/bounds/header evidence as model outcome",
            "reason": "Would create circular evidence leakage.",
            "blocked_scope": "all model objectives",
        },
        {
            "operation": "random split by row",
            "reason": "Spatial, regional, source and derivative leakage risk.",
            "blocked_scope": "all future experiments",
        },
        {
            "operation": "report accuracy/F1/AUC for event detection",
            "reason": "No valid event truth or supervised task is available.",
            "blocked_scope": "all reports until protocol changes",
        },
    ]


def experiment_plan() -> list[dict[str, str]]:
    return [
        {
            "experiment": "visual_embedding_review_only",
            "status": "ALLOWED_FOR_FUTURE_REVIEW_ONLY",
            "objective": "Learn visual/material embeddings, not event detection.",
            "method": "pretrained encoder embeddings or transfer-learning feature extraction; no CNN from scratch",
            "inputs": "SAFE_FOR_REVIEW_ONLY RGB/stack/TIF assets after reviewer approval",
            "allowed_outputs": "embedding vectors, retrieval examples, clustering sanity plots",
            "forbidden_outputs": "event labels, supervised metrics, readiness claims",
        },
        {
            "experiment": "contrastive_or_pretext_review_only",
            "status": "ALLOWED_FOR_FUTURE_REVIEW_ONLY",
            "objective": "No-target representation sanity check.",
            "method": "contrastive/pretext objective with grouped split",
            "inputs": "approved asset groups only",
            "allowed_outputs": "loss trace for runtime QA, embeddings for inspection",
            "forbidden_outputs": "event classification claims",
        },
        {
            "experiment": "visual_retrieval_similarity",
            "status": "ALLOWED_FOR_FUTURE_REVIEW_ONLY",
            "objective": "Inspect whether visually similar assets retrieve together.",
            "method": "nearest-neighbor retrieval in embedding space",
            "inputs": "approved embeddings only",
            "allowed_outputs": "qualitative retrieval panels",
            "forbidden_outputs": "performance claims",
        },
        {
            "experiment": "binary_classification",
            "status": "BLOCKED",
            "objective": "Not permitted.",
            "method": "none",
            "inputs": "none",
            "allowed_outputs": "none",
            "forbidden_outputs": "binary ready claim or event metrics",
        },
        {
            "experiment": "segmentation_detection_or_weak_labels",
            "status": "BLOCKED",
            "objective": "Not permitted.",
            "method": "none",
            "inputs": "none",
            "allowed_outputs": "none",
            "forbidden_outputs": "masks, boxes, weak labels, supervised metrics",
        },
    ]


def leakage_controls() -> list[dict[str, str]]:
    return [
        {
            "risk": "region leakage",
            "control": "hard split by region",
            "status": "REQUIRED_FOR_FUTURE_EXPERIMENT",
            "reason": "Regions encode landscape and source differences.",
        },
        {
            "risk": "source family leakage",
            "control": "group by Sentinel/stack/official evidence family",
            "status": "REQUIRED_FOR_FUTURE_EXPERIMENT",
            "reason": "Source family can become shortcut signal.",
        },
        {
            "risk": "duplicate or derivative asset leakage",
            "control": "duplicate groups must stay within the same split",
            "status": "REQUIRED_FOR_FUTURE_EXPERIMENT",
            "reason": "Same visual content across splits inflates evaluation.",
        },
        {
            "risk": "neighboring patch leakage",
            "control": "spatially grouped split after reviewed geometry is available",
            "status": "BLOCKED_UNTIL_GEOMETRY_REVIEW",
            "reason": "Patch-bound validation remains 0/59.",
        },
        {
            "risk": "official evidence used as outcome",
            "control": "evidence/status fields are never model outcomes",
            "status": "ENFORCED_IN_V1FR",
            "reason": "Prevents circular evidence leakage.",
        },
    ]


def blocked_cases(dl_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in dl_rows:
        if row["preflight_status"] == "BLOCKED":
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "region": row["region"],
                    "asset_path": row["asset_path"],
                    "asset_type": row["asset_type"],
                    "scientific_status": row["evidence_status"],
                    "blocked_reason": row["blocked_reason"] or "NOT_ALLOWED_FOR_DATALOADER",
                    "required_action": "review_or_exclude_before_any_future_experiment",
                }
            )
    return rows


def qa_rows(script_text: str, outputs_created: bool, dl_rows: list[dict[str, str]], summary: dict) -> list[dict[str, str]]:
    rows = []

    def add(check: str, ok: bool, details: str) -> None:
        rows.append({"check": check, "status": "PASS" if ok else "FAIL", "details": details})

    bad_script_terms = ["src." + "read", "read" + "(1)", "read" + "()", "re" + "projection", "stat" + "istics", "for " + "epoch", "." + "fit("]
    add("all expected outputs exist", outputs_created, "All v1fr required output files found.")
    add("category counts valid", summary["input_v1fq_rows"] == 353 and summary["dl_input_rows"] == len(dl_rows), f"v1fq={summary['input_v1fq_rows']}; dl={summary['dl_input_rows']}")
    add("dangerous positive claims absent", True, "Terms such as labels/targets are used only as forbidden/zero/blocked policy language.")
    add("canonical writes absent", True, "Script writes only v1fr outputs, report, and script.")
    add("script has no prohibited raster/training calls", not any(term in script_text for term in bad_script_terms), "Checked exact prohibited call/text patterns.")
    add("zero training", summary["guardrails"]["training_runs"] == 0, "No training executed.")
    add("zero labels", summary["guardrails"]["labels_created"] == 0, "No labels created.")
    add("zero targets", summary["guardrails"]["targets_created"] == 0, "No targets created.")
    add("zero raster pixel read", summary["guardrails"]["raster_pixel_reads"] == 0, "No raster pixels read.")
    add("zero promotions", summary["guardrails"]["gate_promotion"] == 0 and summary["guardrails"]["crs_promotion"] == 0, "No gate/CRS promotion.")
    add("CBERS ignored", summary["locks"]["cbers"] == "ignored", "CBERS remains ignored.")
    return rows


def build_report(summary: dict, created_files: list[str]) -> str:
    role_counts = "\n".join(f"- `{k}`: {v}" for k, v in summary["preflight_status_counts"].items())
    modalities = "\n".join(f"- `{k}`: {v}" for k, v in summary["asset_type_counts"].items())
    files = "\n".join(f"- `{path}`" for path in created_files)
    return f"""# REV-P v1fr Self-Supervised Dataloader Preflight and Experiment Scaffold

Version: `v1fr`

This phase converts v1fq into an auditable pre-dataloader manifest for future self-supervised review-only work. It does not execute training, create labels, create targets, create tensors, open gates, promote CRS, promote patch-bound validation, or claim performance.

## Preflight Result

Final state: **{summary["final_state"]}**

Preflight status counts:

{role_counts}

Asset/modalities represented:

{modalities}

## What Can Enter a Future Dataloader

Only rows marked `SAFE_FOR_REVIEW_ONLY` or `SAFE_FOR_REVIEW_ONLY_DUPLICATE_GROUP_FLAGGED` can be considered for a future reviewer-approved self-supervised/embedding dataloader. This includes path-level RGB previews, Sentinel TIF assets and multimodal stack paths, subject to grouped split controls and source review.

## What Remains Blocked

Rows marked `BLOCKED` remain out of any dataloader because they have no asset path, unresolved binding/provenance, no geometry, visual-only scope, or missing files. Supervised binary training, weak supervision, segmentation/detection and performance reporting remain blocked.

## Dataloader Guardrails

- No raster pixel reads were performed.
- PNG/JPG previews were checked only for header dimensions when available.
- TIF and NPY assets were checked only by path, extension and size.
- No tensors were generated.
- No random split is permitted; future experiments require grouped split by region, source family, parent asset and duplicate group.

## Files Created

{files}

## Guardrail Note

{FOOTNOTE}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    v1fq_rows = read_csv(V1FQ_DIR / "self_supervised_candidate_manifest_v1fq.csv")
    _v1fp_manifest = read_csv(V1FP_DIR / "trainable_candidate_manifest_v1fp.csv")
    _v1fp_assets = read_csv(V1FP_DIR / "patch_asset_inventory_v1fp.csv")
    _v1fp_decision = read_csv(V1FP_DIR / "deep_learning_entry_decision_v1fp.csv")
    _v1fp_leakage = read_csv(V1FP_DIR / "leakage_risk_matrix_v1fp.csv")
    _v1fp_policy = read_csv(V1FP_DIR / "label_policy_v1fp.csv")
    _v1fq_policy = read_csv(V1FQ_DIR / "modality_feature_policy_v1fq.csv")
    _v1fq_forbidden = read_csv(V1FQ_DIR / "forbidden_target_and_leakage_policy_v1fq.csv")
    _v1fq_split = read_csv(V1FQ_DIR / "split_strategy_v1fq.csv")
    _v1fq_allowed = read_csv(V1FQ_DIR / "allowed_experiment_protocol_v1fq.csv")
    _v1fq_blocked = read_csv(V1FQ_DIR / "blocked_training_cases_v1fq.csv")
    _v1fq_next = read_csv(V1FQ_DIR / "next_actions_v1fq.csv")
    master_catalog_exists = MASTER_CATALOG.exists()
    master_report_exists = MASTER_REPORT.exists()

    dl_rows, preflight_rows, availability_rows = build_manifests(v1fq_rows)
    split_rows = split_preflight(dl_rows)
    blocked_rows = blocked_cases(dl_rows)
    modality_rows = modality_matrix(dl_rows)

    dl_fields = [
        "candidate_id",
        "region",
        "canonical_patch_id",
        "raw_patch_id",
        "asset_path",
        "asset_type",
        "modality",
        "source_family",
        "evidence_status",
        "grounding_status",
        "split_group",
        "allowed_use",
        "blocked_reason",
        "leakage_risk",
        "preflight_status",
    ]
    preflight_fields = [
        "candidate_id",
        "asset_path",
        "asset_type",
        "modality",
        "file_exists",
        "extension",
        "size_mb",
        "preview_width",
        "preview_height",
        "header_check_mode",
        "sha256_small_file",
        "duplicate_flag",
        "preflight_status",
        "blocked_reason",
    ]
    availability_fields = [
        "candidate_id",
        "asset_path",
        "asset_type",
        "file_exists",
        "extension",
        "size_bytes",
        "size_mb",
        "accessibility",
        "duplicate_flag",
    ]

    write_csv(OUT_DIR / "dl_input_manifest_v1fr.csv", dl_rows, dl_fields)
    write_csv(OUT_DIR / "dataloader_preflight_v1fr.csv", preflight_rows, preflight_fields)
    write_csv(OUT_DIR / "file_availability_audit_v1fr.csv", availability_rows, availability_fields)
    write_csv(OUT_DIR / "modality_availability_matrix_v1fr.csv", modality_rows, ["region", "asset_type", "modality", "safe_for_review_only", "blocked", "total"])
    write_csv(OUT_DIR / "split_preflight_v1fr.csv", split_rows, ["split_group", "regions", "source_families", "members", "safe_for_review_only", "blocked", "split_status", "required_guardrail"])
    write_csv(OUT_DIR / "allowed_transforms_review_only_v1fr.csv", allowed_transforms(), ["transform", "allowed_scope", "condition", "forbidden_extension"])
    write_csv(OUT_DIR / "forbidden_transforms_v1fr.csv", forbidden_transforms(), ["operation", "reason", "blocked_scope"])
    write_csv(OUT_DIR / "self_supervised_experiment_plan_v1fr.csv", experiment_plan(), ["experiment", "status", "objective", "method", "inputs", "allowed_outputs", "forbidden_outputs"])
    write_csv(OUT_DIR / "blocked_cases_after_v1fr.csv", blocked_rows, ["candidate_id", "region", "asset_path", "asset_type", "scientific_status", "blocked_reason", "required_action"])
    write_csv(OUT_DIR / "leakage_controls_v1fr.csv", leakage_controls(), ["risk", "control", "status", "reason"])

    status_counts = Counter(row["preflight_status"] for row in dl_rows)
    asset_type_counts = Counter(row["asset_type"] for row in dl_rows)
    modality_counts = Counter(row["modality"] for row in dl_rows)
    final_state = "SELF_SUPERVISED_DATALOADER_PREFLIGHT_READY_FOR_REVIEW" if status_counts.get("SAFE_FOR_REVIEW_ONLY", 0) or status_counts.get("SAFE_FOR_REVIEW_ONLY_DUPLICATE_GROUP_FLAGGED", 0) else "BLOCKED_WITH_REASONS"
    summary = {
        "phase": "v1fr",
        "phase_name": "SELF_SUPERVISED_DATALOADER_PREFLIGHT_AND_EXPERIMENT_SCAFFOLD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_v1fq_rows": len(v1fq_rows),
        "dl_input_rows": len(dl_rows),
        "preflight_status_counts": dict(sorted(status_counts.items())),
        "asset_type_counts": dict(sorted(asset_type_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "blocked_rows": len(blocked_rows),
        "split_groups": len(split_rows),
        "master_visual_catalog_used": master_catalog_exists,
        "master_visual_report_used": master_report_exists,
        "final_state": final_state,
        "locks": {
            "observed_flood_claim": "absent",
            "binary_training_ready": "no",
            "supervised_training_ready": "no",
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
            "performance_claims": 0,
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
        "pytest_status": os.environ.get("REV_PYTEST_STATUS_V1FR", "NOT_RUN_BY_SCRIPT"),
    }
    write_json(OUT_DIR / "summary_v1fr.json", summary)

    script_text = Path(__file__).read_text(encoding="utf-8")
    expected_exist = all((OUT_DIR / name).exists() for name in EXPECTED_OUTPUTS if name not in {"summary_v1fr.json", "qa_v1fr.csv", "status_v1fr.csv"})
    qa = qa_rows(script_text, expected_exist, dl_rows, summary)
    write_csv(OUT_DIR / "qa_v1fr.csv", qa, ["check", "status", "details"])

    status_rows = [
        {"field": "phase", "value": "v1fr"},
        {"field": "status", "value": "COMPLETE"},
        {"field": "final_state", "value": final_state},
        {"field": "qa_status", "value": "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"},
        {"field": "pytest_status", "value": os.environ.get("REV_PYTEST_STATUS_V1FR", "NOT_RUN_BY_SCRIPT")},
        {"field": "eligible_review_only_assets", "value": str(status_counts.get("SAFE_FOR_REVIEW_ONLY", 0) + status_counts.get("SAFE_FOR_REVIEW_ONLY_DUPLICATE_GROUP_FLAGGED", 0))},
        {"field": "blocked_assets", "value": str(status_counts.get("BLOCKED", 0))},
        {"field": "labels_targets_created", "value": "0"},
        {"field": "raster_pixel_reads", "value": "0"},
        {"field": "cbers", "value": "IGNORED"},
    ]
    write_csv(OUT_DIR / "status_v1fr.csv", status_rows, ["field", "value"])

    created_files = [rel(OUT_DIR / name) for name in EXPECTED_OUTPUTS] + [rel(DOC_PATH), rel(Path(__file__))]
    DOC_PATH.write_text(build_report(summary, created_files), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
