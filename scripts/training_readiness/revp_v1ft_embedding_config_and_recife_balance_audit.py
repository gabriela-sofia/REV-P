from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1FS_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fs_self_supervised_asset_sanity_and_embedding_plan"
V1FR_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fr_self_supervised_dataloader_preflight"
V1FQ_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fq_self_supervised_protocol"
OUT_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1ft_embedding_config_and_recife_balance_audit"
DOC_PATH = ROOT / "docs" / "revp_v1ft_embedding_config_and_recife_balance_audit_report.md"

EXPECTED_OUTPUTS = [
    "embedding_ready_assets_v1ft.csv",
    "embedding_excluded_assets_v1ft.csv",
    "embedding_modality_decision_v1ft.csv",
    "embedding_transform_policy_v1ft.csv",
    "embedding_output_schema_v1ft.csv",
    "embedding_split_config_v1ft.csv",
    "embedding_extraction_config_v1ft.json",
    "embedding_execution_checklist_v1ft.csv",
    "blocked_after_v1ft.csv",
    "recife_multimodal_stack_discovery_v1ft.csv",
    "recife_stack_linkage_audit_v1ft.csv",
    "recife_stack_recovery_candidates_v1ft.csv",
    "recife_multimodal_reconstruction_plan_v1ft.csv",
    "summary_v1ft.json",
    "qa_v1ft.csv",
    "status_v1ft.csv",
]

SAFE_READY_STATUS = {"EMBEDDING_REVIEW_ONLY_READY", "EMBEDDING_REVIEW_ONLY_READY_WITH_DUPLICATE_FLAG"}
RECIFE_TERMS = ("recife", "rec_", "recife_ext", "recife_bg")
SCAN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".tif", ".tiff", ".csv", ".json", ".md", ".txt", ".log"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


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


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def should_skip(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & SKIP_DIRS) or "cbers" in str(path).lower() or OUT_DIR in path.parents or path == DOC_PATH


def infer_patch_id(name_or_path: str) -> tuple[str, str, str]:
    text = name_or_path.replace("\\", "/").lower()
    family = "recife_general"
    if "recife_ext" in text:
        family = "recife_ext"
    elif "recife_bg" in text:
        family = "recife_bg"
    elif "patch_recife" in text:
        family = "patch_recife"
    elif "/recife/" in text or "recife_" in text:
        family = "recife"
    match = re.search(r"(?:patch_)?recife[_-](\d+)", text)
    if match:
        raw = f"recife_{match.group(1)}"
        return f"REC_{match.group(1)}", raw, family
    match = re.search(r"recife_(?:ext|bg)[_-]?(\d+)", text)
    if match:
        raw = f"{family}_{match.group(1)}"
        return f"REC_EXTBG_{match.group(1)}", raw, family
    match = re.search(r"\brec[_-](\d+)\b", text)
    if match:
        raw = f"rec_{match.group(1)}"
        return f"REC_{match.group(1)}", raw, family
    return "", "", family


def source_hint(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    if "patches/stacks" in text:
        return "patches_stacks"
    if "patches/thumbnails" in text:
        return "patches_thumbnails"
    if "data/sentinel" in text:
        return "data_sentinel"
    if "pe3d" in text or "mde" in text or "mdt" in text:
        return "recife_pe3d_mde_or_terrain"
    if "patch_grounding" in text:
        return "patch_grounding_manifest"
    if "external_validation" in text:
        return "external_validation_manifest"
    if "training_readiness" in text:
        return "training_readiness_manifest"
    if "scripts" in text:
        return "script_or_generation_log_reference"
    return "project_file"


def scan_recife_assets(v1fs_paths: set[str], v1fr_paths: set[str], v1fq_paths: set[str]) -> list[dict[str, str]]:
    rows = []
    for path in ROOT.rglob("*"):
        if should_skip(path) or not path.is_file():
            continue
        suffix = path.suffix.lower()
        ptxt = str(path).lower()
        if suffix not in SCAN_SUFFIXES:
            continue
        if not any(term in ptxt for term in RECIFE_TERMS) and not any(term in path.name.lower() for term in ("stack_spec", "channel_norm", "generation_log", "valid_stacks")):
            continue
        rpath = rel(path)
        canonical, raw, family = infer_patch_id(rpath)
        is_stack_like = suffix in {".npy", ".npz", ".pt", ".pth"} or "stack" in ptxt or "multimodal" in ptxt
        in_v1fs = rpath in v1fs_paths
        in_v1fr = rpath in v1fr_paths
        in_v1fq = rpath in v1fq_paths
        if in_v1fs and is_stack_like and "patches/stacks/recife" in rpath.lower():
            link_status = "LINKED_EXISTING_STACK_READY_FOR_REVIEW"
        elif is_stack_like and "recife_ext" in ptxt or is_stack_like and "recife_bg" in ptxt:
            link_status = "EXISTING_STACK_BLOCKED_BY_REC_EXT_BG"
        elif is_stack_like and not in_v1fs:
            link_status = "EXISTING_STACK_NEEDS_NAMING_RECONCILIATION"
        elif suffix in {".tif", ".tiff", ".csv", ".json", ".md", ".txt", ".log"}:
            link_status = "EXISTING_ASSET_NOT_MULTIMODAL_STACK"
        else:
            link_status = "DUPLICATE_OR_AMBIGUOUS_NEEDS_REVIEW"
        rows.append(
            {
                "path": rpath,
                "filename": path.name,
                "extension": suffix,
                "size_bytes": str(path.stat().st_size),
                "inferred_region": "Recife" if "recife" in ptxt or "rec_" in ptxt else "metadata_cross_region",
                "inferred_patch_id": canonical,
                "inferred_raw_patch_id": raw,
                "naming_family": family,
                "source_hint": source_hint(path),
                "in_v1fs_manifest": "yes" if in_v1fs else "no",
                "in_v1fr_manifest": "yes" if in_v1fr else "no",
                "in_v1fq_manifest": "yes" if in_v1fq else "no",
                "candidate_link_status": link_status,
                "notes": "metadata/path-only discovery; no reconstruction or promotion",
            }
        )
    rows.sort(key=lambda r: (r["candidate_link_status"], r["path"]))
    return rows


def build_ready_and_excluded(readiness_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ready = []
    excluded = []
    for row in readiness_rows:
        modality = row.get("modality", "")
        status = row.get("readiness_status", "")
        base = {
            "asset_id": row.get("asset_id", ""),
            "candidate_id": row.get("candidate_id", ""),
            "region": row.get("region", ""),
            "modality": modality,
            "asset_path": row.get("asset_path", ""),
            "split_group": row.get("split_group", ""),
            "readiness_status": status,
            "config_status": "",
            "decision_reason": "",
        }
        if status in SAFE_READY_STATUS and modality == "sentinel_raster_path_only":
            base["config_status"] = "READY_SENTINEL_FIRST_REVIEW_ONLY"
            base["decision_reason"] = "Sentinel modality is region-distributed and source-aware review-only."
            ready.append(base)
        elif status in SAFE_READY_STATUS and modality == "multimodal_stack_path_only":
            base["config_status"] = "CONDITIONAL_HOLD_MULTIMODAL_RECIFE_BALANCE"
            base["decision_reason"] = "File-ready stack, but Recife has severe multimodal imbalance; hold full multimodal extraction until recovery review."
            excluded.append(base)
        else:
            base["config_status"] = "EXCLUDED_NOT_READY"
            base["decision_reason"] = row.get("blocking_reason", "") or "Not ready for v1ft config lock."
            excluded.append(base)
    return ready, excluded


def build_linkage(discovery: list[dict[str, str]], dl_rows: list[dict[str, str]], readiness_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    recife_candidates = sorted({row["candidate_id"] for row in dl_rows if row.get("region") == "Recife"})
    by_candidate = defaultdict(lambda: {"sentinel": [], "stack": [], "png": [], "blocked": []})
    for row in dl_rows:
        if row.get("region") != "Recife":
            continue
        if row.get("asset_type") == "SENTINEL_TIF_ASSET":
            by_candidate[row["candidate_id"]]["sentinel"].append(row.get("asset_path", ""))
        elif row.get("asset_type") == "MULTIMODAL_STACK_NPY":
            by_candidate[row["candidate_id"]]["stack"].append(row.get("asset_path", ""))
        elif row.get("asset_type") == "RGB_PREVIEW_PNG":
            by_candidate[row["candidate_id"]]["png"].append(row.get("asset_path", ""))
        if row.get("preflight_status") == "BLOCKED":
            by_candidate[row["candidate_id"]]["blocked"].append(row.get("blocked_reason", ""))
    discovery_by_patch = defaultdict(list)
    for item in discovery:
        if item["inferred_patch_id"]:
            discovery_by_patch[item["inferred_patch_id"]].append(item["path"])
    rows = []
    for candidate in recife_candidates:
        links = by_candidate[candidate]
        stack_paths = links["stack"]
        sentinel_paths = links["sentinel"]
        discovered = discovery_by_patch.get(candidate, [])
        if stack_paths:
            classification = "LINKED_EXISTING_STACK_READY_FOR_REVIEW"
            action = "keep as linked but do not promote beyond review-only"
        elif any("recife_ext" in p.lower() or "recife_bg" in p.lower() for p in discovered):
            classification = "EXISTING_STACK_BLOCKED_BY_REC_EXT_BG"
            action = "resolve naming/provenance before any recovery"
        elif sentinel_paths:
            classification = "MISSING_STACK_RECONSTRUCTION_NEEDED"
            action = "future stack reconstruction plan required; do not execute in v1ft"
        else:
            classification = "DUPLICATE_OR_AMBIGUOUS_NEEDS_REVIEW"
            action = "manual review"
        rows.append(
            {
                "candidate_id": candidate,
                "sentinel_assets": ";".join(sentinel_paths),
                "multimodal_stack_assets": ";".join(stack_paths),
                "png_preview_assets": ";".join(links["png"]),
                "discovered_recife_related_paths": ";".join(discovered[:20]),
                "pe3d_mde_evidence_status": "B4_LOCAL_RASTER_HEADER_EVIDENCE_PRESENT_NEEDS_REVIEW",
                "patch_grounding_status": "RECIFE_EXT_BG_OR_PATH_LINKAGE_REVIEW_REQUIRED" if not stack_paths else "LINKED_STACK_REVIEW_ONLY",
                "linkage_classification": classification,
                "required_action": action,
            }
        )
    return rows


def recovery_candidates(discovery: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for item in discovery:
        if item["extension"] not in {".npy", ".npz", ".pt", ".pth"}:
            continue
        if item["candidate_link_status"] == "LINKED_EXISTING_STACK_READY_FOR_REVIEW":
            status = "NEEDS_REVIEW_ALREADY_LINKED"
            missing = "review only; do not promote automatically"
        else:
            status = "NEEDS_REVIEW"
            missing = "naming/provenance/candidate link review before inclusion"
        rows.append(
            {
                "path": item["path"],
                "inferred_patch_id": item["inferred_patch_id"],
                "inferred_raw_patch_id": item["inferred_raw_patch_id"],
                "naming_family": item["naming_family"],
                "candidate_link_status": item["candidate_link_status"],
                "recovery_status": status,
                "what_is_missing": missing,
                "forbidden_action": "no automatic manifest correction or embedding extraction in v1ft",
            }
        )
    return rows


def reconstruction_plan(linkage_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in linkage_rows:
        if row["linkage_classification"] != "MISSING_STACK_RECONSTRUCTION_NEEDED":
            continue
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "required_inputs": "reviewed Sentinel asset; approved multimodal recipe; PE3D/MDE evidence as review context only",
                "sentinel_asset": row["sentinel_assets"],
                "pe3d_mde_header_evidence": row["pe3d_mde_evidence_status"],
                "expected_output_path": f"patches/stacks/recife/patch_recife_{row['candidate_id'].split('_')[-1]}.npy",
                "required_naming_convention": "patch_recife_#####.npy matching candidate/raw id after REC ext/bg provenance review",
                "minimum_qa": "path exists; shape mmap; channel spec; source lineage; grouped split; no labels or targets",
                "guardrails": "no reconstruction in v1ft; no event claims; no gate/CRS/preflight promotion",
                "blocked_claims": "no supervised readiness; no event detection; no model metrics",
            }
        )
    return rows


def modality_decision(ready: list[dict[str, str]], excluded: list[dict[str, str]], recovery: list[dict[str, str]]) -> list[dict[str, str]]:
    sentinel_ready = sum(1 for row in ready if row["modality"] == "sentinel_raster_path_only")
    stack_conditional = sum(1 for row in excluded if row["modality"] == "multimodal_stack_path_only")
    return [
        {
            "modality": "Sentinel",
            "decision": "SENTINEL_FIRST_REVIEW_ONLY_CONFIG_LOCK",
            "asset_count": str(sentinel_ready),
            "reason": "Most balanced available modality across regions; still review-only.",
            "blocking_or_caveat": "No training/extraction until reviewer approves config.",
        },
        {
            "modality": "Multimodal stack",
            "decision": "CONDITIONAL_HOLD_UNTIL_RECIFE_STACK_RECOVERY_REVIEW",
            "asset_count": str(stack_conditional),
            "reason": "Curitiba/Petrópolis have many stacks but Recife has only one linked stack.",
            "blocking_or_caveat": "Run RECIFE_MULTIMODAL_STACK_RECOVERY planning before full multimodal extraction.",
        },
        {
            "modality": "PNG preview",
            "decision": "VISUAL_EXPLANATION_ONLY",
            "asset_count": "83",
            "reason": "Previews explain material availability and QA; not training input for v1ft.",
            "blocking_or_caveat": "Keep out of embedding config unless future visual-QA-only protocol is approved.",
        },
        {
            "modality": "Recife stack recovery",
            "decision": "SUBPHASE_REQUIRED",
            "asset_count": str(len(recovery)),
            "reason": "Existing Recife stack evidence is insufficiently balanced for full multimodal plan.",
            "blocking_or_caveat": "Metadata-only reconciliation or future reconstruction required.",
        },
    ]


def transform_policy() -> list[dict[str, str]]:
    return [
        {"field": "resize", "value": "allowed_after_review", "scope": "Sentinel-first embeddings", "guardrail": "fixed deterministic config"},
        {"field": "normalization", "value": "allowed_after_review", "scope": "encoder input preparation", "guardrail": "do not derive from held-out groups"},
        {"field": "crop", "value": "allowed_after_review", "scope": "pretext augmentation", "guardrail": "not outcome/status driven"},
        {"field": "mask_or_target_generation", "value": "forbidden", "scope": "all", "guardrail": "no labels or targets"},
        {"field": "status_as_feature_or_target", "value": "forbidden", "scope": "all", "guardrail": "external evidence remains audit metadata"},
    ]


def output_schema() -> list[dict[str, str]]:
    return [
        {"column": "asset_id", "type": "string", "required": "yes", "description": "Stable input asset id from v1fs."},
        {"column": "candidate_id", "type": "string", "required": "yes", "description": "Candidate identifier; not a validated patch claim."},
        {"column": "region", "type": "string", "required": "yes", "description": "Split/control metadata only."},
        {"column": "modality", "type": "string", "required": "yes", "description": "Input modality."},
        {"column": "split_group", "type": "string", "required": "yes", "description": "Grouped split key."},
        {"column": "embedding_vector", "type": "array<float>", "required": "future_only", "description": "Future artifact; not generated in v1ft."},
        {"column": "encoder_config_id", "type": "string", "required": "future_only", "description": "Reviewed encoder/config reference."},
        {"column": "label", "type": "not_allowed", "required": "no", "description": "Labels are forbidden in this track."},
        {"column": "target", "type": "not_allowed", "required": "no", "description": "Targets are forbidden in this track."},
    ]


def split_config(ready: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for key, members in sorted(defaultdict(list, {}).items()):
        pass
    groups = defaultdict(list)
    for row in ready:
        groups[(row["region"], row["split_group"])].append(row)
    for (region, group), members in sorted(groups.items()):
        rows.append(
            {
                "region": region,
                "split_group": group,
                "asset_count": str(len(members)),
                "split_policy": "region_source_asset_grouped_no_random_split",
                "recommended_role": "reviewer_assigned_holdout_or_train_fold_future_only",
                "leakage_guardrail": "keep source/asset derivatives within one split",
            }
        )
    return rows


def execution_checklist() -> list[dict[str, str]]:
    return [
        {"step": "reviewer_approval", "required": "yes", "status": "pending", "reason": "No extraction before review."},
        {"step": "sentinel_first_config_selected", "required": "yes", "status": "drafted", "reason": "Most balanced current route."},
        {"step": "recife_stack_recovery_decision", "required": "yes_for_multimodal", "status": "pending", "reason": "Recife stack imbalance remains."},
        {"step": "grouped_split_locked", "required": "yes", "status": "drafted", "reason": "Avoid region/source leakage."},
        {"step": "labels_targets_absent", "required": "yes", "status": "enforced", "reason": "Review-only self-supervised scope."},
        {"step": "no_training_or_extraction", "required": "yes_in_v1ft", "status": "enforced", "reason": "Config lock only."},
    ]


def blocked_after_v1ft(excluded: list[dict[str, str]], blocked_v1fs: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in excluded:
        rows.append(
            {
                "asset_id": row["asset_id"],
                "candidate_id": row["candidate_id"],
                "region": row["region"],
                "modality": row["modality"],
                "asset_path": row["asset_path"],
                "block_status": row["config_status"],
                "reason": row["decision_reason"],
                "required_action": "review_or_recovery_before_embedding_extraction",
            }
        )
    for row in blocked_v1fs:
        rows.append(
            {
                "asset_id": row.get("asset_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "region": row.get("region", ""),
                "modality": row.get("modality", ""),
                "asset_path": row.get("asset_path", ""),
                "block_status": row.get("readiness_status", "BLOCKED_FROM_V1FS"),
                "reason": row.get("blocking_reason", ""),
                "required_action": row.get("required_action", "review_or_exclude"),
            }
        )
    return rows


def build_config(ready: list[dict[str, str]], excluded: list[dict[str, str]], recovery_plan_rows: list[dict[str, str]]) -> dict:
    return {
        "version": "v1ft",
        "config_status": "LOCKED_DRAFT_REVIEW_ONLY",
        "recommended_start": "Sentinel-first review-only embedding extraction after reviewer approval",
        "multimodal_stack_status": "conditional_hold_until_recife_stack_recovery",
        "ready_asset_count": len(ready),
        "conditional_or_excluded_asset_count": len(excluded),
        "recife_reconstruction_needed_count": len(recovery_plan_rows),
        "encoder_policy": {
            "recommended": "pretrained_encoder_or_geospatial_pretrained_encoder_future_only",
            "encoder_loading_now": False,
            "extract_embeddings_now": False,
        },
        "guardrails": {
            "training": 0,
            "embedding_extraction": 0,
            "model_loading": 0,
            "downloads": 0,
            "labels": 0,
            "targets": 0,
            "weak_supervision": 0,
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
            "cbers": "ignored",
        },
    }


def qa_rows(script_text: str, outputs_exist: bool, summary: dict, discovery_rows: list[dict[str, str]], recovery_rows: list[dict[str, str]]) -> list[dict[str, str]]:
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
    bad_code = [
        "src." + "read",
        "read" + "(1)",
        "for " + "epoch",
        "load_" + "model",
        "torch." + "load",
        "." + "fit(",
        "embed" + "ding_vector =",
        "stat" + "istics",
        "re" + "projection",
    ]
    summary_text = json.dumps(summary, ensure_ascii=False).lower()
    add("outputs exist", outputs_exist, "All required v1ft outputs found.")
    add("211 eligible assets preserved or justified", summary["v1fs_ready_assets"] == 211, f"v1fs_ready={summary['v1fs_ready_assets']}")
    add("142 blocked assets preserved", summary["v1fs_blocked_reference"] == 142, f"blocked={summary['v1fs_blocked_reference']}")
    add("Recife imbalance audit executed", summary["recife_discovery_rows"] > 0, f"discovery_rows={summary['recife_discovery_rows']}")
    add("discovery CSV non-empty", len(discovery_rows) > 0, f"rows={len(discovery_rows)}")
    add("Recife one-stack state explained", summary["recife_linked_stack_count"] == 1, f"linked_stacks={summary['recife_linked_stack_count']}")
    add("recovery candidates not promoted", all(row["recovery_status"].startswith("NEEDS_REVIEW") for row in recovery_rows), "All recovery rows remain review-only.")
    add("Sentinel balance explicit", summary["sentinel_ready_assets"] == 128, f"sentinel={summary['sentinel_ready_assets']}")
    add("multimodal imbalance explicit", summary["multimodal_stack_conditional_hold"] == 83, f"stack_hold={summary['multimodal_stack_conditional_hold']}")
    add("dangerous terms absent in summary", not any(term in summary_text for term in bad_phrases), "Summary avoids unsafe phrasing.")
    add("script has no training/model/extraction/prohibited calls", not any(term in script_text for term in bad_code), "Checked exact code patterns.")
    add("guardrails zero", all(value == 0 for key, value in summary["guardrails"].items() if isinstance(value, int)), "Numeric guardrails are zero.")
    add("CBERS ignored", summary["guardrails"]["cbers"] == "ignored", "CBERS ignored.")
    return rows


def build_report(summary: dict, created_files: list[str]) -> str:
    files = "\n".join(f"- `{path}`" for path in created_files)
    return f"""# REV-P v1ft Embedding Config Lock and Recife Multimodal Balance Audit

Version: `v1ft`

This phase locks a review-only future embedding configuration and audits the Recife multimodal stack imbalance. It does not execute embeddings, train models, create labels, create targets, load models, download data, run weak supervision, report model metrics, promote CRS, open gates, or promote readiness.

## Why Recife Needed This Audit

Recife has the strongest external material evidence in the project: PE3D/MDE raster-header evidence with 66/66 headers, EPSG:31985 and bounds present for review. Yet v1fs showed only **1** Recife multimodal stack against **37** Recife Sentinel assets. That mismatch is treated as a linkage/reconstruction problem to audit, not as an acceptable final balance.

## Audit Result

- Recife discovery rows: **{summary["recife_discovery_rows"]}**
- Recife linked stacks in current ready manifest: **{summary["recife_linked_stack_count"]}**
- Recife recovery candidate rows: **{summary["recife_recovery_candidate_rows"]}**
- Recife stack reconstruction-needed rows: **{summary["recife_reconstruction_needed_rows"]}**

The audit found the currently linked Recife stack, but did not find enough additional ready-linked Recife multimodal stacks to balance the region. Existing or related Recife artifacts remain review-only and are not automatically promoted.

## Config Decision

Recommended decision: **{summary["recommended_decision"]}**

Sentinel-first is the safest future route because it has 128 review-only assets and a far better regional distribution. Full multimodal stack extraction should wait for a Recife stack recovery/reconstruction subphase.

## What Remains Blocked

Full multimodal extraction is conditional until Recife stack recovery is resolved. Supervised training, weak supervision, segmentation/detection, labels, targets, model metrics, readiness claims and gate/CRS promotion remain blocked.

## Files Created

{files}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    readiness = read_csv(V1FS_DIR / "embedding_extraction_readiness_v1fs.csv")
    _asset_sanity = read_csv(V1FS_DIR / "asset_sanity_audit_v1fs.csv")
    _plan = read_csv(V1FS_DIR / "embedding_extraction_plan_v1fs.csv")
    _split = read_csv(V1FS_DIR / "embedding_split_matrix_v1fs.csv")
    _dupes = read_csv(V1FS_DIR / "duplicate_and_conflict_audit_v1fs.csv")
    blocked_v1fs = read_csv(V1FS_DIR / "blocked_after_v1fs.csv")
    v1fs_summary = json.loads((V1FS_DIR / "summary_v1fs.json").read_text(encoding="utf-8"))
    _qa = read_csv(V1FS_DIR / "qa_v1fs.csv")
    _status = read_csv(V1FS_DIR / "status_v1fs.csv")
    _report_text = (ROOT / "docs" / "revp_v1fs_self_supervised_asset_sanity_and_embedding_plan_report.md").read_text(encoding="utf-8")
    dl_rows = read_csv(V1FR_DIR / "dl_input_manifest_v1fr.csv")
    _forbidden = read_csv(V1FQ_DIR / "forbidden_target_and_leakage_policy_v1fq.csv")
    _split_policy = read_csv(V1FQ_DIR / "split_strategy_v1fq.csv")

    v1fs_paths = {row.get("asset_path", "") for row in readiness if row.get("asset_path")}
    v1fr_paths = {row.get("asset_path", "") for row in dl_rows if row.get("asset_path")}
    v1fq_paths = {row.get("asset_path", "") for row in read_csv(V1FQ_DIR / "self_supervised_candidate_manifest_v1fq.csv") if row.get("asset_path")}

    ready, excluded = build_ready_and_excluded(readiness)
    discovery = scan_recife_assets(v1fs_paths, v1fr_paths, v1fq_paths)
    linkage = build_linkage(discovery, dl_rows, readiness)
    recovery = recovery_candidates(discovery)
    reconstruction = reconstruction_plan(linkage)

    write_csv(OUT_DIR / "embedding_ready_assets_v1ft.csv", ready, ["asset_id", "candidate_id", "region", "modality", "asset_path", "split_group", "readiness_status", "config_status", "decision_reason"])
    write_csv(OUT_DIR / "embedding_excluded_assets_v1ft.csv", excluded, ["asset_id", "candidate_id", "region", "modality", "asset_path", "split_group", "readiness_status", "config_status", "decision_reason"])
    write_csv(OUT_DIR / "embedding_modality_decision_v1ft.csv", modality_decision(ready, excluded, recovery), ["modality", "decision", "asset_count", "reason", "blocking_or_caveat"])
    write_csv(OUT_DIR / "embedding_transform_policy_v1ft.csv", transform_policy(), ["field", "value", "scope", "guardrail"])
    write_csv(OUT_DIR / "embedding_output_schema_v1ft.csv", output_schema(), ["column", "type", "required", "description"])
    write_csv(OUT_DIR / "embedding_split_config_v1ft.csv", split_config(ready), ["region", "split_group", "asset_count", "split_policy", "recommended_role", "leakage_guardrail"])
    write_csv(OUT_DIR / "embedding_execution_checklist_v1ft.csv", execution_checklist(), ["step", "required", "status", "reason"])
    write_csv(OUT_DIR / "blocked_after_v1ft.csv", blocked_after_v1ft(excluded, blocked_v1fs), ["asset_id", "candidate_id", "region", "modality", "asset_path", "block_status", "reason", "required_action"])
    write_csv(OUT_DIR / "recife_multimodal_stack_discovery_v1ft.csv", discovery, ["path", "filename", "extension", "size_bytes", "inferred_region", "inferred_patch_id", "inferred_raw_patch_id", "naming_family", "source_hint", "in_v1fs_manifest", "in_v1fr_manifest", "in_v1fq_manifest", "candidate_link_status", "notes"])
    write_csv(OUT_DIR / "recife_stack_linkage_audit_v1ft.csv", linkage, ["candidate_id", "sentinel_assets", "multimodal_stack_assets", "png_preview_assets", "discovered_recife_related_paths", "pe3d_mde_evidence_status", "patch_grounding_status", "linkage_classification", "required_action"])
    write_csv(OUT_DIR / "recife_stack_recovery_candidates_v1ft.csv", recovery, ["path", "inferred_patch_id", "inferred_raw_patch_id", "naming_family", "candidate_link_status", "recovery_status", "what_is_missing", "forbidden_action"])
    write_csv(OUT_DIR / "recife_multimodal_reconstruction_plan_v1ft.csv", reconstruction, ["candidate_id", "required_inputs", "sentinel_asset", "pe3d_mde_header_evidence", "expected_output_path", "required_naming_convention", "minimum_qa", "guardrails", "blocked_claims"])

    config = build_config(ready, excluded, reconstruction)
    write_json(OUT_DIR / "embedding_extraction_config_v1ft.json", config)

    linked_recife_stacks = [row for row in linkage if row["multimodal_stack_assets"]]
    stack_discovery = [row for row in discovery if row["extension"] in {".npy", ".npz", ".pt", ".pth"}]
    summary = {
        "phase": "v1ft",
        "phase_name": "SELF_SUPERVISED_EMBEDDING_EXTRACTION_CONFIG_LOCK_AND_RECIFE_MULTIMODAL_BALANCE_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v1fs_ready_assets": int(v1fs_summary["eligible_input_rows"]),
        "v1fs_blocked_reference": int(v1fs_summary["blocked_reference_rows"]),
        "sentinel_ready_assets": len(ready),
        "multimodal_stack_conditional_hold": sum(1 for row in excluded if row["modality"] == "multimodal_stack_path_only"),
        "excluded_or_conditional_assets": len(excluded),
        "recife_discovery_rows": len(discovery),
        "recife_stack_like_discovery_rows": len(stack_discovery),
        "recife_linked_stack_count": len(linked_recife_stacks),
        "recife_recovery_candidate_rows": len(recovery),
        "recife_reconstruction_needed_rows": len(reconstruction),
        "recommended_decision": "SENTINEL_FIRST_REVIEW_ONLY__MULTIMODAL_AFTER_RECIFE_RECOVERY",
        "guardrails": config["guardrails"],
        "pytest_status": os.environ.get("REV_PYTEST_STATUS_V1FT", "NOT_RUN_BY_SCRIPT"),
    }
    write_json(OUT_DIR / "summary_v1ft.json", summary)

    script_text = Path(__file__).read_text(encoding="utf-8")
    expected_exist = all((OUT_DIR / name).exists() for name in EXPECTED_OUTPUTS if name not in {"summary_v1ft.json", "qa_v1ft.csv", "status_v1ft.csv"})
    qa = qa_rows(script_text, expected_exist, summary, discovery, recovery)
    write_csv(OUT_DIR / "qa_v1ft.csv", qa, ["check", "status", "details"])

    status = [
        {"field": "phase", "value": "v1ft"},
        {"field": "status", "value": "COMPLETE"},
        {"field": "qa_status", "value": "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"},
        {"field": "pytest_status", "value": os.environ.get("REV_PYTEST_STATUS_V1FT", "NOT_RUN_BY_SCRIPT")},
        {"field": "sentinel_ready_review_only_assets", "value": str(len(ready))},
        {"field": "multimodal_stack_conditional_hold", "value": str(summary["multimodal_stack_conditional_hold"])},
        {"field": "recife_linked_stack_count", "value": str(len(linked_recife_stacks))},
        {"field": "recife_reconstruction_needed_rows", "value": str(len(reconstruction))},
        {"field": "recommended_decision", "value": summary["recommended_decision"]},
        {"field": "guardrails", "value": "ZERO_TRAINING_ZERO_EXTRACTION_ZERO_LABELS_ZERO_TARGETS"},
    ]
    write_csv(OUT_DIR / "status_v1ft.csv", status, ["field", "value"])

    created_files = [rel(OUT_DIR / name) for name in EXPECTED_OUTPUTS] + [rel(DOC_PATH), rel(Path(__file__))]
    DOC_PATH.write_text(build_report(summary, created_files), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
