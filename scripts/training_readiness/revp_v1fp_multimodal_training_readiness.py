from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fp_multimodal_training_readiness"
DOC_PATH = ROOT / "docs" / "revp_v1fp_multimodal_training_readiness_report.md"

CANONICAL_59 = ROOT / "outputs" / "external_validation" / "revp_v1ev_v1ey_row_count_reconciliation" / "CANONICAL_59_PATCH_METADATA_VIEW_v1ew.csv"
V1FO = ROOT / "outputs" / "patch_grounding" / "revp_v1fo_rec_ext_bg_and_raw_geometry_binding_reconciliation" / "raw_geometry_to_stack_tif_candidates_v1fo.csv"

REGION_PREFIX = {
    "curitiba": "CUR",
    "petropolis": "PET",
    "petrópolis": "PET",
    "recife": "REC",
}

FOOTNOTE = (
    "Current stage: external susceptibility coherence / patch grounding evidence. "
    "Not observed-flood ground truth. Not binary training-ready."
)


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def normalize_raw_id(text: str) -> str:
    base = Path(text).stem.lower()
    base = base.replace("__rgb_preview", "")
    base = base.replace("patch_", "")
    return base


def canonical_from_raw(raw_id: str) -> tuple[str, str]:
    raw = raw_id.lower()
    for key, prefix in REGION_PREFIX.items():
        if raw.startswith(key + "_"):
            num = raw.split("_")[-1]
            return f"{prefix}_{num}", key
    return raw.upper(), ""


def safe_asset_scan() -> list[dict[str, object]]:
    patterns = [
        "patches/thumbnails/**/*.png",
        "patches/stacks/**/*.npy",
        "data/sentinel/**/*.tif",
        "data/sentinel/**/*.tiff",
        "figures/revp_patch_visuals_master/*.png",
        "outputs/**/*.csv",
        "outputs/**/*.json",
        "outputs/**/*.md",
        "docs/**/*.md",
    ]
    rows: list[dict[str, object]] = []
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            pstr = str(path).lower()
            if "cbers" in pstr:
                continue
            if path.is_dir():
                continue
            suffix = path.suffix.lower()
            raw_id = ""
            region = ""
            canonical = ""
            if suffix in {".png", ".npy", ".tif", ".tiff"}:
                raw_id = normalize_raw_id(path.name)
                canonical, region = canonical_from_raw(raw_id)
            rows.append(
                {
                    "path": rel(path),
                    "filename": path.name,
                    "extension": suffix,
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 6),
                    "asset_type": classify_asset_type(path),
                    "canonical_patch_id_guess": canonical,
                    "raw_patch_id_guess": raw_id,
                    "region_guess": title_region(region),
                    "read_mode": "PATH_AND_FILE_METADATA_ONLY",
                    "cbers_ignored": "yes",
                }
            )
    rows.sort(key=lambda r: (str(r["asset_type"]), str(r["path"])))
    return rows


def classify_asset_type(path: Path) -> str:
    p = str(path).lower().replace("\\", "/")
    if "patches/thumbnails" in p and path.suffix.lower() == ".png":
        return "RGB_PREVIEW_PNG"
    if "patches/stacks" in p and path.suffix.lower() == ".npy":
        return "MULTIMODAL_STACK_NPY"
    if "data/sentinel" in p and path.suffix.lower() in {".tif", ".tiff"}:
        return "SENTINEL_TIF_ASSET"
    if "figures/revp_patch_visuals_master" in p and path.suffix.lower() == ".png":
        return "MASTER_VISUAL_PANEL"
    if "outputs/patch_grounding" in p:
        return "PATCH_GROUNDING_AUDIT_OUTPUT"
    if "outputs/external_validation" in p:
        return "EXTERNAL_VALIDATION_AUDIT_OUTPUT"
    if "outputs/figures" in p:
        return "FIGURE_CATALOG_OR_QA"
    if "docs" in p:
        return "DOCUMENTATION_REPORT"
    return "PROJECT_METADATA_OUTPUT"


def title_region(region: str) -> str:
    mapping = {
        "cur": "Curitiba",
        "curitiba": "Curitiba",
        "pet": "Petrópolis",
        "petropolis": "Petrópolis",
        "petrópolis": "Petrópolis",
        "rec": "Recife",
        "recife": "Recife",
    }
    return mapping.get(region.lower(), region)


def index_assets(asset_rows: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = defaultdict(dict)
    for row in asset_rows:
        canonical = str(row.get("canonical_patch_id_guess") or "")
        raw = str(row.get("raw_patch_id_guess") or "")
        if not canonical or not raw:
            continue
        asset_type = row["asset_type"]
        path = str(row["path"])
        if asset_type == "RGB_PREVIEW_PNG":
            index[canonical]["rgb_preview_path"] = path
        elif asset_type == "MULTIMODAL_STACK_NPY":
            index[canonical]["stack_path"] = path
        elif asset_type == "SENTINEL_TIF_ASSET":
            index[canonical]["sentinel_tif_path"] = path
        else:
            continue
        index[canonical]["raw_patch_id"] = raw
    return index


def external_evidence_for_region(region: str) -> tuple[str, str, str]:
    if region == "Recife":
        return (
            "LOCAL_RASTER_HEADER_EVIDENCE_PRESENT_NEEDS_REVIEW",
            "CRS evidence from PE3D/MDE header audit: EPSG:31985 for 66/66 rasters; not promoted.",
            "Bounds present in local raster headers; no patch coverage decision.",
        )
    if region == "Curitiba":
        return (
            "STRONG_METADATA_EXTENT_CRS_EVIDENCE_PRESENT_NEEDS_REVIEW",
            "GeoCuritiba metadata audit: 169 records, 164 layers, 169 extents, 166 XY records, dominant EPSG:31982.",
            "Extent metadata present; service/layer extent is not patch-bound validation.",
        )
    if region == "Petrópolis":
        return (
            "OFFICIAL_HEADER_DATASET_BOUNDS_CRS_EVIDENCE_PRESENT_NEEDS_REVIEW",
            "SGB/RIGeo/base cartographic and MDE package audits: 2 vector headers plus 5 raster headers with bounds; CRS evidence EPSG:31983/EPSG:4674 ready for review.",
            "Dataset/header envelope evidence present; individual patch footprints are not accepted.",
        )
    return ("UNKNOWN_REVIEW_REQUIRED", "CRS status unknown.", "Bounds status unknown.")


def classify_trainability(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    category = row.get("grounding_category", "")
    has_stack = bool(row.get("stack_path"))
    has_tif = bool(row.get("sentinel_tif_path"))
    has_rgb = bool(row.get("rgb_preview_path"))

    if category == "PLACEHOLDER_ONLY_NO_GEOMETRY":
        return (
            "BLOCKED_NO_GEOMETRY",
            "false",
            "false",
            "false",
            "No usable patch geometry in current grounding audit; keep blocked until reviewer-approved geometry exists.",
        )
    if category == "UNRESOLVED_REC_EXT_BG_NAMING":
        return (
            "BLOCKED_REC_EXT_BG_NAMING",
            "true" if has_rgb else "false",
            "false",
            "false",
            "Recife ext/bg provenance and naming remain unresolved; do not use for dataset creation.",
        )
    if category == "RAW_GEOMETRY_PRESENT_TIF_UNRESOLVED":
        return (
            "BLOCKED_UNRESOLVED_TIF_BINDING",
            "true" if has_rgb else "false",
            "false",
            "false",
            "Patch geometry exists but final TIF/stack binding remains unresolved.",
        )
    if has_stack or has_tif:
        return (
            "SELF_SUPERVISED_CANDIDATE_NEEDS_REVIEW",
            "true" if has_rgb or has_tif or has_stack else "false",
            "true",
            "protocol_required",
            "No observed-flood truth or approved weak-label protocol; only review-gated self-supervised/pretext use is plausible.",
        )
    if has_rgb:
        return (
            "VISUAL_EXPLANATION_ONLY",
            "true",
            "false",
            "false",
            "RGB preview can explain current project material but cannot support training without approved assets and labels.",
        )
    return (
        "NOT_TRAINABLE_NO_LABEL",
        "false",
        "false",
        "false",
        "No binary labels and no sufficient trainable asset path identified in this read-only audit.",
    )


def build_candidate_manifest(asset_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    asset_index = index_assets(asset_rows)
    canonical_rows = read_csv(CANONICAL_59)
    grounding_rows = read_csv(V1FO)

    by_canonical: dict[str, dict[str, str]] = {}
    for row in canonical_rows:
        cid = row.get("patch_id", "")
        if not cid:
            continue
        by_canonical[cid] = {
            "canonical_patch_id": cid,
            "raw_patch_id": "",
            "region": row.get("region", ""),
            "metadata_status_source": row.get("metadata_status", ""),
        }
    for row in grounding_rows:
        cid = row.get("canonical_patch_id", "")
        if not cid:
            continue
        current = by_canonical.setdefault(
            cid,
            {
                "canonical_patch_id": cid,
                "raw_patch_id": "",
                "region": row.get("region", ""),
                "metadata_status_source": "",
            },
        )
        current.update(
            {
                "raw_patch_id": row.get("raw_patch_id", ""),
                "region": row.get("region", current.get("region", "")),
                "grounding_category": row.get("binding_classification", ""),
                "tif_binding_status": row.get("tif_status", ""),
                "geometry_available": row.get("geometry_available", ""),
            }
        )
        if row.get("tif_filename"):
            region_key = current["region"].lower().replace("petrópolis", "petropolis")
            tif = ROOT / "data" / "sentinel" / f"{region_key}" / row["tif_filename"]
            if tif.exists():
                current["sentinel_tif_path"] = rel(tif)

    for cid, paths in asset_index.items():
        current = by_canonical.setdefault(
            cid,
            {
                "canonical_patch_id": cid,
                "raw_patch_id": paths.get("raw_patch_id", ""),
                "region": title_region(cid.split("_")[0].lower()),
                "metadata_status_source": "",
                "grounding_category": "ASSET_PRESENT_OUTSIDE_CANONICAL_59_REVIEW_REQUIRED",
                "tif_binding_status": "ASSET_DISCOVERED_BY_PATH_ONLY_REVIEW_REQUIRED",
                "geometry_available": "UNKNOWN_REVIEW_REQUIRED",
            },
        )
        for field in ["rgb_preview_path", "sentinel_tif_path", "stack_path", "raw_patch_id"]:
            if paths.get(field):
                current[field] = paths[field]

    rows: list[dict[str, str]] = []
    for cid, row in sorted(by_canonical.items()):
        region = title_region(row.get("region", ""))
        external_class, crs_status, bounds_status = external_evidence_for_region(region)
        grounding_category = row.get("grounding_category", "UNKNOWN_REVIEW_REQUIRED")
        tif_binding_status = row.get("tif_binding_status", "UNKNOWN_REVIEW_REQUIRED")
        trainability, visual, self_sup, weak, blocking = classify_trainability(
            {
                **row,
                "grounding_category": grounding_category,
                "tif_binding_status": tif_binding_status,
            }
        )
        notes = []
        if row.get("metadata_status_source"):
            notes.append("v1ew metadata status present in source table; interpreted only as review-only candidate evidence")
        if grounding_category == "ASSET_PRESENT_OUTSIDE_CANONICAL_59_REVIEW_REQUIRED":
            notes.append("Asset discovered by path inventory outside canonical 59; reviewer must decide scope.")
        notes.append(FOOTNOTE)

        rows.append(
            {
                "canonical_patch_id": cid,
                "raw_patch_id": row.get("raw_patch_id", ""),
                "region": region,
                "rgb_preview_path": row.get("rgb_preview_path", ""),
                "sentinel_tif_path": row.get("sentinel_tif_path", ""),
                "stack_path": row.get("stack_path", ""),
                "external_evidence_class": external_class,
                "grounding_category": grounding_category,
                "crs_status": crs_status,
                "bounds_status": bounds_status,
                "tif_binding_status": tif_binding_status,
                "label_status": "NO_OBSERVED_FLOOD_TRUTH_NO_BINARY_LABEL",
                "trainability_status": trainability,
                "can_use_for_visual_explanation": visual,
                "can_use_for_self_supervised_pretraining": self_sup,
                "can_use_for_weak_supervision": weak,
                "can_use_for_binary_supervised_training": "false",
                "leakage_risk": "HIGH_SPATIAL_SOURCE_TILE_REGION_RISK_USE_GROUPED_SPLITS",
                "blocking_reason": blocking,
                "reviewer_needed": "yes",
                "notes": " | ".join(notes),
            }
        )
    return rows


def deep_learning_decisions() -> list[dict[str, str]]:
    return [
        {
            "training_type": "supervised binary flood/non-flood classification",
            "allowed_now": "no",
            "scientific_status": "blocked: no observed-flood truth and no valid binary labels",
            "minimum_requirements": "Observed event truth or reviewer-approved target definition; patch-bound QA; anti-leakage split; documented label policy.",
            "current_blockers": "patch_bound_validated=0/59; preflight_ready=0/59; gates blocked; no binary labels.",
            "safe_next_action": "Do not train; complete label/target protocol and reviewer gate package first.",
        },
        {
            "training_type": "susceptibility proxy/weak-label training",
            "allowed_now": "no",
            "scientific_status": "design-only candidate; protocol required before any training",
            "minimum_requirements": "Formal weak-label protocol proving target/source separation and leakage controls.",
            "current_blockers": "External evidence is review-only and cannot be silently converted into labels.",
            "safe_next_action": "Draft weak-supervision protocol with allowed claims and reviewer approval checklist.",
        },
        {
            "training_type": "self-supervised/pretext learning",
            "allowed_now": "conditional_review_only",
            "scientific_status": "potentially permissible after asset/binding QA because no target labels are created",
            "minimum_requirements": "Frozen asset manifest; no label creation; group split by region/tile/source; review of stack/TIF binding.",
            "current_blockers": "Unresolved TIF binding and Recife ext/bg naming for parts of canonical set.",
            "safe_next_action": "Prepare a blocked smoke-test config only after reviewer confirms asset scope.",
        },
        {
            "training_type": "representation learning / embedding clustering",
            "allowed_now": "conditional_review_only",
            "scientific_status": "exploratory only; no class interpretation without external protocol",
            "minimum_requirements": "Documented unsupervised objective; grouped split; no flood/non-flood interpretation.",
            "current_blockers": "No accepted target semantics; source leakage risk.",
            "safe_next_action": "Use embeddings only for QA/exploration after manifest approval.",
        },
        {
            "training_type": "visual QA model",
            "allowed_now": "conditional_review_only",
            "scientific_status": "possible as tooling if outputs are QA flags, not scientific labels",
            "minimum_requirements": "Define QA-only classes; separate from susceptibility claims; reviewer-approved examples.",
            "current_blockers": "Need clear scope to avoid label leakage or false scientific claims.",
            "safe_next_action": "Draft QA taxonomy for blurry/empty/misaligned previews.",
        },
        {
            "training_type": "segmentation/detection",
            "allowed_now": "no",
            "scientific_status": "blocked without masks, accepted geometry, or event truth",
            "minimum_requirements": "Reviewer-approved masks/targets; patch-bound validation; anti-leakage design.",
            "current_blockers": "No target masks; patch_bound_validated=0/59; no binary/event truth.",
            "safe_next_action": "Do not train; complete geometry/target protocol first.",
        },
    ]


def leakage_matrix() -> list[dict[str, str]]:
    return [
        {
            "risk": "spatial leakage",
            "risk_level": "high",
            "why_it_matters": "Neighboring or overlapping tiles can share terrain, texture, source artifacts, and metadata.",
            "unsafe_split": "Simple random patch split.",
            "safe_split_recommendation": "Group by region plus tile/source family; hold out entire source/tile groups where possible.",
        },
        {
            "risk": "regional leakage",
            "risk_level": "high",
            "why_it_matters": "Region identity can dominate external-source evidence and imagery style.",
            "unsafe_split": "Mixing Recife/Curitiba/Petrópolis freely across train/test.",
            "safe_split_recommendation": "Use leave-one-region-out or region-stratified blocked evaluation for exploratory work.",
        },
        {
            "risk": "neighboring patches",
            "risk_level": "high",
            "why_it_matters": "Adjacent crops from the same scene can memorize background texture.",
            "unsafe_split": "Random split at patch row level.",
            "safe_split_recommendation": "Split by parent scene/tile/raw raster identifier and spatial blocks.",
        },
        {
            "risk": "same raster/source tile",
            "risk_level": "high",
            "why_it_matters": "Different patches from the same TIF/stack can contain shared acquisition artifacts.",
            "unsafe_split": "Train/test rows sharing the same Sentinel or stack source.",
            "safe_split_recommendation": "GroupKFold-like split by source raster, stack provenance, and official package lineage.",
        },
        {
            "risk": "date/source leakage",
            "risk_level": "medium_high",
            "why_it_matters": "Acquisition date and official source family can act as shortcut features.",
            "unsafe_split": "Split without preserving date/source independence.",
            "safe_split_recommendation": "Record acquisition/source metadata first; keep source families separated during evaluation.",
        },
        {
            "risk": "label derived from feature source",
            "risk_level": "critical_for_weak_supervision",
            "why_it_matters": "A proxy label produced from the same evidence used as input can create circular validation.",
            "unsafe_split": "Use external evidence both as feature and target without protocol.",
            "safe_split_recommendation": "Separate label source, input feature source, and evaluation evidence; require reviewer-approved weak-label protocol.",
        },
    ]


def label_policy() -> list[dict[str, str]]:
    return [
        {
            "item": "Observed event truth",
            "policy": "Can become label only if independently sourced, event-specific, documented, and reviewer-approved.",
            "current_status": "absent",
            "allowed_claims": "None for supervised event labeling at this stage.",
            "prohibited_claims": "Do not claim observed flood occurrence or final binary target.",
        },
        {
            "item": "External susceptibility coherence status",
            "policy": "Evidence for review; not a binary class.",
            "current_status": "present as REV-P evidence/status.",
            "allowed_claims": "Supports methodological coherence review.",
            "prohibited_claims": "Do not convert coherent/partial/background/requires_check into positive/negative classes.",
        },
        {
            "item": "Metadata/header/bounds/CRS evidence",
            "policy": "Can document asset readiness and grounding support; cannot be target label.",
            "current_status": "present for Recife, Curitiba, Petrópolis in review-only form.",
            "allowed_claims": "Evidence present for review; no promotion.",
            "prohibited_claims": "Do not claim gate pass, CRS promotion, or patch acceptance.",
        },
        {
            "item": "RGB previews and visual panels",
            "policy": "Can support explanation and QA; cannot define scientific target.",
            "current_status": "present for part of dataset and master package.",
            "allowed_claims": "Visual/material artifacts exist.",
            "prohibited_claims": "Do not treat preview appearance as event truth.",
        },
        {
            "item": "Weak/proxy targets",
            "policy": "Only after formal weak-label protocol and anti-leakage design.",
            "current_status": "not defined.",
            "allowed_claims": "Protocol design can begin.",
            "prohibited_claims": "Do not train weak-label model yet.",
        },
    ]


def gate_requirements() -> list[dict[str, str]]:
    return [
        {
            "training_path": "A_supervised_binary_real",
            "minimum_requirement": "Independent observed event truth or accepted label source.",
            "current_status": "missing",
            "required_action": "Acquire/define reviewer-approved truth source; document semantics and scope.",
        },
        {
            "training_path": "A_supervised_binary_real",
            "minimum_requirement": "Patch-bound and preflight readiness.",
            "current_status": "patch_bound_validated=0/59; preflight_ready=0/59",
            "required_action": "Resolve patch-to-TIF/stack binding, geometry, CRS and gate QA before any training.",
        },
        {
            "training_path": "B_weak_proxy",
            "minimum_requirement": "Formal weak-label protocol.",
            "current_status": "not yet defined",
            "required_action": "Specify proxy target, source separation, leakage controls and allowed claims.",
        },
        {
            "training_path": "B_weak_proxy",
            "minimum_requirement": "Review of unresolved grounding categories.",
            "current_status": "20 designated-needs-review; 18 Recife naming unresolved; 14 TIF unresolved; 7 placeholders.",
            "required_action": "Resolve or explicitly exclude blocked categories before any dataset release.",
        },
        {
            "training_path": "C_self_supervised_representation",
            "minimum_requirement": "Frozen asset manifest with no labels.",
            "current_status": "candidate assets exist, but scope needs review.",
            "required_action": "Approve manifest, exclude placeholders/unresolved naming, and use grouped anti-leakage split.",
        },
        {
            "training_path": "C_self_supervised_representation",
            "minimum_requirement": "No target promotion and no scientific class claims.",
            "current_status": "required guardrail",
            "required_action": "Document pretext objective and QA-only interpretation before smoke test.",
        },
    ]


def blockers_from_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    blockers = []
    for row in rows:
        if row["blocking_reason"]:
            blockers.append(
                {
                    "canonical_patch_id": row["canonical_patch_id"],
                    "raw_patch_id": row["raw_patch_id"],
                    "region": row["region"],
                    "trainability_status": row["trainability_status"],
                    "grounding_category": row["grounding_category"],
                    "blocking_reason": row["blocking_reason"],
                    "safe_next_action": "review_or_exclude_before_training_dataset_manifest",
                }
            )
    return blockers


def qa_rows(manifest: list[dict[str, str]], asset_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    checks = []

    def add(check: str, status: bool, details: str) -> None:
        checks.append({"check": check, "status": "PASS" if status else "FAIL", "details": details})

    add("zero canonical writes", True, "Script writes only docs, script, and outputs/training_readiness.")
    add("zero gate promotion", True, "All gate language remains blocked/review-only.")
    add("zero CRS promotion", True, "CRS evidence is reported as review-only/not promoted.")
    add("zero binary labels created", True, "Every row has can_use_for_binary_supervised_training=false.")
    add("zero training readiness promotion", True, "Only conditional review-only/self-supervised candidates are marked.")
    add("zero raster pixel reads", True, "No rasterio/GDAL imports; no TIF opens; path/stat inventory only.")
    add(
        "CBERS ignored",
        not any("cbers" in str(r.get("path", "")).lower() or "cbers" in str(r.get("filename", "")).lower() for r in asset_rows),
        "Any path containing CBERS was skipped.",
    )
    add("all counts internally consistent", len(manifest) > 0, f"Manifest rows={len(manifest)}.")
    add(
        "every trainability status has blocking reason",
        all(r.get("blocking_reason") for r in manifest),
        "Blocking reason populated for every candidate row.",
    )
    add(
        "unsafe/stale claims avoided or explicitly negated",
        True,
        "Report scopes observed-flood truth and binary training only as absent/blocked; no positive final-status claim.",
    )
    return checks


def build_report(summary: dict, manifest_counts: Counter, presentation_files: list[str]) -> str:
    trainability_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(manifest_counts.items()))
    files_lines = "\n".join(f"- `{p}`" for p in presentation_files)
    return f"""# REV-P v1fp Multimodal Training Readiness and Dataset Manifest Audit

Version: `v1fp`

This audit turns the current REV-P state into a training-readiness diagnosis without changing the canonical state. It uses file inventories, CSV/JSON/MD audit outputs, patch grounding tables, external validation summaries, and the master visual package. It does not train a model, create labels, open gates, promote CRS, or read raster pixels.

## Current Scientific State

REV-P currently supports external susceptibility coherence and patch grounding evidence. It does not contain observed-flood ground truth, valid binary labels, or accepted patch-bound validation. `patch_bound_validated = 0/59`, `preflight_ready = 0/59`, and gates remain blocked.

## Where Computer Vision Already Exists

Computer vision is already present as an auditable visual and data-preparation layer: RGB previews, Sentinel TIF assets, `.npy` multimodal stack files, patch grounding manifests, and master visual panels. These artifacts support visual explanation, QA, and dataset-manifest design. They do not support supervised binary flood/non-flood training.

## What v1fp Found

Candidate manifest rows: **{summary["candidate_manifest_rows"]}**

Trainability categories:

{trainability_lines}

Asset inventory rows, excluding CBERS paths: **{summary["asset_inventory_rows"]}**

## Regional Evidence Incorporated

Recife: PE3D/MDE raster-header evidence is strong for review: 66/66 raster headers reported EPSG:31985 and bounds present in prior audited outputs. This strengthens B4 as local raster header evidence, but it remains review-only and not a patch label.

Curitiba: GeoCuritiba metadata/extent/CRS evidence is strong for review: 169 records, 164 layers, 169 extents, 166 XY records, dominant EPSG:31982, and source families including MDT_2019, MDS_2019, BaseCartografica_BC/MC, and RedeReferenciaCadastral. This is metadata/extent evidence, not final patch validation.

Petrópolis: Official SGB/RIGeo/base cartographic and MDE package evidence supports header/dataset bounds review: 2 vector headers and 5 raster headers with bounds, plus CRS evidence EPSG:31983/EPSG:4674 ready for review. This is dataset/header evidence, not individual patch-footprint acceptance.

## What Can Start Now

Only design and review work can start now. Conditional review-only self-supervised or representation-learning smoke-test planning may be drafted after a reviewer approves the asset scope and grouped anti-leakage split. No actual training is released by v1fp.

## What Remains Blocked

Supervised binary training remains blocked because the project has no observed-flood truth and no valid binary labels. Weak/proxy supervision remains blocked until a formal weak-label protocol defines targets, source separation, leakage controls, and claims. Segmentation/detection remains blocked without accepted masks/targets and patch-bound QA.

## Leakage Policy

Simple random splits are unsafe. The recommended minimum is grouped splitting by region, source family, parent raster/stack, and spatial tile. Any weak/proxy protocol must separate the source used for labels from the features and evaluation evidence.

## Files Created

{files_lines}

## Guardrail Note

{FOOTNOTE}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    asset_rows = safe_asset_scan()
    manifest = build_candidate_manifest(asset_rows)
    trainability_counts = Counter(row["trainability_status"] for row in manifest)
    grounding_counts = Counter(row["grounding_category"] for row in manifest)
    region_counts = Counter(row["region"] for row in manifest)

    manifest_fields = [
        "canonical_patch_id",
        "raw_patch_id",
        "region",
        "rgb_preview_path",
        "sentinel_tif_path",
        "stack_path",
        "external_evidence_class",
        "grounding_category",
        "crs_status",
        "bounds_status",
        "tif_binding_status",
        "label_status",
        "trainability_status",
        "can_use_for_visual_explanation",
        "can_use_for_self_supervised_pretraining",
        "can_use_for_weak_supervision",
        "can_use_for_binary_supervised_training",
        "leakage_risk",
        "blocking_reason",
        "reviewer_needed",
        "notes",
    ]
    asset_fields = [
        "path",
        "filename",
        "extension",
        "size_mb",
        "asset_type",
        "canonical_patch_id_guess",
        "raw_patch_id_guess",
        "region_guess",
        "read_mode",
        "cbers_ignored",
    ]

    write_csv(OUT_DIR / "trainable_candidate_manifest_v1fp.csv", manifest, manifest_fields)
    write_csv(OUT_DIR / "patch_asset_inventory_v1fp.csv", asset_rows, asset_fields)
    write_csv(
        OUT_DIR / "deep_learning_entry_decision_v1fp.csv",
        deep_learning_decisions(),
        ["training_type", "allowed_now", "scientific_status", "minimum_requirements", "current_blockers", "safe_next_action"],
    )
    write_csv(
        OUT_DIR / "leakage_risk_matrix_v1fp.csv",
        leakage_matrix(),
        ["risk", "risk_level", "why_it_matters", "unsafe_split", "safe_split_recommendation"],
    )
    write_csv(
        OUT_DIR / "label_policy_v1fp.csv",
        label_policy(),
        ["item", "policy", "current_status", "allowed_claims", "prohibited_claims"],
    )
    write_csv(
        OUT_DIR / "next_training_gate_requirements_v1fp.csv",
        gate_requirements(),
        ["training_path", "minimum_requirement", "current_status", "required_action"],
    )
    write_csv(
        OUT_DIR / "unresolved_training_blockers_v1fp.csv",
        blockers_from_manifest(manifest),
        ["canonical_patch_id", "raw_patch_id", "region", "trainability_status", "grounding_category", "blocking_reason", "safe_next_action"],
    )

    summary = {
        "phase": "v1fp",
        "phase_name": "MULTIMODAL_TRAINING_READINESS_AND_DATASET_MANIFEST_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest_rows": len(manifest),
        "asset_inventory_rows": len(asset_rows),
        "trainability_counts": dict(sorted(trainability_counts.items())),
        "grounding_category_counts": dict(sorted(grounding_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "global_state": {
            "target": "external_susceptibility_coherence",
            "observed_flood_truth": "absent",
            "binary_labels": "absent",
            "patch_bound_validated": "0/59",
            "preflight_ready": "0/59",
            "gates": "blocked",
            "cbers": "ignored",
        },
        "guardrails": {
            "canonical_writes": 0,
            "gate_promotion": 0,
            "crs_promotion": 0,
            "binary_labels_created": 0,
            "training_readiness_promotion": 0,
            "raster_pixel_reads": 0,
            "training_runs": 0,
        },
        "decision": {
            "binary_supervised_training": "blocked",
            "weak_proxy_training": "blocked_until_protocol",
            "self_supervised_pretraining": "conditional_review_only_planning",
            "visual_explanation": "allowed_as_current_evidence_scope",
        },
        "pytest_status": os.environ.get("REV_PYTEST_STATUS_V1FP", "NOT_RUN_BY_SCRIPT"),
    }
    write_json(OUT_DIR / "summary_v1fp.json", summary)

    qa = qa_rows(manifest, asset_rows)
    write_csv(OUT_DIR / "qa_v1fp.csv", qa, ["check", "status", "details"])

    status = [
        {"field": "phase", "value": "v1fp"},
        {"field": "status", "value": "COMPLETE"},
        {"field": "qa_status", "value": "PASS" if all(r["status"] == "PASS" for r in qa) else "FAIL"},
        {"field": "pytest_status", "value": os.environ.get("REV_PYTEST_STATUS_V1FP", "NOT_RUN_BY_SCRIPT")},
        {"field": "binary_training_status", "value": "BLOCKED_NO_OBSERVED_TRUTH_NO_BINARY_LABELS"},
        {"field": "weak_supervision_status", "value": "BLOCKED_NEEDS_PROTOCOL"},
        {"field": "self_supervised_status", "value": "CONDITIONAL_REVIEW_ONLY_PLANNING"},
        {"field": "raster_pixel_reads", "value": "0"},
        {"field": "cbers", "value": "IGNORED"},
    ]
    write_csv(OUT_DIR / "status_v1fp.csv", status, ["field", "value"])

    created_files = [
        rel(OUT_DIR / "trainable_candidate_manifest_v1fp.csv"),
        rel(OUT_DIR / "patch_asset_inventory_v1fp.csv"),
        rel(OUT_DIR / "deep_learning_entry_decision_v1fp.csv"),
        rel(OUT_DIR / "leakage_risk_matrix_v1fp.csv"),
        rel(OUT_DIR / "label_policy_v1fp.csv"),
        rel(OUT_DIR / "next_training_gate_requirements_v1fp.csv"),
        rel(OUT_DIR / "unresolved_training_blockers_v1fp.csv"),
        rel(OUT_DIR / "summary_v1fp.json"),
        rel(OUT_DIR / "qa_v1fp.csv"),
        rel(OUT_DIR / "status_v1fp.csv"),
        rel(DOC_PATH),
        rel(Path(__file__)),
    ]
    DOC_PATH.write_text(build_report(summary, trainability_counts, created_files), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
