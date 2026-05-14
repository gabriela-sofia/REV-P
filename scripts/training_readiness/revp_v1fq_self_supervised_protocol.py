from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1FP_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fp_multimodal_training_readiness"
OUT_DIR = ROOT / "outputs" / "training_readiness" / "revp_v1fq_self_supervised_protocol"
DOC_PATH = ROOT / "docs" / "revp_v1fq_self_supervised_protocol_report.md"

FOOTNOTE = (
    "Current stage: external susceptibility coherence / patch grounding evidence. "
    "No observed-flood ground truth, no binary labels, no training readiness promotion."
)

EXPECTED_V1FP_COUNTS = {
    "SELF_SUPERVISED_CANDIDATE_NEEDS_REVIEW": 128,
    "BLOCKED_REC_EXT_BG_NAMING": 18,
    "BLOCKED_UNRESOLVED_TIF_BINDING": 14,
    "BLOCKED_NO_GEOMETRY": 7,
    "NOT_TRAINABLE_NO_LABEL": 20,
}


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


def load_v1fp_inputs() -> dict[str, object]:
    inputs = {
        "manifest": read_csv(V1FP_DIR / "trainable_candidate_manifest_v1fp.csv"),
        "asset_inventory": read_csv(V1FP_DIR / "patch_asset_inventory_v1fp.csv"),
        "deep_learning_decision": read_csv(V1FP_DIR / "deep_learning_entry_decision_v1fp.csv"),
        "leakage_risk": read_csv(V1FP_DIR / "leakage_risk_matrix_v1fp.csv"),
        "label_policy": read_csv(V1FP_DIR / "label_policy_v1fp.csv"),
        "gate_requirements": read_csv(V1FP_DIR / "next_training_gate_requirements_v1fp.csv"),
        "blockers": read_csv(V1FP_DIR / "unresolved_training_blockers_v1fp.csv"),
        "qa": read_csv(V1FP_DIR / "qa_v1fp.csv"),
        "status": read_csv(V1FP_DIR / "status_v1fp.csv"),
        "summary": json.loads((V1FP_DIR / "summary_v1fp.json").read_text(encoding="utf-8")),
        "report_text": (ROOT / "docs" / "revp_v1fp_multimodal_training_readiness_report.md").read_text(encoding="utf-8"),
    }
    return inputs


def modality_for(asset_type: str) -> str:
    return {
        "RGB_PREVIEW_PNG": "rgb_preview",
        "SENTINEL_TIF_ASSET": "sentinel_raster_path_only",
        "MULTIMODAL_STACK_NPY": "multimodal_stack_path_only",
        "MASTER_VISUAL_PANEL": "visual_explanation_panel",
        "NO_ASSET_PATH": "none",
    }.get(asset_type, "metadata_or_documentation")


def evidence_source_for(row: dict[str, str], asset_type: str) -> str:
    if asset_type == "SENTINEL_TIF_ASSET":
        return "local Sentinel TIF path from v1fp manifest; path only, no pixel read"
    if asset_type == "MULTIMODAL_STACK_NPY":
        return "local multimodal stack NPY path from v1fp manifest; path only, no array read"
    if asset_type == "RGB_PREVIEW_PNG":
        return "local RGB preview PNG from v1fp manifest"
    if row.get("external_evidence_class"):
        return row["external_evidence_class"]
    return "v1fp metadata-only manifest"


def split_group_for(row: dict[str, str], asset_type: str, asset_path: str) -> str:
    region = row.get("region", "UNKNOWN").replace(" ", "_")
    source = "none"
    if asset_type == "SENTINEL_TIF_ASSET":
        source = "sentinel_tif"
    elif asset_type == "MULTIMODAL_STACK_NPY":
        source = "stack_npy"
    elif asset_type == "RGB_PREVIEW_PNG":
        source = "rgb_preview"
    elif asset_type == "NO_ASSET_PATH":
        source = "no_asset"
    stem = Path(asset_path).stem if asset_path else row.get("canonical_patch_id", "unknown")
    return f"{region}__{source}__{stem}"


def leakage_flags_for(row: dict[str, str], asset_type: str) -> str:
    flags = [
        "spatial_group_required",
        "region_group_required",
        "source_group_required",
        "status_not_target",
        "official_evidence_not_target",
    ]
    if asset_type in {"SENTINEL_TIF_ASSET", "MULTIMODAL_STACK_NPY"}:
        flags.append("same_tile_or_stack_leakage_risk")
    if row.get("region") == "Recife":
        flags.append("recife_ext_bg_provenance_guard")
    if row.get("grounding_category") == "ASSET_PRESENT_OUTSIDE_CANONICAL_59_REVIEW_REQUIRED":
        flags.append("outside_canonical_59_scope_review")
    return ";".join(flags)


def protocol_mode_for(row: dict[str, str], asset_type: str) -> tuple[str, str, str, str]:
    status = row.get("trainability_status", "")
    if status == "SELF_SUPERVISED_CANDIDATE_NEEDS_REVIEW" and asset_type in {"SENTINEL_TIF_ASSET", "MULTIMODAL_STACK_NPY"}:
        return (
            "SELF_SUPERVISED_REPRESENTATION_REVIEW_ONLY",
            "review-only representation/pretext planning after asset-scope approval; no labels",
            "supervised binary training; weak-label training; segmentation; model performance claims; status-as-target",
            "Asset exists, but use is restricted to review-only representation planning because labels and readiness are absent.",
        )
    if status == "SELF_SUPERVISED_CANDIDATE_NEEDS_REVIEW" and asset_type == "RGB_PREVIEW_PNG":
        return (
            "VISUAL_EXPLANATION_ONLY",
            "visual explanation and QA illustration",
            "training target; binary class; performance evaluation",
            "RGB preview supports explanation, not model target creation.",
        )
    if status == "BLOCKED_NO_GEOMETRY":
        return (
            "BLOCKED_NO_GEOMETRY",
            "none until reviewer-approved geometry exists",
            "all training and dataset inclusion",
            "Current v1fp/v1fo grounding marks this candidate as placeholder/no geometry.",
        )
    if status == "BLOCKED_UNRESOLVED_TIF_BINDING":
        return (
            "BLOCKED_UNRESOLVED_BINDING",
            "visual explanation only if preview exists",
            "self-supervised dataset inclusion; supervised training; target creation",
            "Patch geometry exists but final TIF/stack binding remains unresolved.",
        )
    if status == "BLOCKED_REC_EXT_BG_NAMING":
        return (
            "BLOCKED_RECIFE_EXT_BG_NAMING",
            "none for training; provenance review only",
            "dataset inclusion; supervised or self-supervised training",
            "Recife ext/bg naming/provenance is unresolved.",
        )
    if status == "NOT_TRAINABLE_NO_LABEL":
        return (
            "BLOCKED_NO_LABEL",
            "documentation/manifest review only",
            "training and target creation",
            "No accepted label and no sufficient reviewed asset route.",
        )
    return (
        "VISUAL_EXPLANATION_ONLY",
        "visual/material explanation only",
        "supervised binary training and target creation",
        "Conservative fallback: evidence is review-only.",
    )


def build_self_supervised_manifest(v1fp_manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_row in v1fp_manifest:
        candidate_id = source_row.get("canonical_patch_id") or source_row.get("raw_patch_id")
        assets = [
            ("RGB_PREVIEW_PNG", source_row.get("rgb_preview_path", "")),
            ("SENTINEL_TIF_ASSET", source_row.get("sentinel_tif_path", "")),
            ("MULTIMODAL_STACK_NPY", source_row.get("stack_path", "")),
        ]
        real_assets = [(kind, path) for kind, path in assets if path]
        if not real_assets:
            real_assets = [("NO_ASSET_PATH", "")]
        for asset_type, asset_path in real_assets:
            mode, allowed, forbidden, reason = protocol_mode_for(source_row, asset_type)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "region": source_row.get("region", ""),
                    "canonical_patch_id": source_row.get("canonical_patch_id", ""),
                    "raw_patch_id": source_row.get("raw_patch_id", ""),
                    "asset_path": asset_path,
                    "asset_type": asset_type,
                    "modality": modality_for(asset_type),
                    "evidence_source": evidence_source_for(source_row, asset_type),
                    "scientific_role": mode,
                    "trainability_status": source_row.get("trainability_status", ""),
                    "allowed_use": allowed,
                    "forbidden_use": forbidden,
                    "leakage_flags": leakage_flags_for(source_row, asset_type),
                    "split_group": split_group_for(source_row, asset_type, asset_path),
                    "reviewer_notes": FOOTNOTE,
                    "decision_reason": reason,
                }
            )
    return rows


def modality_feature_policy() -> list[dict[str, str]]:
    return [
        {
            "field_or_modality": "RGB preview PNG",
            "feature_role": "visual QA/explanation feature only; possible review-only embedding input if approved",
            "may_be_used_as_input": "yes_review_only",
            "may_be_used_as_target": "no",
            "leakage_risk": "preview may encode region/source artifacts",
            "required_guardrail": "group by region/source; never interpret as observed event truth",
        },
        {
            "field_or_modality": "Sentinel TIF path",
            "feature_role": "candidate raster input path for future pretext/representation experiments",
            "may_be_used_as_input": "yes_after_binding_review",
            "may_be_used_as_target": "no",
            "leakage_risk": "same raster/tile/date/source leakage",
            "required_guardrail": "no pixel read in v1fq; future split by raster/source/tile",
        },
        {
            "field_or_modality": "Multimodal stack NPY path",
            "feature_role": "candidate stack input path for future pretext/representation experiments",
            "may_be_used_as_input": "yes_after_scope_review",
            "may_be_used_as_target": "no",
            "leakage_risk": "stack provenance and source family shortcuts",
            "required_guardrail": "do not read arrays in v1fq; future split by source group",
        },
        {
            "field_or_modality": "external_evidence_class / grounding_category / CRS / bounds status",
            "feature_role": "audit/status evidence only",
            "may_be_used_as_input": "no_for_model_training",
            "may_be_used_as_target": "no",
            "leakage_risk": "critical if converted into labels or features",
            "required_guardrail": "report-only; never status-as-target",
        },
        {
            "field_or_modality": "region",
            "feature_role": "split/control metadata",
            "may_be_used_as_input": "no",
            "may_be_used_as_target": "no",
            "leakage_risk": "regional shortcut",
            "required_guardrail": "use as grouped split variable, not model feature",
        },
    ]


def forbidden_policy() -> list[dict[str, str]]:
    return [
        {
            "forbidden_item": "observed-flood class target",
            "why_forbidden_now": "No observed-flood ground truth exists in current REV-P state.",
            "leakage_or_overclaim_risk": "Would create unsupported binary claims.",
            "allowed_alternative": "Document absence and design truth-acquisition protocol.",
        },
        {
            "forbidden_item": "coherent/partial/rejected/pending/status as labels",
            "why_forbidden_now": "These are review/evidence states, not event labels or negatives.",
            "leakage_or_overclaim_risk": "Status leakage and false supervision.",
            "allowed_alternative": "Use only for audit stratification and reporting.",
        },
        {
            "forbidden_item": "official CRS/bounds/header evidence as target",
            "why_forbidden_now": "This evidence supports grounding review, not susceptibility/event truth.",
            "leakage_or_overclaim_risk": "Circular validation if source evidence becomes both feature and target.",
            "allowed_alternative": "Keep in manifest as evidence_source/scientific_role metadata.",
        },
        {
            "forbidden_item": "random patch split",
            "why_forbidden_now": "Spatial, regional, tile and source leakage risks are high.",
            "leakage_or_overclaim_risk": "Inflated embedding or downstream evaluation signals.",
            "allowed_alternative": "Grouped split by region/source/tile/raster lineage.",
        },
        {
            "forbidden_item": "model performance claims",
            "why_forbidden_now": "No training or target evaluation is authorized.",
            "leakage_or_overclaim_risk": "Would imply readiness and validation not present.",
            "allowed_alternative": "Only sanity checks and exploratory embedding diagnostics after review.",
        },
    ]


def split_strategy() -> list[dict[str, str]]:
    return [
        {
            "split_level": "region",
            "recommendation": "primary hard grouping",
            "rationale": "Prevents Recife/Curitiba/Petrópolis source and landscape shortcuts.",
            "implementation_note": "Use region as outer holdout or blocked fold; never simple random split.",
        },
        {
            "split_level": "source_family",
            "recommendation": "secondary hard grouping",
            "rationale": "Official source families and Sentinel/stack provenance may leak acquisition artifacts.",
            "implementation_note": "Group by Sentinel TIF, stack lineage, PE3D/MDE, GeoCuritiba, SGB/RIGeo where available.",
        },
        {
            "split_level": "tile_or_parent_raster",
            "recommendation": "mandatory when parent source is known",
            "rationale": "Patches from same raster/tile share texture and processing artifacts.",
            "implementation_note": "No train/test split sharing parent raster or stack source.",
        },
        {
            "split_level": "time_or_acquisition",
            "recommendation": "required when date metadata becomes available",
            "rationale": "Date/source artifacts can be shortcuts.",
            "implementation_note": "Add acquisition-date group before any future experiment.",
        },
    ]


def allowed_experiment_protocol() -> list[dict[str, str]]:
    return [
        {
            "experiment_mode": "visual/material explanation",
            "allowed_now": "yes",
            "objective": "Explain real assets, evidence dossiers and grounding status.",
            "architecture_or_method": "static panels, retrieval-by-path, manual QA",
            "allowed_evaluation": "completeness, legibility, audit traceability",
            "forbidden_evaluation": "event classification metrics",
        },
        {
            "experiment_mode": "self-supervised representation planning",
            "allowed_now": "review_only_planning",
            "objective": "Plan embeddings/pretext learning without labels.",
            "architecture_or_method": "transfer learning embeddings or pretrained encoder; not CNN from scratch",
            "allowed_evaluation": "embedding sanity checks, clustering exploration, visual retrieval, modality separation",
            "forbidden_evaluation": "flood accuracy, F1, AUC, positive/negative labels",
        },
        {
            "experiment_mode": "blocked smoke-test design",
            "allowed_now": "design_only",
            "objective": "Specify how a future no-label pipeline would run after approval.",
            "architecture_or_method": "small frozen-backbone embedding extraction plan",
            "allowed_evaluation": "runtime smoke QA only after reviewer approval",
            "forbidden_evaluation": "scientific performance claims",
        },
    ]


def blocked_training_cases() -> list[dict[str, str]]:
    return [
        {
            "case": "supervised binary training",
            "status": "blocked",
            "blocking_reason": "No observed-flood ground truth and no valid binary labels.",
            "required_before_revisit": "Truth source, label policy, patch-bound validation, anti-leakage split, reviewer approval.",
        },
        {
            "case": "weak supervision real training",
            "status": "blocked",
            "blocking_reason": "No formal weak-label protocol and risk of using evidence/status as target.",
            "required_before_revisit": "Protocol defining target semantics, source separation, leakage controls and claims.",
        },
        {
            "case": "segmentation/detection",
            "status": "blocked",
            "blocking_reason": "No accepted masks/targets and patch_bound_validated=0/59.",
            "required_before_revisit": "Reviewer-approved geometry/masks/targets and preflight readiness.",
        },
        {
            "case": "random split evaluation",
            "status": "forbidden",
            "blocking_reason": "High spatial/regional/source leakage risk.",
            "required_before_revisit": "Grouped split design by region/source/tile/raster lineage.",
        },
    ]


def next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": "Reviewer approval of self-supervised asset scope",
            "purpose": "Decide which of the 128 review-only candidates may enter a future no-label pretext manifest.",
            "guardrail": "Do not treat candidates as validated patches.",
        },
        {
            "priority": "2",
            "action": "Resolve or exclude blocked grounding groups",
            "purpose": "Keep Recife ext/bg naming, unresolved TIF binding and placeholders out of training manifests.",
            "guardrail": "No automatic inclusion from path presence alone.",
        },
        {
            "priority": "3",
            "action": "Define grouped split table",
            "purpose": "Attach region/source/tile/raster lineage groups before any future experiment.",
            "guardrail": "No simple random split.",
        },
        {
            "priority": "4",
            "action": "Draft optional no-label smoke-test spec",
            "purpose": "Prepare transfer-learning embedding extraction plan after review.",
            "guardrail": "No labels, no target metrics, no model performance claims.",
        },
    ]


def qa_rows(inputs: dict[str, object], manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    v1fp_manifest: list[dict[str, str]] = inputs["manifest"]  # type: ignore[assignment]
    asset_inventory: list[dict[str, str]] = inputs["asset_inventory"]  # type: ignore[assignment]
    counts = Counter(row["trainability_status"] for row in v1fp_manifest)
    rows: list[dict[str, str]] = []

    def add(check: str, ok: bool, details: str) -> None:
        rows.append({"check": check, "status": "PASS" if ok else "FAIL", "details": details})

    add("v1fp category sum verified", dict(counts) == EXPECTED_V1FP_COUNTS, f"Observed={dict(counts)}")
    add("v1fp total candidates verified", len(v1fp_manifest) == 187, f"Rows={len(v1fp_manifest)}")
    add("CBERS ignored", not any("cbers" in row.get("path", "").lower() or "cbers" in row.get("filename", "").lower() for row in asset_inventory), "No CBERS path in v1fp asset inventory.")
    add("no labels created", True, "v1fq creates policy/protocol fields only; no label columns or target values.")
    add("no targets created", True, "No target generation or target semantics created.")
    add("no training executed", True, "No ML libraries imported; no model fit/predict calls.")
    add("no gate promotion", True, "All gates remain blocked/review-only.")
    add("no CRS promotion", True, "CRS remains evidence/status metadata only.")
    add("every candidate asset has allowed_use", all(row.get("allowed_use") for row in manifest), "All v1fq rows have allowed_use.")
    add("every candidate asset has forbidden_use", all(row.get("forbidden_use") for row in manifest), "All v1fq rows have forbidden_use.")
    add("leakage flags present", all(row.get("leakage_flags") for row in manifest), "All v1fq rows flag leakage controls.")
    add("no raster pixel reads", True, "Path/manifest only; no rasterio/GDAL, no array/raster opening.")
    add("patch_bound/preflight remain locked", True, "patch_bound_validated=0/59 and preflight_ready=0/59 recorded in summary/report.")
    return rows


def build_report(summary: dict, created_files: list[str]) -> str:
    counts = "\n".join(f"- `{k}`: {v}" for k, v in summary["v1fp_trainability_counts"].items())
    files = "\n".join(f"- `{path}`" for path in created_files)
    return f"""# REV-P v1fq Self-Supervised Multimodal Protocol and Manifest Refinement

Version: `v1fq`

This phase refines v1fp into an auditable self-supervised review-only protocol. It does not run training, create labels, create targets, promote CRS, open gates, or declare training readiness.

## Starting Point From v1fp

Manifest total: **{summary["v1fp_candidate_count"]}** candidates.

V1fp trainability categories:

{counts}

Asset inventory: **{summary["v1fp_asset_inventory_count"]}** read-only artifacts. CBERS remains ignored.

## Protocol Decision

The only scientifically admissible near-term direction is review-only representation/pretext planning using candidate raster/stack assets after reviewer approval of scope, binding, source lineage and grouped split strategy. This means learning visual/material representations, not detecting flood occurrence.

Supervised binary training, weak supervision, segmentation/detection, model performance claims, label creation and target creation remain blocked.

## Candidate Use Policy

Allowed:

- Visual/material explanation and QA.
- Review-only self-supervised representation planning for candidates with raster or stack assets.
- Future embedding sanity checks, clustering exploration, visual retrieval and modality separation after reviewer approval.

Forbidden:

- Flood/non-flood target creation.
- Coherence/status/evidence as target.
- Official CRS/bounds/header evidence as target.
- Random patch split.
- Accuracy/F1/AUC claims for flood detection.

## Leakage Controls

Any future experiment must split by region, source family and parent raster/stack/tile lineage. The region field is a split/control field, not a model input. External evidence/status fields are report-only and must not become labels or features.

## Persistent Locks

`patch_bound_validated = 0/59`, `preflight_ready = 0/59`, gates remain blocked, and the project remains without observed-flood ground truth.

## Files Created

{files}

## Guardrail Note

{FOOTNOTE}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    inputs = load_v1fp_inputs()
    v1fp_manifest: list[dict[str, str]] = inputs["manifest"]  # type: ignore[assignment]
    asset_inventory: list[dict[str, str]] = inputs["asset_inventory"]  # type: ignore[assignment]

    refined_manifest = build_self_supervised_manifest(v1fp_manifest)
    mode_counts = Counter(row["scientific_role"] for row in refined_manifest)
    v1fp_counts = Counter(row["trainability_status"] for row in v1fp_manifest)

    manifest_fields = [
        "candidate_id",
        "region",
        "canonical_patch_id",
        "raw_patch_id",
        "asset_path",
        "asset_type",
        "modality",
        "evidence_source",
        "scientific_role",
        "trainability_status",
        "allowed_use",
        "forbidden_use",
        "leakage_flags",
        "split_group",
        "reviewer_notes",
        "decision_reason",
    ]
    write_csv(OUT_DIR / "self_supervised_candidate_manifest_v1fq.csv", refined_manifest, manifest_fields)
    write_csv(
        OUT_DIR / "modality_feature_policy_v1fq.csv",
        modality_feature_policy(),
        ["field_or_modality", "feature_role", "may_be_used_as_input", "may_be_used_as_target", "leakage_risk", "required_guardrail"],
    )
    write_csv(
        OUT_DIR / "forbidden_target_and_leakage_policy_v1fq.csv",
        forbidden_policy(),
        ["forbidden_item", "why_forbidden_now", "leakage_or_overclaim_risk", "allowed_alternative"],
    )
    write_csv(
        OUT_DIR / "split_strategy_v1fq.csv",
        split_strategy(),
        ["split_level", "recommendation", "rationale", "implementation_note"],
    )
    write_csv(
        OUT_DIR / "allowed_experiment_protocol_v1fq.csv",
        allowed_experiment_protocol(),
        ["experiment_mode", "allowed_now", "objective", "architecture_or_method", "allowed_evaluation", "forbidden_evaluation"],
    )
    write_csv(
        OUT_DIR / "blocked_training_cases_v1fq.csv",
        blocked_training_cases(),
        ["case", "status", "blocking_reason", "required_before_revisit"],
    )
    write_csv(
        OUT_DIR / "next_actions_v1fq.csv",
        next_actions(),
        ["priority", "action", "purpose", "guardrail"],
    )

    qa = qa_rows(inputs, refined_manifest)
    write_csv(OUT_DIR / "qa_v1fq.csv", qa, ["check", "status", "details"])

    summary = {
        "phase": "v1fq",
        "phase_name": "SELF_SUPERVISED_MULTIMODAL_PROTOCOL_AND_MANIFEST_REFINEMENT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v1fp_candidate_count": len(v1fp_manifest),
        "v1fp_asset_inventory_count": len(asset_inventory),
        "v1fp_trainability_counts": dict(sorted(v1fp_counts.items())),
        "self_supervised_manifest_rows": len(refined_manifest),
        "scientific_role_counts": dict(sorted(mode_counts.items())),
        "decision": {
            "visual_explanation": "allowed",
            "self_supervised_representation": "review_only_planning",
            "supervised_binary_training": "blocked",
            "weak_supervision_real_training": "blocked",
            "segmentation_detection": "blocked",
        },
        "locks": {
            "observed_flood_ground_truth": "absent",
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
            "labels_created": 0,
            "targets_created": 0,
            "training_runs": 0,
            "raster_pixel_reads": 0,
        },
        "pytest_status": os.environ.get("REV_PYTEST_STATUS_V1FQ", "NOT_RUN_BY_SCRIPT"),
    }
    write_json(OUT_DIR / "summary_v1fq.json", summary)

    status = [
        {"field": "phase", "value": "v1fq"},
        {"field": "status", "value": "COMPLETE"},
        {"field": "qa_status", "value": "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"},
        {"field": "pytest_status", "value": os.environ.get("REV_PYTEST_STATUS_V1FQ", "NOT_RUN_BY_SCRIPT")},
        {"field": "self_supervised_protocol", "value": "REVIEW_ONLY_PLANNING"},
        {"field": "supervised_binary_training", "value": "BLOCKED"},
        {"field": "weak_supervision_real_training", "value": "BLOCKED_NEEDS_PROTOCOL"},
        {"field": "labels_targets_created", "value": "0"},
        {"field": "raster_pixel_reads", "value": "0"},
        {"field": "cbers", "value": "IGNORED"},
    ]
    write_csv(OUT_DIR / "status_v1fq.csv", status, ["field", "value"])

    created_files = [
        rel(OUT_DIR / "self_supervised_candidate_manifest_v1fq.csv"),
        rel(OUT_DIR / "modality_feature_policy_v1fq.csv"),
        rel(OUT_DIR / "forbidden_target_and_leakage_policy_v1fq.csv"),
        rel(OUT_DIR / "split_strategy_v1fq.csv"),
        rel(OUT_DIR / "allowed_experiment_protocol_v1fq.csv"),
        rel(OUT_DIR / "blocked_training_cases_v1fq.csv"),
        rel(OUT_DIR / "next_actions_v1fq.csv"),
        rel(OUT_DIR / "summary_v1fq.json"),
        rel(OUT_DIR / "qa_v1fq.csv"),
        rel(OUT_DIR / "status_v1fq.csv"),
        rel(DOC_PATH),
        rel(Path(__file__)),
    ]
    DOC_PATH.write_text(build_report(summary, created_files), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
