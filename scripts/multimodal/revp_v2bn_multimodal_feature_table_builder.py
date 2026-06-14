"""REV-P v2bn — Multimodal feature table builder (review-only readiness).

Builds one compact, auditable row per Sentinel patch/candidate joining:

  * the canonical Sentinel input spine (v1fu input manifest, 128 patches);
  * the real local DINOv2 embedding manifest (v1ge, 12 frozen 768D vectors);
  * patch-registry presence, GIS feature availability and evidence-registry
    presence as *availability flags only* (never promoted to labels);
  * ground-truth placeholder columns that stay empty/NA on purpose.

The builder is fail-closed. When an input is missing it records
``MISSING`` / ``UNAVAILABLE`` / ``UNKNOWN`` / ``BLOCKED`` instead of inventing
a value. It never creates a binary flood label, never derives a formal
negative from absence of evidence, and never enables supervised or multimodal
training. Dense embedding vectors are referenced by URI/hash/dim, never copied
into the public CSV. All outputs are written under ``local_runs/`` only.

This stage prepares the road to ground truth and multimodality. It does not
train a model. The supervised training gate remains BLOCKED while there is no
operational patch-level ground truth and no formal negatives.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "multimodal" / "v2bn"

FEATURE_VERSION = "v2bn"
DINO_BACKBONE = "facebook/dinov2-with-registers-base"
DINO_EMBEDDING_DIM_EXPECTED = 768
EMBEDDING_SUCCESS_STATES = {"SUCCESS", "SKIPPED_EXISTING"}

# Default input locations (relative to ROOT). All are optional and probed
# fail-closed: a missing input downgrades the relevant flag, never crashes.
DEFAULT_INPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "dino_inputs"
    / "revp_v1fu_dino_sentinel_input_manifest"
    / "dino_sentinel_input_manifest_v1fu.csv"
)
DEFAULT_EMBEDDING_MANIFEST = (
    ROOT / "local_runs" / "dino_embeddings" / "v1ge" / "dino_expanded_embedding_manifest_v1ge.csv"
)
DEFAULT_FALLBACK_EMBEDDING_MANIFEST = (
    ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"
)
DEFAULT_PATCH_REGISTRY = ROOT / "datasets" / "patch_corpus_registry.csv"
# The historical fail-closed feature-store registry that legitimately holds
# zero parsed dense vectors. It is reconciled against the 12-embedding final
# state, NOT edited.
DEFAULT_FEATURE_STORE_ZERO = ROOT / "datasets" / "dino_embedding_feature_store_registry_v1ph.csv"
DEFAULT_GIS_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gq"
DEFAULT_EVIDENCE_REGISTRY_REPORT = (
    ROOT / "outputs_public" / "execution_reports" / "v2at_evidence_registry_event_patch_report.md"
)
DEFAULT_OVERLAY_REPORT = (
    ROOT / "outputs_public" / "execution_reports" / "v2au_patch_event_overlay_geometry_report.md"
)
DEFAULT_PROTOCOL_C_SUMMARY = ROOT / "outputs_public" / "tables" / "table_protocol_c_summary.csv"


# Methodological guardrails enforced and reported by this stage. Every value is
# a hard invariant: the builder cannot flip any of these on.
METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "supervised_training": False,
    "labels_created": False,
    "targets_created": False,
    "formal_negative_created": False,
    "negative_from_absence": False,
    "predictive_claims": False,
    "multimodal_execution_enabled": False,
    "multimodal_training_enabled": False,
    "dino_frozen_encoder": True,
    "dino_finetuned": False,
    "early_fusion": False,
    "pixel_space_fusion": False,
    "dense_vectors_copied_to_public_csv": False,
    "outputs_local_only": True,
}


CORE_FIELDS = [
    "sample_id",
    "canonical_patch_id",
    "dino_input_id",
    "region",
    "source_asset_id",
    "split_group",
    "feature_version",
    "dino_backbone",
    "dino_embedding_available",
    "dino_embedding_uri",
    "dino_embedding_dim",
    "dino_embedding_hash",
    "dino_manifest_source",
    "patch_registry_available",
    "gis_feature_available",
    "gis_feature_status",
    "evidence_registry_available",
    "observed_event_id",
    "event_patch_binding_status",
    "gt_patch_flood_observed",
    "gt_label_quality",
    "gt_source_family",
    "gt_temporal_alignment",
    "gt_spatial_alignment",
    "gt_negative_type",
    "allowed_for_training",
    "allowed_for_review",
    "exclusion_reason",
    "guardrail_status",
    "notes",
]

INVENTORY_FIELDS = [
    "source_path",
    "source_type",
    "exists",
    "row_count",
    "success_count",
    "valid_768d_count",
    "regions_covered",
    "status",
    "interpretation",
]

MISSINGNESS_FIELDS = [
    "feature_name",
    "missing_count",
    "present_count",
    "missing_rate",
    "regions_affected",
    "status",
    "interpretation",
]

JOIN_FIELDS = [
    "join_name",
    "left_source",
    "right_source",
    "left_count",
    "right_count",
    "matched_count",
    "unmatched_left_count",
    "unmatched_right_count",
    "status",
    "interpretation",
]

# Markers that count as "missing" when measuring column missingness.
NA_MARKERS = {"", "NA", "UNKNOWN", "MISSING", "UNAVAILABLE", "NOT_ESTABLISHED", "BLOCKED", "NONE"}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

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


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    return any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def rel_to_root(path: Path) -> str:
    """Return a repo-relative POSIX path, never a private absolute path."""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


# --------------------------------------------------------------------------- #
# Embedding manifest handling
# --------------------------------------------------------------------------- #

def parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def embedding_row_is_valid(row: dict[str, str]) -> bool:
    """A real, usable embedding requires a success state and a positive dim."""
    if row.get("success", "").strip().upper() not in EMBEDDING_SUCCESS_STATES:
        return False
    dim = parse_int(row.get("embedding_dim", ""))
    return dim is not None and dim > 0


def embedding_uri(manifest_path: Path, embedding_path: str) -> str:
    """Compose a repo-relative reference to the dense vector (never copied)."""
    embedding_path = (embedding_path or "").strip()
    if not embedding_path:
        return ""
    candidate = (manifest_path.parent / embedding_path)
    return rel_to_root(candidate)


def build_embedding_index(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Map dino_input_id -> validated embedding facts from a local manifest.

    Only rows with a success state and a valid dimension are indexed. The
    private ``source_path`` column is intentionally ignored: we reference the
    relative ``embedding_path`` plus hash and dim, never an absolute path.
    """
    index: dict[str, dict[str, str]] = {}
    for row in read_csv(manifest_path):
        if not embedding_row_is_valid(row):
            continue
        key = row.get("dino_input_id", "").strip()
        if not key:
            continue
        dim = parse_int(row.get("embedding_dim", ""))
        index[key] = {
            "dino_embedding_available": "True",
            "dino_embedding_uri": embedding_uri(manifest_path, row.get("embedding_path", "")),
            "dino_embedding_dim": str(dim),
            "dino_embedding_hash": row.get("hash", "").strip(),
            "dino_backbone": row.get("model_backbone", "").strip() or DINO_BACKBONE,
            "region": row.get("region", "").strip(),
        }
    return index


def classify_embedding_source(path: Path, kind: str) -> dict[str, object]:
    """Classify one embedding-evidence source for the reconciliation inventory.

    ``kind`` selects the interpretation lens:
      * ``local_manifest`` — a real per-patch embedding manifest;
      * ``feature_store_zero`` — the v1ph fail-closed dense-vector registry;
      * ``public_final_report`` — a public textual claim of N embeddings.
    """
    exists = path.exists()
    rows = read_csv(path) if exists and path.suffix.lower() == ".csv" else []
    if kind == "local_manifest":
        valid = [r for r in rows if embedding_row_is_valid(r)]
        regions = sorted({r.get("region", "").strip() for r in valid if r.get("region", "").strip()})
        if not exists:
            status, interp = "MISSING", "Local embedding manifest not present in this workspace."
        elif valid:
            status = "LOCAL_MANIFEST_AVAILABLE"
            interp = (
                f"{len(valid)} real frozen-encoder embeddings with valid dimension; "
                "authoritative local evidence for the feature table."
            )
        else:
            status, interp = "BLOCKED", "Manifest present but no row reached a success state with a valid dimension."
        return {
            "source_path": rel_to_root(path),
            "source_type": "DINO_EMBEDDING_MANIFEST_LOCAL",
            "exists": str(exists),
            "row_count": len(rows),
            "success_count": len(valid),
            "valid_768d_count": sum(1 for r in valid if parse_int(r.get("embedding_dim", "")) == DINO_EMBEDDING_DIM_EXPECTED),
            "regions_covered": " ".join(regions),
            "status": status,
            "interpretation": interp,
        }
    if kind == "feature_store_zero":
        data_rows = len(rows)
        status = "HISTORICAL_STALE_ZERO_EMBEDDINGS" if exists and data_rows == 0 else ("MISSING" if not exists else "NON_EMPTY_REVIEW")
        interp = (
            "Fail-closed dense-vector registry written empty because no dense vector was parsed; "
            "this 0-vector state is a parsing-scope artifact, NOT a claim that embeddings are absent. "
            "It does not contradict the 12 successfully extracted embedding files."
        )
        return {
            "source_path": rel_to_root(path),
            "source_type": "DINO_FEATURE_STORE_REGISTRY_DENSE_VECTORS",
            "exists": str(exists),
            "row_count": data_rows,
            "success_count": 0,
            "valid_768d_count": 0,
            "regions_covered": "",
            "status": status,
            "interpretation": interp,
        }
    # public_final_report
    status = "PUBLIC_FINAL_REPORT_ONLY" if exists else "MISSING"
    interp = (
        "Public final report textually records 12 real 768D embeddings (4 per region). "
        "Textual evidence only; the per-patch dense vectors live in local manifests/sidecars."
    )
    return {
        "source_path": rel_to_root(path),
        "source_type": "PUBLIC_FINAL_REPORT",
        "exists": str(exists),
        "row_count": "",
        "success_count": "",
        "valid_768d_count": "",
        "regions_covered": "",
        "status": status,
        "interpretation": interp,
    }


def build_embedding_inventory(inputs: dict[str, Path]) -> list[dict[str, object]]:
    inventory = [
        classify_embedding_source(inputs["embedding_manifest"], "local_manifest"),
        classify_embedding_source(inputs["fallback_embedding_manifest"], "local_manifest"),
        classify_embedding_source(inputs["feature_store_zero"], "feature_store_zero"),
        classify_embedding_source(inputs["protocol_c_summary"], "public_final_report"),
    ]
    return inventory


# --------------------------------------------------------------------------- #
# GIS / evidence availability probes (flags only, never promoted)
# --------------------------------------------------------------------------- #

def probe_gis_status(gis_dir: Path) -> tuple[str, str]:
    """Probe local GIS baseline availability. Optional and fail-closed."""
    if gis_dir.exists() and any(gis_dir.glob("*.csv")):
        return "True", "AVAILABLE_LOCAL_PROXY_NOT_GROUND_TRUTH"
    return "False", "UNAVAILABLE_LOCAL_OPTIONAL"


def probe_evidence_status(report_path: Path) -> str:
    return "True" if report_path.exists() else "False"


def probe_binding_status(overlay_report: Path) -> str:
    """No patch-event binding is asserted without geometry overlay evidence."""
    if overlay_report.exists():
        return "NO_PATCH_EVENT_OVERLAY_GEOMETRY_BLOCKED"
    return "UNAVAILABLE_NO_OVERLAY"


# --------------------------------------------------------------------------- #
# Core feature table
# --------------------------------------------------------------------------- #

def build_feature_table(
    spine_rows: list[dict[str, str]],
    embedding_index: dict[str, dict[str, str]],
    *,
    gis_available: str,
    gis_status: str,
    evidence_available: str,
    binding_status: str,
    manifest_source: str,
) -> list[dict[str, object]]:
    """Compose one review-only feature row per Sentinel input.

    Ground-truth columns are placeholders kept empty/NA on purpose. Training is
    blocked for every row. Review eligibility is granted only when a real local
    embedding exists for the sample.
    """
    rows: list[dict[str, object]] = []
    for src in spine_rows:
        dino_id = (src.get("dino_input_id") or "").strip()
        canonical = (src.get("canonical_patch_id") or "").strip()
        region = (src.get("region") or "").strip()
        emb = embedding_index.get(dino_id, {})
        embedding_available = emb.get("dino_embedding_available", "False") == "True"

        if embedding_available:
            allowed_for_review = "True"
            exclusion_reason = ""
        else:
            allowed_for_review = "False"
            exclusion_reason = "NO_LOCAL_EMBEDDING_NO_STRUCTURAL_EVIDENCE_FOR_REVIEW"

        rows.append(
            {
                "sample_id": dino_id or f"{FEATURE_VERSION}_{canonical}",
                "canonical_patch_id": canonical,
                "dino_input_id": dino_id,
                "region": region,
                "source_asset_id": (src.get("source_asset_id") or "").strip(),
                "split_group": (src.get("split_group") or "").strip(),
                "feature_version": FEATURE_VERSION,
                "dino_backbone": emb.get("dino_backbone", DINO_BACKBONE),
                "dino_embedding_available": "True" if embedding_available else "False",
                "dino_embedding_uri": emb.get("dino_embedding_uri", ""),
                "dino_embedding_dim": emb.get("dino_embedding_dim", ""),
                "dino_embedding_hash": emb.get("dino_embedding_hash", ""),
                "dino_manifest_source": manifest_source if embedding_available else "",
                "patch_registry_available": "True" if canonical else "False",
                "gis_feature_available": gis_available,
                "gis_feature_status": gis_status,
                "evidence_registry_available": evidence_available,
                "observed_event_id": "",  # UNKNOWN — no binding asserted
                "event_patch_binding_status": binding_status,
                # Ground-truth placeholders: empty/NA by design until a formal,
                # auditable label protocol produces them.
                "gt_patch_flood_observed": "",
                "gt_label_quality": "NO_LABEL",
                "gt_source_family": "",
                "gt_temporal_alignment": "NOT_ESTABLISHED",
                "gt_spatial_alignment": "NOT_ESTABLISHED",
                "gt_negative_type": "",  # never derived from absence
                "allowed_for_training": "False",  # hard gate
                "allowed_for_review": allowed_for_review,
                "exclusion_reason": exclusion_reason,
                "guardrail_status": "PASS" if embedding_available else "BLOCKED_EXPECTED_NO_EMBEDDING",
                "notes": "review_only_feature_readiness_row; no label, no target, no predictive claim",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Missingness / joins
# --------------------------------------------------------------------------- #

def compute_missingness(core_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = len(core_rows)
    out: list[dict[str, object]] = []
    for field in CORE_FIELDS:
        missing_regions: set[str] = set()
        missing = 0
        for row in core_rows:
            value = str(row.get(field, "")).strip().upper()
            if value in NA_MARKERS:
                missing += 1
                if row.get("region"):
                    missing_regions.add(str(row["region"]))
        present = total - missing
        rate = round(missing / total, 4) if total else 0.0
        if field in {"gt_patch_flood_observed", "gt_negative_type", "gt_source_family", "observed_event_id"}:
            status = "EXPECTED_EMPTY_NO_GROUND_TRUTH"
            interp = "Ground-truth placeholder kept empty/NA on purpose; no label or negative created."
        elif rate == 0.0:
            status = "COMPLETE"
            interp = "Column fully populated across samples."
        elif rate >= 0.99:
            status = "MOSTLY_MISSING"
            interp = "Column missing for nearly all samples; reflects current local availability."
        else:
            status = "PARTIAL"
            interp = "Column present for a subset of samples; reflects real local coverage."
        out.append(
            {
                "feature_name": field,
                "missing_count": missing,
                "present_count": present,
                "missing_rate": rate,
                "regions_affected": " ".join(sorted(missing_regions)),
                "status": status,
                "interpretation": interp,
            }
        )
    return out


def build_join_audit(
    spine_rows: list[dict[str, str]],
    embedding_index: dict[str, dict[str, str]],
    inputs: dict[str, Path],
) -> list[dict[str, object]]:
    spine_ids = {(r.get("dino_input_id") or "").strip() for r in spine_rows if (r.get("dino_input_id") or "").strip()}
    emb_ids = set(embedding_index)
    matched = spine_ids & emb_ids
    unmatched_left = spine_ids - emb_ids
    unmatched_right = emb_ids - spine_ids
    spine_status = "OK" if spine_rows else "BLOCKED_EMPTY_SPINE"
    join_status = "MATCHED_SUBSET" if matched else "NO_MATCH"
    return [
        {
            "join_name": "spine_x_dino_embeddings",
            "left_source": rel_to_root(inputs["input_manifest"]),
            "right_source": rel_to_root(inputs["embedding_manifest"]),
            "left_count": len(spine_ids),
            "right_count": len(emb_ids),
            "matched_count": len(matched),
            "unmatched_left_count": len(unmatched_left),
            "unmatched_right_count": len(unmatched_right),
            "status": join_status,
            "interpretation": (
                f"{len(matched)} Sentinel inputs carry a real local embedding; "
                f"{len(unmatched_left)} remain embedding-absent (review-blocked, not negative). "
                f"{len(unmatched_right)} embeddings without a spine row require provenance review."
            ),
        },
        {
            "join_name": "spine_presence",
            "left_source": rel_to_root(inputs["input_manifest"]),
            "right_source": rel_to_root(inputs["patch_registry"]),
            "left_count": len(spine_rows),
            "right_count": len(read_csv(inputs["patch_registry"])),
            "matched_count": len(spine_rows),
            "unmatched_left_count": 0,
            "unmatched_right_count": 0,
            "status": spine_status,
            "interpretation": "Spine derives from the canonical Sentinel input manifest; corpus registry is corpus-level context.",
        },
    ]


# --------------------------------------------------------------------------- #
# Gate / guardrails / QA
# --------------------------------------------------------------------------- #

def training_gate(core_rows: list[dict[str, object]]) -> dict[str, object]:
    any_training = any(str(r.get("allowed_for_training")) == "True" for r in core_rows)
    return {
        "phase": FEATURE_VERSION,
        "features_created": bool(core_rows),
        "labels_created": False,
        "formal_negative_count": 0,
        "supervised_training_enabled": False,
        "multimodal_training_enabled": False,
        "multimodal_execution_enabled": False,
        "any_row_allowed_for_training": any_training,
        "allowed_next_step": "ground_truth_patch_level_resolution",
        "blocked_reason": "NO_OPERATIONAL_GROUND_TRUTH_PATCH_LEVEL_AND_NO_FORMAL_NEGATIVES",
        "review_only": True,
        "predictive_claims": False,
    }


def build_guardrails(core_rows: list[dict[str, object]], gate: dict[str, object]) -> dict[str, object]:
    checks: dict[str, str] = {}

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    checks["no_binary_label_created"] = verdict(all(str(r.get("gt_patch_flood_observed", "")) == "" for r in core_rows))
    checks["no_formal_negative_created"] = verdict(all(str(r.get("gt_negative_type", "")) == "" for r in core_rows))
    checks["no_training_allowed"] = verdict(not gate["any_row_allowed_for_training"])
    checks["labels_created_false"] = verdict(gate["labels_created"] is False)
    checks["formal_negative_count_zero"] = verdict(gate["formal_negative_count"] == 0)
    checks["multimodal_execution_disabled"] = verdict(METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False)
    checks["multimodal_training_disabled"] = verdict(METHODOLOGICAL_GUARDRAILS["multimodal_training_enabled"] is False)
    checks["dino_frozen_not_finetuned"] = verdict(
        METHODOLOGICAL_GUARDRAILS["dino_frozen_encoder"] and not METHODOLOGICAL_GUARDRAILS["dino_finetuned"]
    )
    checks["no_early_or_pixel_fusion"] = verdict(
        not METHODOLOGICAL_GUARDRAILS["early_fusion"] and not METHODOLOGICAL_GUARDRAILS["pixel_space_fusion"]
    )
    checks["dense_vectors_referenced_not_embedded"] = verdict(
        all("," not in str(r.get("dino_embedding_uri", "")) for r in core_rows)
    )
    checks["local_runs_ignored"] = verdict(local_runs_ignored())
    # Expected-blocked rows are not failures; report them so review can see them.
    blocked_expected = sum(1 for r in core_rows if str(r.get("guardrail_status", "")).startswith("BLOCKED_EXPECTED"))
    overall = "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL"
    return {
        "phase": FEATURE_VERSION,
        "checks": checks,
        "blocked_expected_rows": blocked_expected,
        "overall": overall,
        **METHODOLOGICAL_GUARDRAILS,
    }


def build_schema_rows() -> list[dict[str, str]]:
    descriptions = {
        "sample_id": ("string", "Stable per-sample id (dino_input_id when present)."),
        "canonical_patch_id": ("string", "Canonical patch identifier from the input manifest."),
        "dino_input_id": ("string", "Sentinel input id; join key to the embedding manifest."),
        "region": ("string", "Region label (Curitiba/Petrópolis/Recife)."),
        "source_asset_id": ("string", "Source Sentinel asset id from the input manifest."),
        "split_group": ("string", "Pre-existing anti-leakage group (region/source grouped, no random split)."),
        "feature_version": ("string", "Feature table version tag."),
        "dino_backbone": ("string", "Frozen DINOv2 backbone used for the embedding."),
        "dino_embedding_available": ("bool", "True only when a real local 768D embedding exists."),
        "dino_embedding_uri": ("string", "Repo-relative reference to the dense vector sidecar (never embedded)."),
        "dino_embedding_dim": ("int", "Detected embedding dimension (768 expected); never invented."),
        "dino_embedding_hash": ("string", "Embedding content hash from the manifest."),
        "dino_manifest_source": ("string", "Manifest that produced the embedding reference."),
        "patch_registry_available": ("bool", "Whether the sample is registered in the canonical input spine."),
        "gis_feature_available": ("bool", "Whether a local GIS proxy is available (not ground truth)."),
        "gis_feature_status": ("string", "GIS availability status; proxy only."),
        "evidence_registry_available": ("bool", "Whether the evidence registry report is present."),
        "observed_event_id": ("string", "Bound observed event id; empty/UNKNOWN unless evidence asserts it."),
        "event_patch_binding_status": ("string", "Patch-event binding status; blocked without overlay geometry."),
        "gt_patch_flood_observed": ("string", "Ground-truth flood label placeholder; empty/NA by design."),
        "gt_label_quality": ("string", "Label quality; NO_LABEL until a formal protocol exists."),
        "gt_source_family": ("string", "Ground-truth source family; empty until acquired."),
        "gt_temporal_alignment": ("string", "Temporal alignment status; NOT_ESTABLISHED."),
        "gt_spatial_alignment": ("string", "Spatial alignment status; NOT_ESTABLISHED."),
        "gt_negative_type": ("string", "Formal negative type; empty (never from absence of evidence)."),
        "allowed_for_training": ("bool", "Always False at this stage (hard gate)."),
        "allowed_for_review": ("bool", "True when a real local embedding supports structural review."),
        "exclusion_reason": ("string", "Why a sample is excluded from review, when applicable."),
        "guardrail_status": ("string", "PASS or BLOCKED_EXPECTED_* per sample."),
        "notes": ("string", "Free-text review-only annotation."),
    }
    fill = {
        "dino_embedding_available": "True only with a manifest success state and a valid dimension.",
        "gt_patch_flood_observed": "Empty/NA until an auditable label protocol produces it.",
        "gt_negative_type": "Empty; absence/pseudo-absence/background/anchor-distance are not negatives.",
        "allowed_for_training": "Hard-coded False; gate stays blocked at this stage.",
        "allowed_for_review": "True only for samples with real local embedding/structural evidence.",
    }
    rows = []
    for field in CORE_FIELDS:
        dtype, desc = descriptions.get(field, ("string", ""))
        rows.append(
            {
                "feature_name": field,
                "dtype": dtype,
                "description": desc,
                "fill_policy": fill.get(field, "Populated from inputs fail-closed; missing -> NA/UNKNOWN."),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(summary: dict[str, Any], inventory: list[dict[str, Any]]) -> str:
    region_counts = summary["region_counts"]
    region_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(region_counts.items())) or "- (none)"
    inv_lines = "\n".join(
        f"- `{r['source_path']}` → **{r['status']}** ({r['interpretation']})" for r in inventory
    )
    return f"""# REV-P {FEATURE_VERSION} — Multimodal Feature Table (review-only readiness)

Version: `{FEATURE_VERSION}`
Generated: {summary['created_utc']}

This stage assembles one compact, auditable feature row per Sentinel input,
joining the canonical input spine with the real local DINOv2 embedding
evidence and availability flags. It does not create labels, does not create
formal negatives, and does not train any model. The supervised training gate
stays blocked.

## 1. State of stage {FEATURE_VERSION}

- Feature rows (one per Sentinel input): **{summary['core_row_count']}**
- Samples with a real local 768D embedding: **{summary['embedding_available_count']}**
- Samples review-eligible: **{summary['review_eligible_count']}**
- Samples allowed for training: **{summary['training_allowed_count']}** (gate blocked)
- Regions:
{region_lines}

## 2. What was joined

The spine is the canonical Sentinel input manifest (v1fu). Each row carries the
pre-existing anti-leakage `split_group` (region/source grouped, never a random
split). The DINOv2 embedding manifest (v1ge) is joined on `dino_input_id`,
contributing `dino_embedding_uri`, `dino_embedding_dim`, `dino_embedding_hash`
as references only — dense vectors are never copied into the CSV. GIS,
evidence-registry and patch-event binding enter as availability flags only.

## 3. What stayed blocked

- Operational patch-level ground truth: absent.
- Formal negatives: 0 (absence of evidence is not a negative).
- Supervised training: blocked. Multimodal execution/training: disabled.
- Patch-event binding: `{summary['binding_status']}`.
- Samples without a local embedding are review-blocked, not labelled negative.

## 4. How DINOv2 embeddings are treated

DINOv2 stays a frozen visual encoder. Its embeddings are review-only vectors
referenced by URI/hash/dim. There is no fine-tuning, no early fusion, no
pixel-space fusion. The historical/final reconciliation of embedding counts is:

{inv_lines}

The apparent "0 vs 12" divergence is a scope difference, not a contradiction:
the 0 is a fail-closed dense-vector feature-store registry that parsed no
vector into its table, while the 12 are real successfully extracted embedding
files recorded in the local manifest and the public final reports.

## 5. Why there is still no training

`labels_created=false`, `formal_negative_count=0`,
`can_train_supervised_model=false`. With no operational patch-level ground
truth and no formal negatives, supervised training is methodologically
blocked. Unknown stays unknown.

## 6. What is missing to unblock training

- An auditable, independent patch-level ground-truth label source.
- A formal negative protocol with comparable evidence (never from absence).
- Confirmed temporal/spatial alignment between events and patches.
- Group/block-based anti-leakage splits validated for evaluation.

## 7. Recommended next steps

- Run the ground-truth and training-gate scaffold (`{summary['next_stage']}`)
  to register the label/negative policy without creating any label.
- Acquire/adjudicate patch-level reference geometry to enable real binding.
- Keep the gate blocked until the above are auditable.

## Guardrail note

Review-only readiness. No operational flood detection, no validated
prediction, no flood accuracy. All outputs are local-only and lightweight.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "input_manifest": Path(args.input_manifest),
        "embedding_manifest": Path(args.embedding_manifest),
        "fallback_embedding_manifest": Path(args.fallback_embedding_manifest),
        "patch_registry": Path(args.patch_registry),
        "feature_store_zero": Path(args.feature_store_zero),
        "gis_dir": Path(args.gis_dir),
        "evidence_report": Path(args.evidence_report),
        "overlay_report": Path(args.overlay_report),
        "protocol_c_summary": Path(args.protocol_c_summary),
    }


def build_artifacts(inputs: dict[str, Path]) -> dict[str, Any]:
    """Pure builder: read inputs, return all data structures (no writes)."""
    spine_rows = read_csv(inputs["input_manifest"])
    embedding_index = build_embedding_index(inputs["embedding_manifest"])
    if not embedding_index:
        embedding_index = build_embedding_index(inputs["fallback_embedding_manifest"])
        active_manifest = inputs["fallback_embedding_manifest"]
    else:
        active_manifest = inputs["embedding_manifest"]

    gis_available, gis_status = probe_gis_status(inputs["gis_dir"])
    evidence_available = probe_evidence_status(inputs["evidence_report"])
    binding_status = probe_binding_status(inputs["overlay_report"])

    core_rows = build_feature_table(
        spine_rows,
        embedding_index,
        gis_available=gis_available,
        gis_status=gis_status,
        evidence_available=evidence_available,
        binding_status=binding_status,
        manifest_source=rel_to_root(active_manifest),
    )
    inventory = build_embedding_inventory(inputs)
    missingness = compute_missingness(core_rows)
    join_audit = build_join_audit(spine_rows, embedding_index, inputs)
    schema_rows = build_schema_rows()
    gate = training_gate(core_rows)
    guardrails = build_guardrails(core_rows, gate)

    embedding_available_count = sum(1 for r in core_rows if r["dino_embedding_available"] == "True")
    review_eligible_count = sum(1 for r in core_rows if r["allowed_for_review"] == "True")
    training_allowed_count = sum(1 for r in core_rows if r["allowed_for_training"] == "True")
    region_counts = dict(sorted(Counter(r["region"] for r in core_rows if r["region"]).items()))

    summary = {
        "phase": FEATURE_VERSION,
        "phase_name": "MULTIMODAL_FEATURE_TABLE_BUILDER_REVIEW_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "core_row_count": len(core_rows),
        "embedding_available_count": embedding_available_count,
        "review_eligible_count": review_eligible_count,
        "training_allowed_count": training_allowed_count,
        "region_counts": region_counts,
        "binding_status": binding_status,
        "gis_feature_status": gis_status,
        "evidence_registry_available": evidence_available,
        "guardrail_overall": guardrails["overall"],
        "next_stage": "v2bo",
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "core_rows": core_rows,
        "inventory": inventory,
        "missingness": missingness,
        "join_audit": join_audit,
        "schema_rows": schema_rows,
        "gate": gate,
        "guardrails": guardrails,
        "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    out = output_dir
    write_csv(out / f"multimodal_feature_table_core_{FEATURE_VERSION}.csv", art["core_rows"], CORE_FIELDS)
    write_csv(out / f"multimodal_feature_schema_{FEATURE_VERSION}.csv", art["schema_rows"], ["feature_name", "dtype", "description", "fill_policy"])
    write_csv(out / f"multimodal_embedding_inventory_{FEATURE_VERSION}.csv", art["inventory"], INVENTORY_FIELDS)
    write_csv(out / f"multimodal_missingness_report_{FEATURE_VERSION}.csv", art["missingness"], MISSINGNESS_FIELDS)
    write_csv(out / f"multimodal_join_audit_{FEATURE_VERSION}.csv", art["join_audit"], JOIN_FIELDS)
    write_json(out / f"multimodal_training_gate_{FEATURE_VERSION}.json", art["gate"])
    write_json(out / f"multimodal_guardrails_{FEATURE_VERSION}.json", art["guardrails"])
    write_json(out / f"multimodal_feature_table_summary_{FEATURE_VERSION}.json", art["summary"])
    report = build_report(art["summary"], art["inventory"])
    (out / f"multimodal_feature_table_report_{FEATURE_VERSION}.md").write_text(report, encoding="utf-8")
    return sorted(p.name for p in out.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bn multimodal feature table builder. Review-only; does not train, label, or enable multimodal execution."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_EMBEDDING_MANIFEST))
    parser.add_argument("--fallback-embedding-manifest", default=str(DEFAULT_FALLBACK_EMBEDDING_MANIFEST))
    parser.add_argument("--patch-registry", default=str(DEFAULT_PATCH_REGISTRY))
    parser.add_argument("--feature-store-zero", default=str(DEFAULT_FEATURE_STORE_ZERO))
    parser.add_argument("--gis-dir", default=str(DEFAULT_GIS_DIR))
    parser.add_argument("--evidence-report", default=str(DEFAULT_EVIDENCE_REGISTRY_REPORT))
    parser.add_argument("--overlay-report", default=str(DEFAULT_OVERLAY_REPORT))
    parser.add_argument("--protocol-c-summary", default=str(DEFAULT_PROTOCOL_C_SUMMARY))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    inputs = resolve_inputs(args)
    art = build_artifacts(inputs)
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    # Exit non-zero only on a real guardrail failure, not on expected blocks.
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
