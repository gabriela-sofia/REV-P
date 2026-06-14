"""REV-P v2bo — Ground-truth and training-gate scaffold (no labels created).

Prepares the *protocol* for patch-level ground truth without producing a single
label. It emits:

  * a patch registry scaffold with empty/NA label columns, one row per
    candidate patch;
  * an explicit label policy (what may become a label, and only how);
  * an explicit negative policy (absence/pseudo-absence/background/anchor
    distance are NOT formal negatives; unknown stays unknown);
  * a training gate that stays BLOCKED.

It is fail-closed: it reads the v2bn feature table when available and falls
back to the canonical input manifest. It never writes a flood label, never
derives a negative from absence, never enables training. Outputs are local-only
and lightweight.
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
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bo"

STAGE = "v2bo"

DEFAULT_FEATURE_TABLE = (
    ROOT / "local_runs" / "multimodal" / "v2bn" / "multimodal_feature_table_core_v2bn.csv"
)
DEFAULT_INPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "dino_inputs"
    / "revp_v1fu_dino_sentinel_input_manifest"
    / "dino_sentinel_input_manifest_v1fu.csv"
)


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "targets_created": False,
    "formal_negative_created": False,
    "negative_from_absence": False,
    "supervised_training": False,
    "multimodal_execution_enabled": False,
    "multimodal_training_enabled": False,
    "predictive_claims": False,
    "unknown_stays_unknown": True,
}


SCAFFOLD_FIELDS = [
    "canonical_patch_id",
    "region",
    "candidate_event_id",
    "positive_evidence_status",
    "negative_evidence_status",
    "temporal_alignment_status",
    "spatial_alignment_status",
    "human_review_required",
    "gt_patch_flood_observed",
    "label_quality",
    "allowed_for_training",
    "blocked_reason",
]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
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


# --------------------------------------------------------------------------- #
# Scaffold construction
# --------------------------------------------------------------------------- #

def load_candidate_patches(feature_table: Path, input_manifest: Path) -> tuple[list[dict[str, str]], str]:
    """Return (patches, source_tag). Fail-closed: feature table -> manifest -> empty."""
    rows = read_csv(feature_table)
    if rows:
        return rows, "v2bn_feature_table"
    rows = read_csv(input_manifest)
    if rows:
        return rows, "v1fu_input_manifest"
    return [], "NONE"


def build_scaffold_rows(patches: list[dict[str, str]]) -> list[dict[str, Any]]:
    """One scaffold row per patch. Every label-bearing column stays empty/NA.

    No positive label, no negative type, no training permission is created.
    Human review is required for every candidate before any label can exist.
    """
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for patch in patches:
        canonical = (patch.get("canonical_patch_id") or "").strip()
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        rows.append(
            {
                "canonical_patch_id": canonical,
                "region": (patch.get("region") or "").strip(),
                "candidate_event_id": "",  # UNKNOWN — no event bound yet
                "positive_evidence_status": "NOT_ESTABLISHED",
                "negative_evidence_status": "NOT_ESTABLISHED",
                "temporal_alignment_status": "NOT_ESTABLISHED",
                "spatial_alignment_status": "NOT_ESTABLISHED",
                "human_review_required": "True",
                "gt_patch_flood_observed": "",  # empty by design
                "label_quality": "NO_LABEL",
                "allowed_for_training": "False",
                "blocked_reason": "NO_OPERATIONAL_GROUND_TRUTH_PATCH_LEVEL",
            }
        )
    return rows


def label_policy() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "labels_created": False,
        "principle": "A label may exist only after an independent, auditable, reviewer-approved protocol.",
        "what_may_become_a_label": [
            {
                "item": "independent_observed_event_truth",
                "condition": "event-specific, independently sourced, documented, temporally and spatially aligned to the patch, double-reviewed and adjudicated",
                "current_status": "absent",
            },
            {
                "item": "reviewer_approved_patch_level_reference",
                "condition": "explicit human adjudication with traceable evidence and recorded uncertainty",
                "current_status": "absent",
            },
        ],
        "what_may_not_become_a_label": [
            "embedding similarity, outlier or cluster membership",
            "GIS vulnerability proxy",
            "external contextual coherence status",
            "metadata/header/bounds/CRS evidence",
            "RGB preview appearance",
        ],
        "label_quality_scale": ["NO_LABEL", "CANDIDATE_REVIEW", "REVIEWER_CONFIRMED", "ADJUDICATED"],
        "current_label_quality_for_all_patches": "NO_LABEL",
    }


def negative_policy() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "formal_negative_count": 0,
        "negative_from_absence": False,
        "rules": [
            {"rule": "absence_of_evidence_is_not_negative", "enforced": True},
            {"rule": "pseudo_absence_is_not_formal_negative", "enforced": True},
            {"rule": "random_background_is_not_formal_negative", "enforced": True},
            {"rule": "distance_from_anchor_is_not_formal_negative", "enforced": True},
            {
                "rule": "matched_negatives_require_formal_criteria_and_comparable_evidence",
                "enforced": True,
                "condition": "only with an explicit, reviewer-approved matching protocol and comparable observational evidence",
            },
            {"rule": "unknown_stays_unknown", "enforced": True},
        ],
        "allowed_negative_types_now": [],
        "blocked_reason": "NO_FORMAL_NEGATIVE_PROTOCOL_AND_NO_COMPARABLE_EVIDENCE",
    }


def training_gate(scaffold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    any_training = any(str(r.get("allowed_for_training")) == "True" for r in scaffold_rows)
    any_label = any(str(r.get("gt_patch_flood_observed", "")).strip() != "" for r in scaffold_rows)
    return {
        "phase": STAGE,
        "scaffold_created": bool(scaffold_rows),
        "candidate_patch_count": len(scaffold_rows),
        "labels_created": any_label,
        "formal_negative_count": 0,
        "supervised_training_enabled": False,
        "multimodal_training_enabled": False,
        "multimodal_execution_enabled": False,
        "any_row_allowed_for_training": any_training,
        "allowed_next_step": "acquire_independent_patch_level_ground_truth_then_human_adjudication",
        "blocked_reason": "NO_OPERATIONAL_GROUND_TRUTH_PATCH_LEVEL_AND_NO_FORMAL_NEGATIVES",
        "review_only": True,
        "predictive_claims": False,
        "future_baselines_when_unblocked": [
            "logistic_regression_on_frozen_embeddings",
            "random_forest",
            "hist_gradient_boosting_or_xgboost",
            "shallow_mlp",
        ],
        "future_validation_policy": "group_or_block_split_not_random",
    }


def build_qa(scaffold_rows: list[dict[str, Any]], gate: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("no_label_created", all(str(r.get("gt_patch_flood_observed", "")) == "" for r in scaffold_rows), "all gt_patch_flood_observed empty")
    add("all_label_quality_no_label", all(r.get("label_quality") == "NO_LABEL" for r in scaffold_rows), "label_quality fixed to NO_LABEL")
    add("no_training_allowed", not gate["any_row_allowed_for_training"], "allowed_for_training False for all rows")
    add("labels_created_false", gate["labels_created"] is False, "training gate reports labels_created False")
    add("formal_negative_count_zero", gate["formal_negative_count"] == 0, "no formal negatives")
    add("human_review_required_all", all(r.get("human_review_required") == "True" for r in scaffold_rows), "every candidate requires human review")
    add("multimodal_disabled", METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False, "multimodal execution disabled")
    add("local_runs_ignored", local_runs_ignored(), ".gitignore checked")
    return checks


def build_report(summary: dict[str, Any]) -> str:
    region_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(summary["region_counts"].items())) or "- (none)"
    return f"""# REV-P {STAGE} — Ground-Truth and Training-Gate Scaffold (no labels created)

Version: `{STAGE}`
Generated: {summary['created_utc']}
Patch source: `{summary['patch_source']}`

This scaffold prepares the patch-level ground-truth *protocol* without creating
any label. Every label-bearing column is intentionally empty/NA. The supervised
training gate stays blocked.

## Candidate patches

- Candidate patch rows: **{summary['candidate_patch_count']}**
- Patches with a label: **{summary['labelled_patch_count']}** (must stay 0)
- Patches allowed for training: **{summary['training_allowed_count']}** (gate blocked)
- Regions:
{region_lines}

## Label policy

A label may exist only after an independent, auditable, reviewer-approved
protocol with temporal and spatial alignment plus double review and
adjudication. Embeddings, GIS proxies, contextual coherence, metadata/CRS
evidence and preview appearance may NOT become labels. Current label quality
for every patch: `NO_LABEL`.

## Negative policy

Absence of evidence is not a negative. Pseudo-absence, random background and
distance from an anchor are not formal negatives. Matched negatives require an
explicit, reviewer-approved matching protocol with comparable observational
evidence. Unknown stays unknown. Formal negative count: 0.

## Training gate

`labels_created=false`, `formal_negative_count=0`,
`supervised_training_enabled=false`. Training is blocked. When unblocked by
auditable ground truth, the first models must be light baselines over frozen
embeddings (logistic regression, random forest, HistGradientBoosting/XGBoost,
shallow MLP), evaluated with group/block splits — never a simple random split.

## Guardrail note

Review-only scaffold. No labels, no negatives from absence, no training, no
predictive claims. Outputs are local-only and lightweight.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(feature_table: Path, input_manifest: Path) -> dict[str, Any]:
    patches, source_tag = load_candidate_patches(feature_table, input_manifest)
    scaffold_rows = build_scaffold_rows(patches)
    gate = training_gate(scaffold_rows)
    qa = build_qa(scaffold_rows, gate)
    region_counts = dict(sorted(Counter(r["region"] for r in scaffold_rows if r["region"]).items()))
    summary = {
        "phase": STAGE,
        "phase_name": "GROUND_TRUTH_AND_TRAINING_GATE_SCAFFOLD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_source": source_tag,
        "candidate_patch_count": len(scaffold_rows),
        "labelled_patch_count": sum(1 for r in scaffold_rows if str(r.get("gt_patch_flood_observed", "")).strip()),
        "training_allowed_count": sum(1 for r in scaffold_rows if str(r.get("allowed_for_training")) == "True"),
        "region_counts": region_counts,
        "qa_status": "PASS" if all(c["status"] == "PASS" for c in qa) else "FAIL",
        **{k: v for k, v in gate.items() if k not in {"phase", "candidate_patch_count"}},
    }
    return {
        "scaffold_rows": scaffold_rows,
        "label_policy": label_policy(),
        "negative_policy": negative_policy(),
        "gate": gate,
        "qa": qa,
        "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"gt_patch_registry_scaffold_{STAGE}.csv", art["scaffold_rows"], SCAFFOLD_FIELDS)
    write_json(output_dir / f"gt_label_policy_{STAGE}.json", art["label_policy"])
    write_json(output_dir / f"gt_negative_policy_{STAGE}.json", art["negative_policy"])
    write_json(output_dir / f"gt_training_gate_{STAGE}.json", art["gate"])
    write_csv(output_dir / f"gt_scaffold_qa_{STAGE}.csv", art["qa"], ["check", "status", "detail"])
    (output_dir / f"gt_scaffold_report_{STAGE}.md").write_text(build_report(art["summary"]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bo ground-truth and training-gate scaffold. Creates no labels and enables no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--feature-table", default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(Path(args.feature_table), Path(args.input_manifest))
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["summary"]["qa_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
