"""Tests for v1r8 / v1r8b (SUSC-21b): local coverage expansion + expanded refinement.

The two invariants that matter here:
  1. The expansion never acquires or regenerates imagery, and its vectors are
     only usable because the reproduction gate proved the local encoder
     reproduces the already-published v1r3 vectors.
  2. The expanded refinement inherits every v1r7 guardrail unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPANSION = ROOT / "datasets" / "dino_recife_coverage_expansion_embeddings_v1r8.csv"
AUDIT = ROOT / "datasets" / "dino_recife_coverage_expansion_audit_v1r8.csv"
BASE = ROOT / "datasets" / "dino_recife_sedec_all_embeddings_v1r3.csv"
QA8 = ROOT / "datasets" / "dino_recife_expanded_refinement_qa_v1r8.csv"
QUEUE8 = ROOT / "datasets" / "dino_recife_expanded_refinement_review_queue_v1r8.csv"
BOUNDARY8 = ROOT / "datasets" / "dino_evidence_refinement_boundary_matrix_v1r8.csv"
QA7 = ROOT / "datasets" / "dino_recife_evidence_refinement_qa_v1r7.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def kv(path: Path, key_col: str) -> dict[str, str]:
    rows = read_rows(path)
    value_col = [c for c in rows[0] if c != key_col][0]
    return {r[key_col]: r[value_col] for r in rows}


def test_expansion_acquired_and_regenerated_nothing() -> None:
    a = kv(AUDIT, "audit_key")
    assert a["new_imagery_downloaded"] == "0"
    assert a["previews_regenerated"] == "0"
    assert a["labels_created"] == "0"
    assert a["classifier_trained"] == "false"
    for r in read_rows(EXPANSION):
        assert r["new_imagery_downloaded"] == "false"
        assert r["preview_regenerated"] == "false"


def test_reproduction_gate_passed_so_vectors_are_comparable() -> None:
    a = kv(AUDIT, "audit_key")
    assert a["reproduction_gate_verdict"] == "PASS_NEW_VECTORS_COMPARABLE_WITH_V1R3"
    assert float(a["reproduction_gate_max_abs_deviation"]) <= 1e-4
    assert float(a["reproduction_gate_min_cosine"]) >= 0.9999


def test_expansion_uses_the_audited_preview_modality_only() -> None:
    """Guards against silently embedding the 64x64 previews from generate_stacks.py."""
    a = kv(AUDIT, "audit_key")
    assert a["preview_size_enforced"] == "256x256"
    for r in read_rows(EXPANSION):
        assert (r["image_width"], r["image_height"]) == ("256", "256")
        assert r["model_name"] == "dinov2-with-registers-base"
        assert r["embedding_dim"] == "768"


def test_expansion_does_not_overwrite_or_duplicate_v1r3() -> None:
    base = {r["patch_id"] for r in read_rows(BASE)}
    new = {r["patch_id"] for r in read_rows(EXPANSION)}
    assert not (base & new)
    assert len(base) + len(new) == int(kv(AUDIT, "audit_key")["n_local_previews"])


def test_expansion_carries_the_review_only_guardrails() -> None:
    for r in read_rows(EXPANSION):
        assert r["can_create_label"] == "false"
        assert r["can_train_model"] == "false"
        assert r["target_created"] == "false"
        assert r["review_only"] == "true"
        assert r["dino_allowed_use"] == "REVIEW_ONLY_REPRESENTATION"


def test_expanded_refinement_kept_the_preregistered_thresholds() -> None:
    a, b = kv(QA8, "qa_key"), kv(QA7, "qa_key")
    for k in ("preregistered_phi_H", "preregistered_phi_L",
              "preregistered_min_support_points", "preregistered_utility_criterion",
              "unit_of_analysis", "similarity_space"):
        assert a[k] == b[k], f"{k} drifted between v1r7 and v1r8b"
    assert int(a["n_patches_total"]) > int(b["n_patches_total"])


def test_expanded_refinement_boundary_and_queue_stay_review_only() -> None:
    row = read_rows(BOUNDARY8)[0]
    assert row["refinement_can_create_label"] == "false"
    assert row["refinement_can_train_classifier"] == "false"
    assert row["refinement_can_enter_firth_model"] == "false"
    assert row["refinement_role"] == "REVIEW_QUEUE_PRIORITIZATION_ONLY"
    for r in read_rows(QUEUE8):
        assert r["promotion_allowed"] == "false"
        assert r["review_status"] == "PENDING_HUMAN_REVIEW"


def test_expanded_ranking_is_suppressed_because_the_result_is_null() -> None:
    a = kv(QA8, "qa_key")
    assert a["verdict"].startswith("REFINEMENT_SIGNAL_NOT_ESTABLISHED")
    assert not any(r["review_rank"].strip() for r in read_rows(QUEUE8))


def test_spearman_reports_the_effective_n_not_the_nominal_one() -> None:
    """Patches with no v12 point carry NaN; the QA string must say how many
    pairs were actually used."""
    a = kv(QA8, "qa_key")
    for key in ("qa_score_vs_frac_positive_all_patches_spearman",
                "qa_score_vs_frac_positive_targets_spearman"):
        assert "n_used=" in a[key]
    used = int(a["qa_score_vs_frac_positive_all_patches_spearman"]
               .split("n_used=")[1].split()[0])
    assert used < int(a["n_patches_total"])
