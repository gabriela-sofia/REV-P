"""Tests for v1r7 (SUSC-21) DINO evidence refinement in Recife.

The pipeline depends on two PRIVATE inputs (the v12 dataset and the Sentinel
patch directory), so these tests cover the two things that must hold
regardless of those paths:

  1. Guardrails on the committed outputs: no label, no classifier, no event
     validation, no entry into the Firth model -- and fail-closed behaviour
     when the private paths are absent.
  2. The refinement arithmetic itself, on a synthetic fixture where the
     correct answer is known by construction.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1r7_dino_evidence_refinement_recife.py"
SCORES = ROOT / "datasets" / "dino_recife_evidence_refinement_scores_v1r7.csv"
QUEUE = ROOT / "datasets" / "dino_recife_evidence_refinement_review_queue_v1r7.csv"
QA = ROOT / "datasets" / "dino_recife_evidence_refinement_qa_v1r7.csv"
BOUNDARY = ROOT / "datasets" / "dino_evidence_refinement_boundary_matrix_v1r7.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("revp_v1r7", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

def test_boundary_matrix_forbids_label_training_and_model_entry() -> None:
    row = read_rows(BOUNDARY)[0]
    assert row["refinement_can_create_label"] == "false"
    assert row["refinement_can_train_classifier"] == "false"
    assert row["refinement_can_validate_event"] == "false"
    assert row["refinement_can_enter_firth_model"] == "false"
    assert row["refinement_can_prioritize_review"] == "true"
    assert row["refinement_role"] == "REVIEW_QUEUE_PRIORITIZATION_ONLY"
    assert row["unit_of_analysis"] == "PATCH"
    assert row["pseudoreplication_checked"] == "true"


def test_review_queue_never_promotes_a_candidate() -> None:
    rows = read_rows(QUEUE)
    assert rows, "queue must not be empty after a successful run"
    for r in rows:
        assert r["promotion_allowed"] == "false"
        assert r["review_status"] == "PENDING_HUMAN_REVIEW"
        assert r["evidence_role"] == "REVIEW_QUEUE_PRIORITIZATION_ONLY"


def test_scores_carry_no_label_or_training_permission() -> None:
    for r in read_rows(SCORES):
        assert r["refinement_can_create_label"] == "false"
        assert r["refinement_can_train_classifier"] == "false"


def test_qa_reports_zero_labels_and_no_new_acquisition() -> None:
    qa = {r["qa_key"]: r["qa_value"] for r in read_rows(QA)}
    assert qa["labels_created"] == "0"
    assert qa["classifier_trained"] == "false"
    assert qa["new_embeddings_extracted"] == "0"
    assert qa["new_imagery_downloaded"] == "0"
    assert qa["unit_of_analysis"] == "PATCH"
    assert int(qa["qa_pseudoreplication_cluster_n"]) == int(qa["n_patches_total"])


def test_ranking_is_suppressed_when_the_utility_criterion_fails() -> None:
    """The pre-registered contract: a null refinement must not be shipped as
    an ordered queue."""
    qa = {r["qa_key"]: r["qa_value"] for r in read_rows(QA)}
    signal_established = qa["verdict"].startswith("REFINEMENT_SIGNAL_PRESENT")
    ranks = [r["review_rank"] for r in read_rows(QUEUE)]
    if signal_established:
        assert all(v.strip() for v in ranks)
    else:
        assert not any(v.strip() for v in ranks)


def test_fail_closed_without_private_paths(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items()
           if k not in ("REVP_V1R7_IN_V12", "REVP_V1R7_SENTINEL_DIR",
                        "REVP_V1R5_IN_V12", "REVP_V1R5_SENTINEL_DIR")}
    env["REVP_V1R7_OUT_SCORES"] = str(tmp_path / "s.csv")
    env["REVP_V1R7_OUT_QUEUE"] = str(tmp_path / "q.csv")
    env["REVP_V1R7_OUT_QA"] = str(tmp_path / "qa.csv")
    env["REVP_V1R7_OUT_BOUNDARY"] = str(tmp_path / "b.csv")
    res = subprocess.run([sys.executable, str(SCRIPT)], cwd=SCRIPT.parent,
                         env=env, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    qa = {r["qa_key"]: r["qa_value"] for r in read_rows(tmp_path / "qa.csv")}
    assert qa["status"] == "FAIL_CLOSED_MISSING_PRIVATE_PATH_ENV"
    assert read_rows(tmp_path / "s.csv") == []


# --------------------------------------------------------------------------- #
# Refinement arithmetic on a synthetic fixture
# --------------------------------------------------------------------------- #

def test_seed_separation_is_positive_when_seeds_are_structured(module) -> None:
    """Two tight, well-separated groups must yield a positive separation."""
    a = np.array([1.0, 0.0]), np.array([0.99, 0.14])
    b = np.array([0.0, 1.0]), np.array([0.14, 0.99])
    x = np.vstack([a[0], a[1], b[0], b[1]])
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T
    assert module._seed_separation(sim, [0, 1], [2, 3]) > 0.5


def test_seed_separation_is_near_zero_when_groups_are_interleaved(module) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 16))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T
    sep = module._seed_separation(sim, list(range(0, 10)), list(range(10, 20)))
    assert abs(sep) < 0.25


def test_haversine_matches_a_known_distance(module) -> None:
    # Recife city centre to Olinda historic centre, ~6 km.
    d = module._haversine_km(-8.0631, -34.8711, -8.0089, -34.8553)
    assert 5.0 < d < 7.5


def test_thresholds_are_the_preregistered_values() -> None:
    """Guards against silent tuning of the seed definition in the primary route."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("REVP_V1R7_PHI", "REVP_V1R7_MIN_SUPPORT"))}
    res = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util,sys;"
         f"spec=importlib.util.spec_from_file_location('m',r'{SCRIPT}');"
         "m=importlib.util.module_from_spec(spec);"
         f"sys.path.insert(0,r'{SCRIPT.parent}');spec.loader.exec_module(m);"
         "print(m.PHI_H, m.PHI_L, m.MIN_SUPPORT, m.N_PERM, m.RANDOM_SEED)"],
        env=env, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["0.75", "0.25", "2", "20000", "0"]
