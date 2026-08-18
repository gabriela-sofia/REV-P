"""SUSC-14C official reference expansion tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

MATCHED = DAT / "susc_14c_matched_official_occurrences_v1.csv"
RESOLVED = DAT / "susc_14c_resolved_ambiguous_occurrences_v1.csv"
LINKAGE = DAT / "susc_14c_event_patch_linkage_v1.csv"
REPORT = OUT / "SUSC_14C_official_reference_expansion_report.md"
MANDATORY = (
    "O SUSC-14C expande a aquisicao de referencias oficiais e reprocessa ocorrencias de cheia para aumentar "
    "vinculos observacionais evento-patch. Todos os vinculos permanecem review-only, sem ground truth, sem "
    "treino supervisionado e sem score v7 automatico."
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=ROOT, text=True,
                          capture_output=True, timeout=900, check=False)


def test_pipeline_and_validator_pass():
    for name in (
        "audit_susc_14c_14b_matching_failures.py",
        "acquire_susc_14c_additional_official_references_batches.py",
        "parse_susc_14c_additional_official_references.py",
        "build_susc_14c_reference_feature_union.py",
        "match_susc_14c_occurrences_to_reference_union.py",
        "resolve_susc_14c_ambiguous_occurrence_matches.py",
        "build_susc_14c_event_patch_linkage.py",
        "run_susc_14c_observational_diagnostics.py",
        "validate_susc_14c_official_reference_expansion.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_governance_closed_and_score_v7_absent():
    for path in (MATCHED, RESOLVED, LINKAGE):
        rows = read_csv(path)
        assert rows
        assert all(row.get("allowed_for_training") == "false" for row in rows)
        assert all(row.get("can_be_ground_truth") == "false" for row in rows)
        assert all(row.get("can_be_used_as_ground_truth") == "false" for row in rows)
        assert all(row.get("review_only") == "true" for row in rows)
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_neighborhood_ambiguous_and_street_constraints():
    for row in read_csv(RESOLVED):
        if row.get("georeferencing_status") == "blocked_ambiguous_address":
            assert row.get("evidence_level") == "rejected_not_observed_event"
    for row in read_csv(LINKAGE):
        if row.get("spatial_relation") == "weak_neighborhood_context":
            assert row.get("patch_id") == "REGION_LEVEL_NO_PATCH_RESOLUTION"
            assert row.get("can_be_used_for_observational_evaluation") == "false"
        if row.get("georeferencing_status") == "blocked_ambiguous_address":
            assert row.get("linkage_confidence") != "moderate"
            assert row.get("can_be_used_for_observational_evaluation") == "false"
        if row.get("matched_feature_type") == "street_segment":
            assert row.get("linkage_confidence") != "strong"


def test_moderate_evaluation_requires_official_match():
    for row in read_csv(LINKAGE):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            assert row.get("linkage_confidence") == "moderate"
            assert row.get("georeferencing_status") in {"official_address_point_match", "official_street_segment_match"}


def test_report_mandatory_statement():
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
