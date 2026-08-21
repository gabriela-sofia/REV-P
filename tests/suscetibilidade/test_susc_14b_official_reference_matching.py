"""SUSC-14B official reference matching tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

MATCHED = DAT / "susc_14b_matched_official_occurrences_v1.csv"
LINKAGE = DAT / "susc_14b_event_patch_linkage_v1.csv"
REPORT = OUT / "SUSC_14B_official_reference_matching_report.md"
MANDATORY = (
    "O SUSC-14B usa referencias espaciais oficiais para tentar associar ocorrencias oficiais sem coordenada "
    "a patches territoriais. Mesmo quando ha match por endereco, logradouro ou eixo viario, o vinculo "
    "permanece review-only, nao cria ground truth, nao libera treino supervisionado e nao autoriza score v7 automatico."
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=ROOT, text=True,
                          capture_output=True, timeout=420, check=False)


def test_pipeline_and_validator_pass():
    for name in (
        "discover_susc_14b_official_spatial_references_live.py",
        "acquire_susc_14b_official_spatial_references.py",
        "parse_susc_14b_official_spatial_references.py",
        "build_susc_14b_address_normalization_index.py",
        "match_susc_14b_occurrences_to_official_references.py",
        "build_susc_14b_event_patch_linkage.py",
        "run_susc_14b_observational_diagnostics.py",
        "validate_susc_14b_official_reference_matching.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_governance_closed_and_score_v7_absent():
    for path in (MATCHED, LINKAGE):
        rows = read_csv(path)
        assert rows
        assert all(row.get("allowed_for_training") == "false" for row in rows)
        assert all(row.get("can_be_ground_truth") == "false" for row in rows)
        assert all(row.get("can_be_used_as_ground_truth") == "false" for row in rows)
        assert all(row.get("review_only") == "true" for row in rows)
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_neighborhood_and_street_segment_constraints():
    for row in read_csv(LINKAGE):
        if row.get("spatial_relation") == "same_neighborhood_context":
            assert row.get("patch_id") == "REGION_LEVEL_NO_PATCH_RESOLUTION"
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
