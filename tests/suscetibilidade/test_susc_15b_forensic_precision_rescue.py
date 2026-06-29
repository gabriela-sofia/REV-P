"""SUSC-15B forensic precision rescue tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

CATALOG = DAT / "susc_15b_precision_event_catalog_v1.csv"
LINKAGE = DAT / "susc_15b_precision_event_patch_linkage_v1.csv"
REPORT = OUT / "SUSC_15B_forensic_precision_rescue_report.md"
MANDATORY = (
    "SUSC-15B e review-only: nao aceita bairro-only como sucesso, nao aceita "
    "street segment sem numero/intersecao como calibracao, nao cria ground "
    "truth, nao libera treino supervisionado e nao cria score v7 automatico."
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_pipeline_and_validator_pass():
    for name in (
        "audit_susc_15b_hidden_coordinate_and_id_fields.py",
        "discover_susc_15b_nonobvious_event_layers.py",
        "acquire_susc_15b_precision_sources_deep.py",
        "parse_susc_15b_precision_sources_deep.py",
        "join_susc_15b_occurrence_ids_to_official_tables.py",
        "build_susc_15b_precision_event_catalog.py",
        "build_susc_15b_precision_event_patch_linkage.py",
        "run_susc_15b_precision_readiness.py",
        "validate_susc_15b_forensic_precision_rescue.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_governance_closed_and_no_score_v7():
    for path in (CATALOG, LINKAGE):
        rows = read_csv(path)
        assert rows
        assert all(row.get("allowed_for_training") == "false" for row in rows)
        assert all(row.get("can_be_ground_truth") == "false" for row in rows)
        assert all(row.get("can_be_used_as_ground_truth") == "false" for row in rows)
        assert all(row.get("review_only") == "true" for row in rows)
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_weak_tiers_are_not_calibration_eligible():
    weak = {
        "T5_official_street_segment_candidate",
        "T6_neighborhood_context_only",
        "T7_rejected_or_non_flood",
    }
    for row in read_csv(CATALOG):
        if row.get("precision_tier") in weak:
            assert row.get("calibration_eligible") == "false"


def test_report_mandatory_statement():
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
