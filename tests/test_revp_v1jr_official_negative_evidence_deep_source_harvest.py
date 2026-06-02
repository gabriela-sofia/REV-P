"""Tests for REV-P v1jr official negative evidence deep source harvest."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jr_official_negative_evidence_deep_source_harvest.py"
COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--read-source-queues",
    "--harvest-official-sources",
    "--extract-evidence-snippets",
    "--emit-harvest",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    env = os.environ.copy()
    env.pop("RUN_REVP_INTEGRATION", None)
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (DATASETS / "official_negative_evidence_deep_source_harvest_registry.csv").exists()


def test_harvest_public_outputs_exist() -> None:
    run_once()
    for path in [
        DATASETS / "official_negative_evidence_deep_source_harvest_registry.csv",
        DATASETS / "official_negative_evidence_deep_source_harvest_audit.csv",
        DATASETS / "schemas/official_negative_evidence_deep_source_harvest_registry_schema.csv",
        DATASETS / "schemas/official_negative_evidence_deep_source_harvest_audit_schema.csv",
        DOCS / "protocolo_c_colheita_profunda_evidencia_negativa_v1jr.md",
        DOCS / "protocolo_c_relatorio_colheita_profunda_evidencia_negativa_v1jr.md",
    ]:
        assert path.exists(), path


def test_required_terms_are_audited() -> None:
    run_once()
    audit = read_csv(DATASETS / "official_negative_evidence_deep_source_harvest_audit.csv")
    terms = {row["required_term"] for row in audit}
    assert "sem indício de movimento de massa" in terms
    assert "Petrópolis 2022 CPRM sem ocorrência" in terms


def test_absence_of_record_is_not_promoted() -> None:
    run_once()
    rows = read_csv(DATASETS / "official_negative_evidence_deep_source_harvest_audit.csv")
    assert all(row["candidate_status"] != "FORMAL_NEGATIVE_READY" for row in rows)
    assert all("absence of registry hit is not negative evidence" in row["notes"] for row in rows)


def test_public_outputs_no_private_paths_or_raw_files() -> None:
    run_once()
    for path in [
        DATASETS / "official_negative_evidence_deep_source_harvest_registry.csv",
        DATASETS / "official_negative_evidence_deep_source_harvest_audit.csv",
    ]:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert r"C:\Users\gabriela" not in text
        assert "Documents\\REV-P" not in text
        assert not any(token in text for token in [".pdf", ".html", ".bin"]) or "LOCAL_ONLY" in text
