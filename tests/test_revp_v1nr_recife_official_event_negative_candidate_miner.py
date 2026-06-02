"""Tests for v1nr Recife candidate miner."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nr_recife_official_event_negative_candidate_miner.py"
POS = ROOT / "datasets/recife_official_positive_candidate_registry.csv"
NEG = ROOT / "datasets/recife_official_negative_candidate_registry.csv"
REJECT = ROOT / "datasets/recife_official_candidate_rejection_registry.csv"
HITS = ROOT / "datasets/recife_official_candidate_keyword_hit_registry.csv"
SCHEMA = ROOT / "datasets/schemas/recife_official_candidate_registry_schema.csv"
ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nr_mines_positive_and_strict_negative_candidates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "recife_fixture.csv").write_text(
        "data;endereco;bairro;descricao;situacao;vistoria_risco\n"
        "2022-05-25;Rua A;Ibura;alagamento em canal;aberto;\n"
        "2022-05-26;Rua B;Ibura;vistoria de barreira sem risco;finalizado;sem risco\n"
        "2022-05-27;Rua C;Boa Viagem;pedido finalizado;finalizado;\n"
        "2022-05-28;;;;sem ocorrencia;\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["REVP_RECIFE_RAW_DIR"] = str(raw)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(rows(POS)) >= 1
    negatives = rows(NEG)
    assert len(negatives) == 1
    assert negatives[0]["semantic_trigger"] == "sem risco"
    assert any(row["rejection_reason"] == "REJECTED_STATUS_ONLY_NOT_NEGATIVE_FORMAL" for row in rows(REJECT))
    assert any(row["rejection_reason"] == "FORMAL_NEGATIVE_MINIMUM_FIELDS_MISSING" for row in rows(REJECT))


def test_v1nr_schema_and_no_public_forbidden_claims() -> None:
    assert "candidate_id" in {row["field"] for row in rows(SCHEMA)}
    for path in [POS, NEG, REJECT, HITS]:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABS_PATH.search(text)
        for forbidden in ["can_train_model,true", "can_create_operational_label,true", "operational_label,true", "ground_truth,true"]:
            assert forbidden not in text
