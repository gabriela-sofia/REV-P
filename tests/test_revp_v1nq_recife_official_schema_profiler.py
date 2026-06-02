"""Tests for v1nq Recife schema profiler."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nq_recife_official_schema_profiler.py"
PROFILE = ROOT / "datasets/recife_official_schema_profile.csv"
ROLES = ROOT / "datasets/recife_official_field_role_matrix.csv"
VOCAB = ROOT / "datasets/recife_official_vocabulary_audit.csv"
READINESS = ROOT / "datasets/recife_official_temporal_spatial_field_readiness.csv"
SCHEMA = ROOT / "datasets/schemas/recife_official_vocabulary_audit_schema.csv"
DOC = ROOT / "docs/metodologia_cientifica/protocolo_c_recife_perfil_schema_v1nq.md"
ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nq_profiles_columns_and_vocabularies(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "recife_fixture.csv").write_text(
        "data;endereco;bairro;vistoria_risco;processo_situacao;SERVICO_DESCRICAO;SITUACAO\n2022-05-25;Rua A;Ibura;sem risco;finalizado;boca de lobo;concluido\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["REVP_RECIFE_RAW_DIR"] = str(raw)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert any(row["field_role"] == "TEMPORAL_KEY" for row in rows(ROLES))
    assert any(row["field_role"] == "SPATIAL_KEY" for row in rows(ROLES))
    assert any(row["target_vocabulary"] == "vistoria_risco" and row["value"] == "sem risco" for row in rows(VOCAB))
    assert rows(READINESS)[0]["has_temporal_key"] == "true"


def test_v1nq_outputs_safe_and_vocab_does_not_imply_negative() -> None:
    assert "can_imply_negative" in {row["field"] for row in rows(SCHEMA)}
    assert {row["can_imply_negative"] for row in rows(VOCAB)} == {"false"}
    for path in [PROFILE, ROLES, VOCAB, READINESS, DOC]:
        assert not ABS_PATH.search(path.read_text(encoding="utf-8", errors="replace"))
