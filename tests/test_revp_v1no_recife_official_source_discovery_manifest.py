"""Tests for v1no Recife official source discovery."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1no_recife_official_source_discovery_manifest.py"
MANIFEST = ROOT / "datasets/recife_official_source_resource_manifest.csv"
SUMMARY = ROOT / "datasets/recife_official_source_discovery_summary.csv"
SCHEMA = ROOT / "datasets/schemas/recife_official_source_resource_manifest_schema.csv"
DOC = ROOT / "docs/metodologia_cientifica/protocolo_c_recife_fontes_oficiais_v1no.md"
ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\")


def write_fixture(path: Path) -> None:
    payload = {
        "demandas-dos-cidadaos-e-servicos-dados-vivos-recife": {
            "resources": [
                {"id": "sedec_solic", "name": "SEDEC Solicitacoes Tempo Real", "url": "https://dados.recife.pe.gov.br/sedec.csv", "format": "CSV"},
                {"id": "sedec_vist", "name": "SEDEC Vistorias Tempo Real", "url": "https://dados.recife.pe.gov.br/vist.csv", "format": "CSV"},
                {"id": "sedec_dict", "name": "Dicionario SEDEC Chamados", "url": "https://dados.recife.pe.gov.br/dict.csv", "format": "CSV"},
            ]
        },
        "central-de-atendimento-de-servicos-da-emlurb-156": {
            "resources": [{"id": "emlurb_2022", "name": "Historico EMLURB 156 2022", "url": "https://dados.recife.pe.gov.br/emlurb_2022.csv", "format": "CSV"}]
        },
        "pedidos-ao-portal-da-transparencia": {
            "resources": [{"id": "lai_2022", "name": "Pedidos LAI 2022", "url": "https://dados.recife.pe.gov.br/lai.csv", "format": "CSV"}]
        },
        "manifestacoes-recebidas-via-ouvidoria": {
            "resources": [{"id": "ouv_2022", "name": "Ouvidoria 2022", "url": "https://dados.recife.pe.gov.br/ouv.csv", "format": "CSV"}]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1no_discovers_official_resources_from_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "ckan.json"
    write_fixture(fixture)
    env = os.environ.copy()
    env["REVP_RECIFE_DISCOVERY_FIXTURE"] = str(fixture)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert MANIFEST.exists() and SUMMARY.exists() and DOC.exists()
    manifest = rows(MANIFEST)
    assert len(manifest) == 6
    assert {row["official_source"] for row in manifest} == {"true"}
    assert "OFFICIAL_NEGATIVE_CANDIDATE" in {row["source_role"] for row in manifest}
    assert "SERVICE_DEMAND_CONTEXT" in {row["source_role"] for row in manifest}


def test_v1no_schema_and_public_outputs_are_safe() -> None:
    schema_fields = {row["field"] for row in rows(SCHEMA)}
    for field in ["source_id", "resource_url", "official_source", "source_role"]:
        assert field in schema_fields
    for path in [MANIFEST, SUMMARY, DOC]:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABS_PATH.search(text)
        assert "ground_truth,true" not in text
