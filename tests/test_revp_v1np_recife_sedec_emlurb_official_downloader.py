"""Tests for v1np Recife official downloader with fixture files."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DISCOVERY = ROOT / "scripts/protocolo_c/revp_v1no_recife_official_source_discovery_manifest.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1np_recife_sedec_emlurb_official_downloader.py"
MANIFEST = ROOT / "datasets/recife_official_download_manifest.csv"
INVENTORY = ROOT / "datasets/recife_official_raw_file_inventory.csv"
FAILURES = ROOT / "datasets/recife_official_download_failures.csv"
SCHEMA = ROOT / "datasets/schemas/recife_official_download_manifest_schema.csv"
ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_fixture(tmp_path: Path) -> dict[str, str]:
    ckan = tmp_path / "ckan.json"
    ckan.write_text(
        json.dumps(
            {
                "demandas-dos-cidadaos-e-servicos-dados-vivos-recife": {"resources": [{"id": "sedec_solic", "name": "SEDEC Solicitacoes Tempo Real", "url": "https://example/sedec.csv", "format": "CSV"}]},
                "central-de-atendimento-de-servicos-da-emlurb-156": {"resources": [{"id": "emlurb_2022", "name": "Historico EMLURB 156 2022", "url": "https://example/emlurb.csv", "format": "CSV"}]},
                "pedidos-ao-portal-da-transparencia": {"resources": []},
                "manifestacoes-recebidas-via-ouvidoria": {"resources": []},
            }
        ),
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "downloads"
    raw_dir = tmp_path / "raw"
    fixture_dir.mkdir()
    (fixture_dir / "sedec_solic.csv").write_text("data;bairro;descricao;situacao\n2022-05-25;Ibura;alagamento em canal;aberto\n", encoding="utf-8")
    (fixture_dir / "emlurb_2022.csv").write_text("data;endereco;SERVICO_DESCRICAO;SITUACAO\n25/05/2022;Rua A;boca de lobo obstruida;finalizado\n", encoding="utf-8")
    return {"REVP_RECIFE_DISCOVERY_FIXTURE": str(ckan), "REVP_RECIFE_DOWNLOAD_FIXTURE_DIR": str(fixture_dir), "REVP_RECIFE_RAW_DIR": str(raw_dir)}


def test_v1np_downloads_fixture_csvs_to_local_only_inventory(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(prepare_fixture(tmp_path))
    subprocess.run([sys.executable, str(DISCOVERY), "--force", "--emit-evidence"], cwd=ROOT, env=env, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(rows(MANIFEST)) == 2
    assert len(rows(INVENTORY)) == 2
    assert all(row["raw_storage_policy"] == "RAW_ONLY_LOCAL_RUNS_NOT_VERSIONED" for row in rows(INVENTORY))
    assert all(row["download_status"] == "DOWNLOAD_OK" for row in rows(MANIFEST))


def test_v1np_public_outputs_are_safe_and_schema_exists() -> None:
    assert "row_count" in {row["field"] for row in rows(SCHEMA)}
    for path in [MANIFEST, INVENTORY, FAILURES]:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABS_PATH.search(text)
        assert "can_train_model,true" not in text
