"""Testes do marco MV2-DATA-03 — Live Metadata Probe & Private Raster Canary (Vertente B).

Cobrem a transição controlada para API/canary real: preflight existe; candidates oficiais
(não corpus) com can_use_for_corpus_day10=false; probes/canary não executam sem config/env;
no máximo 1 canary; nenhum corpus target baixado; nenhum raster/segredo/caminho privado em
outputs_public; Dia 10/sandbox do corpus não desbloqueiam; risk/impact matrices completas;
schemas carregam; summary coerente; validador passa; regressões DATA-02/DATA-01.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_live_api_canary"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_03_common as common  # noqa: E402
import mv2_data_03_run_live_api_canary as runner  # noqa: E402
import mv2_data_03_validate_live_api_canary as validator  # noqa: E402


def _read(name: str) -> list[dict[str, str]]:
    with (PUBLIC_DIR / name).open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _summary() -> dict:
    return json.loads((PUBLIC_DIR / "mv2_data_03_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    # Ambiente fail-closed: sem env vars de permissão/canary.
    for var in [common.ENV_ALLOW_NETWORK, common.ENV_ALLOW_METADATA_CALLS,
                common.ENV_ALLOW_RASTER_DOWNLOAD, common.ENV_ALLOW_RASTER_CANARY]:
        os.environ.pop(var, None)
    rc = runner.main()
    assert rc == 0, "orquestrador MV2-DATA-03 falhou"


# 1. config/env preflight existe
def test_preflight_exists() -> None:
    rows = _read("mv2_data_03_config_env_preflight.csv")
    assert rows
    # secret_value_logged sempre false
    assert all(r["secret_value_logged"] == "false" for r in rows)


# 2. official canary candidates existem
def test_candidates_exist() -> None:
    rows = _read("mv2_data_03_official_canary_candidates.csv")
    assert 1 <= len(rows) <= 2


# 3. canary candidates não são corpus targets
def test_candidates_not_corpus() -> None:
    corpus = common.corpus_patch_ids()
    for r in _read("mv2_data_03_official_canary_candidates.csv"):
        assert r["official_refpatch_id"] not in corpus
        assert not r["corpus_patch_id"].strip()
        assert r["canary_role"] == "TECHNICAL_CANARY_ONLY"


# 4. can_use_for_corpus_day10=false
def test_can_use_for_corpus_false() -> None:
    for r in _read("mv2_data_03_official_canary_candidates.csv"):
        assert r["can_use_for_corpus_day10"] == "false"
    for r in _read("mv2_data_03_canary_metadata_consensus.csv"):
        assert r["can_use_for_corpus_day10"] == "false"


# 5. probes não executam sem config
def test_probes_not_executed_without_config() -> None:
    for name in ["mv2_data_03_live_gee_metadata_probe.csv",
                 "mv2_data_03_live_cdse_stac_metadata_probe.csv",
                 "mv2_data_03_live_cdse_odata_metadata_probe.csv"]:
        rows = _read(name)
        assert rows
        assert all(r["executed"] == "false" for r in rows)


# 6. consensus registry-only funciona
def test_consensus_registry_only() -> None:
    rows = _read("mv2_data_03_canary_metadata_consensus.csv")
    assert rows
    assert all(r["consensus_status"] == "REGISTRY_ONLY_STRONG" for r in rows)
    assert all(r["canary_download_allowed"] == "false" for r in rows)


# 7. canary executor não executa sem env var
def test_canary_executor_not_executed() -> None:
    rows = _read("mv2_data_03_private_raster_canary_execution_manifest.csv")
    for r in rows:
        assert r["executed"] == "false"
        assert r["execution_status"] == "NOT_EXECUTED_GUARDRAIL"
        assert "REV_P_ALLOW_RASTER_CANARY!=YES" in r["blocking_reason"]


# 8. no máximo 1 canary pode executar
def test_max_one_canary() -> None:
    rows = _read("mv2_data_03_private_raster_canary_execution_manifest.csv")
    assert len(rows) <= 1
    assert sum(1 for r in rows if r["executed"] == "true") <= 1


# 9. nenhum corpus target é baixado
def test_no_corpus_download() -> None:
    s = _summary()
    assert s["corpus_targets_downloaded"] == 0


# 10. nenhum raster em outputs_public
def test_no_raster_public() -> None:
    raster_ext = {".tif", ".tiff", ".jp2", ".safe", ".npy", ".npz", ".png", ".jpg", ".zip", ".nc", ".hdf"}
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file():
            assert p.suffix.lower() not in raster_ext, p.name


# 11. private path não vaza
def test_no_private_path_leak() -> None:
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert "C:/Users" not in txt and "C:\\Users" not in txt, p.name


# 12. segredo não vaza
def test_no_secret_leak() -> None:
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert "eyJ" not in txt
            assert "BEGIN PRIVATE KEY" not in txt


# 13. Dia 10 corpus não desbloqueia
def test_day10_not_unlocked() -> None:
    assert _summary()["corpus_day10_unlocked"] is False


# 14. sandbox não desbloqueia
def test_sandbox_not_unlocked() -> None:
    assert _summary()["sandbox_unlocked"] is False


# 15. labels/silver/gold/negatives continuam zero
def test_no_label_silver_gold_negative() -> None:
    s = _summary()
    for k in ["labels_created", "silver_created", "gold_created", "negatives_created",
              "spectral_features_created", "crops_created", "native_rasters_created",
              "public_raster_leaks"]:
        assert s[k] == 0, k
    assert s["can_train"] is False


# 16. risk matrix contém riscos obrigatórios
def test_risk_matrix_complete() -> None:
    present = {r["risk_type"] for r in _read("mv2_data_03_live_api_canary_risk_matrix.csv")}
    for rt in common.RISK_TYPES:
        assert rt in present, rt


# 17. corpus impact matrix contém linhas obrigatórias
def test_impact_matrix_rows() -> None:
    comps = {r["component"] for r in _read("mv2_data_03_corpus_impact_matrix.csv")}
    for c in ["API auth validated", "metadata probe validated", "raster private path validated",
              "checksum validation", "band/SCL validation", "corpus lineage still missing",
              "Dia 10 still blocked", "sandbox still blocked"]:
        assert c in comps, c
    for r in _read("mv2_data_03_corpus_impact_matrix.csv"):
        assert r["affects_128_targets"] == "false"
        assert r["affects_day10"] == "false"
        assert r["affects_sandbox"] == "false"


# 18. schemas carregam
def test_schemas_load() -> None:
    for n in [
        "schema_mv2_data_03_official_canary_candidates.json",
        "schema_mv2_data_03_canary_metadata_consensus.json",
        "schema_mv2_data_03_private_raster_canary_execution_manifest.json",
        "schema_mv2_data_03_private_raster_canary_validation.json",
        "schema_mv2_data_03_corpus_impact_matrix.json",
    ]:
        d = json.loads((SCHEMA_DIR / n).read_text(encoding="utf-8"))
        assert d["schema_id"] and d["required_fields"]


# 19. summary JSON coerente
def test_summary_coherent() -> None:
    s = _summary()
    assert s["official_canary_candidates"] == len(_read("mv2_data_03_official_canary_candidates.csv"))
    assert s["registry_only_strong"] == sum(
        1 for r in _read("mv2_data_03_canary_metadata_consensus.csv")
        if r["consensus_status"] == "REGISTRY_ONLY_STRONG"
    )
    assert s["config_found"] in (True, False)
    assert s["gee_metadata_probes_executed"] == 0


# 20. validador passa
def test_validator_passes() -> None:
    assert validator.main() == 0


# 21. regressão DATA-02 passa
def test_data02_regression() -> None:
    import mv2_data_02_validate_api_raster_harness as v02
    assert v02.main() == 0


# 22. regressão DATA-01 passa
def test_data01_regression() -> None:
    import mv2_data_01_validate_lineage_targets as v01
    assert v01.main() == 0


# extra: private raster validation aceita NO_RASTER_EXECUTED_OK
def test_private_validation_no_raster_ok() -> None:
    rows = _read("mv2_data_03_private_raster_canary_validation.csv")
    assert rows
    assert all(r["validation_status"] == "NO_RASTER_EXECUTED_OK" for r in rows)
