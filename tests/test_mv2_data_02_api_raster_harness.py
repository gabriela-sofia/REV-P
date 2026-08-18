"""Testes do marco MV2-DATA-02 — Sentinel API & Raster Access Harness (Vertente B).

Cobrem as travas fail-closed FORTES: config default bloqueada; clientes não executam por
padrão; OData não baixa; lineage não promove sem metadata; canary bloqueado mesmo com
lineage forte; executor exige env var; bulk download sempre bloqueado; nenhum raster/segredo/
caminho privado em outputs_public; Dia 10 não desbloqueia sem raster privado validado;
labels/silver/gold/negatives/treino zero; risk matrix completa; schemas carregam; summary
coerente; validador passa; regressão DATA-01 passa.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_api_raster_harness"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_02_api_config as config  # noqa: E402
import mv2_data_02_common as common  # noqa: E402
import mv2_data_02_run_api_raster_harness as runner  # noqa: E402
import mv2_data_02_validate_api_raster_harness as validator  # noqa: E402


def _read(name: str) -> list[dict[str, str]]:
    with (PUBLIC_DIR / name).open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _summary() -> dict:
    return json.loads((PUBLIC_DIR / "mv2_data_02_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    # Garante ambiente fail-closed (sem env var de canary / config privada).
    os.environ.pop(config.CANARY_CONFIRM_ENV_VAR, None)
    os.environ.pop(config.CONFIG_ENV_VAR, None)
    rc = runner.main()
    assert rc == 0, "orquestrador MV2-DATA-02 falhou"


# 1. config default é fail-closed
def test_config_default_fail_closed() -> None:
    cfg = config.load_config()
    for k in ["gee_enabled", "cdse_stac_enabled", "cdse_odata_enabled", "allow_network",
              "allow_metadata_calls", "allow_raster_download", "allow_canary_download"]:
        assert cfg[k] is False, k


# 2. provider registry existe
def test_provider_registry_exists() -> None:
    rows = _read("mv2_data_02_api_provider_registry.csv")
    ids = {r["provider_id"] for r in rows}
    assert {"GEE", "CDSE_STAC", "CDSE_ODATA"} <= ids
    # nenhum segredo: credential_env_vars é só nome de env var
    for r in rows:
        assert "eyJ" not in r["credential_env_vars"]


# 3. STAC client não executa HTTP por padrão
def test_stac_not_executed_by_default() -> None:
    rows = _read("mv2_data_02_cdse_stac_metadata_results.csv")
    assert rows
    assert all(r["executed"] == "false" for r in rows)
    assert all(r["metadata_status"] == "NOT_EXECUTED_CONFIG_BLOCKED" for r in rows)


# 4. OData client não baixa por padrão
def test_odata_no_download_by_default() -> None:
    rows = _read("mv2_data_02_cdse_odata_metadata_results.csv")
    assert rows
    assert all(r["download_executed"] == "false" for r in rows)


# 5. GEE client não executa por padrão
def test_gee_not_executed_by_default() -> None:
    rows = _read("mv2_data_02_gee_metadata_results.csv")
    assert rows
    assert all(r["executed"] == "false" for r in rows)


# 6. lineage resolution não promove sem metadata
def test_lineage_not_promoted_without_metadata() -> None:
    rows = _read("mv2_data_02_api_lineage_resolution.csv")
    assert all(r["lineage_status"] == "API_LINEAGE_NOT_EXECUTED" for r in rows)
    assert all(r["can_promote_to_raster_canary"] == "false" for r in rows)
    assert all(r["can_promote_to_stac_dry_run"] == "false" for r in rows)


# 7. canary plan fica bloqueado sem config (mesmo com lineage forte de registry)
def test_canary_plan_blocked_by_default() -> None:
    rows = _read("mv2_data_02_raster_canary_plan.csv")
    assert len(rows) <= 3
    for r in rows:
        assert r["download_allowed_now"] == "false"
        assert r["crop_allowed_now"] == "false"
        assert r["canary_allowed_by_config"] == "false"
        assert r["private_output_dir"]


# 8. canary executor não executa sem env var
def test_canary_executor_not_executed_without_env() -> None:
    rows = _read("mv2_data_02_raster_canary_execution_manifest.csv")
    for r in rows:
        assert r["executed"] == "false"
        assert r["execution_status"] == "NOT_EXECUTED_GUARDRAIL"
        assert "env_REV_P_ALLOW_RASTER_CANARY!=YES" in r["unmet_requirements"]


# 9. bulk download sempre bloqueado
def test_bulk_download_always_blocked() -> None:
    rows = _read("mv2_data_02_api_lineage_resolution.csv")
    assert all(r["can_promote_to_bulk_download"] == "false" for r in rows)


# 10. nenhum raster em outputs_public
def test_no_raster_in_public() -> None:
    raster_ext = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog", ".npy", ".npz",
                  ".png", ".jpg", ".zip", ".nc", ".hdf"}
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file():
            assert p.suffix.lower() not in raster_ext, p.name


# 11. nenhum segredo em outputs_public
def test_no_secret_in_public() -> None:
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert "eyJ" not in txt
            assert "BEGIN RSA PRIVATE KEY" not in txt
            assert "BEGIN PRIVATE KEY" not in txt


# 12. nenhum caminho privado absoluto em outputs_public
def test_no_private_absolute_path() -> None:
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert "C:/Users" not in txt and "C:\\Users" not in txt, p.name


# 13. private raster validator aceita NO_RASTER_EXECUTED
def test_private_validator_accepts_no_raster() -> None:
    rows = _read("mv2_data_02_private_raster_validation.csv")
    assert rows
    assert all(r["validation_status"] == "NO_RASTER_EXECUTED" for r in rows)
    assert all(r["public_leak_detected"] == "false" for r in rows)


# 14. Dia 10 não desbloqueia sem raster validado
def test_day10_not_unlocked() -> None:
    s = _summary()
    assert s["can_unlock_day10_now"] is False
    assert s["private_rasters_validated"] == 0


# 15. labels/silver/gold/negatives continuam zero
def test_no_label_silver_gold_negative() -> None:
    s = _summary()
    for k in ["labels_created", "silver_created", "gold_created", "negatives_created",
              "crops_created", "native_rasters_created", "spectral_features_created",
              "public_raster_leaks", "downloads_executed"]:
        assert s[k] == 0, k


# 16. can_train=false
def test_can_train_false() -> None:
    s = _summary()
    assert s["can_train"] is False
    assert s["sandbox_status"] == "bloqueado"


# 17. risk matrix contém riscos obrigatórios
def test_risk_matrix_complete() -> None:
    present = {r["risk_type"] for r in _read("mv2_data_02_api_raster_risk_matrix.csv")}
    for rt in common.RISK_TYPES:
        assert rt in present, rt


# 18. schemas carregam
def test_schemas_load() -> None:
    for n in [
        "schema_mv2_data_02_api_provider_registry.json",
        "schema_mv2_data_02_api_lineage_resolution.json",
        "schema_mv2_data_02_raster_canary_plan.json",
        "schema_mv2_data_02_raster_canary_execution_manifest.json",
        "schema_mv2_data_02_private_raster_validation.json",
    ]:
        d = json.loads((SCHEMA_DIR / n).read_text(encoding="utf-8"))
        assert d["schema_id"] and d["required_fields"]


# 19. summary JSON coerente
def test_summary_coherent() -> None:
    s = _summary()
    assert s["total_targets"] == len(_read("mv2_data_02_api_lineage_resolution.csv"))
    assert s["providers_configured"] == len(_read("mv2_data_02_api_provider_registry.csv"))
    assert s["raster_canary_candidates"] == len(_read("mv2_data_02_raster_canary_plan.csv"))
    assert s["gee_metadata_calls_executed"] == 0
    assert s["stac_metadata_calls_executed"] == 0
    assert s["odata_metadata_calls_executed"] == 0


# 20. validador passa
def test_validator_passes() -> None:
    assert validator.main() == 0


# 21. regressão DATA-01 passa
def test_data01_regression() -> None:
    import mv2_data_01_validate_lineage_targets as v01
    assert v01.main() == 0


# extra: config de exemplo público não contém segredo
def test_example_config_no_secret() -> None:
    path = PUBLIC_DIR / "mv2_data_02_api_config.example.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["allow_network"] is False
    assert d["allow_raster_download"] is False
    # só o NOME da env var de credencial, nunca o segredo
    assert d["cdse_token_env_var"] == "CDSE_ACCESS_TOKEN"
    txt = path.read_text(encoding="utf-8")
    assert "eyJ" not in txt
