"""Testes do marco MV2-DATA-04 — Corpus Metadata Probe (Vertente B).

Cobrem o probe metadata-only do corpus: batch ≤15 balanceado sem canary; sem janela
temporal GEE/STAC bloqueiam; sem product_id OData bloqueia; nenhuma query aberta;
lineage não promove sem metadata; nenhum download/crop/raster/feature; labels/silver/gold/
negatives zero; Dia 10/sandbox não desbloqueiam; template existe; risk matrix/schemas/summary
coerentes; validador passa; regressões DATA-03/02/01.
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
PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_corpus_metadata_probe"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_04_common as common  # noqa: E402
import mv2_data_04_run_corpus_metadata_probe as runner  # noqa: E402
import mv2_data_04_validate_corpus_metadata_probe as validator  # noqa: E402


def _read(name: str) -> list[dict[str, str]]:
    with (PUBLIC_DIR / name).open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _summary() -> dict:
    return json.loads((PUBLIC_DIR / "mv2_data_04_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    for var in [common.ENV_ALLOW_NETWORK, common.ENV_ALLOW_METADATA_CALLS]:
        os.environ.pop(var, None)
    rc = runner.main()
    assert rc == 0, "orquestrador MV2-DATA-04 falhou"


# 1. batch existe
def test_batch_exists() -> None:
    assert _read("mv2_data_04_corpus_metadata_batch.csv")


# 2. batch <= 15
def test_batch_max_15() -> None:
    assert len(_read("mv2_data_04_corpus_metadata_batch.csv")) <= 15


# 3. batch balanceado quando possível (<=5 por região)
def test_batch_balanced() -> None:
    from collections import Counter
    dist = Counter(r["region"] for r in _read("mv2_data_04_corpus_metadata_batch.csv"))
    for region, cnt in dist.items():
        assert cnt <= 5, (region, cnt)


# 4. batch não inclui official canary
def test_batch_no_canary() -> None:
    canary = common.official_canary_ids()
    for r in _read("mv2_data_04_corpus_metadata_batch.csv"):
        assert r["patch_id"] not in canary
        assert r["target_id"] not in canary
        assert r["is_official_canary"] == "false"


# 5. sem temporal window, GEE bloqueia
def test_gee_blocked_no_temporal() -> None:
    rows = _read("mv2_data_04_gee_corpus_metadata_probe.csv")
    assert rows
    for r in rows:
        if r["temporal_window_status"] != "PRESENT":
            assert r["metadata_status"] == "BLOCKED_NO_TEMPORAL_WINDOW"
            assert r["executed"] == "false"


# 6. sem temporal window, STAC bloqueia
def test_stac_blocked_no_temporal() -> None:
    rows = _read("mv2_data_04_cdse_stac_corpus_metadata_probe.csv")
    assert rows
    for r in rows:
        if r["temporal_window_status"] != "PRESENT":
            assert r["metadata_status"] == "BLOCKED_NO_TEMPORAL_WINDOW"
            assert r["executed"] == "false"


# 7. sem product_id, OData bloqueia
def test_odata_blocked_no_product() -> None:
    rows = _read("mv2_data_04_cdse_odata_metadata_probe.csv")
    assert rows
    for r in rows:
        assert r["metadata_status"] == "BLOCKED_NO_PRODUCT_ID"
        assert r["executed"] == "false"


# 8. nenhuma query aberta ampla (sem janela => bloqueio, nunca execução)
def test_no_open_ended_query() -> None:
    for name in ["mv2_data_04_gee_corpus_metadata_probe.csv",
                 "mv2_data_04_cdse_stac_corpus_metadata_probe.csv"]:
        for r in _read(name):
            if r["temporal_window_status"] != "PRESENT":
                assert r["executed"] == "false"


# 9. corpus lineage não promove sem metadata
def test_lineage_not_promoted() -> None:
    rows = _read("mv2_data_04_corpus_api_lineage_resolution.csv")
    assert rows
    for r in rows:
        assert r["lineage_status"] == "CORPUS_LINEAGE_BLOCKED_NO_TEMPORAL_WINDOW"
        assert r["can_stac_dry_run"] == "false"
        assert r["can_unlock_day10_now"] == "false"


# 10-13. nenhum download/crop/raster/feature
def test_no_download_crop_raster_feature() -> None:
    s = _summary()
    for k in ["downloads_executed", "crops_created", "native_rasters_created",
              "spectral_features_created"]:
        assert s[k] == 0, k


# 14. labels/silver/gold/negatives continuam zero
def test_no_label_silver_gold_negative() -> None:
    s = _summary()
    for k in ["labels_created", "silver_created", "gold_created", "negatives_created"]:
        assert s[k] == 0, k


# 15. Dia 10 não desbloqueia
def test_day10_not_unlocked() -> None:
    assert _summary()["corpus_day10_unlocked"] is False


# 16. sandbox não desbloqueia
def test_sandbox_not_unlocked() -> None:
    s = _summary()
    assert s["sandbox_unlocked"] is False
    assert s["can_train"] is False


# 17. temporal window template existe
def test_temporal_template_exists() -> None:
    rows = _read("mv2_data_04_temporal_window_template.csv")
    assert rows
    # sem evidência, campos temporais vazios
    for r in rows:
        assert not r["temporal_window_start"].strip()
        assert not r["temporal_window_end"].strip()


# 18. risk matrix contém riscos obrigatórios
def test_risk_matrix_complete() -> None:
    present = {r["risk_type"] for r in _read("mv2_data_04_risk_matrix.csv")}
    for rt in common.RISK_TYPES:
        assert rt in present, rt


# 19. schemas carregam
def test_schemas_load() -> None:
    for n in [
        "schema_mv2_data_04_corpus_metadata_batch.json",
        "schema_mv2_data_04_corpus_api_lineage_resolution.json",
        "schema_mv2_data_04_temporal_window_gap_matrix.json",
        "schema_mv2_data_04_temporal_window_template.json",
    ]:
        d = json.loads((SCHEMA_DIR / n).read_text(encoding="utf-8"))
        assert d["schema_id"] and d["required_fields"]


# 20. summary coerente
def test_summary_coherent() -> None:
    s = _summary()
    batch = _read("mv2_data_04_corpus_metadata_batch.csv")
    assert s["batch_targets"] == len(batch)
    assert s["batch_targets"] <= 15
    assert s["total_targets"] == 128
    assert s["gee_probes_executed"] == 0
    assert s["blocked_no_temporal_window"] == sum(
        1 for r in _read("mv2_data_04_gee_corpus_metadata_probe.csv")
        if r["metadata_status"] == "BLOCKED_NO_TEMPORAL_WINDOW"
    )


# 21. validador passa
def test_validator_passes() -> None:
    assert validator.main() == 0


# 22. regressão DATA-03 passa
def test_data03_regression() -> None:
    import mv2_data_03_validate_live_api_canary as v03
    assert v03.main() == 0


# 23. regressão DATA-02 passa
def test_data02_regression() -> None:
    import mv2_data_02_validate_api_raster_harness as v02
    assert v02.main() == 0


# 24. regressão DATA-01 passa
def test_data01_regression() -> None:
    import mv2_data_01_validate_lineage_targets as v01
    assert v01.main() == 0


# extra: nenhum raster/segredo/caminho privado em outputs_public
def test_outputs_clean() -> None:
    raster = {".tif", ".tiff", ".jp2", ".safe", ".npz", ".npy", ".png", ".jpg", ".zip"}
    for p in PUBLIC_DIR.rglob("*"):
        if p.is_file():
            assert p.suffix.lower() not in raster, p.name
            if p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                assert "C:/Users" not in txt and "C:\\Users" not in txt
                assert "eyJ" not in txt
