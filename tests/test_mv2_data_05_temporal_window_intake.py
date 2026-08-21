"""Testes do marco MV2-DATA-05 — Temporal Window Intake & Validation (Vertente B).

Cobrem o intake/validação/gate de janela temporal: template vazio não promove; rejeição de
data futura, start>end, target inexistente, asset/patch mismatch, sem fonte, sem review,
janela ampla demais; GEE/STAC só com promoção; OData nunca; nenhuma API/download/crop/feature;
labels/silver/gold/negatives zero; Dia 10/sandbox bloqueados; risk/schemas/summary; validador;
regressões DATA-04/03/02/01.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_intake"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_05_common as common  # noqa: E402
import mv2_data_05_run_temporal_window_intake as runner  # noqa: E402
import mv2_data_05_validate_temporal_evidence as evidence  # noqa: E402
import mv2_data_05_validate_temporal_window_intake as validator  # noqa: E402


def _read(name: str) -> list[dict[str, str]]:
    with (PUBLIC_DIR / name).open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _summary() -> dict:
    return json.loads((PUBLIC_DIR / "mv2_data_05_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    rc = runner.main()
    assert rc == 0, "orquestrador MV2-DATA-05 falhou"


# --- helpers para classificação pura ---
def _corpus_sample() -> tuple[dict, dict]:
    corpus = common.corpus_index()
    tid = next(iter(corpus))
    return corpus, corpus[tid]


def _strong_row() -> dict:
    _, c = _corpus_sample()
    return {
        "target_id": c["target_id"], "asset_id": c["asset_id"], "patch_id": c["patch_id"],
        "region": c.get("region", ""), "temporal_window_start": "2022-02-01",
        "temporal_window_end": "2022-02-20", "temporal_window_source": "evento_defesa_civil",
        "source_ref": "doc://evento/2022", "review_status": "APPROVED",
    }


# 1. input discovery existe
def test_input_discovery_exists() -> None:
    assert _read("mv2_data_05_input_discovery.csv")


# 2. normalized temporal windows existe
def test_normalized_exists() -> None:
    assert _read("mv2_data_05_normalized_temporal_windows.csv")


# 3. template vazio não promove
def test_empty_template_no_promotion() -> None:
    s = _summary()
    assert s["filled_template_found"] is False
    assert s["temporal_promoted_strong"] == 0
    assert s["temporal_promoted_partial"] == 0
    for r in _read("mv2_data_05_temporal_promotion_gate.csv"):
        assert r["promotion_status"] == "TEMPORAL_WINDOW_BLOCKED_EMPTY"


# 4. target inexistente não promove
def test_unknown_target() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["target_id"] = "MV2_DATA_01_TGT_99999"
    res = evidence.evaluate(row, corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_INVALID"


# 5. asset mismatch não promove
def test_asset_mismatch() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["asset_id"] = "WRONG_ASSET"
    res = evidence.evaluate(row, corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_CONFLICT"


# 6. patch mismatch não promove
def test_patch_mismatch() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["patch_id"] = "WRONG_PATCH"
    res = evidence.evaluate(row, corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_CONFLICT"


# 7. data futura invalida
def test_future_date() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["temporal_window_start"] = "2030-01-01"
    row["temporal_window_end"] = "2030-01-10"
    res = evidence.evaluate(row, corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_INVALID"
    assert res["future_date"] == "true"


# 8. start > end invalida
def test_start_after_end() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["temporal_window_start"] = "2022-03-01"
    row["temporal_window_end"] = "2022-02-01"
    res = evidence.evaluate(row, corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_INVALID"
    assert res["start_after_end"] == "true"


# 9. source_ref ausente bloqueia strong
def test_source_missing_blocks_strong() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["temporal_window_source"] = ""
    row["source_ref"] = ""
    res = evidence.evaluate(row, corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_INVALID"


# 10. review_status ausente bloqueia strong (vira PARTIAL)
def test_review_missing_blocks_strong() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["review_status"] = "PENDING_HUMAN_FILL"
    res = evidence.evaluate(row, corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_VALID_PARTIAL"


# 11. janela ampla demais bloqueia strong
def test_too_broad_blocks_strong() -> None:
    corpus, _ = _corpus_sample()
    row = _strong_row()
    row["temporal_window_start"] = "2022-01-01"
    row["temporal_window_end"] = "2022-06-01"  # ~150 dias
    res = evidence.evaluate(row, corpus)
    assert res["too_broad"] == "true"
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_REVIEW_REQUIRED"


# extra: caminho STRONG funciona (sanidade da lógica positiva)
def test_strong_path_ok() -> None:
    corpus, _ = _corpus_sample()
    res = evidence.evaluate(_strong_row(), corpus)
    assert res["evidence_status"] == "TEMPORAL_EVIDENCE_VALID_STRONG"


# 12. GEE/STAC só habilitam com temporal strong/partial
def test_gee_stac_only_with_promotion() -> None:
    for r in _read("mv2_data_05_temporal_promotion_gate.csv"):
        promoted = r["promotion_status"] in {"TEMPORAL_WINDOW_PROMOTED_STRONG", "TEMPORAL_WINDOW_PROMOTED_PARTIAL"}
        if r["can_enable_gee_metadata_probe"] == "true" or r["can_enable_stac_metadata_probe"] == "true":
            assert promoted


# 13. OData não habilita sem product_id
def test_odata_never_enabled() -> None:
    for r in _read("mv2_data_05_temporal_promotion_gate.csv"):
        assert r["can_enable_odata_probe"] == "false"
    for r in _read("mv2_data_05_probe_ready_batch.csv"):
        assert r["can_probe_odata_metadata"] == "false"


# 14. nenhum GEE/STAC/OData executado
def test_no_api_executed() -> None:
    s = _summary()
    assert s["gee_calls_executed"] == 0
    assert s["stac_calls_executed"] == 0
    assert s["odata_calls_executed"] == 0


# 15. nenhum download/crop/raster/feature
def test_no_download_crop_raster_feature() -> None:
    s = _summary()
    for k in ["downloads_executed", "crops_created", "native_rasters_created", "spectral_features_created"]:
        assert s[k] == 0, k


# 16. labels/silver/gold/negatives continuam zero
def test_no_label_silver_gold_negative() -> None:
    s = _summary()
    for k in ["labels_created", "silver_created", "gold_created", "negatives_created"]:
        assert s[k] == 0, k


# 17. Dia 10 não desbloqueia
def test_day10_not_unlocked() -> None:
    assert _summary()["corpus_day10_unlocked"] is False
    for r in _read("mv2_data_05_temporal_promotion_gate.csv"):
        assert r["can_unlock_day10_now"] == "false"


# 18. sandbox não desbloqueia
def test_sandbox_not_unlocked() -> None:
    s = _summary()
    assert s["sandbox_unlocked"] is False
    assert s["can_train"] is False


# 19. risk matrix contém riscos obrigatórios
def test_risk_matrix_complete() -> None:
    present = {r["risk_type"] for r in _read("mv2_data_05_risk_matrix.csv")}
    for rt in common.RISK_TYPES:
        assert rt in present, rt


# 20. schemas carregam
def test_schemas_load() -> None:
    for n in [
        "schema_mv2_data_05_normalized_temporal_windows.json",
        "schema_mv2_data_05_temporal_evidence_validation.json",
        "schema_mv2_data_05_temporal_promotion_gate.json",
        "schema_mv2_data_05_probe_ready_batch.json",
    ]:
        d = json.loads((SCHEMA_DIR / n).read_text(encoding="utf-8"))
        assert d["schema_id"] and d["required_fields"]


# 21. summary coerente
def test_summary_coherent() -> None:
    s = _summary()
    assert s["total_input_rows"] == len(_read("mv2_data_05_normalized_temporal_windows.csv"))
    assert s["probe_ready_gee"] == sum(
        1 for r in _read("mv2_data_05_probe_ready_batch.csv") if r["can_probe_gee_metadata"] == "true"
    )


# 22. validador passa
def test_validator_passes() -> None:
    assert validator.main() == 0


# 23-26. regressões
def test_data04_regression() -> None:
    import mv2_data_04_validate_corpus_metadata_probe as v04
    assert v04.main() == 0


def test_data03_regression() -> None:
    import mv2_data_03_validate_live_api_canary as v03
    assert v03.main() == 0


def test_data02_regression() -> None:
    import mv2_data_02_validate_api_raster_harness as v02
    assert v02.main() == 0


def test_data01_regression() -> None:
    import mv2_data_01_validate_lineage_targets as v01
    assert v01.main() == 0
