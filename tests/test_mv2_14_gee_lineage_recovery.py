"""Testes do marco MV2-14 — GEE Scene Lineage Recovery & Temporal Metadata Binding.

Cobrem as travas fail-closed: nenhum candidate vira strong sem evidência; tile T23KPR
não auto-vincula a Petrópolis; datetime futuro inválido; região não basta; STAC/labels/
crops/features continuam 0; Dia 10 não desbloqueia sem raster nativo; template não vem
preenchido; schemas carregam; summary coerente.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_gee_lineage_recovery"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_14_gee_lineage_common as common  # noqa: E402
import mv2_14_run_gee_lineage_recovery as runner  # noqa: E402
import mv2_14_validate_gee_lineage_guardrails as validator  # noqa: E402

TEMPLATE_FILL = ["scene_id", "acquisition_datetime", "tile", "cloud_cover",
                 "gee_collection", "gee_export_task", "gee_script_path_or_name"]


def _read(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    rc = runner.main()
    assert rc == 0, "orquestrador MV2-14 falhou"


# 1. schemas existem e têm campos obrigatórios
def test_schemas_have_required_fields() -> None:
    for n in [
        "schema_mv2_14_lineage_candidates.json",
        "schema_mv2_14_lineage_binding_matrix.json",
        "schema_mv2_14_post_lineage_stac_gate.json",
        "schema_mv2_14_day10_post_lineage_gate.json",
        "schema_mv2_14_manual_lineage_template.json",
    ]:
        d = json.loads((SCHEMA_DIR / n).read_text(encoding="utf-8"))
        assert d["required_fields"] and d["schema_id"], n


# 2. medium binding index contém os 128 ou registra divergência
def test_medium_binding_index_count() -> None:
    rows = _read("mv2_14_medium_binding_index.csv")
    binding = common.read_csv_rows(common.MV2_13_BINDING_MATRIX)
    expected = sum(1 for b in binding if common.clean(b.get("binding_strength")) == "BINDING_MEDIUM")
    assert len(rows) == expected


# 3. nenhum candidate review vira strong sem evidência
def test_candidate_review_never_strong() -> None:
    lineage = _read("mv2_14_lineage_binding_matrix.csv")
    for lr in lineage:
        if lr["lineage_status"] == "LINEAGE_CANDIDATE_REVIEW":
            assert lr["recovered_scene_id"] == "", lr["lineage_binding_id"]
            assert lr["stac_readiness_after_lineage"] != "READY_FOR_STAC_REAL_REVIEW_REQUIRED"


# 4. tile T23KPR não auto-vincula cena a Petrópolis
def test_no_tile_auto_join() -> None:
    review = _read("mv2_14_strong_scene_anchor_review.csv")
    assert review, "esperava cenas para revisão"
    for r in review:
        assert r["auto_join_allowed"] == "false", r["scene_anchor_id"]
    s = json.loads((OUTPUT_DIR / "mv2_14_lineage_summary.json").read_text(encoding="utf-8"))
    assert s["auto_joins_performed"] == 0


# 5. datetime futuro é invalidado
def test_future_datetime_invalidated() -> None:
    lineage = _read("mv2_14_lineage_binding_matrix.csv")
    for lr in lineage:
        dt = lr["recovered_datetime"]
        if dt:
            assert common.classify_datetime(dt[:10])[1] != "INVALID_FUTURE" or lr["lineage_status"] == "LINEAGE_INVALID"


# 6. cidade/região não basta para lineage (no máximo CANDIDATE_REVIEW)
def test_region_only_is_review_not_recovered() -> None:
    lineage = _read("mv2_14_lineage_binding_matrix.csv")
    for lr in lineage:
        # Petrópolis sem cena por patch deve ser REVIEW, nunca RECOVERED.
        if lr["region"] == "Petropolis" and not lr["recovered_scene_id"]:
            assert lr["lineage_status"] in {"LINEAGE_CANDIDATE_REVIEW", "LINEAGE_NOT_FOUND"}, lr["lineage_binding_id"]


# 7. filename/texto fraco não basta para lineage forte
def test_text_candidates_low_confidence() -> None:
    cands = _read("mv2_14_lineage_candidates.csv")
    for c in cands:
        if c["extraction_method"] == "text_regex":
            assert c["extraction_confidence"] == "LOW", c["candidate_id"]


# 8-12. STAC/downloads/crops/features/labels continuam 0
def test_no_stac_no_artifacts() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_14_lineage_summary.json").read_text(encoding="utf-8"))
    for key in ["stac_real_executed", "downloads_executed", "crops_created",
                "native_rasters_created", "spectral_features_created", "labels_created",
                "silver_created", "gold_created", "negatives_created"]:
        assert s[key] == 0, key


# 13. can_train false
def test_can_train_false() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_14_lineage_summary.json").read_text(encoding="utf-8"))
    assert s["can_train"] is False


# 14. Dia 10 não desbloqueia sem raster nativo
def test_day10_not_unlocked_now() -> None:
    day10 = _read("mv2_14_day10_post_lineage_gate.csv")
    for d in day10:
        if d["can_unlock_day10_now"] == "true":
            assert d["has_native_raster"] == "true", d["day10_post_lineage_id"]
    s = json.loads((OUTPUT_DIR / "mv2_14_lineage_summary.json").read_text(encoding="utf-8"))
    assert s["can_unlock_day10_now"] is False


# 15. template manual não vem preenchido com lineage inventado
def test_template_not_prefilled() -> None:
    template = _read("mv2_14_manual_lineage_template.csv")
    assert template
    for t in template:
        for f in TEMPLATE_FILL:
            assert t[f] == "", f"{f} pré-preenchido em {t['patch_id']}"
        assert t["review_status"] == "PENDING_GEE_CHECK"


# 16. outputs públicos são leves
def test_outputs_light() -> None:
    assert validator.find_violations() == []


# 17. nenhum caminho absoluto privado nos outputs
def test_no_private_absolute_paths() -> None:
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert "C:/Users" not in txt and "C:\\Users" not in txt, p.name


# 18. risk matrix contém riscos obrigatórios
def test_risk_matrix_required_types() -> None:
    rows = _read("mv2_14_lineage_risk_matrix.csv")
    present = {r["risk_type"] for r in rows}
    for risk in common.RISK_TYPES:
        assert risk in present, risk


# 19. summary JSON coerente com CSVs
def test_summary_consistent() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_14_lineage_summary.json").read_text(encoding="utf-8"))
    lineage = _read("mv2_14_lineage_binding_matrix.csv")
    total = (s["lineage_recovered_strong"] + s["lineage_recovered_partial"]
             + s["lineage_candidate_review"] + s["lineage_conflict"]
             + s["lineage_invalid"] + s["lineage_not_found"])
    assert total == len(lineage) == s["total_medium_bindings_input"]


# 20. regressão MV2-13 disponível localmente ou registrada como indisponível
def test_mv2_13_regression_available_or_recorded() -> None:
    mv13 = PROJECT_ROOT / "tests" / "test_mv2_13_sentinel_binding.py"
    if not mv13.exists():
        pytest.skip("NOT_RUN_BRANCH_OR_ARTIFACT_UNAVAILABLE: teste MV2-13 ausente nesta tree")
    # presente: apenas confirma que o binding matrix do MV2-13 existe como insumo
    assert common.MV2_13_BINDING_MATRIX.exists()


# Reforço: estado fail-closed do bloco
def test_zero_strong_zero_dry_run_is_consistent() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_14_lineage_summary.json").read_text(encoding="utf-8"))
    if s["stac_dry_run_eligible_after_lineage"] > 0:
        # se houver dry-run, deve haver lineage forte correspondente
        assert s["lineage_recovered_strong"] > 0
