"""Testes do marco MV2-13 — Sentinel Binding & Spectral Eligibility Gate.

Cobrem as travas fail-closed: binding fraco/none/conflito/inválido nunca autoriza
STAC real; dry-run não executa nem cria crop; Dia 10 não desbloqueia sem raster
Sentinel nativo; PNG/NPZ/render nunca é raster nativo; labels/silver/gold/negativos
permanecem zero; outputs públicos são leves; schemas carregam; etc.

Os artefatos são (re)gerados a partir do orquestrador, de forma determinística.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_sentinel_binding"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_13_run_sentinel_binding as runner  # noqa: E402
import mv2_13_sentinel_binding_common as common  # noqa: E402
import mv2_13_validate_sentinel_binding_guardrails as validator  # noqa: E402

REAL_OK = "STAC_REAL_FUTURE_ELIGIBLE"
BAD_BINDINGS = {"BINDING_WEAK", "BINDING_NONE", "BINDING_CONFLICT", "BINDING_INVALID"}


def _read(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    rc = runner.main()
    assert rc == 0, "orquestrador MV2-13 falhou"


# 1. schemas carregam e têm campos obrigatórios
def test_schemas_load_with_required_fields() -> None:
    names = [
        "schema_mv2_13_sentinel_binding.json",
        "schema_mv2_13_stac_gate.json",
        "schema_mv2_13_day10_spectral_gate.json",
        "schema_mv2_13_manual_recovery_queue.json",
        "schema_mv2_13_binding_risk_matrix.json",
    ]
    for n in names:
        data = json.loads((SCHEMA_DIR / n).read_text(encoding="utf-8"))
        assert data["required_fields"], n
        assert data["schema_id"], n


def test_binding_matrix_matches_schema_fields() -> None:
    schema = json.loads((SCHEMA_DIR / "schema_mv2_13_sentinel_binding.json").read_text(encoding="utf-8"))
    rows = _read("mv2_13_binding_matrix.csv")
    assert rows
    for field in schema["required_fields"]:
        assert field in rows[0], field


# 2. normalized anchors preservam total esperado
def test_normalized_anchors_total() -> None:
    rows = _read("mv2_13_normalized_anchors.csv")
    src = common.load_anchors()
    assert len(rows) == len(src), "divergência no total de âncoras normalizadas"


# 3. binding matrix existe e tem colunas obrigatórias
def test_binding_matrix_columns() -> None:
    rows = _read("mv2_13_binding_matrix.csv")
    for col in common.BINDING_COLUMNS:
        assert col in rows[0], col


# 4 + 5. weak/none/conflict/invalid nunca autorizam STAC real
def test_bad_bindings_never_stac_real() -> None:
    binding = _read("mv2_13_binding_matrix.csv")
    gate = {g["binding_id"]: g for g in _read("mv2_13_stac_gate_matrix.csv")}
    for b in binding:
        if b["binding_strength"] in BAD_BINDINGS:
            assert b["stac_status"] != REAL_OK, b["binding_id"]
            assert b["can_stac_real_next_step"] != "true", b["binding_id"]
            g = gate.get(b["binding_id"], {})
            assert g.get("can_stac_real_next_step") != "true", b["binding_id"]


# 6 + 7. STAC dry-run não executa download nem cria crop
def test_dry_run_never_executes() -> None:
    dry = _read("mv2_13_stac_dry_run_plan.csv")
    for r in dry:
        assert r["would_download"] == "false", r["dry_run_query_id"]
        assert r["would_create_crop"] == "false", r["dry_run_query_id"]
        assert r["execution_status"] == "NOT_EXECUTED_DRY_RUN_ONLY", r["dry_run_query_id"]


# 8. Dia 10 não desbloqueia sem raster nativo
def test_day10_not_unlocked_without_native_raster() -> None:
    day10 = _read("mv2_13_day10_spectral_gate.csv")
    for d in day10:
        if d["can_unlock_day10_now"] == "true":
            assert d["has_native_raster"] == "true", d["day10_gate_id"]


# 9. PNG/NPZ/render não contam como raster nativo
def test_render_never_native_raster() -> None:
    assets = _read("mv2_13_asset_inventory_normalized.csv")
    for a in assets:
        if a["is_native_sentinel_raster"] == "true":
            assert a["is_visual_render"] != "true", a["asset_id"]
            assert a["is_npz_visual"] != "true", a["asset_id"]
            assert a["is_dinov2_embedding"] != "true", a["asset_id"]


# 10 + 11 + 12. missing crs/geometry/patch/asset bloqueiam STAC real
def test_missing_fields_block_stac_real() -> None:
    gate = _read("mv2_13_stac_gate_matrix.csv")
    binding = {b["binding_id"]: b for b in _read("mv2_13_binding_matrix.csv")}
    for g in gate:
        if g["can_stac_real_next_step"] == "true":
            b = binding[g["binding_id"]]
            assert b["patch_id"] and b["asset_id"], g["binding_id"]
            assert g["geometry_ok"] == "true", g["binding_id"]
            assert b["crs_status"] != "ABSENT", g["binding_id"]
            assert b["scene_id"] or b["acquisition_datetime"], g["binding_id"]


# 13 + 14. labels/silver/gold/negatives zero; can_train false
def test_no_labels_silver_gold_negatives_no_train() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_13_binding_summary.json").read_text(encoding="utf-8"))
    for key in ["labels_created", "silver_created", "gold_created", "negatives_created",
                "crops_created", "spectral_features_created", "native_rasters_created",
                "stac_real_executed", "downloads_executed"]:
        assert s[key] == 0, key
    assert s["can_train"] is False
    assert s["sandbox_status"] == "bloqueado"


# 15. outputs_public não tem bruto pesado
def test_no_heavy_public_outputs() -> None:
    assert validator.find_violations() == []


# 16. risk matrix contém riscos obrigatórios
def test_risk_matrix_has_required_risks() -> None:
    rows = _read("mv2_13_binding_risk_matrix.csv")
    present = {r["risk_type"] for r in rows}
    for risk in common.RISK_TYPES:
        assert risk in present, risk


# 17. manual recovery queue contém ações para GEE/scene_id/patch_id/asset_id/geometry/CRS
def test_recovery_queue_covers_core_actions() -> None:
    rows = _read("mv2_13_manual_recovery_queue.csv")
    blob = " ".join((r["missing_field"] + " " + r["recovery_action"] + " " + r["expected_source"]).lower() for r in rows)
    for token in ["gee", "scene_id", "patch_id", "asset_id", "geometry", "crs", "datetime"]:
        assert token.lower() in blob, token


# 18. summary JSON coerente com CSVs
def test_summary_consistent_with_csvs() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_13_binding_summary.json").read_text(encoding="utf-8"))
    binding = _read("mv2_13_binding_matrix.csv")
    assert s["total_anchors"] == len(binding)
    total = (s["binding_strong"] + s["binding_medium"] + s["binding_weak"]
             + s["binding_none"] + s["binding_conflict"] + s["binding_invalid"])
    assert total == len(binding)


# 19. git diff --cached vazio (sem fragilidade: tolera ambiente sem git)
def test_no_staged_files() -> None:
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        pytest.skip("git indisponível")
    assert out == "", f"há arquivos staged: {out}"


# 20. nenhum caminho absoluto privado nos outputs
def test_no_private_absolute_paths() -> None:
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert "C:/Users" not in txt and "C:\\Users" not in txt, p.name


# Reforço: STAC real e dry-run eligible coerentes com a realidade fail-closed atual
def test_stac_real_zero_and_native_raster_zero() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_13_binding_summary.json").read_text(encoding="utf-8"))
    assert s["stac_real_future_eligible"] == 0
    assert s["native_rasters_present_in_corpus"] == 0
    assert s["can_unlock_day10_now"] is False
