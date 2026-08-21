"""Testes do marco MV2-12 — Data Intake, Evidence Backlog & Download Readiness.

Cobrem as travas metodológicas obrigatórias:
- outputs_public/mv2_data_readiness só contém arquivos leves (sem bruto pesado);
- raster espectral nativo NUNCA é inferido de PNG/NPZ;
- download readiness não marca publish_to_outputs_public=true para bruto;
- nenhum dado crítico (P0/P1) é marcado resolvido sem path, fonte ou justificativa.

Os testes (re)geram os artefatos a partir dos scripts MV2-12, de modo a serem
determinísticos e independentes de execução prévia.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_readiness"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_12_build_download_readiness as download_mod  # noqa: E402
import mv2_12_build_missing_data_matrix as matrix_mod  # noqa: E402
import mv2_12_data_readiness_common as common  # noqa: E402
import mv2_12_event_geometry_backlog as event_mod  # noqa: E402
import mv2_12_scan_local_data_candidates as scan_mod  # noqa: E402
import mv2_12_sentinel_native_raster_backlog as raster_mod  # noqa: E402
import mv2_12_validate_no_heavy_public_outputs as validate_mod  # noqa: E402


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    common.ensure_output_dir()
    scan_mod.main()
    matrix_mod.main()
    download_mod.main()
    event_mod.main()
    raster_mod.main()


# --------------------------------------------------------------------------- #
# 1. outputs_public/mv2_data_readiness não contém bruto pesado
# --------------------------------------------------------------------------- #


def test_no_heavy_raw_in_public_outputs() -> None:
    violations = validate_mod.find_violations()
    assert violations == [], f"violações de bruto pesado: {violations}"


def test_validator_exit_code_zero() -> None:
    assert validate_mod.main() == 0


def test_only_light_extensions_present() -> None:
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() in validate_mod.ALLOWED_EXTENSIONS, path


# --------------------------------------------------------------------------- #
# 2. raster espectral nativo NUNCA inferido de PNG/NPZ
# --------------------------------------------------------------------------- #


def test_raster_backlog_does_not_infer_spectral_from_render() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_sentinel_native_raster_backlog.csv")
    assert rows, "backlog de raster vazio"
    for row in rows:
        has_render = row["has_png_render"] == "true" or row["has_npz_visual"] == "true"
        has_native = row["has_native_raster"] == "true"
        supports = row["can_support_spectral_baseline"] == "true"
        # Render visual jamais habilita baseline espectral.
        if has_render and not has_native:
            assert not supports, row["region"]
        # Baseline espectral exige raster nativo verificado.
        if supports:
            assert has_native, row["region"]


def test_scan_marks_png_npz_as_non_spectral() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_local_recovery_candidates.csv")
    for row in rows:
        if row["extension"] in {".png", ".jpg", ".jpeg", ".npz"}:
            # Nunca promovido a canônico e nunca tratado como espectral.
            assert row["can_promote_to_canonical"] == "false", row["candidate_path"]


def test_native_spectral_and_visual_extensions_disjoint() -> None:
    assert not (
        common.NATIVE_SPECTRAL_EXTENSIONS & common.VISUAL_RENDER_EXTENSIONS
    )
    assert ".png" not in common.NATIVE_SPECTRAL_EXTENSIONS
    assert ".npz" not in common.NATIVE_SPECTRAL_EXTENSIONS


# --------------------------------------------------------------------------- #
# 3. download readiness não publica bruto
# --------------------------------------------------------------------------- #


def test_download_readiness_never_publishes_raw() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_download_readiness.csv")
    assert rows, "download readiness vazio"
    for row in rows:
        assert row["publish_to_outputs_public"] == "false", row["source_name"]
        assert not row["download_target_dir"].startswith("outputs_public"), row["source_name"]


def test_download_targets_are_local_only() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_download_readiness.csv")
    allowed_prefixes = ("data_quarantine", "external_data_local", "data_raw_local")
    for row in rows:
        assert row["download_target_dir"].startswith(allowed_prefixes), row["source_name"]


# --------------------------------------------------------------------------- #
# 4. nenhum dado crítico marcado resolvido sem evidência
# --------------------------------------------------------------------------- #


def test_no_critical_resolved_without_evidence() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_missing_data_matrix.csv")
    assert rows
    for row in rows:
        if row["priority"] in {"P0", "P1"} and row["current_status"] == "PRESENTE_VALIDADO":
            assert row["expected_location"], row["missing_data_id"]
            assert row["local_search_result"], row["missing_data_id"]
            assert row["blocking_reason"] or row["notes"], row["missing_data_id"]


def test_status_and_purpose_in_taxonomy() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_missing_data_matrix.csv")
    for row in rows:
        assert row["current_status"] in common.STATUS_VALUES, row["missing_data_id"]
        for part in row["scientific_purpose"].split("|"):
            assert part in common.PURPOSE_VALUES, row["missing_data_id"]


def test_absence_not_treated_as_negative() -> None:
    """Ausência de evidência nunca habilita negativo formal; G6 fica review-only."""
    rows = _read(OUTPUT_DIR / "mv2_12_missing_data_matrix.csv")
    negatives = [r for r in rows if "NEGATIVO_FORMAL" in r["scientific_purpose"]]
    assert negatives, "esperava ao menos uma linha de requisito de negativo formal"
    for row in negatives:
        assert row["current_status"] in {"NAO_NECESSARIO_AGORA", "DESCONHECIDO"}, row["missing_data_id"]
        assert row["can_be_done_now"] == "false", row["missing_data_id"]


# --------------------------------------------------------------------------- #
# 5. eventos: nenhum promovido a G4/silver sem geometria observada
# --------------------------------------------------------------------------- #


def test_event_geometry_no_g4_without_footprint() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_event_geometry_backlog.csv")
    assert len(rows) == 9, "esperava 9 eventos candidatos do Protocolo C"
    for row in rows:
        if row["has_observed_footprint"] == "false":
            assert row["can_support_g4"] == "false", row["event_id"]
            assert row["can_support_silver"] == "false", row["event_id"]


def test_petropolis_kept_as_mass_movement() -> None:
    rows = _read(OUTPUT_DIR / "mv2_12_event_geometry_backlog.csv")
    for row in rows:
        if row["region"] == "Petropolis":
            assert row["phenomenon_type"] == "mass_movement", row["event_id"]
