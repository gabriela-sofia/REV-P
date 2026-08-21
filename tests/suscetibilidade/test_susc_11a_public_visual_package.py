"""SUSC-11A public visual package tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
PKG = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11A_public_visual_package"
MANIFEST = PKG / "SUSC_11A_visual_package_manifest.json"
README = PKG / "README.md"

REQUIRED_TABLES = ["top_patches_for_tcc.csv", "score_v6_by_region.csv", "cases_for_tcc_with_caution.csv",
                   "limitations_table.csv", "methodology_artifacts_index.csv"]


def test_structure_exists():
    for sub in ("figures", "tables", "geojson", "slides_content"):
        assert (PKG / sub).exists(), sub
    assert README.exists()
    assert MANIFEST.exists()


def test_nine_figures():
    assert len(list((PKG / "figures").glob("*.svg"))) >= 9


def test_required_tables():
    for t in REQUIRED_TABLES:
        assert (PKG / "tables" / t).exists(), t


def test_eight_slides():
    slides = sorted(p.name for p in (PKG / "slides_content").glob("*.md"))
    assert len(slides) >= 8


def test_figures_are_svg_with_footer():
    for f in (PKG / "figures").glob("*.svg"):
        txt = f.read_text(encoding="utf-8")
        assert txt.startswith("<svg")
        assert "review-only" in txt


def test_manifest_governance():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["can_be_used_as_ground_truth"] is False
    assert m["allowed_for_training"] is False
    assert m["review_only"] is True


def test_readme_mandatory():
    assert "As figuras não representam ground truth" in README.read_text(encoding="utf-8")
