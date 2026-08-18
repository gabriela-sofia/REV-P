"""Tests for SUSC-17D validacao tecnica de evidencia observacional."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_17d_validacao_tecnica_evidencia_observacional"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight_validacao_tecnica.csv",
    OUT / "preflight_validacao_tecnica.json",
    OUT / "decisoes_validacao_tecnica.csv",
    OUT / "matriz_consistencia_features.csv",
    OUT / "resumo_por_status_validacao.csv",
    OUT / "resumo_por_regiao.csv",
    OUT / "bloqueios_validacao_tecnica.csv",
    OUT / "gate_prontidao_17b.csv",
    OUT / "susc_17d_validacao_tecnica_summary.json",
    REPORTS / "SUSC_17D_VALIDACAO_TECNICA_EVIDENCIA_OBSERVACIONAL.md",
    SCHEMAS / "susc_17d_validacao_tecnica_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17d_validacao_tecnica_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17d_validacao_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_schema()
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def base_decision(**overrides) -> dict:
    row = {
        "item_validacao_id": "S17D_TEST_0001",
        "candidate_event_id": "S17C_TEST",
        "source_id": "SRC_TEST",
        "source_type": "official_observed_event_polygon",
        "geometry_id": "GEOM_TEST",
        "patch_id": "PATCH_TEST",
        "cidade": "Recife",
        "regiao": "REC",
        "tipo_fenomeno": "inundacao",
        "data_evento": "2022-05-28",
        "classe_vinculo": "exact_polygon_overlap",
        "pontuacao_confiabilidade_fonte": "85",
        "pontuacao_qualidade_temporal": "90",
        "pontuacao_qualidade_geometrica": "85",
        "pontuacao_qualidade_vinculo_patch": "85",
        "pontuacao_compatibilidade_fenomeno": "85",
        "pontuacao_incerteza_espacial": "85",
        "pontuacao_consistencia_features": "70",
        "pontuacao_risco_vazamento_temporal": "80",
        "pontuacao_severidade_bloqueios": "85",
        "confianca_tecnica_final": "alta",
        "status_validacao_tecnica": "aceito_para_avaliacao_review_only",
        "accepted_for_evaluation": "true",
        "accepted_for_calibration_candidate": "false",
        "eligible_for_training": "false",
        "ground_truth": "false",
        "review_only": "true",
        "score_v7_allowed": "false",
        "not_ground_truth_reason": "validacao tecnica review-only; nao e verdade de referencia",
        "justificativa_tecnica": "fonte, data, geometria e vinculo fortes",
        "bloqueios_criticos": "nenhum",
    }
    row.update(overrides)
    return row


def test_build_validator_e_determinismo():
    result = run_script("build_susc_17d_validacao_tecnica_evidencia_observacional.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    cards_first = {path: path.read_bytes() for path in sorted((OUT / "cartoes_evidencia").glob("*.md"))}
    assert cards_first
    result = run_script("build_susc_17d_validacao_tecnica_evidencia_observacional.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    assert {path: path.read_bytes() for path in sorted((OUT / "cartoes_evidencia").glob("*.md"))} == cards_first
    result = run_script("validate_susc_17d_validacao_tecnica_evidencia_observacional.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_item_forte_aceito_para_avaliacao_review_only():
    decisions = read_csv(OUT / "decisoes_validacao_tecnica.csv")
    summary = json.loads((OUT / "susc_17d_validacao_tecnica_summary.json").read_text(encoding="utf-8"))
    accepted = [row for row in decisions if row["accepted_for_evaluation"] == "true"]
    assert len(decisions) == 5
    assert accepted
    assert {row["status_validacao_tecnica"] for row in accepted} == {"aceito_para_avaliacao_review_only"}
    assert {row["ground_truth"] for row in decisions} == {"false"}
    assert {row["eligible_for_training"] for row in decisions} == {"false"}
    assert {row["score_v7_allowed"] for row in decisions} == {"false"}
    assert summary["accepted_for_evaluation_count"] == 5
    assert summary["accepted_for_calibration_candidate_count"] == 0


def test_geometria_textual_nao_e_aceita():
    s17d = _load_common()
    link = {"patch_id": "PATCH", "geometry_id": "GEOM", "link_class": "exact_polygon_overlap", "uncertainty_m": "25"}
    geom = {"has_geometry_object": "false", "crs": "not_available", "geometry_type": "unresolved_geometry"}
    target = {"source_type": "official_observed_event_point", "event_date_precision": "exact_day", "phenomenon_type": "flood"}
    blockers = s17d._critical_blockers(link, geom, target, 80)
    assert "geometria_nao_resolvida" in blockers
    assert s17d._status(link, geom, target, blockers) == "geometria_insuficiente"


def test_alerta_e_area_de_risco_rejeitados():
    s17d = _load_common()
    link = {"patch_id": "PATCH", "geometry_id": "GEOM", "link_class": "exact_polygon_overlap", "uncertainty_m": "25"}
    geom = {"has_geometry_object": "true", "crs": "EPSG:4326", "geometry_type": "polygon"}
    for source_type in ["alert_only", "risk_area_not_event"]:
        target = {"source_type": source_type, "event_date_precision": "exact_day", "phenomenon_type": "flood"}
        blockers = s17d._critical_blockers(link, geom, target, 80)
        assert "classe_de_fonte_inadequada" in blockers
        assert s17d._status(link, geom, target, blockers) == "rejeitado"


def test_same_region_only_rejeitado_por_incompatibilidade_espacial():
    s17d = _load_common()
    link = {"patch_id": "PATCH", "geometry_id": "GEOM", "link_class": "same_region_only", "uncertainty_m": "not_available"}
    geom = {"has_geometry_object": "true", "crs": "EPSG:4326", "geometry_type": "polygon"}
    target = {"source_type": "official_observed_event_polygon", "event_date_precision": "exact_day", "phenomenon_type": "flood"}
    blockers = s17d._critical_blockers(link, geom, target, 80)
    assert "classe_de_vinculo_nao_forte" in blockers
    assert s17d._status(link, geom, target, blockers) == "incompatibilidade_espacial"


def test_calibracao_sem_avaliacao_falha():
    s17d = _load_common()
    row = base_decision(
        accepted_for_evaluation="false",
        accepted_for_calibration_candidate="true",
        status_validacao_tecnica="ambiguo",
    )
    assert any("calibracao_sem_avaliacao" in err for err in s17d.validate_decision_rows([row]))


def test_ground_truth_treino_score_v7_proibidos():
    s17d = _load_common()
    row = base_decision(ground_truth="true", eligible_for_training="true", score_v7_allowed="true")
    errors = s17d.validate_decision_rows([row])
    assert any("ground_truth_true_proibido" in err for err in errors)
    assert any("eligible_for_training_true_proibido" in err for err in errors)
    assert any("score_v7_allowed_true_proibido" in err for err in errors)


def test_justificativa_obrigatoria():
    s17d = _load_common()
    row = base_decision(justificativa_tecnica="")
    assert any("justificativa" in err for err in s17d.validate_decision_rows([row]))


def test_vazamento_temporal_critico_bloqueia_aceite():
    s17d = _load_common()
    row = base_decision(pontuacao_risco_vazamento_temporal="10")
    assert any("vazamento" in err for err in s17d.validate_decision_rows([row]))


def test_vocabulario_publico_proibido_falha():
    s17d = _load_common()
    hits = s17d.public_text_violations_text("texto publico com agente, LLM e IA")
    assert {"agente", "LLM", "IA"}.issubset(set(hits))
