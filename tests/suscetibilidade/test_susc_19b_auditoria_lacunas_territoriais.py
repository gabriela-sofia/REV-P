"""Testes para SUSC-19B auditoria e preenchimento de lacunas territoriais."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_19b_auditoria_lacunas_territoriais"


def _load_common():
    path = SCRIPTS / "susc_19b_lacunas_territoriais_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s19b_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def test_auditoria_le_os_300_patches_do_19a():
    aud = _read(S.AUDITORIA_MISS)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert len(aud) == summary["total_patches"]
    assert summary["total_patches"] == 300


def test_missingness_territorial_detectado():
    aud = _read(S.AUDITORIA_MISS)
    assert all("faltam" in r["status_missingness"] for r in aud)
    assert all(r["MapBiomas_class_majority"] == "NA" for r in aud)


def test_feature_sem_fonte_nao_pode_ser_marcada_presente():
    for r in _read(S.MAT_EXTRAIDA):
        if r["landcover_source"] in ("not_available", "", "ausente"):
            for col in ("water_prop_19b", "exposed_soil_prop_19b", "impervious_proxy_19b", "MapBiomas_class_majority", "MapBiomas_class_distribution"):
                assert r[col] in ("NA", "", "not_available"), f"{r['patch_id']}:{col} presente sem fonte"


def test_extracao_local_so_preenche_com_fonte():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if not summary["fonte_local_utilizavel"]:
        assert summary["territorial_features_preenchidas_localmente"] == 0


def test_ausencia_de_fonte_local_gera_pacote_gee():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if not summary["fonte_local_utilizavel"]:
        assert S.GEE_JS.exists()
        assert S.GEE_MD.exists()
        assert S.GEE_MANIFEST.exists()
        assert S.GEE_SCHEMA.exists()


def test_pacote_gee_tem_manifesto_e_schema():
    manifest = _read(S.GEE_MANIFEST)
    assert len(manifest) == 300
    assert all("patch_id" in r and "xmin" in r for r in manifest)
    schema = json.loads(S.GEE_SCHEMA.read_text(encoding="utf-8"))
    assert "patch_id" in schema["required"]
    assert schema["properties"]["review_only"]["const"] is True


def test_score_v6_intacto():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


def test_ground_truth_trainable_score_v7_proibidos():
    for r in _read(S.MAT_19B):
        assert r["ground_truth"] == "false"
        assert r["eligible_for_training"] == "false"
        assert r["score_v7_allowed"] == "false"
        assert r["score_oficial"] == "false"
        assert r["substituir_score_v6"] == "false"
        assert r["usar_em_treino"] == "false"


def test_output_19b_versionavel():
    ignored = S.git_ignored([OUT / "summary.json", S.MAT_19B, S.GEE_JS])
    assert S.rel(OUT / "summary.json") not in ignored
    assert S.rel(S.GEE_JS) not in ignored


def test_cobertura_territorial_comparada():
    cmp = _read(S.COMPARACAO)
    assert cmp
    for r in cmp:
        assert "coverage_territorial_19a" in r
        assert "coverage_territorial_19b" in r


def test_vocabulario_publico_proibido_falha():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo}")


def test_saidas_publicas_sem_vocabulario_proibido():
    assert S.validate_public_text() == []


def test_validator_completo_passa():
    summary = S.build_all()
    assert S.validate_outputs(summary) == []
