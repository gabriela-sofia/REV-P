"""Testes para SUSC-GT08 - Preparacao Integrada de AOI e Busca Sentinel-1."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
SUSC = ROOT / "outputs_public" / "suscetibilidade"


def _load(name, filename):
    path = SCRIPTS / filename
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load("gt08_common", "susc_gt08_sentinel1_readiness_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


# janela dos canarios (2022-05-24): pre 04-24..05-23, post 05-24..05-31
PRE_WIN = ("2022-04-24", "2022-05-23")
POST_WIN = ("2022-05-24", "2022-05-31")
AOI = (-34.9, -8.1, -34.88, -8.08)  # bbox de patch em Recife


def _scene(**over):
    base = dict(scene_id="S1A_IW_GRDH_1SDV_20220516T075410_x", product_id="PID",
                acquisition_date="2022-05-16", product_type="GRD_HD", acquisition_mode="IW",
                orbit_path="111", orbit_frame="621", orbit_direction="DESCENDING",
                scene_bbox=(-35.4, -9.8, -32.8, -7.8))
    base.update(over)
    return base


# 1. leitura GT07 + 5 canarios -----------------------------------------------
def test_le_gt07_5_canarios():
    fila = _read(SUSC / "susc_gt07_fila_qa_canarios.csv")
    alvos = _read(S.TARGETS_CSV)
    assert len(alvos) == len(fila) == 5
    assert alvos[0]["sentinel1_target_id"].startswith("S1_")


# 2. resolucao AOI forte/parcial/ausente -------------------------------------
def test_aoi_forte():
    r = S.resolve_aoi({"bbox": "-34.9,-8.1,-34.88,-8.08", "geometry_available": "true"}, {})
    assert r["aoi_status"] == "aoi_forte_por_bbox_ou_geometria_patch"


def test_aoi_parcial_e_ausente_e_invalida():
    parc = S.resolve_aoi({"centroid_lon": "-34.9", "centroid_lat": "-8.1"}, {})
    assert parc["aoi_status"] == "aoi_parcial_por_centroide_ou_ponto"
    aus = S.resolve_aoi({}, {})
    assert aus["aoi_status"] == "aoi_ausente"
    inv = S.resolve_aoi({"bbox": "-34.9,-8.1,-34.88"}, {})  # so 3 valores
    assert inv["aoi_status"] == "aoi_invalida"
    ctx = S.resolve_aoi({}, {"city": "Recife"})
    assert ctx["aoi_status"] == "aoi_contextual_insuficiente"


def test_todos_canarios_aoi_forte():
    for a in _read(S.AOI_CSV):
        assert a["aoi_status"] == "aoi_forte_por_bbox_ou_geometria_patch"
        assert a["sufficient_for_search"] == "true"


# 3. query spec com GRD/IW/VV,VH/no_download/metadata_only -------------------
def test_query_spec():
    for q in _read(S.QUERY_CSV):
        assert q["product_type"] == "GRD"
        assert q["acquisition_mode"] == "IW"
        assert q["polarization_required"] == "VV,VH"
        assert q["metadata_only"] == "true"
        assert q["no_download"] == "true"
        assert q["remote_execution_allowed_now"] == "false"


# 4. ausencia de catalogo nao quebra build -----------------------------------
def test_catalogo_ausente_nao_quebra():
    assert S.load_scenes(Path("nao_existe_xyz.json")) == []
    pair = S._pair_contract([], [], aoi_ok=True, has_window=True)
    assert pair["status"] == "sem_par_valido"


# 5. catalogo sintetico gera cenas -------------------------------------------
def test_catalogo_sintetico(tmp_path):
    cat = {"results": [{
        "granuleName": "S1A_IW_GRDH_1SDV_20220521T080221_x", "productID": "PID1",
        "startTime": "2022-05-21T08:02:21Z", "dataset": "Sentinel-1A", "productType": "GRD_HD",
        "beamMode": "IW", "polarization": "VV+VH", "flightDirection": "DESCENDING",
        "path": 9, "frame": 620, "wkt": "POLYGON ((-35.4 -9.8, -32.8 -9.8, -32.8 -7.8, -35.4 -7.8, -35.4 -9.8))",
    }]}
    p = tmp_path / "asf_sentinel1_grd_metadata_recife_event.json"
    p.write_text(json.dumps(cat), encoding="utf-8")
    scenes = S.load_scenes(p)
    assert len(scenes) == 1
    assert scenes[0]["scene_id"] == "S1A_IW_GRDH_1SDV_20220521T080221_x"
    assert scenes[0]["acquisition_date"] == "2022-05-21"
    assert scenes[0]["scene_bbox"] is not None


# 6. classificacao pre/pos/fora/wrong/overlap --------------------------------
def test_classify_scene():
    assert S.classify_scene(_scene(acquisition_date="2022-05-16"), AOI, PRE_WIN, POST_WIN) == "pre_event_candidate"
    assert S.classify_scene(_scene(acquisition_date="2022-05-28"), AOI, PRE_WIN, POST_WIN) == "post_event_candidate"
    assert S.classify_scene(_scene(acquisition_date="2022-06-09"), AOI, PRE_WIN, POST_WIN) == "outside_window"
    assert S.classify_scene(_scene(product_type="SLC"), AOI, PRE_WIN, POST_WIN) == "wrong_product"
    assert S.classify_scene(_scene(acquisition_mode="EW"), AOI, PRE_WIN, POST_WIN) == "wrong_mode"
    assert S.classify_scene(_scene(scene_bbox=(0, 0, 1, 1)), AOI, PRE_WIN, POST_WIN) == "insufficient_overlap"
    assert S.classify_scene(_scene(scene_id="", acquisition_date=""), AOI, PRE_WIN, POST_WIN) == "missing_metadata"


# 7. contrato de pareamento exige pre+pos ------------------------------------
def test_pareamento_pre_pos():
    pre = [_scene(acquisition_date="2022-05-16")]
    post = [_scene(acquisition_date="2022-05-28")]
    pair = S._pair_contract(pre, post, aoi_ok=True, has_window=True)
    assert pair["status"] == "par_completo_pre_pos"
    assert pair["same_orbit"] is True  # mesma path/frame/direcao
    assert S._pair_contract(pre, [], True, True)["status"] == "falta_cena_pos"
    assert S._pair_contract([], post, True, True)["status"] == "falta_cena_pre"
    assert S._pair_contract(pre, post, aoi_ok=False, has_window=True)["status"] == "bloqueado_por_aoi_ou_janela"
    # todos os canarios reais tem par completo
    for p in _read(S.PAIRING_CSV):
        assert p["pairing_status"] == "par_completo_pre_pos"


# 8. guardrails de nao execucao ----------------------------------------------
def test_nao_execucao():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    for k in ("download_executado", "sar_executado", "footprint_criado", "geometria_criada", "qa_executado", "internet_usada", "stac_api_usada"):
        assert manifest[k] == 0
    assert manifest["can_execute_search_now_true_count"] == 0
    assert manifest["can_download_now_true_count"] == 0
    for t in _read(S.TARGETS_CSV):
        assert t["can_execute_search_now"] == "false"
        assert t["can_download_now"] == "false"
    for s in _read(S.SCENES_CSV):
        assert s["metadata_only"] == "true"
        assert s["no_download_in_this_milestone"] == "true"


# 9. guardrails de nao treino/ground truth/score_v7 --------------------------
def test_nao_treino():
    for t in _read(S.TARGETS_CSV):
        assert t["eligible_for_training"] == "false"
        assert t["eligible_for_ground_truth"] == "false"
        assert t["trainable"] == "false"
        assert t["score_v7_candidate"] == "false"
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_promovidos"] == 0


# 10. relatorio publico em portugues -----------------------------------------
def test_relatorio_portugues():
    h1 = S._report_h1(S.REPORT.read_text(encoding="utf-8"))
    assert S.title_is_portuguese(h1)
    txt = S.REPORT.read_text(encoding="utf-8").lower()
    assert "nunca feature pré-evento" in txt or "nunca feature pre-evento" in txt


# 11. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.TARGETS_CSV, S.AOI_CSV, S.QUERY_CSV, S.CATALOG_CSV, S.SCENES_CSV,
            S.PAIRING_CSV, S.PLAN_CSV, S.BLOCKERS_CSV, S.SUMMARY_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 12. schemas aceitam validos e rejeitam invalidos ---------------------------
def test_schema_aceita():
    jsonschema = pytest.importorskip("jsonschema")
    for csvf, schf in [(S.TARGETS_CSV, S.SCHEMA_TARGET), (S.QUERY_CSV, S.SCHEMA_QUERY),
                       (S.SCENES_CSV, S.SCHEMA_SCENE), (S.PAIRING_CSV, S.SCHEMA_PAIR)]:
        schema = json.loads(schf.read_text(encoding="utf-8"))
        v = jsonschema.Draft202012Validator(schema)
        for r in _read(csvf)[:40]:
            assert list(v.iter_errors(r)) == []


def test_schema_rejeita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_QUERY.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    inv = dict(_read(S.QUERY_CSV)[0])
    inv["no_download"] = "false"
    assert list(v.iter_errors(inv))
    inv2 = dict(_read(S.QUERY_CSV)[0])
    inv2["remote_execution_allowed_now"] = "true"
    assert list(v.iter_errors(inv2))


# 13. GT08 nao modifica GT01-07 ----------------------------------------------
def test_nao_modifica_marcos_anteriores():
    import hashlib

    def digest():
        h = {}
        for p in (sorted(SUSC.glob("susc_gt0[1234567]_*"))
                  + sorted((ROOT / "schemas" / "suscetibilidade").glob("susc_gt0[1234567]_*"))):
            h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return h

    before = digest()
    S.build_all()
    assert before == digest()


# 14. manifesto lista sha256 -------------------------------------------------
def test_manifesto_sha256():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    assert len(arts) == 16  # 10 publicos + 6 schemas
    for a in arts:
        assert len(a["sha256"]) == 64


# 15. recomendacao GT09 do resultado real ------------------------------------
def test_recomendacao_gt09():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["proximo_marco"].startswith("GT09")
    # ha pares completos -> pacote visual manual
    assert "Pacote Visual Manual" in manifest["proximo_marco"]
    assert S._next_milestone([{"aoi_status": "aoi_ausente", "pairing_status": "sem_par_valido", "ready_for_metadata_search": "false"}]).endswith("Resolucao de AOI")


# extra: validador completo passa + cenas locais reais existem ---------------
def test_validador_completo_e_cenas_reais():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []
    assert manifest["catalogos_locais"] >= 1
    assert manifest["cenas_locais_avaliadas"] >= 1
