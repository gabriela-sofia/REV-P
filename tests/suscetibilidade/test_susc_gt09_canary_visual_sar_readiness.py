"""Testes para SUSC-GT09 - Esteira Canaria de Revisao Visual e Pre-Execucao SAR."""

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


S = _load("gt09_common", "susc_gt09_canary_visual_sar_readiness_common.py")


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def _scene(sid, date, path="111", frame="621", pol="VV+VH", direction="DESCENDING"):
    return {"scene_id": sid, "product_id": "P_" + sid, "acquisition_datetime": date + "T07:54:10Z",
            "polarization": pol, "orbit_path": path, "orbit_frame": frame, "orbit_direction": direction,
            "aoi_overlap": "true"}


# 1. leitura GT08 + 5 canarios -----------------------------------------------
def test_le_gt08_5_canarios():
    alvos = _read(SUSC / "susc_gt08_alvos_sentinel1_canarios.csv")
    gate = _read(S.GATE_CSV)
    assert len(gate) == len(alvos) == 5


# 2-3. selecao do melhor par + preferencia mesma orbita ----------------------
def test_selecao_par_mesma_orbita():
    pre = [_scene("PRE_A", "2022-05-16", path="111", frame="621"),
           _scene("PRE_B", "2022-05-21", path="9", frame="620")]
    post = [_scene("POST_A", "2022-05-28", path="111", frame="621")]
    best, alt, allp = S.select_pairs(pre, post, "2022-05-24")
    assert best["pre"]["scene_id"] == "PRE_A"  # mesma orbita vence
    assert best["same_orbit"] is True
    assert len(alt) == 1


def test_pares_reais_mesma_orbita():
    for p in _read(S.PAIRS_CSV):
        if p["selected"] == "true":
            assert p["same_orbit_pair"] == "true"
            assert p["aoi_intersection"] == "true"


# 4-5. rejeicao fora de janela / sem overlap (via GT08 classificacao) --------
def test_pares_so_de_candidatas_validas():
    # os pares GT09 so usam cenas ja classificadas pre/post pelo GT08 (dentro da janela + overlap)
    scenes = {s["scene_id"]: s for s in _read(SUSC / "susc_gt08_cenas_candidatas_locais_sentinel1.csv")}
    for p in _read(S.PAIRS_CSV):
        if p["selected"] == "true":
            assert scenes[p["pre_scene_id"]]["scene_classification"] == "pre_event_candidate"
            assert scenes[p["post_scene_id"]]["scene_classification"] == "post_event_candidate"


# 6. dossie completo ---------------------------------------------------------
def test_dossie_completo():
    dos = _read(S.DOSSIER_CSV)
    assert len(dos) == 5
    for d in dos:
        assert d["pre_scene_id"] != "not_available"
        assert d["post_scene_id"] != "not_available"
        assert d["remains_review_only"] == "true"


# 7. inventario local sem gerar imagem ---------------------------------------
def test_inventario_sem_gerar_imagem():
    inv = _read(S.ASSETS_CSV)
    assert inv
    for a in inv:
        assert a["image_generated_in_this_milestone"] == "false"
        assert a["downloaded_in_this_milestone"] == "false"
        assert a["asset_class"] in S.ASSET_CLASSES


# 8. contrato renderizacao sem execucao --------------------------------------
def test_render_sem_execucao():
    for r in _read(S.RENDER_CSV):
        assert r["renderizacao_executada"] == "false"
        assert r["raster_baixado"] == "false"
        assert r["footprint_criado"] == "false"


# 9. dry-run com etapas nao executadas ---------------------------------------
def test_dry_run_nao_executado():
    dr = _read(S.DRYRUN_CSV)
    assert len(dr) == 55  # 11 etapas x 5 canarios
    for d in dr:
        assert d["executado_agora"] == "false"


# 10. handoff QA sem accepted/rejected ---------------------------------------
def test_handoff_sem_accepted_rejected():
    for h in _read(S.HANDOFF_CSV):
        assert h["current_qa_status"] not in ("accepted", "rejected")
        assert h["accepted_future_is_review_only"] == "true"
        assert h["handoff_status"] in S.HANDOFF_STATUS


# 11. readiness gate correto -------------------------------------------------
def test_readiness_gate():
    for g in _read(S.GATE_CSV):
        assert g["gate_state"] in S.GATE_STATES
        if g["gate_state"] == "pronto_para_renderizacao_visual_futura":
            assert g["aoi_forte"] == "true"
            assert g["par_selecionado"] == "true"
            assert g["scene_ids_presentes"] == "true"
    # logica pura
    assert S.gate_state(False, True, True, True, True) == "bloqueado_por_aoi"
    assert S.gate_state(True, False, False, True, True) == "bloqueado_por_par_pre_pos"
    assert S.gate_state(True, True, True, True, True) == "pronto_para_renderizacao_visual_futura"


# 12. guardrails de leakage --------------------------------------------------
def test_leakage():
    for lk in _read(S.LEAKAGE_CSV):
        assert lk["post_scene_used_as_pre_event_feature"] == "false"
        assert lk["footprint_is_evaluation_reference_only"] == "true"
    for p in _read(S.PAIRS_CSV):
        assert "nunca feature pre-evento" in p["post_event_scene_role_pt"].lower()


# 13. guardrails nao download/SAR/GEE/footprint ------------------------------
def test_guardrails_execucao():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    for k in ("render_executada", "raster_baixado", "footprint_criado", "geometria_criada",
              "sar_executado", "qa_executado", "imagem_gerada", "download_executado", "internet_usada"):
        assert manifest[k] == 0
    assert manifest["dry_run_executado_agora_count"] == 0


# 14. guardrails nao treino/ground truth/score_v7 ----------------------------
def test_guardrails_treino():
    for g in _read(S.GATE_CSV):
        assert g["eligible_for_training"] == "false"
        assert g["eligible_for_ground_truth"] == "false"
        assert g["trainable"] == "false"
        assert g["score_v7_candidate"] == "false"
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["positive_strong_promovidos"] == 0
    assert manifest["accepted_atual"] == 0 and manifest["rejected_atual"] == 0


# 15. relatorio publico em portugues -----------------------------------------
def test_relatorio_portugues():
    h1 = S._report_h1(S.REPORT.read_text(encoding="utf-8"))
    assert S.title_is_portuguese(h1)


# 16. build deterministico ----------------------------------------------------
def test_build_deterministico():
    outs = [S.PAIRS_CSV, S.ALT_PAIRS_CSV, S.DOSSIER_CSV, S.ASSETS_CSV, S.RENDER_CSV,
            S.DRYRUN_CSV, S.HANDOFF_CSV, S.GATE_CSV, S.LEAKAGE_CSV, S.SUMMARY_CSV, S.REPORT]
    S.build_all()
    a = {p.name: p.read_bytes() for p in outs}
    S.build_all()
    b = {p.name: p.read_bytes() for p in outs}
    assert a == b


# 17. schemas aceitam validos e rejeitam invalidos ---------------------------
def test_schema_aceita():
    jsonschema = pytest.importorskip("jsonschema")
    for csvf, schf in [(S.PAIRS_CSV, S.SCHEMA_PAIR), (S.DOSSIER_CSV, S.SCHEMA_DOSSIER),
                       (S.ASSETS_CSV, S.SCHEMA_ASSET), (S.RENDER_CSV, S.SCHEMA_RENDER),
                       (S.DRYRUN_CSV, S.SCHEMA_DRYRUN), (S.GATE_CSV, S.SCHEMA_GATE)]:
        schema = json.loads(schf.read_text(encoding="utf-8"))
        v = jsonschema.Draft202012Validator(schema)
        for r in _read(csvf)[:40]:
            assert list(v.iter_errors(r)) == []


def test_schema_rejeita():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(S.SCHEMA_RENDER.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    inv = dict(_read(S.RENDER_CSV)[0])
    inv["renderizacao_executada"] = "true"
    assert list(v.iter_errors(inv))
    schema2 = json.loads(S.SCHEMA_GATE.read_text(encoding="utf-8"))
    v2 = jsonschema.Draft202012Validator(schema2)
    inv2 = dict(_read(S.GATE_CSV)[0])
    inv2["eligible_for_training"] = "true"
    assert list(v2.iter_errors(inv2))


# 18. GT09 nao modifica GT01-08 ----------------------------------------------
def test_nao_modifica_marcos_anteriores():
    import hashlib

    def digest():
        h = {}
        for p in (sorted(SUSC.glob("susc_gt0[12345678]_*"))
                  + sorted((ROOT / "schemas" / "suscetibilidade").glob("susc_gt0[12345678]_*"))):
            h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return h

    before = digest()
    S.build_all()
    assert before == digest()


# 19. manifesto lista sha256 -------------------------------------------------
def test_manifesto_sha256():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    assert len(arts) == 19  # 12 publicos + 7 schemas
    for a in arts:
        assert len(a["sha256"]) == 64


# 20. recomendacao GT10 do resultado real ------------------------------------
def test_recomendacao_gt10():
    manifest = json.loads(S.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["proximo_marco"].startswith("GT10")
    # ha ativos visuais + prontos -> renderizacao visual local
    assert "Renderizacao Visual Local" in manifest["proximo_marco"]


# extra: validador completo passa + ativos reais -----------------------------
def test_validador_e_ativos_reais():
    manifest = S.build_all()
    assert S.validate_outputs(manifest) == []
    assert manifest["ativos_locais_total"] >= 1
    assert manifest["pares_mesma_orbita"] == 5
