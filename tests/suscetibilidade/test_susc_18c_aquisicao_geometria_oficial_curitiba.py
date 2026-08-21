"""Tests for SUSC-18C aquisicao de geometria oficial de Curitiba."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18c_aquisicao_geometria_oficial_curitiba"
CARDS = OUT / "cartoes_curitiba"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "auditoria_local_geometria_curitiba.csv",
    OUT / "aquisicao_fontes_oficiais_curitiba.csv",
    OUT / "manifesto_aquisicao_curitiba.csv",
    OUT / "fila_aquisicao_externa_curitiba.csv",
    OUT / "curitiba_geometrias_ocorrencia_normalizadas.csv",
    OUT / "curitiba_vinculos_evento_patch.csv",
    OUT / "curitiba_features_por_vinculo.csv",
    OUT / "curitiba_fila_extracao_features.csv",
    OUT / "curitiba_referencia_observacional.csv",
    OUT / "solicitacao_geometria_ocorrencia_curitiba.md",
    OUT / "schema_resposta_esperada_curitiba.json",
    OUT / "modelo_planilha_resposta_curitiba.csv",
    OUT / "gate_curitiba_pos_18c.csv",
    OUT / "gate_prontidao_17b_pos_18c.csv",
    OUT / "resumo_por_status.csv",
    OUT / "summary.json",
    OUT / "preflight.json",
    CARDS / "S18C_REF_0002.md",
    REPORTS / "SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md",
    SCHEMAS / "susc_18c_curitiba_geometria_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_18c_curitiba_geometria_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18c_curitiba_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def _first_patch_bbox():
    patches = S.load_cur_patch_polygons()
    assert patches
    patch_id, ring = next(iter(patches.items()))
    return patch_id, S._ring_bbox(ring)


def _official_base(geometry_type):
    return {
        "candidate_event_id": "S17C_REF_0060",
        "data_evento": "2022-01-15",
        "tipo_fenomeno": "flood_inundation_alagamento",
        "geometry_type": geometry_type,
        "geometry_source": "registro oficial de ocorrencia",
        "crs": "EPSG:4326",
        "is_event_specific": "true",
    }


def test_build_gera_outputs_obrigatorios():
    S.build_all()
    missing = [str(p) for p in EXPECTED if not p.exists()]
    assert not missing


def test_validator_passa():
    assert S.validate() == 0


def test_ponto_oficial_vira_geometria_resolvida_e_patch_link():
    _, bbox = _first_patch_bbox()
    rec = _official_base("point")
    rec.update({"lon": str((bbox[0] + bbox[2]) / 2), "lat": str((bbox[1] + bbox[3]) / 2)})
    norm = S.normalize_geometry(rec, 1)
    links = S.overlay_geometry(norm, S.load_cur_patch_polygons())
    assert norm["status_geometria"] == "geometria_oficial_ponto"
    assert any(classe == "point_within_patch" for _, classe, _ in links)


def test_poligono_oficial_vira_geometria_resolvida():
    _, bbox = _first_patch_bbox()
    rec = _official_base("polygon")
    rec["bbox"] = ",".join(str(v) for v in bbox)
    norm = S.normalize_geometry(rec, 2)
    links = S.overlay_geometry(norm, S.load_cur_patch_polygons())
    assert norm["status_geometria"] == "geometria_oficial_poligono"
    assert any(classe == "exact_polygon_overlap" for _, classe, _ in links)


def test_bbox_oficial_vira_geometria_resolvida():
    _, bbox = _first_patch_bbox()
    rec = _official_base("bbox")
    rec["bbox"] = ",".join(str(v) for v in bbox)
    norm = S.normalize_geometry(rec, 3)
    links = S.overlay_geometry(norm, S.load_cur_patch_polygons())
    assert norm["status_geometria"] == "geometria_oficial_bbox"
    assert any(classe == "bbox_overlap" for _, classe, _ in links)


def test_bairro_textual_bloqueado():
    rec = _official_base("textual")
    rec["bairro"] = "bairro informado sem coordenada"
    assert S.classify_geometria_record(rec) == "bloqueada_textual"


def test_centroide_de_bairro_bloqueado():
    rec = _official_base("point")
    rec.update({"lon": "-49.2", "lat": "-25.4", "is_neighborhood_centroid": "true"})
    assert S.classify_geometria_record(rec) == "bloqueada_area_administrativa"


def test_area_administrativa_bloqueada():
    rec = _official_base("polygon")
    rec.update({"geometry_source": "area administrativa municipal", "bbox": "-49.3,-25.5,-49.2,-25.4"})
    assert S.classify_geometria_record(rec) == "bloqueada_area_administrativa"


def test_sem_geometria_gera_solicitacao_formal_e_linhas_bloqueadas():
    S.build_all()
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["geometria_encontrada"] is False
    assert summary["status_final_18c"] == "18C_CURITIBA_GEOMETRIA_AUSENTE_COM_SOLICITACAO_FORMAL"
    assert summary["patch_links_curitiba"] == 4
    assert summary["patch_links_fortes_curitiba"] == 0
    assert (OUT / "solicitacao_geometria_ocorrencia_curitiba.md").exists()

    geoms = _read(OUT / "curitiba_geometrias_ocorrencia_normalizadas.csv")
    vinculos = _read(OUT / "curitiba_vinculos_evento_patch.csv")
    features = _read(OUT / "curitiba_features_por_vinculo.csv")
    assert geoms and all(r["status_geometria"] == "ausente" for r in geoms)
    assert vinculos and all(r["classe_vinculo"] == "insufficient_for_patch_link" for r in vinculos)
    assert all(r["vinculo_forte"] == "false" for r in vinculos)
    assert all(r["geometry_id"] == "not_available" for r in vinculos)
    assert all(r["patch_id"] == "not_available" for r in vinculos)
    assert all(r["review_only"] == "true" for r in vinculos)
    assert not any(r["classe_vinculo"] in S.CLASSE_VINCULO_FORTE for r in vinculos)
    assert all("sem geometry_id oficial de ocorrencia" in r["justificativa_tecnica"] for r in vinculos)
    assert features and features[0]["lacunas"].startswith("sem_geometria_de_ocorrencia")


def test_referencia_curitiba_0060_bloqueada_para_solicitacao_externa():
    refs = _read(OUT / "curitiba_referencia_observacional.csv")
    alvo = [r for r in refs if r["candidate_event_id"] == "S17C_REF_0060"]
    assert alvo
    assert alvo[0]["status_referencia_observacional"] == "bloqueado_sem_geometria"
    assert alvo[0]["uso_permitido"] == "solicitacao_externa_geometria_oficial"
    assert alvo[0]["ground_truth"] == "false"
    assert alvo[0]["eligible_for_training"] == "false"
    assert alvo[0]["score_v7_allowed"] == "false"


def test_same_region_only_nao_e_forte():
    assert "same_region_only" not in S.CLASSE_VINCULO_FORTE
    refs = _read(OUT / "curitiba_referencia_observacional.csv")
    assert any(r["classe_vinculo"] == "same_region_only" for r in refs)


def test_feature_sem_fonte_falha():
    rows = [{
        "patch_link_id": "X",
        "possui_fisico": "true",
        "possui_espectral": "false",
        "possui_chuva": "false",
        "fonte_fisico": "",
        "fonte_espectral": "ausente",
        "fonte_chuva": "ausente",
    }]
    assert any("possui_fisico_sem_fonte" in e for e in S.validate_features_rows(rows))


def test_validator_rejeita_classe_forte_sem_geometry_id_real():
    rows = [{
        "patch_link_id": "X",
        "candidate_event_id": "S17C_REF_0060",
        "geometry_id": "not_available",
        "geometry_type": "none",
        "patch_id": "not_available",
        "classe_vinculo": "point_within_patch",
        "metrica": "0m",
        "vinculo_forte": "false",
        "review_only": "true",
        "justificativa_tecnica": "invalido",
    }]
    errs = S.validate_vinculos_rows(rows)
    assert any("classe_forte_sem_geometry_id" in e for e in errs)
    assert any("sem_geometria_com_classe_forte" in e for e in errs)


def test_ground_truth_treino_score_v7_e_17b_proibidos():
    refs = _read(OUT / "curitiba_referencia_observacional.csv")
    assert refs
    for row in refs:
        assert row["ground_truth"] == "false"
        assert row["eligible_for_training"] == "false"
        assert row["score_v7_allowed"] == "false"
        assert row["review_only"] == "true"
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_17b_criado"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["status_17b"] == "17B_BLOQUEADO_POR_GEOMETRIA"
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*.csv"))


def test_vocabulario_publico_proibido_falha_e_outputs_estao_limpos():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo}")
    assert S.public_text_violations_files() == []


def test_schema_resposta_exige_campos_minimos_publicos():
    schema = json.loads((OUT / "schema_resposta_esperada_curitiba.json").read_text(encoding="utf-8"))
    for field in ["candidate_event_id", "data_evento", "tipo_fenomeno", "geometry_type", "geometry_source", "crs"]:
        assert field in schema["required"]
