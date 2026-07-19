"""Tests for SUSC-18D protocolo externo e ingestao Curitiba."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_18d_protocolo_externo_curitiba"

EXPECTED = [
    OUT / "oficio_solicitacao_defesa_civil_curitiba.md",
    OUT / "oficio_solicitacao_geocuritiba_ippuc.md",
    OUT / "checklist_envio_solicitacao.csv",
    OUT / "protocolo_acompanhamento_resposta.csv",
    OUT / "schema_resposta_oficial_curitiba.json",
    OUT / "modelo_resposta_oficial_curitiba.csv",
    OUT / "instrucoes_ingestao_resposta.md",
    OUT / "status_protocolo_externo_curitiba.csv",
    OUT / "matriz_ingestao_resposta_curitiba.csv",
    OUT / "gate_curitiba_pos_18d.csv",
    OUT / "gate_prontidao_17b_pos_18d.csv",
    OUT / "summary.json",
    OUT / "preflight.json",
    OUT / "geometrias_resposta_oficial_curitiba.csv",
    OUT / "vinculos_resposta_patch_curitiba.csv",
    OUT / "relatorio_ingestao_resposta_curitiba.md",
]


def _load_common():
    path = SCRIPTS / "susc_18d_protocolo_externo_curitiba_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18d_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all(records=[], input_files=[])


def _first_patch_bbox():
    patches = S.s18c.load_cur_patch_polygons()
    assert patches
    _, ring = next(iter(patches.items()))
    return S.s18c._ring_bbox(ring)


def _base(geometry_type):
    return {
        "candidate_event_id": "S17C_REF_0060",
        "data_evento": "2022-01-15",
        "tipo_fenomeno": "flood_inundation_alagamento",
        "geometry_type": geometry_type,
        "geometry_source": "registro oficial de ocorrencia",
        "crs": "EPSG:4326",
        "is_event_specific": "true",
    }


def test_build_sem_resposta_gera_outputs_e_aguardando_resposta():
    summary = S.build_all(records=[], input_files=[])
    missing = [str(p) for p in EXPECTED if not p.exists()]
    assert not missing
    assert summary["status_18d"] == "18D_AGUARDANDO_RESPOSTA_OFICIAL"
    assert summary["status_17b"] == "17B_BLOQUEADO_POR_GEOMETRIA"
    assert summary["strong_patch_links"] == 0
    matriz = _read(OUT / "matriz_ingestao_resposta_curitiba.csv")
    assert matriz[0]["status_ingestao"] == "aguardando_resposta"


def test_validator_passa_sem_resposta():
    assert S.validate() == 0


def test_resposta_com_ponto_oficial_gera_point_within_patch():
    bbox = _first_patch_bbox()
    rec = _base("point")
    rec.update({"lon": str((bbox[0] + bbox[2]) / 2), "lat": str((bbox[1] + bbox[3]) / 2)})
    norms, links = S.process_records([rec])
    assert norms[0]["status_geometria"] == "geometria_oficial_ponto"
    assert any(l["classe_vinculo"] == "point_within_patch" and l["vinculo_forte"] == "true" for l in links)


def test_resposta_com_poligono_oficial_gera_exact_polygon_overlap():
    bbox = _first_patch_bbox()
    rec = _base("polygon")
    rec["bbox"] = ",".join(str(v) for v in bbox)
    norms, links = S.process_records([rec])
    assert norms[0]["status_geometria"] == "geometria_oficial_poligono"
    assert any(l["classe_vinculo"] == "exact_polygon_overlap" and l["vinculo_forte"] == "true" for l in links)


def test_resposta_com_bbox_oficial_gera_bbox_overlap():
    bbox = _first_patch_bbox()
    rec = _base("bbox")
    rec["bbox"] = ",".join(str(v) for v in bbox)
    norms, links = S.process_records([rec])
    assert norms[0]["status_geometria"] == "geometria_oficial_bbox"
    assert any(l["classe_vinculo"] == "bbox_overlap" and l["vinculo_forte"] == "true" for l in links)


def test_bairro_textual_bloqueado():
    rec = _base("textual")
    rec["bairro"] = "bairro sem coordenada"
    norm = S.normalize_response_record(rec)
    assert norm["status_geometria"] == "bloqueada_textual"
    assert norm["geometria_de_ocorrencia"] == "false"


def test_centroide_bloqueado():
    rec = _base("point")
    rec.update({"lon": "-49.2", "lat": "-25.4", "is_neighborhood_centroid": "true"})
    norm = S.normalize_response_record(rec)
    assert norm["status_geometria"] == "bloqueada_area_administrativa"
    assert norm["geometria_de_ocorrencia"] == "false"


def test_ground_truth_treino_score_v7_proibidos_e_17b_nao_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["ground_truth_true_count"] == 0
    assert summary["eligible_for_training_true_count"] == 0
    assert summary["score_v7_allowed_true_count"] == 0
    assert summary["benchmark_17b_criado"] is False
    assert summary["score_v7_created"] is False
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*.csv"))


def test_vocabulario_publico_limpo():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex", "automacao"]:
        assert S.public_text_violations_text(f"texto com {termo}")
    assert S.public_text_violations_files() == []
