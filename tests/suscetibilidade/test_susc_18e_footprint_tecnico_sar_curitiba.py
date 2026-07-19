"""Tests for SUSC-18E footprint tecnico SAR Curitiba."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_18e_footprint_tecnico_sar_curitiba"

EXPECTED = [
    OUT / "curitiba_aoi_tecnica_43_patches.csv",
    OUT / "curitiba_aoi_tecnica_43_patches.geojson",
    OUT / "janelas_sar_curitiba.csv",
    OUT / "auditoria_sar_local_curitiba.csv",
    OUT / "footprint_sar_curitiba.csv",
    OUT / "estatisticas_sar_por_patch_curitiba.csv",
    OUT / "vinculos_tecnicos_sar_patch_curitiba.csv",
    OUT / "features_por_vinculo_sar_curitiba.csv",
    OUT / "comparacao_sar_score_v6_curitiba.csv",
    OUT / "fila_download_sar_curitiba.csv",
    OUT / "instrucoes_download_sar_curitiba.md",
    OUT / "gate_footprint_sar_curitiba.csv",
    OUT / "gate_prontidao_17b_pos_18e.csv",
    OUT / "summary.json",
    OUT / "preflight.json",
    OUT / "pacote_gee" / "gee_sentinel1_curitiba_2022_01_15.js",
    OUT / "pacote_gee" / "gee_sentinel1_curitiba_2022_01_15.md",
    OUT / "pacote_gee" / "gee_export_manifest_curitiba.csv",
    OUT / "pacote_gee" / "expected_outputs_schema.json",
    ROOT / "schemas" / "suscetibilidade" / "susc_18e_sar_curitiba_schema_v1.json",
    ROOT / "outputs_public" / "reports" / "SUSC_18E_FOOTPRINT_TECNICO_SAR_CURITIBA.md",
]


def _load_common():
    path = SCRIPTS / "susc_18e_sar_curitiba_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18e_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


def test_aoi_gerada_de_43_patches_reais():
    aoi = _read(OUT / "curitiba_aoi_tecnica_43_patches.csv")
    assert len(aoi) == 1
    assert aoi[0]["patch_count"] == "43"
    assert aoi[0]["uso_permitido"] == "aoi_tecnica_para_busca_sar"
    geo = json.loads((OUT / "curitiba_aoi_tecnica_43_patches.geojson").read_text(encoding="utf-8"))
    assert geo["features"][0]["properties"]["patch_count"] == 43


def test_aoi_nao_e_geometria_de_ocorrencia():
    aoi = _read(OUT / "curitiba_aoi_tecnica_43_patches.csv")[0]
    assert aoi["geometria_de_ocorrencia"] == "false"
    assert aoi["ground_truth"] == "false"
    assert "nao e geometria de ocorrencia" in aoi["justificativa_tecnica"]


def test_sem_raster_local_gera_pacote_gee_sem_footprint():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["local_sar_raster_sufficient"] is False
    assert summary["footprint_produced"] is False
    assert summary["gee_package_created"] is True
    assert summary["status_18e"] == "18E_PACOTE_GEE_SENTINEL1_PRONTO"
    missing = [str(p) for p in EXPECTED if not p.exists()]
    assert not missing


def test_footprint_local_gera_vinculo_tecnico_nao_forte():
    patches = S.s18c.load_cur_patch_polygons()
    patch_id, ring = next(iter(patches.items()))
    bbox = S.s18c._ring_bbox(ring)
    footprint = {
        "footprint_id": "S18E_FP_TESTE",
        "candidate_event_id": "S17C_REF_0060",
        "footprint_produzido": "true",
        "bbox": ",".join(str(v) for v in bbox),
    }
    links = S.build_technical_links([footprint], patches)
    assert any(l["patch_id"] == patch_id for l in links)
    assert all(l["classe_vinculo"] == "technical_footprint_overlap" for l in links)
    assert all(l["vinculo_forte"] == "false" for l in links)


def test_vinculo_tecnico_nao_e_ground_truth():
    links = _read(OUT / "vinculos_tecnicos_sar_patch_curitiba.csv")
    assert links
    assert all(l["ground_truth"] == "false" for l in links)
    assert all(l["eligible_for_training"] == "false" for l in links)
    assert all(l["score_v7_allowed"] == "false" for l in links)
    assert all(l["vinculo_forte"] == "false" for l in links)


def test_feature_pos_evento_proibida_como_pre_evento():
    errors = S.validate_feature_rows([{
        "feature_id": "X",
        "periodo_evento": "pos_evento",
        "usavel_pre_evento": "true",
        "ground_truth": "false",
        "eligible_for_training": "false",
        "score_v7_allowed": "false",
        "review_only": "true",
    }])
    assert any("feature_pos_evento" in e for e in errors)


def test_score_v6_nao_alterado_e_score_v7_proibido():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    rows = _read(OUT / "comparacao_sar_score_v6_curitiba.csv")
    assert rows
    assert all(r["score_v6_alterado"] == "false" for r in rows)
    assert all(r["score_v7_criado"] == "false" for r in rows)


def test_17b_nao_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_17b_criado"] is False
    assert summary["status_17b"] == "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL"
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*"))


def test_vocabulario_publico_proibido():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex", "automacao"]:
        assert S.public_text_violations_text(f"texto com {termo}")
    assert S.public_text_violations_files() == []


def test_status_invalido_falha():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["status_18e"] = "18E_OK_GENERICO"
    assert any("status_18e_fora_enum" in e for e in S.validate_summary(summary))
