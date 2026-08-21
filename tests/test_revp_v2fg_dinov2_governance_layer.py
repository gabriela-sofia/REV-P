"""REV-P v2fg -- camada de governança DINOv2 da API.

Cobre o que a etapa exige: dimensionalidade/normalização do embedding,
similaridade, gate OOD por limiar, seleção de medoid, incompatibilidade
territorial, contrato da API e o comportamento quando NÃO existem
embeddings/medoids válidos.

Fronteira testada explicitamente: a camada não altera o bloco físico do
contrato (score/CI/features_used).
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
DINO_SCRIPTS = ROOT / "scripts" / "dino"
API_SCRIPTS = ROOT / "outputs_public" / "data" / "linha_causal" / "susc_20e_api_contrato_inferencia_recife" / "scripts"
for _p in (str(DINO_SCRIPTS), str(API_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import revp_v2fg_dinov2_embedder as emb  # noqa: E402
import revp_v2fg_dinov2_governance_engine as gov  # noqa: E402
from revp_v2fg_dinov2_embedder import Dinov2Embedder  # noqa: E402
from revp_v2fg_dinov2_governance_engine import Dinov2GovernanceEngine  # noqa: E402

MANIFEST = ROOT / "datasets" / "dinov2_governance_medoids_v2fg.json"
CORPUS = ROOT / "datasets" / "dinov2_governance_corpus_v2fg.csv"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def engine() -> Dinov2GovernanceEngine:
    if not MANIFEST.exists():
        pytest.skip("manifesto v2fg ausente -- rode revp_v2fg_build_dinov2_governance_corpus.py")
    return Dinov2GovernanceEngine()


# --------------------------------------------------------------------------- #
# 1. Embedder: dimensionalidade e normalização
# --------------------------------------------------------------------------- #

def test_embedder_constantes_batem_com_o_backbone_real_do_projeto():
    assert emb.MODEL_NAME == "facebook/dinov2-with-registers-base"
    assert emb.EMBEDDING_DIM == 768


def test_embedder_mock_exige_optin_explicito(monkeypatch):
    monkeypatch.delenv("REVP_DINOV2_ALLOW_MOCK", raising=False)
    assert Dinov2Embedder().is_mock is False
    assert Dinov2Embedder(mock=True).is_mock is True
    monkeypatch.setenv("REVP_DINOV2_ALLOW_MOCK", "true")
    assert Dinov2Embedder().is_mock is True


def test_embedder_mock_produz_768d_l2_normalizado_e_deterministico(tmp_path):
    img = tmp_path / "patch.bin"
    img.write_bytes(b"conteudo-deterministico-de-teste")
    embedder = Dinov2Embedder(mock=True)
    vec = embedder.embed_image(img)
    assert vec is not None
    assert len(vec) == 768
    assert emb.is_unit_norm(vec)
    assert vec == embedder.embed_image(img)
    # e o vetor sai carimbado como mock -- nunca se confunde com dado real
    assert embedder.describe()["is_mock"] is True
    assert embedder.describe()["backend"] == "mock"


def test_embedder_sem_l2_nao_normaliza(tmp_path):
    img = tmp_path / "patch.bin"
    img.write_bytes(b"outro-conteudo")
    vec = Dinov2Embedder(mock=True, l2_normalize_output=False).embed_image(img)
    assert vec is not None and len(vec) == 768
    assert not emb.is_unit_norm(vec)


def test_embedder_arquivo_inexistente_retorna_none(tmp_path):
    assert Dinov2Embedder(mock=True).embed_image(tmp_path / "nao_existe.png") is None


def test_embedder_sem_backend_real_nao_inventa_vetor(tmp_path, monkeypatch):
    """Sem pesos locais e sem download, o extrator falha fechado: None, não
    um vetor sintético silencioso."""
    monkeypatch.delenv("REVP_DINOV2_ALLOW_MOCK", raising=False)
    img = tmp_path / "patch.bin"
    img.write_bytes(b"x")
    embedder = Dinov2Embedder(model_path=str(tmp_path / "modelo_inexistente"), allow_download=False)
    assert embedder.embed_image(img) is None
    assert embedder.backend == emb.BACKEND_UNAVAILABLE
    assert embedder.load_error


def test_l2_normalize_recusa_vetor_degenerado():
    assert emb.l2_normalize([0.0] * 768) is None
    assert emb.l2_normalize([3.0, 4.0]) == [0.6, 0.8]


# --------------------------------------------------------------------------- #
# 2. Similaridade
# --------------------------------------------------------------------------- #

def test_cosseno_de_vetor_com_ele_mesmo_e_um():
    v = emb.l2_normalize([1.0, 2.0, 3.0])
    assert gov.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosseno_e_invariante_a_escala_e_simetrico():
    a, b = [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]
    assert gov.cosine_similarity(a, b) == pytest.approx(0.5)
    assert gov.cosine_similarity(a, [10.0 * x for x in b]) == pytest.approx(0.5)
    assert gov.cosine_similarity(b, a) == pytest.approx(gov.cosine_similarity(a, b))


def test_cosseno_recusa_dimensao_incompativel_e_norma_zero():
    with pytest.raises(ValueError):
        gov.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        gov.cosine_similarity([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize("vec,ok", [
    ([1.0] + [0.0] * 767, True),
    ([1.0] + [0.0] * 766, False),          # 767D
    ([0.5] + [0.0] * 767, False),          # norma 0.5
    ([0.0] * 768, False),                  # norma zero
    ([float("nan")] + [0.0] * 767, False), # não finito
    (None, False),
])
def test_validate_embedding(vec, ok):
    assert gov.validate_embedding(vec)[0] is ok


# --------------------------------------------------------------------------- #
# 3. Corpus e medoids reais persistidos
# --------------------------------------------------------------------------- #

def test_manifesto_real_existe_e_e_coerente(engine):
    manifest = engine.manifest
    assert manifest["manifest_version"] == gov.MANIFEST_VERSION
    assert manifest["model"]["model_name"] == "facebook/dinov2-with-registers-base"
    assert manifest["model"]["embedding_dim"] == 768
    corpus = manifest["corpus"]
    # contagens obrigatórias da etapa E2/E3
    assert corpus["candidates"] == corpus["valid"] + corpus["blocked"]
    assert corpus["valid"] > 0
    assert corpus["source_files"], "manifesto sem fonte real registrada"
    for src in corpus["source_files"]:
        assert (ROOT / src["path"]).exists(), f"fonte declarada nao existe: {src['path']}"


def test_todo_medoid_persistido_e_768d_l2_normalizado(engine):
    assert engine.available
    for medoid in engine.manifest["medoids"]:
        vec = medoid["embedding"]
        assert len(vec) == 768
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-4)
        assert (ROOT / medoid["source_file"]).exists()


def test_medoid_e_o_patch_de_maior_similaridade_media_no_recorte(engine):
    """Reexecuta a definição publicada sobre o próprio corpus e confere que o
    medoid persistido é mesmo o vencedor."""
    import csv

    with CORPUS.open(encoding="utf-8-sig", newline="") as fh:
        valid = [r for r in csv.DictReader(fh) if r["status"] == "VALID"]

    for medoid in engine.manifest["medoids"]:
        if medoid["scope_kind"] != "region":
            continue
        ids = [r["patch_id"] for r in valid if r["region"] == medoid["region"]]
        vectors = {pid: engine.embedding_for_patch(pid) for pid in ids}
        assert all(v is not None for v in vectors.values())
        means = {
            pid: sum(gov.cosine_similarity(vectors[pid], vectors[o]) for o in ids if o != pid)
            / (len(ids) - 1)
            for pid in ids
        }
        assert max(means, key=lambda p: means[p]) == medoid["patch_id"]
        assert means[medoid["patch_id"]] == pytest.approx(
            medoid["mean_cosine_within_scope"], abs=1e-6)


def test_corpus_nao_contem_mock_nem_fixture(engine):
    import csv

    with CORPUS.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "VALID":
                continue
            assert "mock" not in row["source_file"].lower()
            assert "fixture" not in row["source_file"].lower()
            assert "dinov2-with-registers-base" in row["model_name"]
            assert row["observed_dim"] == "768"


def test_medoid_de_recorte_unitario_nao_e_inventado():
    from revp_v2fg_build_dinov2_governance_corpus import compute_medoid

    vectors = {"REC_00001": emb.l2_normalize([1.0] + [0.0] * 767)}
    assert compute_medoid(["REC_00001"], vectors) is None


# --------------------------------------------------------------------------- #
# 4. Gate OOD e seleção de medoid
# --------------------------------------------------------------------------- #

def test_medoid_do_proprio_corpus_e_in_domain_e_escolhe_a_si_mesmo(engine):
    medoid = next(m for m in engine.manifest["medoids"] if m["scope_kind"] == "region")
    result = engine.evaluate(embedding=medoid["embedding"], requested_region=medoid["region"])
    assert result["status"] == gov.STATUS_IN_DOMAIN
    assert result["nearest_medoid_patch_id"] == medoid["patch_id"]
    assert result["suggested_region"] == medoid["region"]
    assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert result["territorial_match"] == gov.TERRITORIAL_MATCH
    # ranking completo, não só o vencedor
    assert len(result["ranking"]) == len(
        [m for m in engine.manifest["medoids"] if m["scope_kind"] == "region"])
    assert result["ranking"] == sorted(
        result["ranking"], key=lambda r: -r["cosine_similarity"])


def test_gate_ood_dispara_para_vetor_ortogonal_ao_corpus(engine):
    medoid = engine.medoids[0]["embedding"]
    # vetor construído para ser quase ortogonal a todos os medoids reais:
    # componente de sinal alternado, depois ortogonalizado contra o medoid.
    raw = [((-1.0) ** i) for i in range(768)]
    dot = sum(x * y for x, y in zip(raw, medoid))
    orth = emb.l2_normalize([x - dot * y for x, y in zip(raw, medoid)])
    result = engine.evaluate(embedding=orth, requested_region="recife")
    assert result["ood_threshold"] is not None
    assert result["cosine_similarity"] < result["ood_threshold"]
    assert result["status"] == gov.STATUS_OUT_OF_DOMAIN
    assert any("gate OOD" in n for n in result["notes"])


def test_limiar_ood_e_configuravel_e_muda_o_estado(engine):
    medoid = next(m for m in engine.manifest["medoids"] if m["scope_kind"] == "region")
    strict = Dinov2GovernanceEngine(ood_threshold=1.5)
    assert strict.ood_threshold == 1.5
    assert strict.threshold_source == "argumento_explicito"
    out = strict.evaluate(embedding=medoid["embedding"], requested_region=medoid["region"])
    assert out["status"] == gov.STATUS_OUT_OF_DOMAIN

    loose = Dinov2GovernanceEngine(ood_threshold=-1.0)
    assert loose.evaluate(embedding=medoid["embedding"])["status"] == gov.STATUS_IN_DOMAIN


def test_limiar_ood_pode_vir_do_ambiente(engine, monkeypatch):
    monkeypatch.setenv("REVP_DINOV2_OOD_THRESHOLD", "0.99")
    e = Dinov2GovernanceEngine()
    assert e.ood_threshold == pytest.approx(0.99)
    assert e.threshold_source == "env:REVP_DINOV2_OOD_THRESHOLD"


def test_limiar_default_vem_do_manifesto_com_base_declarada(engine):
    gate = engine.manifest["ood_gate"]
    assert gate["threshold_default"] is not None
    assert "percentil" in gate["threshold_basis"]
    assert engine.ood_threshold == pytest.approx(gate["threshold_default"])


def test_patch_do_corpus_resolve_embedding_real(engine):
    pid = engine.known_patch_ids()[0]
    vec = engine.embedding_for_patch(pid)
    assert vec is not None and len(vec) == 768
    result = engine.evaluate(patch_id=pid)
    assert result["status"] in (gov.STATUS_IN_DOMAIN, gov.STATUS_OUT_OF_DOMAIN)
    assert result["query_patch_id"] == pid


# --------------------------------------------------------------------------- #
# 5. Incompatibilidade territorial
# --------------------------------------------------------------------------- #

def test_incompatibilidade_territorial_e_explicita(engine):
    regions = {m["region"]: m for m in engine.manifest["medoids"] if m["scope_kind"] == "region"}
    assert len(regions) >= 2
    target = regions["RECIFE"] if "RECIFE" in regions else next(iter(regions.values()))
    other = next(r for r in regions if r != target["region"])
    result = engine.evaluate(embedding=target["embedding"], requested_region=other)
    assert result["suggested_region"] == target["region"]
    assert result["requested_region"] == other
    assert result["territorial_match"] == gov.TERRITORIAL_MISMATCH
    assert any("incompatibilidade territorial" in n for n in result["notes"])


def test_sem_regiao_solicitada_nao_ha_veredito_territorial(engine):
    medoid = engine.medoids[0]
    result = engine.evaluate(embedding=medoid["embedding"])
    assert result["territorial_match"] == gov.TERRITORIAL_NA
    assert result["requested_region"] is None


def test_nome_de_regiao_do_region_registry_e_normalizado():
    assert gov.normalize_region("recife") == "RECIFE"
    assert gov.normalize_region("petropolis") == "PET"
    assert gov.normalize_region("curitiba") == "CURITIBA"
    assert gov.normalize_region("sao paulo") == "SAO PAULO"


def test_concordancia_territorial_do_corpus_e_medida_e_registrada(engine):
    diag = engine.manifest["diagnostics"]["territorial_concordance"]
    assert 0.0 <= diag["rate"] <= 1.0
    assert diag["concordant"] + 0 <= diag["total"]
    # a leitura honesta do número precisa estar no artefato, não só no código
    assert "revisao humana" in diag["reading"] or "revisão humana" in diag["reading"]


# --------------------------------------------------------------------------- #
# 6. Ausência de embeddings/medoids válidos
# --------------------------------------------------------------------------- #

def test_sem_manifesto_a_governanca_reporta_indisponivel(tmp_path):
    e = Dinov2GovernanceEngine(manifest_path=tmp_path / "nao_existe.json",
                               corpus_path=tmp_path / "nao_existe.csv")
    assert e.available is False
    result = e.evaluate(requested_region="recife")
    assert result["status"] == gov.STATUS_UNAVAILABLE
    assert result["cosine_similarity"] is None
    assert result["ranking"] == []
    assert result["notes"] and "manifesto" in result["notes"][0]


def test_manifesto_de_versao_incompativel_e_recusado(tmp_path, engine):
    bad = copy.deepcopy(engine.manifest)
    bad["manifest_version"] = "dinov2_governance.v0.0"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    e = Dinov2GovernanceEngine(manifest_path=path, corpus_path=CORPUS)
    assert e.available is False
    assert "versao_incompativel" in e.unavailable_reason


def test_manifesto_com_medoid_invalido_e_recusado(tmp_path, engine):
    bad = copy.deepcopy(engine.manifest)
    bad["medoids"][0]["embedding"] = [0.5] * 768  # norma != 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    e = Dinov2GovernanceEngine(manifest_path=path, corpus_path=CORPUS)
    assert e.available is False
    assert "medoid_invalido" in e.unavailable_reason


def test_patch_fora_do_corpus_nao_inventa_evidencia(engine):
    result = engine.evaluate(patch_id="REC_99999999", requested_region="recife")
    assert result["status"] == gov.STATUS_NO_VISUAL_EVIDENCE
    assert result["cosine_similarity"] is None
    assert result["nearest_medoid_patch_id"] is None
    assert "nao esta no corpus" in result["notes"][0]


def test_sem_patch_e_sem_vetor_o_estado_e_explicito(engine):
    result = engine.evaluate(requested_region="recife")
    assert result["status"] == gov.STATUS_NO_VISUAL_EVIDENCE
    assert result["notes"]


def test_embedding_malformado_e_rejeitado_com_estado_proprio(engine):
    result = engine.evaluate(embedding=[0.1] * 10, requested_region="recife")
    assert result["status"] == gov.STATUS_INVALID_EMBEDDING
    assert "dimensao_invalida" in result["notes"][0]


# --------------------------------------------------------------------------- #
# 7. Contrato da API
# --------------------------------------------------------------------------- #

def test_contrato_expoe_bloco_de_governanca_com_campos_obrigatorios():
    from contract_schema import DinoGovernance, ScoreResponse

    assert "dino_governance" in ScoreResponse.model_fields
    fields = DinoGovernance.model_fields
    for name in ("status", "cosine_similarity", "suggested_region", "nearest_medoid_patch_id",
                 "requested_region", "territorial_match", "ood_threshold", "ranking",
                 "audit", "notes", "affects_score"):
        assert name in fields, f"campo ausente no contrato: {name}"


def test_contrato_declara_que_a_governanca_nunca_soma_ao_score():
    from contract_schema import DinoGovernance

    block = DinoGovernance(status="in_domain")
    assert block.affects_score is False
    with pytest.raises(Exception):
        DinoGovernance(status="in_domain", affects_score=True)


def test_contrato_recusa_status_de_governanca_desconhecido():
    from contract_schema import DinoGovernance

    with pytest.raises(Exception):
        DinoGovernance(status="tudo_certo")


def test_request_aceita_patch_visual_opcional_sem_quebrar_chamadas_antigas():
    from contract_schema import ScoreRequest

    geom = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]}
    legacy = ScoreRequest(request_id="r1", region={"geometry": geom})
    assert legacy.visual_patch_id is None
    novo = ScoreRequest(request_id="r2", region={"geometry": geom}, visual_patch_id="REC_00205")
    assert novo.visual_patch_id == "REC_00205"


def test_resultado_da_engine_valida_contra_o_contrato(engine):
    from contract_schema import DinoGovernance

    medoid = engine.medoids[0]
    block = DinoGovernance(**engine.evaluate(embedding=medoid["embedding"],
                                             requested_region="recife"))
    assert block.status in ("in_domain", "out_of_domain")
    assert block.audit.manifest_path == "datasets/dinov2_governance_medoids_v2fg.json"
    assert block.audit.model_name == "facebook/dinov2-with-registers-base"
    assert block.audit.corpus_valid == engine.manifest["corpus"]["valid"]
    assert block.affects_score is False


def test_auditoria_nao_emite_url(engine):
    audit = engine.audit_block()
    blob = json.dumps(audit, ensure_ascii=False).lower()
    for token in ("http://", "https://", "www."):
        assert token not in blob


def test_ponte_da_api_nunca_devolve_none_nem_levanta():
    import dino_governance_bridge

    out = dino_governance_bridge.evaluate(requested_region="recife")
    assert isinstance(out, dict) and "status" in out

    class _Quebrada:
        def evaluate(self, **kwargs):
            raise RuntimeError("falha simulada")

    out = dino_governance_bridge.evaluate(requested_region="recife", engine=_Quebrada())
    assert out["status"] == gov.STATUS_NO_VISUAL_EVIDENCE
    assert "falha_interna_da_governanca_dinov2" in out["notes"][0]


def test_ponte_resolve_patch_por_bbox_e_por_hint():
    import dino_governance_bridge as bridge

    bboxes = {"00205": (-35.0, -8.1, -34.9, -8.0)}
    assert bridge.resolve_patch_id(None, -8.05, -34.95, bboxes) == "REC_00205"
    assert bridge.resolve_patch_id(None, -20.0, -40.0, bboxes) is None
    assert bridge.resolve_patch_id(None, None, None, bboxes) is None
    # hint explícito tem precedência
    assert bridge.resolve_patch_id("cur_00038", -8.05, -34.95, bboxes) == "CUR_00038"


@pytest.fixture(scope="module")
def api_client():
    """Cliente HTTP real da API SUSC-20E. Pula quando o motor físico não pode
    ser carregado neste ambiente (numpy/sklearn/pyproj/rasterio ou os dados
    do v12) -- a governança visual não é responsável por isso."""
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("httpx")
    try:
        import app as api_app
    except Exception as exc:  # pragma: no cover - depende do ambiente
        pytest.skip(f"motor fisico SUSC-20D indisponivel neste ambiente: {exc}")
    from fastapi.testclient import TestClient

    with TestClient(api_app.app) as client:
        yield client, api_app


def test_api_devolve_governanca_em_resposta_bloqueada(api_client):
    """Gate de região sem modelo (Curitiba) continua `insufficient_data`, mas a
    governança visual aparece na resposta -- nada é bloqueado em silêncio."""
    client, _ = api_client
    geom = {"type": "Polygon", "coordinates": [[
        [-49.30, -25.50], [-49.30, -25.45], [-49.25, -25.45], [-49.25, -25.50], [-49.30, -25.50]]]}
    body = client.post("/score", json={
        "request_id": "v2fg-curitiba", "region": {"geometry": geom},
        "visual_patch_id": "CUR_00402"}).json()
    assert body["status"] == "insufficient_data"
    assert body["score"]["value"] is None
    block = body["dino_governance"]
    assert block["status"] == "in_domain"
    assert block["suggested_region"] == "CURITIBA"
    assert block["territorial_match"] == "match"
    assert block["affects_score"] is False
    assert block["ranking"]


def test_api_score_fisico_nao_muda_com_a_governanca(api_client):
    """Mesma geometria, com e sem evidência visual: score, CI e features
    idênticos. É o teste que trava o DINOv2 fora do modelo de Firth."""
    client, api_app = api_client
    if not getattr(api_app, "_engine", None) or not api_app._engine.known_points:
        pytest.skip("motor SUSC-20D sem pontos conhecidos carregados")
    point = api_app._engine.known_points[0]
    lat, lon, d = float(point["lat"]), float(point["lon"]), 0.002
    geom = {"type": "Polygon", "coordinates": [[
        [lon - d, lat - d], [lon - d, lat + d], [lon + d, lat + d],
        [lon + d, lat - d], [lon - d, lat - d]]]}

    sem = client.post("/score", json={
        "request_id": "v2fg-sem", "region": {"geometry": geom}}).json()
    com = client.post("/score", json={
        "request_id": "v2fg-com", "region": {"geometry": geom},
        "visual_patch_id": "CUR_00402"}).json()

    assert sem["status"] == com["status"] == "ok"
    assert sem["score"] == com["score"]
    assert sem["features_used"] == com["features_used"]

    assert sem["dino_governance"]["status"] == "no_visual_evidence"
    assert com["dino_governance"]["territorial_match"] == "mismatch"
    assert com["dino_governance"]["requested_region"] == "RECIFE"
    assert com["dino_governance"]["suggested_region"] == "CURITIBA"
    # a divergência vira limitação explícita, sem mexer no score
    assert any("dinov2_governanca" in lim for lim in com["limitations"])
    assert not any("dinov2_governanca" in lim for lim in sem["limitations"])


def test_governanca_nao_altera_o_bloco_fisico_do_contrato():
    """O contrato físico continua exatamente o mesmo -- a camada só ACRESCENTA
    um bloco. Se algum dia `score`/`features_used` passar a depender do DINO,
    este teste quebra."""
    from contract_schema import ScoreResponse

    fisico = {"request_id", "status", "region_maturity", "score", "features_used",
              "evidence", "limitations", "data_version", "generated_at"}
    assert fisico.issubset(set(ScoreResponse.model_fields))
    assert set(ScoreResponse.model_fields) - fisico == {"dino_governance"}
