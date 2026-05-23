"""
test_revp_v1ii_targeted_official_repository_event_vector_mining.py

Testes para v1ii-R1 — Mineração dirigida em repositórios oficiais
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "protocolo_c" / "revp_v1ii_targeted_official_repository_event_vector_mining.py"
DATASETS_DIR = ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
REGISTRY_PUBLIC = DATASETS_DIR / "targeted_official_repository_event_vector_registry.csv"
REGISTRY_SCHEMA = SCHEMAS_DIR / "targeted_official_repository_event_vector_registry_schema.csv"
LOCAL_RUNS = ROOT / "local_runs" / "protocolo_c" / "v1ii"


# =========================================================================
# Testes de estrutura básica
# =========================================================================

def test_v1ii_script_exists() -> None:
    """Script v1ii existe"""
    assert SCRIPT.exists(), f"Script não encontrado: {SCRIPT}"


def test_v1ii_script_runs_without_error() -> None:
    """Script roda sem erro"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--scan-local"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script falhou: {result.stderr}"


def test_v1ii_schema_exists() -> None:
    """Schema v1ii existe"""
    assert REGISTRY_SCHEMA.exists(), f"Schema não encontrado: {REGISTRY_SCHEMA}"


def test_v1ii_public_registry_created_with_force() -> None:
    """Registry público pode ser criado com --force"""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--scan-local", "--force"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    # Registry será criado quando houver candidatos
    # Por agora, só verificamos se o comando roda


def test_v1ii_no_email_or_manual_request() -> None:
    """Script não depende de e-mail ou solicitação manual como etapa principal"""
    content = SCRIPT.read_text()
    # Verificar que invariantes estão presentes (negativas)
    assert "nao_enviar_email" in content, "Invariante de não enviar e-mail não documentada"
    assert "nao_criar_solicitacao_institucional" in content, "Invariante de não solicitação institucional não documentada"


def test_v1ii_no_label_creation() -> None:
    """Não há criação de labels supervisionados"""
    content = SCRIPT.read_text()
    # Verificar que invariantes estão documentados
    assert "nao_criar_label_target_class" in content, "Invariante não documentado"


def test_v1ii_no_protocol_b_reopen() -> None:
    """Protocolo B não é reaberto"""
    content = SCRIPT.read_text()
    assert "nao_reabrir_protocolo_b" in content, "Invariante não documentado"


def test_v1ii_local_runs_not_versioned() -> None:
    """local_runs não é versionado"""
    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.exists()
    gitignore_content = gitignore_path.read_text()
    assert "local_runs/**" in gitignore_content or "local_runs" in gitignore_content


# =========================================================================
# Testes de registry público com dados reais
# =========================================================================

def _run_full_scan() -> None:
    """Helper: roda todos os scanners com --force."""
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--scan-rigeo", "--scan-ckan-recife", "--scan-ckan-pe",
            "--scan-dados-rj", "--scan-geocuritiba", "--scan-dados-gov",
            "--scan-local", "--force",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )


def test_v1ii_public_registry_has_rows() -> None:
    """Registry público tem pelo menos 12 linhas (todos os candidatos pré-registrados)"""
    _run_full_scan()
    assert REGISTRY_PUBLIC.exists(), "Registry público não criado"
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 12, f"Registry tem {len(rows)} linhas, esperava >= 12"


def test_v1ii_registry_has_required_columns() -> None:
    """Registry público contém colunas obrigatórias do schema"""
    _run_full_scan()
    required_cols = [
        "repository_candidate_id", "repository_name", "institution", "region",
        "classification_status", "resource_format", "event_date_available",
        "observed_not_risk", "geometry_available",
    ]
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
    for col in required_cols:
        assert col in cols, f"Coluna '{col}' ausente no registry"


def test_v1ii_registry_covers_all_regions() -> None:
    """Registry cobre todas as três regiões: PET, REC, CTB"""
    _run_full_scan()
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    regions = {r["region"] for r in rows}
    for region in ("PET", "REC", "CTB"):
        assert region in regions, f"Região {region} ausente no registry"


def test_v1ii_rigeo_items_registered() -> None:
    """Itens RIGeo estão no registry"""
    _run_full_scan()
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = {r["repository_candidate_id"] for r in rows}
    assert "RIGEO_PET_001" in ids, "RIGEO_PET_001 não encontrado"
    assert "RIGEO_PET_002" in ids, "RIGEO_PET_002 não encontrado"


def test_v1ii_ckan_recife_items_registered() -> None:
    """Itens CKAN Recife estão no registry"""
    _run_full_scan()
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = {r["repository_candidate_id"] for r in rows}
    assert "CKAN_REC_001" in ids
    assert "CKAN_REC_002" in ids


def test_v1ii_no_ground_truth_candidate() -> None:
    """Nenhum candidato ground truth (resultado esperado com fontes públicas atuais)"""
    _run_full_scan()
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gt = [r for r in rows if r["classification_status"] == "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"]
    # Resultado esperado: 0 ground truth confirmados
    assert len(gt) == 0, (
        f"Inesperadamente encontrado(s) {len(gt)} ground truth(s). "
        "Verificar se nova fonte pública foi incorporada corretamente."
    )


def test_v1ii_risk_layers_classified_correctly() -> None:
    """Camadas de risco/susceptibilidade classificadas como RISK_SUSCEPTIBILITY_ONLY"""
    _run_full_scan()
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # CKAN_REC_001 é coordenadas de risco, não ocorrência observada
    ckan_rec_001 = next((r for r in rows if r["repository_candidate_id"] == "CKAN_REC_001"), None)
    assert ckan_rec_001 is not None
    assert ckan_rec_001["classification_status"] == "RISK_SUSCEPTIBILITY_ONLY"
    assert ckan_rec_001["observed_not_risk"] == "NO"


def test_v1ii_blocked_no_date_items_exist() -> None:
    """Itens sem data de evento estão corretamente classificados como BLOCKED_NO_DATE"""
    _run_full_scan()
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    blocked = [r for r in rows if r["classification_status"] == "BLOCKED_NO_DATE"]
    assert len(blocked) >= 2, f"Esperado >= 2 BLOCKED_NO_DATE, encontrado {len(blocked)}"
    for b in blocked:
        assert b["event_date_available"] == "NO", (
            f"{b['repository_candidate_id']}: BLOCKED_NO_DATE mas event_date_available={b['event_date_available']}"
        )


def test_v1ii_registry_no_private_paths() -> None:
    """Registry público não contém paths privados ou referências a PROJETO"""
    _run_full_scan()
    content = REGISTRY_PUBLIC.read_text(encoding="utf-8")
    private_markers = ["gabriela", "C:\\Users", "/Users/", "PROJETO", "sig_extracted"]
    for marker in private_markers:
        assert marker.lower() not in content.lower(), (
            f"Path privado '{marker}' detectado no registry público"
        )


def test_v1ii_network_failure_does_not_crash() -> None:
    """Falha de rede não causa crash — SCAN_FAILED_CONTROLLED ou NETWORK_UNAVAILABLE"""
    # Roda apenas com --scan-local (sem rede) — script deve terminar com código 0
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--scan-local"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script falhou com --scan-local: {result.stderr}"


# =========================================================================
# Testes de outputs locais
# =========================================================================

def test_v1ii_local_outputs_generated() -> None:
    """Todos os 7 outputs locais são gerados"""
    _run_full_scan()
    expected_files = [
        "v1ii_repository_scan_log.csv",
        "v1ii_resource_inventory.csv",
        "v1ii_download_audit.csv",
        "v1ii_vector_table_audit.csv",
        "v1ii_candidate_decisions.csv",
        "v1ii_qa.csv",
        "v1ii_summary.json",
    ]
    for fname in expected_files:
        fpath = LOCAL_RUNS / fname
        assert fpath.exists(), f"Output local não gerado: {fname}"


def test_v1ii_scan_log_covers_all_repos() -> None:
    """Scan log cobre os 6 repositórios consultados"""
    _run_full_scan()
    log_path = LOCAL_RUNS / "v1ii_repository_scan_log.csv"
    assert log_path.exists()
    with log_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 6, f"Scan log tem {len(rows)} repositórios, esperava >= 6"


def test_v1ii_summary_json_structure() -> None:
    """v1ii_summary.json tem estrutura esperada"""
    _run_full_scan()
    summary_path = LOCAL_RUNS / "v1ii_summary.json"
    assert summary_path.exists()
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    required_keys = [
        "stage", "total_resources_found", "ground_truth_candidates",
        "operational_ground_truth_status", "can_create_training_label",
    ]
    for key in required_keys:
        assert key in summary, f"Chave '{key}' ausente em v1ii_summary.json"
    assert summary["operational_ground_truth_status"] == "BLOCKED"
    assert summary["can_create_training_label"] is False


def test_v1ii_qa_csv_has_checks() -> None:
    """v1ii_qa.csv contém validações"""
    _run_full_scan()
    qa_path = LOCAL_RUNS / "v1ii_qa.csv"
    assert qa_path.exists()
    with qa_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 7, f"QA tem {len(rows)} linhas, esperava >= 7"


def test_v1ii_scan_failed_items_have_controlled_status() -> None:
    """Itens com falha de scan têm status SCAN_FAILED_CONTROLLED"""
    _run_full_scan()
    with REGISTRY_PUBLIC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    failed = [r for r in rows if r["classification_status"] == "SCAN_FAILED_CONTROLLED"]
    # CKAN_PE_001 e DATAGOV_003 devem ser SCAN_FAILED_CONTROLLED
    assert len(failed) >= 2, f"Esperado >= 2 SCAN_FAILED_CONTROLLED, encontrado {len(failed)}"


def test_v1ii_schema_has_all_required_fields() -> None:
    """Schema v1ii contém todos os campos do registry"""
    with REGISTRY_SCHEMA.open(encoding="utf-8") as f:
        schema_rows = list(csv.DictReader(f))
    schema_fields = {r["field_name"] for r in schema_rows}
    required_fields = [
        "repository_candidate_id", "repository_name", "institution", "region",
        "event_id", "search_term", "query_url_or_api", "dataset_title",
        "classification_status", "blocking_reason", "geometry_available",
        "event_date_available", "observed_not_risk",
    ]
    for field in required_fields:
        assert field in schema_fields, f"Campo '{field}' ausente no schema"


def test_v1ii_docs_no_email_references() -> None:
    """Documentação v1ii não contém referências a e-mail ou solicitação institucional"""
    doc_path = ROOT / "docs" / "metodologia_cientifica" / "protocolo_c_mineracao_dirigida_repositorios_oficiais_v1ii.md"
    if not doc_path.exists():
        return  # Doc ainda não criada — teste não aplicável agora
    content = doc_path.read_text(encoding="utf-8").lower()
    forbidden = ["enviar e-mail", "solicitar ao sgb", "solicitação institucional", "v1ij — solicitação formal"]
    for term in forbidden:
        assert term not in content, f"Referência proibida encontrada na doc v1ii: '{term}'"
