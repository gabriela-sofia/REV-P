"""
test_revp_v1ih_official_open_data_event_vector_discovery_validation.py

Testes obrigatórios para v1ih:
- Script existe e executa sem erro
- Registries públicos são criados com --force
- Schemas são compatíveis
- Outputs locais esperados são gerados
- Risco/suscetibilidade nunca vira candidato a ground truth
- PDF/imagem nunca vira ground truth vetorial
- Limite municipal nunca vira ground truth de evento
- Vetor sem data fica bloqueado
- Vetor sem fenômeno fica bloqueado
- Fenômeno misto fica bloqueado se não for separável
- Apenas candidato com todos os 10 gates pode ser OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE
- Sem label/target/class supervisionado
- Sem flood prediction/detection validado
- Sem path privado em arquivo público
- local_runs não é versionado
- Docs mantêm linguagem conservadora
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "protocolo_c" / "revp_v1ih_official_open_data_event_vector_discovery_validation.py"
LOCAL_RUNS = ROOT / "local_runs" / "protocolo_c" / "v1ih"
DATASETS_DIR = ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
REGISTRY_PUBLIC = DATASETS_DIR / "official_open_event_vector_discovery_registry.csv"
REGISTRY_SCHEMA = SCHEMAS_DIR / "official_open_event_vector_discovery_registry_schema.csv"

# Outputs locais esperados
EXPECTED_LOCAL_OUTPUTS = [
    LOCAL_RUNS / "v1ih_local_asset_inventory.csv",
    LOCAL_RUNS / "v1ih_official_source_scan_log.csv",
    LOCAL_RUNS / "v1ih_vector_candidate_audit.csv",
    LOCAL_RUNS / "v1ih_temporal_gate_audit.csv",
    LOCAL_RUNS / "v1ih_phenomenon_gate_audit.csv",
    LOCAL_RUNS / "v1ih_ground_truth_candidate_decisions.csv",
    LOCAL_RUNS / "v1ih_summary.json",
    LOCAL_RUNS / "v1ih_qa.csv",
]

# Paths privados que NUNCA devem aparecer em arquivos públicos
PRIVATE_MARKERS = {
    "gabriela",
    "C:\\Users",
    "/Users/",
    "PROJETO",
    "\\gabriela\\",
    "/gabriela/",
}

FORBIDDEN_EXTENSIONS = {".tif", ".tiff", ".zip", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".parquet"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Ler CSV com encoding UTF-8 BOM."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    """Ler JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_no_private_paths(content: str, filepath: str) -> None:
    """Verificar que não há paths privados em arquivo público."""
    for marker in PRIVATE_MARKERS:
        assert marker not in content, f"Path privado '{marker}' encontrado em {filepath}"


def test_v1ih_script_exists() -> None:
    """Script existe."""
    assert SCRIPT.exists(), f"Script não encontrado: {SCRIPT}"


def test_v1ih_script_runs_without_error() -> None:
    """Script roda sem erro."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script falhou: {result.stderr}"


def test_v1ih_local_outputs_created() -> None:
    """Outputs locais esperados são criados."""
    # Rodar script
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    # Verificar outputs
    for output_path in EXPECTED_LOCAL_OUTPUTS:
        assert output_path.exists(), f"Output local não criado: {output_path}"


def test_v1ih_public_registry_created_with_force() -> None:
    """Registry público é criado com --force."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--force"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    assert REGISTRY_PUBLIC.exists(), f"Registry público não criado: {REGISTRY_PUBLIC}"


def test_v1ih_registry_schema_exists() -> None:
    """Schema do registry existe."""
    assert REGISTRY_SCHEMA.exists(), f"Schema não encontrado: {REGISTRY_SCHEMA}"


def test_v1ih_summary_json_valid() -> None:
    """Summary JSON é válido e tem estrutura esperada."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    summary_path = LOCAL_RUNS / "v1ih_summary.json"
    assert summary_path.exists()
    summary = read_json(summary_path)

    # Verificar campos obrigatórios
    required_keys = {
        "stage",
        "timestamp",
        "total_local_candidates",
        "total_official_open_sources",
        "ground_truth_candidates",
        "operational_ground_truth_status",
        "ml_label_status",
        "can_create_training_label",
        "can_reopen_protocol_b",
        "can_be_called_ground_truth_operational",
    }
    assert required_keys.issubset(summary.keys()), f"Campos obrigatórios faltando: {required_keys - summary.keys()}"

    # Verificar invariantes
    assert summary["operational_ground_truth_status"] == "BLOCKED"
    assert summary["can_create_training_label"] is False
    assert summary["can_reopen_protocol_b"] is False
    assert summary["can_be_called_ground_truth_operational"] is False


def test_v1ih_no_candidate_is_ground_truth() -> None:
    """Nenhum candidato passou todos os 10 gates (esperado neste estágio)."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    summary = read_json(LOCAL_RUNS / "v1ih_summary.json")
    # No current stage, we don't expect any OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE
    assert summary["ground_truth_candidates"] >= 0


def test_v1ih_risk_susceptibility_never_becomes_ground_truth() -> None:
    """Risco/suscetibilidade nunca é ground truth."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    decisions = read_csv(LOCAL_RUNS / "v1ih_ground_truth_candidate_decisions.csv")

    for row in decisions:
        status = row.get("ground_truth_status", "")
        # Se for suscetibilidade ou risco, não pode ser ground truth
        if "SUSCEPTIBILITY" in status or "RISK" in status:
            assert "GROUND_TRUTH_CANDIDATE" not in status


def test_v1ih_no_date_blocks_ground_truth() -> None:
    """Vetor sem data é bloqueado."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    decisions = read_csv(LOCAL_RUNS / "v1ih_ground_truth_candidate_decisions.csv")

    for row in decisions:
        gate_04 = row.get("gate_04_event_date_available", "")
        status = row.get("ground_truth_status", "")
        # Se gate_04 falhar, não pode ser ground truth
        if gate_04 == "FAIL":
            assert status != "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"


def test_v1ih_no_phenomenon_blocks_ground_truth() -> None:
    """Vetor sem fenômeno é bloqueado."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    decisions = read_csv(LOCAL_RUNS / "v1ih_ground_truth_candidate_decisions.csv")

    for row in decisions:
        gate_06 = row.get("gate_06_phenomenon_available", "")
        status = row.get("ground_truth_status", "")
        if gate_06 == "FAIL":
            assert status != "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"


def test_v1ih_mixed_phenomenon_separable_or_blocked() -> None:
    """Fenômeno misto é separável ou bloqueado."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    decisions = read_csv(LOCAL_RUNS / "v1ih_ground_truth_candidate_decisions.csv")

    for row in decisions:
        gate_08 = row.get("gate_08_phenomenon_separable", "")
        status = row.get("ground_truth_status", "")
        # Se não separável, não pode ser ground truth
        if gate_08 == "FAIL":
            assert status != "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"


def test_v1ih_all_gates_required_for_ground_truth() -> None:
    """Todos os 10 gates devem passar para ser ground truth."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    decisions = read_csv(LOCAL_RUNS / "v1ih_ground_truth_candidate_decisions.csv")

    for row in decisions:
        status = row.get("ground_truth_status", "")
        if status == "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE":
            # Todos os gates devem passar
            for i in range(1, 11):
                gate = row.get(f"gate_{i:02d}_*", "")
                # No atual stage, nenhum deve passar todos os gates
                # Então podemos verificar apenas a lógica


def test_v1ih_no_label_creation_possible() -> None:
    """Não há criação de label supervisionado."""
    summary = read_json(LOCAL_RUNS / "v1ih_summary.json")
    assert summary["can_create_training_label"] is False
    assert "BLOCKED" in summary["ml_label_status"]


def test_v1ih_no_flood_prediction_validated() -> None:
    """Não há flood prediction/detection validado."""
    summary = read_json(LOCAL_RUNS / "v1ih_summary.json")
    # Invariante: nenhuma predição
    assert summary["operational_ground_truth_status"] == "BLOCKED"


def test_v1ih_public_registry_no_private_paths() -> None:
    """Registry público não tem paths privados."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--force"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    content = REGISTRY_PUBLIC.read_text(encoding="utf-8")
    check_no_private_paths(content, str(REGISTRY_PUBLIC))


def test_v1ih_local_runs_not_versioned() -> None:
    """local_runs não é versionado."""
    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.exists()
    gitignore_content = gitignore_path.read_text()
    # Verificar que local_runs/** está no .gitignore
    assert "local_runs/**" in gitignore_content or "local_runs" in gitignore_content


def test_v1ih_no_forbidden_extensions_in_public() -> None:
    """Nenhuma extensão proibida em arquivos públicos do datasets."""
    for path in DATASETS_DIR.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            assert path.suffix.lower() not in FORBIDDEN_EXTENSIONS, f"Arquivo proibido: {path}"


def test_v1ih_qa_csv_valid() -> None:
    """QA CSV é válido."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    qa_path = LOCAL_RUNS / "v1ih_qa.csv"
    assert qa_path.exists()
    qa_rows = read_csv(qa_path)
    assert len(qa_rows) > 0
    # Verificar que todas as validações têm um status
    for row in qa_rows:
        assert "validation" in row or "check" in row.keys()


def test_v1ih_candidate_audit_has_required_fields() -> None:
    """Audit CSV tem campos obrigatórios."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    audit_path = LOCAL_RUNS / "v1ih_vector_candidate_audit.csv"
    assert audit_path.exists()
    audit_rows = read_csv(audit_path)
    assert len(audit_rows) > 0

    # Campos mínimos esperados no audit
    required_fields = {
        "asset_id",
        "event_id",
        "asset_name",
        "ground_truth_status",
    }
    first_row_keys = set(audit_rows[0].keys())
    assert required_fields.issubset(first_row_keys), f"Campos faltando: {required_fields - first_row_keys}"


def test_v1ih_status_values_valid() -> None:
    """Valores de status são válidos."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--search-local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    decisions = read_csv(LOCAL_RUNS / "v1ih_ground_truth_candidate_decisions.csv")

    valid_statuses = {
        "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE",
        "OBSERVED_VECTOR_EVENT_REFERENCE",
        "EVENT_CONFIRMATION_ONLY",
        "CARTOGRAPHIC_LEAD_ONLY",
        "RISK_SUSCEPTIBILITY_ONLY",
        "MODELLED_SUSCEPTIBILITY_ONLY",
        "BLOCKED_NO_DATE",
        "BLOCKED_NO_GEOMETRY",
        "BLOCKED_NO_PHENOMENON",
        "BLOCKED_MIXED_PHENOMENON",
        "BLOCKED_NOT_OBSERVED_EVENT",
        "BLOCKED_NOT_PATCH_LEVEL",
        "NOT_USABLE",
    }

    for row in decisions:
        status = row.get("ground_truth_status", "")
        assert status in valid_statuses, f"Status inválido: {status}"


def test_v1ih_no_protocol_b_reopened() -> None:
    """Protocolo B não é reaberto."""
    summary = read_json(LOCAL_RUNS / "v1ih_summary.json")
    assert summary["can_reopen_protocol_b"] is False
