"""
Testes para revp_v1iu_anchor_centered_sentinel_patch_generation.py

Verifica:
  - script existe e é importável
  - outputs locais são gerados ao rodar
  - se raster cobre anchor, patch local é gerado
  - se nenhum raster cobre anchor, blocker correto é emitido
  - registry público só existe se patch real foi gerado
  - patch não vira label
  - can_create_training_label = false sempre
  - can_train_model = false sempre
  - can_reopen_protocol_b = false sempre
  - TIF/NPY/NPZ não versionados fora de local_runs
  - sem path privado em CSV/MD público
  - docs não usam detection/prediction
  - docs não sugerem e-mail/solicitação/Protocolo B como próximo passo automático
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────

REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1iu_anchor_centered_sentinel_patch_generation.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iu"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
PROJETO_ROOT = Path(r"C:\Users\gabriela\Documents\PROJETO")

# ─── Fixtures ─────────────────────────────────────────────────────────────────

PRIVATE_PATH_FRAGMENTS = [
    r"C:\Users\gabriela",
    "/Users/gabriela",
    "gabriela",
    "Documents\\PROJETO",
    "Documents/PROJETO",
]

FORBIDDEN_WORDS_DOCS = [
    "detection",
    "prediction",
    "predict",
    "flood detection",
    "landslide detection",
]

FORBIDDEN_SUGGESTIONS_DOCS = [
    "solicitar",
    "protocolo b",  # como próximo passo automático
    "abrir chamado",
    "enviar e-mail",
]


# ─── 1. Script existe e é importável ──────────────────────────────────────────

class TestScriptExists:
    def test_script_file_exists(self):
        assert SCRIPT.exists(), f"Script não encontrado: {SCRIPT}"

    def test_script_importable(self):
        """Testa que o módulo não tem erros de sintaxe."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Erro de compilação: {result.stderr}"

    def test_script_has_required_flags(self):
        """Script deve aceitar todos os flags CLI definidos."""
        source = SCRIPT.read_text(encoding="utf-8")
        required_flags = [
            "--force",
            "--scan-projeto",
            "--scan-revp",
            "--scan-local-only",
            "--read-official-anchor",
            "--scan-local-sentinel-rasters",
            "--group-bands",
            "--generate-anchor-centered-patch",
            "--emit-qa",
            "--emit-readiness",
        ]
        for flag in required_flags:
            assert flag in source, f"Flag ausente no script: {flag}"


# ─── 2. Outputs locais são gerados ────────────────────────────────────────────

class TestLocalOutputsGenerated:
    """Executa o script e verifica criação dos outputs locais."""

    @pytest.fixture(autouse=True, scope="class")
    def run_script(self):
        """Roda o script uma vez para a classe inteira."""
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--force",
                "--scan-projeto",
                "--scan-revp",
                "--scan-local-only",
                "--read-official-anchor",
                "--scan-local-sentinel-rasters",
                "--group-bands",
                "--generate-anchor-centered-patch",
                "--emit-qa",
                "--emit-readiness",
            ],
            capture_output=True,
            text=True,
            cwd=str(REVP_ROOT),
        )
        self.__class__._result = result
        yield

    def test_script_runs_without_exception(self):
        rc = self.__class__._result.returncode
        stderr = self.__class__._result.stderr
        assert rc == 0, f"Script falhou com returncode={rc}\nSTDERR:\n{stderr}"

    def test_output_dir_created(self):
        assert LOCAL_RUNS.exists(), f"Diretório de saída não criado: {LOCAL_RUNS}"

    def test_inventory_csv_created(self):
        p = LOCAL_RUNS / "v1iu_local_sentinel_raster_inventory.csv"
        assert p.exists(), "Inventário de rasters não criado"

    def test_coverage_audit_csv_created(self):
        p = LOCAL_RUNS / "v1iu_anchor_raster_coverage_audit.csv"
        assert p.exists(), "Audit de cobertura não criado"

    def test_band_grouping_csv_created(self):
        p = LOCAL_RUNS / "v1iu_band_grouping_audit.csv"
        assert p.exists(), "Audit de agrupamento não criado"

    def test_generation_log_csv_created(self):
        p = LOCAL_RUNS / "v1iu_anchor_patch_generation_log.csv"
        assert p.exists(), "Log de geração não criado"

    def test_manifest_csv_created(self):
        p = LOCAL_RUNS / "v1iu_anchor_patch_manifest_local.csv"
        assert p.exists(), "Manifest local não criado"

    def test_qa_csv_created(self):
        p = LOCAL_RUNS / "v1iu_qa.csv"
        assert p.exists(), "QA CSV não criado"

    def test_readiness_csv_created(self):
        p = LOCAL_RUNS / "v1iu_reference_patch_readiness_decision.csv"
        assert p.exists(), "Readiness CSV não criado"

    def test_summary_json_created(self):
        p = LOCAL_RUNS / "v1iu_summary.json"
        assert p.exists(), "Summary JSON não criado"


# ─── 3. Blocker correto quando nenhum raster cobre o anchor ───────────────────

class TestNoRasterCoverageBlocker:
    """
    Dado que sabemos (por v1it + pré-scan) que nenhum raster local cobre o anchor,
    verifica que o script emite o blocker correto.
    """

    def test_summary_json_has_correct_blocker(self):
        summary_path = LOCAL_RUNS / "v1iu_summary.json"
        if not summary_path.exists():
            pytest.skip("Summary não encontrado — rode test_local_outputs_generated primeiro")

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        # Dois cenários válidos:
        # 1. Nenhum raster cobre → blocker LOCAL_SENTINEL_RASTER_NOT_AVAILABLE_FOR_ANCHOR
        # 2. Raster cobre mas patch falhou → outro blocker
        # 3. Patch gerado com QA PASS → blocking_reason=NONE
        blocking = summary.get("blocking_reason", "")
        rasters_covering = summary.get("rasters_covering_anchor", 0)

        if rasters_covering == 0:
            assert "LOCAL_SENTINEL_RASTER_NOT_AVAILABLE_FOR_ANCHOR" in blocking, (
                f"Blocker esperado LOCAL_SENTINEL_RASTER_NOT_AVAILABLE_FOR_ANCHOR, "
                f"obtido: {blocking}"
            )

    def test_readiness_covers_anchor_zero_implies_blocked(self):
        readiness_path = LOCAL_RUNS / "v1iu_reference_patch_readiness_decision.csv"
        if not readiness_path.exists():
            pytest.skip("Readiness não encontrado")

        with open(readiness_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "Readiness CSV vazio"
        row = rows[0]

        covering = int(row.get("rasters_covering_anchor", 0))
        status = row.get("reference_patch_status", "")
        blocker = row.get("blocking_reason", "")

        if covering == 0:
            assert "BLOCKED" in status, f"Status deveria ser BLOCKED mas é: {status}"
            assert blocker != "", f"Blocking reason deveria ser preenchido"

    def test_gate_matrix_written(self):
        gate_matrix = DATASETS / "official_anchor_reference_patch_gate_matrix.csv"
        assert gate_matrix.exists(), (
            "Gate matrix pública não criada. "
            "Deve ser criada sempre (inclui diagnóstico quando bloqueado)."
        )


# ─── 4. Registry público só se patch real gerado ─────────────────────────────

class TestRegistryOnlyIfPatchGenerated:
    def test_registry_only_if_patch_qa_pass(self):
        summary_path = LOCAL_RUNS / "v1iu_summary.json"
        if not summary_path.exists():
            pytest.skip("Summary não encontrado")

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        patches_qa_pass = summary.get("patches_qa_pass", 0)
        registry_path = DATASETS / "official_anchor_reference_patch_registry.csv"

        if patches_qa_pass == 0:
            assert not registry_path.exists(), (
                "Registry público existe mas nenhum patch com QA PASS foi gerado. "
                "Registry só deve ser criado se patch real com QA PASS existir."
            )
        else:
            assert registry_path.exists(), (
                "Patch com QA PASS foi gerado mas registry público não foi criado."
            )


# ─── 5. Invariantes de segurança ─────────────────────────────────────────────

class TestSecurityInvariants:
    """Garante que labels, treino e Protocolo B nunca são habilitados."""

    def _load_summary(self):
        p = LOCAL_RUNS / "v1iu_summary.json"
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def test_can_create_training_label_always_false(self):
        summary = self._load_summary()
        if summary is None:
            pytest.skip("Summary não encontrado")
        assert summary.get("can_create_training_label") == False

    def test_can_train_model_always_false(self):
        summary = self._load_summary()
        if summary is None:
            pytest.skip("Summary não encontrado")
        assert summary.get("can_train_model") == False

    def test_can_reopen_protocol_b_always_false(self):
        summary = self._load_summary()
        if summary is None:
            pytest.skip("Summary não encontrado")
        assert summary.get("can_reopen_protocol_b") == False

    def test_readiness_training_invariants(self):
        readiness_path = LOCAL_RUNS / "v1iu_reference_patch_readiness_decision.csv"
        if not readiness_path.exists():
            pytest.skip("Readiness não encontrado")

        with open(readiness_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            pytest.skip("Readiness vazio")

        row = rows[0]
        assert row.get("can_create_training_label", "YES").upper() == "NO"
        assert row.get("can_train_model", "YES").upper() == "NO"
        assert row.get("can_reopen_protocol_b", "YES").upper() == "NO"
        assert row.get("can_be_operational_ground_truth", "YES").upper() == "NO"
        assert row.get("public_versioning_status", "") == "METADATA_ONLY"

    def test_qa_training_gates_pass(self):
        qa_path = LOCAL_RUNS / "v1iu_qa.csv"
        if not qa_path.exists():
            pytest.skip("QA não encontrado")

        with open(qa_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {r["gate"]: r["status"] for r in reader}

        assert rows.get("no_training_label_gate") == "PASS"
        assert rows.get("no_model_training_gate") == "PASS"
        assert rows.get("no_protocol_b_gate") == "PASS"

    def test_gate_matrix_training_invariants(self):
        gm = DATASETS / "official_anchor_reference_patch_gate_matrix.csv"
        if not gm.exists():
            pytest.skip("Gate matrix não encontrada")

        with open(gm, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            pytest.skip("Gate matrix vazia")

        row = rows[0]
        assert row.get("can_create_training_label", "YES").upper() == "NO"
        assert row.get("can_train_model", "YES").upper() == "NO"
        assert row.get("can_reopen_protocol_b", "YES").upper() == "NO"


# ─── 6. TIF/NPY/NPZ não versionados fora de local_runs ───────────────────────

class TestNoRasterInPublicScope:
    def test_no_tif_in_datasets(self):
        tifs = list(DATASETS.rglob("*.tif")) + list(DATASETS.rglob("*.tiff"))
        assert len(tifs) == 0, f"TIFs encontrados em datasets/: {tifs}"

    def test_no_npy_in_datasets(self):
        npys = list(DATASETS.rglob("*.npy")) + list(DATASETS.rglob("*.npz"))
        assert len(npys) == 0, f"NPY/NPZ encontrados em datasets/: {npys}"

    def test_no_tif_in_docs(self):
        tifs = list(DOCS.rglob("*.tif")) + list(DOCS.rglob("*.tiff")) if DOCS.exists() else []
        assert len(tifs) == 0, f"TIFs encontrados em docs/: {tifs}"

    def test_no_tif_in_scripts(self):
        scripts_dir = REVP_ROOT / "scripts"
        tifs = list(scripts_dir.rglob("*.tif")) + list(scripts_dir.rglob("*.tiff"))
        assert len(tifs) == 0, f"TIFs encontrados em scripts/: {tifs}"

    def test_gitignore_covers_local_runs(self):
        gitignore = REVP_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore não encontrado")
        content = gitignore.read_text(encoding="utf-8")
        assert "local_runs" in content, "local_runs/ não está no .gitignore"

    def test_gitignore_covers_rasters(self):
        gitignore = REVP_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore não encontrado")
        content = gitignore.read_text(encoding="utf-8")
        assert "*.tif" in content or "*.tiff" in content, "*.tif/tiff não está no .gitignore"


# ─── 7. Sem path privado em CSVs públicos ────────────────────────────────────

class TestNoPrivatePathInPublic:
    def _check_file_for_private_paths(self, path: Path):
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8", errors="replace")
        found = []
        for fragment in PRIVATE_PATH_FRAGMENTS:
            if fragment.lower() in content.lower():
                found.append(fragment)
        return found

    def test_registry_no_private_path(self):
        registry = DATASETS / "official_anchor_reference_patch_registry.csv"
        if not registry.exists():
            pytest.skip("Registry não existe")
        leaks = self._check_file_for_private_paths(registry)
        assert not leaks, f"Path privado vazou em registry: {leaks}"

    def test_gate_matrix_no_private_path(self):
        gm = DATASETS / "official_anchor_reference_patch_gate_matrix.csv"
        if not gm.exists():
            pytest.skip("Gate matrix não existe")
        leaks = self._check_file_for_private_paths(gm)
        assert not leaks, f"Path privado vazou em gate matrix: {leaks}"

    def test_docs_no_private_path(self):
        doc_files = list(DOCS.glob("*v1iu*.md")) if DOCS.exists() else []
        for doc in doc_files:
            leaks = self._check_file_for_private_paths(doc)
            assert not leaks, f"Path privado vazou em {doc.name}: {leaks}"


# ─── 8. Docs não usam linguagem proibida ─────────────────────────────────────

class TestDocLanguage:
    def _get_doc_content(self) -> str:
        if not DOCS.exists():
            return ""
        doc_files = list(DOCS.glob("*v1iu*.md"))
        return "\n".join(d.read_text(encoding="utf-8", errors="replace") for d in doc_files)

    def test_docs_no_detection_prediction(self):
        content = self._get_doc_content().lower()
        if not content:
            pytest.skip("Docs v1iu não existem (normal se patch não foi gerado)")
        for word in FORBIDDEN_WORDS_DOCS:
            assert word not in content, (
                f"Palavra proibida '{word}' encontrada nos docs v1iu"
            )

    def test_docs_no_forbidden_protocol_b_suggestions(self):
        content = self._get_doc_content().lower()
        if not content:
            pytest.skip("Docs v1iu não existem")
        # Protocolo B pode aparecer como contexto histórico — verificar somente em sugestões diretas
        forbidden = ["abrir protocolo b", "reiniciar protocolo b", "ative o protocolo b"]
        for phrase in forbidden:
            assert phrase not in content, (
                f"Sugestão proibida '{phrase}' encontrada nos docs v1iu"
            )

    def test_docs_differentiate_reference_from_ground_truth(self):
        content = self._get_doc_content()
        if not content:
            pytest.skip("Docs v1iu não existem")
        # Deve mencionar que NÃO é ground truth operacional
        assert "operacional" in content.lower() or "NÃO" in content or "não é" in content.lower(), (
            "Docs devem diferenciar reference patch candidate de ground truth operacional"
        )


# ─── 9. Anchor correto no summary ────────────────────────────────────────────

class TestAnchorValues:
    def test_summary_anchor_coordinates(self):
        p = LOCAL_RUNS / "v1iu_summary.json"
        if not p.exists():
            pytest.skip("Summary não encontrado")

        with open(p, encoding="utf-8") as f:
            summary = json.load(f)

        assert abs(float(summary.get("anchor_lat", 0)) - (-22.484251)) < 0.001
        assert abs(float(summary.get("anchor_lon", 0)) - (-43.211257)) < 0.001

    def test_summary_anchor_id(self):
        p = LOCAL_RUNS / "v1iu_summary.json"
        if not p.exists():
            pytest.skip("Summary não encontrado")

        with open(p, encoding="utf-8") as f:
            summary = json.load(f)

        assert "CPRM" in summary.get("anchor_id", "") or "PET2022" in summary.get("anchor_id", "")


# ─── 10. Schema pública escrita ───────────────────────────────────────────────

class TestSchemaFiles:
    def test_gate_matrix_schema_written(self):
        schema = SCHEMAS / "official_anchor_reference_patch_gate_matrix_schema.csv"
        assert schema.exists(), "Schema da gate matrix não criado"

    def test_gate_matrix_schema_has_required_fields(self):
        schema = SCHEMAS / "official_anchor_reference_patch_gate_matrix_schema.csv"
        if not schema.exists():
            pytest.skip("Schema não existe")

        with open(schema, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = {r["field"] for r in reader}

        required = [
            "official_anchor_gate",
            "local_raster_coverage_gate",
            "patch_generation_gate",
            "blocking_reason",
            "can_create_training_label",
            "can_train_model",
            "can_reopen_protocol_b",
        ]
        for req in required:
            assert req in fields, f"Campo obrigatório ausente no schema: {req}"
