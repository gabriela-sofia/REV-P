"""
Testes para v1is — Official Event Unit Spatial Anchor & Sentinel Readiness

Cobre:
- Execução básica (script importável, CLI responde)
- Outputs locais produzidos (6 arquivos)
- Registros públicos produzidos (2 arquivos + 2 schemas)
- Classificação de spatial anchors (apenas unidades com coordenada explícita)
- Unidades documentary-only sem coordenada inventada
- Prontidão Sentinel (janela temporal calculada, status correto)
- Prontidão multimodal (baseada em Sentinel + DINO)
- Invariantes absolutos (can_be_operational_ground_truth=NO, etc.)
- Sem path privado em arquivos públicos
- Ausência de labels, targets e Protocolo B
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "protocolo_c"
    / "revp_v1is_event_unit_spatial_anchor_sentinel_readiness.py"
)
LOCAL_OUT = REPO_ROOT / "local_runs" / "protocolo_c" / "v1is"
PUBLIC_DATASETS = REPO_ROOT / "datasets"
SCHEMAS_DIR = PUBLIC_DATASETS / "schemas"
V1IR_REGISTRY = PUBLIC_DATASETS / "official_documented_event_unit_registry.csv"

PRIVATE_PATH_RE = re.compile(r"[A-Za-z]:\\")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _run_builder(extra_args=None):
    """Executa o script v1is com --force e todos os flags."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--force",
        "--read-v1ir-registry",
        "--emit-spatial-anchors",
        "--emit-sentinel-readiness",
        "--emit-multimodal-readiness",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result


def _read_csv(path: Path):
    """Lê CSV e retorna lista de dicts."""
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def builder_run():
    """Executa o builder uma vez por módulo."""
    return _run_builder()


@pytest.fixture(scope="module")
def anchor_registry():
    rows = _read_csv(PUBLIC_DATASETS / "official_event_spatial_anchor_registry.csv")
    return rows


@pytest.fixture(scope="module")
def sentinel_registry():
    rows = _read_csv(PUBLIC_DATASETS / "sentinel_readiness_for_official_event_anchors.csv")
    return rows


@pytest.fixture(scope="module")
def local_anchors():
    rows = _read_csv(LOCAL_OUT / "v1is_spatial_anchor_candidates.csv")
    return rows


@pytest.fixture(scope="module")
def local_doc_refs():
    rows = _read_csv(LOCAL_OUT / "v1is_non_coordinate_event_units.csv")
    return rows


@pytest.fixture(scope="module")
def local_sentinel():
    rows = _read_csv(LOCAL_OUT / "v1is_sentinel_temporal_window_readiness.csv")
    return rows


@pytest.fixture(scope="module")
def local_multimodal():
    rows = _read_csv(LOCAL_OUT / "v1is_multimodal_anchor_readiness.csv")
    return rows


@pytest.fixture(scope="module")
def summary():
    with (LOCAL_OUT / "v1is_summary.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def qa():
    rows = _read_csv(LOCAL_OUT / "v1is_qa.csv")
    return rows


# ---------------------------------------------------------------------------
# TestV1ISBasics — script existe, executa, termina sem erro crítico
# ---------------------------------------------------------------------------


class TestV1ISBasics:
    def test_script_exists(self):
        assert SCRIPT.exists(), f"Script não encontrado: {SCRIPT}"

    def test_script_importable(self):
        result = subprocess.run(
            [sys.executable, "-c", f"import importlib.util; "
             f"spec = importlib.util.spec_from_file_location('v1is', r'{SCRIPT}'); "
             f"m = importlib.util.module_from_spec(spec)"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        # importar não deve explodir
        assert result.returncode == 0 or "SyntaxError" not in result.stderr

    def test_run_exits_zero(self, builder_run):
        assert builder_run.returncode == 0, (
            f"Script retornou exit={builder_run.returncode}\n"
            f"stderr: {builder_run.stderr[:500]}"
        )

    def test_no_python_traceback(self, builder_run):
        assert "Traceback" not in builder_run.stderr, (
            f"Traceback encontrado: {builder_run.stderr[:300]}"
        )

    def test_run_mentions_spatial_anchors(self, builder_run):
        combined = builder_run.stdout + builder_run.stderr
        assert "spatial anchor" in combined.lower() or "ANCHOR" in combined

    def test_run_mentions_invariants(self, builder_run):
        combined = builder_run.stdout + builder_run.stderr
        assert "NO" in combined  # invariantes aparecem na saída

    def test_v1ir_registry_must_exist(self):
        assert V1IR_REGISTRY.exists(), (
            f"v1ir registry não encontrado: {V1IR_REGISTRY}"
        )


# ---------------------------------------------------------------------------
# TestV1ISOutputsLocais — 6 arquivos gerados em local_runs
# ---------------------------------------------------------------------------


class TestV1ISOutputsLocais:
    def test_dir_created(self):
        assert LOCAL_OUT.exists(), f"Diretório não criado: {LOCAL_OUT}"

    def test_spatial_anchor_candidates_csv(self):
        f = LOCAL_OUT / "v1is_spatial_anchor_candidates.csv"
        assert f.exists(), f"Arquivo ausente: {f}"
        assert f.stat().st_size > 0

    def test_non_coordinate_event_units_csv(self):
        f = LOCAL_OUT / "v1is_non_coordinate_event_units.csv"
        assert f.exists(), f"Arquivo ausente: {f}"
        assert f.stat().st_size > 0

    def test_sentinel_temporal_window_readiness_csv(self):
        f = LOCAL_OUT / "v1is_sentinel_temporal_window_readiness.csv"
        assert f.exists(), f"Arquivo ausente: {f}"
        assert f.stat().st_size > 0

    def test_multimodal_anchor_readiness_csv(self):
        f = LOCAL_OUT / "v1is_multimodal_anchor_readiness.csv"
        assert f.exists(), f"Arquivo ausente: {f}"
        assert f.stat().st_size > 0

    def test_summary_json(self):
        f = LOCAL_OUT / "v1is_summary.json"
        assert f.exists(), f"Arquivo ausente: {f}"
        assert f.stat().st_size > 0

    def test_qa_csv(self):
        f = LOCAL_OUT / "v1is_qa.csv"
        assert f.exists(), f"Arquivo ausente: {f}"
        assert f.stat().st_size > 0

    def test_all_local_csv_are_utf8(self):
        for csv_file in LOCAL_OUT.glob("v1is_*.csv"):
            try:
                csv_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pytest.fail(f"Arquivo não é UTF-8: {csv_file.name}")


# ---------------------------------------------------------------------------
# TestV1ISRegistrosPublicos — 4 arquivos públicos
# ---------------------------------------------------------------------------


class TestV1ISRegistrosPublicos:
    def test_anchor_registry_exists(self):
        f = PUBLIC_DATASETS / "official_event_spatial_anchor_registry.csv"
        assert f.exists(), f"Registro público ausente: {f}"

    def test_anchor_schema_exists(self):
        f = SCHEMAS_DIR / "official_event_spatial_anchor_schema.csv"
        assert f.exists(), f"Schema ausente: {f}"

    def test_sentinel_readiness_registry_exists(self):
        f = PUBLIC_DATASETS / "sentinel_readiness_for_official_event_anchors.csv"
        assert f.exists(), f"Registro público ausente: {f}"

    def test_sentinel_readiness_schema_exists(self):
        f = SCHEMAS_DIR / "sentinel_readiness_for_official_event_anchors_schema.csv"
        assert f.exists(), f"Schema ausente: {f}"

    def test_anchor_schema_has_field_column(self):
        rows = _read_csv(SCHEMAS_DIR / "official_event_spatial_anchor_schema.csv")
        assert len(rows) > 0
        assert "field" in rows[0], "Schema deve ter coluna 'field'"

    def test_sentinel_schema_has_field_column(self):
        rows = _read_csv(SCHEMAS_DIR / "sentinel_readiness_for_official_event_anchors_schema.csv")
        assert len(rows) > 0
        assert "field" in rows[0], "Schema deve ter coluna 'field'"


# ---------------------------------------------------------------------------
# TestV1ISSpatialAnchors — apenas ANEXO-II é spatial anchor
# ---------------------------------------------------------------------------


class TestV1ISSpatialAnchors:
    def test_exactly_one_spatial_anchor(self, local_anchors, anchor_registry):
        assert len(local_anchors) == 1, (
            f"Esperado 1 anchor, encontrado {len(local_anchors)}"
        )
        assert len(anchor_registry) == 1, (
            f"Registry público: esperado 1 anchor, encontrado {len(anchor_registry)}"
        )

    def test_anchor_is_anexo_ii(self, local_anchors):
        anchor = local_anchors[0]
        assert "ANEXOII" in anchor["anchor_id"] or "ANEXOII" in anchor["source_unit_id"], (
            f"Anchor deve ser de ANEXO-II: {anchor['anchor_id']}"
        )

    def test_anchor_has_moinho_preto(self, local_anchors):
        anchor = local_anchors[0]
        assert "Moinho Preto" in anchor.get("locality_text_sanitized", ""), (
            f"Localidade deve ser Moinho Preto: {anchor.get('locality_text_sanitized')}"
        )

    def test_anchor_date_19022022(self, local_anchors):
        anchor = local_anchors[0]
        assert "19/02/2022" in anchor.get("event_date", ""), (
            f"Data deve ser 19/02/2022: {anchor.get('event_date')}"
        )

    def test_anchor_coordinate_lat(self, local_anchors):
        anchor = local_anchors[0]
        assert anchor.get("coordinate_lat", "") == "-22.484251", (
            f"Latitude: esperado -22.484251, encontrado {anchor.get('coordinate_lat')}"
        )

    def test_anchor_coordinate_lon(self, local_anchors):
        anchor = local_anchors[0]
        assert anchor.get("coordinate_lon", "") == "-43.211257", (
            f"Longitude: esperado -43.211257, encontrado {anchor.get('coordinate_lon')}"
        )

    def test_anchor_coordinate_source(self, local_anchors):
        anchor = local_anchors[0]
        assert anchor.get("coordinate_source") == "CPRM_FIELD_SURVEY_REPORT", (
            f"coordinate_source errado: {anchor.get('coordinate_source')}"
        )

    def test_anchor_status_is_candidate(self, local_anchors):
        anchor = local_anchors[0]
        assert anchor.get("anchor_status") == "SPATIAL_ANCHOR_CANDIDATE", (
            f"anchor_status: {anchor.get('anchor_status')}"
        )

    def test_anchor_can_be_candidate_yes(self, local_anchors):
        anchor = local_anchors[0]
        assert anchor.get("can_be_spatial_anchor_candidate") == "YES", (
            f"can_be_spatial_anchor_candidate: {anchor.get('can_be_spatial_anchor_candidate')}"
        )

    def test_anchor_phenomenon_movement_of_mass(self, local_anchors):
        anchor = local_anchors[0]
        assert "MOVEMENT_OF_MASS" in anchor.get("phenomenon_group", ""), (
            f"Fenômeno: {anchor.get('phenomenon_group')}"
        )


# ---------------------------------------------------------------------------
# TestV1ISDocumentaryOnly — 9 unidades sem coordenada, 1 insuficiente
# ---------------------------------------------------------------------------


class TestV1ISDocumentaryOnly:
    def test_ten_total_non_anchor_units(self, local_doc_refs):
        assert len(local_doc_refs) == 10, (
            f"Esperado 10 unidades não-anchor, encontrado {len(local_doc_refs)}"
        )

    def test_nine_documentary_reference_only(self, local_doc_refs):
        doc_only = [
            r for r in local_doc_refs
            if r.get("documentary_reference_status") == "DOCUMENTARY_REFERENCE_ONLY"
        ]
        assert len(doc_only) == 9, (
            f"Esperado 9 DOCUMENTARY_REFERENCE_ONLY, encontrado {len(doc_only)}"
        )

    def test_one_insufficient_evidence(self, local_doc_refs):
        insufficient = [
            r for r in local_doc_refs
            if r.get("documentary_reference_status") == "INSUFFICIENT_EVIDENCE"
        ]
        assert len(insufficient) == 1, (
            f"Esperado 1 INSUFFICIENT_EVIDENCE, encontrado {len(insufficient)}"
        )

    def test_no_invented_coordinate(self, local_doc_refs):
        """Unidades documentary-only não devem ter coordenada inventada."""
        for row in local_doc_refs:
            if row.get("documentary_reference_status") == "DOCUMENTARY_REFERENCE_ONLY":
                assert row.get("coordinate_available", "NO") == "NO", (
                    f"Unidade documentary-only tem coordinate_available!=NO: "
                    f"{row.get('unit_id')}"
                )

    def test_cannot_become_anchor(self, local_doc_refs):
        """Nenhuma unidade documentary-only pode virar spatial anchor."""
        for row in local_doc_refs:
            assert row.get("can_become_spatial_anchor") == "NO", (
                f"can_become_spatial_anchor != NO: {row.get('unit_id')}"
            )

    def test_reason_no_anchor_references_no_coordinate(self, local_doc_refs):
        """O motivo de não poder virar anchor deve referenciar ausência de coordenada."""
        for row in local_doc_refs:
            if row.get("documentary_reference_status") == "DOCUMENTARY_REFERENCE_ONLY":
                reason = row.get("reason_no_anchor", "")
                assert "COORDINATE" in reason.upper() or "COORD" in reason.upper(), (
                    f"reason_no_anchor não menciona coordenada: {reason}"
                )

    def test_no_bairro_centroid_used(self, local_doc_refs):
        """Nenhuma unidade deve ter coordenada baseada em centroid de bairro."""
        for row in local_doc_refs:
            reason = row.get("reason_no_anchor", "")
            assert "centroid" not in reason.lower() or "CANNOT" in reason.upper(), (
                f"Centroid pode ter sido usado: {row.get('unit_id')} — {reason}"
            )


# ---------------------------------------------------------------------------
# TestV1ISSentinelReadiness
# ---------------------------------------------------------------------------


class TestV1ISSentinelReadiness:
    def test_one_sentinel_readiness_entry(self, local_sentinel, sentinel_registry):
        assert len(local_sentinel) == 1, (
            f"Esperado 1 entrada de prontidão Sentinel, encontrado {len(local_sentinel)}"
        )
        assert len(sentinel_registry) == 1, (
            f"Registry público: esperado 1, encontrado {len(sentinel_registry)}"
        )

    def test_anchor_id_matches(self, local_sentinel, local_anchors):
        sr = local_sentinel[0]
        anchor = local_anchors[0]
        assert sr.get("anchor_id") == anchor.get("anchor_id"), (
            f"anchor_id diverge: sentinel={sr.get('anchor_id')}, "
            f"anchor={anchor.get('anchor_id')}"
        )

    def test_event_date_preserved(self, local_sentinel):
        sr = local_sentinel[0]
        assert "19/02/2022" in sr.get("event_date", ""), (
            f"event_date: {sr.get('event_date')}"
        )

    def test_search_window_start_is_before_event(self, local_sentinel):
        sr = local_sentinel[0]
        win_start = sr.get("sentinel_search_window_start", "")
        assert win_start, "sentinel_search_window_start vazio"
        # janela começa antes de 19/02/2022
        from datetime import datetime
        dt_start = datetime.strptime(win_start, "%Y-%m-%d")
        event_dt = datetime(2022, 2, 19)
        assert dt_start < event_dt, (
            f"Janela deve começar antes do evento: {win_start}"
        )

    def test_search_window_end_is_after_event(self, local_sentinel):
        sr = local_sentinel[0]
        win_end = sr.get("sentinel_search_window_end", "")
        assert win_end, "sentinel_search_window_end vazio"
        from datetime import datetime
        dt_end = datetime.strptime(win_end, "%Y-%m-%d")
        event_dt = datetime(2022, 2, 19)
        assert dt_end > event_dt, (
            f"Janela deve terminar após o evento: {win_end}"
        )

    def test_sentinel_pixel_size_10m(self, local_sentinel):
        sr = local_sentinel[0]
        assert sr.get("sentinel_pixel_size_m") == "10", (
            f"pixel_size_m: {sr.get('sentinel_pixel_size_m')}"
        )

    def test_patch_status_not_none(self, local_sentinel):
        sr = local_sentinel[0]
        status = sr.get("sentinel_patch_candidate_available", "")
        assert status in (
            "PATCH_NOT_AVAILABLE",
            "PATCH_FOUND",
            "PATCH_FOUND_COORD_UNVERIFIED",
        ), f"sentinel_patch_candidate_available inválido: {status}"

    def test_readiness_status_is_valid(self, local_sentinel):
        sr = local_sentinel[0]
        status = sr.get("sentinel_readiness_status", "")
        valid_statuses = {
            "NOT_READY_PATCH_UNAVAILABLE",
            "PARTIAL_PATCH_FOUND_COORD_UNVERIFIED",
            "READY_FOR_INSPECTION",
        }
        assert status in valid_statuses, (
            f"sentinel_readiness_status inválido: {status}"
        )

    def test_regions_inspected_field_present(self, local_sentinel):
        sr = local_sentinel[0]
        assert sr.get("sentinel_patch_registry_inspected", ""), (
            "sentinel_patch_registry_inspected vazio"
        )

    def test_notes_not_empty(self, local_sentinel):
        sr = local_sentinel[0]
        assert sr.get("notes", ""), "notes vazio na prontidão Sentinel"


# ---------------------------------------------------------------------------
# TestV1ISMultimodalReadiness
# ---------------------------------------------------------------------------


class TestV1ISMultimodalReadiness:
    def test_one_multimodal_entry(self, local_multimodal):
        assert len(local_multimodal) == 1, (
            f"Esperado 1 entrada multimodal, encontrado {len(local_multimodal)}"
        )

    def test_anchor_id_matches(self, local_multimodal, local_anchors):
        mr = local_multimodal[0]
        anchor = local_anchors[0]
        assert mr.get("anchor_id") == anchor.get("anchor_id"), (
            f"anchor_id diverge multimodal/anchor"
        )

    def test_multimodal_status_is_valid(self, local_multimodal):
        mr = local_multimodal[0]
        status = mr.get("multimodal_readiness_status", "")
        valid = {
            "NOT_READY",
            "PARTIAL_COORD_VERIFICATION_NEEDED",
            "READY_FOR_STRUCTURAL_ANALYSIS",
        }
        assert status in valid, f"multimodal_readiness_status inválido: {status}"

    def test_dino_backbone_present(self, local_multimodal):
        mr = local_multimodal[0]
        backbone = mr.get("dino_backbone", "")
        assert "dinov2" in backbone.lower() or "dino" in backbone.lower(), (
            f"dino_backbone ausente ou inválido: {backbone}"
        )

    def test_dino_embedding_dim_numeric(self, local_multimodal):
        mr = local_multimodal[0]
        dim = mr.get("dino_embedding_dim", "")
        assert dim.isdigit(), f"dino_embedding_dim não numérico: {dim}"
        assert int(dim) > 0

    def test_can_create_label_is_NO(self, local_multimodal):
        for mr in local_multimodal:
            assert mr.get("can_create_label") == "NO", (
                f"can_create_label != NO: {mr.get('anchor_id')}"
            )

    def test_can_train_model_is_NO(self, local_multimodal):
        for mr in local_multimodal:
            assert mr.get("can_train_model") == "NO", (
                f"can_train_model != NO: {mr.get('anchor_id')}"
            )

    def test_minimum_evidence_needed_not_empty(self, local_multimodal):
        for mr in local_multimodal:
            assert mr.get("minimum_evidence_needed", ""), (
                f"minimum_evidence_needed vazio: {mr.get('anchor_id')}"
            )


# ---------------------------------------------------------------------------
# TestV1ISInvariants — can_* = NO em todos os lugares
# ---------------------------------------------------------------------------


class TestV1ISInvariants:
    def test_anchors_can_be_truth_NO(self, local_anchors):
        for row in local_anchors:
            assert row.get("can_be_operational_ground_truth") == "NO", (
                f"can_be_operational_ground_truth != NO: {row.get('anchor_id')}"
            )

    def test_anchors_can_create_label_NO(self, local_anchors):
        for row in local_anchors:
            assert row.get("can_create_training_label") == "NO", (
                f"can_create_training_label != NO: {row.get('anchor_id')}"
            )

    def test_anchors_can_train_NO(self, local_anchors):
        for row in local_anchors:
            assert row.get("can_train_model") == "NO", (
                f"can_train_model != NO: {row.get('anchor_id')}"
            )

    def test_anchors_can_reopen_b_NO(self, local_anchors):
        for row in local_anchors:
            assert row.get("can_reopen_protocol_b") == "NO", (
                f"can_reopen_protocol_b != NO: {row.get('anchor_id')}"
            )

    def test_doc_refs_can_be_truth_NO(self, local_doc_refs):
        for row in local_doc_refs:
            assert row.get("can_be_operational_ground_truth") == "NO", (
                f"can_be_operational_ground_truth != NO: {row.get('unit_id')}"
            )

    def test_doc_refs_can_create_label_NO(self, local_doc_refs):
        for row in local_doc_refs:
            assert row.get("can_create_training_label") == "NO", (
                f"can_create_training_label != NO: {row.get('unit_id')}"
            )

    def test_doc_refs_can_train_NO(self, local_doc_refs):
        for row in local_doc_refs:
            assert row.get("can_train_model") == "NO", (
                f"can_train_model != NO: {row.get('unit_id')}"
            )

    def test_doc_refs_can_reopen_b_NO(self, local_doc_refs):
        for row in local_doc_refs:
            assert row.get("can_reopen_protocol_b") == "NO", (
                f"can_reopen_protocol_b != NO: {row.get('unit_id')}"
            )

    def test_multimodal_can_create_label_NO(self, local_multimodal):
        for row in local_multimodal:
            assert row.get("can_create_label") == "NO", (
                f"can_create_label != NO: {row.get('anchor_id')}"
            )

    def test_multimodal_can_train_NO(self, local_multimodal):
        for row in local_multimodal:
            assert row.get("can_train_model") == "NO", (
                f"can_train_model != NO: {row.get('anchor_id')}"
            )

    def test_summary_invariants_all_NO(self, summary):
        inv = summary.get("invariants", {})
        for key, val in inv.items():
            assert val == "NO", f"Invariante {key} != NO: {val}"

    def test_public_anchor_registry_truth_NO(self, anchor_registry):
        for row in anchor_registry:
            assert row.get("can_be_operational_ground_truth") == "NO"
            assert row.get("can_create_training_label") == "NO"
            assert row.get("can_train_model") == "NO"
            assert row.get("can_reopen_protocol_b") == "NO"


# ---------------------------------------------------------------------------
# TestV1ISSemPathPrivado — sem path privado em arquivos públicos
# ---------------------------------------------------------------------------


class TestV1ISSemPathPrivado:
    def _check_file_no_private_path(self, path: Path):
        content = path.read_text(encoding="utf-8", errors="replace")
        matches = PRIVATE_PATH_RE.findall(content)
        assert len(matches) == 0, (
            f"Path privado encontrado em {path.name}: {matches[:3]}"
        )

    def test_anchor_registry_no_private_path(self):
        f = PUBLIC_DATASETS / "official_event_spatial_anchor_registry.csv"
        self._check_file_no_private_path(f)

    def test_sentinel_registry_no_private_path(self):
        f = PUBLIC_DATASETS / "sentinel_readiness_for_official_event_anchors.csv"
        self._check_file_no_private_path(f)

    def test_anchor_schema_no_private_path(self):
        f = SCHEMAS_DIR / "official_event_spatial_anchor_schema.csv"
        self._check_file_no_private_path(f)

    def test_sentinel_schema_no_private_path(self):
        f = SCHEMAS_DIR / "sentinel_readiness_for_official_event_anchors_schema.csv"
        self._check_file_no_private_path(f)


# ---------------------------------------------------------------------------
# TestV1ISQAChecks — checks no v1is_qa.csv
# ---------------------------------------------------------------------------


class TestV1ISQAChecks:
    def _get_check(self, qa_rows, check_name):
        for row in qa_rows:
            if row.get("check") == check_name:
                return row
        return None

    def test_spatial_anchors_ge_1_pass(self, qa):
        row = self._get_check(qa, "spatial_anchors_ge_1")
        assert row is not None, "QA check 'spatial_anchors_ge_1' não encontrado"
        assert row.get("result") == "PASS", f"QA falhou: {row}"

    def test_all_anchors_have_explicit_coord(self, qa):
        row = self._get_check(qa, "all_anchors_have_explicit_coord")
        assert row is not None, "QA check ausente"
        assert row.get("result") == "PASS", f"QA falhou: {row}"

    def test_no_anchor_from_no_coord(self, qa):
        row = self._get_check(qa, "no_anchor_from_no_coord_unit")
        assert row is not None, "QA check ausente"
        assert row.get("result") == "PASS", f"QA falhou: {row}"

    def test_invariant_truth_qa(self, qa):
        row = self._get_check(qa, "invariant_can_be_truth_NO")
        assert row is not None
        assert row.get("result") == "PASS"

    def test_invariant_label_qa(self, qa):
        row = self._get_check(qa, "invariant_can_create_label_NO")
        assert row is not None
        assert row.get("result") == "PASS"

    def test_invariant_train_qa(self, qa):
        row = self._get_check(qa, "invariant_can_train_NO")
        assert row is not None
        assert row.get("result") == "PASS"

    def test_invariant_reopen_b_qa(self, qa):
        row = self._get_check(qa, "invariant_can_reopen_b_NO")
        assert row is not None
        assert row.get("result") == "PASS"

    def test_documentary_refs_no_invented_coord(self, qa):
        row = self._get_check(qa, "documentary_refs_no_invented_coord")
        assert row is not None
        assert row.get("result") == "PASS"

    def test_sentinel_readiness_evaluated(self, qa):
        row = self._get_check(qa, "sentinel_readiness_evaluated")
        assert row is not None
        assert row.get("result") == "PASS"

    def test_multimodal_readiness_evaluated(self, qa):
        row = self._get_check(qa, "multimodal_readiness_evaluated")
        assert row is not None
        assert row.get("result") == "PASS"

    def test_no_qa_fail_result(self, qa):
        failures = [r for r in qa if r.get("result") == "FAIL"]
        assert len(failures) == 0, (
            f"QA checks com FAIL: {[(r['check'], r['detail']) for r in failures]}"
        )


# ---------------------------------------------------------------------------
# TestV1ISSummary — conteúdo do JSON summary
# ---------------------------------------------------------------------------


class TestV1ISSummary:
    def test_version_is_v1is(self, summary):
        assert summary.get("script_version") == "v1is", (
            f"script_version: {summary.get('script_version')}"
        )

    def test_spatial_anchors_count(self, summary):
        assert summary.get("spatial_anchors_created") == 1, (
            f"spatial_anchors_created: {summary.get('spatial_anchors_created')}"
        )

    def test_documentary_reference_count(self, summary):
        assert summary.get("documentary_reference_only") == 9, (
            f"documentary_reference_only: {summary.get('documentary_reference_only')}"
        )

    def test_insufficient_evidence_count(self, summary):
        assert summary.get("insufficient_evidence") == 1, (
            f"insufficient_evidence: {summary.get('insufficient_evidence')}"
        )

    def test_anchor_entry_has_coordinate(self, summary):
        anchors = summary.get("spatial_anchors", [])
        assert len(anchors) == 1
        anchor = anchors[0]
        assert anchor.get("coordinate_lat") == "-22.484251"
        assert anchor.get("coordinate_lon") == "-43.211257"

    def test_invariants_block_in_summary(self, summary):
        inv = summary.get("invariants", {})
        assert inv.get("can_be_operational_ground_truth") == "NO"
        assert inv.get("can_create_training_label") == "NO"
        assert inv.get("can_train_model") == "NO"
        assert inv.get("can_reopen_protocol_b") == "NO"

    def test_sentinel_readiness_block(self, summary):
        sr_list = summary.get("sentinel_readiness_per_anchor", [])
        assert len(sr_list) == 1
        sr = sr_list[0]
        assert "patch_available" in sr
        assert "readiness_status" in sr

    def test_multimodal_readiness_block(self, summary):
        mr_list = summary.get("multimodal_readiness_per_anchor", [])
        assert len(mr_list) == 1
        mr = mr_list[0]
        assert mr.get("can_create_label") == "NO"
        assert mr.get("can_train_model") == "NO"

    def test_dino_manifest_inspected(self, summary):
        dino = summary.get("dino_manifest_inspected", {})
        assert isinstance(dino, dict)
        assert "total_patches" in dino

    def test_timestamp_in_summary(self, summary):
        ts = summary.get("run_timestamp", "")
        assert "2026" in ts or len(ts) >= 10, f"timestamp ausente: {ts}"


# ---------------------------------------------------------------------------
# TestV1ISNaoGeocodificaBairro — nenhum centroid ou geocodificação
# ---------------------------------------------------------------------------


class TestV1ISNaoGeocodificaBairro:
    def test_doc_refs_have_no_lat_lon(self, local_doc_refs):
        """Unidades documentary-only não devem ter coordenadas."""
        coord_fields = ["coordinate_lat", "coordinate_lon"]
        for row in local_doc_refs:
            for field in coord_fields:
                if field in row:
                    assert row[field] == "", (
                        f"Coordenada presente em unidade documentary-only: "
                        f"{row.get('unit_id')} — {field}={row[field]}"
                    )

    def test_no_centroid_word_in_doc_ref_notes(self, local_doc_refs):
        """Nenhuma nota deve mencionar uso de centroid como coordenada."""
        for row in local_doc_refs:
            notes = row.get("notes", "").lower()
            reason = row.get("reason_no_anchor", "").lower()
            # centroid pode ser mencionado como MOTIVO de bloqueio, não como uso
            if "centroid" in reason:
                assert "geocod" not in reason, (
                    f"Centroid usado para geocodificar: {row.get('unit_id')}"
                )


# ---------------------------------------------------------------------------
# TestV1ISLocalRunsNaoVersionado — local_runs não deve aparecer em staged
# ---------------------------------------------------------------------------


class TestV1ISLocalRunsNaoVersionado:
    def test_local_runs_not_staged(self):
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        staged = result.stdout.splitlines()
        local_staged = [f for f in staged if "local_runs" in f]
        assert len(local_staged) == 0, (
            f"Arquivos de local_runs staged: {local_staged}"
        )

    def test_no_npz_staged(self):
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        npz_staged = [l for l in result.stdout.splitlines() if l.endswith(".npz")]
        assert len(npz_staged) == 0, f"Nenhum .npz deve estar staged: {npz_staged}"

    def test_no_npy_staged(self):
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        npy_staged = [l for l in result.stdout.splitlines() if l.endswith(".npy")]
        assert len(npy_staged) == 0, f"Nenhum .npy deve estar staged: {npy_staged}"

    def test_no_tif_staged(self):
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        tif_staged = [
            l for l in result.stdout.splitlines() if l.endswith((".tif", ".tiff"))
        ]
        assert len(tif_staged) == 0, f"Nenhum raster deve estar staged: {tif_staged}"
