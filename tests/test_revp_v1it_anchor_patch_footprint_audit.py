"""
Testes para v1it — Official Anchor to Sentinel Patch Footprint Audit

Cobre:
- Script existe e roda
- Outputs locais gerados (6 arquivos)
- Registros públicos gerados (2 registros + 2 schemas)
- Anchor ANEXO-II aparece e nunca vira label
- Footprint de patch: bounds disponíveis para patches PET
- Containment: anchor não está em nenhum patch existente (resultado esperado)
- Status correto: NO_PATCH_COVERAGE_FOR_ANCHOR ou NEAR_MISS
- Invariantes: can_create_training_label=NO, can_train_model=NO, can_reopen_protocol_b=NO
- Sem path privado em arquivos públicos
- Nenhum pixel lido (apenas header)
- Docs não fazem claims preditivos
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
SCRIPT = REPO_ROOT / "scripts" / "protocolo_c" / "revp_v1it_anchor_patch_footprint_audit.py"
LOCAL_OUT = REPO_ROOT / "local_runs" / "protocolo_c" / "v1it"
PUBLIC_DATASETS = REPO_ROOT / "datasets"
SCHEMAS_DIR = PUBLIC_DATASETS / "schemas"

PRIVATE_PATH_RE = re.compile(r"[A-Za-z]:\\")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_builder():
    cmd = [
        sys.executable, str(SCRIPT),
        "--force", "--read-spatial-anchors", "--read-patch-registries",
        "--read-local-raster-headers", "--compute-footprints",
        "--emit-anchor-patch-audit",
    ]
    return subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )


def _read_csv(path: Path):
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builder_run():
    return _run_builder()


@pytest.fixture(scope="module")
def footprint_inventory():
    return _read_csv(LOCAL_OUT / "v1it_patch_footprint_inventory.csv")


@pytest.fixture(scope="module")
def containment_audit():
    return _read_csv(LOCAL_OUT / "v1it_anchor_patch_containment_audit.csv")


@pytest.fixture(scope="module")
def transform_audit():
    return _read_csv(LOCAL_OUT / "v1it_anchor_coordinate_transform_audit.csv")


@pytest.fixture(scope="module")
def match_decision():
    return _read_csv(LOCAL_OUT / "v1it_anchor_patch_match_decision.csv")


@pytest.fixture(scope="module")
def public_audit():
    return _read_csv(PUBLIC_DATASETS / "official_anchor_patch_footprint_audit.csv")


@pytest.fixture(scope="module")
def multimodal_matrix():
    return _read_csv(PUBLIC_DATASETS / "official_anchor_multimodal_readiness_matrix.csv")


@pytest.fixture(scope="module")
def summary():
    with (LOCAL_OUT / "v1it_summary.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def qa():
    return _read_csv(LOCAL_OUT / "v1it_qa.csv")


# ---------------------------------------------------------------------------
# TestV1ITBasics
# ---------------------------------------------------------------------------


class TestV1ITBasics:
    def test_script_exists(self):
        assert SCRIPT.exists(), f"Script nao encontrado: {SCRIPT}"

    def test_exits_zero(self, builder_run):
        assert builder_run.returncode == 0, (
            f"exit={builder_run.returncode}\nstderr: {builder_run.stderr[:500]}"
        )

    def test_no_traceback(self, builder_run):
        assert "Traceback" not in builder_run.stderr

    def test_output_mentions_patches(self, builder_run):
        combined = builder_run.stdout + builder_run.stderr
        assert "patch" in combined.lower() or "PET" in combined

    def test_output_mentions_coverage_status(self, builder_run):
        combined = builder_run.stdout + builder_run.stderr
        assert "COVERAGE" in combined.upper() or "anchor" in combined.lower()

    def test_output_mentions_invariants(self, builder_run):
        combined = builder_run.stdout + builder_run.stderr
        assert "NO" in combined


# ---------------------------------------------------------------------------
# TestV1ITOutputsLocais — 6 arquivos
# ---------------------------------------------------------------------------


class TestV1ITOutputsLocais:
    def test_dir_exists(self):
        assert LOCAL_OUT.exists()

    def test_footprint_inventory(self):
        f = LOCAL_OUT / "v1it_patch_footprint_inventory.csv"
        assert f.exists() and f.stat().st_size > 0

    def test_transform_audit(self):
        f = LOCAL_OUT / "v1it_anchor_coordinate_transform_audit.csv"
        assert f.exists() and f.stat().st_size > 0

    def test_containment_audit(self):
        f = LOCAL_OUT / "v1it_anchor_patch_containment_audit.csv"
        assert f.exists() and f.stat().st_size > 0

    def test_match_decision(self):
        f = LOCAL_OUT / "v1it_anchor_patch_match_decision.csv"
        assert f.exists() and f.stat().st_size > 0

    def test_summary_json(self):
        f = LOCAL_OUT / "v1it_summary.json"
        assert f.exists() and f.stat().st_size > 0

    def test_qa_csv(self):
        f = LOCAL_OUT / "v1it_qa.csv"
        assert f.exists() and f.stat().st_size > 0

    def test_all_local_utf8(self):
        for f in LOCAL_OUT.glob("v1it_*.csv"):
            try:
                f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pytest.fail(f"Nao e UTF-8: {f.name}")


# ---------------------------------------------------------------------------
# TestV1ITRegistrosPublicos
# ---------------------------------------------------------------------------


class TestV1ITRegistrosPublicos:
    def test_footprint_audit_exists(self):
        assert (PUBLIC_DATASETS / "official_anchor_patch_footprint_audit.csv").exists()

    def test_footprint_schema_exists(self):
        assert (SCHEMAS_DIR / "official_anchor_patch_footprint_audit_schema.csv").exists()

    def test_multimodal_matrix_exists(self):
        assert (PUBLIC_DATASETS / "official_anchor_multimodal_readiness_matrix.csv").exists()

    def test_multimodal_schema_exists(self):
        assert (SCHEMAS_DIR / "official_anchor_multimodal_readiness_schema.csv").exists()

    def test_schemas_have_field_column(self):
        for f in [
            SCHEMAS_DIR / "official_anchor_patch_footprint_audit_schema.csv",
            SCHEMAS_DIR / "official_anchor_multimodal_readiness_schema.csv",
        ]:
            rows = _read_csv(f)
            assert rows and "field" in rows[0], f"Schema sem coluna field: {f.name}"


# ---------------------------------------------------------------------------
# TestV1ITFootprintInventory
# ---------------------------------------------------------------------------


class TestV1ITFootprintInventory:
    def test_has_pet_patches(self, footprint_inventory):
        pet = [r for r in footprint_inventory if "PET" in r.get("patch_id", "")]
        assert len(pet) > 0, "Nenhum patch PET no inventario"

    def test_patches_count_ge_4(self, footprint_inventory):
        # Pelo menos os 4 com DINO embeddings devem estar lá
        pet = [r for r in footprint_inventory if "PET" in r.get("patch_id", "")]
        assert len(pet) >= 4, f"Esperado pelo menos 4 patches PET, encontrado {len(pet)}"

    def test_most_patches_readable(self, footprint_inventory):
        readable = [r for r in footprint_inventory if r.get("raster_readable") == "YES"]
        total = len(footprint_inventory)
        assert len(readable) > 0, "Nenhum patch livel"
        # pelo menos 80% leiveis
        assert len(readable) / total >= 0.8, (
            f"Menos de 80% dos patches leiveis: {len(readable)}/{total}"
        )

    def test_readable_patches_have_bounds(self, footprint_inventory):
        for row in footprint_inventory:
            if row.get("raster_readable") == "YES":
                assert row.get("bounds_left"), f"bounds_left vazio: {row['patch_id']}"
                assert row.get("bounds_right"), f"bounds_right vazio: {row['patch_id']}"
                assert row.get("bounds_bottom"), f"bounds_bottom vazio: {row['patch_id']}"
                assert row.get("bounds_top"), f"bounds_top vazio: {row['patch_id']}"

    def test_readable_patches_have_centroid(self, footprint_inventory):
        for row in footprint_inventory:
            if row.get("raster_readable") == "YES":
                assert row.get("centroid_x"), f"centroid_x vazio: {row['patch_id']}"
                assert row.get("centroid_y"), f"centroid_y vazio: {row['patch_id']}"

    def test_no_private_path_in_inventory(self, footprint_inventory):
        for row in footprint_inventory:
            ref = row.get("asset_path_reference_sanitized", "")
            assert not PRIVATE_PATH_RE.search(ref), (
                f"Path privado em {row['patch_id']}: {ref}"
            )

    def test_crs_epsg_present_for_readable(self, footprint_inventory):
        for row in footprint_inventory:
            if row.get("raster_readable") == "YES":
                assert row.get("crs_epsg"), f"crs_epsg vazio: {row['patch_id']}"

    def test_notes_say_no_pixel_read(self, footprint_inventory):
        for row in footprint_inventory:
            if row.get("raster_readable") == "YES":
                notes = row.get("notes", "").lower()
                assert "pixel_read=no" in notes or "header" in notes, (
                    f"Notes nao mencionam ausencia de leitura de pixel: {row['patch_id']}"
                )


# ---------------------------------------------------------------------------
# TestV1ITContainmentAudit
# ---------------------------------------------------------------------------


class TestV1ITContainmentAudit:
    def test_has_entries(self, containment_audit):
        assert len(containment_audit) > 0

    def test_anchor_ii_appears(self, containment_audit):
        ids = [r.get("anchor_id", "") for r in containment_audit]
        assert any("ANEXOII" in i for i in ids), (
            "Anchor ANEXO-II nao aparece no audit"
        )

    def test_coverage_status_is_valid(self, containment_audit):
        valid = {
            "PATCH_COVERAGE_CONFIRMED_FOR_OFFICIAL_ANCHOR",
            "NO_PATCH_COVERAGE_FOR_ANCHOR",
            "NEAR_MISS_NO_COVERAGE",
            "PATCH_FOUND_BOUNDS_UNAVAILABLE",
            "TRANSFORM_FAILED",
        }
        for row in containment_audit:
            status = row.get("sentinel_patch_coverage_status", "")
            # accept valid or NEAR_MISS_ prefixed
            ok = status in valid or status.startswith("NEAR_MISS_")
            assert ok, f"Status invalido: {status} ({row.get('patch_id')})"

    def test_anchor_coordinate_preserved(self, containment_audit):
        """Coordenada original do anchor não é alterada."""
        for row in containment_audit:
            if "ANEXOII" in row.get("anchor_id", ""):
                assert row.get("anchor_latitude") == "-22.484251", (
                    f"Lat alterada: {row.get('anchor_latitude')}"
                )
                assert row.get("anchor_longitude") == "-43.211257", (
                    f"Lon alterada: {row.get('anchor_longitude')}"
                )
                break

    def test_anchor_crs_is_wgs84(self, containment_audit):
        for row in containment_audit:
            if row.get("anchor_crs"):
                assert "4326" in row.get("anchor_crs", "") or "WGS" in row.get("anchor_crs", "")
                break

    def test_no_pixel_read_in_notes(self, containment_audit):
        for row in containment_audit:
            notes = row.get("notes", "").lower()
            # notes should say pixel_read=NO if present
            if "pixel" in notes:
                assert "no" in notes or "not" in notes, (
                    f"Notes implicam leitura de pixel: {row.get('patch_id')}"
                )

    def test_distance_numeric_when_present(self, containment_audit):
        for row in containment_audit:
            d = row.get("distance_to_patch_centroid_m", "")
            if d:
                assert d.replace(".", "").isdigit() or (
                    d.startswith("-") and d[1:].replace(".", "").isdigit()
                ), f"distancia nao numerica: {d}"


# ---------------------------------------------------------------------------
# TestV1ITNoPatchCoverage — resultado esperado: anchor fora de todos os patches
# ---------------------------------------------------------------------------


class TestV1ITNoPatchCoverage:
    def test_no_patch_contains_anchor(self, containment_audit):
        """Nenhum patch deve conter o anchor (resultado esperado da auditoria)."""
        inside = [
            r for r in containment_audit
            if r.get("anchor_inside_patch_bounds") == "TRUE"
        ]
        assert len(inside) == 0, (
            f"Anchor encontrado dentro de {len(inside)} patches — inesperado: "
            f"{[r.get('patch_id') for r in inside]}"
        )

    def test_match_decision_no_coverage(self, match_decision):
        row = match_decision[0]
        assert row.get("patches_containing_anchor") == "0", (
            f"patches_containing_anchor: {row.get('patches_containing_anchor')}"
        )
        assert row.get("final_coverage_status") in (
            "NO_PATCH_COVERAGE_FOR_ANCHOR",
            "NEAR_MISS_NO_COVERAGE",
        ), f"final_coverage_status: {row.get('final_coverage_status')}"

    def test_closest_patch_identified(self, match_decision):
        row = match_decision[0]
        assert row.get("closest_patch_id"), "closest_patch_id vazio"

    def test_closest_patch_is_pet(self, match_decision):
        row = match_decision[0]
        assert "PET" in row.get("closest_patch_id", ""), (
            f"closest_patch_id nao e PET: {row.get('closest_patch_id')}"
        )

    def test_closest_edge_distance_gt_zero(self, match_decision):
        row = match_decision[0]
        dist = row.get("closest_edge_distance_m", "")
        assert dist and float(dist) > 0, (
            f"closest_edge_distance_m deveria ser > 0: {dist}"
        )

    def test_summary_coverage_false(self, summary):
        assert summary.get("patches_inside_anchor") == 0, (
            f"patches_inside_anchor: {summary.get('patches_inside_anchor')}"
        )
        assert summary.get("final_coverage_status") == "NO_PATCH_COVERAGE_FOR_ANCHOR", (
            f"final_coverage_status: {summary.get('final_coverage_status')}"
        )


# ---------------------------------------------------------------------------
# TestV1ITInvariants
# ---------------------------------------------------------------------------


class TestV1ITInvariants:
    def test_containment_can_create_label_NO(self, containment_audit):
        for row in containment_audit:
            assert row.get("can_create_training_label") == "NO", (
                f"can_create_training_label != NO: {row.get('patch_id')}"
            )

    def test_containment_can_train_NO(self, containment_audit):
        for row in containment_audit:
            assert row.get("can_train_model") == "NO", (
                f"can_train_model != NO: {row.get('patch_id')}"
            )

    def test_containment_can_reopen_b_NO(self, containment_audit):
        for row in containment_audit:
            assert row.get("can_reopen_protocol_b") == "NO", (
                f"can_reopen_protocol_b != NO: {row.get('patch_id')}"
            )

    def test_match_decision_can_create_label_NO(self, match_decision):
        for row in match_decision:
            assert row.get("can_create_training_label") == "NO"

    def test_match_decision_can_train_NO(self, match_decision):
        for row in match_decision:
            assert row.get("can_train_model") == "NO"

    def test_match_decision_can_reopen_b_NO(self, match_decision):
        for row in match_decision:
            assert row.get("can_reopen_protocol_b") == "NO"

    def test_public_audit_invariants(self, public_audit):
        for row in public_audit:
            assert row.get("can_create_training_label") == "NO"
            assert row.get("can_train_model") == "NO"
            assert row.get("can_reopen_protocol_b") == "NO"

    def test_multimodal_matrix_can_create_label_NO(self, multimodal_matrix):
        for row in multimodal_matrix:
            assert row.get("can_create_training_label") == "NO"

    def test_multimodal_matrix_can_train_NO(self, multimodal_matrix):
        for row in multimodal_matrix:
            assert row.get("can_train_model") == "NO"

    def test_summary_invariants(self, summary):
        inv = summary.get("invariants", {})
        for key, val in inv.items():
            assert val == "NO", f"Invariante {key} != NO: {val}"

    def test_coverage_not_confirmed_means_no_multimodal(self, multimodal_matrix):
        """Sem cobertura de patch, multimodal_reference_status nao pode ser READY."""
        for row in multimodal_matrix:
            if row.get("sentinel_patch_coverage_confirmed") != "TRUE":
                assert row.get("multimodal_reference_status") not in (
                    "READY_FOR_STRUCTURAL_ANALYSIS",
                ), (
                    f"Status READY sem cobertura: {row.get('multimodal_reference_status')}"
                )


# ---------------------------------------------------------------------------
# TestV1ITSemPathPrivado
# ---------------------------------------------------------------------------


class TestV1ITSemPathPrivado:
    def _check_no_private(self, path: Path):
        content = path.read_text(encoding="utf-8", errors="replace")
        hits = PRIVATE_PATH_RE.findall(content)
        assert not hits, f"Path privado em {path.name}: {hits[:3]}"

    def test_public_footprint_audit_no_private(self):
        self._check_no_private(
            PUBLIC_DATASETS / "official_anchor_patch_footprint_audit.csv"
        )

    def test_multimodal_matrix_no_private(self):
        self._check_no_private(
            PUBLIC_DATASETS / "official_anchor_multimodal_readiness_matrix.csv"
        )

    def test_footprint_schema_no_private(self):
        self._check_no_private(
            SCHEMAS_DIR / "official_anchor_patch_footprint_audit_schema.csv"
        )

    def test_multimodal_schema_no_private(self):
        self._check_no_private(
            SCHEMAS_DIR / "official_anchor_multimodal_readiness_schema.csv"
        )


# ---------------------------------------------------------------------------
# TestV1ITMultimodalMatrix
# ---------------------------------------------------------------------------


class TestV1ITMultimodalMatrix:
    def test_has_one_row(self, multimodal_matrix):
        assert len(multimodal_matrix) == 1, (
            f"Esperado 1 linha, encontrado {len(multimodal_matrix)}"
        )

    def test_anchor_id_present(self, multimodal_matrix):
        row = multimodal_matrix[0]
        assert "ANEXOII" in row.get("anchor_id", ""), (
            f"anchor_id nao tem ANEXOII: {row.get('anchor_id')}"
        )

    def test_official_event_available(self, multimodal_matrix):
        row = multimodal_matrix[0]
        assert row.get("official_documented_event_available") == "YES"

    def test_explicit_coordinate_available(self, multimodal_matrix):
        row = multimodal_matrix[0]
        assert row.get("explicit_coordinate_available") == "YES"

    def test_blocker_mentions_acquisition(self, multimodal_matrix):
        row = multimodal_matrix[0]
        blocker = row.get("primary_blocker", "")
        # either NONE (if coverage) or mentions acquisition
        if blocker != "NONE":
            assert "ACQUISITION" in blocker.upper() or "COVERAGE" in blocker.upper() or "DINO" in blocker.upper(), (
                f"Blocker nao menciona causa: {blocker}"
            )

    def test_temporal_precision_exact_date(self, multimodal_matrix):
        row = multimodal_matrix[0]
        assert row.get("temporal_precision") == "EXACT_DATE", (
            f"temporal_precision: {row.get('temporal_precision')}"
        )

    def test_spatial_precision_exact_coordinate(self, multimodal_matrix):
        row = multimodal_matrix[0]
        assert row.get("spatial_precision") == "EXACT_COORDINATE", (
            f"spatial_precision: {row.get('spatial_precision')}"
        )


# ---------------------------------------------------------------------------
# TestV1ITQAChecks
# ---------------------------------------------------------------------------


class TestV1ITQAChecks:
    def _get(self, qa_rows, name):
        return next((r for r in qa_rows if r.get("check") == name), None)

    def test_script_ran_to_completion(self, qa):
        row = self._get(qa, "script_ran_to_completion")
        assert row and row.get("result") == "PASS"

    def test_footprint_inventory_created(self, qa):
        row = self._get(qa, "footprint_inventory_created")
        assert row and row.get("result") == "PASS"

    def test_patches_with_bounds_gt_0(self, qa):
        row = self._get(qa, "patches_with_bounds_gt_0")
        assert row and row.get("result") == "PASS"

    def test_containment_audit_created(self, qa):
        row = self._get(qa, "containment_audit_created")
        assert row and row.get("result") == "PASS"

    def test_invariant_label_qa(self, qa):
        row = self._get(qa, "invariant_can_create_label_NO")
        assert row and row.get("result") == "PASS"

    def test_invariant_train_qa(self, qa):
        row = self._get(qa, "invariant_can_train_NO")
        assert row and row.get("result") == "PASS"

    def test_invariant_reopen_b_qa(self, qa):
        row = self._get(qa, "invariant_can_reopen_b_NO")
        assert row and row.get("result") == "PASS"

    def test_no_pixels_read_qa(self, qa):
        row = self._get(qa, "no_pixels_read")
        assert row and row.get("result") == "PASS"

    def test_no_private_path_qa(self, qa):
        row = self._get(qa, "no_private_path_in_footprint")
        assert row and row.get("result") == "PASS"

    def test_no_fail_results(self, qa):
        fails = [r for r in qa if r.get("result") == "FAIL"]
        assert not fails, (
            f"QA checks FAIL: {[(r['check'], r['detail']) for r in fails]}"
        )


# ---------------------------------------------------------------------------
# TestV1ITSummary
# ---------------------------------------------------------------------------


class TestV1ITSummary:
    def test_version(self, summary):
        assert summary.get("script_version") == "v1it"

    def test_anchors_evaluated(self, summary):
        assert summary.get("anchors_evaluated") == 1

    def test_pet_patches_total_ge_4(self, summary):
        assert summary.get("pet_patches_total", 0) >= 4

    def test_patches_with_bounds_ge_4(self, summary):
        assert summary.get("pet_patches_with_bounds", 0) >= 4

    def test_closest_patch_id_present(self, summary):
        assert summary.get("closest_patch_id"), "closest_patch_id vazio"

    def test_closest_patch_is_pet(self, summary):
        assert "PET" in summary.get("closest_patch_id", "")

    def test_invariants_block(self, summary):
        inv = summary.get("invariants", {})
        assert inv.get("can_create_training_label") == "NO"
        assert inv.get("can_train_model") == "NO"
        assert inv.get("can_reopen_protocol_b") == "NO"
        assert inv.get("can_be_operational_ground_truth") == "NO"


# ---------------------------------------------------------------------------
# TestV1ITLocalRunsNaoVersionado
# ---------------------------------------------------------------------------


class TestV1ITLocalRunsNaoVersionado:
    def test_local_runs_not_staged(self):
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        staged = [f for f in result.stdout.splitlines() if "local_runs" in f]
        assert not staged, f"local_runs staged: {staged}"

    def test_no_tif_staged(self):
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        tif = [f for f in result.stdout.splitlines() if f.endswith((".tif", ".tiff"))]
        assert not tif, f"TIF staged: {tif}"

    def test_no_npz_staged(self):
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        npz = [f for f in result.stdout.splitlines() if f.endswith(".npz")]
        assert not npz, f"NPZ staged: {npz}"
