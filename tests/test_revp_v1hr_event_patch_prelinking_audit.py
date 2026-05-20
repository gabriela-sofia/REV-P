"""
Audit tests for v1hr — Pré-ligação Evento–Patch e Geocodificação Manual Controlada.

This test suite verifies that all v1hr artifacts are present, structurally correct,
and free of dangerous methodological claims. It does NOT execute any overlay,
geocoding, or data acquisition — it only audits metadata artifacts.

Ground truth operacional: NOT_ESTABLISHED
Protocolo B: BLOCKED
Multimodal: HOLD
DINO: SUPPORT_ONLY
"""

import csv
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

BASE = Path(__file__).parent.parent
DATASETS = BASE / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = BASE / "docs"
TEMPLATES = DOCS / "templates"
METODOLOGIA = DOCS / "metodologia_cientifica"

EXPECTED_EVENTS = {
    "REC_2022_05_24_30", "REC_2023_02_05_06", "REC_2024_06_14_16",
    "PET_2022_02_15", "PET_2022_03_20_21", "PET_2024_03_21_28",
    "CTB_2022_01_15_16", "CTB_2023_10_28_30", "CTB_2024_02_18_20",
}

PETROPOLIS_EVENTS = {"PET_2022_02_15", "PET_2022_03_20_21", "PET_2024_03_21_28"}

REQUIRED_DEPENDENCY_TYPES = {
    "SOURCE_GEOMETRY",
    "MANUAL_GEOCODING",
    "LICENSE_PROVENANCE",
    "SENTINEL_TEMPORAL_SEARCH",
    "HUMAN_REVIEW",
}

DANGEROUS_PATTERNS = [
    r"ground truth operacional\s+(estabelecido|validado|confirmado|criado)",
    r"(flood|inunda\w+)\s+(label|detection|prediction)\s+(criado|estabelecido|confirmado|validado)",
    r"training\s+label\s+(criado|gerado|estabelecido)",
    r"supervised\s+training\s+(iniciado|autorizado|realizado)",
    r"patch.{0,40}(validado|confirmado)\s+(como|como sendo)\s+inunda\w+",
    r"overlay\s+(executado|realizado|conclu\w+)",
    r"geocodifica\w+\s+autom\w+\s+realizada",
    r"coordenada\s+(criada|gerada|definida)\s+automaticamente",
]

SAFE_CONTEXTS = [
    "forbidden_use",
    "forbidden_claim",
    "bloqueado",
    "blocked",
    "não cria",
    "não gera",
    "nenhum",
    "proibido",
    "cannot",
    "not_established",
    "hold",
    "sempre incluir",
    "claim_proibido",
    "nunca",
    "não pode",
    "não permite",
    "não executa",
    "não estabelece",
]

NEW_MD_FILES = [
    METODOLOGIA / "protocolo_c_pre_ligacao_evento_patch.md",
    TEMPLATES / "protocolo_c_ficha_geocodificacao_manual.md",
    TEMPLATES / "protocolo_c_revisao_pre_overlay_evento_patch.md",
]


# ===========================================================================
# 1 — Files exist
# ===========================================================================


class TestFilesExist:
    def test_event_patch_linking_preflight_registry(self):
        assert (DATASETS / "event_patch_linking_preflight_registry.csv").exists()

    def test_manual_geocoding_target_registry(self):
        assert (DATASETS / "manual_geocoding_target_registry.csv").exists()

    def test_event_sentinel_temporal_window_registry(self):
        assert (DATASETS / "event_sentinel_temporal_window_registry.csv").exists()

    def test_patch_linking_dependency_registry(self):
        assert (DATASETS / "patch_linking_dependency_registry.csv").exists()

    def test_event_patch_linking_preflight_schema(self):
        assert (SCHEMAS / "event_patch_linking_preflight_schema.csv").exists()

    def test_manual_geocoding_target_schema(self):
        assert (SCHEMAS / "manual_geocoding_target_schema.csv").exists()

    def test_event_sentinel_temporal_window_schema(self):
        assert (SCHEMAS / "event_sentinel_temporal_window_schema.csv").exists()

    def test_patch_linking_dependency_schema(self):
        assert (SCHEMAS / "patch_linking_dependency_schema.csv").exists()

    def test_template_geocodificacao_manual(self):
        assert (TEMPLATES / "protocolo_c_ficha_geocodificacao_manual.md").exists()

    def test_template_revisao_pre_overlay(self):
        assert (TEMPLATES / "protocolo_c_revisao_pre_overlay_evento_patch.md").exists()

    def test_methodology_doc_pre_ligacao(self):
        assert (METODOLOGIA / "protocolo_c_pre_ligacao_evento_patch.md").exists()


# ===========================================================================
# 2 — Schema fields
# ===========================================================================


class TestSchemaFields:
    def _schema_fields(self, schema_file):
        with open(SCHEMAS / schema_file, encoding="utf-8") as f:
            return [row["field_name"] for row in csv.DictReader(f)]

    def test_preflight_schema_required_fields(self):
        fields = self._schema_fields("event_patch_linking_preflight_schema.csv")
        for required in [
            "preflight_id", "observed_event_id", "region", "patch_scope",
            "promotion_allowed", "can_create_training_label", "protocol_b_status",
            "multimodal_status", "dino_usage_status", "pre_link_status",
            "patch_overlay_status", "forbidden_claim",
        ]:
            assert required in fields, f"Missing from preflight schema: {required}"

    def test_geocoding_schema_required_fields(self):
        fields = self._schema_fields("manual_geocoding_target_schema.csv")
        for required in [
            "geocoding_target_id", "observed_event_id", "locality_name",
            "geocoding_status", "requires_official_confirmation",
            "cannot_establish_ground_truth_alone", "forbidden_use",
        ]:
            assert required in fields, f"Missing from geocoding schema: {required}"

    def test_sentinel_window_schema_required_fields(self):
        fields = self._schema_fields("event_sentinel_temporal_window_schema.csv")
        for required in [
            "temporal_window_id", "observed_event_id",
            "pre_event_window_start", "pre_event_window_end",
            "event_window_start", "event_window_end",
            "post_event_window_start", "post_event_window_end",
            "sentinel_1_relevance", "acquisition_status",
            "cannot_establish_ground_truth_alone", "forbidden_use",
        ]:
            assert required in fields, f"Missing from sentinel window schema: {required}"

    def test_dependency_schema_required_fields(self):
        fields = self._schema_fields("patch_linking_dependency_schema.csv")
        for required in [
            "dependency_id", "observed_event_id", "dependency_type",
            "required_before_overlay", "required_before_ground_reference",
            "current_status", "forbidden_if_missing",
        ]:
            assert required in fields, f"Missing from dependency schema: {required}"


# ===========================================================================
# 3 — Preflight registry
# ===========================================================================


class TestPreflightRegistry:
    def _load(self):
        with open(DATASETS / "event_patch_linking_preflight_registry.csv", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_minimum_row_count(self):
        assert len(self._load()) >= 9

    def test_all_events_covered(self):
        event_ids = {r["observed_event_id"] for r in self._load()}
        for event_id in EXPECTED_EVENTS:
            assert event_id in event_ids, f"Event missing from preflight: {event_id}"

    def test_promotion_always_false(self):
        for row in self._load():
            assert row["promotion_allowed"] == "false", \
                f"promotion_allowed != false for {row['preflight_id']}"

    def test_training_label_always_false(self):
        for row in self._load():
            assert row["can_create_training_label"] == "false", \
                f"can_create_training_label != false for {row['preflight_id']}"

    def test_protocol_b_always_blocked(self):
        for row in self._load():
            assert row["protocol_b_status"] == "BLOCKED", \
                f"protocol_b_status != BLOCKED for {row['preflight_id']}"

    def test_multimodal_always_hold(self):
        for row in self._load():
            assert row["multimodal_status"] == "HOLD", \
                f"multimodal_status != HOLD for {row['preflight_id']}"

    def test_dino_always_support_only(self):
        for row in self._load():
            assert row["dino_usage_status"] == "SUPPORT_ONLY", \
                f"dino_usage_status != SUPPORT_ONLY for {row['preflight_id']}"

    def test_no_overlay_executed(self):
        for row in self._load():
            assert row["patch_overlay_status"] != "EXECUTED", \
                f"patch_overlay_status=EXECUTED for {row['preflight_id']}"

    def test_forbidden_claim_not_empty(self):
        for row in self._load():
            assert row["forbidden_claim"].strip(), \
                f"Empty forbidden_claim for {row['preflight_id']}"

    def test_pet_2024_not_ready(self):
        rows = [r for r in self._load() if r["observed_event_id"] == "PET_2024_03_21_28"]
        assert rows, "PET_2024_03_21_28 missing from preflight"
        for r in rows:
            assert r["pre_link_status"] in {
                "NOT_READY_FOR_PATCH_LINKING", "BLOCKED_PENDING_GEOMETRY"
            }, f"PET_2024 should be NOT_READY or BLOCKED, got {r['pre_link_status']}"

    def test_region_level_scope_exists_for_all_events(self):
        region_level = [r for r in self._load() if r["patch_scope"] == "REGION_LEVEL"]
        event_ids = {r["observed_event_id"] for r in region_level}
        for event_id in EXPECTED_EVENTS:
            assert event_id in event_ids, \
                f"No REGION_LEVEL preflight row for event {event_id}"


# ===========================================================================
# 4 — Geocoding target registry
# ===========================================================================


class TestGeocodingTargetRegistry:
    def _load(self):
        with open(DATASETS / "manual_geocoding_target_registry.csv", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_minimum_total_targets(self):
        assert len(self._load()) >= 22

    def test_recife_minimum_targets(self):
        recife = [r for r in self._load() if r["region"] == "Recife"]
        assert len(recife) >= 6, f"Recife has only {len(recife)} geocoding targets"

    def test_petropolis_minimum_targets(self):
        pet = [r for r in self._load() if r["region"] == "Petrópolis"]
        assert len(pet) >= 8, f"Petrópolis has only {len(pet)} geocoding targets"

    def test_curitiba_minimum_targets(self):
        ctr = [r for r in self._load() if r["region"] == "Curitiba"]
        assert len(ctr) >= 4, f"Curitiba has only {len(ctr)} geocoding targets"

    def test_geocoding_status_never_geocoded(self):
        for row in self._load():
            assert row["geocoding_status"] != "GEOCODED", \
                f"geocoding_status=GEOCODED for {row['geocoding_target_id']}"

    def test_requires_official_confirmation_always_true(self):
        for row in self._load():
            assert row["requires_official_confirmation"] == "true", \
                f"requires_official_confirmation != true for {row['geocoding_target_id']}"

    def test_cannot_establish_ground_truth_always_true(self):
        for row in self._load():
            assert row["cannot_establish_ground_truth_alone"] == "true", \
                f"cannot_establish_ground_truth_alone != true for {row['geocoding_target_id']}"

    def test_forbidden_use_not_empty(self):
        for row in self._load():
            assert row["forbidden_use"].strip(), \
                f"Empty forbidden_use for {row['geocoding_target_id']}"

    def test_all_events_have_at_least_one_geocoding_target(self):
        event_ids = {r["observed_event_id"] for r in self._load()}
        for event_id in EXPECTED_EVENTS:
            assert event_id in event_ids, \
                f"No geocoding targets for event {event_id}"

    def test_no_coordinate_pattern_in_locality_name(self):
        coord_re = re.compile(r"-?\d{1,3}\.\d{4,}")
        for row in self._load():
            assert not coord_re.search(row["locality_name"]), \
                f"Coordinate-like value in locality_name: {row['locality_name']}"

    def test_no_coordinate_pattern_in_notes(self):
        coord_re = re.compile(r"-?\d{1,3}\.\d{4,}")
        for row in self._load():
            notes = row.get("notes", "")
            assert not coord_re.search(notes), \
                f"Coordinate-like value in notes for {row['geocoding_target_id']}: {notes[:80]}"


# ===========================================================================
# 5 — Sentinel temporal window registry
# ===========================================================================


class TestSentinelTemporalWindows:
    def _load(self):
        with open(DATASETS / "event_sentinel_temporal_window_registry.csv", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_exactly_nine_windows(self):
        rows = self._load()
        assert len(rows) == 9, f"Expected 9 sentinel windows, got {len(rows)}"

    def test_one_window_per_event(self):
        event_ids = [r["observed_event_id"] for r in self._load()]
        assert len(event_ids) == len(set(event_ids)), \
            "Duplicate event IDs in sentinel windows"

    def test_all_events_have_window(self):
        event_ids = {r["observed_event_id"] for r in self._load()}
        for event_id in EXPECTED_EVENTS:
            assert event_id in event_ids, f"No sentinel window for event {event_id}"

    def test_acquisition_status_not_acquired(self):
        for row in self._load():
            assert row["acquisition_status"] == "NOT_ACQUIRED", \
                f"acquisition_status != NOT_ACQUIRED for {row['temporal_window_id']}"

    def test_sentinel_1_relevance_high(self):
        for row in self._load():
            assert row["sentinel_1_relevance"] == "HIGH", \
                f"sentinel_1_relevance != HIGH for {row['temporal_window_id']}"

    def test_cannot_establish_ground_truth_always_true(self):
        for row in self._load():
            assert row["cannot_establish_ground_truth_alone"] == "true", \
                f"cannot_establish_ground_truth_alone != true for {row['temporal_window_id']}"

    def test_can_support_future_review_true(self):
        for row in self._load():
            assert row["can_support_future_review"] == "true", \
                f"can_support_future_review != true for {row['temporal_window_id']}"

    def test_pre_event_window_dates(self):
        for row in self._load():
            event_start = date.fromisoformat(row["event_date_start"])
            pre_start = date.fromisoformat(row["pre_event_window_start"])
            pre_end = date.fromisoformat(row["pre_event_window_end"])
            assert pre_start == event_start - timedelta(days=14), \
                f"pre_event_window_start wrong for {row['temporal_window_id']}"
            assert pre_end == event_start - timedelta(days=1), \
                f"pre_event_window_end wrong for {row['temporal_window_id']}"

    def test_post_event_window_dates(self):
        for row in self._load():
            event_end = date.fromisoformat(row["event_date_end"])
            post_start = date.fromisoformat(row["post_event_window_start"])
            post_end = date.fromisoformat(row["post_event_window_end"])
            assert post_start == event_end + timedelta(days=1), \
                f"post_event_window_start wrong for {row['temporal_window_id']}"
            assert post_end == event_end + timedelta(days=14), \
                f"post_event_window_end wrong for {row['temporal_window_id']}"

    def test_event_window_matches_event_dates(self):
        for row in self._load():
            assert row["event_window_start"] == row["event_date_start"], \
                f"event_window_start != event_date_start for {row['temporal_window_id']}"
            assert row["event_window_end"] == row["event_date_end"], \
                f"event_window_end != event_date_end for {row['temporal_window_id']}"

    def test_forbidden_use_not_empty(self):
        for row in self._load():
            assert row["forbidden_use"].strip(), \
                f"Empty forbidden_use for {row['temporal_window_id']}"

    def test_temporal_alignment_status_prepared_metadata_only(self):
        for row in self._load():
            assert row["temporal_alignment_status"] == "PREPARED_METADATA_ONLY", \
                f"temporal_alignment_status != PREPARED_METADATA_ONLY for {row['temporal_window_id']}"


# ===========================================================================
# 6 — Dependency registry
# ===========================================================================


class TestDependencyRegistry:
    def _load(self):
        with open(DATASETS / "patch_linking_dependency_registry.csv", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_minimum_dependency_count(self):
        assert len(self._load()) >= 45, \
            f"Expected at least 45 dependencies, got {len(self._load())}"

    def test_no_closed_dependencies(self):
        for row in self._load():
            assert row["current_status"] != "CLOSED", \
                f"Closed dependency found: {row['dependency_id']}"

    def test_all_events_have_required_dependency_types(self):
        rows = self._load()
        for event_id in EXPECTED_EVENTS:
            event_deps = {r["dependency_type"] for r in rows if r["observed_event_id"] == event_id}
            for dep_type in REQUIRED_DEPENDENCY_TYPES:
                assert dep_type in event_deps, \
                    f"Event {event_id} missing dependency type {dep_type}"

    def test_petropolis_has_phenomenon_separation(self):
        rows = self._load()
        for event_id in PETROPOLIS_EVENTS:
            event_deps = {r["dependency_type"] for r in rows if r["observed_event_id"] == event_id}
            assert "PHENOMENON_SEPARATION" in event_deps, \
                f"Petrópolis event {event_id} missing PHENOMENON_SEPARATION"

    def test_required_before_ground_reference_always_true(self):
        for row in self._load():
            assert row["required_before_ground_reference"] == "true", \
                f"required_before_ground_reference != true for {row['dependency_id']}"

    def test_forbidden_if_missing_not_empty(self):
        for row in self._load():
            assert row["forbidden_if_missing"].strip(), \
                f"Empty forbidden_if_missing for {row['dependency_id']}"

    def test_all_events_have_dependencies(self):
        rows = self._load()
        event_ids_with_deps = {r["observed_event_id"] for r in rows}
        for event_id in EXPECTED_EVENTS:
            assert event_id in event_ids_with_deps, \
                f"No dependencies for event {event_id}"

    def test_human_review_required_before_ground_reference(self):
        rows = self._load()
        for event_id in EXPECTED_EVENTS:
            human_reviews = [
                r for r in rows
                if r["observed_event_id"] == event_id
                and r["dependency_type"] == "HUMAN_REVIEW"
            ]
            assert human_reviews, f"No HUMAN_REVIEW dependency for {event_id}"
            for hr in human_reviews:
                assert hr["required_before_ground_reference"] == "true"


# ===========================================================================
# 7 — Templates
# ===========================================================================


class TestTemplates:
    def _read(self, filename):
        return (TEMPLATES / filename).read_text(encoding="utf-8")

    def test_geocodificacao_manual_has_key_placeholders(self):
        content = self._read("protocolo_c_ficha_geocodificacao_manual.md")
        for placeholder in [
            "[GEOCODING_TARGET_ID]", "[OBSERVED_EVENT_ID]", "[REGIAO]",
            "[LOCALIDADE]", "[TIPO_DE_LOCALIDADE]", "[FONTE]", "[URL]",
            "[CRS_ESPERADO]", "[LICENCA]", "[REVISOR_FUNCAO]",
        ]:
            assert placeholder in content, f"Placeholder missing: {placeholder}"

    def test_geocodificacao_manual_warns_against_approximate_geocoding(self):
        content = self._read("protocolo_c_ficha_geocodificacao_manual.md")
        assert "buscador online" in content or "aproximada" in content.lower() or \
               "oficial" in content.lower()

    def test_geocodificacao_manual_blocks_ground_truth(self):
        content = self._read("protocolo_c_ficha_geocodificacao_manual.md")
        assert "ground truth" in content.lower()
        assert "cannot_establish_ground_truth_alone" in content or \
               "nunca" in content.lower() or "never" in content.lower()

    def test_revisao_pre_overlay_has_key_placeholders(self):
        content = self._read("protocolo_c_revisao_pre_overlay_evento_patch.md")
        for placeholder in [
            "[PRE_OVERLAY_REVIEW_ID]", "[PATCH_IDS]", "[SENTINEL_STATUS]",
            "[PODE_EXECUTAR_OVERLAY_FUTURO]", "[DECISAO]", "[REVISOR_FUNCAO]",
        ]:
            assert placeholder in content, f"Placeholder missing: {placeholder}"

    def test_revisao_pre_overlay_promotion_always_false(self):
        content = self._read("protocolo_c_revisao_pre_overlay_evento_patch.md")
        assert "false" in content
        assert "ground reference" in content.lower()

    def test_revisao_pre_overlay_has_decision_options(self):
        content = self._read("protocolo_c_revisao_pre_overlay_evento_patch.md")
        for decision in [
            "READY_FOR_FUTURE_OVERLAY",
            "REQUEST_SOURCE_GEOMETRY",
            "BLOCK_PATCH_LINKING",
            "BLOCK_OPERATIONAL_USE",
        ]:
            assert decision in content, f"Decision option missing: {decision}"

    def test_revisao_pre_overlay_blocks_label(self):
        content = self._read("protocolo_c_revisao_pre_overlay_evento_patch.md")
        assert "label de treino" in content.lower() or "training label" in content.lower()
        assert "false" in content


# ===========================================================================
# 8 — Methodology documents
# ===========================================================================


class TestMethodologyDocuments:
    def _read(self, filename):
        return (METODOLOGIA / filename).read_text(encoding="utf-8")

    def test_pre_ligacao_doc_states_metadata_only(self):
        content = self._read("protocolo_c_pre_ligacao_evento_patch.md")
        assert "metadata-only" in content.lower() or "metadata only" in content.lower()

    def test_pre_ligacao_doc_mentions_no_overlay(self):
        content = self._read("protocolo_c_pre_ligacao_evento_patch.md")
        assert "overlay" in content.lower()

    def test_pre_ligacao_doc_blocks_ground_truth(self):
        content = self._read("protocolo_c_pre_ligacao_evento_patch.md")
        assert "ground truth operacional" in content.lower()

    def test_pre_ligacao_doc_protocol_b_blocked(self):
        content = self._read("protocolo_c_pre_ligacao_evento_patch.md")
        assert "BLOCKED" in content or "bloqueado" in content.lower()

    def test_pre_ligacao_doc_lists_all_output_registries(self):
        content = self._read("protocolo_c_pre_ligacao_evento_patch.md")
        for registry in [
            "event_patch_linking_preflight_registry",
            "manual_geocoding_target_registry",
            "event_sentinel_temporal_window_registry",
            "patch_linking_dependency_registry",
        ]:
            assert registry in content, f"Registry not mentioned in pre-linking doc: {registry}"

    def test_referencias_observacionais_references_v1hr(self):
        content = self._read("protocolo_c_referencias_observacionais_candidatas.md")
        assert "v1hr" in content or "pré-ligação" in content.lower() \
               or "pre_ligacao" in content.lower() or "pre-ligacao" in content.lower()

    def test_dossies_doc_references_v1hr(self):
        content = self._read("protocolo_c_dossies_eventos_candidatos.md")
        assert "v1hr" in content or "pré-ligação" in content.lower() \
               or "pre_ligacao" in content.lower()


# ===========================================================================
# 9 — Security: no dangerous claims
# ===========================================================================


class TestNoDangerousClaims:
    @pytest.mark.parametrize("md_file", NEW_MD_FILES)
    def test_no_dangerous_claims_in_new_docs(self, md_file):
        text = md_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            line_lower = line.lower()
            if any(ctx in line_lower for ctx in SAFE_CONTEXTS):
                continue
            for pattern in DANGEROUS_PATTERNS:
                assert not re.search(pattern, line, re.IGNORECASE), \
                    f"Dangerous claim in {md_file.name}: {line.strip()[:120]}"

    def test_no_private_paths_in_registries(self):
        private_re = re.compile(r"C:\\\\Users\\\\|C:/Users/|/home/\w")
        for registry_name in [
            "event_patch_linking_preflight_registry.csv",
            "manual_geocoding_target_registry.csv",
            "event_sentinel_temporal_window_registry.csv",
            "patch_linking_dependency_registry.csv",
        ]:
            text = (DATASETS / registry_name).read_text(encoding="utf-8")
            assert not private_re.search(text), \
                f"Private path found in {registry_name}"

    def test_no_coordinates_in_geocoding_notes(self):
        coord_re = re.compile(r"-?\d{1,3}\.\d{4,}")
        with open(DATASETS / "manual_geocoding_target_registry.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                notes = row.get("notes", "")
                assert not coord_re.search(notes), \
                    f"Coordinate in notes for {row['geocoding_target_id']}: {notes[:80]}"

    def test_no_coordinates_in_geocoding_locality_name(self):
        coord_re = re.compile(r"-?\d{1,3}\.\d{4,}")
        with open(DATASETS / "manual_geocoding_target_registry.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert not coord_re.search(row["locality_name"]), \
                    f"Coordinate in locality_name: {row['locality_name']}"

    def test_dependency_registry_no_closed_status(self):
        with open(DATASETS / "patch_linking_dependency_registry.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["current_status"] != "CLOSED", \
                    f"CLOSED dependency: {row['dependency_id']}"

    def test_sentinel_windows_all_not_acquired(self):
        with open(DATASETS / "event_sentinel_temporal_window_registry.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["acquisition_status"] == "NOT_ACQUIRED", \
                    f"Non NOT_ACQUIRED in sentinel window: {row['temporal_window_id']}"

    def test_preflight_no_ground_truth_established_claim(self):
        with open(DATASETS / "event_patch_linking_preflight_registry.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                allowed = row.get("allowed_claim", "").lower()
                assert "ground truth operacional estabelecido" not in allowed, \
                    f"Dangerous allowed_claim in preflight: {row['preflight_id']}"
