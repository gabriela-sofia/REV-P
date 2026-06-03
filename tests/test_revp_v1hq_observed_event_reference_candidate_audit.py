"""
v1hq — Referências observacionais candidatas.
Audita documentos, schemas, registries e templates da primeira camada
de eventos observados candidatos do Protocolo C.
"""
import csv
import os
import re
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..")
DATASETS = os.path.join(BASE, "datasets")
SCHEMAS = os.path.join(DATASETS, "schemas")
DOCS = os.path.join(BASE, "docs", "metodologia_cientifica")
TEMPLATES = os.path.join(BASE, "docs", "templates")

CANDIDATE_REG = os.path.join(DATASETS, "observed_event_reference_candidate_registry.csv")
GAP_REG = os.path.join(DATASETS, "observed_event_reference_gap_registry.csv")
DECISION_REG = os.path.join(DATASETS, "observed_event_reference_decision_registry.csv")
MANUAL_REG = os.path.join(DATASETS, "manual_external_evidence_needed_registry.csv")

CANDIDATE_SCHEMA = os.path.join(SCHEMAS, "observed_event_reference_candidate_schema.csv")
GAP_SCHEMA = os.path.join(SCHEMAS, "observed_event_reference_gap_schema.csv")
DECISION_SCHEMA = os.path.join(SCHEMAS, "observed_event_reference_decision_schema.csv")
MANUAL_SCHEMA = os.path.join(SCHEMAS, "manual_external_evidence_needed_schema.csv")

DOC_REFS = os.path.join(DOCS, "protocolo_c_referencias_observacionais_candidatas.md")
DOC_DIAG = os.path.join(DOCS, "protocolo_c_diagnostico_dados_externos_validos.md")

TEMPLATE_INTAKE = os.path.join(TEMPLATES, "protocolo_c_intake_fonte_observacional_manual.md")
TEMPLATE_REVIEW = os.path.join(TEMPLATES, "protocolo_c_revisao_evento_observado.md")

VALID_REGIONS = {"Recife", "Petrópolis", "Curitiba"}
EXPECTED_EVENTS = 9
EVENTS_PER_REGION = 3
MIN_HIGH_PRIORITY = 3

DANGEROUS_PATTERNS = [
    r"flood prediction",
    r"flood detection",
    r"predição de enchente",
    r"detecção de enchente",
    r"operational ground truth established",
    r"ground truth operacional estabelecido",
    r"flood label[^s]",
    r"label de enchente",
    r"training label[^s]",
    r"label de treino[^s]",
    r"supervised training[^;,]",
    r"supervised validation",
    r"modelo preditivo[^;,]",
    r"detecção operacional[^;,]",
]

SAFE_CONTEXTS = [
    "forbidden_use",
    "forbidden_claim",
    "bloqueado",
    "bloqueada",
    "blocked",
    "não cria",
    "não gera",
    "não está",
    "não treina",
    "não prediz",
    "nenhum",
    "proibido",
    "sem ",
    "cannot",
    "not established",
    "NOT_ESTABLISHED",
    "hold",
    "HOLD",
    "BLOCKED",
    "sempre incluir",
    "claim_proibido",
    "add other",
    "adicionar outros",
]


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _check_dangerous_claims(text, filepath):
    lines = text.splitlines()
    violations = []
    for i, line in enumerate(lines, 1):
        line_lower = line.lower()
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, line_lower):
                is_safe = any(ctx in line_lower for ctx in SAFE_CONTEXTS)
                if not is_safe:
                    violations.append(f"  Line {i}: {line.strip()[:120]}")
    return violations


# ─── 1. Arquivos existem ────────────────────────────────────────────────────

class TestFilesExist:
    def test_doc_refs_exists(self):
        assert os.path.exists(DOC_REFS), f"Missing: {DOC_REFS}"

    def test_doc_diag_exists(self):
        assert os.path.exists(DOC_DIAG), f"Missing: {DOC_DIAG}"

    def test_candidate_schema_exists(self):
        assert os.path.exists(CANDIDATE_SCHEMA)

    def test_gap_schema_exists(self):
        assert os.path.exists(GAP_SCHEMA)

    def test_decision_schema_exists(self):
        assert os.path.exists(DECISION_SCHEMA)

    def test_manual_schema_exists(self):
        assert os.path.exists(MANUAL_SCHEMA)

    def test_candidate_registry_exists(self):
        assert os.path.exists(CANDIDATE_REG)

    def test_gap_registry_exists(self):
        assert os.path.exists(GAP_REG)

    def test_decision_registry_exists(self):
        assert os.path.exists(DECISION_REG)

    def test_manual_registry_exists(self):
        assert os.path.exists(MANUAL_REG)

    def test_template_intake_exists(self):
        assert os.path.exists(TEMPLATE_INTAKE)

    def test_template_review_exists(self):
        assert os.path.exists(TEMPLATE_REVIEW)


# ─── 2. Schemas têm campos obrigatórios ─────────────────────────────────────

class TestSchemaFields:
    CANDIDATE_REQUIRED = [
        "observed_event_id", "region", "event_name", "event_type",
        "date_start", "date_end", "temporal_precision_level",
        "spatial_precision_level", "primary_source_type", "primary_source_name",
        "primary_source_url", "observed_event_confirmed", "source_traceable",
        "g1_event_confirmation", "g2_source_availability",
        "g3_temporal_alignment", "g4_spatial_alignment_triage",
        "priority_level", "ground_reference_candidate_status",
        "operational_ground_truth_status", "protocol_b_status",
        "multimodal_status", "dino_usage_status",
        "can_advance_to_source_review", "can_advance_to_patch_linking",
        "can_be_used_as_training_label", "can_reopen_protocol_b",
        "requires_review_gate", "missing_evidence",
    ]

    GAP_REQUIRED = [
        "gap_id", "observed_event_id", "region", "gap_type",
        "gap_description", "required_for_gate",
        "required_for_patch_level_reference", "priority_level",
        "suggested_action", "status",
    ]

    DECISION_REQUIRED = [
        "decision_id", "observed_event_id", "region", "decision_level",
        "decision_status", "decision_reason", "gates_closed",
        "can_promote_to_ground_reference", "can_generate_training_label",
        "can_reopen_protocol_b", "next_required_step", "review_owner",
    ]

    MANUAL_REQUIRED = [
        "needed_id", "region", "observed_event_id", "evidence_item",
        "evidence_category", "target_provider", "expected_format",
        "priority_level", "acquisition_mode", "local_only_if_acquired",
        "public_metadata_allowed", "license_review_required",
        "can_help_close_gate", "cannot_establish_ground_truth_alone",
        "current_status", "forbidden_use",
    ]

    def _check_schema(self, schema_path, required_fields):
        rows = _read_csv(schema_path)
        field_names = {r["field_name"] for r in rows}
        missing = [f for f in required_fields if f not in field_names]
        assert not missing, f"Missing schema fields in {os.path.basename(schema_path)}: {missing}"

    def test_candidate_schema_fields(self):
        self._check_schema(CANDIDATE_SCHEMA, self.CANDIDATE_REQUIRED)

    def test_gap_schema_fields(self):
        self._check_schema(GAP_SCHEMA, self.GAP_REQUIRED)

    def test_decision_schema_fields(self):
        self._check_schema(DECISION_SCHEMA, self.DECISION_REQUIRED)

    def test_manual_schema_fields(self):
        self._check_schema(MANUAL_SCHEMA, self.MANUAL_REQUIRED)


# ─── 3. Registry principal ──────────────────────────────────────────────────

class TestCandidateRegistry:
    @pytest.fixture(scope="class")
    def rows(self):
        return _read_csv(CANDIDATE_REG)

    def test_exactly_nine_events(self, rows):
        assert len(rows) == EXPECTED_EVENTS, f"Expected {EXPECTED_EVENTS} events, got {len(rows)}"

    def test_three_events_per_region(self, rows):
        from collections import Counter
        counts = Counter(r["region"] for r in rows)
        for region in VALID_REGIONS:
            assert counts[region] == EVENTS_PER_REGION, \
                f"Region '{region}' has {counts[region]} events, expected {EVENTS_PER_REGION}"

    def test_at_least_three_high_priority(self, rows):
        high = [r for r in rows if r["priority_level"] == "HIGH"]
        assert len(high) >= MIN_HIGH_PRIORITY, \
            f"Only {len(high)} HIGH priority events, need at least {MIN_HIGH_PRIORITY}"

    def test_all_observed_event_confirmed_true(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["observed_event_confirmed"] != "true"]
        assert not bad, f"observed_event_confirmed != true: {bad}"

    def test_all_source_traceable_true(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["source_traceable"] != "true"]
        assert not bad, f"source_traceable != true: {bad}"

    def test_all_g1_closed(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["g1_event_confirmation"] != "CLOSED"]
        assert not bad, f"G1 not CLOSED: {bad}"

    def test_all_g2_closed(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["g2_source_availability"] != "CLOSED"]
        assert not bad, f"G2 not CLOSED: {bad}"

    def test_all_g3_closed(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["g3_temporal_alignment"] != "CLOSED"]
        assert not bad, f"G3 not CLOSED: {bad}"

    def test_at_least_eight_g4_closed(self, rows):
        closed = [r for r in rows if r["g4_spatial_alignment_triage"] == "CLOSED"]
        assert len(closed) >= 8, \
            f"Only {len(closed)} events with G4 CLOSED, need at least 8"

    def test_pet_2024_has_g4_partial(self, rows):
        pet_2024 = next((r for r in rows if r["observed_event_id"] == "PET_2024_03_21_28"), None)
        assert pet_2024 is not None, "PET_2024_03_21_28 not found"
        assert pet_2024["g4_spatial_alignment_triage"] == "PARTIAL", \
            f"PET_2024_03_21_28 G4 should be PARTIAL, got {pet_2024['g4_spatial_alignment_triage']}"

    def test_all_operational_ground_truth_not_established(self, rows):
        bad = [r["observed_event_id"] for r in rows
               if r["operational_ground_truth_status"] != "NOT_ESTABLISHED"]
        assert not bad, f"operational_ground_truth_status != NOT_ESTABLISHED: {bad}"

    def test_all_protocol_b_blocked(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["protocol_b_status"] != "BLOCKED"]
        assert not bad, f"protocol_b_status != BLOCKED: {bad}"

    def test_all_multimodal_hold(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["multimodal_status"] != "HOLD"]
        assert not bad, f"multimodal_status != HOLD: {bad}"

    def test_all_dino_support_only(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["dino_usage_status"] != "SUPPORT_ONLY"]
        assert not bad, f"dino_usage_status != SUPPORT_ONLY: {bad}"

    def test_no_training_label(self, rows):
        bad = [r["observed_event_id"] for r in rows
               if r["can_be_used_as_training_label"] != "false"]
        assert not bad, f"can_be_used_as_training_label != false: {bad}"

    def test_no_reopen_protocol_b(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["can_reopen_protocol_b"] != "false"]
        assert not bad, f"can_reopen_protocol_b != false: {bad}"

    def test_all_require_review_gate(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["requires_review_gate"] != "true"]
        assert not bad, f"requires_review_gate != true: {bad}"

    def test_all_can_advance_to_source_review(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["can_advance_to_source_review"] != "true"]
        assert not bad, f"can_advance_to_source_review != true: {bad}"

    def test_no_patch_linking_allowed(self, rows):
        bad = [r["observed_event_id"] for r in rows if r["can_advance_to_patch_linking"] != "false"]
        assert not bad, f"can_advance_to_patch_linking != false: {bad}"


# ─── 4. Registry de lacunas ─────────────────────────────────────────────────

class TestGapRegistry:
    @pytest.fixture(scope="class")
    def candidate_ids(self):
        return {r["observed_event_id"] for r in _read_csv(CANDIDATE_REG)}

    @pytest.fixture(scope="class")
    def gap_rows(self):
        return _read_csv(GAP_REG)

    def test_all_event_ids_exist_in_candidate_registry(self, gap_rows, candidate_ids):
        unknown = {r["observed_event_id"] for r in gap_rows} - candidate_ids
        assert not unknown, f"Gap registry has unknown event IDs: {unknown}"

    def test_at_least_three_gaps_per_event(self, gap_rows, candidate_ids):
        from collections import Counter
        counts = Counter(r["observed_event_id"] for r in gap_rows)
        for event_id in candidate_ids:
            assert counts[event_id] >= 3, \
                f"Event {event_id} has only {counts[event_id]} gaps, need at least 3"

    def test_no_resolved_gaps(self, gap_rows):
        resolved = [r["gap_id"] for r in gap_rows if r["status"] not in ("OPEN", "PARTIAL", "BLOCKED")]
        assert not resolved, f"Gaps marked as resolved: {resolved}"

    def test_all_events_have_patch_overlay_gap(self, gap_rows, candidate_ids):
        events_with_overlay_gap = {r["observed_event_id"] for r in gap_rows
                                   if r["gap_type"] == "PATCH_OVERLAY_NOT_DONE"}
        missing = candidate_ids - events_with_overlay_gap
        assert not missing, f"Events without PATCH_OVERLAY_NOT_DONE gap: {missing}"

    def test_all_events_have_review_gate_gap(self, gap_rows, candidate_ids):
        events_with_review_gap = {r["observed_event_id"] for r in gap_rows
                                  if r["gap_type"] == "REVIEW_GATE_NOT_DONE"}
        missing = candidate_ids - events_with_review_gap
        assert not missing, f"Events without REVIEW_GATE_NOT_DONE gap: {missing}"

    def test_all_events_have_license_gap(self, gap_rows, candidate_ids):
        events_with_license_gap = {r["observed_event_id"] for r in gap_rows
                                   if r["gap_type"] == "LICENSE_PROVENANCE_PENDING"}
        missing = candidate_ids - events_with_license_gap
        assert not missing, f"Events without LICENSE_PROVENANCE_PENDING gap: {missing}"


# ─── 5. Registry de decisões ─────────────────────────────────────────────────

class TestDecisionRegistry:
    @pytest.fixture(scope="class")
    def candidate_ids(self):
        return {r["observed_event_id"] for r in _read_csv(CANDIDATE_REG)}

    @pytest.fixture(scope="class")
    def dec_rows(self):
        return _read_csv(DECISION_REG)

    def test_one_decision_per_event(self, dec_rows, candidate_ids):
        from collections import Counter
        counts = Counter(r["observed_event_id"] for r in dec_rows)
        for event_id in candidate_ids:
            assert counts[event_id] == 1, \
                f"Event {event_id} has {counts[event_id]} decisions, expected 1"

    def test_all_decision_ids_are_candidate_events(self, dec_rows, candidate_ids):
        unknown = {r["observed_event_id"] for r in dec_rows} - candidate_ids
        assert not unknown, f"Decision registry has unknown event IDs: {unknown}"

    def test_no_promote_to_ground_reference(self, dec_rows):
        bad = [r["observed_event_id"] for r in dec_rows
               if r["can_promote_to_ground_reference"] != "false"]
        assert not bad, f"can_promote_to_ground_reference != false: {bad}"

    def test_no_generate_training_label(self, dec_rows):
        bad = [r["observed_event_id"] for r in dec_rows
               if r["can_generate_training_label"] != "false"]
        assert not bad, f"can_generate_training_label != false: {bad}"

    def test_no_reopen_protocol_b(self, dec_rows):
        bad = [r["observed_event_id"] for r in dec_rows
               if r["can_reopen_protocol_b"] != "false"]
        assert not bad, f"can_reopen_protocol_b != false: {bad}"

    def test_pet_2024_needs_more_spatial_evidence(self, dec_rows):
        pet = next((r for r in dec_rows if r["observed_event_id"] == "PET_2024_03_21_28"), None)
        assert pet is not None, "PET_2024_03_21_28 decision not found"
        assert pet["decision_status"] == "NEEDS_MORE_SPATIAL_EVIDENCE", \
            f"PET_2024_03_21_28 decision_status should be NEEDS_MORE_SPATIAL_EVIDENCE"

    def test_no_protocol_b_reopening_in_decisions(self, dec_rows):
        bad = [r["observed_event_id"] for r in dec_rows
               if r["can_reopen_protocol_b"] != "false"]
        assert not bad, f"Decisions allowing Protocolo B reopening: {bad}"


# ─── 6. Registry de evidências manuais ──────────────────────────────────────

class TestManualEvidenceRegistry:
    @pytest.fixture(scope="class")
    def manual_rows(self):
        return _read_csv(MANUAL_REG)

    def test_contains_all_three_regions(self, manual_rows):
        regions = {r["region"] for r in manual_rows}
        for region in VALID_REGIONS:
            assert region in regions, f"Region '{region}' not found in manual evidence registry"

    def test_recife_has_at_least_six_items(self, manual_rows):
        recife_items = [r for r in manual_rows if r["region"] == "Recife"]
        assert len(recife_items) >= 6, \
            f"Recife has {len(recife_items)} manual evidence items, need at least 6"

    def test_petropolis_has_at_least_six_items(self, manual_rows):
        pet_items = [r for r in manual_rows if r["region"] == "Petrópolis"]
        assert len(pet_items) >= 6, \
            f"Petrópolis has {len(pet_items)} manual evidence items, need at least 6"

    def test_curitiba_has_at_least_six_items(self, manual_rows):
        ctb_items = [r for r in manual_rows if r["region"] == "Curitiba"]
        assert len(ctb_items) >= 6, \
            f"Curitiba has {len(ctb_items)} manual evidence items, need at least 6"

    def test_all_cannot_establish_ground_truth_alone(self, manual_rows):
        bad = [r["needed_id"] for r in manual_rows
               if r["cannot_establish_ground_truth_alone"] != "true"]
        assert not bad, f"Items where cannot_establish_ground_truth_alone != true: {bad}"

    def test_no_item_with_acquired_status(self, manual_rows):
        acquired = [r["needed_id"] for r in manual_rows if r["current_status"] == "ACQUIRED"]
        assert not acquired, f"Items marked as ACQUIRED: {acquired}"

    def test_all_forbidden_use_blocks_ground_truth(self, manual_rows):
        bad = []
        for r in manual_rows:
            forbidden = r["forbidden_use"].lower()
            if "ground truth" not in forbidden and "ground_truth" not in forbidden:
                bad.append(r["needed_id"])
        assert not bad, f"Items without ground truth in forbidden_use: {bad}"

    def test_all_forbidden_use_blocks_flood_label(self, manual_rows):
        bad = []
        for r in manual_rows:
            forbidden = r["forbidden_use"].lower()
            if "flood label" not in forbidden and "label" not in forbidden:
                bad.append(r["needed_id"])
        assert not bad, f"Items without flood label in forbidden_use: {bad}"

    def test_all_forbidden_use_blocks_training_label(self, manual_rows):
        bad = []
        for r in manual_rows:
            forbidden = r["forbidden_use"].lower()
            if "training label" not in forbidden and "training" not in forbidden:
                bad.append(r["needed_id"])
        assert not bad, f"Items without training label in forbidden_use: {bad}"


# ─── 7. Templates ────────────────────────────────────────────────────────────

class TestTemplates:
    INTAKE_REQUIRED_PLACEHOLDERS = [
        "[SOURCE_ID]", "[OBSERVED_EVENT_ID]", "[REGIAO]",
        "[FONTE]", "[URL]", "[LICENCA]",
        "[GATES_QUE_PODE_FECHAR]", "[GATES_QUE_NAO_FECHA]",
        "[CLAIM_PERMITIDO]", "[CLAIM_PROIBIDO]",
    ]

    REVIEW_REQUIRED_PLACEHOLDERS = [
        "[REVIEW_ID]", "[OBSERVED_EVENT_ID]", "[REGIAO]",
        "[REVISOR_FUNCAO]", "[FONTES_REVISADAS]",
        "[PODE_AVANCAR_PARA_PATCH_LINKING]",
        "[PODE_PROMOVER_GROUND_REFERENCE]", "[PODE_GERAR_LABEL]",
    ]

    def test_intake_has_required_placeholders(self):
        text = _read_text(TEMPLATE_INTAKE)
        for ph in self.INTAKE_REQUIRED_PLACEHOLDERS:
            assert ph in text, f"Intake template missing placeholder: {ph}"

    def test_review_has_required_placeholders(self):
        text = _read_text(TEMPLATE_REVIEW)
        for ph in self.REVIEW_REQUIRED_PLACEHOLDERS:
            assert ph in text, f"Review template missing placeholder: {ph}"

    def test_intake_mentions_local_only(self):
        text = _read_text(TEMPLATE_INTAKE).lower()
        assert "local-only" in text or "local_only" in text, \
            "Intake template does not mention local-only"

    def test_review_blocks_label_creation(self):
        text = _read_text(TEMPLATE_REVIEW).lower()
        assert "false" in text or "proibido" in text or "não cria" in text, \
            "Review template does not block label/ground truth"

    def test_review_has_decision_options(self):
        text = _read_text(TEMPLATE_REVIEW)
        assert "ACCEPT_AS_OBSERVED_EVENT_CANDIDATE" in text
        assert "BLOCK_FOR_OPERATIONAL_USE" in text


# ─── 8. Documentos metodológicos ────────────────────────────────────────────

class TestMethodologyDocuments:
    KEY_TERMS = [
        "evento observado candidato",
        "referência observacional candidata",
        "ground truth operacional",
        "Protocolo B",
        "multimodal",
        "DINO",
    ]

    def test_doc_refs_has_key_terms(self):
        text = _read_text(DOC_REFS).lower()
        for term in self.KEY_TERMS:
            assert term.lower() in text, f"DOC_REFS missing term: '{term}'"

    def test_doc_diag_has_key_terms(self):
        text = _read_text(DOC_DIAG).lower()
        for term in ["recife", "petrópolis", "curitiba", "local_only", "ground truth"]:
            assert term.lower() in text, f"DOC_DIAG missing term: '{term}'"

    def test_doc_refs_states_ground_truth_not_established(self):
        text = _read_text(DOC_REFS)
        assert "não está estabelecido" in text or "NOT_ESTABLISHED" in text or \
               "não possui ground truth" in text, \
            "DOC_REFS must state ground truth is not established"

    def test_doc_refs_states_protocol_b_blocked(self):
        text = _read_text(DOC_REFS)
        assert "BLOCKED" in text or "bloqueado" in text.lower(), \
            "DOC_REFS must state Protocolo B is BLOCKED"

    def test_doc_refs_states_multimodal_hold(self):
        text = _read_text(DOC_REFS)
        assert "HOLD" in text or "hold" in text.lower(), \
            "DOC_REFS must state multimodal is HOLD"

    def test_doc_refs_states_dino_support_only(self):
        text = _read_text(DOC_REFS)
        assert "support-only" in text.lower() or "SUPPORT_ONLY" in text or \
               "review-only" in text.lower(), \
            "DOC_REFS must state DINO is support/review only"


# ─── 9. Segurança — nenhum claim perigoso não contextualizado ───────────────

class TestNoDangerousClaims:
    FILES_TO_CHECK = [
        (DOC_REFS, "protocolo_c_referencias_observacionais_candidatas.md"),
        (DOC_DIAG, "protocolo_c_diagnostico_dados_externos_validos.md"),
        (TEMPLATE_INTAKE, "protocolo_c_intake_fonte_observacional_manual.md"),
        (TEMPLATE_REVIEW, "protocolo_c_revisao_evento_observado.md"),
    ]

    @pytest.mark.parametrize("filepath,label", FILES_TO_CHECK)
    def test_no_dangerous_claims_in_docs(self, filepath, label):
        text = _read_text(filepath)
        violations = _check_dangerous_claims(text, filepath)
        assert not violations, \
            f"Dangerous unconstrained claims in {label}:\n" + "\n".join(violations)

    def test_no_private_paths_in_candidates(self):
        text = _read_text(CANDIDATE_REG)
        bad_patterns = [r"C:\\Users\\", r"/home/", r"C:/Users/"]
        for pattern in bad_patterns:
            assert not re.search(pattern, text, re.IGNORECASE), \
                f"Private path pattern '{pattern}' found in candidate registry"

    def test_no_private_paths_in_gaps(self):
        text = _read_text(GAP_REG)
        bad_patterns = [r"C:\\Users\\", r"/home/", r"C:/Users/"]
        for pattern in bad_patterns:
            assert not re.search(pattern, text, re.IGNORECASE), \
                f"Private path pattern '{pattern}' found in gap registry"

    def test_no_heavy_files_in_schemas(self):
        schema_dir = SCHEMAS
        for fname in os.listdir(schema_dir):
            if fname.startswith("observed_event") or fname.startswith("manual_external"):
                fpath = os.path.join(schema_dir, fname)
                text = _read_text(fpath)
                for ext in [".tif", ".npz", ".npy", ".shp", ".geotiff"]:
                    assert ext not in text.lower(), \
                        f"Heavy file extension '{ext}' referenced in {fname}"

    def test_candidate_registry_no_operational_ground_truth_established(self):
        text = _read_text(CANDIDATE_REG)
        assert "ESTABLISHED" not in text.replace("NOT_ESTABLISHED", ""), \
            "Candidate registry contains 'ESTABLISHED' without 'NOT_' prefix"
