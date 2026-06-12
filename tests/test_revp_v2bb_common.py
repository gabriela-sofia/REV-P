import os

import pytest

import scripts.protocolo_c.revp_v2bb_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("source_type,name,kwargs,expected", [
    ("TECHNICAL_REPORT", "report", {"confirmed": True}, "OFFICIAL_EVENT_REPORT"),
    ("OFFICIAL_BULLETIN", "bulletin", {"confirmed": True, "has_map": True}, "OFFICIAL_PDF_MAP"),
    ("OFFICIAL_MUNICIPAL", "prefeitura", {"confirmed": True}, "OFFICIAL_TEXTUAL_LOCATION"),
    ("JOURNALISTIC", "news", {"confirmed": True}, "JOURNALISTIC_DATED_LOCATION"),
    ("INSTITUTIONAL_POST", "post", {"confirmed": True}, "INSTITUTIONAL_POST"),
    ("OTHER", "map", {"confirmed": True, "has_map": True}, "OFFICIAL_MAP_IMAGE"),
    ("OTHER", "quickview", {"confirmed": True, "quickview": True}, "QUICKVIEW_ONLY"),
    ("OFFICIAL_GEOLOGICAL", "SGB CPRM", {"confirmed": True}, "SUSCEPTIBILITY_CONTEXT_ONLY"),
    ("ACADEMIC_PAPER", "paper", {"confirmed": True}, "REVIEW_ONLY_VISUAL_SUPPORT"),
    ("OTHER", "conflict", {"confirmed": True, "conflict": True}, "CONFLICTING_SECONDARY_EVIDENCE"),
    ("OTHER", "missing", {"confirmed": False}, "NO_SECONDARY_EVIDENCE_FOUND"),
    ("OTHER", "context", {"confirmed": True}, "CONTEXT_ONLY"),
])
def test_secondary_class(source_type, name, kwargs, expected):
    assert common.secondary_class(source_type, name, **kwargs) == expected


@pytest.mark.parametrize("text,city,patch,expected", [
    ("patch-123 affected", "Curitiba", "patch-123", "PATCH_LEVEL"),
    ("Bairro Cajuru", "Curitiba", "", "NEIGHBORHOOD_LEVEL"),
    ("Rua das Flores", "Curitiba", "", "NEIGHBORHOOD_LEVEL"),
    ("Curitiba", "Curitiba", "", "MUNICIPALITY_LEVEL"),
    ("Parana", "Curitiba", "", "REGIONAL"),
    ("", "Curitiba", "", "UNKNOWN"),
])
def test_location_relation(text, city, patch, expected):
    assert common.location_relation(text, city, patch) == expected


@pytest.mark.parametrize("event,evidence,delta,relation", [
    ("2024-01-10", "2024-01-10", "0", "DURING"),
    ("2024-01-10", "2024-01-09", "-1", "BEFORE"),
    ("2024-01-10", "2024-01-11", "1", "AFTER"),
    ("bad", "bad", "", "UNKNOWN"),
])
def test_temporal_relation(event, evidence, delta, relation):
    assert common.temporal_relation(event, evidence) == (delta, relation)


@pytest.mark.parametrize("evidence,boundary,expected", [
    ("OFFICIAL_MAP_IMAGE", False, "MAP_ONLY"),
    ("OFFICIAL_PDF_MAP", False, "MAP_ONLY"),
    ("QUICKVIEW_ONLY", False, "MAP_ONLY"),
    ("OFFICIAL_EVENT_REPORT", False, "TEXT_ONLY"),
    ("OFFICIAL_TEXTUAL_LOCATION", False, "TEXT_ONLY"),
    ("CONTEXT_ONLY", False, "NONE"),
    ("OFFICIAL_EVENT_REPORT", True, "EXPLICIT_BOUNDARY"),
])
def test_geometry_relation(evidence, boundary, expected):
    assert common.geometry_relation(evidence, boundary) == expected


@pytest.mark.parametrize("evidence,location,date_ok,conflict,expected", [
    ("OFFICIAL_EVENT_REPORT", "NEIGHBORHOOD_LEVEL", True, False, True),
    ("OFFICIAL_MAP_IMAGE", "PATCH_LEVEL", True, False, True),
    ("JOURNALISTIC_DATED_LOCATION", "NEIGHBORHOOD_LEVEL", True, False, True),
    ("OFFICIAL_EVENT_REPORT", "MUNICIPALITY_LEVEL", True, False, False),
    ("QUICKVIEW_ONLY", "PATCH_LEVEL", True, False, False),
    ("SUSCEPTIBILITY_CONTEXT_ONLY", "PATCH_LEVEL", True, False, False),
    ("OFFICIAL_EVENT_REPORT", "PATCH_LEVEL", False, False, False),
    ("OFFICIAL_EVENT_REPORT", "PATCH_LEVEL", True, True, False),
])
def test_can_reduce_uncertainty(evidence, location, date_ok, conflict, expected):
    assert common.can_reduce_uncertainty(evidence, location, date_ok, conflict) is expected


def row(evidence_class, location="MUNICIPALITY_LEVEL", reduce=False):
    return {"evidence_class": evidence_class, "location_relation": location, "can_reduce_uncertainty": str(reduce).lower()}


@pytest.mark.parametrize("rows,status,basis,allowed", [
    ([row("OFFICIAL_MAP_IMAGE", "NEIGHBORHOOD_LEVEL", True)], "READY_FOR_HUMAN_DIGITIZATION_REVIEW", "OFFICIAL_MAP_IMAGE", True),
    ([row("OFFICIAL_PDF_MAP", "PATCH_LEVEL", True)], "READY_FOR_HUMAN_DIGITIZATION_REVIEW", "OFFICIAL_PDF_MAP", True),
    ([row("OFFICIAL_EVENT_REPORT", "NEIGHBORHOOD_LEVEL", True)], "READY_FOR_HUMAN_DIGITIZATION_REVIEW", "OFFICIAL_EVENT_REPORT_WITH_LOCALITY", True),
    ([row("OFFICIAL_TEXTUAL_LOCATION")], "NEEDS_MORE_SECONDARY_EVIDENCE", "TEXTUAL_ONLY", False),
    ([row("QUICKVIEW_ONLY")], "NEEDS_MORE_SECONDARY_EVIDENCE", "TEXTUAL_ONLY", False),
    ([row("CONTEXT_ONLY")], "ONLY_CONTEXT_AVAILABLE", "NONE", False),
    ([row("NO_SECONDARY_EVIDENCE_FOUND")], "ONLY_CONTEXT_AVAILABLE", "NONE", False),
    ([row("CONFLICTING_SECONDARY_EVIDENCE")], "HAZARD_AMBIGUITY_BLOCKS_DIGITIZATION", "NONE", False),
])
def test_digitization_decision(rows, status, basis, allowed):
    assert common.digitization_decision(rows) == (status, basis, allowed)


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2bb_orchestrator_manifest.csv"])
def test_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def test_selected_packets_six():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert len(rows) == 6
    assert all(row["selection_status"] == "SELECTED_FOR_SECONDARY_EVIDENCE_EXPANSION" for row in rows)


def test_recife_excluded():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert not any(row["region"] == "Recife" for row in rows)


def test_expanded_targets_78():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert len(rows) == 78
    assert sum(row["target_origin"] == "REUSED_V2BA_AUDITED_SOURCE" for row in rows) == 12
    assert sum(row["target_origin"] == "EXPANDED_SECONDARY_TARGET" for row in rows) == 66


def test_each_packet_has_13_targets():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert {sum(row["review_packet_id"] == packet for row in rows) for packet in {row["review_packet_id"] for row in rows}} == {13}


def test_probe_offline_and_no_raw():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert len(rows) == 78
    assert all(row["probe_mode"] == "OFFLINE_DETERMINISTIC" for row in rows)
    assert all(row["raw_payload_cached"] == "false" and row["raw_data_versioned"] == "false" for row in rows)


def test_classification_78():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert len(rows) == 78
    assert sum(row["evidence_class"] == "NO_SECONDARY_EVIDENCE_FOUND" for row in rows) == 66


def test_actual_classes_do_not_reduce():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert all(row["can_reduce_uncertainty"] == "false" for row in rows)


def test_correlation_78():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert len(rows) == 78
    assert all(row["geometry_relation"] in {"TEXT_ONLY", "MAP_ONLY", "NONE"} for row in rows)


def test_uncertainty_remains_very_high():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    assert len(rows) == 6
    assert all(row["previous_overall_uncertainty"] == "VERY_HIGH" for row in rows)
    assert all(row["updated_overall_uncertainty"] == "VERY_HIGH" and row["uncertainty_reduced"] == "false" for row in rows)


def test_digitization_matrix_all_false():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert len(rows) == 6
    assert all(row["can_digitize_candidate"] == "false" for row in rows)
    assert all(row["digitization_review_status"] in {"NEEDS_MORE_SECONDARY_EVIDENCE", "ONLY_CONTEXT_AVAILABLE"} for row in rows)


@pytest.mark.parametrize("field", [
    "adjudication_id", "review_packet_id", "region", "city", "patch_id", "event_date",
    "temporal_support_summary", "secondary_evidence_summary", "geometry_evidence_summary",
    "updated_uncertainty", "digitization_review_status", "recommended_human_decision_options",
    "current_truth_status", "current_label_status",
])
def test_adjudication_fields(field):
    assert field in common.load_csv(common.dataset_path(common.OUTPUTS[7]))[0]


def test_adjudication_truth_and_label_status():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert all(row["current_truth_status"] == "NOT_GROUND_TRUTH" for row in rows)
    assert all(row["current_label_status"] == "NO_LABEL" for row in rows)


@pytest.mark.parametrize("option", [
    "DIGITIZE_CANDIDATE_FOR_NEXT_REVIEW", "KEEP_GEOMETRY_MISSING", "REQUEST_MORE_EVIDENCE",
    "MARK_HAZARD_AMBIGUOUS", "EXCLUDE_FROM_GEOMETRY_REVIEW",
])
def test_decision_options(option):
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert all(option in row["recommended_human_decision_options"] for row in rows)


def test_packet_index_six():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert len(rows) == 6
    assert all(row["current_truth_status"] == "NOT_GROUND_TRUTH" for row in rows)
    assert all(row["next_action_rank_1"] == "SEARCH_MORE_SPECIFIC_SECONDARY_SPATIAL_EVIDENCE" for row in rows)
    assert all(row["recife_next_action"] == "RESOLVE_RECIFE_TEMPORAL_GAP_WITH_CEMADEN_OR_SECONDARY_STATIONS" for row in rows)


def test_markdowns_six_and_guardrails():
    paths = [name for name in os.listdir(common.doc_path("secondary_evidence_packets")) if name.endswith(".md")]
    assert len(paths) == 6
    assert all("Nao e ground truth; nao cria label; nao cria negativo; nao treina modelo." in open(common.doc_path("secondary_evidence_packets", name), encoding="utf-8").read() for name in paths)


def test_probe_summaries_six():
    assert len([name for name in os.listdir(common.doc_path("source_probe_summaries")) if name.endswith(".md")]) == 6


def test_adjudication_table_doc():
    assert os.path.exists(common.doc_path("adjudication_tables", common.OUTPUTS[7]))


@pytest.mark.parametrize("field", ["can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model"])
def test_zero_promotions(field):
    for output in common.OUTPUTS[:9]:
        assert all(row.get(field) == "false" for row in common.load_csv(common.dataset_path(output)))


def test_cache_marker():
    assert open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[9]))
    assert len(rows) == 10
    assert all(row["status"] == "PASS" and row["violation_count"] == "0" for row in rows)


def test_orchestrator_ok():
    rows = common.load_csv(common.dataset_path("v2bb_orchestrator_manifest.csv"))
    assert len(rows) == 10
    assert all(row["status"] == "OK" for row in rows)


def test_readme_next_action():
    text = open(common.doc_path("README.md"), encoding="utf-8").read()
    assert "SEARCH_MORE_SPECIFIC_SECONDARY_SPATIAL_EVIDENCE" in text
    assert "Ground truth, labels, negativos e treino: 0" in text
