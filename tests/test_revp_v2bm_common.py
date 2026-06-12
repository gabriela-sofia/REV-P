import csv
import os

import pytest

import scripts.protocolo_c.revp_v2bm_common as common


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


CTB = [("CTB_2023_10_28_30", "FACT_v2at_0002", "SEED_v2bc_0001"),
       ("CTB_2022_01_15_16", "FACT_v2at_0007", "SEED_v2bc_0002"),
       ("CTB_2024_02_18_20", "FACT_v2at_0008", "SEED_v2bc_0003")]
PET = [("PET_2022_02_15", "FACT_v2at_0001"), ("PET_2022_03_20_21", "FACT_v2at_0004"),
       ("PET_2024_03_21_28", "FACT_v2at_0009")]


def seed(workspace, monkeypatch, ctb_strong=True, ctb_local=True, ctb_preview="READY_FOR_HUMAN_REVIEW",
         ctb_patch="HIGH", ctb_date="UNKNOWN", pet_ready=True):
    datasets = os.path.join(workspace, "datasets")
    os.makedirs(datasets, exist_ok=True)
    common.DATASET_DIR = datasets
    common.DOCS_DIR = os.path.join(workspace, "docs_v2bm")
    common.PUBLIC_DIR = os.path.join(workspace, "outputs_public")
    common.REFRESH = False

    _write(os.path.join(datasets, common.INPUTS["vl_registry"]),
           ["reference_status", "phenomenon_scope", "evidence_score", "uncertainty_level", "event_window"],
           [{"reference_status": "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE",
             "phenomenon_scope": "LANDSLIDE_SCARS_WITH_FLOOD_EVENT_CONTEXT", "evidence_score": "0.76",
             "uncertainty_level": "MODERATE", "event_window": "2022-05-24 a 2022-06-02"}])

    _write(os.path.join(datasets, common.INPUTS["bc_curitiba"]),
           ["seed_candidate_id", "event_patch_package_id", "candidate_id", "city", "region", "window_start",
            "window_end", "station_id", "station_name", "station_role", "missing_rate", "precip_signal_status",
            "temporal_evidence_strength"],
           [{"seed_candidate_id": f"SEEDC_{i}", "event_patch_package_id": epp, "candidate_id": cid, "city": "Curitiba",
             "region": "Curitiba", "window_start": "2023-10-21", "window_end": "2023-10-30",
             "station_id": "A807" if ctb_local else "A999", "station_name": "CURITIBA",
             "station_role": "LOCAL" if ctb_local else "REGIONAL_PROXY",
             "missing_rate": "0.000" if ctb_strong else "0.500",
             "precip_signal_status": "PRECIPITATION_PRESENT",
             "temporal_evidence_strength": "STRONG" if ctb_strong else "WEAK"}
            for i, (cid, epp, _sid) in enumerate(CTB, 1)])
    _write(os.path.join(datasets, common.INPUTS["bc_seed_registry"]),
           ["seed_id", "candidate_id", "region"],
           [{"seed_id": sid, "candidate_id": cid, "region": "Curitiba"} for cid, _e, sid in CTB])
    _write(os.path.join(datasets, common.INPUTS["bf_crosswalk"]),
           ["seed_id", "updated_crosswalk_status", "visual_review_status", "dino_link_status"],
           [{"seed_id": sid, "updated_crosswalk_status": "VISUAL_REVIEW_READY_WITH_WEAK_LINK",
             "visual_review_status": ctb_preview, "dino_link_status": "MODERATE"} for _c, _e, sid in CTB])
    _write(os.path.join(datasets, common.INPUTS["bf_lineage"]),
           ["seed_id", "date_confidence", "patch_link_confidence", "dino_link_confidence", "overall_lineage_confidence"],
           [{"seed_id": sid, "date_confidence": ctb_date, "patch_link_confidence": ctb_patch,
             "dino_link_confidence": "MODERATE", "overall_lineage_confidence": "LOW"} for _c, _e, sid in CTB])

    _write(os.path.join(datasets, common.INPUTS["bc_non_selected"]),
           ["queue_id", "event_patch_package_id", "candidate_id", "region", "city"],
           [{"queue_id": f"Q{i}", "event_patch_package_id": epp, "candidate_id": cid, "region": "Petropolis",
             "city": "Petropolis"} for i, (cid, epp) in enumerate(PET, 1)])
    _write(os.path.join(datasets, common.INPUTS["ay_precip"]),
           ["event_patch_package_id", "source_name", "station_id", "temporal_support_status", "precip_signal_status"],
           [{"event_patch_package_id": epp, "source_name": "INMET", "station_id": "A610",
             "temporal_support_status": "TEMPORAL_EVIDENCE_READY_FOR_REVIEW" if pet_ready else "TEMPORAL_EVIDENCE_NOT_READY",
             "precip_signal_status": "PRECIPITATION_PRESENT" if pet_ready else "UNKNOWN"} for _c, epp in PET])
    return datasets


@pytest.fixture
def workspace(tmp_path):
    saved = (common.DATASET_DIR, common.DOCS_DIR, common.PUBLIC_DIR, common.REFRESH)
    yield str(tmp_path)
    common.DATASET_DIR, common.DOCS_DIR, common.PUBLIC_DIR, common.REFRESH = saved


def _load(name):
    with open(common.dataset_path(name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _full(workspace, monkeypatch, **kw):
    seed(workspace, monkeypatch, **kw)
    for fn in (common.run_load_cross_region_state, common.run_apply_refined_protocol_policy,
               common.run_reassess_curitiba_candidates, common.run_reassess_petropolis_candidates,
               common.run_build_cross_region_candidate_registry, common.run_build_cross_region_gate_table,
               common.run_build_cross_region_evidence_scorecard, common.run_generate_reapplication_report,
               common.run_guardrail_regression):
        fn()


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


def test_forbidden_invariants_false():
    for field in ("can_create_operational_label", "can_create_negative", "can_train_model", "raw_data_versioned"):
        assert common.INVARIANTS[field] == "false"


def test_no_truth_invariants_true():
    for field in ("regional_proxy_is_not_local_station", "sentinel_preview_is_not_event_truth",
                  "dino_signal_is_not_truth", "patch_boundary_is_not_event_geometry",
                  "temporal_reference_is_not_label", "visual_review_reference_is_not_label",
                  "contextual_reference_is_not_label", "no_supervised_training_target_created"):
        assert common.INVARIANTS[field] == "true"


# --------------------------------------------------------------------------- #
# Task 1 - state
# --------------------------------------------------------------------------- #

def test_state_three_regions(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_load_cross_region_state()
    assert {r["region"] for r in rows} == {"Recife", "Curitiba", "Petropolis"}


def test_state_recife_already_validated(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["region"]: r for r in common.run_load_cross_region_state()}
    assert rows["Recife"]["current_status_before_v2bm"] == "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE"


def test_state_curitiba_limitations_mention_date(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["region"]: r for r in common.run_load_cross_region_state()}
    assert "acquisition_date" in rows["Curitiba"]["key_limitations"]


def test_state_petropolis_proxy(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["region"]: r for r in common.run_load_cross_region_state()}
    assert "regional proxy" in rows["Petropolis"]["key_limitations"].lower()


# --------------------------------------------------------------------------- #
# Task 2 - policy
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("item", ["PUBLIC_SOURCE_LICENSE", "MANUAL_HUMAN_REVIEW", "RASTER_CARTOGRAPHIC_EVIDENCE",
                                  "VECTOR_CRS", "SENTINEL_VISUAL", "DINO_SIGNAL", "REGIONAL_PROXY",
                                  "LOCAL_STRONG_SERIES", "INSTRUMENT_GAP"])
def test_policy_items_present(workspace, monkeypatch, item):
    seed(workspace, monkeypatch)
    items = {r["policy_item"] for r in common.run_apply_refined_protocol_policy()}
    assert item in items


def test_policy_license_not_blocker(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["policy_item"]: r for r in common.run_apply_refined_protocol_policy()}
    assert rows["PUBLIC_SOURCE_LICENSE"]["still_blocks_operational_label"] == "false"
    assert "not a blocker" in rows["PUBLIC_SOURCE_LICENSE"]["refined_interpretation"]


def test_policy_manual_review_not_required(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["policy_item"]: r for r in common.run_apply_refined_protocol_policy()}
    assert "Automated" in rows["MANUAL_HUMAN_REVIEW"]["refined_interpretation"]


def test_policy_raster_inherited_from_recife(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["policy_item"]: r for r in common.run_apply_refined_protocol_policy()}
    assert rows["RASTER_CARTOGRAPHIC_EVIDENCE"]["applies_to_region"] == "RECIFE"


def test_policy_dino_blocks_label(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["policy_item"]: r for r in common.run_apply_refined_protocol_policy()}
    assert rows["DINO_SIGNAL"]["still_blocks_operational_label"] == "true"


# --------------------------------------------------------------------------- #
# Task 3 - Curitiba
# --------------------------------------------------------------------------- #

def test_curitiba_three_seeds(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    assert len(common.run_reassess_curitiba_candidates()) == 3


def test_curitiba_temporal_reference(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_curitiba_candidates()
    assert all(r["updated_protocol_status"] == "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE" for r in rows)


def test_curitiba_local_station(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_curitiba_candidates()
    assert all(r["station_id"] == "A807" and r["station_role"] == "LOCAL" for r in rows)


def test_curitiba_no_full_spatial_without_date(workspace, monkeypatch):
    seed(workspace, monkeypatch, ctb_date="UNKNOWN")
    rows = common.run_reassess_curitiba_candidates()
    assert all(r["sentinel_date_status"] == "UNKNOWN" for r in rows)
    assert all(r["updated_protocol_status"] != "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE" for r in rows)


def test_curitiba_dino_not_label(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_curitiba_candidates()
    assert all(r["dino_link_status"] == "MODERATE" for r in rows)
    assert all(r["can_create_operational_label"] == "false" for r in rows)


def test_curitiba_preview_not_truth(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_curitiba_candidates()
    assert all("review" in r["remaining_limitation"].lower() for r in rows)


def test_curitiba_visual_review_when_temporal_weak(workspace, monkeypatch):
    seed(workspace, monkeypatch, ctb_strong=False, ctb_preview="READY_FOR_HUMAN_REVIEW", ctb_patch="HIGH")
    rows = common.run_reassess_curitiba_candidates()
    assert all(r["updated_protocol_status"] == "PROTOCOL_VALIDATED_VISUAL_REVIEW_REFERENCE" for r in rows)


def test_curitiba_score_capped(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_curitiba_candidates()
    assert all(float(r["evidence_score"]) <= 0.70 for r in rows)


def test_curitiba_no_training(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_curitiba_candidates()
    assert all(r["can_train_model"] == "false" for r in rows)


# --------------------------------------------------------------------------- #
# Task 4 - Petropolis
# --------------------------------------------------------------------------- #

def test_petropolis_three_packages(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    assert len(common.run_reassess_petropolis_candidates()) == 3


def test_petropolis_regional_temporal_context(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_petropolis_candidates()
    assert all(r["updated_protocol_status"] == "PROTOCOL_VALIDATED_REGIONAL_TEMPORAL_CONTEXT" for r in rows)


def test_petropolis_proxy_not_local(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_petropolis_candidates()
    assert all(r["station_role"] == "REGIONAL_PROXY" for r in rows)
    assert all("not become a local" in r["proxy_limitation"] for r in rows)


def test_petropolis_review_only_when_not_ready(workspace, monkeypatch):
    seed(workspace, monkeypatch, pet_ready=False)
    rows = common.run_reassess_petropolis_candidates()
    assert all(r["updated_protocol_status"] in {"REMAIN_REVIEW_ONLY_CONTEXT", "BLOCKED_FOR_CANDIDATE_REFERENCE"}
               for r in rows)


def test_petropolis_uncertainty_high(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_petropolis_candidates()
    assert all(r["uncertainty_level"] == "HIGH" for r in rows)


def test_petropolis_no_label(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reassess_petropolis_candidates()
    assert all(r["can_create_operational_label"] == "false" for r in rows)


# --------------------------------------------------------------------------- #
# Task 5 - registry
# --------------------------------------------------------------------------- #

def test_registry_three_regions(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_reassess_curitiba_candidates()
    common.run_reassess_petropolis_candidates()
    rows = {r["region"]: r for r in common.run_build_cross_region_candidate_registry()}
    assert set(rows) == {"Recife", "Curitiba", "Petropolis"}


def test_registry_recife_candidate_reference(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_reassess_curitiba_candidates()
    common.run_reassess_petropolis_candidates()
    rows = {r["region"]: r for r in common.run_build_cross_region_candidate_registry()}
    assert rows["Recife"]["reference_status"] == "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE"


def test_registry_curitiba_temporal(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_reassess_curitiba_candidates()
    common.run_reassess_petropolis_candidates()
    rows = {r["region"]: r for r in common.run_build_cross_region_candidate_registry()}
    assert rows["Curitiba"]["reference_status"] == "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE"


def test_registry_petropolis_contextual(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_reassess_curitiba_candidates()
    common.run_reassess_petropolis_candidates()
    rows = {r["region"]: r for r in common.run_build_cross_region_candidate_registry()}
    assert rows["Petropolis"]["reference_status"] == "PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE"


def test_registry_score_ordering(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_reassess_curitiba_candidates()
    common.run_reassess_petropolis_candidates()
    rows = {r["region"]: float(r["evidence_score"]) for r in common.run_build_cross_region_candidate_registry()}
    assert rows["Recife"] >= rows["Curitiba"] >= rows["Petropolis"]


@pytest.mark.parametrize("region", ["Recife", "Curitiba", "Petropolis"])
def test_registry_forbidden_use(workspace, monkeypatch, region):
    seed(workspace, monkeypatch)
    common.run_reassess_curitiba_candidates()
    common.run_reassess_petropolis_candidates()
    rows = {r["region"]: r for r in common.run_build_cross_region_candidate_registry()}
    for token in ("SUPERVISED_LABEL", "NEGATIVE_LABEL", "TRAINING_TARGET"):
        assert token in rows[region]["forbidden_use"]


# --------------------------------------------------------------------------- #
# Task 6 - gate table
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("region", ["Recife", "Curitiba", "Petropolis"])
def test_gate_table_c7_blocked(workspace, monkeypatch, region):
    _full(workspace, monkeypatch)
    rows = [r for r in _load(common.OUTPUTS["gate_table"])
            if r["region"] == region and r["gate_id"] == "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH"]
    assert rows and rows[0]["gate_status"] == "NOT_CREATED_BLOCKED_FOR_TRAINING"


@pytest.mark.parametrize("region", ["Recife", "Curitiba", "Petropolis"])
def test_gate_table_eight_gates(workspace, monkeypatch, region):
    _full(workspace, monkeypatch)
    rows = [r for r in _load(common.OUTPUTS["gate_table"]) if r["region"] == region]
    assert len(rows) == 8


def test_gate_table_no_label_training(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    rows = _load(common.OUTPUTS["gate_table"])
    assert all(r["operational_label_allowed"] == "false" and r["training_allowed"] == "false" for r in rows)


def test_gate_table_curitiba_c3_pending(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    rows = {(r["region"], r["gate_id"]): r for r in _load(common.OUTPUTS["gate_table"])}
    assert "PENDING" in rows[("Curitiba", "C3_SPATIAL_ANCHOR")]["gate_status"]


def test_gate_table_petropolis_c2_proxy(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    rows = {(r["region"], r["gate_id"]): r for r in _load(common.OUTPUTS["gate_table"])}
    assert "PROXY" in rows[("Petropolis", "C2_VALID_SERIES_OR_STATION")]["gate_status"]


# --------------------------------------------------------------------------- #
# Task 7 - scorecard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("region", ["Recife", "Curitiba", "Petropolis"])
@pytest.mark.parametrize("axis", ["PROVENANCE", "TEMPORALITY", "SPATIAL_CARTOGRAPHIC_EVIDENCE",
                                  "VISUAL_REVIEW_CONTEXT", "HYDROLOGICAL_CONTEXT", "DINO_REVIEW_SIGNAL",
                                  "GEOMETRY_VECTOR_READYNESS", "MODEL_LABEL_READYNESS"])
def test_scorecard_axis_present(workspace, monkeypatch, region, axis):
    seed(workspace, monkeypatch)
    rows = common.run_build_cross_region_evidence_scorecard()
    assert any(r["region"] == region and r["evidence_axis"] == axis for r in rows)


def test_scorecard_no_axis_supports_label(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_cross_region_evidence_scorecard()
    assert all(r["supports_operational_label"] == "false" for r in rows)


def test_scorecard_curitiba_temporal_high(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {(r["region"], r["evidence_axis"]): r for r in common.run_build_cross_region_evidence_scorecard()}
    assert rows[("Curitiba", "TEMPORALITY")]["score"] == "1.0"


def test_scorecard_recife_spatial_high(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {(r["region"], r["evidence_axis"]): r for r in common.run_build_cross_region_evidence_scorecard()}
    assert rows[("Recife", "SPATIAL_CARTOGRAPHIC_EVIDENCE")]["score"] == "1.0"


def test_scorecard_petropolis_dino_zero(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {(r["region"], r["evidence_axis"]): r for r in common.run_build_cross_region_evidence_scorecard()}
    assert rows[("Petropolis", "DINO_REVIEW_SIGNAL")]["score"] == "0.0"


# --------------------------------------------------------------------------- #
# Task 8 - report / packets / public
# --------------------------------------------------------------------------- #

def test_report_sections(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    body = open(common.doc_path("reports", "protocol_c_cross_region_reapplication_report.md"), encoding="utf-8").read()
    for section in ("## 1. Resumo", "## 6. Matriz comparativa", "## 10. Como isso fortalece o TCC"):
        assert section in body


@pytest.mark.parametrize("region", ["recife", "curitiba", "petropolis"])
def test_region_packets_written(workspace, monkeypatch, region):
    _full(workspace, monkeypatch)
    assert os.path.exists(common.doc_path("region_packets", f"{region}.md"))


def test_public_outputs_written(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    for rel in common.PUBLIC_FILES.values():
        assert os.path.exists(common.public_path(rel))


def test_public_outputs_only_csv_md(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    for rel in common.PUBLIC_FILES.values():
        assert os.path.splitext(rel)[1].lower() in {".csv", ".md"}


def test_readme_written(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    body = open(common.doc_path("README.md"), encoding="utf-8").read()
    assert "Curitiba" in body and "Petropolis" in body and "C7" in body


# --------------------------------------------------------------------------- #
# Guardrail regression
# --------------------------------------------------------------------------- #

def test_guardrail_pass(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    rows = common.run_guardrail_regression()
    assert rows and all(r["status"] == "PASS" for r in rows)


def test_guardrail_detects_label(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    reg = _load(common.OUTPUTS["registry"])
    reg[0]["can_create_operational_label"] = "true"
    common.write_csv(common.dataset_path(common.OUTPUTS["registry"]), reg)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


def test_guardrail_detects_c7_unblocked(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    gt = _load(common.OUTPUTS["gate_table"])
    for r in gt:
        if r["gate_id"] == "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH":
            r["gate_status"] = "CREATED"
    common.write_csv(common.dataset_path(common.OUTPUTS["gate_table"]), gt)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


def test_guardrail_detects_training_flag(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    sc = _load(common.OUTPUTS["scorecard"])
    sc[0]["can_train_model"] = "true"
    common.write_csv(common.dataset_path(common.OUTPUTS["scorecard"]), sc)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


@pytest.mark.parametrize("name_key", ["state", "policy", "curitiba", "petropolis", "registry", "gate_table", "scorecard"])
def test_no_forbidden_true(workspace, monkeypatch, name_key):
    _full(workspace, monkeypatch)
    for r in _load(common.OUTPUTS[name_key]):
        for field in ("can_create_operational_label", "can_create_negative", "can_train_model"):
            assert common.clean(r.get(field, "false")).lower() != "true"


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def test_orchestrator_end_to_end(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    manifest = common.run_orchestrator()
    assert manifest[0]["step_name"] == "refresh_v2bl_chain"
    assert manifest[0]["status"] == "SKIPPED"
    for name in common.OUTPUTS.values():
        assert os.path.exists(common.dataset_path(name))


def test_orchestrator_manifest_length(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    manifest = common.run_orchestrator()
    assert len(manifest) == 10
    assert all(m["status"] in {"OK", "SKIPPED"} for m in manifest)


def test_orchestrator_docs_and_public(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    assert os.path.exists(common.doc_path("evidence_cache", ".gitignore"))
    for rel in common.PUBLIC_FILES.values():
        assert os.path.exists(common.public_path(rel))


def test_orchestrator_expands_three_regions(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    regions = {r["region"] for r in _load(common.OUTPUTS["registry"])}
    assert regions == {"Recife", "Curitiba", "Petropolis"}


def test_zero_label_negative_training(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    for name in (common.OUTPUTS["registry"], common.OUTPUTS["curitiba"], common.OUTPUTS["petropolis"]):
        for r in _load(name):
            assert r["can_create_operational_label"] == "false"
            assert r["can_create_negative"] == "false"
            assert r["can_train_model"] == "false"


def test_evidence_cache_gitignore(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    assert "*" in open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read()
