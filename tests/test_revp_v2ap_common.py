import csv

import pytest

import scripts.protocolo_c.revp_v2ap_common as common


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    protocol = datasets / "protocolo_c"
    docs = tmp_path / "docs" / "protocolo_c" / "v2ap_patch_geometry_sentinel_crosswalk"
    cfg = tmp_path / "configs" / "protocolo_c"
    for path in (datasets, protocol, docs, cfg):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(common, "DATASET_ROOT", str(datasets))
    monkeypatch.setattr(common, "DATASET_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    return datasets, protocol, docs


# Candidate specs: 1 C4 (event geometry), 6 C3, 1 C2, 1 C1 (no spatial).
_SPECS = [
    ("PET_2022_02_15", "Petropolis", "C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW",
     "STRONG_GROUND_REFERENCE_CANDIDATE", "HIGH", "true", 88, "2022-02-15", "2022-02-16"),
    ("REC_2022_05_24_30", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     "STRONG_GROUND_REFERENCE_CANDIDATE", "HIGH", "false", 90, "2022-05-24", "2022-05-30"),
    ("REC_2023_02_05_06", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     "STRONG_GROUND_REFERENCE_CANDIDATE", "HIGH", "false", 80, "2023-02-05", "2023-02-06"),
    ("REC_2024_06_14_16", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     "STRONG_GROUND_REFERENCE_CANDIDATE", "HIGH", "false", 90, "2024-06-14", "2024-06-16"),
    ("CTB_2022_01_15_16", "Curitiba", "C3_STRONG_REFERENCE_CANDIDATE",
     "STRONG_GROUND_REFERENCE_CANDIDATE", "HIGH", "false", 85, "2022-01-15", "2022-01-16"),
    ("CTB_2023_10_28_30", "Curitiba", "C3_STRONG_REFERENCE_CANDIDATE",
     "STRONG_GROUND_REFERENCE_CANDIDATE", "HIGH", "false", 90, "2023-10-28", "2023-10-30"),
    ("CTB_2024_02_18_20", "Curitiba", "C3_STRONG_REFERENCE_CANDIDATE",
     "STRONG_GROUND_REFERENCE_CANDIDATE", "HIGH", "false", 85, "2024-02-18", "2024-02-20"),
    ("PET_2022_03_20_21", "Petropolis", "C2_DOCUMENTED_OBSERVED_EVENT",
     "MODERATE_GROUND_REFERENCE_CANDIDATE", "MEDIUM", "false", 75, "2022-03-20", "2022-03-21"),
    ("PET_2024_03_21_28", "Petropolis", "C1_CONTEXTUAL_OBSERVED_EVENT",
     "NEEDS_MORE_SPATIAL_EVIDENCE", "MEDIUM", "false", 60, "2024-03-21", "2024-03-28"),
]


def install_v2ao_stack(protocol, datasets, with_sentinel=True):
    form, matrix, levels, ranking, spatial, observed = [], [], [], [], [], []
    for cid, region, level, decision, conf, has_map, score, ds, de in _SPECS:
        form.append({"candidate_id": cid, "geometry_or_map_available": has_map,
                     "spatial_anchor_confirmed": "true" if level != "C1_CONTEXTUAL_OBSERVED_EVENT" else "false",
                     "sentinel_crosswalk_confirmed": "false"})
        matrix.append({"candidate_id": cid, "region": region, "event_name": f"Evento {cid}",
                       "human_review_decision": decision, "confidence_band": conf,
                       "spatial_strength": "HIGH" if has_map == "true" else "MEDIUM",
                       "source_strength": "STRONG", "remaining_blockers": "NEEDS_SENTINEL_CROSSWALK"})
        levels.append({"candidate_id": cid, "reference_level": level})
        ranking.append({"candidate_id": cid, "readiness_score": str(score),
                        "recommended_next_step": "x"})
        anchors = 3 if level != "C1_CONTEXTUAL_OBSERVED_EVENT" else 1
        for _ in range(anchors):
            spatial.append({"candidate_id": cid, "anchor_type": "bairro",
                            "geometry_available": "false", "coordinate_available": "false"})
        observed.append({"observed_event_id": cid, "region": region,
                         "date_start": ds, "date_end": de})
    write_csv(protocol / "v2ao_human_review_form_filled.csv",
              ["candidate_id", "geometry_or_map_available", "spatial_anchor_confirmed", "sentinel_crosswalk_confirmed"], form)
    write_csv(protocol / "v2ao_reference_candidate_matrix.csv",
              ["candidate_id", "region", "event_name", "human_review_decision", "confidence_band",
               "spatial_strength", "source_strength", "remaining_blockers"], matrix)
    write_csv(protocol / "v2ao_reference_level_classification.csv",
              ["candidate_id", "reference_level"], levels)
    write_csv(protocol / "v2ao_patch_truth_boundary_audit.csv",
              ["candidate_id", "patch_truth_allowed"], [{"candidate_id": c[0], "patch_truth_allowed": "false"} for c in _SPECS])
    write_csv(protocol / "v2ao_human_review_trace.csv",
              ["candidate_id", "decision_field"], [{"candidate_id": c[0], "decision_field": "x"} for c in _SPECS])
    write_csv(protocol / "v2ao_ground_truth_promotion_blocker_audit.csv",
              ["candidate_id", "ground_truth_blocked"], [{"candidate_id": c[0], "ground_truth_blocked": "true"} for c in _SPECS])
    write_csv(protocol / "v2ao_final_candidate_ranking.csv",
              ["candidate_id", "readiness_score", "recommended_next_step"], ranking)
    write_csv(protocol / "v2an_spatial_anchor_registry.csv",
              ["candidate_id", "anchor_type", "geometry_available", "coordinate_available"], spatial)
    write_csv(datasets / "observed_event_reference_candidate_registry.csv",
              ["observed_event_id", "region", "date_start", "date_end"], observed)
    if with_sentinel:
        # synthetic Sentinel asset registry with explicit manifest date matching a Recife event
        write_csv(datasets / "official_anchor_sentinel_patch_registry_fixture.csv",
                  ["reference_patch_id", "region", "scene_date", "anchor_latitude", "anchor_longitude", "crs"],
                  [{"reference_patch_id": "REC_00001", "region": "REC", "scene_date": "2022-05-26",
                    "anchor_latitude": "-8.05", "anchor_longitude": "-34.9", "crs": "EPSG:4326"}])
        # a DINO crosswalk file that must be marked unusable
        write_csv(datasets / "dino_protocol_c_crosswalk_fixture.csv",
                  ["patch_id", "region", "scene_date"],
                  [{"patch_id": "REC_00002", "region": "REC", "scene_date": "2022-05-25"}])


def install_all(tmp_path, monkeypatch, with_sentinel=True):
    datasets, protocol, docs = set_env(tmp_path, monkeypatch)
    install_v2ao_stack(protocol, datasets, with_sentinel)
    return datasets, protocol, docs


# --- common tests ----------------------------------------------------------
def test_loads_v2ao_stack(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    stack = common.load_v2ao_candidate_stack()
    assert len(stack["levels"]) == 9


def test_fails_when_v2ao_missing(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.load_v2ao_candidate_stack()


def test_extract_date_and_window():
    assert common.extract_date_from_text_safe("scene_2022-05-26_s2.tif") == "2022-05-26"
    assert common.extract_date_from_text_safe("no date here") == ""
    assert common.date_window_overlap("2022-05-26", "2022-05-26", "2022-05-16", "2022-06-07")
    assert not common.date_window_overlap("2021-01-01", "2021-01-01", "2022-05-16", "2022-06-07")


def test_normalize_helpers():
    assert common.normalize_region("", "REC_x") == "Recife"
    assert common.normalize_patch_id(" pet_00016 ") == "PET_00016"
    assert common.normalize_date("20220526") == "2022-05-26"


def test_assert_no_operational_promotion():
    common.assert_no_operational_promotion([{"forbidden_use": "ground_truth|label"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"patch_truth_allowed": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"note": "sentinel_date_inferred=true"}])


def test_assert_no_fake_sentinel_crosswalk():
    common.assert_no_fake_sentinel_crosswalk([{
        "can_be_used_as_explicit_crosswalk": "true",
        "crosswalk_evidence_type": "manifest_field_explicit"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_sentinel_crosswalk([{
            "can_be_used_as_explicit_crosswalk": "true",
            "crosswalk_evidence_type": "dino_similarity"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_sentinel_crosswalk([{"crosswalk_inferred": "true"}])


def test_assert_no_fake_geometry():
    common.assert_no_fake_geometry([{"has_event_geometry": "false"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_geometry([{"geometry_invented": "true"}])


def test_assert_output_is_v2ap():
    common.assert_output_is_v2ap("datasets/protocolo_c/v2ap_x.csv")
    with pytest.raises(ValueError):
        common.assert_output_is_v2ap("datasets/protocolo_c/v2ao_x.csv")


def test_scan_sentinel_assets_marks_dino_unusable(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    assets = common.scan_repo_for_sentinel_assets()
    dino = [a for a in assets if "dino" in a["path"].lower()]
    assert dino
    assert all(a["safe_to_use_as_crosswalk_evidence"] == "false" for a in dino)
    nondino = [a for a in assets if a["safe_to_use_as_crosswalk_evidence"] == "true"]
    assert nondino  # the official fixture with explicit date is usable
