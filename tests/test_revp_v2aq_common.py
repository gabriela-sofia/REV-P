import csv
import json

import pytest

import scripts.protocolo_c.revp_v2aq_common as common


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
    docs = tmp_path / "docs" / "protocolo_c" / "v2aq_event_geometry_patch_link"
    geojson = docs / "geojson_candidates"
    cfg = tmp_path / "configs" / "protocolo_c"
    for path in (datasets, protocol, docs, geojson, cfg):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_ROOT", str(datasets))
    monkeypatch.setattr(common, "DATASET_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "GEOJSON_DIR", str(geojson))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    return datasets, protocol, docs, geojson


# candidate, region, level, anchors, has_event_geometry, has_explicit_crosswalk
_SPECS = [
    ("PET_2022_02_15", "Petropolis", "C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW",
     ["municipio", "mapa_ou_laudo"], "true", "false", 88),
    ("REC_2023_02_05_06", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     ["municipio", "rua", "ponto_de_alagamento"], "false", "true", 80),
    ("CTB_2023_10_28_30", "Curitiba", "C3_STRONG_REFERENCE_CANDIDATE",
     ["municipio", "rua", "ponto_de_alagamento"], "false", "false", 90),
    ("REC_2022_05_24_30", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     ["municipio", "bairro", "corredor_de_rio"], "false", "false", 90),
    ("REC_2024_06_14_16", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     ["municipio", "bairro"], "false", "false", 90),
    ("CTB_2022_01_15_16", "Curitiba", "C3_STRONG_REFERENCE_CANDIDATE",
     ["municipio", "bairro"], "false", "false", 85),
    ("CTB_2024_02_18_20", "Curitiba", "C3_STRONG_REFERENCE_CANDIDATE",
     ["municipio", "bairro"], "false", "false", 85),
    ("PET_2022_03_20_21", "Petropolis", "C2_DOCUMENTED_OBSERVED_EVENT",
     ["municipio", "mapa_ou_laudo"], "true", "false", 75),
    ("PET_2024_03_21_28", "Petropolis", "C1_CONTEXTUAL_OBSERVED_EVENT",
     ["municipio"], "false", "false", 60),
]


def install_stack(protocol, datasets):
    selection, geometry, link, scores, crosswalk = [], [], [], [], []
    levels, ranking, trace, spatial, observed = [], [], [], [], []
    patch_reg = [{"registry_item_id": "PR_0", "source_file": "datasets/protocolo_c/patch_fixture.csv",
                  "patch_id": "patch_id", "region": "REC", "has_geometry": "true", "has_bbox": "true",
                  "has_centroid": "false", "has_crs": "true", "geometry_source": "registry_column_present",
                  "geometry_status": "HAS_GEOMETRY_COLUMNS", "safe_for_patch_link": "true", "notes": "x"}]
    for cid, region, level, anchors, has_geom, has_xc, score in _SPECS:
        selection.append({"candidate_id": cid, "region": region, "reference_level": level,
                          "human_review_decision": "STRONG_GROUND_REFERENCE_CANDIDATE",
                          "confidence_band": "HIGH", "readiness_score": str(score),
                          "included_in_v2ap": "true", "selection_reason": "x",
                          "dominant_blocker": "x", "forbidden_use": "ground_truth|label"})
        gstat = "EVENT_AND_PATCH_GEOMETRY_READY" if has_geom == "true" else "PATCH_GEOMETRY_READY"
        geometry.append({"geometry_readiness_id": "g", "candidate_id": cid, "region": region,
                         "spatial_anchor_count": str(len(anchors)), "strongest_anchor_type": "municipio",
                         "has_event_geometry": has_geom, "has_event_coordinates": "false",
                         "has_patch_geometry": "true", "has_patch_bbox": "true",
                         "manual_geometry_collection_needed": "true",
                         "geometry_readiness_status": gstat, "blocking_reason": "x"})
        link.append({"link_readiness_id": "l", "candidate_id": cid, "region": region,
                     "has_event_anchor": "true", "has_event_geometry": has_geom,
                     "has_patch_geometry": "true", "has_sentinel_crosswalk_candidate": has_xc,
                     "has_explicit_sentinel_crosswalk": has_xc,
                     "patch_event_link_status": "x", "patch_level_reference_candidate": "false",
                     "patch_truth_allowed": "false", "required_next_action": "x"})
        scores.append({"score_id": "s", "candidate_id": cid, "region": region,
                       "event_reference_level": level, "geometry_score": "0", "crosswalk_score": "0",
                       "patch_registry_score": "20", "overall_patch_reference_score": "20",
                       "readiness_band": "MEDIUM", "dominant_blocker": "x",
                       "patch_reference_status": "EVENT_REFERENCE_ONLY",
                       "can_create_ground_truth": "false", "can_create_label": "false",
                       "protocol_b_status": "BLOCKED"})
        within = "true" if has_xc == "true" else "false"
        crosswalk.append({"crosswalk_candidate_id": "c", "candidate_id": cid, "region": region,
                          "asset_id": "A", "patch_id_detected": (region[:3].upper() + "_00001") if has_xc == "true" else "",
                          "asset_date": "2022-05-25" if has_xc == "true" else "",
                          "event_date_start": "2022-05-24", "event_date_end": "2022-05-30",
                          "within_temporal_window": within, "region_match": "true",
                          "patch_id_match": "false", "crosswalk_evidence_type": "manifest_field_explicit" if has_xc == "true" else "none",
                          "crosswalk_candidate_status": "x", "can_be_used_as_explicit_crosswalk": has_xc,
                          "blocking_reason": "x"})
        levels.append({"candidate_id": cid, "reference_level": level})
        ranking.append({"candidate_id": cid, "readiness_score": str(score), "recommended_next_step": "x"})
        trace.append({"candidate_id": cid, "decision_field": "x"})
        for at in anchors:
            spatial.append({"candidate_id": cid, "region": region, "anchor_text": f"{at} text",
                            "anchor_type": at, "geometry_available": "false", "coordinate_available": "false"})
        observed.append({"observed_event_id": cid, "region": region,
                         "date_start": "2022-05-24", "date_end": "2022-05-30"})
    geom_packet = [{"collection_id": "GC", "candidate_id": c[0]} for c in _SPECS]
    cross_packet = [{"collection_id": "XC", "candidate_id": c[0]} for c in _SPECS]
    boundary = [{"candidate_id": c[0], "patch_truth_allowed": "false"} for c in _SPECS]
    write_csv(protocol / "v2ap_candidate_selection.csv", list(selection[0].keys()), selection)
    write_csv(protocol / "v2ap_spatial_geometry_readiness.csv", list(geometry[0].keys()), geometry)
    write_csv(protocol / "v2ap_patch_registry_inventory.csv", list(patch_reg[0].keys()), patch_reg)
    write_csv(protocol / "v2ap_sentinel_crosswalk_candidates.csv", list(crosswalk[0].keys()), crosswalk)
    write_csv(protocol / "v2ap_patch_event_link_readiness.csv", list(link[0].keys()), link)
    write_csv(protocol / "v2ap_geometry_collection_packet.csv", list(geom_packet[0].keys()), geom_packet)
    write_csv(protocol / "v2ap_crosswalk_collection_packet.csv", list(cross_packet[0].keys()), cross_packet)
    write_csv(protocol / "v2ap_patch_reference_readiness_scores.csv", list(scores[0].keys()), scores)
    write_csv(protocol / "v2ap_patch_truth_boundary_update.csv", list(boundary[0].keys()), boundary)
    write_csv(protocol / "v2ao_reference_level_classification.csv", list(levels[0].keys()), levels)
    write_csv(protocol / "v2ao_final_candidate_ranking.csv", list(ranking[0].keys()), ranking)
    write_csv(protocol / "v2ao_human_review_trace.csv", list(trace[0].keys()), trace)
    write_csv(protocol / "v2an_spatial_anchor_registry.csv", list(spatial[0].keys()), spatial)
    write_csv(datasets / "observed_event_reference_candidate_registry.csv", list(observed[0].keys()), observed)


def install_all(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = set_env(tmp_path, monkeypatch)
    install_stack(protocol, datasets)
    return datasets, protocol, docs, geojson


# --- common tests ----------------------------------------------------------
def test_loads_v2ap_stack(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    stack = common.load_v2ap_stack()
    assert len(stack["geometry"]) == 9


def test_fails_when_v2ap_missing(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.load_v2ap_stack()


def test_derive_spread(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    derived = common.derive_candidates()
    assert len(derived) == 9
    statuses = {common._event_geometry_status(d) for d in derived}
    assert "OFFICIAL_MAP_DIGITIZATION_REQUIRED" in statuses
    assert "TEXTUAL_ANCHOR_ONLY" in statuses
    assert "INSUFFICIENT_GEOMETRY" in statuses


def test_assert_no_operational_promotion():
    common.assert_no_operational_promotion([{"forbidden_use": "ground_truth|label"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"patch_truth_allowed": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"note": "geometry_inferred=true"}])


def test_assert_no_fake_geometry():
    common.assert_no_fake_geometry([{"geometry_status": "TEXTUAL_ANCHOR_ONLY"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_geometry([{"geometry_invented": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_geometry([{
            "geometry_status": "EXPLICIT_EVENT_GEOMETRY_AVAILABLE",
            "geometry_source_type": "textual_anchor"}])


def test_assert_no_fake_overlay():
    common.assert_no_fake_overlay([{"overlay_executed": "false"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_overlay([{"overlay_executed": "true"}])


def test_assert_no_label_creation():
    common.assert_no_label_creation([{"can_create_label": "false"}])
    with pytest.raises(ValueError):
        common.assert_no_label_creation([{"can_create_label": "true"}])


def test_assert_output_is_v2aq():
    common.assert_output_is_v2aq("datasets/protocolo_c/v2aq_x.csv")
    with pytest.raises(ValueError):
        common.assert_output_is_v2aq("datasets/protocolo_c/v2ap_x.csv")


def test_geojson_real_geometry_from_explicit(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    # inject an explicit-geometry event candidate and export
    cols = common.EVENT_GEOMETRY_COLUMNS
    row = {c: "" for c in cols}
    row.update({"event_geometry_id": "EG", "candidate_id": "X", "region": "Recife",
                "geometry_status": "EXPLICIT_POINT_OR_COORDINATE_AVAILABLE",
                "geometry_source_type": "explicit_coordinate", "manual_digitization_required": "false",
                "can_use_for_patch_link_review": "true", "can_use_for_ground_truth": "false",
                "explicit_geometry_geojson": json.dumps({"type": "Point", "coordinates": [-34.9, -8.05]})})
    write_csv(protocol / "v2aq_event_geometry_candidates.csv", cols, [row])
    index = common.run_geojson_candidate_exporter(common.parse_args([]))
    assert index[0]["geometry_present"] == "true"
    fc = json.load(open(geojson / "v2aq_event_geometry_x.geojson", encoding="utf-8"))
    assert fc["features"][0]["geometry"]["type"] == "Point"
    assert fc["features"][0]["properties"]["patch_truth_allowed"] is False
