import csv
import json

import pytest

import scripts.protocolo_c.revp_v2as_common as common


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    protocol = datasets / "protocolo_c"
    docs = tmp_path / "docs" / "protocolo_c" / "v2as_official_geometry_deep_probe"
    geojson = docs / "geojson_candidates"
    cache = docs / "evidence_cache"
    cfg = tmp_path / "configs" / "protocolo_c"
    for path in (datasets, protocol, docs, geojson, cache, cfg):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_ROOT", str(datasets))
    monkeypatch.setattr(common, "DATASET_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "GEOJSON_DIR", str(geojson))
    monkeypatch.setattr(common, "CACHE_DIR", str(cache))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.delenv(common.NETWORK_ENV, raising=False)
    return datasets, protocol, docs, geojson, cache


# candidate, region, level, band, anchor, geom_status, can_digit, has_xc, patch
_SPECS = [
    ("PET_2022_02_15", "Petropolis", "C4_READY_FOR_EXTERNAL_VALIDATION_REVIEW",
     "HIGH_PRIORITY", "mapa_ou_laudo", "OFFICIAL_MAP_DIGITIZATION_REQUIRED", True, False, ""),
    ("REC_2023_02_05_06", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     "HIGH_PRIORITY", "rua", "OFFICIAL_MAP_DIGITIZATION_REQUIRED", True, True, "REC_00001"),
    ("REC_2022_05_24_30", "Recife", "C3_STRONG_REFERENCE_CANDIDATE",
     "MEDIUM_PRIORITY", "bairro", "TEXTUAL_ANCHOR_ONLY", False, True, "REC_00002"),
    ("PET_2024_03_21_28", "Petropolis", "C1_CONTEXTUAL_OBSERVED_EVENT",
     "LOW_PRIORITY", "municipio", "INSUFFICIENT_GEOMETRY", False, False, ""),
]
_BAND_TO_ANCHOR = {"HIGH_PRIORITY": "HIGH", "MEDIUM_PRIORITY": "MEDIUM", "LOW_PRIORITY": "LOW"}


def install_stack(protocol, datasets):
    priority, readiness, license_crs, sources = [], [], [], []
    event_geom, anchor, crosswalk, patch_match = [], [], [], []
    spatial, metadata, observed, stub = [], [], [], []
    for rank, (cid, region, level, band, atype, gstatus, can_d, has_xc, patch) in enumerate(_SPECS, 1):
        priority.append({"priority_id": f"GDP_{cid}", "candidate_id": cid, "region": region,
                         "reference_level": level, "anchor_strength_band": _BAND_TO_ANCHOR[band],
                         "strongest_anchor_type": atype, "geometry_candidate_status": gstatus,
                         "manual_digitization_required": "true",
                         "patch_link_review_packet_exists": "true", "priority_rank": str(rank),
                         "priority_band": band, "priority_reason": "x"})
        dstatus = ("GEOMETRY_DIGITIZATION_READY" if can_d else
                   "INSUFFICIENT_SPATIAL_EVIDENCE" if gstatus == "INSUFFICIENT_GEOMETRY"
                   else "NEEDS_SPECIFIC_LOCALITY_GEOMETRY")
        readiness.append({"readiness_id": f"DR_{cid}", "candidate_id": cid, "region": region,
                          "can_digitize_now": "true" if can_d else "false",
                          "digitization_status": dstatus})
        license_crs.append({"checklist_id": f"LC_{cid}", "candidate_id": cid, "source": "x",
                            "license_status": "UNKNOWN_NEEDS_REVIEW",
                            "crs_status": "NOT_DOCUMENTED_NEEDS_ASSIGNMENT",
                            "reuse_allowed": "false", "coordinate_system_documented": "false"})
        sources.append({"source_registry_id": f"OGS_{cid}_primary", "candidate_id": cid,
                        "region": region, "source_name": f"Fonte oficial {region}",
                        "source_url_or_document": f"https://example.gov.br/{cid.lower()}",
                        "source_type": "OFFICIAL_MUNICIPAL", "source_role": "primary",
                        "raw_data_downloaded": "false", "raw_data_versioned": "false"})
        event_geom.append({"event_geometry_id": f"EG_{cid}", "candidate_id": cid, "region": region,
                           "geometry_status": gstatus, "manual_digitization_required": "true",
                           "can_use_for_ground_truth": "false", "explicit_geometry_geojson": ""})
        anchor.append({"anchor_strength_id": f"AS_{cid}", "candidate_id": cid,
                       "strongest_anchor_type": atype, "anchor_strength_band": _BAND_TO_ANCHOR[band]})
        crosswalk.append({"join_id": f"JN_{cid}", "candidate_id": cid, "patch_id": patch,
                          "has_crosswalk_candidate": "true" if has_xc else "false",
                          "patch_truth_allowed": "false"})
        patch_match.append({"match_candidate_id": f"MT_{cid}", "candidate_id": cid,
                            "patch_id": patch, "overlay_executed": "false"})
        spatial.append({"spatial_anchor_id": f"SA_{cid}", "candidate_id": cid, "region": region,
                        "anchor_text": f"{atype} documentado", "anchor_type": atype,
                        "geometry_available": "false", "coordinate_available": "false",
                        "notes": "Ancora textual; sem coordenada inventada."})
        metadata.append({"metadata_id": f"MD_{cid}", "candidate_id": cid, "source_role": "primary",
                         "mentions_geometry_or_map": "false",
                         "limitations": "Metadados leves; sem download de bruto."})
        observed.append({"observed_event_id": cid, "region": region,
                         "primary_source_name": f"Fonte oficial {region}",
                         "primary_source_url": f"https://example.gov.br/{cid.lower()}",
                         "primary_source_type": "OFFICIAL_MUNICIPAL"})
        stub.append({"candidate_id": cid, "note": "stub"})
    write_csv(protocol / "v2ar_geometry_digitization_priority.csv", list(priority[0].keys()), priority)
    write_csv(protocol / "v2ar_digitization_readiness_matrix.csv", list(readiness[0].keys()), readiness)
    write_csv(protocol / "v2ar_license_crs_checklist.csv", list(license_crs[0].keys()), license_crs)
    write_csv(protocol / "v2ar_official_geometry_source_registry.csv", list(sources[0].keys()), sources)
    # remaining v2ar inputs required by load_v2ar_stack (presence only)
    for name in ("v2ar_source_access_metadata_probe.csv", "v2ar_geometry_extraction_attempts.csv",
                 "v2ar_geojson_candidate_index.csv", "v2ar_geojson_export_validation.csv",
                 "v2ar_patch_link_external_validation_packet.csv",
                 "v2ar_digitization_task_refinement.csv", "v2ar_patch_truth_boundary_audit.csv"):
        write_csv(protocol / name, list(stub[0].keys()), stub)
    write_csv(protocol / "v2aq_event_geometry_candidates.csv", list(event_geom[0].keys()), event_geom)
    write_csv(protocol / "v2aq_spatial_anchor_strength.csv", list(anchor[0].keys()), anchor)
    write_csv(protocol / "v2aq_crosswalk_geometry_join_candidates.csv", list(crosswalk[0].keys()), crosswalk)
    write_csv(protocol / "v2aq_patch_geometry_match_candidates.csv", list(patch_match[0].keys()), patch_match)
    write_csv(protocol / "v2an_spatial_anchor_registry.csv", list(spatial[0].keys()), spatial)
    write_csv(protocol / "v2an_document_metadata_registry.csv", list(metadata[0].keys()), metadata)
    write_csv(datasets / "observed_event_reference_candidate_registry.csv",
              list(observed[0].keys()), observed)


def install_all(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = set_env(tmp_path, monkeypatch)
    install_stack(protocol, datasets)
    return datasets, protocol, docs, geojson, cache


def inject_explicit_geometry(protocol, candidate_id, geometry):
    rows = read_csv(protocol / "v2aq_event_geometry_candidates.csv")
    for r in rows:
        if r["candidate_id"] == candidate_id:
            r["geometry_status"] = "EXPLICIT_POINT_OR_COORDINATE_AVAILABLE"
            r["explicit_geometry_geojson"] = json.dumps(geometry)
    write_csv(protocol / "v2aq_event_geometry_candidates.csv", list(rows[0].keys()), rows)


# --- common tests ----------------------------------------------------------
def test_loads_v2ar_stack(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    stack = common.load_v2ar_stack()
    assert len(stack["priority"]) == 4


def test_fails_when_v2ar_missing(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.load_v2ar_stack()


def test_network_disabled_by_default(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    assert common.is_network_enabled() is False
    probe = common.fetch_small_to_ignored_cache("https://example.gov.br/x", "PET_2022_02_15")
    assert probe["fetch_status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN"
    assert probe["cached_temporarily"] == "false"


def test_detect_geometry_payload_formats():
    assert common.detect_geometry_payload(json.dumps(
        {"type": "Point", "coordinates": [-34.9, -8.05]}))["explicit_geometry_found"]
    assert common.detect_geometry_payload("[-35.0,-8.1,-34.8,-7.9]")["bbox_found"]
    assert common.detect_geometry_payload("POINT(-43.18 -22.5)")["wkt_found"]
    assert common.detect_geometry_payload(
        "<coordinates>-49.3,-25.4,0</coordinates>")["kml_found"]
    assert common.detect_geometry_payload("lat,lon\n-8.05,-34.9\n")["csv_point_found"]


def test_detect_rejects_broad_text_and_geocoding():
    det = common.detect_geometry_payload("Bairro Boa Viagem, Recife, muito atingido pelas chuvas")
    assert det["explicit_geometry_found"] is False
    assert det["geometry"] is None


def test_validate_geojson_geometry_payload():
    assert common.validate_geojson_geometry_payload({"type": "Point", "coordinates": [-34.9, -8.05]})
    assert not common.validate_geojson_geometry_payload({"type": "Point", "coordinates": ["x", None]})
    assert not common.validate_geojson_geometry_payload({"type": "Nope", "coordinates": [1, 2]})


def test_representative_lat_lon():
    assert common.representative_lat_lon({"type": "Point", "coordinates": [-34.9, -8.05]}) == (-8.05, -34.9)
    poly = {"type": "Polygon", "coordinates": [[[-35, -8], [-34, -8], [-34, -7], [-35, -8]]]}
    assert common.representative_lat_lon(poly) == (-8.0, -35.0)


def test_assert_no_operational_promotion():
    common.assert_no_operational_promotion([{"forbidden_use": common.FORBIDDEN_USE}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"patch_truth_allowed": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"note": "geometry_inferred=true"}])


def test_assert_no_raw_data_versioned():
    common.assert_no_raw_data_versioned([{"raw_data_versioned": "false"}])
    with pytest.raises(ValueError):
        common.assert_no_raw_data_versioned([{"raw_data_versioned": "true"}])


def test_assert_no_fake_geometry():
    common.assert_no_fake_geometry([{"explicit_geometry_found": "false"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_geometry([{"geometry_invented": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_geometry([{"geocoded_as_geometry": "true"}])


def test_assert_output_is_v2as():
    common.assert_output_is_v2as("datasets/protocolo_c/v2as_x.csv")
    common.assert_output_is_v2as("docs/protocolo_c/v2as_official_geometry_deep_probe/evidence_cache/.gitignore")
    with pytest.raises(ValueError):
        common.assert_output_is_v2as("datasets/protocolo_c/v2ar_x.csv")


def test_ensure_cache_dir_writes_gitignore(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    common.ensure_cache_dir()
    gi = common.cache_path(".gitignore")
    assert common.read_text(gi) == "*\n!.gitignore\n"


def test_detect_candidate_geometry_from_explicit(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    inject_explicit_geometry(protocol, "PET_2022_02_15", {"type": "Point", "coordinates": [-43.18, -22.5]})
    eg = {r["candidate_id"]: r for r in read_csv(protocol / "v2aq_event_geometry_candidates.csv")}
    det = common.detect_candidate_geometry("PET_2022_02_15", eg["PET_2022_02_15"])
    assert det["explicit_geometry_found"] is True
    assert det["geometry"]["type"] == "Point"
