import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def test_geometry_blocker_forbids_overlay_and_ground_reference(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_geometry_blocker_attacher()
    assert rows
    assert all(r["no_overlay_executed"] == "true" for r in rows)
    assert all(r["no_coordinates_invented"] == "true" for r in rows)
    assert all(r["overlay_blocker"] for r in rows)
    assert all(r["ground_reference_blocker"] == "ground_reference_forbidden_no_observed_geometry" for r in rows)
    assert all(r["label_blocker"] == "training_label_forbidden" for r in rows)


def test_recife_no_coordinates_petropolis_geometry_missing(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_geometry_blocker_attacher()
    rec = [r for r in rows if r["region"] == "REC"]
    assert rec and all(r["coordinate_status"] == "LOCALITY_ONLY_NO_COORDINATES" for r in rec)
    pet = [r for r in rows if r["region"] == "PET"]
    assert pet and all(r["geometry_status"] == "GEOMETRY_STILL_MISSING" for r in pet)
