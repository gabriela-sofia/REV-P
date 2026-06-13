"""v2bb autofill and replay readiness tests."""

import scripts.v2bb_public_geometry_retrieval_feed_builder as engine
from tests.v2bb_test_helpers import dirs, feature, read_csv, write_geojson


def test_autofill_candidates_created_only_for_valid_geometry(tmp_path):
    paths = dirs(tmp_path)
    write_geojson(paths["external_dir"]/"raw"/"patch.geojson", feature("patch_boundary", "REC_00019"))
    engine.run("build_feeds", **paths)
    created = engine.update_autofill(engine.resolve_dirs(**paths))
    assert len(created) == 1
    assert read_csv(created[0])[0]["review_status"] == "provided_unreviewed"


def test_replay_readiness_stays_tp0_without_valid_pair():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    row = engine.load_csv(root/"datasets"/"v2bb_replay_readiness_update.csv")[0]
    assert row["can_attempt_v2az_replay"] == "false"
    assert row["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE_WITH_PUBLIC_SEARCH_DOSSIER"
