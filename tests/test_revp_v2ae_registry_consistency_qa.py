import shutil

from tests.test_revp_v2ae_canonical_region_registry_builder import (
    install_base_inputs, set_env, FIXTURE_DIR,
)
import scripts.protocolo_c.revp_v2ae_common as common


def _build_all(data):
    install_base_inputs(data)
    common.run_canonical_region_registry_builder(common.parse_args([]))
    common.run_canonical_event_registry_builder(common.parse_args([]))
    common.run_canonical_event_patch_registry_builder(common.parse_args([]))


def test_consistency_qa_passes_on_consistent_registries(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    _build_all(data)
    rows = common.run_registry_consistency_qa(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0
    assert any(r["check_name"] == "three_regions_present" and r["status"] == "PASS" for r in rows)


def test_consistency_qa_detects_inconsistent_region_registry(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    _build_all(data)
    # Replace the canonical region registry with a 2-region inconsistent fixture.
    shutil.copy(FIXTURE_DIR / "inconsistent_region_registry.csv",
                str(data / "v2ae_canonical_region_registry.csv"))
    rows = common.run_registry_consistency_qa(common.parse_args([]))
    assert any(r["check_name"] == "three_regions_present" and r["status"] == "FAIL" for r in rows)
