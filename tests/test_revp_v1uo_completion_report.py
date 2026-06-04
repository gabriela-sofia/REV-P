import os

import scripts.protocolo_c.revp_v1uo_multiregion_common as common
from tests.test_revp_v1uo_multiregion_event_registry_builder import make_base


def test_completion_report_writes_technical_docs_and_manifest(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    docs = tmp_path / "docs" / "metodologia_cientifica"
    configs = tmp_path / "configs" / "protocolo_c"
    docs.mkdir(parents=True)
    configs.mkdir(parents=True)
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(configs))
    monkeypatch.setattr(common, "V1UO_ARTIFACTS", [])
    common.run_multiregion_event_registry_builder(str(data / "v1uo_multiregion_event_registry.csv"))
    common.run_public_source_registry_builder(str(data / "v1uo_public_source_registry.csv"))
    common.run_region_adapter_factory(str(data / "v1uo_region_adapter_registry.csv"))
    common.run_multiregion_public_discovery_runner(str(data / "v1uo_multiregion_public_discovery_registry.csv"))
    common.run_multiregion_schema_audit_runner(str(data / "v1uo_multiregion_schema_audit_registry.csv"))
    common.run_multiregion_candidate_router(str(data / "v1uo_multiregion_candidate_router.csv"))
    common.run_multiregion_evidence_matrix_builder(str(data / "v1uo_multiregion_evidence_matrix.csv"))
    common.run_ground_truth_opportunity_ranker(str(data / "v1uo_ground_truth_opportunity_ranker.csv"))
    common.run_event_patch_package_prebuilder(str(data / "v1uo_event_patch_package_prebuild_registry.csv"))
    result = common.run_completion_report()
    assert result["top_ranked_event"] == "PET_2024_03_21_28"
    assert os.path.exists(docs / "protocolo_c_status_atual_v1uo.md")
    assert os.path.exists(data / "v1uo_versionable_artifacts_manifest.csv")
