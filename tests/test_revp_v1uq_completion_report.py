import os

import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, write_csv


def test_completion_report_writes_docs_next_actions_and_manifest(tmp_path, monkeypatch):
    data, docs, configs, _, _ = set_env(tmp_path, monkeypatch)
    monkeypatch.setattr(common, "V1UQ_ARTIFACTS", [])
    for name, cols in [
        ("v1uq_petropolis_pdf_text_extraction_registry.csv", common.TEXT_COLUMNS),
        ("v1uq_petropolis_page_level_evidence_registry.csv", common.PAGE_COLUMNS),
        ("v1uq_petropolis_locality_term_audit.csv", common.LOCALITY_COLUMNS),
        ("v1uq_petropolis_event_date_linkage_audit.csv", common.DATE_COLUMNS),
        ("v1uq_petropolis_missing_geodata_signal_audit.csv", common.MISSING_GEODATA_COLUMNS),
    ]:
        write_csv(data / name, cols, [])
    write_csv(data / "v1uq_petropolis_event_status_registry.csv", common.EVENT_STATUS_COLUMNS, [{
        "event_id": "PET_2022_02_15", "v1uq_status": "PHENOMENON_SEPARATION_PARTIAL_TEXTUAL_NO_GEOMETRY", "phenomenon_separation_status": "PHENOMENON_SEPARATION_PARTIAL_TEXTUAL", "documentary_evidence_strength": "MODERATE", "has_missing_geodata_signal": "true", "has_observed_geometry": "false", "ground_truth_operational": "false", "can_create_ground_reference": "false", "can_create_training_label": "false", "can_advance_to_overlay_preflight": "false", "main_blocker": "GEOMETRY_STILL_MISSING", "recommended_next_action": "v1ur - Petropolis Public Geodata Path Recovery", "notes": "",
    }])
    result = common.run_completion_report()
    assert result["next_action"] == "v1ur - Petropolis Public Geodata Path Recovery"
    assert os.path.exists(docs / "protocolo_c_status_atual_v1uq.md")
    assert os.path.exists(configs / "v1uq_petropolis_pdf_text_policy.yaml")
    assert os.path.exists(data / "v1uq_versionable_artifacts_manifest.csv")
