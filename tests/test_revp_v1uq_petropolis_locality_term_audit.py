import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, seed_page_text


def test_locality_audit_does_not_geocode_or_enable_overlay(tmp_path, monkeypatch):
    data, _, _, _, staging = set_env(tmp_path, monkeypatch)
    seed_page_text(data, staging, ["Petropolis Quitandinha Rio Quitandinha Centro."])
    common.run_pdf_structure_inventory()
    common.run_phenomenon_term_indexer()
    common.run_page_level_evidence_builder()
    rows = common.run_locality_term_audit()
    assert rows
    assert all(r["can_support_overlay"] == "false" for r in rows)
    assert "Petropolis" not in rows[0]["locality_token_hash"]
