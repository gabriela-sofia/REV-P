import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, seed_page_text


def test_page_level_evidence_combines_structure_terms_and_keeps_ground_reference_false(tmp_path, monkeypatch):
    data, _, _, _, staging = set_env(tmp_path, monkeypatch)
    seed_page_text(data, staging, ["15/02/2022 alagamento em Quitandinha com mapa."])
    common.run_pdf_structure_inventory()
    common.run_phenomenon_term_indexer()
    rows = common.run_page_level_evidence_builder()
    assert rows[0]["locality_signal_strength"] != "NONE"
    assert rows[0]["date_signal_strength"] == "STRONG"
    assert rows[0]["can_create_ground_reference"] == "false"
