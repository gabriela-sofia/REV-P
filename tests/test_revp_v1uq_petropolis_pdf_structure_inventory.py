import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, seed_page_text


def test_structure_inventory_detects_map_table_annex_signals(tmp_path, monkeypatch):
    data, _, _, _, staging = set_env(tmp_path, monkeypatch)
    seed_page_text(data, staging, ["Figura e mapa em anexo com tabela e geodados SIG."])
    rows = common.run_pdf_structure_inventory()
    assert rows[0]["likely_map_page"] == "true"
    assert rows[0]["likely_table_page"] == "true"
    assert rows[0]["likely_annex_page"] == "true"
    assert rows[0]["geodata_reference_signal"] == "true"
