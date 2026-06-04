import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, seed_page_text


def test_missing_geodata_audit_detects_signal_without_inventing_file(tmp_path, monkeypatch):
    data, _, _, _, staging = set_env(tmp_path, monkeypatch)
    seed_page_text(data, staging, ["O anexo digital SIG cita shapefile e camada vetorial."])
    rows = common.run_missing_geodata_signal_audit()
    assert rows
    assert rows[0]["can_be_resolved_by_public_search"] == "true"
    assert rows[0]["public_path_hint"] == "SGB_RIGEO_PUBLIC_SOURCE_RECHECK"
