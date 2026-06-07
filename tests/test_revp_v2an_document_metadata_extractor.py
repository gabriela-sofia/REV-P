import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_metadata_light_only(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    common.run_source_access_probe(common.parse_args([]))
    rows = common.run_document_metadata_extractor(common.parse_args([]))
    assert rows
    for r in rows:
        assert len(r["evidence_fragment_safe"]) <= 140
        assert r["metadata_extraction_status"] in ("EXTRACTED_LIGHT", "SKIPPED_NO_ACCESS")
    assert (docs / "v2an_document_metadata_registry.md").exists()
