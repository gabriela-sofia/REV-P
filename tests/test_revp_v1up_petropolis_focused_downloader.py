import os

import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env, write_csv


def test_focused_downloader_uses_safe_filename(tmp_path, monkeypatch):
    data, _, _, raw = set_env(tmp_path, monkeypatch)
    write_csv(data / "v1up_petropolis_sgb_rigeo_registry.csv", common.RIGEO_COLUMNS, [{
        "rigeo_record_id": "RIGEO_v1up_0000", "event_id": "PET_2022_02_15",
        "item_url": "https://rigeo.sgb.gov.br/handle/doc/22668", "title": "item",
        "publication_year": "2022", "bitstream_url": "https://example.org/a/b/anexo.zip",
        "bitstream_name": "anexo.zip", "format_hint": "zip", "is_public": "true",
        "is_event_specific": "true", "is_geometry_candidate": "true",
        "is_context_only": "false", "blocking_reason": "", "notes": "",
    }])
    write_csv(data / "v1up_petropolis_rj_public_portal_registry.csv", common.PORTAL_COLUMNS, [])
    monkeypatch.setattr(common, "read_bytes_url", lambda *a, **k: b"zip-bytes")
    rows = common.run_focused_downloader(allow_web=True, download=True)
    assert rows[0]["downloaded"] == "true"
    assert rows[0]["safe_filename"].startswith("PET_2022_02_15__SGB_RIGEO__RIGEO_v1up_0000__")
    assert os.path.exists(raw / rows[0]["safe_filename"])
