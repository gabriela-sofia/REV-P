import os

import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_orientation_packet_is_short_technical_and_safe(tmp_path, monkeypatch):
    data, docs = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_orientation_meeting_packet_builder(common.parse_args([]))
    assert len(rows) == 5
    assert os.path.exists(docs / "protocolo_c_v2aj_orientation_meeting_packet.md")
    assert all("operational" not in r["recommended_wording"].lower() for r in rows)
