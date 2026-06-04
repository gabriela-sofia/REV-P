import scripts.protocolo_c.revp_v1um_recife_common as common
from tests.test_revp_v1um_recife_locality_candidate_sampler import make_base


def test_normalizer_uses_hash_tokens_and_never_geocodes(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    rows = common.run_locality_text_normalizer(str(data / "norm.csv"))
    assert rows[0]["normalized_locality_token"].startswith("neighborhood_hash_")
    assert rows[0]["can_support_overlay"] == "false"
    assert "no_geocoding" in rows[0]["notes"]
