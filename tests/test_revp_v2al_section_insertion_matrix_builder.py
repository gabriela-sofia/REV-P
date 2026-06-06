import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_insertion_matrix_maps_drafts(tmp_path, monkeypatch):
    data, _, _ = install_all(tmp_path, monkeypatch)
    common.run_section_insertion_matrix_builder(common.parse_args([]))
    rows = read_csv(data / "v2al_section_insertion_matrix.csv")
    sources = [r["v2ak_source_draft"] for r in rows]
    assert any("metodologia" in s for s in sources)
    assert any("resultados" in s for s in sources)
    assert any("discussao" in s for s in sources)
    assert any("limitacoes" in s for s in sources)
    assert any("briefing" in s for s in sources)
    assert all(r["insertion_mode"] == "manual_review_required" for r in rows)
    assert all(r["required_review"] == "true" for r in rows)
    briefing = [r for r in rows if "briefing" in r["v2ak_source_draft"]][0]
    assert "corpo" in briefing["recommended_position"].lower() or \
        "reuniao" in briefing["target_tcc_section"].lower()
