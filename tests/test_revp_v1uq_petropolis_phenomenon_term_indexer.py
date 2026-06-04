import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, seed_page_text


def test_phenomenon_index_separates_flood_landslide_mixed_and_context(tmp_path, monkeypatch):
    data, _, _, _, staging = set_env(tmp_path, monkeypatch)
    seed_page_text(data, staging, [
        "alagamento e inundacao",
        "deslizamento e movimento de massa",
        "alagamento e deslizamento",
        "risco geologico e suscetibilidade",
    ])
    rows = common.run_phenomenon_term_indexer()
    classes = {r["phenomenon_class"] for r in rows}
    assert "FLOOD_OR_INUNDATION" in classes or "URBAN_FLOODING" in classes
    assert "LANDSLIDE_OR_MASS_MOVEMENT" in classes
    assert "MIXED_HYDRO_GEO" in classes
    assert "RISK_OR_SUSCEPTIBILITY_CONTEXT" in classes
