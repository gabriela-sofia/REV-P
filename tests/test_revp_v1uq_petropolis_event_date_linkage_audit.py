import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, seed_page_text


def test_date_linkage_does_not_treat_generic_year_as_exact_event(tmp_path, monkeypatch):
    data, _, _, _, staging = set_env(tmp_path, monkeypatch)
    seed_page_text(data, staging, ["2022", "15/02/2022"])
    rows = common.run_event_date_linkage_audit()
    by_class = {r["date_signal_class"]: r for r in rows}
    assert by_class["EVENT_YEAR"]["can_support_temporal_gate"] == "false"
    assert by_class["EXACT_EVENT_DATE"]["can_support_temporal_gate"] == "true"
