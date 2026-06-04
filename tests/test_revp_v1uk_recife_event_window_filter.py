from scripts.protocolo_c.revp_v1uk_recife_common import parse_date, window_type


def test_event_window_filter_identifies_inside_and_outside():
    assert window_type(parse_date("25/05/2022")) == "event_core_window"
    assert window_type(parse_date("22/05/2022")) == "pre_event_3d"
    assert window_type(parse_date("18/05/2022")) == "pre_event_7d"
    assert window_type(parse_date("10/07/2022")) == "outside_event_window"
