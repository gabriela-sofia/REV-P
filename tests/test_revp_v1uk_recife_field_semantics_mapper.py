from scripts.protocolo_c.revp_v1uk_recife_common import field_role


def test_field_mapper_maps_expected_variations():
    assert field_role("Data") == "event_date"
    assert field_role("solicitacao_data") == "event_date"
    assert field_role("Bairro") == "neighborhood"
    assert field_role("logradouro") == "address"
    assert field_role("latitude") == "latitude"
    assert field_role("longitude") == "longitude"
    assert field_role("Defesa Civil") == "unmapped"
