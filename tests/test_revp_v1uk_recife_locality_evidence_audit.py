from scripts.protocolo_c.revp_v1uk_recife_common import locality_status_for_row, role_fields


def test_textual_locality_does_not_become_coordinate():
    row = {"Bairro": "Ibura", "Endereco": "Rua A"}
    roles = role_fields(row.keys())
    assert locality_status_for_row(row, roles) == "ADDRESS_TEXT_AVAILABLE"
    assert not roles.get("latitude")
    assert not roles.get("longitude")
