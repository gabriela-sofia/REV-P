import scripts.protocolo_c.revp_v1un_recife_common as common


def test_safe_claims_are_non_operational_and_prohibited_claims_include_patch_label_truth(tmp_path):
    safe, prohibited = common.run_safe_claims_generator(str(tmp_path / "safe.csv"), str(tmp_path / "prohibited.csv"))
    safe_text = " ".join(r["claim_text"].lower() for r in safe)
    prohibited_text = " ".join(r["claim_text"].lower() for r in prohibited)
    assert "validated at patch level" not in safe_text
    assert "training label was created" in prohibited_text
    assert "ground truth was created" in prohibited_text
    assert "patch contains observed flooding" in prohibited_text
    assert all(r["allowed"] == "true" for r in safe)
    assert all(r["allowed"] == "false" for r in prohibited)
