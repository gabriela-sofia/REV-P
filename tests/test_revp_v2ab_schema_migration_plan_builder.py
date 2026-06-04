from tests.test_revp_v2ab_patch_namespace_inventory import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ab_common as common


def test_migration_plan_does_not_modify_prior_outputs(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    # Snapshot a prior output before building the plan.
    prior = data / "v2aa_patch_date_candidate_consolidation.csv"
    before = prior.read_bytes()
    rows = common.run_schema_migration_plan_builder(common.parse_args([]))
    assert rows
    assert all(r["implementation_started"] == "false" for r in rows)
    assert all(r["target_version"].startswith("v2ac") for r in rows)
    # Plan names concrete schema actions (add/deprecate/validate).
    actions = {r["action_type"] for r in rows}
    assert actions & {"add_field", "add_blocker", "deprecate_field", "add_validation"}
    # The prior output is untouched.
    assert prior.read_bytes() == before
