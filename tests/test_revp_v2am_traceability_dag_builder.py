import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all, read_csv


def test_dag_nodes_edges_and_mmd(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    nodes, edges = common.run_traceability_dag_builder(common.parse_args([]))
    node_ids = {n["node_id"] for n in nodes}
    for nid in ("N_v2ah_stop_gate", "N_v2ai_assignments", "N_v2aj_claims",
                "N_v2ak_drafts", "N_v2al_bundles", "N_v2am_atlas"):
        assert nid in node_ids
    assert all(e["promotion_created"] == "false" for e in edges)
    rels = {(e["source_node"], e["target_node"]) for e in edges}
    assert ("N_v2ah_review_queue", "N_v2ai_assignments") in rels
    assert ("N_v2al_bundles", "N_v2am_atlas") in rels
    mmd = (atlas / "v2am_traceability_dag.mmd").read_text(encoding="utf-8")
    assert mmd.startswith("flowchart TD")
    edge_rows = read_csv(data / "v2am_traceability_dag_edges.csv")
    assert all(r["promotion_created"] == "false" for r in edge_rows)
