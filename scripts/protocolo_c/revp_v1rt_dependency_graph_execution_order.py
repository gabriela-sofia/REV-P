"""REV-P v1rt — Dependency graph and execution order.

Produces a CSV edge list of inter-block dependencies and an explicit execution
order. No networkx required; pure CSV. No science, no labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rs_v1rz_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p, assert_clean_rows, guardrail_row,
    write_csv_with_header, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_EDGES = _p("REVP_V1RT_OUT_EDGES", DATASETS / "protocol_c_dependency_graph_edges_v1rt.csv")
OUT_ORDER = _p("REVP_V1RT_OUT_ORDER", DATASETS / "protocol_c_execution_order_v1rt.csv")
OUT_SUMMARY = _p("REVP_V1RT_OUT_SUMMARY", DATASETS / "protocol_c_dependency_graph_summary_v1rt.csv")
SCHEMA_EDGES = _p("REVP_V1RT_SCHEMA_EDGES", SCHEMAS / "protocol_c_dependency_graph_edges_v1rt_schema.csv")
SCHEMA_ORDER = _p("REVP_V1RT_SCHEMA_ORDER", SCHEMAS / "protocol_c_execution_order_v1rt_schema.csv")
SCHEMA_SUM = _p("REVP_V1RT_SCHEMA_SUM", SCHEMAS / "protocol_c_dependency_graph_summary_v1rt_schema.csv")
DOC = _p("REVP_V1RT_DOC", DOCS / "revp_v1rt_dependency_graph_execution_order.md")

EDGE_FIELDS = ["edge_id", "source_block", "source_artifact", "target_block",
               "target_artifact", "dependency_type", "required",
               "fail_closed_behavior", "notes"]
ORDER_FIELDS = ["order_id", "seq", "block", "key_script", "depends_on",
                "status_gate", "outputs_to", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

# Static dependency edges between blocks
_EDGES: list[tuple[str, ...]] = [
    # DINO chain
    ("E001", "DINO_REPRESENTATION_V1PG_V1PM", "dino_artifact_discovery_v1pg.csv",
     "DINO_HARNESS_V1PN_V1PT", "dino_backend_model_probe_v1pp.csv", "INPUT", "true", "FAIL_CLOSED"),
    ("E002", "DINO_HARNESS_V1PN_V1PT", "dino_controlled_smoke_embedding_results_v1pq.csv",
     "DINO_VISUAL_V1PU_V1PZ", "dino_visual_asset_queue_v1pu.csv", "INPUT", "true", "FAIL_CLOSED"),
    ("E003", "DINO_VISUAL_V1PU_V1PZ", "dino_embedding_execution_queue_v1po.csv",
     "DINO_BRIDGE_V1QA_V1QF", "dino_dry_run_execution_plan_v1qc.csv", "INPUT", "true", "FAIL_CLOSED"),
    ("E004", "DINO_BRIDGE_V1QA_V1QF", "dino_dry_run_execution_plan_v1qc.csv",
     "DINO_SMOKE_V1QG_V1QM", "dino_controlled_smoke_embedding_results_v1pq.csv", "INPUT", "true", "FAIL_CLOSED"),
    ("E005", "DINO_SMOKE_V1QG_V1QM", "dino_controlled_smoke_embedding_results_v1pq.csv",
     "DINO_LOCAL_V1QN_V1QT", "dino_local_execution_config_v1qq.csv", "INPUT", "true", "FAIL_CLOSED"),
    # P0
    ("E006", "DINO_LOCAL_V1QN_V1QT", "dino_local_readiness_bundle_summary_v1qt.csv",
     "GROUND_REF_P0_V1QU_V1QZ", "protocol_c_official_evidence_source_requirements_v1qu.csv", "CONTEXT", "false", "CONTINUE"),
    ("E007", "GROUND_REF_P0_V1QU_V1QZ", "protocol_c_event_patch_review_sample_v1qv.csv",
     "EXTERNAL_INTAKE_P1_V1RA_V1RF", "protocol_c_external_collection_task_board_v1ra.csv", "INPUT", "true", "FAIL_CLOSED"),
    # P1 -> P2
    ("E008", "EXTERNAL_INTAKE_P1_V1RA_V1RF", "protocol_c_external_event_candidates_v1rd.csv",
     "REVIEW_GATE_P2_V1RG_V1RM", "protocol_c_review_response_intake_template_v1rg.csv", "INPUT", "false", "WAITING"),
    # P2 -> P3
    ("E009", "REVIEW_GATE_P2_V1RG_V1RM", "protocol_c_review_supervisor_gate_scientific_summary_v1rm.csv",
     "DASHBOARD_P3_V1RN_V1RR", "protocol_c_scientific_roadmap_summary_v1rr.csv", "INPUT", "false", "CONTINUE"),
    # P0 -> P3
    ("E010", "GROUND_REF_P0_V1QU_V1QZ", "protocol_c_double_review_packet_manifest_v1qw.csv",
     "REVIEW_GATE_P2_V1RG_V1RM", "protocol_c_review_response_intake_template_v1rg.csv", "INPUT", "true", "FAIL_CLOSED"),
    # P3 -> Integration
    ("E011", "DASHBOARD_P3_V1RN_V1RR", "protocol_c_scientific_roadmap_summary_v1rr.csv",
     "INTEGRATION_V1RS_V1RZ", "protocol_c_integrated_artifact_inventory_v1rs.csv", "INPUT", "false", "CONTINUE"),
]

# Execution order
_ORDER: list[tuple[str, ...]] = [
    ("1", "DINO_REPRESENTATION_V1PG_V1PM", "v1pg/v1ph/v1pi/v1pj/v1pk/v1pl/v1pm",
     "—", "DINO_EMBEDDINGS_NOT_FOUND_FAIL_CLOSED",
     "DINO_HARNESS_V1PN_V1PT", "requires local model"),
    ("2", "DINO_HARNESS_V1PN_V1PT", "v1pn/v1po/v1pp/v1pq/v1pr/v1ps/v1pt",
     "DINO_REPRESENTATION_V1PG_V1PM", "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED",
     "DINO_VISUAL_V1PU_V1PZ", "requires local model"),
    ("3", "DINO_VISUAL_V1PU_V1PZ", "v1pu/v1pv/v1pw/v1px/v1py/v1pz",
     "DINO_HARNESS_V1PN_V1PT", "DINO_VISUAL_QUEUE_READY_REVIEW_ONLY",
     "DINO_BRIDGE_V1QA_V1QF", ""),
    ("4", "DINO_BRIDGE_V1QA_V1QF", "v1qa/v1qb/v1qc/v1qd/v1qe/v1qf",
     "DINO_VISUAL_V1PU_V1PZ", "DINO_EXECUTION_BRIDGE_READY_DRY_RUN_MODEL_MISSING",
     "DINO_SMOKE_V1QG_V1QM", ""),
    ("5", "DINO_SMOKE_V1QG_V1QM", "v1qg/v1qh/v1qi/v1qj/v1qk/v1ql/v1qm",
     "DINO_BRIDGE_V1QA_V1QF", "DINO_SMOKE_EMBEDDINGS_MODEL_MISSING_FAIL_CLOSED",
     "DINO_LOCAL_V1QN_V1QT", "requires local model"),
    ("6", "DINO_LOCAL_V1QN_V1QT", "v1qn/v1qo/v1qp/v1qq/v1qr/v1qs/v1qt",
     "DINO_SMOKE_V1QG_V1QM", "LOCAL_DINO_READINESS_MODEL_MISSING_FAIL_CLOSED",
     "GROUND_REF_P0_V1QU_V1QZ", "requires local model"),
    ("7", "GROUND_REF_P0_V1QU_V1QZ", "v1qu/v1qv/v1qw/v1qx/v1qy/v1qz",
     "—", "GROUND_REFERENCE_REVIEW_NOT_COMPLETED_FAIL_CLOSED",
     "EXTERNAL_INTAKE_P1_V1RA_V1RF", "parallel to DINO chain"),
    ("8", "EXTERNAL_INTAKE_P1_V1RA_V1RF", "v1ra/v1rb/v1rc/v1rd/v1re/v1rf",
     "GROUND_REF_P0_V1QU_V1QZ", "EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS",
     "REVIEW_GATE_P2_V1RG_V1RM", "requires manual documents"),
    ("9", "REVIEW_GATE_P2_V1RG_V1RM", "v1rg/v1rh/v1ri/v1rj/v1rk/v1rl/v1rm",
     "EXTERNAL_INTAKE_P1_V1RA_V1RF;GROUND_REF_P0_V1QU_V1QZ",
     "REVIEW_SUPERVISOR_GATE_WAITING_MANUAL_RESPONSES",
     "DASHBOARD_P3_V1RN_V1RR", "requires manual review responses"),
    ("10", "DASHBOARD_P3_V1RN_V1RR", "v1rn/v1ro/v1rp/v1rq/v1rr",
     "REVIEW_GATE_P2_V1RG_V1RM", "REV_P_WAITING_EXTERNAL_DOCUMENT_INTAKE",
     "INTEGRATION_V1RS_V1RZ", ""),
    ("11", "INTEGRATION_V1RS_V1RZ", "v1rs/v1rt/v1ru/v1rv/v1rw/v1rx/v1ry/v1rz",
     "DASHBOARD_P3_V1RN_V1RR", "INTEGRATION_HARDENING_WAITING_EXTERNAL_EVIDENCE",
     "—", "hardening and commit prep"),
]


def run(datasets: Path | None = None) -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    for e in _EDGES:
        row = {
            "edge_id": e[0], "source_block": e[1], "source_artifact": e[2],
            "target_block": e[3], "target_artifact": e[4], "dependency_type": e[5],
            "required": e[6], "fail_closed_behavior": e[7], "notes": "",
        }
        edges.append(row)

    order: list[dict[str, Any]] = []
    for o in _ORDER:
        seq, block, key, depends, gate, outputs, notes = o
        row = {"order_id": f"V1RT_O{int(seq):02d}", "seq": seq, "block": block,
               "key_script": key, "depends_on": depends, "status_gate": gate,
               "outputs_to": outputs, "notes": notes}
        order.append(row)

    write_csv_with_header(OUT_EDGES, edges, EDGE_FIELDS)
    write_csv_with_header(OUT_ORDER, order, ORDER_FIELDS)
    write_schema_safe(SCHEMA_EDGES, EDGE_FIELDS, "v1rt_edges")
    write_schema_safe(SCHEMA_ORDER, ORDER_FIELDS, "v1rt_order")

    summary = [
        {"stat_key": "total_edges", "stat_value": str(len(edges))},
        {"stat_key": "execution_steps", "stat_value": str(len(order))},
        {"stat_key": "required_edges", "stat_value": str(sum(1 for e in edges if e["required"] == "true"))},
        {"stat_key": "stage", "stat_value": "v1rt"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_safe(SCHEMA_SUM, SUM_FIELDS, "v1rt_summary")

    write_doc(DOC, "v1rt — Dependency Graph and Execution Order", [
        "## Objetivo",
        "CSV edge list das dependências inter-bloco e ordem de execução explícita "
        "(DINO v1pg-v1qt, P0-P3 v1qu-v1rr, integration v1rs-v1rz).",
        "## Resultado",
        f"Arestas: {len(edges)}. Passos de execução: {len(order)}.",
        "## Bloqueios conhecidos",
        "DINO chain: modelo local ausente. P1: documentos externos pendentes. "
        "P2: respostas A/B pendentes. P3: estado final de espera.",
    ])
    print(f"[v1rt] edges={len(edges)} order={len(order)}")
    return {"edges": len(edges), "order": len(order)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rt dependency graph").parse_args()
    run()
