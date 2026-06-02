"""REV-P v1tx — Case dossier exporter.

One readable evidence dossier per case, consolidating the v1tn-v1tw automated
review results. Review-only; no operational labels/targets/ground truth.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tx_v1ub_tcc_dossier_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails,
    build_dossier_sections, DOSSIER_SECTION_KEYS,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_DOS = _p("REVP_V1TX_OUT_DOS", DATASETS / "protocol_c_case_dossier_v1tx.csv")
OUT_SEC = _p("REVP_V1TX_OUT_SEC", DATASETS / "protocol_c_case_dossier_sections_v1tx.csv")
OUT_SUM = _p("REVP_V1TX_OUT_SUM", DATASETS / "protocol_c_case_dossier_summary_v1tx.csv")
SCHEMA_D = _p("REVP_V1TX_SCHEMA_D", SCHEMAS / "protocol_c_case_dossier_v1tx_schema.csv")
SCHEMA_E = _p("REVP_V1TX_SCHEMA_E", SCHEMAS / "protocol_c_case_dossier_sections_v1tx_schema.csv")
SCHEMA_S = _p("REVP_V1TX_SCHEMA_S", SCHEMAS / "protocol_c_case_dossier_summary_v1tx_schema.csv")
DOC = _p("REVP_V1TX_DOC", DOCS / "revp_v1tx_case_dossier_exporter.md")

DOS_FIELDS = ["case_id", "region", "hazard_type", "event_window"] + \
    DOSSIER_SECTION_KEYS + [
    "review_only", "automated_review",
    "internal_review_automated_for_review_only",
    "requires_external_observational_evidence_for_operational_claim",
    "automatic_c3_promotion", "c4_opened",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative", "notes"]
SEC_FIELDS = ["case_id", "section_order", "section_key", "section_text",
              "review_only", "automated_review"]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    flow = {r.get("case_id", ""): r for r in
            read_csv_safe(DATASETS / "protocol_c_single_flow_review_export_v1ts.csv")}
    sup = {r.get("case_id", ""): r for r in
           read_csv_safe(DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")}
    proof = {r.get("case_id", ""): r for r in
             read_csv_safe(DATASETS / "protocol_c_proof_of_review_only_validation_audit_v1tu.csv")}

    dos_rows: list[dict[str, Any]] = []
    sec_rows: list[dict[str, Any]] = []
    for c in cases:
        cid = c.get("case_id", "")
        if cid.startswith("FAIL_CLOSED"):
            continue
        sections = build_dossier_sections(
            c, flow.get(cid, {}), sup.get(cid, {}), proof.get(cid, {}))
        row: dict[str, Any] = {
            "case_id": cid, "region": c.get("region", ""),
            "hazard_type": c.get("hazard_type", ""),
            "event_window": c.get("event_window", ""), "notes": "",
        }
        row.update(sections)
        row.update(guardrail_row_review())
        dos_rows.append(row)
        for i, k in enumerate(DOSSIER_SECTION_KEYS):
            sec_rows.append({
                "case_id": cid, "section_order": str(i), "section_key": k,
                "section_text": sections[k],
                "review_only": "true", "automated_review": "true",
            })

    if not dos_rows:
        dos_rows = [{
            "case_id": "FAIL_CLOSED_NO_CASES", "region": "", "hazard_type": "",
            "event_window": "", "notes": "no inputs",
            **{k: "" for k in DOSSIER_SECTION_KEYS}, **guardrail_row_review(),
        }]
        sec_rows = [{
            "case_id": "FAIL_CLOSED_NO_CASES", "section_order": "0",
            "section_key": "identificacao", "section_text": "no inputs",
            "review_only": "true", "automated_review": "true",
        }]

    for label, rws in (("v1tx_dos", dos_rows), ("v1tx_sec", sec_rows)):
        viol = scan_guardrails(rws, label)
        if viol:
            raise ValueError(f"Guardrail violations {label}: {viol[:3]}")

    write_csv_with_header(OUT_DOS, dos_rows, DOS_FIELDS)
    write_csv_with_header(OUT_SEC, sec_rows, SEC_FIELDS)
    write_schema(SCHEMA_D, DOS_FIELDS, "v1tx_dossier")
    write_schema(SCHEMA_E, SEC_FIELDS, "v1tx_sections")

    summary = [
        {"stat_key": "dossiers_total", "stat_value": str(len(dos_rows))},
        {"stat_key": "sections_total", "stat_value": str(len(sec_rows))},
        {"stat_key": "sections_per_case", "stat_value": str(len(DOSSIER_SECTION_KEYS))},
        {"stat_key": "stage", "stat_value": "v1tx"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tx_summary")

    write_doc(DOC, "v1tx — Case Dossier Exporter", [
        "## Objetivo",
        "Exportar um dossie de evidencia legivel por caso a partir da revisao "
        "automatizada review-only (v1tn-v1tw).",
        f"## Resultado\nDossies: {len(dos_rows)}. Secoes: {len(sec_rows)}.",
        "## Limitacao",
        "DINO/hidromet sao contexto; ausencia nao e negativo. Sem C3 automatico, "
        "sem C4, sem ground truth/rotulo/target/negativo formal. Fonte externa "
        "exigida para afirmacao operacional.",
    ])
    print(f"[v1tx] dossiers={len(dos_rows)} sections={len(sec_rows)}")
    return {"dossiers": len(dos_rows), "sections": len(sec_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tx case dossier exporter").parse_args()
    run()
