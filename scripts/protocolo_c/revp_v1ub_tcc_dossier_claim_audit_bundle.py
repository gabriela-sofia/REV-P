"""REV-P v1ub — TCC dossier claim audit + bundle.

Audits the v1tx-v1ua dossier artefacts for forbidden operational claims and
review-layer labels, then bundles a manifest and scientific summary with a
final status. Review-only; fail-closed on any forbidden claim.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tx_v1ub_tcc_dossier_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, safe_relpath, scan_forbidden_claims,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_AUD = _p("REVP_V1UB_OUT_AUD", DATASETS / "protocol_c_tcc_dossier_claim_audit_v1ub.csv")
OUT_MAN = _p("REVP_V1UB_OUT_MAN", DATASETS / "protocol_c_tcc_dossier_bundle_manifest_v1ub.csv")
OUT_SUM = _p("REVP_V1UB_OUT_SUM", DATASETS / "protocol_c_tcc_dossier_bundle_scientific_summary_v1ub.csv")
SCHEMA_A = _p("REVP_V1UB_SCHEMA_A", SCHEMAS / "protocol_c_tcc_dossier_claim_audit_v1ub_schema.csv")
SCHEMA_M = _p("REVP_V1UB_SCHEMA_M", SCHEMAS / "protocol_c_tcc_dossier_bundle_manifest_v1ub_schema.csv")
SCHEMA_S = _p("REVP_V1UB_SCHEMA_S", SCHEMAS / "protocol_c_tcc_dossier_bundle_scientific_summary_v1ub_schema.csv")
DOC = _p("REVP_V1UB_DOC", DOCS / "revp_v1ub_tcc_dossier_claim_audit_bundle.md")

AUD_FIELDS = ["audit_id", "source_file", "rows_scanned", "claim_violations",
              "audit_status", "sample_hits",
              "review_only", "automated_review",
              "requires_external_observational_evidence_for_operational_claim",
              "automatic_c3_promotion", "c4_opened",
              "can_create_operational_label", "can_train_model", "target_created",
              "ground_truth_operational", "formal_negative",
              "dino_validates_event", "hydromet_validates_event",
              "hydromet_is_negative_evidence", "absence_as_negative", "notes"]
MAN_FIELDS = ["artifact_id", "stage", "artifact_path", "rows", "role",
              "review_only", "automated_review"]
SUM_FIELDS = ["metric_key", "metric_value", "review_only", "automated_review"]

SCANNED = [
    ("v1tx", "protocol_c_case_dossier_v1tx.csv", "dossier"),
    ("v1tx", "protocol_c_case_dossier_sections_v1tx.csv", "dossier_sections"),
    ("v1ty", "protocol_c_final_evidence_matrix_v1ty.csv", "matrix"),
    ("v1tz", "protocol_c_tcc_latex_table_fragments_v1tz.csv", "latex_fragments"),
    ("v1ua", "protocol_c_tcc_narrative_draft_v1ua.csv", "narrative"),
]


def run() -> dict[str, Any]:
    aud_rows: list[dict[str, Any]] = []
    man_rows: list[dict[str, Any]] = []
    total_violations = 0
    dossier_count = 0

    for stage, fname, role in SCANNED:
        data = read_csv_safe(DATASETS / fname)
        if role == "dossier":
            dossier_count = len([r for r in data
                                 if not r.get("case_id", "").startswith("FAIL_CLOSED")])
        hits: list[str] = []
        for r in data:
            for k, v in r.items():
                if k in ("review_only", "automated_review",
                         "internal_review_automated_for_review_only"):
                    continue
                hits.extend(scan_forbidden_claims(str(v)))
        viols = len(hits)
        total_violations += viols
        aud_rows.append({
            "audit_id": f"V1UB_{stage}_{role}", "source_file": fname,
            "rows_scanned": str(len(data)), "claim_violations": str(viols),
            "audit_status": "CLAIM_CLEAN" if viols == 0 else "CLAIM_VIOLATION_FAIL_CLOSED",
            "sample_hits": ";".join(sorted(set(hits))[:3]), "notes": "",
            **guardrail_row_review(),
        })
        man_rows.append({
            "artifact_id": f"V1UB_{stage.upper()}_{role.upper()}", "stage": stage,
            "artifact_path": safe_relpath(DATASETS / fname), "rows": str(len(data)),
            "role": role, "review_only": "true", "automated_review": "true",
        })

    if total_violations > 0:
        final_status = "TCC_DOSSIER_BUNDLE_CLAIM_VIOLATION_FAIL_CLOSED"
    elif dossier_count > 0:
        final_status = "TCC_DOSSIER_BUNDLE_CLAIM_SAFE_READY_FOR_TCC"
    else:
        final_status = "TCC_DOSSIER_BUNDLE_EMPTY_FAIL_CLOSED"

    for label, rws in (("v1ub_aud", aud_rows), ("v1ub_man", man_rows)):
        viol = scan_guardrails(rws, label)
        if viol:
            raise ValueError(f"Guardrail violations {label}: {viol[:3]}")

    write_csv_with_header(OUT_AUD, aud_rows, AUD_FIELDS)
    write_csv_with_header(OUT_MAN, man_rows, MAN_FIELDS)
    write_schema(SCHEMA_A, AUD_FIELDS, "v1ub_audit")
    write_schema(SCHEMA_M, MAN_FIELDS, "v1ub_manifest")

    sci_rows = [
        {"metric_key": "dossiers_total", "metric_value": str(dossier_count)},
        {"metric_key": "files_audited", "metric_value": str(len(aud_rows))},
        {"metric_key": "claim_violations", "metric_value": str(total_violations)},
        {"metric_key": "automatic_c3_promotions", "metric_value": "0"},
        {"metric_key": "c4_opened_count", "metric_value": "0"},
        {"metric_key": "labels_created", "metric_value": "0"},
        {"metric_key": "targets_created", "metric_value": "0"},
        {"metric_key": "ground_truth_operational_created", "metric_value": "0"},
        {"metric_key": "formal_negatives_created", "metric_value": "0"},
        {"metric_key": "final_status", "metric_value": final_status},
    ]
    for r in sci_rows:
        r["review_only"] = "true"
        r["automated_review"] = "true"
    write_csv_with_header(OUT_SUM, sci_rows, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1ub_scientific_summary")

    write_doc(DOC, "v1ub — TCC Dossier Claim Audit + Bundle", [
        "## Objetivo",
        "Auditar os artefatos de dossie (v1tx-v1ua) contra claims operacionais "
        "proibidos e rotulos de revisao indevidos, e consolidar manifesto e "
        "resumo cientifico com status final.",
        f"## Resultado\nDossies: {dossier_count}. Arquivos auditados: {len(aud_rows)}. "
        f"Violacoes de claim: {total_violations}. Status final: {final_status}.",
        "## Limitacao",
        "DINO/hidromet sao contexto; ausencia nao e negativo. Sem C3 automatico "
        "(= 0), C4 fechado, sem ground truth/rotulo/target/negativo formal. Fonte "
        "observacional externa exigida para afirmacao operacional.",
    ])
    print(f"[v1ub] files={len(aud_rows)} dossiers={dossier_count} "
          f"violations={total_violations} final={final_status}")
    return {"files": len(aud_rows), "dossiers": dossier_count,
            "violations": total_violations, "final_status": final_status}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ub dossier claim audit bundle").parse_args()
    run()
