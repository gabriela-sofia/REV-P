"""REV-P v1rq — Methodological claims audit.

Audits recent Protocol C CSV outputs for forbidden operational claims:
operational labels, training targets, operational ground truth, DINO-as-proof,
absence-as-negative, formal-negative opened, absolute paths, and forbidden
literal path-dir exposure. Also checks that bundle docs carry their mandatory
review-only disclaimers. Read-only; emits one audit row per (artifact, claim).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qu_v1qz_ground_reference_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    detect_absolute_path,
    guardrail_row,
    read_csv_safe,
    safe_relpath,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_AUDIT = _p("REVP_V1RQ_OUT_AUDIT", DATASETS / "protocol_c_methodological_claims_audit_v1rq.csv")
OUT_SUMMARY = _p("REVP_V1RQ_OUT_SUMMARY", DATASETS / "protocol_c_methodological_claims_audit_summary_v1rq.csv")
SCHEMA_AUDIT = _p("REVP_V1RQ_SCHEMA_AUDIT", SCHEMAS / "protocol_c_methodological_claims_audit_v1rq_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RQ_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_methodological_claims_audit_summary_v1rq_schema.csv")
DOC = _p("REVP_V1RQ_DOC", DOCS / "revp_v1rq_methodological_claims_audit.md")

AUDIT_FIELDS = [
    "audit_id", "artifact", "artifact_type", "claim_type", "status",
    "evidence", "severity", "review_only", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

# claim_type -> field that must never be "true"
_FORBIDDEN_TRUE_CLAIMS = {
    "OPERATIONAL_LABEL": "can_create_operational_label",
    "TRAINING_MODEL": "can_train_model",
    "TARGET_CREATED": "target_created",
    "GROUND_TRUTH_OPERATIONAL": "ground_truth_operational",
    "FORMAL_NEGATIVE": "formal_negative",
    "DINO_VALIDATES_EVENT": "dino_validates_event",
    "ABSENCE_AS_NEGATIVE": "absence_as_negative",
}

# Forbidden literal path-dir token (kept split so this source/output never embeds it)
_FORBIDDEN_LITERAL = "local" + "_runs"

# Bundle docs that must carry a mandatory disclaimer phrase
_DOC_DISCLAIMERS = {
    "revp_v1qz_ground_reference_partial_validation_bundle.md": "nao validam evento",
    "revp_v1rf_external_intake_bundle.md": "negativos formais por ausencia",
    "revp_v1rm_review_supervisor_gate_bundle.md": "permanece review-only",
}


def _audit_csv(path: Path, idx_start: int) -> list[dict[str, Any]]:
    rel = safe_relpath(path)
    rows = read_csv_safe(path)
    out: list[dict[str, Any]] = []
    idx = idx_start

    def emit(claim, violated, evidence, severity="critical"):
        nonlocal idx
        row = {
            "audit_id": f"V1RQ_AUD_{idx:05d}", "artifact": rel, "artifact_type": "CSV",
            "claim_type": claim, "status": "VIOLATION" if violated else "CLEAN",
            "evidence": evidence[:80], "severity": severity, "notes": "",
        }
        row.update(guardrail_row())
        out.append(row)
        idx += 1

    for claim, field in _FORBIDDEN_TRUE_CLAIMS.items():
        hits = sum(1 for r in rows if str(r.get(field, "false")).strip().lower() == "true")
        emit(claim, hits > 0, f"{field}=true x{hits}" if hits else "clean")

    abs_hits = sum(1 for r in rows for v in r.values() if detect_absolute_path(str(v)))
    emit("ABSOLUTE_PATH", abs_hits > 0, f"abs_path x{abs_hits}" if abs_hits else "clean")

    literal_hits = sum(1 for r in rows for v in r.values() if _FORBIDDEN_LITERAL in str(v).lower())
    emit("FORBIDDEN_LITERAL_EXPOSURE", literal_hits > 0,
         f"literal x{literal_hits}" if literal_hits else "clean")

    return out


def run(datasets: Path | None = None) -> dict[str, Any]:
    ds = datasets or DATASETS
    targets = sorted(set(ds.glob("protocol_c_*v1q*.csv")) | set(ds.glob("protocol_c_*v1r*.csv")))
    targets = [p for p in targets if "schema" not in p.name]

    rows: list[dict[str, Any]] = []
    idx = 0
    for path in targets:
        checks = _audit_csv(path, idx)
        idx += len(checks)
        rows.extend(checks)

    # Doc disclaimers
    docs_dir = DOC.parent
    for fname, phrase in _DOC_DISCLAIMERS.items():
        dpath = docs_dir / fname
        present = False
        if dpath.exists():
            present = phrase.lower() in dpath.read_text(encoding="utf-8", errors="replace").lower()
        row = {
            "audit_id": f"V1RQ_AUD_{idx:05d}", "artifact": safe_relpath(dpath),
            "artifact_type": "DOC", "claim_type": "MANDATORY_DISCLAIMER",
            "status": "CLEAN" if present else "VIOLATION",
            "evidence": f"phrase_present={present}", "severity": "high", "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
        idx += 1

    assert_clean_rows(rows, "v1rq_audit")
    write_csv_with_header(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_schema_safe(SCHEMA_AUDIT, AUDIT_FIELDS, "v1rq_audit")

    violations = sum(1 for r in rows if r["status"] == "VIOLATION")
    by_claim: dict[str, int] = {}
    for r in rows:
        if r["status"] == "VIOLATION":
            by_claim[r["claim_type"]] = by_claim.get(r["claim_type"], 0) + 1

    overall = "CLAIMS_AUDIT_CLEAN" if violations == 0 else "CLAIMS_AUDIT_VIOLATIONS_FOUND"
    summary = [
        {"stat_key": "audit_status", "stat_value": overall},
        {"stat_key": "artifacts_scanned", "stat_value": str(len(targets))},
        {"stat_key": "audit_rows", "stat_value": str(len(rows))},
        {"stat_key": "violations", "stat_value": str(violations)},
    ]
    for claim, n in sorted(by_claim.items()):
        summary.append({"stat_key": f"violation_{claim.lower()}", "stat_value": str(n)})
    summary.append({"stat_key": "stage", "stat_value": "v1rq"})
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rq_summary")

    write_doc(
        DOC,
        "v1rq — Methodological Claims Audit",
        [
            "## Objetivo",
            "Auditar CSVs recentes do Protocolo C contra claims proibidos: label operacional, "
            "target de treino, ground truth operacional, DINO-como-prova, ausencia-como-negativo, "
            "negativo formal aberto, path absoluto, exposicao de literal de diretorio.",
            "## Resultado",
            f"Status: {overall}. Artefatos: {len(targets)}. Linhas de auditoria: {len(rows)}. "
            f"Violacoes: {violations}.",
            "## Guardrails",
            "Auditoria read-only. Verifica tambem que os docs de bundle carregam o disclaimer "
            "review-only obrigatorio.",
        ],
    )
    print(f"[v1rq] status={overall} scanned={len(targets)} rows={len(rows)} violations={violations}")
    return {"status": overall, "scanned": len(targets), "rows": len(rows), "violations": violations}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rq methodological claims audit").parse_args()
    run()
