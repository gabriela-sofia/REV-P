"""REV-P v1qt — Final local readiness bundle.

Consolidates v1qn-v1qs into a manifest, quality checks, and scientific summary
for the local DINO execution readiness block.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qn_v1qt_local_readiness_common import (
    DATASETS, DOCS, READINESS_PHRASE, SCHEMAS,
    _p, assert_no_forbidden_true, read_csv, read_csv_header,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)

OUT_MAN  = _p("REVP_V1QT_OUT_MAN",  DATASETS / "dino_local_readiness_manifest_v1qt.csv")
OUT_QC   = _p("REVP_V1QT_OUT_QC",   DATASETS / "dino_local_readiness_quality_checks_v1qt.csv")
OUT_SUM  = _p("REVP_V1QT_OUT_SUM",  DATASETS / "dino_local_readiness_scientific_summary_v1qt.csv")
SCH_MAN  = _p("REVP_V1QT_SCH_MAN",  SCHEMAS / "dino_local_readiness_manifest_v1qt_schema.csv")
SCH_QC   = _p("REVP_V1QT_SCH_QC",   SCHEMAS / "dino_local_readiness_quality_checks_v1qt_schema.csv")
SCH_SUM  = _p("REVP_V1QT_SCH_SUM",  SCHEMAS / "dino_local_readiness_scientific_summary_v1qt_schema.csv")
DOC      = _p("REVP_V1QT_DOC",       DOCS / "revp_v1qt_local_readiness_bundle.md")

MAN_FIELDS = ["artifact_id", "stage", "filename", "rows", "header_present", "role"]
QC_FIELDS  = ["check_id", "check_name", "expected", "observed", "passed", "notes"]
SUM_FIELDS = ["summary_id", "metric", "value", "interpretation",
              "methodological_status", "writing_use"]

# Optional override for the datasets directory (testing support)
_IN_DS = _p("REVP_V1QT_IN_DATASETS", DATASETS)

ARTIFACTS = [
    ("v1qn", "dino_local_root_environment_audit_v1qn.csv",    "root_env_audit"),
    ("v1qn", "dino_local_root_environment_summary_v1qn.csv",  "root_env_summary"),
    ("v1qo", "dino_smoke_asset_local_reconciliation_v1qo.csv","asset_reconciliation"),
    ("v1qo", "dino_smoke_asset_local_reconciliation_summary_v1qo.csv","asset_rec_summary"),
    ("v1qp", "dino_manifest_crosswalk_repair_suggestions_v1qp.csv","crosswalk_suggestions"),
    ("v1qp", "dino_manifest_crosswalk_repair_summary_v1qp.csv","crosswalk_summary"),
    ("v1qq", "revp_dino_local_execution_checklist_v1qq.csv",  "execution_checklist"),
    ("v1qr", "dino_local_smoke_run_readiness_gate_v1qr.csv",  "readiness_gate"),
    ("v1qr", "dino_local_smoke_run_readiness_summary_v1qr.csv","readiness_summary"),
    ("v1qs", "dino_tcc_table_local_readiness_v1qs.csv",       "tcc_readiness"),
    ("v1qs", "dino_tcc_table_local_blockers_v1qs.csv",        "tcc_blockers"),
]


def _stat(fname: str, key: str, default: str = "0") -> str:
    for r in read_csv(_IN_DS / fname):
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def _count(fname: str) -> str:
    p = _IN_DS / fname
    return str(len(read_csv(p))) if p.exists() else "MISSING"


def build_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (stage, fname, role) in enumerate(ARTIFACTS, 1):
        p = _IN_DS / fname
        rows.append({
            "artifact_id": f"V1QT_ART_{i:03d}", "stage": stage, "filename": fname,
            "rows": _count(fname),
            "header_present": str(bool(read_csv_header(p))).lower(),
            "role": role,
        })
    return rows


def build_qc() -> tuple[list[dict[str, Any]], int, int]:
    roots_exist   = int(_stat("dino_local_root_environment_summary_v1qn.csv", "roots_existing", "0") or "0")
    model_set     = _stat("dino_local_root_environment_summary_v1qn.csv", "model_path_set", "false")
    model_exists  = _stat("dino_local_root_environment_summary_v1qn.csv", "model_path_exists", "false")
    allow_dl      = _stat("dino_local_root_environment_summary_v1qn.csv", "model_allow_download", "true")
    reconciled    = int(_stat("dino_smoke_asset_local_reconciliation_summary_v1qo.csv","exact_matches","0") or "0") + \
                    int(_stat("dino_smoke_asset_local_reconciliation_summary_v1qo.csv","partial_matches","0") or "0")
    gate_status   = _stat("dino_local_smoke_run_readiness_summary_v1qr.csv", "final_status", "MISSING")
    labels        = _stat("dino_local_smoke_run_readiness_summary_v1qr.csv", "labels_created", "0")
    targets       = _stat("dino_local_smoke_run_readiness_summary_v1qr.csv", "targets_created", "0")

    checks = [
        ("roots_configured",      ">=1", str(roots_exist),   roots_exist >= 1,   True),
        ("model_path_set",        "true", model_set,         model_set == "true", True),
        ("allow_download_false",  "false", allow_dl,         allow_dl == "false", True),
        ("labels_created_zero",   "0",    labels,            labels == "0",       True),
        ("targets_created_zero",  "0",    targets,           targets == "0",      True),
        ("no_abs_paths_in_outputs","true","enforced",         True,                True),
        ("readiness_gate_computed","not_MISSING", gate_status, gate_status != "MISSING", False),
    ]
    rows: list[dict[str, Any]] = []
    passed = 0
    for i, (name, exp, obs, ok, _block) in enumerate(checks, 1):
        if ok:
            passed += 1
        rows.append({"check_id": f"V1QT_QC_{i:03d}", "check_name": name,
                     "expected": exp, "observed": obs, "passed": str(ok).lower(), "notes": ""})
    return rows, passed, len(checks)


def _final_status() -> str:
    gate = _stat("dino_local_smoke_run_readiness_summary_v1qr.csv", "final_status", "MISSING")
    model_set = _stat("dino_local_root_environment_summary_v1qn.csv", "model_path_set", "false")
    unresolved = int(_stat("dino_smoke_asset_local_reconciliation_summary_v1qo.csv","unresolved","0") or "0")
    labels = _stat("dino_local_smoke_run_readiness_summary_v1qr.csv", "labels_created", "0")
    if labels != "0":
        return "LOCAL_DINO_READINESS_GUARDRAIL_FAIL_CLOSED"
    if gate == "READY_FOR_MANUAL_REAL_SMOKE_RUN":
        return "LOCAL_DINO_READINESS_READY_FOR_MANUAL_SMOKE_RUN"
    if model_set != "true":
        return "LOCAL_DINO_READINESS_MODEL_MISSING_FAIL_CLOSED"
    if unresolved > 0:
        return "LOCAL_DINO_READINESS_ASSETS_MISSING_FAIL_CLOSED"
    return "LOCAL_DINO_READINESS_DRY_RUN_ONLY"


def build_summary(qc_passed: int, qc_total: int, final: str) -> list[dict[str, Any]]:
    roots  = _stat("dino_local_root_environment_summary_v1qn.csv", "roots_existing", "0")
    model  = _stat("dino_local_root_environment_summary_v1qn.csv", "model_path_set", "false")
    m_rdy  = _stat("dino_local_root_environment_summary_v1qn.csv", "model_path_exists", "false")
    smoke  = _stat("dino_smoke_asset_local_reconciliation_summary_v1qo.csv", "smoke_rows", "0")
    recon  = str(int(_stat("dino_smoke_asset_local_reconciliation_summary_v1qo.csv","exact_matches","0") or "0") +
                 int(_stat("dino_smoke_asset_local_reconciliation_summary_v1qo.csv","partial_matches","0") or "0"))
    unresl = _stat("dino_smoke_asset_local_reconciliation_summary_v1qo.csv", "unresolved", "0")
    gate   = _stat("dino_local_smoke_run_readiness_summary_v1qr.csv", "final_status", "MISSING")

    def s(i: int, m: str, v: str, interp: str, ms: str = "RESULTADO_FINAL",
          use: str = "resultado_negativo_auditavel") -> dict:
        return {"summary_id": f"V1QT_S{i:03d}", "metric": m, "value": v,
                "interpretation": interp, "methodological_status": ms, "writing_use": use}

    return [
        s(1,  "roots_configured",              roots,  "Roots locais configurados e existentes", "AUDITAVEL","metodologia_auditoria"),
        s(2,  "model_path_configured",         model,  "REVP_DINO_MODEL_PATH set"),
        s(3,  "model_ready",                   m_rdy,  "Modelo local acessível (config+pesos)"),
        s(4,  "smoke_sample_rows",             smoke,  "Linhas da amostra smoke"),
        s(5,  "assets_reconciled",             recon,  "Assets smoke com correspondência local"),
        s(6,  "assets_unresolved",             unresl, "Assets smoke sem correspondência local"),
        s(7,  "readiness_gate_status",         gate,   "Status do gate de prontidão v1qr"),
        s(8,  "quality_checks_passed",         f"{qc_passed}/{qc_total}", "QC passados", "AUDITAVEL","metodologia_auditoria"),
        s(9,  "labels_created",                "0",    "Rótulos criados — 0 por design"),
        s(10, "targets_created",               "0",    "Targets criados — 0 por design"),
        s(11, "ground_truth_created",          "0",    "Ground truth criado — 0 por design"),
        s(12, "final_status",                  final,  "Status final do bloco local readiness"),
    ]


def run() -> None:
    manifest = build_manifest()
    qc, qc_passed, qc_total = build_qc()
    final = _final_status()
    summary = build_summary(qc_passed, qc_total, final)

    for rows, label in ((manifest,"v1qt_manifest"), (qc,"v1qt_qc"), (summary,"v1qt_summary")):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)

    write_csv(OUT_MAN,  manifest, MAN_FIELDS)
    write_csv(OUT_QC,   qc,       QC_FIELDS)
    write_csv(OUT_SUM,  summary,  SUM_FIELDS)
    write_schema(SCH_MAN, MAN_FIELDS, "v1qt_local_readiness_manifest")
    write_schema(SCH_QC,  QC_FIELDS,  "v1qt_local_readiness_quality_checks")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qt_local_readiness_scientific_summary")
    write_doc(DOC, "v1qt — Local Readiness Bundle", [
        "## Frase metodológica",
        READINESS_PHRASE,
        "## Objetivo",
        "Consolidar v1qn-v1qs em manifest, QC e summary científico.",
        "## Status final",
        f"**{final}**. QC: {qc_passed}/{qc_total}.",
    ])
    print(f"[v1qt] final={final} qc={qc_passed}/{qc_total}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qt local readiness bundle").parse_args()
    run()
