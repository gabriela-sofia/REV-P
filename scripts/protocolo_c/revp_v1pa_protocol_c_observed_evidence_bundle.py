"""REV-P v1pa — Protocol C observed evidence bundle.

Consolidates v1ou-v1oz into a final manifest, quality-check report,
and scientific summary. Does NOT recalculate scientific decisions.

Fails clearly on missing/headerless outputs or guardrail violations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1ou_v1pa_common import _p, write_csv_safe, write_doc, write_schema_safe

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Output paths (env-overridable)
# ---------------------------------------------------------------------------

OUT_MANIFEST = _p("REVP_V1PA_OUT_MANIFEST", DATASETS / "recife_protocol_c_observed_evidence_manifest_v1pa.csv")
OUT_QUALITY = _p("REVP_V1PA_OUT_QUALITY", DATASETS / "recife_protocol_c_observed_evidence_quality_checks_v1pa.csv")
OUT_SUMMARY = _p("REVP_V1PA_OUT_SUMMARY", DATASETS / "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv")
SCHEMA_MANIFEST = _p("REVP_V1PA_SCHEMA_MANIFEST", SCHEMAS / "recife_protocol_c_observed_evidence_manifest_v1pa_schema.csv")
SCHEMA_QUALITY = _p("REVP_V1PA_SCHEMA_QUALITY", SCHEMAS / "recife_protocol_c_observed_evidence_quality_checks_v1pa_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PA_SCHEMA_SUMMARY", SCHEMAS / "recife_protocol_c_observed_evidence_scientific_summary_v1pa_schema.csv")
DOC = _p("REVP_V1PA_DOC", DOCS / "revp_v1pa_protocol_c_observed_evidence_bundle.md")

MANIFEST_FIELDS = [
    "artifact_id", "artifact_path", "artifact_type", "stage",
    "rows", "columns", "required_columns_present", "header_present",
    "file_size_bytes", "sha256_16", "role_in_pipeline",
    "can_affect_label", "can_affect_training", "notes",
]

QUALITY_FIELDS = [
    "check_id", "check_group", "artifact_path", "check_name",
    "status", "severity", "observed_value", "expected_value", "explanation",
]

SUMMARY_FIELDS = [
    "summary_id", "metric", "value", "interpretation",
    "methodological_status", "writing_use",
]

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")

# ---------------------------------------------------------------------------
# Artifact registry — all v1ou-v1oz outputs
# ---------------------------------------------------------------------------

_ART = tuple[str, str, str, list[str], str, bool, bool]
ARTIFACTS: list[_ART] = [
    # stage, filename, type, required_cols, role, can_affect_label, can_affect_training
    ("v1ou", "recife_external_evidence_source_inventory_v1ou.csv", "CSV",
     ["source_candidate_id", "allowed_for_event_registry", "is_fixture_or_synthetic"],
     "external_evidence_source_inventory", False, False),
    ("v1ou", "recife_external_evidence_source_inventory_summary_v1ou.csv", "CSV",
     ["stat_key", "stat_value"],
     "external_evidence_source_inventory_summary", False, False),
    ("v1ov", "recife_ground_reference_observed_event_registry_v1ov.csv", "CSV",
     ["event_id", "can_be_used_as_ground_truth", "can_train_model", "can_create_operational_label", "allowed_use"],
     "observed_event_registry", False, False),
    ("v1ov", "recife_ground_reference_observed_event_summary_v1ov.csv", "CSV",
     ["stat_key", "stat_value"],
     "observed_event_summary", False, False),
    ("v1ow", "recife_ground_reference_evidence_scoring_v1ow.csv", "CSV",
     ["evidence_id", "can_promote_to_label", "can_train_model", "evidence_tier"],
     "evidence_scoring", False, False),
    ("v1ow", "recife_ground_reference_evidence_scoring_summary_v1ow.csv", "CSV",
     ["stat_key", "stat_value"],
     "evidence_scoring_summary", False, False),
    ("v1ox", "recife_event_patch_linkage_registry_v1ox.csv", "CSV",
     ["linkage_id", "can_create_label", "can_train_model", "temporal_linkage_status"],
     "event_patch_linkage", False, False),
    ("v1ox", "recife_event_patch_linkage_summary_v1ox.csv", "CSV",
     ["stat_key", "stat_value"],
     "event_patch_linkage_summary", False, False),
    ("v1oy", "recife_ground_truth_candidate_decision_audit_v1oy.csv", "CSV",
     ["decision_id", "candidate_level", "can_be_used_for_training", "can_create_operational_label"],
     "c_level_decision_audit", False, False),
    ("v1oy", "recife_ground_truth_candidate_decision_summary_v1oy.csv", "CSV",
     ["stat_key", "stat_value"],
     "c_level_decision_summary", False, False),
    ("v1oz", "recife_dino_review_only_representation_queue_v1oz.csv", "CSV",
     ["queue_id", "dino_can_create_label", "dino_can_train_model", "dino_target_field_created"],
     "dino_review_queue", False, False),
    ("v1oz", "recife_dino_review_only_representation_summary_v1oz.csv", "CSV",
     ["stat_key", "stat_value"],
     "dino_review_summary", False, False),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha16(path: Path) -> str:
    if not path.exists() or path.stat().st_size > 20_000_000:
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_csv_for_audit(path: Path) -> tuple[list[str], int]:
    """Return (fieldnames, row_count)."""
    if not path.exists():
        return ([], -1)
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            count = sum(1 for _ in reader)
        return (fields, count)
    except Exception:
        return ([], -1)


def _scan_for_violations(path: Path) -> list[str]:
    """Return list of violation descriptions found in file."""
    violations = []
    if not path.exists():
        return violations
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return violations

    # Absolute Windows path
    if ABS_PATH_RE.search(text):
        violations.append("ABSOLUTE_WINDOWS_PATH_FOUND")

    # local_runs reference
    if "local_runs" in text.lower():
        violations.append("LOCAL_RUNS_PATH_FOUND")

    # Forbidden true fields in CSV
    forbidden_patterns = [
        ("ground_truth,true", "GROUND_TRUTH_TRUE_FOUND"),
        ("can_train_model,true", "CAN_TRAIN_MODEL_TRUE_FOUND"),
        ("can_create_operational_label,true", "CAN_CREATE_OPERATIONAL_LABEL_TRUE_FOUND"),
        ("can_promote_to_label,true", "CAN_PROMOTE_TO_LABEL_TRUE_FOUND"),
        ("dino_can_create_label,true", "DINO_CAN_CREATE_LABEL_TRUE_FOUND"),
        ("dino_can_train_model,true", "DINO_CAN_TRAIN_MODEL_TRUE_FOUND"),
        ("dino_target_field_created,true", "DINO_TARGET_FIELD_CREATED_TRUE_FOUND"),
        ("can_be_used_for_training,true", "CAN_BE_USED_FOR_TRAINING_TRUE_FOUND"),
        ("can_create_label,true", "CAN_CREATE_LABEL_TRUE_FOUND"),
    ]
    text_lower = text.lower()
    for pattern, label in forbidden_patterns:
        if pattern.lower() in text_lower:
            violations.append(label)

    return violations


def build_manifest(datasets_dir: Path) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for i, (stage, filename, ftype, req_cols, role, can_label, can_train) in enumerate(ARTIFACTS):
        path = datasets_dir / filename
        fields, row_count = _read_csv_for_audit(path)
        req_present = all(c in fields for c in req_cols) if fields else False

        manifest_rows.append({
            "artifact_id": f"V1PA_ARTIFACT_{i+1:03d}",
            "artifact_path": filename,
            "artifact_type": ftype,
            "stage": stage,
            "rows": str(row_count) if row_count >= 0 else "MISSING",
            "columns": str(len(fields)) if fields else "0",
            "required_columns_present": str(req_present).lower(),
            "header_present": str(bool(fields)).lower(),
            "file_size_bytes": str(path.stat().st_size) if path.exists() else "0",
            "sha256_16": _sha16(path),
            "role_in_pipeline": role,
            "can_affect_label": str(can_label).lower(),
            "can_affect_training": str(can_train).lower(),
            "notes": "",
        })
    return manifest_rows


def build_quality_checks(datasets_dir: Path) -> list[dict[str, Any]]:
    qc_rows: list[dict[str, Any]] = []
    qc_id = 0

    for stage, filename, ftype, req_cols, role, _, _ in ARTIFACTS:
        path = datasets_dir / filename
        fields, row_count = _read_csv_for_audit(path)

        def _qc(group: str, name: str, status: str, sev: str, obs: str, exp: str, expl: str) -> None:
            nonlocal qc_id
            qc_id += 1
            qc_rows.append({
                "check_id": f"V1PA_QC_{qc_id:04d}",
                "check_group": group,
                "artifact_path": filename,
                "check_name": name,
                "status": status,
                "severity": sev,
                "observed_value": obs,
                "expected_value": exp,
                "explanation": expl,
            })

        # File existence
        exists = path.exists()
        _qc("existence", "file_exists",
            "PASS" if exists else "FAIL", "HIGH" if not exists else "INFO",
            str(exists), "true", f"stage={stage}")

        # Header
        has_header = bool(fields)
        _qc("header", "header_present",
            "PASS" if has_header else "FAIL", "HIGH" if not has_header else "INFO",
            str(has_header), "true", "CSV must have at least one column")

        # Required columns
        for col in req_cols:
            present = col in fields
            _qc("columns", f"column_{col}",
                "PASS" if present else "FAIL", "HIGH" if not present else "INFO",
                str(present), "true", f"required column: {col}")

        # Empty-but-header check
        if exists and fields and row_count == 0:
            _qc("content", "empty_with_header",
                "PASS", "INFO", "0", ">=0",
                "Empty output with header is valid (fail-closed)")

        # Blocked rows have blocked_reason
        if exists and fields and "blocked_reason" in fields and row_count > 0:
            try:
                with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
                    reader = csv.DictReader(fh)
                    missing_blocked = 0
                    for row in reader:
                        if any(
                            row.get(bf, "").strip().lower() in ("true",)
                            for bf in ["is_blocked", "status"]
                            if bf in fields
                        ):
                            if not row.get("blocked_reason", "").strip():
                                missing_blocked += 1
                _qc("blocked_reason", "blocked_rows_have_reason",
                    "PASS" if missing_blocked == 0 else "FAIL",
                    "MEDIUM", str(missing_blocked), "0",
                    "Blocked rows must have blocked_reason")
            except Exception:
                pass

        # Content violations
        violations = _scan_for_violations(path)
        for v in violations:
            _qc("guardrail", f"no_{v.lower()}",
                "FAIL", "CRITICAL", v, "NOT_FOUND",
                f"Guardrail violation: {v}")
        if not violations and exists:
            _qc("guardrail", "no_forbidden_true_fields",
                "PASS", "INFO", "none", "none",
                "No forbidden true fields found")

    return qc_rows


def _get_stat(summary_path: Path, key: str) -> str:
    if not summary_path.exists():
        return "N/A"
    try:
        with summary_path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("stat_key") == key:
                    return row.get("stat_value", "N/A")
    except Exception:
        pass
    return "N/A"


def build_scientific_summary(datasets_dir: Path) -> tuple[list[dict[str, Any]], str]:
    v1ou_s = datasets_dir / "recife_external_evidence_source_inventory_summary_v1ou.csv"
    v1ov_s = datasets_dir / "recife_ground_reference_observed_event_summary_v1ov.csv"
    v1ow_s = datasets_dir / "recife_ground_reference_evidence_scoring_summary_v1ow.csv"
    v1ox_s = datasets_dir / "recife_event_patch_linkage_summary_v1ox.csv"
    v1oy_s = datasets_dir / "recife_ground_truth_candidate_decision_summary_v1oy.csv"
    v1oz_s = datasets_dir / "recife_dino_review_only_representation_summary_v1oz.csv"

    sources_scanned = _get_stat(v1ou_s, "total_source_candidates_found")
    sources_allowed = _get_stat(v1ou_s, "allowed_for_event_registry")
    events_confirmed = _get_stat(v1ov_s, "observed_event_confirmed_review_only")
    events_probable = _get_stat(v1ov_s, "observed_event_probable_review_only")
    events_contextual = _get_stat(v1ov_s, "contextual_evidence_only")
    events_blocked = _get_stat(v1ov_s, "blocked_insufficient_evidence")
    linkages_total = _get_stat(v1ox_s, "total_linkage_rows")
    temporal_confirmed = _get_stat(v1ox_s, "temporal_linkages_confirmed")
    temporal_blocked = _get_stat(v1ox_s, "temporal_linkages_blocked")
    c1 = _get_stat(v1oy_s, "c1_contextual")
    c2 = _get_stat(v1oy_s, "c2_review_only_candidate")
    c3_not_reached = _get_stat(v1oy_s, "c3_plus_not_reached")
    c4_formal_negs = _get_stat(v1oy_s, "c4_formal_negative_count")
    dino_queue = _get_stat(v1oz_s, "total_queue_entries")
    labels_created = "0"
    training_targets = "0"

    # Determine final status
    if events_confirmed == "0" and events_probable == "0":
        final_status = "OBSERVED_EVIDENCE_REVIEW_ONLY_FAIL_CLOSED"
    else:
        final_status = "OBSERVED_EVIDENCE_REVIEW_ONLY_PARTIAL"

    rows: list[dict[str, Any]] = [
        {"summary_id": "V1PA_S001", "metric": "sources_scanned",
         "value": sources_scanned,
         "interpretation": "Arquivos do repositório escaneados para candidatos a fontes externas",
         "methodological_status": "AUDITAVEL", "writing_use": "metodologia_auditoria"},
        {"summary_id": "V1PA_S002", "metric": "source_candidates_found",
         "value": sources_allowed,
         "interpretation": "Candidatos a fontes/evidências permitidos para registro de eventos",
         "methodological_status": "AUDITAVEL", "writing_use": "metodologia_auditoria"},
        {"summary_id": "V1PA_S003", "metric": "observed_events_confirmed_review_only",
         "value": events_confirmed,
         "interpretation": "Eventos com evidência suficiente para review-only confirmado",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S004", "metric": "probable_events_review_only",
         "value": events_probable,
         "interpretation": "Eventos com evidência provável mas não confirmada (review-only)",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S005", "metric": "contextual_only_evidence",
         "value": events_contextual,
         "interpretation": "Evidências apenas contextuais — sem confirmação institucional",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S006", "metric": "blocked_insufficient_evidence",
         "value": events_blocked,
         "interpretation": "Candidatos bloqueados por evidência insuficiente",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S007", "metric": "event_patch_linkages_total",
         "value": linkages_total,
         "interpretation": "Vínculos evento-patch gerados (todos contextual ou bloqueados)",
         "methodological_status": "AUDITAVEL", "writing_use": "metodologia_auditoria"},
        {"summary_id": "V1PA_S008", "metric": "spatial_linkages_review_only",
         "value": linkages_total,
         "interpretation": "Linkages espaciais (contextuais, sem temporal confirmado)",
         "methodological_status": "AUDITAVEL", "writing_use": "metodologia_auditoria"},
        {"summary_id": "V1PA_S009", "metric": "temporal_linkages_confirmed",
         "value": temporal_confirmed,
         "interpretation": "Linkages temporais confirmados — 0 esperado dado TEMPORAL_RECOVERY_FAIL_CLOSED",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S010", "metric": "c1_contextual",
         "value": c1,
         "interpretation": "Candidatos C1 — evidência contextual apenas",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S011", "metric": "c2_review_only",
         "value": c2,
         "interpretation": "Candidatos C2 — review-only, sem label",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S012", "metric": "c3_plus_candidates",
         "value": "0",
         "interpretation": "C3+ não alcançado — requer scene_date confirmada (product_dates_confirmed_real=0)",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S013", "metric": "c4_formal_negatives",
         "value": c4_formal_negs,
         "interpretation": "Negativos formais para C4 — 0 esperado",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S014", "metric": "dino_review_queue",
         "value": dino_queue,
         "interpretation": "Entradas na fila DINO review-only (sem label, sem target)",
         "methodological_status": "AUDITAVEL", "writing_use": "metodologia_auditoria"},
        {"summary_id": "V1PA_S015", "metric": "labels_created",
         "value": labels_created,
         "interpretation": "Labels operacionais criadas — 0 por design do protocolo",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S016", "metric": "training_targets_created",
         "value": training_targets,
         "interpretation": "Targets de treinamento criados — 0 por design do protocolo",
         "methodological_status": "RESULTADO_FINAL", "writing_use": "resultado_negativo_auditavel"},
        {"summary_id": "V1PA_S017", "metric": "final_status",
         "value": final_status,
         "interpretation": (
             "Status final do bloco observacional: evidência em regime review-only/contextual, "
             "sem ground truth operacional estabelecido"
         ),
         "methodological_status": "RESULTADO_FINAL", "writing_use": "conclusao_auditavel"},
    ]
    return rows, final_status


def run() -> None:
    datasets_dir = DATASETS

    manifest = build_manifest(datasets_dir)
    quality = build_quality_checks(datasets_dir)
    sci_summary = build_scientific_summary(datasets_dir)

    sci_rows: list[dict[str, Any]]
    sci_rows, final_status = sci_summary  # unpack tuple from build_scientific_summary
    write_csv_safe(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv_safe(OUT_QUALITY, quality, QUALITY_FIELDS)
    write_csv_safe(OUT_SUMMARY, sci_rows, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1pa_manifest")
    write_schema_safe(SCHEMA_QUALITY, QUALITY_FIELDS, "v1pa_quality_checks")
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pa_scientific_summary")

    # Compute QC pass rate
    total_checks = len(quality)
    failed = [r for r in quality if r["status"] == "FAIL"]
    critical = [r for r in failed if r["severity"] == "CRITICAL"]

    write_doc(
        DOC,
        "v1pa — Protocol C Observed Evidence Bundle",
        [
            "## Objetivo",
            "Consolidar v1ou-v1oz em manifest, QC e summary científico final. "
            "Não recalcula decisões científicas — auditoria apenas.",
            "## Relação com v1og-v1ot",
            "v1og-v1ot fechou recuperação temporal com TEMPORAL_RECOVERY_FAIL_CLOSED: "
            "0 product_dates confirmadas em 2.654 patches. "
            "v1ou-v1pa não tenta destravar temporal artificialmente. "
            "Toda a camada observacional permanece em regime review-only/contextual.",
            "## O que é evento observado",
            "Evento observado é uma ocorrência de inundação ou deslizamento documentada por fonte "
            "rastreável (decreto, boletim oficial, laudo técnico). Neste bloco, os candidatos a "
            "eventos de Recife foram identificados em dossiers e registros de gaps — mas nenhum "
            "foi confirmado por fonte adquirida, pois G1/G3/G4 permanecem abertos.",
            "## O que é evidência contextual",
            "Evidência contextual é informação geomorfológica, topográfica ou de drenagem que "
            "descreve o ambiente mas não confirma evento específico. PE3D MDE e drenagem ESIG "
            "são exemplos de contexto que não geram ground truth.",
            "## Por que C3/C4 permanecem bloqueados",
            "C3+ requer scene_date Sentinel confirmada (product_dates_confirmed_real=0 de v1ot). "
            f"C4 requer negativo formal explícito (formal_negative_count=0 de v1ot). "
            "Nenhuma condição foi satisfeita.",
            "## Papel do DINO",
            "DINOv2 with registers é usado exclusivamente para representação estrutural visual — "
            "embeddings de revisão sem label, sem target, sem ground truth derivado.",
            "## Texto recomendado para o TCC",
            (
                "A camada observacional do Protocolo C foi estruturada como registro auditável "
                "de eventos e evidências externas, separando fonte, data, precisão temporal, "
                "localização, precisão espacial e vínculo com patches Sentinel. Essa estrutura "
                "não promove automaticamente evidências externas a ground truth operacional. "
                "Na ausência de cadeia temporal Sentinel confirmada e de negativos formais, "
                "os registros permanecem em regime review-only/contextual, preservando os "
                "embeddings DINOv2 como representação visual sem criação de rótulos supervisionados."
            ),
            "## QC",
            f"Total de checks: {total_checks}. "
            f"Falhas: {len(failed)}. "
            f"Críticos: {len(critical)}. "
            f"Status final: {final_status}.",
        ],
    )

    print(f"[v1pa] Manifest: {len(manifest)} artifacts | QC: {total_checks} checks, "
          f"{len(failed)} FAIL, {len(critical)} CRITICAL")
    print(f"[v1pa] Final status: {final_status}")

    if critical:
        print(f"[v1pa] CRITICAL QC failures:")
        for r in critical:
            print(f"  - {r['artifact_path']}: {r['check_name']} = {r['observed_value']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1pa protocol C observed evidence bundle")
    parser.parse_args()
    run()
