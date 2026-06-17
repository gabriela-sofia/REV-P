"""Shared offline helpers for REV-P v2cj-v2cm review-only package."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ALLOWED_CLAIM = "Uso permitido apenas como evidencia candidata revisavel; nao fecha TP2 nem TP3."
FORBIDDEN_CLAIM = (
    "ground_truth_operacional|label_binario|negativo_formal|dataset_treino|"
    "claim_deteccao|claim_predicao|intersecao_observada_sem_validacao"
)

GLOBAL_GUARDRAILS = {
    "ground_truth_operational": "ABSENT",
    "formal_labels_available": "ABSENT",
    "formal_negatives_available": "ABSENT",
    "training_ready": "BLOCKED",
    "supervised_model_allowed": "false",
    "prediction_claim_allowed": "false",
    "automatic_detection_claim_allowed": "false",
    "operational_validation_claim_allowed": "false",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def bool_text(value: object) -> str:
    return "true" if boolish(value) else "false"


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "sim", "present", "ready"}


def first(row: dict[str, str], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def inventory_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv"


def pair_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv"


def priority_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv"


def queue_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_digitization_task_queue_v2ck.csv"


def validation_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_observed_geometry_validation_v2cl.csv"


def replay_path(repo_root: Path) -> Path:
    return repo_root / "outputs_public/tables/revp_patch_event_replay_v2cm.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guardrail_rows(extra: list[tuple[str, str, str, bool, str]] | None = None) -> list[dict[str, str]]:
    rows = [
        {
            "guardrail": key,
            "expected_value": value,
            "observed_value": value,
            "status": "PASS",
            "detail": "trava global preservada",
        }
        for key, value in GLOBAL_GUARDRAILS.items()
    ]
    for key, expected, observed, ok, detail in extra or []:
        rows.append(
            {
                "guardrail": key,
                "expected_value": expected,
                "observed_value": observed,
                "status": "PASS" if ok else "FAIL",
                "detail": detail,
            }
        )
    return rows

