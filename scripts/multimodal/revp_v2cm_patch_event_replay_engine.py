"""REV-P v2cm - blockable patch-event replay engine."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cj_to_v2cm_common import (
    ALLOWED_CLAIM,
    FORBIDDEN_CLAIM,
    guardrail_rows,
    pair_path,
    read_csv,
    replay_path,
    validation_path,
    write_csv,
    write_text,
)


FIELDS = [
    "replay_id",
    "candidate_id",
    "patch_id",
    "region",
    "patch_boundary_available",
    "observed_geometry_validated",
    "crs_compatible",
    "reprojection_required",
    "reprojection_success",
    "intersection_test_executed",
    "candidate_intersection_computed",
    "intersection_area",
    "intersection_ratio_patch",
    "intersection_ratio_event",
    "replay_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

GUARD_FIELDS = ["guardrail", "expected_value", "observed_value", "status", "detail"]


def build_replay(repo_root: Path) -> list[dict[str, str]]:
    pairs = read_csv(pair_path(repo_root))
    validations = {row.get("candidate_id", ""): row for row in read_csv(validation_path(repo_root))}
    rows: list[dict[str, str]] = []
    for idx, pair in enumerate(pairs, 1):
        val = validations.get(pair.get("candidate_id", ""), {})
        patch_ok = pair.get("patch_boundary_available") == "true"
        geom_ok = val.get("validation_status") == "VALIDATED_OBSERVED_GEOMETRY_CANDIDATE"
        crs_ok = val.get("crs_known") == "true"
        prov_ok = val.get("provenance_available") == "true"
        hash_ok = val.get("hash_available") == "true"
        if not patch_ok:
            status, reason = "REPLAY_BLOCKED_NO_PATCH_BOUNDARY", "PATCH_BOUNDARY_AUSENTE"
        elif not geom_ok:
            status, reason = "REPLAY_BLOCKED_NO_OBSERVED_GEOMETRY", val.get("blocking_reason", "GEOMETRIA_OBSERVADA_NAO_VALIDADA")
        elif not crs_ok:
            status, reason = "REPLAY_BLOCKED_MISSING_CRS", "CRS_AUSENTE"
        elif not prov_ok:
            status, reason = "REPLAY_BLOCKED_MISSING_PROVENANCE", "PROVENIENCIA_AUSENTE"
        elif not hash_ok:
            status, reason = "REPLAY_BLOCKED_MISSING_HASH", "HASH_AUSENTE"
        elif val.get("validation_status") == "BLOCKED_VALIDATOR_DEPENDENCY_UNAVAILABLE":
            status, reason = "REPLAY_BLOCKED_VALIDATOR_DEPENDENCY_UNAVAILABLE", "DEPENDENCIA_GEOMETRICA_INDISPONIVEL"
        else:
            status, reason = "REPLAY_READY_NOT_EXECUTED", "REPLAY_PREPARADO_SEM_EXECUCAO_AUTOMATICA"
        executed = status == "REPLAY_EXECUTED_CANDIDATE_ONLY"
        rows.append(
            {
                "replay_id": f"REPLAY_v2cm_{idx:04d}",
                "candidate_id": pair.get("candidate_id", ""),
                "patch_id": pair.get("patch_id", ""),
                "region": pair.get("region", ""),
                "patch_boundary_available": "true" if patch_ok else "false",
                "observed_geometry_validated": "true" if geom_ok else "false",
                "crs_compatible": "true" if crs_ok and geom_ok else "false",
                "reprojection_required": "false",
                "reprojection_success": "false",
                "intersection_test_executed": "true" if executed else "false",
                "candidate_intersection_computed": "true" if executed else "false",
                "intersection_area": "",
                "intersection_ratio_patch": "",
                "intersection_ratio_event": "",
                "replay_status": status,
                "blocking_reason": reason,
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def report(rows: list[dict[str, str]]) -> str:
    executed = sum(1 for row in rows if row["intersection_test_executed"] == "true")
    blocked = sum(1 for row in rows if row["replay_status"].startswith("REPLAY_BLOCKED"))
    return f"""# REV-P v2cm - replay patch-evento bloqueavel

O replay so pode avancar com patch boundary, geometria observada candidata
validada, CRS, proveniencia e hash. Quando esses requisitos faltam, campos de area
permanecem vazios e o status fica bloqueado.

Registros de replay: {len(rows)}.
Bloqueados: {blocked}.
Intersecoes candidatas computadas: {executed}.
"""


def run(repo_root: Path, force: bool = False) -> int:
    rows = build_replay(repo_root)
    out = replay_path(repo_root)
    if out.exists() and not force:
        raise FileExistsError(out)
    write_csv(out, rows, FIELDS)
    guards = guardrail_rows([
        ("intersection_claim_allowed", "false_without_validated_replay", "false", True, "nenhuma intersecao observada afirmada"),
    ])
    write_csv(repo_root / "outputs_public/logs_summary/revp_patch_event_replay_guardrails_v2cm.csv", guards, GUARD_FIELDS)
    write_text(repo_root / "outputs_public/execution_reports/revp_patch_event_replay_report_v2cm.md", report(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run(Path(args.repo_root), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
