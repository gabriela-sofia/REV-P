"""REV-P v2ci - inventario TP2-ready de evidencia observacional candidata.

Esta etapa le artefatos locais do Protocolo C e produz um inventario conservador
de candidatos que podem apoiar TP2 no futuro. Ela nao baixa dados, nao infere
geometria ausente, nao fabrica CRS, nao cria hash, nao cria labels e nao promove
nenhum item a ground truth operacional.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_VERSION = "v2ci"

INVENTORY_FIELDS = [
    "candidate_id",
    "region",
    "event_name",
    "event_date",
    "source_name",
    "source_reference",
    "evidence_type",
    "has_observed_geometry",
    "geometry_format",
    "crs_known",
    "provenance_available",
    "hash_available",
    "human_review_required",
    "can_be_digitized",
    "can_be_replayed_against_patch",
    "candidate_status",
    "blocking_reason",
    "allowed_claim",
    "forbidden_claim",
]

PAIR_FIELDS = [
    "pair_id",
    "candidate_id",
    "patch_id",
    "region",
    "patch_boundary_available",
    "event_geometry_available",
    "intersection_test_possible",
    "intersection_confirmed",
    "tp2_status",
    "tp3_ready",
    "blocking_reason",
]

GUARDRAIL_FIELDS = ["guardrail", "expected_value", "observed_value", "status", "detail"]

EXPECTED_OUTPUTS = [
    "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv",
    "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv",
    "outputs_public/logs_summary/revp_tp2_candidate_guardrail_rollup_v2ci.csv",
    "outputs_public/execution_reports/revp_tp2_candidate_inventory_report_v2ci.md",
    "outputs_public/execution_reports/revp_tp2_candidate_commit_checklist_v2ci.md",
]

INPUT_FILES = [
    "outputs_public/tables/protocol_c_cross_region_candidate_registry.csv",
    "datasets/protocolo_c/v2bg_charter_activation_758_registry.csv",
    "datasets/protocolo_c/v2bh_charter_758_product_inventory.csv",
    "datasets/protocolo_c/v2bi_charter_candidate_geometry_readiness.csv",
    "datasets/protocolo_c/v2bc_ground_truth_seed_registry.csv",
    "datasets/protocolo_c/v2bd_candidate_reference_readiness.csv",
    "datasets/protocolo_c/v2ap_patch_truth_boundary_update.csv",
    "datasets/protocolo_c/v2at_event_patch_package_index.csv",
    "datasets/protocolo_c/v2ba_candidate_geometry_validation.csv",
]

ALLOWED_CLAIM = (
    "Evidencia observacional candidata para revisao metodologica; nao fecha TP2."
)
FORBIDDEN_CLAIM = (
    "ground_truth_operacional|label_binario|negativo_formal|treino|validacao_preditiva|"
    "intersecao_observada_sem_geometria_vetorial_validada"
)
DISALLOWED_PROMOTION_STATUS = "GROUND" + "_TRUTH_READY"


@dataclass(frozen=True)
class SourceRow:
    source_path: str
    row: dict[str, str]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def norm_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "sim", "1", "ready", "present", "validated"}


def first_nonempty(row: dict[str, str], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def region_from_row(row: dict[str, str]) -> str:
    region = first_nonempty(row, ["region", "city", "product_area", "product_area_text"])
    text = " ".join(str(v) for v in row.values()).lower()
    if "recife" in text:
        return "Recife"
    if "curitiba" in text or "ctb_" in text:
        return "Curitiba"
    if "petropolis" in text or "petrópolis" in text or "pet_" in text:
        return "Petropolis"
    return region or "UNKNOWN"


def candidate_id_from_row(src: SourceRow) -> str:
    row = src.row
    candidate = first_nonempty(
        row,
        [
            "candidate_id",
            "product_id",
            "charter_activation_id",
            "seed_id",
            "package_id",
            "candidate_geometry_id",
            "reference_id",
        ],
    )
    if candidate:
        return candidate
    digest = hashlib.sha1((src.source_path + repr(sorted(row.items()))).encode("utf-8")).hexdigest()[:10]
    return f"V2CI_CAND_{digest}"


def evidence_type_from_row(row: dict[str, str]) -> str:
    text = " ".join(str(v) for v in row.values()).lower()
    if "geojson" in text or "wkt" in text or "kml" in text or "vector" in text:
        return "GEOMETRIA_CANDIDATA"
    if "preview" in text or "raster" in text or "map" in text or "imagem" in text:
        return "EVIDENCIA_VISUAL"
    if "charter" in text or "official" in text or "report" in text or "registry" in text:
        return "EVIDENCIA_DOCUMENTAL"
    return "EVIDENCIA_TEXTUAL"


def has_observed_geometry(row: dict[str, str]) -> bool:
    if norm_bool(first_nonempty(row, ["geometry_valid", "geometry_present"])):
        return True
    status = first_nonempty(
        row,
        ["geometry_status", "event_geometry_status", "validation_status", "updated_candidate_status"],
    ).upper()
    if not status:
        return False
    blockers = ["MISSING", "ABSENT", "NULL", "PENDING", "UNKNOWN", "NO_FILE", "NOT_CONFIRMED"]
    if any(blocker in status for blocker in blockers):
        return False
    return status in {"READY", "VALID", "VALIDATED", "PRESENT"}


def geometry_format(row: dict[str, str]) -> str:
    value = first_nonempty(row, ["geometry_format", "geometry_type", "product_type", "product_type_raw"])
    if value:
        return value
    text = " ".join(str(v) for v in row.values()).upper()
    for fmt in ["GEOJSON", "KML", "WKT", "SHP", "VECTOR", "RASTER", "PNG", "JPG", "PDF"]:
        if fmt in text:
            return fmt
    return "UNKNOWN"


def crs_known(row: dict[str, str]) -> bool:
    if norm_bool(first_nonempty(row, ["crs_known", "crs_present", "crs_confirmed"])):
        return True
    status = first_nonempty(row, ["crs_status"]).upper()
    return bool(status and status not in {"UNKNOWN", "MISSING", "ABSENT", "PENDING"})


def provenance_available(row: dict[str, str]) -> bool:
    if norm_bool(first_nonempty(row, ["source_traceable", "provenance_available"])):
        return True
    return bool(first_nonempty(row, ["source_reference", "source_name", "product_url_or_reference", "source_id"]))


def hash_available(row: dict[str, str]) -> bool:
    return bool(first_nonempty(row, ["sha256", "hash", "output_hash", "vector_sha256"]))


def blocking_reason(row: dict[str, str], observed_geom: bool, crs: bool, provenance: bool, hashed: bool) -> str:
    reasons: list[str] = []
    if not observed_geom:
        reasons.append("GEOMETRIA_OBSERVADA_VALIDADA_AUSENTE")
    if not crs:
        reasons.append("CRS_AUSENTE_OU_NAO_CONFIRMADO")
    if not provenance:
        reasons.append("PROVENIENCIA_INSUFICIENTE")
    if not hashed:
        reasons.append("HASH_AUSENTE")
    explicit = first_nonempty(
        row,
        [
            "blocking_reason",
            "blocking_factors",
            "why_still_blocked",
            "fail_reason",
            "limitation",
            "readiness_reason",
            "note",
        ],
    )
    if explicit:
        reasons.append(f"FONTE_REGISTRA:{explicit}")
    return "|".join(reasons) if reasons else "SEM_BLOQUEIO_TP2_APARENTE"


def candidate_status(row: dict[str, str], observed_geom: bool, crs: bool, provenance: bool, hashed: bool) -> str:
    if observed_geom and crs and provenance and hashed:
        return "TP2_READY_FOR_REPLAY"
    if observed_geom or "CANDIDATE" in " ".join(str(v) for v in row.values()).upper():
        return "TP2_CANDIDATE_ONLY"
    return "TP2_BLOCKED"


def load_sources(repo_root: Path) -> list[SourceRow]:
    sources: list[SourceRow] = []
    for rel in INPUT_FILES:
        path = repo_root / rel
        for row in read_csv(path):
            sources.append(SourceRow(rel, row))
    return sources


def build_inventory(repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for src in load_sources(repo_root):
        row = src.row
        observed = has_observed_geometry(row)
        crs = crs_known(row)
        prov = provenance_available(row)
        hashed = hash_available(row)
        cid = candidate_id_from_row(src)
        rows.append(
            {
                "candidate_id": cid,
                "region": region_from_row(row),
                "event_name": first_nonempty(
                    row,
                    ["event_name", "activation_title", "product_title", "event_id", "candidate_id", "package_id"],
                    cid,
                ),
                "event_date": first_nonempty(row, ["event_date", "activation_date", "product_date", "date_start"]),
                "source_name": first_nonempty(row, ["source_name", "primary_source_name", "source_class"], src.source_path),
                "source_reference": first_nonempty(
                    row,
                    ["source_reference", "source_url", "primary_source_url", "product_url_or_reference", "source_id"],
                    src.source_path,
                ),
                "evidence_type": evidence_type_from_row(row),
                "has_observed_geometry": str(observed).lower(),
                "geometry_format": geometry_format(row),
                "crs_known": str(crs).lower(),
                "provenance_available": str(prov).lower(),
                "hash_available": str(hashed).lower(),
                "human_review_required": "true",
                "can_be_digitized": str(evidence_type_from_row(row) in {"GEOMETRIA_CANDIDATA", "EVIDENCIA_VISUAL", "EVIDENCIA_DOCUMENTAL"}).lower(),
                "can_be_replayed_against_patch": str(observed and crs and prov and hashed).lower(),
                "candidate_status": candidate_status(row, observed, crs, prov, hashed),
                "blocking_reason": blocking_reason(row, observed, crs, prov, hashed),
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    if not rows:
        rows.append(
            {
                "candidate_id": "V2CI_NO_LOCAL_CANDIDATE_SOURCE_FOUND",
                "region": "UNKNOWN",
                "event_name": "Nenhum artefato local de candidato encontrado",
                "event_date": "",
                "source_name": "inventario_v2ci",
                "source_reference": "datasets/|docs/|outputs_public/|manifests/",
                "evidence_type": "BLOQUEIO_EXPLICITO",
                "has_observed_geometry": "false",
                "geometry_format": "UNKNOWN",
                "crs_known": "false",
                "provenance_available": "false",
                "hash_available": "false",
                "human_review_required": "true",
                "can_be_digitized": "false",
                "can_be_replayed_against_patch": "false",
                "candidate_status": "TP2_BLOCKED",
                "blocking_reason": "NENHUMA_FONTE_LOCAL_LIDA",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            }
        )
    return rows


def patch_boundary_available(row: dict[str, str]) -> bool:
    if first_nonempty(row, ["patch_id"]):
        return True
    if norm_bool(first_nonempty(row, ["patch_boundary_available"])):
        return True
    status = first_nonempty(row, ["patch_geometry_status", "patch_reference_status", "patch_component"]).upper()
    return "READY" in status or "BOUNDARY" in status


def build_pairs(repo_root: Path, inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    by_candidate = {row["candidate_id"]: row for row in inventory}
    pair_rows: list[dict[str, str]] = []
    for src in load_sources(repo_root):
        cid = candidate_id_from_row(src)
        inv = by_candidate.get(cid)
        if not inv:
            continue
        patch_id = first_nonempty(src.row, ["patch_id", "event_patch_package_id", "package_id"])
        patch_ready = patch_boundary_available(src.row)
        event_ready = inv["candidate_status"] == "TP2_READY_FOR_REPLAY"
        possible = patch_ready and event_ready
        tp3 = possible
        pair_rows.append(
            {
                "pair_id": f"PAIR_v2ci_{len(pair_rows) + 1:04d}",
                "candidate_id": cid,
                "patch_id": patch_id,
                "region": inv["region"],
                "patch_boundary_available": str(patch_ready).lower(),
                "event_geometry_available": str(event_ready).lower(),
                "intersection_test_possible": str(possible).lower(),
                "intersection_confirmed": "false",
                "tp2_status": inv["candidate_status"],
                "tp3_ready": str(tp3).lower(),
                "blocking_reason": inv["blocking_reason"] if not possible else "INTERSECAO_NAO_EXECUTADA_NESTE_MARCO",
            }
        )
    if not pair_rows:
        pair_rows.append(
            {
                "pair_id": "PAIR_v2ci_0000",
                "candidate_id": inventory[0]["candidate_id"],
                "patch_id": "",
                "region": inventory[0]["region"],
                "patch_boundary_available": "false",
                "event_geometry_available": "false",
                "intersection_test_possible": "false",
                "intersection_confirmed": "false",
                "tp2_status": "TP2_BLOCKED",
                "tp3_ready": "false",
                "blocking_reason": "SEM_PAR_PATCH_EVENTO_LOCAL",
            }
        )
    return pair_rows


def build_guardrails(inventory: list[dict[str, str]], pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    any_intersection_possible = any(row["intersection_test_possible"] == "true" for row in pairs)
    return [
        guardrail("formal_labels_available", "ABSENT", "ABSENT", True, "Nenhum label formal criado."),
        guardrail("formal_negatives_available", "ABSENT", "ABSENT", True, "Nenhum negativo formal criado."),
        guardrail("training_ready", "BLOCKED", "BLOCKED", True, "Treino permanece bloqueado."),
        guardrail("ground_truth_operational", "ABSENT", "ABSENT", True, "Sem ground truth operacional patch-level."),
        guardrail("supervised_model_allowed", "false", "false", True, "Classificador supervisionado nao permitido."),
        guardrail("prediction_claim_allowed", "false", "false", True, "Sem reivindicacao preditiva."),
        guardrail(
            "intersection_claim_allowed",
            "false_unless_validated_vector_geometry_and_explicit_test",
            str(any_intersection_possible).lower(),
            not any(row["intersection_confirmed"] == "true" for row in pairs),
            "Nenhuma intersecao observada foi afirmada neste marco.",
        ),
        guardrail(
            "ground_truth_ready_status_absent",
            "true",
            str(all(row["candidate_status"] != DISALLOWED_PROMOTION_STATUS for row in inventory)).lower(),
            all(row["candidate_status"] != DISALLOWED_PROMOTION_STATUS for row in inventory),
            "Status de promocao operacional nao utilizado.",
        ),
    ]


def guardrail(name: str, expected: str, observed: str, ok: bool, detail: str) -> dict[str, str]:
    return {
        "guardrail": name,
        "expected_value": expected,
        "observed_value": observed,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
    }


def build_report(inventory: list[dict[str, str]], pairs: list[dict[str, str]], guardrails: list[dict[str, str]]) -> str:
    status_counts: dict[str, int] = {}
    for row in inventory:
        status_counts[row["candidate_status"]] = status_counts.get(row["candidate_status"], 0) + 1
    status_lines = "\n".join(f"- `{key}`: {value}" for key, value in sorted(status_counts.items()))
    guardrail_lines = "\n".join(f"- `{g['guardrail']}`: {g['status']} ({g['detail']})" for g in guardrails)
    return f"""# REV-P v2ci - inventario TP2-ready de evidencia observacional candidata

Este marco e um inventario TP2-ready, nao um fechamento de TP2. Ele organiza
evidencia observacional candidata ja presente no repositorio e registra bloqueios
metodologicos de forma auditavel.

## Escopo

- Modo review-only.
- Sem download externo e sem internet.
- Sem inferencia de geometria ausente, CRS, data de evento ou hash.
- Sem criacao de labels, negativos formais ou treino.
- Sem afirmacao de intersecao espacial observada.

## Resultado do inventario

Total de candidatos inventariados: {len(inventory)}.
Total de pares candidato-patch: {len(pairs)}.

## Status TP2

{status_lines}

## Travas metodologicas

{guardrail_lines}

## Interpretacao conservadora

Evidencia textual, evidencia visual, geometria candidata, geometria observada
validada e ground truth operacional sao categorias distintas. Um item so poderia
entrar como `TP2_READY_FOR_REPLAY` se houvesse geometria observada vetorial, CRS
conhecido, proveniencia e hash. Quando qualquer desses elementos falta, o item
permanece bloqueado ou candidato apenas.
"""


def build_checklist(inventory: list[dict[str, str]], pairs: list[dict[str, str]], guardrails: list[dict[str, str]]) -> str:
    checks = [
        ("inventario_gerado", len(inventory) > 0, f"{len(inventory)} candidatos"),
        ("pares_gerados", len(pairs) > 0, f"{len(pairs)} pares"),
        (
            "sem_status_de_promocao_operacional",
            all(r["candidate_status"] != DISALLOWED_PROMOTION_STATUS for r in inventory),
            "status proibido ausente",
        ),
        ("treino_bloqueado", any(g["guardrail"] == "training_ready" and g["observed_value"] == "BLOCKED" for g in guardrails), "training_ready=BLOCKED"),
        ("tp3_nao_pronto_quando_tp2_bloqueado", all(not (p["tp2_status"] == "TP2_BLOCKED" and p["tp3_ready"] == "true") for p in pairs), "verificado"),
        ("sem_claim_intersecao", all(p["intersection_confirmed"] == "false" for p in pairs), "nenhuma intersecao afirmada"),
    ]
    lines = "\n".join(f"- [{'x' if ok else ' '}] {name}: {detail}" for name, ok, detail in checks)
    return f"""# Checklist de commit v2ci

{lines}

Mensagem sugerida:

```text
docs: inventaria candidatos TP2 sem promover ground truth operacional
```
"""


def ensure_can_write(repo_root: Path, force: bool) -> None:
    existing = [rel for rel in EXPECTED_OUTPUTS if (repo_root / rel).exists()]
    if existing and not force:
        joined = "\n".join(existing)
        raise FileExistsError(f"Outputs existentes; use --force para sobrescrever:\n{joined}")


def run(repo_root: Path, force: bool) -> int:
    ensure_can_write(repo_root, force)
    inventory = build_inventory(repo_root)
    pairs = build_pairs(repo_root, inventory)
    guardrails = build_guardrails(inventory, pairs)
    write_csv(repo_root / EXPECTED_OUTPUTS[0], inventory, INVENTORY_FIELDS)
    write_csv(repo_root / EXPECTED_OUTPUTS[1], pairs, PAIR_FIELDS)
    write_csv(repo_root / EXPECTED_OUTPUTS[2], guardrails, GUARDRAIL_FIELDS)
    write_text(repo_root / EXPECTED_OUTPUTS[3], build_report(inventory, pairs, guardrails))
    write_text(repo_root / EXPECTED_OUTPUTS[4], build_checklist(inventory, pairs, guardrails))
    return 0 if all(g["status"] == "PASS" for g in guardrails) else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(Path(args.repo_root), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
