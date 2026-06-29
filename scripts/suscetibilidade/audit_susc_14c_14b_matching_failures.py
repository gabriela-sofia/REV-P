"""SUSC-14C FASE 1 - audit SUSC-14B matching failures."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv, write_markdown  # noqa: E402

MATCHED_14B = ROOT / "datasets" / "suscetibilidade" / "susc_14b_matched_official_occurrences_v1.csv"
FEATURES_14B = ROOT / "datasets" / "suscetibilidade" / "susc_14b_official_spatial_reference_features_v1.csv"
REGISTRY_14B = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_registry_v1.csv"
DOWNLOADS_14B = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_download_manifest_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_14B_matching_failure_audit.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_14B_matching_failure_report.md"

FIELDS = ["failure_type", "rank", "key", "count", "example_occurrence_id", "review_only", "notes"]


def _street_type(name: str) -> str:
    n = g14.normalize_street(name)
    if n.startswith("rua "):
        return "rua"
    if n.startswith("avenida "):
        return "avenida"
    if n.startswith("travessa "):
        return "travessa"
    if n.startswith("estrada "):
        return "estrada"
    if n.startswith("praca "):
        return "praca"
    return ""


def _classify(row: dict, reference_names: set[str]) -> list[str]:
    out = []
    status = row.get("georeferencing_status", "")
    method = row.get("match_method", "")
    street = row.get("normalized_street", "")
    hood = row.get("normalized_neighborhood", "")
    raw = row.get("raw_address", "")
    if status in {"blocked_no_official_reference", "blocked_ambiguous_address", "official_neighborhood_only"}:
        out.append(status if status != "official_neighborhood_only" else "weak_neighborhood_only")
    if status == "blocked_ambiguous_address":
        out.append("low_similarity" if float(row.get("name_similarity") or 0) < 0.92 else "candidate_tie")
    if not street:
        out.append("missing_street_name")
    if not hood:
        out.append("missing_neighborhood")
    if any(x in g14.normalize_text(raw) for x in ("proximo", "frente", "lado", "esquina", "lote", "quadra", "bloco")):
        out.append("number/complement_noise")
    if _street_type(street) and street.replace(_street_type(street), "", 1).strip() in reference_names:
        out.append("street_type_mismatch")
    if any(abbrev in g14.normalize_text(raw).split() for abbrev in ("r", "av", "tv", "est", "pc")):
        out.append("abbreviation_mismatch")
    if method in {"no_official_reference_for_address", "no_official_layer_for_address"}:
        out.append("reference_not_processed_due_batch_cap")
    return out or ["other_review_required"]


def _top(counter: Counter, examples: dict, failure_type: str, limit: int = 25) -> list[dict]:
    rows = []
    for rank, (key, count) in enumerate(counter.most_common(limit), start=1):
        rows.append({
            "failure_type": failure_type,
            "rank": rank,
            "key": key,
            "count": count,
            "example_occurrence_id": examples.get(key, ""),
            "review_only": "true",
            "notes": "ranking de gargalo 14B para priorizacao 14C",
        })
    return rows


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Audit of SUSC-14B Matching Failures")
    print("=" * 60)
    matched = read_csv(MATCHED_14B)
    features = read_csv(FEATURES_14B)
    registry = read_csv(REGISTRY_14B)
    downloads = read_csv(DOWNLOADS_14B)
    ref_names = {r.get("normalized_name", "") for r in features if r.get("normalized_name")}
    failure_counts = Counter()
    missing_streets, ambiguous_streets, hood_failures = Counter(), Counter(), Counter()
    missing_examples, amb_examples, hood_examples = {}, {}, {}
    for row in matched:
        types = _classify(row, ref_names)
        for t in types:
            failure_counts[t] += 1
        status = row.get("georeferencing_status", "")
        street = row.get("normalized_street") or "SEM_LOGRADOURO"
        hood = row.get("normalized_neighborhood") or "SEM_BAIRRO"
        if status == "blocked_no_official_reference":
            missing_streets[street] += 1
            missing_examples.setdefault(street, row.get("occurrence_id", ""))
            hood_failures[hood] += 1
            hood_examples.setdefault(hood, row.get("occurrence_id", ""))
        elif status == "blocked_ambiguous_address":
            ambiguous_streets[street] += 1
            amb_examples.setdefault(street, row.get("occurrence_id", ""))
            hood_failures[hood] += 1
            hood_examples.setdefault(hood, row.get("occurrence_id", ""))
        elif status == "official_neighborhood_only":
            hood_failures[hood] += 1
            hood_examples.setdefault(hood, row.get("occurrence_id", ""))
    high_priority = Counter()
    hp_examples = {}
    for row in downloads:
        if row.get("download_status") == "not_attempted_batch_cap":
            ref = next((r for r in registry if r.get("reference_id") == row.get("reference_id")), {})
            key = f"{ref.get('region','')}|p{ref.get('priority','')}|{row.get('source_title','')[:90]}"
            high_priority[key] += 1
            hp_examples.setdefault(key, row.get("reference_id", ""))
    rows = []
    rows.extend(_top(missing_streets, missing_examples, "top_missing_streets"))
    rows.extend(_top(ambiguous_streets, amb_examples, "top_ambiguous_streets"))
    rows.extend(_top(hood_failures, hood_examples, "top_neighborhoods_with_many_failures"))
    rows.extend(_top(high_priority, hp_examples, "sources_not_attempted_but_high_priority"))
    rows.extend(_top(failure_counts, defaultdict(str), "failure_class_counts", 50))
    write_csv(AUDIT, rows, FIELDS)
    write_markdown(REPORT, f"""# SUSC-14C - auditoria dos gargalos herdados do SUSC-14B

Status: review-only. Sem ground truth, sem treino supervisionado e sem score v7.

- Ocorrencias auditadas: **{len(matched)}**
- Features oficiais 14B disponiveis: **{len(features)}**
- Falhas por classe: **{dict(failure_counts)}**
- Ruas sem referencia ranqueadas: **{len(missing_streets)}**
- Ruas ambiguas ranqueadas: **{len(ambiguous_streets)}**
- Bairros com falhas ranqueados: **{len(hood_failures)}**
- Fontes nao tentadas por teto de lote: **{sum(1 for r in downloads if r.get('download_status') == 'not_attempted_batch_cap')}**

A auditoria direciona o 14C para recursos oficiais ainda nao tentados e para
casos onde a normalizacao/fuzzy matching pode reduzir ambiguidade sem inventar
coordenadas ou promover bairro para patch-level.
""")
    print(f"audit rows: {len(rows)} | failures: {dict(failure_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
