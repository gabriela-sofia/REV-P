"""SUSC-17A Reference Evidence Protocol (review-only, fail-closed).

Formalizes the SUSC-16D stub into an auditable protocol that classifies
observed reference evidence using already-existing project artifacts only. It
separates four logical objects:

* ``event_record``             - the observed event identity;
* ``source_footprint``         - the official/technical footprint geometry;
* ``derived_patch_link``       - the footprint->patch linkage;
* ``score_evaluation_reference`` - a patch link usable to evaluate score v6.

It NEVER changes score v6, creates score v7, creates training data, supervised
labels, models, ground truth or true negatives. Observed footprints stay
candidate evidence; documentary/geometry absence is never a true negative. Only
the three strong classes may ever support strong evaluation, and none becomes
ground truth.

No data is invented: missing fields are filled with controlled absence values
(``not_available``), and every record keeps lineage to its source artifact.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17A.md"
SCHEMA = SCHEMAS / "susc_17a_reference_evidence_protocol_schema_v1.json"

# --- inputs (read-only) ---------------------------------------------------- #
CATALOG = DAT / "susc_16a_observed_footprint_catalog_v1.csv"
LINKAGE = DAT / "susc_16a_footprint_patch_linkage_v1.csv"
UNIFIED = DAT / "susc_16c_unified_observational_analysis_table_v1.csv"
QUALITY_16B = OUT / "SUSC_16B_footprint_evidence_quality_audit.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

# --- outputs --------------------------------------------------------------- #
REGISTRY = OUT / "susc_17a_reference_evidence_registry.csv"
QUALITY_TIERS = OUT / "susc_17a_reference_evidence_quality_tiers.csv"
CLASS_POLICY = OUT / "susc_17a_reference_evidence_class_policy.json"
SUMMARY = OUT / "susc_17a_reference_evidence_summary.json"
REPORT = OUT / "SUSC_17A_REFERENCE_EVIDENCE_PROTOCOL_REPORT.md"

REQUIRED_INPUTS = [AUDIT, SCHEMA, CATALOG, LINKAGE, UNIFIED, SCORE_V6]

MANDATORY = (
    "O SUSC-17A formaliza o protocolo de evidencia observacional review-only. "
    "Ele separa event_record, source_footprint, derived_patch_link e "
    "score_evaluation_reference, trata footprint como evidencia candidata e nao "
    "como ground truth, nao transforma ausencia documental em negativo, nao "
    "altera o score v6 oficial, nao cria score v7, nao cria treino, modelo, "
    "label supervisionado ou ground truth."
)

# --- evidence class taxonomy ---------------------------------------------- #
STRONG_CLASSES = {
    "official_observed_event_polygon",
    "official_observed_event_point",
    "technical_remote_sensing_flood_footprint",
}
MODERATE_CLASSES = {"official_address_resolved"}
CONTEXT_CLASSES = {
    "street_neighborhood_context_only",
    "risk_area_not_event",
    "alert_only",
    "administrative_disaster_record",
    "documentary_context",
}
EXCLUDED_CLASSES = {"rejected_non_event", "insufficient_reference"}
ALL_CLASSES = STRONG_CLASSES | MODERATE_CLASSES | CONTEXT_CLASSES | EXCLUDED_CLASSES

# Classes that represent an actually-occurred observation (vs risk/alert/reject).
OCCURRENCE_CLASSES = STRONG_CLASSES | MODERATE_CLASSES | {"administrative_disaster_record"}

# Classes that need fine geometry to validate patch-level. administrative and
# context classes never validate patch automatically.
PATCH_VALIDATING_CLASSES = STRONG_CLASSES | MODERATE_CLASSES

QUALITY_TIER_DEFS = [
    ("TIER_A_STRONG_GEOMETRY_DATED",
     "Strong class with fine geometry, explicit event date and patch link",
     "fine_geometry", "explicit_date", "patch_linked", "true", "true", "true"),
    ("TIER_B_MODERATE_GEOMETRY_PATCH_LINKED",
     "Strong class with footprint geometry and a resolved patch link, no explicit date",
     "footprint_geometry", "not_available", "patch_linked", "true", "true", "true"),
    ("TIER_C_COARSE_GEOMETRY_UNLINKED",
     "Strong class with footprint geometry but no resolved patch link yet",
     "footprint_geometry", "not_available", "not_linked", "true", "false", "false"),
    ("TIER_D_CONTEXT_NO_PATCH_VALIDATION",
     "Context/administrative class that never validates patch-level automatically",
     "coarse_or_textual", "any", "any", "false", "false", "false"),
    ("TIER_E_INSUFFICIENT",
     "Geometry parse failed or rejected; never enters evaluation or calibration",
     "not_available", "not_available", "not_linked", "false", "false", "false"),
]

REGISTRY_FIELDS = [
    "reference_evidence_id", "source_id", "event_id", "geometry_id",
    "patch_link_id", "patch_id", "event_date", "pre_event_window",
    "post_event_window", "city", "region", "phenomenon_type", "geometry_type",
    "spatial_resolution", "temporal_resolution", "source_authority",
    "source_confidence", "annotation_method", "uncertainty_m", "link_quality",
    "evidence_class", "quality_tier", "reference_level",
    "represents_observed_occurrence", "eligible_for_evaluation",
    "eligible_for_strong_evaluation", "eligible_for_calibration",
    "eligible_for_score_v7_future", "review_only", "trainable", "ground_truth",
    "not_ground_truth_reason", "no_negative_control_reason", "source_artifact",
    "lineage_notes",
]

NOT_GT_REASON = (
    "observed_footprint_and_official_geometry_are_candidate_evidence_not_ground_truth"
)
NO_NEG_REASON = (
    "documentary_or_geometry_absence_is_not_a_true_negative_in_this_protocol"
)
CATALOG_ARTIFACT = "datasets/suscetibilidade/susc_16a_observed_footprint_catalog_v1.csv"
LINKAGE_ARTIFACT = "datasets/suscetibilidade/susc_16a_footprint_patch_linkage_v1.csv"
UNIFIED_ARTIFACT = "datasets/suscetibilidade/susc_16c_unified_observational_analysis_table_v1.csv"

CITY_BY_REGION = {
    "recife": "Recife", "curitiba": "Curitiba", "petropolis": "Petropolis",
    "unknown": "unknown",
}


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p.relative_to(ROOT)) for p in missing))


def _na(value: str | None) -> str:
    value = (value or "").strip()
    return value if value else "not_available"


def _classify(geometry_type: str) -> str:
    geometry_type = (geometry_type or "").strip().lower()
    if geometry_type == "polygon":
        return "official_observed_event_polygon"
    if geometry_type in ("point_set", "point", "multipoint"):
        return "official_observed_event_point"
    # geometry could not be resolved (empty / json_parse_failed) -> insufficient
    return "insufficient_reference"


def _spatial_resolution(evidence_class: str) -> str:
    if evidence_class == "official_observed_event_polygon":
        return "coarse_bbox_polygon"
    if evidence_class == "official_observed_event_point":
        return "point_cluster"
    return "not_available"


def _link_quality(spatial_relation: str, overlap_ratio: str) -> str:
    try:
        overlap = float(overlap_ratio)
    except (TypeError, ValueError):
        overlap = 0.0
    if "covers" in (spatial_relation or "") and overlap >= 0.99:
        return "high_spatial_overlap"
    if overlap >= 0.5:
        return "moderate_spatial_overlap"
    return "low_spatial_overlap"


def _eligibility(evidence_class: str, has_patch_link: bool) -> dict[str, str]:
    can_validate = evidence_class in PATCH_VALIDATING_CLASSES
    is_strong = evidence_class in STRONG_CLASSES and has_patch_link
    return {
        "eligible_for_evaluation": "true" if can_validate else "false",
        "eligible_for_strong_evaluation": "true" if is_strong else "false",
        "eligible_for_calibration": "true" if is_strong else "false",
        "eligible_for_score_v7_future": "false",
    }


def _quality_tier(evidence_class: str, has_patch_link: bool) -> str:
    if evidence_class in EXCLUDED_CLASSES:
        return "TIER_E_INSUFFICIENT"
    if evidence_class in CONTEXT_CLASSES:
        return "TIER_D_CONTEXT_NO_PATCH_VALIDATION"
    if evidence_class in STRONG_CLASSES and has_patch_link:
        return "TIER_B_MODERATE_GEOMETRY_PATCH_LINKED"
    return "TIER_C_COARSE_GEOMETRY_UNLINKED"


def _base_governance() -> dict[str, str]:
    return {
        "review_only": "true",
        "trainable": "false",
        "ground_truth": "false",
        "not_ground_truth_reason": NOT_GT_REASON,
        "no_negative_control_reason": NO_NEG_REASON,
    }


def _row(evidence_class: str, has_patch_link: bool, **fields) -> dict[str, str]:
    elig = _eligibility(evidence_class, has_patch_link)
    row = {
        "phenomenon_type": "flood",
        "source_authority": "local_project_official_artifact",
        "source_confidence": "moderate",
        "annotation_method": "local_geometry_parse",
        "uncertainty_m": "not_available",
        "evidence_class": evidence_class,
        "quality_tier": _quality_tier(evidence_class, has_patch_link),
        "represents_observed_occurrence": "true" if evidence_class in OCCURRENCE_CLASSES else "false",
        **elig,
        **_base_governance(),
    }
    row.update(fields)
    # Every field is coerced to a non-empty controlled string (no invented data).
    return {field: _na(str(row.get(field, ""))) for field in REGISTRY_FIELDS}


def build_registry() -> int:
    _require_inputs()
    catalog = read_csv(CATALOG)
    eligible = [r for r in catalog if r.get("eligible_for_patch_observational_evaluation") == "true"]
    catalog_by_footprint = {r["footprint_id"]: r for r in eligible}
    linkage_by_pair = {
        (r["footprint_id"], r["patch_id"]): r for r in read_csv(LINKAGE)
    }
    events = [r for r in read_csv(UNIFIED) if r.get("is_event_patch") == "true"]

    rows: list[dict[str, str]] = []
    linked_footprints: set[str] = set()

    # 1. derived_patch_link / score_evaluation_reference rows (the 65 16C links).
    for ev in events:
        footprint_id = ev["footprint_id"]
        patch_id = ev["patch_id"]
        linked_footprints.add(footprint_id)
        cat = catalog_by_footprint.get(footprint_id, {})
        link = linkage_by_pair.get((footprint_id, patch_id), {})
        geometry_type = cat.get("geometry_type", "")
        evidence_class = _classify(geometry_type)
        event_id = _na(cat.get("event_id") or link.get("event_id"))
        link_id = _na(link.get("linkage_id"))
        rows.append(_row(
            evidence_class, True,
            reference_evidence_id=f"REF17A_{link_id}",
            source_id=footprint_id,
            event_id=event_id,
            geometry_id=event_id,
            patch_link_id=link_id,
            patch_id=patch_id,
            event_date=_na(cat.get("event_date") or link.get("event_date")),
            pre_event_window="not_available",
            post_event_window="not_available",
            city=CITY_BY_REGION.get(ev.get("region", ""), _na(ev.get("region"))),
            region=_na(ev.get("region")),
            geometry_type=_na(geometry_type),
            spatial_resolution=_spatial_resolution(evidence_class),
            temporal_resolution="not_available",
            link_quality=_link_quality(link.get("spatial_relation", ""), link.get("overlap_ratio", "")),
            reference_level="score_evaluation_reference",
            source_artifact=f"{UNIFIED_ARTIFACT}|{LINKAGE_ARTIFACT}|{CATALOG_ARTIFACT}",
            lineage_notes=(
                f"patch link from SUSC-16C ({ev.get('case_id', '')}); spatial_relation="
                f"{_na(link.get('spatial_relation'))}; overlap_ratio={_na(link.get('overlap_ratio'))}; "
                "event identity inferred from local geometry candidate, no separate official event registry; "
                "no explicit event date -> temporal windows not_available"),
        ))

    # 2. source_footprint rows for eligible footprints not yet linked to patches.
    for footprint_id, cat in sorted(catalog_by_footprint.items()):
        if footprint_id in linked_footprints:
            continue
        geometry_type = cat.get("geometry_type", "")
        evidence_class = _classify(geometry_type)
        event_id = _na(cat.get("event_id"))
        region = _na(cat.get("region"))
        rows.append(_row(
            evidence_class, False,
            reference_evidence_id=f"REF17A_FP_{footprint_id}",
            source_id=footprint_id,
            event_id=event_id,
            geometry_id=event_id,
            patch_link_id="not_available",
            patch_id="not_available",
            event_date=_na(cat.get("event_date")),
            pre_event_window="not_available",
            post_event_window="not_available",
            city=CITY_BY_REGION.get(region, region),
            region=region,
            geometry_type=_na(geometry_type),
            spatial_resolution=_spatial_resolution(evidence_class),
            temporal_resolution="not_available",
            link_quality="not_available",
            reference_level="source_footprint",
            source_artifact=CATALOG_ARTIFACT,
            lineage_notes=(
                "eligible footprint from SUSC-16A catalog without a resolved patch link; "
                f"raw geometry_type={_na(geometry_type)}; not operationally evaluable until a "
                "patch link and event date exist"),
        ))

    rows.sort(key=lambda r: r["reference_evidence_id"])
    write_csv(REGISTRY, rows, REGISTRY_FIELDS)
    print(f"registry -> {REGISTRY.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def build_quality_tiers() -> int:
    rows_registry = read_csv(REGISTRY) if REGISTRY.exists() else []
    counts = Counter(r["quality_tier"] for r in rows_registry)
    rows = []
    for (tier, desc, geom, temporal, link, evalable, strong, calib) in QUALITY_TIER_DEFS:
        rows.append({
            "quality_tier": tier,
            "description": desc,
            "geometry_requirement": geom,
            "temporal_requirement": temporal,
            "patch_link_requirement": link,
            "eligible_for_evaluation": evalable,
            "eligible_for_strong_evaluation": strong,
            "eligible_for_calibration": calib,
            "record_count": counts.get(tier, 0),
            "review_only": "true",
        })
    write_csv(QUALITY_TIERS, rows)
    print(f"quality tiers -> {QUALITY_TIERS.relative_to(ROOT)}")
    return 0


def _class_policy_entry(evidence_class: str) -> dict:
    if evidence_class in STRONG_CLASSES:
        tier_class = "strong"
    elif evidence_class in MODERATE_CLASSES:
        tier_class = "moderate"
    elif evidence_class in CONTEXT_CLASSES:
        tier_class = "context"
    else:
        tier_class = "excluded"
    return {
        "evidence_class": evidence_class,
        "tier_class": tier_class,
        "can_be_strong_evaluation": evidence_class in STRONG_CLASSES,
        "can_be_moderate_evaluation": evidence_class in MODERATE_CLASSES,
        "validates_patch_level": evidence_class in PATCH_VALIDATING_CLASSES,
        "represents_observed_occurrence": evidence_class in OCCURRENCE_CLASSES,
        "eligible_for_calibration": evidence_class in STRONG_CLASSES,
        "requires_uncertainty_m": evidence_class in MODERATE_CLASSES,
        "can_be_ground_truth": False,
        "can_be_true_negative": False,
    }


def build_class_policy() -> int:
    policy = {
        "protocol": "susc_17a_reference_evidence_protocol",
        "schema": str(SCHEMA.relative_to(ROOT)).replace("\\", "/"),
        "note": (
            "Avaliacao forte somente para official_observed_event_polygon, "
            "official_observed_event_point e technical_remote_sensing_flood_footprint. "
            "official_address_resolved pode ser avaliacao moderada (nunca forte sem QA "
            "e exige uncertainty_m). Classes de contexto/administrativas nao validam "
            "patch-level. rejected_non_event e insufficient_reference nunca entram em "
            "avaliacao ou calibracao. Nenhuma classe e ground truth ou negativo verdadeiro."),
        "strong_evaluation_classes": sorted(STRONG_CLASSES),
        "moderate_evaluation_classes": sorted(MODERATE_CLASSES),
        "context_classes_no_patch_validation": sorted(CONTEXT_CLASSES),
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "documentary_absence_is_true_negative": False,
        "footprint_is_ground_truth": False,
        "official_score_v6_changed": False,
        "score_v7_created": False,
        "review_only": True,
        "classes": [_class_policy_entry(c) for c in [
            "official_observed_event_polygon",
            "official_observed_event_point",
            "technical_remote_sensing_flood_footprint",
            "official_address_resolved",
            "street_neighborhood_context_only",
            "risk_area_not_event",
            "alert_only",
            "administrative_disaster_record",
            "documentary_context",
            "rejected_non_event",
            "insufficient_reference",
        ]],
    }
    write_json(CLASS_POLICY, policy)
    print(f"class policy -> {CLASS_POLICY.relative_to(ROOT)}")
    return 0


def build_summary() -> int:
    rows = read_csv(REGISTRY)
    patch_links = [r for r in rows if r["patch_link_id"] != "not_available"]
    strong_rows = [r for r in rows if r["eligible_for_strong_evaluation"] == "true"]
    distinct_strong_sources = len({r["source_id"] for r in strong_rows})
    dated_strong = sum(1 for r in strong_rows if r["event_date"] != "not_available")
    sufficient_strong = distinct_strong_sources >= 5 and dated_strong > 0
    blocks_17b = []
    if dated_strong == 0:
        blocks_17b.append("no_event_dates_for_temporal_pre_post_windows")
    if distinct_strong_sources < 5:
        blocks_17b.append("too_few_distinct_strong_footprint_sources")
    if len({r["region"] for r in strong_rows}) < 2:
        blocks_17b.append("strong_references_concentrated_in_one_region")
    next_milestone = (
        "SUSC-17B Event-Based Mini-Benchmark Review-Only"
        if sufficient_strong else
        "SUSC-17C Sentinel-1/SAR Canary (review-only) para produzir footprints "
        "tecnicos datados; SUSC-17B permanece bloqueado ate haver janelas temporais "
        "e diversidade de footprints fortes")
    summary = {
        "total_reference_records": len(rows),
        "total_unique_events": len({r["event_id"] for r in rows if r["event_id"] != "not_available"}),
        "total_unique_footprints": len({r["source_id"] for r in rows}),
        "total_patch_links": len(patch_links),
        "count_by_evidence_class": dict(sorted(Counter(r["evidence_class"] for r in rows).items())),
        "count_by_quality_tier": dict(sorted(Counter(r["quality_tier"] for r in rows).items())),
        "count_by_reference_level": dict(sorted(Counter(r["reference_level"] for r in rows).items())),
        "eligible_for_evaluation_count": sum(1 for r in rows if r["eligible_for_evaluation"] == "true"),
        "eligible_for_strong_evaluation_count": len(strong_rows),
        "eligible_for_calibration_count": sum(1 for r in rows if r["eligible_for_calibration"] == "true"),
        "eligible_for_score_v7_future_count": 0,
        "distinct_strong_sources": distinct_strong_sources,
        "dated_strong_references": dated_strong,
        "sufficient_strong_for_benchmark": sufficient_strong,
        "blocks_17b": blocks_17b,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "score_v6_changed": False,
        "score_v7_created": False,
        "readiness_status": "REFERENCE_PROTOCOL_READY_REVIEW_ONLY",
        "next_recommended_milestone": next_milestone,
    }
    write_json(SUMMARY, summary)
    print(f"summary -> {SUMMARY.relative_to(ROOT)}")
    return 0


def build_report() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report = f"""# SUSC-17A - Reference Evidence Protocol (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `eligible_for_score_v7_future_count=0`; `readiness_status=REFERENCE_PROTOCOL_READY_REVIEW_ONLY`.

{MANDATORY}

## 1. O que o 17A resolve

O 17A transforma o stub criado no SUSC-16D em um protocolo formal e auditavel de evidencia observacional. Ele cataloga, classifica e qualifica eventos, footprints, fontes, incerteza e elegibilidade de uso, definindo o contrato que um futuro mini-benchmark (17B) poderia consumir, sem ainda criar benchmark, score v7, treino, label ou ground truth.

## 2. Por que separar event_record / source_footprint / derived_patch_link / score_evaluation_reference

Esses quatro objetos tem regimes de confianca diferentes. O `event_record` e a identidade do evento; o `source_footprint` e a geometria oficial/tecnica; o `derived_patch_link` e a ligacao footprint->patch; o `score_evaluation_reference` e o link efetivamente usavel para avaliar o score v6. Mistura-los esconderia que uma geometria existe mas nao tem link de patch, ou que um link existe mas sem data do evento. O registry mantem os quatro niveis explicitos no campo `reference_level`.

## 3. Por que footprint nao e ground truth

Os footprints sao geometrias oficiais candidatas (parse local, `moderate_candidate`), sem data explicita, sem validacao QA forte e sem confirmacao independente patch-a-patch. Eles indicam onde houve inundacao reportada, mas nao provam o rotulo de cada patch. Por isso `ground_truth=false` e `not_ground_truth_reason` em todas as linhas.

## 4. Por que ausencia documental nao e negativo

A ausencia de footprint ou de documento sobre um patch significa apenas que nao ha evidencia registrada, nao que o evento nao ocorreu. Tratar ausencia como negativo verdadeiro criaria rotulos falsos. O protocolo marca `no_negative_control_reason` em todas as linhas e nao cria nenhum controle negativo.

## 5. Classes em avaliacao forte, moderada ou apenas contexto

- Forte (valida patch-level, elegivel a calibracao): `official_observed_event_polygon`, `official_observed_event_point`, `technical_remote_sensing_flood_footprint`.
- Moderada (nunca forte automatica, exige `uncertainty_m` e QA): `official_address_resolved`.
- Apenas contexto (nao valida patch-level): `street_neighborhood_context_only`, `risk_area_not_event`, `alert_only`, `administrative_disaster_record`, `documentary_context`.
- Excluidas de avaliacao/calibracao: `rejected_non_event`, `insufficient_reference`.

## 6. Campos preenchidos como ausencia controlada e por que

- `event_date`, `pre_event_window`, `post_event_window`, `temporal_resolution`: `not_available` porque os footprints elegiveis do SUSC-16A nao tem data explicita extraida.
- `uncertainty_m`: `not_available` porque nenhuma incerteza metrica foi quantificada nas fontes locais.
- `patch_link_id`, `patch_id`, `link_quality`: `not_available` para footprints sem link de patch resolvido.
- `geometry_type`/`evidence_class` insuficiente para 6 footprints cujo geometry parse falhou (`insufficient_reference`).

Nenhum desses campos foi inventado.

## 7. O que ainda bloqueia o 17B

{json.dumps(summary.get('blocks_17b', []), ensure_ascii=False)}

Em numeros: {summary.get('eligible_for_strong_evaluation_count')} referencias fortes patch-linked, mas vindas de apenas {summary.get('distinct_strong_sources')} footprints distintos, em regiao unica, com {summary.get('dated_strong_references')} datadas. Sem janelas temporais e sem diversidade, um mini-benchmark de evento seria fragil.

## 8. Numeros do protocolo

- Registros de referencia: {summary.get('total_reference_records')}.
- Eventos unicos: {summary.get('total_unique_events')}.
- Footprints unicos: {summary.get('total_unique_footprints')}.
- Patch links: {summary.get('total_patch_links')}.
- Por classe: {json.dumps(summary.get('count_by_evidence_class', {}), ensure_ascii=False, sort_keys=True)}.
- Por tier: {json.dumps(summary.get('count_by_quality_tier', {}), ensure_ascii=False, sort_keys=True)}.
- Elegiveis a avaliacao: {summary.get('eligible_for_evaluation_count')}.
- Elegiveis a avaliacao forte: {summary.get('eligible_for_strong_evaluation_count')}.
- Elegiveis a calibracao: {summary.get('eligible_for_calibration_count')}.
- Elegiveis a score v7 futuro: {summary.get('eligible_for_score_v7_future_count')}.

## 9. Proximo marco recomendado

{summary.get('next_recommended_milestone')}
"""
    write_markdown(REPORT, report)
    print(f"report -> {REPORT.relative_to(ROOT)}")
    return 0


def run_all() -> int:
    build_registry()
    build_quality_tiers()
    build_class_policy()
    build_summary()
    build_report()
    return 0


# --------------------------------------------------------------------------- #
# Validation (fail-closed)
# --------------------------------------------------------------------------- #

def _load_schema() -> dict:
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def schema_violations(row: dict, schema: dict) -> list[str]:
    """Manual JSON-schema check (required + enum + const), dependency-free."""
    out = []
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        if not str(row.get(field, "")).strip():
            out.append(f"required field empty: {field}")
    for field, spec in props.items():
        if field not in row:
            continue
        value = row.get(field, "")
        if "const" in spec and value != spec["const"]:
            out.append(f"{field}={value} must be {spec['const']}")
        if "enum" in spec and value not in spec["enum"]:
            out.append(f"{field}={value} not in enum")
    return out


def row_governance_violations(row: dict) -> list[str]:
    """Fail-closed governance guard for a single reference record."""
    out = []
    if row.get("review_only") != "true":
        out.append("review_only not true")
    if row.get("trainable") != "false":
        out.append("trainable true")
    if row.get("ground_truth") != "false":
        out.append("ground_truth true")
    if row.get("eligible_for_score_v7_future") not in ("false", "0", ""):
        out.append("eligible_for_score_v7_future positive")
    if not str(row.get("not_ground_truth_reason", "")).strip():
        out.append("missing not_ground_truth_reason")
    if not str(row.get("no_negative_control_reason", "")).strip():
        out.append("missing no_negative_control_reason")
    if str(row.get("negative_ground_truth", "")).strip().lower() in ("true", "1"):
        out.append("negative_ground_truth present")
    if row.get("score_v7_created", "false") == "true":
        out.append("score_v7_created true")
    if row.get("official_score_changed", "false") == "true":
        out.append("official_score_changed true")
    cls = row.get("evidence_class", "")
    # strong eligibility only for strong classes
    if row.get("eligible_for_strong_evaluation") == "true" and cls not in STRONG_CLASSES:
        out.append(f"strong eligibility on non-strong class {cls}")
    # rules 10/11: risk/alert never an occurrence
    if cls in ("risk_area_not_event", "alert_only") and row.get("represents_observed_occurrence") != "false":
        out.append(f"{cls} marked as occurrence")
    # rules 3/12/13: context/administrative classes never validate patch
    if cls in CONTEXT_CLASSES and row.get("eligible_for_evaluation") == "true":
        out.append(f"context class validates patch: {cls}")
    # rule 4: excluded classes never evaluate/calibrate
    if cls in EXCLUDED_CLASSES and (
            row.get("eligible_for_evaluation") == "true"
            or row.get("eligible_for_calibration") == "true"):
        out.append(f"excluded class eligible: {cls}")
    # rule 14: official_address_resolved requires uncertainty_m and never strong
    if cls == "official_address_resolved":
        if not str(row.get("uncertainty_m", "")).strip():
            out.append("official_address_resolved without uncertainty_m")
        if row.get("eligible_for_strong_evaluation") == "true":
            out.append("official_address_resolved marked strong")
    return out


def validate() -> int:
    errors: list[str] = []
    required_files = [AUDIT, SCHEMA, REGISTRY, QUALITY_TIERS, CLASS_POLICY, SUMMARY, REPORT]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        print("FAILED: SUSC-17A validation")
        for error in errors:
            print("  ERROR:", error)
        return 1

    schema = _load_schema()
    rows = read_csv(REGISTRY)
    if not rows:
        errors.append("registry has no records")

    # unique + stable IDs
    ids = [r["reference_evidence_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("reference_evidence_id not unique")
    if ids != sorted(ids):
        errors.append("registry rows not sorted by reference_evidence_id (non-deterministic)")

    for line_no, row in enumerate(rows, start=2):
        for v in schema_violations(row, schema):
            errors.append(f"{REGISTRY.name}:{line_no}: {v}")
        for v in row_governance_violations(row):
            errors.append(f"{REGISTRY.name}:{line_no}: {v}")

    # summary cross-checks
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("eligible_for_score_v7_future_count") not in (0, "0"):
        errors.append("summary: eligible_for_score_v7_future_count > 0")
    if summary.get("review_only") is not True:
        errors.append("summary: review_only not true")
    if summary.get("trainable") is not False:
        errors.append("summary: trainable true")
    if summary.get("ground_truth") is not False:
        errors.append("summary: ground_truth true")
    if summary.get("score_v6_changed") is not False:
        errors.append("summary: score_v6_changed true")
    if summary.get("score_v7_created") is not False:
        errors.append("summary: score_v7_created true")
    if summary.get("readiness_status") != "REFERENCE_PROTOCOL_READY_REVIEW_ONLY":
        errors.append("summary: readiness_status wrong")
    if summary.get("total_reference_records") != len(rows):
        errors.append("summary: total_reference_records mismatch")
    strong_count = sum(1 for r in rows if r["eligible_for_strong_evaluation"] == "true")
    if summary.get("eligible_for_strong_evaluation_count") != strong_count:
        errors.append("summary: strong eval count mismatch")

    # class policy guards
    policy = json.loads(CLASS_POLICY.read_text(encoding="utf-8"))
    strong = {c["evidence_class"] for c in policy.get("classes", []) if c.get("can_be_strong_evaluation")}
    if strong != STRONG_CLASSES:
        errors.append("class policy: strong classes mismatch")
    if any(c.get("can_be_ground_truth") for c in policy.get("classes", [])):
        errors.append("class policy: a class is ground truth")
    if any(c.get("can_be_true_negative") for c in policy.get("classes", [])):
        errors.append("class policy: a class is true negative")
    if len(policy.get("classes", [])) != len(ALL_CLASSES):
        errors.append("class policy: class count mismatch")

    # cross guards shared with the SUSC chain
    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not SCORE_V6.exists():
        errors.append("score v6 official source missing")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")

    if errors:
        print("FAILED: SUSC-17A validation")
        for error in errors:
            print("  ERROR:", error)
        return 1
    print("PASSED: SUSC-17A validations passed")
    return 0
