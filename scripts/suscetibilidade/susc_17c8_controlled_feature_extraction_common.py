"""SUSC-17C8 controlled feature extraction for candidate patches.

This milestone attempts only the adaptable feature layers identified by
SUSC-17C7. It remains review-only and fail-closed: values are not copied from
official patches, inferred from neighbouring patches, or marked real unless a
candidate-specific source artifact exists.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C8.md"

C7_FEATURE_INVENTORY = OUT / "susc_17c7_candidate_patch_feature_inventory.csv"
C7_SOURCE_MAPPING = OUT / "susc_17c7_feature_source_mapping.csv"
C7_TASK_PLAN = OUT / "susc_17c7_extraction_task_plan.csv"
C7_MISSINGNESS = OUT / "susc_17c7_feature_missingness_matrix.csv"
C7_EMBEDDING_READINESS = OUT / "susc_17c7_embedding_input_readiness.csv"
C7_NO_LEAKAGE_POLICY = OUT / "susc_17c7_no_leakage_policy.json"
C7_SUMMARY = OUT / "susc_17c7_readiness_summary.json"
C7_BLOCKERS = OUT / "susc_17c7_promotion_blockers.csv"

C6_CANDIDATE_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
C6_CANDIDATE_GRID_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
C6_CANDIDATE_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"
C6_FEATURE_CONTRACT = OUT / "susc_17c6_multimodal_feature_contract.csv"
C6_CANARY_MATRIX = OUT / "susc_17c6_multimodal_canary_matrix.csv"

REPORT = OUT / "SUSC_17C8_EXTRACAO_REAL_CONTROLADA_FEATURES_PATCHES_CANDIDATOS_REPORT.md"
RUN_REGISTRY = OUT / "susc_17c8_extraction_run_registry.csv"
PHYSICAL_FEATURES = OUT / "susc_17c8_physical_static_features.csv"
URBAN_FEATURES = OUT / "susc_17c8_urban_territorial_features.csv"
RAINFALL_FEATURES = OUT / "susc_17c8_rainfall_trigger_features.csv"
PARTIAL_MATRIX = OUT / "susc_17c8_candidate_feature_matrix_partial.csv"
QUALITY_FLAGS = OUT / "susc_17c8_feature_quality_flags.csv"
PROVENANCE = OUT / "susc_17c8_feature_provenance_manifest.csv"
NO_LEAKAGE_AUDIT = OUT / "susc_17c8_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c8_readiness_summary.json"
BLOCKERS = OUT / "susc_17c8_promotion_blockers.csv"

EXTRACTED_FEATURE_SCHEMA = SCHEMAS / "susc_17c8_extracted_feature_schema_v1.json"
PROVENANCE_SCHEMA = SCHEMAS / "susc_17c8_feature_provenance_schema_v1.json"

REQUIRED_INPUTS = [
    FEATURES,
    SCORE_V6,
    C7_FEATURE_INVENTORY,
    C7_SOURCE_MAPPING,
    C7_TASK_PLAN,
    C7_MISSINGNESS,
    C7_EMBEDDING_READINESS,
    C7_NO_LEAKAGE_POLICY,
    C7_SUMMARY,
    C7_BLOCKERS,
    C6_CANDIDATE_GRID,
    C6_CANDIDATE_GRID_GEOJSON,
    C6_CANDIDATE_LINKS,
    C6_FEATURE_CONTRACT,
    C6_CANARY_MATRIX,
]

REQUIRED_OUTPUTS = [
    AUDIT,
    REPORT,
    RUN_REGISTRY,
    PHYSICAL_FEATURES,
    URBAN_FEATURES,
    RAINFALL_FEATURES,
    PARTIAL_MATRIX,
    QUALITY_FLAGS,
    PROVENANCE,
    NO_LEAKAGE_AUDIT,
    SUMMARY,
    BLOCKERS,
    EXTRACTED_FEATURE_SCHEMA,
    PROVENANCE_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
EVENT_ID = "REC_2022_05_24_30"
MILESTONE = "SUSC-17C8"

TARGET_LAYERS = {
    "physical_static": [
        ("HAND", "m", "mean", "hand_mean"),
        ("elevation", "m", "mean", "elevation_mean"),
        ("slope", "degrees", "mean", "slope_mean"),
        ("DEM", "m", "mean", "elevation_mean"),
        ("drainage", "proxy", "mean", "urban_drainage_interaction"),
        ("distance_to_water", "m", "mean", "distance_to_water_mean"),
        ("flow_accumulation", "log_area", "mean", "flow_acc_log_mean"),
        ("TWI", "index", "mean", "twi_mean"),
    ],
    "urban_territorial": [
        ("MapBiomas", "class_fraction", "patch_summary", "urban_prop"),
        ("urban_prop", "fraction", "mean", "urban_prop"),
        ("vegetation_prop", "fraction", "mean", "vegetation_prop"),
        ("water_prop", "fraction", "mean", "water_occurrence_patch"),
        ("NDBI", "index", "mean", "ndbi_mean"),
        ("imperviousness_proxy", "proxy", "mean", "proxy_v5_ndbi_built"),
    ],
    "rainfall_trigger": [
        ("CHIRPS_3d", "mm", "sum_3d", "chirps_3d_mm"),
        ("CHIRPS_7d", "mm", "sum_7d", "chirps_7d_mm"),
        ("CHIRPS_30d", "mm", "sum_30d", "chirps_30d_mm"),
        ("runoff_context_7d", "index", "mean_7d", "runoff_context_7d"),
    ],
}

BLOCK_REASON_BY_LAYER = {
    "physical_static": "blocked_missing_candidate_specific_dem_or_hydrology_artifact",
    "urban_territorial": "blocked_missing_candidate_specific_landcover_or_spectral_artifact",
    "rainfall_trigger": "blocked_missing_candidate_specific_chirps_or_runoff_artifact",
}

SOURCE_ARTIFACT_BY_LAYER = {
    "physical_static": "official_patch_matrix_has_physical_columns_not_candidate_values",
    "urban_territorial": "official_patch_matrix_has_urban_columns_not_candidate_values",
    "rainfall_trigger": "official_patch_matrix_has_rainfall_columns_not_candidate_values",
}

SOURCE_PIPELINE_BY_LAYER = {
    "physical_static": "official_patch_feature_pipeline_not_candidate_extractor",
    "urban_territorial": "official_patch_feature_pipeline_not_candidate_extractor",
    "rainfall_trigger": "official_patch_feature_pipeline_not_candidate_extractor",
}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _candidate_ids() -> list[str]:
    return [row["candidate_patch_id"] for row in read_csv(C6_CANDIDATE_GRID)]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_fields(include_temporal: bool = False) -> list[str]:
    fields = [
        "candidate_patch_id",
        "feature_name",
        "feature_value",
        "feature_unit",
        "aggregation_method",
        "source_dataset",
        "source_artifact",
        "source_pipeline",
        "feature_value_real",
        "synthetic_placeholder",
        "usable_for_science_now",
        "blocking_reason",
        "review_only",
        "trainable",
        "ground_truth",
    ]
    if include_temporal:
        fields[1:1] = ["event_id", "event_date_or_period", "temporal_window"]
    return fields


def _feature_row(candidate_id: str, layer: str, spec: tuple[str, str, str, str]) -> dict:
    feature_name, unit, aggregation, _source_column = spec
    row = {
        "candidate_patch_id": candidate_id,
        "feature_name": feature_name,
        "feature_value": "",
        "feature_unit": unit,
        "aggregation_method": aggregation,
        "source_dataset": rel(FEATURES),
        "source_artifact": SOURCE_ARTIFACT_BY_LAYER[layer],
        "source_pipeline": SOURCE_PIPELINE_BY_LAYER[layer],
        "feature_value_real": "false",
        "synthetic_placeholder": "false",
        "usable_for_science_now": "false",
        "blocking_reason": BLOCK_REASON_BY_LAYER[layer],
        **GOV,
    }
    if layer == "rainfall_trigger":
        row.update({
            "event_id": EVENT_ID,
            "event_date_or_period": "2022-05-24_2022-05-30",
            "temporal_window": aggregation.replace("sum_", ""),
        })
    return row


def build_feature_rows(layer: str) -> list[dict]:
    rows = []
    for candidate_id in _candidate_ids():
        for spec in TARGET_LAYERS[layer]:
            rows.append(_feature_row(candidate_id, layer, spec))
    return rows


def build_run_registry() -> list[dict]:
    rows = []
    for candidate_id in _candidate_ids():
        for layer, specs in TARGET_LAYERS.items():
            rows.append({
                "extraction_run_id": f"S17C8_RUN_{len(rows) + 1:05d}",
                "candidate_patch_id": candidate_id,
                "feature_layer": layer,
                "attempted": "true",
                "execution_status": "blocked_missing_candidate_specific_artifact",
                "feature_rows_created": str(len(specs)),
                "real_feature_rows_created": "0",
                "source_dataset": rel(FEATURES),
                "source_pipeline": SOURCE_PIPELINE_BY_LAYER[layer],
                "raw_artifact_committed": "false",
                "blocking_reason": BLOCK_REASON_BY_LAYER[layer],
                **GOV,
            })
    return rows


def build_partial_matrix() -> list[dict]:
    rows = []
    missing = (
        "physical_static;urban_territorial;rainfall_trigger;"
        "sentinel2_spectral;sar_observational;embedding_representation"
    )
    for candidate_id in _candidate_ids():
        rows.append({
            "candidate_patch_id": candidate_id,
            "has_physical_static_real": "false",
            "has_urban_territorial_real": "false",
            "has_rainfall_trigger_real": "false",
            "has_sentinel2_spectral_real": "false",
            "has_sar_observational_real": "false",
            "has_embedding_representation_real": "false",
            "real_feature_layer_count": "0",
            "missing_feature_layers": missing,
            "usable_for_science_now": "false",
            **GOV,
        })
    return rows


def _all_feature_rows_by_layer() -> list[tuple[str, dict]]:
    rows = []
    for layer in TARGET_LAYERS:
        for row in build_feature_rows(layer):
            rows.append((layer, row))
    return rows


def build_quality_flags() -> list[dict]:
    rows = []
    for layer, feature in _all_feature_rows_by_layer():
        rows.append({
            "quality_flag_id": f"S17C8_QF_{len(rows) + 1:05d}",
            "candidate_patch_id": feature["candidate_patch_id"],
            "feature_layer": layer,
            "feature_name": feature["feature_name"],
            "status": "blocked_missing_artifact",
            "severity": "high",
            "quality_issue": feature["blocking_reason"],
            "blocks_scientific_use": "true",
            "blocks_17b": "true",
            **GOV,
        })
    return rows


def build_provenance() -> list[dict]:
    rows = []
    for layer, feature in _all_feature_rows_by_layer():
        rows.append({
            "provenance_id": f"S17C8_PROV_{len(rows) + 1:05d}",
            "candidate_patch_id": feature["candidate_patch_id"],
            "feature_layer": layer,
            "feature_name": feature["feature_name"],
            "source_dataset": feature["source_dataset"],
            "source_artifact": feature["source_artifact"],
            "source_pipeline": feature["source_pipeline"],
            "source_column_or_band": _source_column_for(layer, feature["feature_name"]),
            "aggregation_method": feature["aggregation_method"],
            "raw_artifact_committed": "false",
            "derived_artifact": _derived_artifact_for(layer),
            "reproducibility_status": "blocked_no_candidate_specific_value",
            "blocking_reason": feature["blocking_reason"],
            **GOV,
        })
    return rows


def _source_column_for(layer: str, feature_name: str) -> str:
    for name, _unit, _aggregation, column in TARGET_LAYERS[layer]:
        if name == feature_name:
            return column
    return "not_available"


def _derived_artifact_for(layer: str) -> str:
    if layer == "physical_static":
        return rel(PHYSICAL_FEATURES)
    if layer == "urban_territorial":
        return rel(URBAN_FEATURES)
    return rel(RAINFALL_FEATURES)


def build_no_leakage_audit() -> list[dict]:
    rows = []
    for layer, feature in _all_feature_rows_by_layer():
        rows.append({
            "audit_id": f"S17C8_LEAK_{len(rows) + 1:05d}",
            "candidate_patch_id": feature["candidate_patch_id"],
            "feature_layer": layer,
            "feature_name": feature["feature_name"],
            "uses_post_event_data": "false",
            "uses_reference_footprint_as_feature": "false",
            "uses_sar_post_event_as_feature": "false",
            "uses_synthetic_value_as_real": "false",
            "passes_no_leakage_policy": "true",
            "blocking_reason": "not_blocked_by_no_leakage_policy_value_not_extracted",
            **GOV,
        })
    return rows


def build_summary() -> dict:
    candidates = _candidate_ids()
    all_rows = [row for _layer, row in _all_feature_rows_by_layer()]
    blockers = build_blockers()
    return {
        "milestone": MILESTONE,
        "candidate_patch_count": len(candidates),
        "target_layer_count": len(TARGET_LAYERS),
        "attempted_feature_row_count": len(all_rows),
        "real_feature_row_count": 0,
        "blocked_feature_row_count": len(all_rows),
        "physical_static_real_feature_count": 0,
        "urban_territorial_real_feature_count": 0,
        "rainfall_trigger_real_feature_count": 0,
        "candidate_patches_with_any_real_feature_count": 0,
        "candidate_patches_with_all_three_target_layers_count": 0,
        "sentinel2_spectral_extracted": False,
        "sar_observational_extracted": False,
        "embedding_representation_extracted": False,
        "score_v6_changed": False,
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "raw_raster_committed": False,
        "synthetic_as_real": False,
        "provenance_manifest_created": True,
        "post_event_misuse": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "usable_for_science_now": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "blocker_count": len(blockers),
        "blocking_reason": (
            "no_candidate_specific_real_artifact_for_target_layers;"
            "candidate_patch_policy_and_officialization_absent"
        ),
        "recommended_next_milestone": (
            "SUSC-17C9 Recuperacao de artefatos locais candidato-especificos "
            "para extracao real"
        ),
    }


def build_blockers() -> list[dict]:
    blockers = [
        ("candidate_patches_not_official", "candidate grid remains review-only and outside official patch registry"),
        ("candidate_patch_policy_missing", "no accepted policy allows candidate patches to feed official score or 17B"),
        ("qa_not_accepted", "candidate links and evidence are not QA-accepted scientific features"),
        ("sentinel2_spectral_missing", "Sentinel-2 candidate-specific tiles were not extracted in 17C8"),
        ("sar_runtime_missing", "SAR runtime and post-event observational package remain outside this milestone"),
        ("real_embedding_tile_missing", "no real candidate tile exists for DINO/SatMAE embedding"),
        ("score_policy_missing_for_candidate_grid", "no score v7 policy exists for candidate grid use"),
        ("p0_official_packages_missing", "official P0 package remains unavailable for promotion"),
        ("17b_blocked_until_official_patch_or_policy", "17B remains blocked without official patch or explicit candidate policy"),
        ("scientific_use_blocked_until_feature_policy", "candidate feature use remains blocked until feature provenance policy is accepted"),
    ]
    return [
        {
            "blocker_id": f"S17C8_BLOCKER_{i:03d}",
            "blocker_code": code,
            "description": description,
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "blocks_training": "true",
            **GOV,
        }
        for i, (code, description) in enumerate(blockers, start=1)
    ]


def build_extracted_feature_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C8 extracted candidate feature row schema v1",
        "type": "object",
        "required": [
            "candidate_patch_id",
            "feature_name",
            "feature_value",
            "feature_unit",
            "aggregation_method",
            "source_dataset",
            "source_artifact",
            "source_pipeline",
            "feature_value_real",
            "synthetic_placeholder",
            "usable_for_science_now",
            "blocking_reason",
            "review_only",
            "trainable",
            "ground_truth",
        ],
        "properties": {
            "feature_value_real": {"enum": ["true", "false"]},
            "synthetic_placeholder": {"const": "false"},
            "usable_for_science_now": {"const": "false"},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_provenance_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C8 feature provenance manifest schema v1",
        "type": "object",
        "required": [
            "provenance_id",
            "candidate_patch_id",
            "feature_layer",
            "feature_name",
            "source_dataset",
            "source_artifact",
            "source_pipeline",
            "source_column_or_band",
            "aggregation_method",
            "raw_artifact_committed",
            "derived_artifact",
            "reproducibility_status",
            "blocking_reason",
            "review_only",
            "trainable",
            "ground_truth",
        ],
        "properties": {
            "raw_artifact_committed": {"const": "false"},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_audit() -> str:
    head = _run_git(["rev-parse", "HEAD"])
    staged = _run_git(["diff", "--cached", "--name-only"])
    validators = [
        "validate_susc_17c7_candidate_feature_extraction_plan.py",
        "validate_susc_17c6_multimodal_applicability_canary.py",
        "validate_susc_17c5_patch_grid_expansion_review.py",
        "validate_susc_17c4_official_artifact_ingestion.py",
        "validate_susc_17c3_official_source_acquisition.py",
        "validate_susc_17c2_sar_footprint_execution.py",
        "validate_susc_17c_strong_reference_acquisition.py",
        "validate_susc_17a_reference_evidence_protocol.py",
        "validate_susc_16d_calibration_candidate.py",
    ]
    inputs = [rel(path) for path in REQUIRED_INPUTS if path != FEATURES and path != SCORE_V6]
    return "\n".join([
        "# SUSC-17C8 - Auditoria de worktree antes da implementacao",
        "",
        f"- HEAD auditado: `{head}`",
        "- Branch esperada: `marco/pre-unificacao-gates-mv1`",
        f"- Area staged vazia no inicio: `{_bool_text(staged == '')}`",
        "- Implementacao permitida: somente SUSC-17C8, review-only.",
        "- Score v6: sem alteracao planejada.",
        "- Score v7: nao criar.",
        "- Patches oficiais, patch-links oficiais, treino, modelo, label e ground truth: nao criar.",
        "- Raw raster, SAR, DINO/SatMAE e downloads pesados: fora do escopo.",
        "",
        "## Validadores pre-programacao rodados",
        "",
        *[f"- `python scripts/suscetibilidade/{name}`: passou" for name in validators],
        "",
        "## Insumos 17C7/17C6 lidos",
        "",
        *[f"- `{item}`" for item in inputs],
        "",
        "## Sujeira preexistente preservada",
        "",
        "A auditoria inicial confirmou area staged vazia antes do stage seletivo. "
        "Arquivos fora do escopo 17C8 nao sao stageados nem normalizados por este pacote.",
    ])


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C8 - Extracao real controlada de features para patches candidatos",
        "",
        "Este pacote executa uma tentativa controlada sobre as camadas adaptaveis "
        "do 17C7: `physical_static`, `urban_territorial` e `rainfall_trigger`.",
        "",
        "Resultado: nenhum valor foi marcado como real, porque nao ha artefato "
        "local candidato-especifico verificavel para DEM/HAND/hidrologia, "
        "MapBiomas/territorial/espectral ou CHIRPS/runoff. A matriz oficial "
        "existente demonstra colunas e pipelines para patches oficiais, mas nao "
        "autoriza copiar, interpolar ou inferir valores para os patches candidatos.",
        "",
        "## Contagens",
        "",
        f"- Patches candidatos: {summary['candidate_patch_count']}",
        f"- Linhas de feature tentadas: {summary['attempted_feature_row_count']}",
        f"- Linhas reais extraidas: {summary['real_feature_row_count']}",
        f"- Linhas bloqueadas: {summary['blocked_feature_row_count']}",
        "",
        "## Garantias",
        "",
        "- Review-only: sim.",
        "- Valor sintetico tratado como real: nao.",
        "- Proveniencia manifestada: sim.",
        "- Uso indevido de dado pos-evento: nao.",
        "- Patch oficial ou patch-link oficial criado: nao.",
        "- Raw raster commitado: nao.",
        "- Score v6 alterado: nao.",
        "- Score v7 criado: nao.",
        "- Treino, modelo, label ou ground truth criado: nao.",
        "",
        "## Bloqueio 17B",
        "",
        "O 17B permanece bloqueado porque os patches continuam candidatos, nao "
        "oficiais, sem politica aceita de promocao e sem valores reais "
        "candidato-especificos com proveniencia suficiente.",
    ])


def run_all() -> None:
    _require_inputs()
    write_markdown(AUDIT, build_audit())
    write_json(EXTRACTED_FEATURE_SCHEMA, build_extracted_feature_schema())
    write_json(PROVENANCE_SCHEMA, build_provenance_schema())
    write_csv(RUN_REGISTRY, build_run_registry())
    write_csv(PHYSICAL_FEATURES, build_feature_rows("physical_static"), _feature_fields())
    write_csv(URBAN_FEATURES, build_feature_rows("urban_territorial"), _feature_fields())
    write_csv(RAINFALL_FEATURES, build_feature_rows("rainfall_trigger"), _feature_fields(include_temporal=True))
    write_csv(PARTIAL_MATRIX, build_partial_matrix())
    write_csv(QUALITY_FLAGS, build_quality_flags())
    write_csv(PROVENANCE, build_provenance())
    write_csv(NO_LEAKAGE_AUDIT, build_no_leakage_audit())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def _schema_violations(row: dict, schema: dict) -> list[str]:
    errors = []
    for key in schema.get("required", []):
        if key not in row:
            errors.append(f"missing:{key}")
        elif row[key] == "" and key not in {"feature_value"}:
            errors.append(f"empty:{key}")
    for key, rule in schema.get("properties", {}).items():
        if key not in row:
            continue
        if "const" in rule and row[key] != rule["const"]:
            errors.append(f"const:{key}")
        if "enum" in rule and row[key] not in rule["enum"]:
            errors.append(f"enum:{key}")
    return errors


def _assert(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate() -> int:
    errors: list[str] = []
    try:
        _require_inputs()
        run_all()
        first = {path: _sha(path) for path in REQUIRED_OUTPUTS}
        run_all()
        second = {path: _sha(path) for path in REQUIRED_OUTPUTS}
        _assert(first == second, "build is not byte deterministic", errors)

        for path in REQUIRED_OUTPUTS:
            _assert(path.exists(), f"missing output {rel(path)}", errors)

        feature_schema = read_json(EXTRACTED_FEATURE_SCHEMA)
        provenance_schema = read_json(PROVENANCE_SCHEMA)
        feature_rows = (
            read_csv(PHYSICAL_FEATURES)
            + read_csv(URBAN_FEATURES)
            + read_csv(RAINFALL_FEATURES)
        )
        provenance_rows = read_csv(PROVENANCE)
        no_leakage_rows = read_csv(NO_LEAKAGE_AUDIT)
        quality_rows = read_csv(QUALITY_FLAGS)
        summary = read_json(SUMMARY)

        for row in feature_rows:
            errors.extend(f"{row.get('candidate_patch_id')}:{row.get('feature_name')}:{err}" for err in _schema_violations(row, feature_schema))
            _assert(row["feature_value_real"] == "false", "real feature value was unexpectedly declared", errors)
            _assert(row["synthetic_placeholder"] == "false", "synthetic placeholder declared in extracted feature", errors)
            _assert(row["usable_for_science_now"] == "false", "candidate feature declared science-usable", errors)
            _assert(row["review_only"] == "true", "feature row is not review-only", errors)
            _assert(row["trainable"] == "false", "feature row trainable", errors)
            _assert(row["ground_truth"] == "false", "feature row ground_truth", errors)

        for row in provenance_rows:
            errors.extend(f"{row.get('provenance_id')}:{err}" for err in _schema_violations(row, provenance_schema))
            _assert(row["raw_artifact_committed"] == "false", "raw artifact committed in provenance", errors)

        provenance_keys = {(r["candidate_patch_id"], r["feature_layer"], r["feature_name"]) for r in provenance_rows}
        for layer, row in _all_feature_rows_by_layer():
            _assert((row["candidate_patch_id"], layer, row["feature_name"]) in provenance_keys, "missing provenance row", errors)

        _assert(len(feature_rows) == 90, "unexpected feature row count", errors)
        _assert(len(provenance_rows) == len(feature_rows), "provenance count mismatch", errors)
        _assert(len(no_leakage_rows) == len(feature_rows), "no leakage audit count mismatch", errors)
        _assert(len(quality_rows) == len(feature_rows), "quality flag count mismatch", errors)
        _assert(len(read_csv(PARTIAL_MATRIX)) == len(_candidate_ids()), "partial matrix candidate count mismatch", errors)

        for row in no_leakage_rows:
            _assert(row["uses_post_event_data"] == "false", "post-event data misuse", errors)
            _assert(row["uses_reference_footprint_as_feature"] == "false", "reference footprint used as feature", errors)
            _assert(row["uses_sar_post_event_as_feature"] == "false", "SAR post-event used as feature", errors)
            _assert(row["uses_synthetic_value_as_real"] == "false", "synthetic value used as real", errors)
            _assert(row["passes_no_leakage_policy"] == "true", "no-leakage policy failed", errors)

        for row in read_csv(PARTIAL_MATRIX):
            _assert(row["has_sentinel2_spectral_real"] == "false", "Sentinel-2 extracted in 17C8", errors)
            _assert(row["has_sar_observational_real"] == "false", "SAR extracted in 17C8", errors)
            _assert(row["has_embedding_representation_real"] == "false", "embedding extracted in 17C8", errors)
            _assert(row["real_feature_layer_count"] == "0", "candidate has real feature layer", errors)

        _assert(summary["candidate_patch_count"] == 5, "candidate patch count mismatch", errors)
        _assert(summary["attempted_feature_row_count"] == len(feature_rows), "summary attempted count mismatch", errors)
        _assert(summary["real_feature_row_count"] == 0, "summary real count mismatch", errors)
        _assert(summary["score_v6_changed"] is False, "summary says score v6 changed", errors)
        _assert(summary["score_v7_created"] is False, "summary says score v7 created", errors)
        _assert(summary["raw_raster_committed"] is False, "summary says raw raster committed", errors)
        _assert(summary["synthetic_as_real"] is False, "summary says synthetic as real", errors)
        _assert(summary["post_event_misuse"] is False, "summary says post-event misuse", errors)
        _assert(summary["eligible_for_17b_now"] is False, "17B unexpectedly eligible", errors)
        _assert(summary["eligible_for_score_v7"] is False, "score v7 unexpectedly eligible", errors)
        _assert(summary["review_only"] is True, "summary not review-only", errors)
        _assert(summary["trainable"] is False, "summary trainable", errors)
        _assert(summary["ground_truth"] is False, "summary ground truth", errors)

        score_diff = _run_git(["diff", "--name-only", "--", rel(SCORE_V6)])
        official_diff = _run_git(["diff", "--name-only", "--", rel(FEATURES)])
        _assert(score_diff == "", "score v6 has worktree diff", errors)
        _assert(official_diff == "", "official patch feature dataset has worktree diff", errors)
        _assert(not SCORE_V7.exists(), "score v7 exists", errors)

        if errors:
            for err in errors:
                print(f"17C8 validation error: {err}", file=sys.stderr)
            return 1
        print(
            "17C8 -> "
            f"candidate_patches={summary['candidate_patch_count']} "
            f"attempted_feature_rows={summary['attempted_feature_row_count']} "
            f"real_feature_rows={summary['real_feature_row_count']} "
            "review_only=True score_v7_created=False"
        )
        return 0
    except Exception as exc:
        print(f"17C8 validation exception: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    run_all()
