import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Shared helpers for the v2at Evidence Registry + Event-Patch Package Engine.
# --------------------------------------------------------------------------- #

V2AT_ENGINE_PATH = ROOT / "scripts" / "v2at_evidence_registry_event_patch_engine.py"


def _load_v2at_engine():
    spec = importlib.util.spec_from_file_location("v2at_engine", V2AT_ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def v2at_engine():
    return _load_v2at_engine()


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_v2at_dataset(dataset_dir: Path, *, include_conflict=False,
                       include_curitiba_missing=True, context_only_region=False):
    """Create a minimal but realistic v2at input dataset under ``dataset_dir``.

    Mirrors the column contract of the real REV-P registries so the engine reads
    it exactly as it reads production data, but small enough for deterministic
    assertions.
    """
    pc = dataset_dir / "protocolo_c"

    candidates = [
        {"event_patch_candidate_id": "EPC_t_0000", "event_id": "REC_2022_05_24_30",
         "region": "REC", "patch_id": "REC_00205", "linkage_basis": "REGION_ONLY",
         "linkage_status": "CANDIDATE_NON_OPERATIONAL", "event_patch_candidate_only": "true",
         "patch_bound_truth": "false", "can_create_ground_reference": "false",
         "can_create_training_label": "false", "blocker": "SENTINEL_DATE_AND_GEOMETRY_MISSING",
         "notes": "test"},
        {"event_patch_candidate_id": "EPC_t_0001", "event_id": "PET_2022_02_15",
         "region": "PET", "patch_id": "PET_00016", "linkage_basis": "REGION_ONLY",
         "linkage_status": "CANDIDATE_NON_OPERATIONAL", "event_patch_candidate_only": "true",
         "patch_bound_truth": "false", "can_create_ground_reference": "false",
         "can_create_training_label": "false", "blocker": "SENTINEL_DATE_AND_GEOMETRY_MISSING",
         "notes": "test"},
    ]
    if include_curitiba_missing:
        candidates.append(
            {"event_patch_candidate_id": "EPC_t_0002", "event_id": "CUR_EVENT_REGISTRY_MISSING",
             "region": "CUR", "patch_id": "CUR_00007", "linkage_basis": "REGION_ONLY",
             "linkage_status": "BLOCKED_NO_CLEAR_EVENT", "event_patch_candidate_only": "true",
             "patch_bound_truth": "false", "can_create_ground_reference": "false",
             "can_create_training_label": "false", "blocker": "CURITIBA_EVENT_REGISTRY_MISSING",
             "notes": "test"})
    _write_csv(pc / "v1us_event_patch_candidate_registry.csv", candidates)

    dino = [{"dino_attachment_id": f"DINO_t_{i:04d}",
             "event_patch_candidate_id": c["event_patch_candidate_id"],
             "event_id": c["event_id"], "patch_id": c["patch_id"], "region": c["region"],
             "dino_registry": "dino_patch_visual_linkage_registry_v1pv.csv",
             "dino_review_support_status": "DINO_REVIEW_SUPPORT_AVAILABLE",
             "dino_usage": "SUPPORT_ONLY", "can_create_training_label": "false",
             "notes": "DINO is review-only"} for i, c in enumerate(candidates)]
    _write_csv(pc / "v1us_dino_review_support_attachment.csv", dino)

    sentinel = [
        {"confidence_audit_id": "CA_t_0", "patch_id": "REC_2022_05_24_30",
         "selected_sentinel_date": "2022-05-24", "confidence_class": "MEDIUM_CONFIDENCE",
         "confidence_score": "70", "usable_for_temporal_linkage": "true", "blocker": "",
         "notes": "usable"},
    ]
    if include_conflict:
        sentinel.append(
            {"confidence_audit_id": "CA_t_1", "patch_id": "PET_2022_02_15",
             "selected_sentinel_date": "", "confidence_class": "BLOCKED_CONFLICT",
             "confidence_score": "0", "usable_for_temporal_linkage": "false",
             "blocker": "date_conflict_between_sources", "notes": "conflict"})
    _write_csv(pc / "v2aa_sentinel_date_confidence_audit.csv", sentinel)

    _write_csv(pc / "v2ab_event_patch_package_validation.csv", [
        {"validation_id": "VAL_t_0", "event_patch_candidate_id": c["event_patch_candidate_id"],
         "event_id": c["event_id"], "patch_id": c["patch_id"],
         "package_schema_status": "SCHEMA_FIELDS_PRESENT", "missing_required_fields": "",
         "nullable_without_blocker": "0", "namespace_status": "NAMESPACE_RESOLVED",
         "temporal_field_status": "SENTINEL_DATE_RECOVERED", "crosswalk_status": "NO_EXPLICIT_CROSSWALK",
         "unsafe_value_count": "0", "validation_status": "PACKAGE_VALID_WITH_TEMPORAL_BLOCKER",
         "can_create_ground_reference": "false", "can_create_training_label": "false",
         "notes": "test"} for c in candidates])

    _write_csv(dataset_dir / "ground_reference_event_registry.csv", [
        {"event_id": "EVENT_PET2022_CPRM_ANEXOII_19022022",
         "source_event_unit_id": "PET2022_CPRM_ANEXOII_19022022", "source_institution": "SGB/CPRM",
         "source_document_sanitized": "ANEXO-II.pdf", "region": "PET", "municipality": "Petropolis",
         "locality_text_sanitized": "Moinho Preto", "event_or_survey_date": "19/02/2022",
         "temporal_precision": "DAY_EXPLICIT", "phenomenon_group": "MOVEMENT_OF_MASS",
         "coordinate_status": "EXPLICIT_COORDINATE", "latitude": "-22.48", "longitude": "-43.21",
         "spatial_precision": "EXPLICIT_POINT_COORDINATE_REPRESENTATIVE_ANCHOR",
         "source_confidence": "OFFICIAL_CPRM_EXPLICIT_COORDINATE_HIGH", "c_level": "C3_EVENT_PATCH_LINKED",
         "c_level_reason": "linked", "can_be_ground_reference_event": "true",
         "can_be_operational_ground_truth": "false", "can_create_training_label": "false",
         "notes": "test"}])

    external = [
        {"evidence_id": "recife_pe3d", "source_name": "PE3D/MDE", "region": "Recife",
         "evidence_type": "terrain", "institutional_origin": "Programa PE3D", "format": "GeoTIFF",
         "local_status": "EXISTS_PARTIAL", "public_status": "SUMMARY_ONLY", "header_evidence": "x",
         "spatial_coverage": "RMR", "patch_coverage": "18", "evidence_tier": "STRONG",
         "limitations": "partial", "role_in_pipeline": "context", "related_manifest": "m"},
        {"evidence_id": "cur_pref", "source_name": "Prefeitura Curitiba", "region": "Curitiba",
         "evidence_type": "terrain", "institutional_origin": "Prefeitura de Curitiba", "format": "SHP",
         "local_status": "EXISTS_PARTIAL", "public_status": "SUMMARY_ONLY", "header_evidence": "x",
         "spatial_coverage": "Curitiba", "patch_coverage": "14", "evidence_tier": "STRONG",
         "limitations": "partial", "role_in_pipeline": "context", "related_manifest": "m"},
    ]
    if context_only_region:
        external.append(
            {"evidence_id": "media_x", "source_name": "Press report", "region": "Recife",
             "evidence_type": "land_use", "institutional_origin": "MapBiomas Brasil", "format": "raster",
             "local_status": "INDEXED_ONLY", "public_status": "SUMMARY_ONLY", "header_evidence": "x",
             "spatial_coverage": "BR", "patch_coverage": "0", "evidence_tier": "EXPLORATORY",
             "limitations": "context", "role_in_pipeline": "context", "related_manifest": "m"})
    _write_csv(dataset_dir / "external_evidence_registry.csv", external)

    _write_csv(pc / "v2bm_cross_region_candidate_registry.csv", [
        {"reference_id": "XREF_t_001", "region": "Recife", "city": "Recife", "package_id": "p",
         "reference_status": "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE",
         "phenomenon_scope": "LANDSLIDE_SCARS_WITH_FLOOD_EVENT_CONTEXT",
         "evidence_basis": "Charter 758 raster landslide-scars product", "allowed_use": "x",
         "forbidden_use": "x", "uncertainty_level": "MODERATE", "evidence_score": "0.76"},
        {"reference_id": "XREF_t_002", "region": "Curitiba", "city": "Curitiba", "package_id": "p",
         "reference_status": "PROTOCOL_VALIDATED_TEMPORAL_REFERENCE",
         "phenomenon_scope": "URBAN_FLOOD_EVENT_TEMPORAL_CONTEXT",
         "evidence_basis": "A807 LOCAL strong precipitation", "allowed_use": "x",
         "forbidden_use": "x", "uncertainty_level": "MODERATE", "evidence_score": "0.70"},
        {"reference_id": "XREF_t_003", "region": "Petropolis", "city": "Petropolis", "package_id": "p",
         "reference_status": "PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE",
         "phenomenon_scope": "LANDSLIDE_FLOOD_REGIONAL_TEMPORAL_CONTEXT",
         "evidence_basis": "A610 REGIONAL_PROXY temporal context", "allowed_use": "x",
         "forbidden_use": "x", "uncertainty_level": "HIGH", "evidence_score": "0.55"}])

    scorecard = []
    for region in ("Recife", "Curitiba", "Petropolis"):
        scorecard.append({"region": region, "evidence_axis": "TEMPORALITY", "score": "0.6",
                          "score_reason": "dated", "supports_reference_status": "true",
                          "supports_operational_label": "false", "limitation": "not spatial"})
    _write_csv(pc / "v2bm_cross_region_evidence_scorecard.csv", scorecard)

    return dataset_dir


@pytest.fixture
def v2at_dataset(tmp_path):
    def _build(**kwargs):
        ds = tmp_path / "datasets"
        build_v2at_dataset(ds, **kwargs)
        return ds
    return _build


# --------------------------------------------------------------------------- #
# Shared helpers for the v2au Patch-Event Overlay Geometry Engine.
# --------------------------------------------------------------------------- #

V2AU_ENGINE_PATH = ROOT / "scripts" / "v2au_patch_event_overlay_geometry_engine.py"


def _load_v2au_engine():
    spec = importlib.util.spec_from_file_location("v2au_engine", V2AU_ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def v2au_engine():
    return _load_v2au_engine()


_V2AU_PACKAGE_COLUMNS = [
    "package_id", "event_id", "patch_id", "region", "city", "hazard_type",
    "has_patch_overlay", "intersection_ratio", "promotion_candidate_level",
    "promotion_decision", "blocking_reason", "allowed_use", "evidence_score",
    "has_spatial_support", "urban_context",
]

_V2AU_GEOMETRY_COLUMNS = [
    "geometry_role", "linked_event_id", "linked_patch_id", "source_id",
    "source_name", "geometry_type", "geometry_format", "geometry_value",
    "geometry_path", "crs", "latitude", "longitude",
]


def _v2au_package(package_id, event_id, patch_id, region="Recife", **over):
    base = {
        "package_id": package_id, "event_id": event_id, "patch_id": patch_id,
        "region": region, "city": region, "hazard_type": "urban_flood",
        "has_patch_overlay": "false", "intersection_ratio": "UNKNOWN",
        "promotion_candidate_level": "C3",
        "promotion_decision": "C3_CANDIDATE_REFERENCE_HOLD_FOR_OVERLAY",
        "blocking_reason": "NO_PATCH_EVENT_OVERLAY_GEOMETRY",
        "allowed_use": "candidate_reference", "evidence_score": "0.701",
        "has_spatial_support": "true", "urban_context": "true",
    }
    base.update(over)
    return base


def _v2au_geom(role, geometry_format="bbox", value="", crs="EPSG:3857",
               linked_event_id="", linked_patch_id="", **over):
    base = {
        "geometry_role": role, "linked_event_id": linked_event_id,
        "linked_patch_id": linked_patch_id, "source_id": "MANUAL_TEST",
        "source_name": "test geometry", "geometry_type": "", "geometry_format": geometry_format,
        "geometry_value": value, "geometry_path": "", "crs": crs,
        "latitude": "", "longitude": "",
    }
    base.update(over)
    return base


def build_v2au_dataset(dataset_dir, *, packages=None, geometry_sources=None, ground_events=None):
    """Create a minimal v2au input dataset (v2at packages + optional geometries)."""
    if packages is None:
        packages = [_v2au_package("PKG_test000001", "REC_2022_05_24_30", "REC_00205")]
    _write_csv(dataset_dir / "v2at_event_patch_package_registry.csv",
               [{c: p.get(c, "") for c in _V2AU_PACKAGE_COLUMNS} for p in packages])

    if ground_events is None:
        ground_events = []
    if ground_events:
        cols = ["event_id", "region", "event_or_survey_date", "coordinate_status",
                "latitude", "longitude", "phenomenon_group"]
        _write_csv(dataset_dir / "ground_reference_event_registry.csv",
                   [{c: e.get(c, "") for c in cols} for e in ground_events])

    if geometry_sources:
        _write_csv(dataset_dir / "v2au_geometry_sources.csv",
                   [{c: g.get(c, "") for c in _V2AU_GEOMETRY_COLUMNS} for g in geometry_sources])
    return dataset_dir


@pytest.fixture
def v2au_dataset(tmp_path):
    def _build(**kwargs):
        ds = tmp_path / "datasets"
        ds.mkdir(parents=True, exist_ok=True)
        build_v2au_dataset(ds, **kwargs)
        return ds
    return _build


@pytest.fixture
def v2au_make_package():
    return _v2au_package


@pytest.fixture
def v2au_make_geom():
    return _v2au_geom
