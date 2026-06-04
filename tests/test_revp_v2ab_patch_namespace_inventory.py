import csv
import os
import shutil
from pathlib import Path

import scripts.protocolo_c.revp_v2ab_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v2ab"


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    scan = tmp_path / "scan_root"
    staging = tmp_path / "local_only" / "v2ab" / "staging"
    reports = tmp_path / "local_only" / "v2ab" / "reports"
    for p in (data, docs, cfg, scan, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "SCAN_ROOTS", [str(scan)])
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data, scan


# Patch-source fixtures scanned for namespaces / crosswalks.
CORE_SCAN_FIXTURES = [
    "v1us_event_patch_candidate_registry.csv",
    "dino_patch_visual_linkage_fixture.csv",
    "official_anchor_sentinel_patch_fixture.csv",
    "recife_sentinel_patch_fixture.csv",
    "conflicting_namespace_patch_fixture.csv",
]


def install_base_inputs(data, scan, with_explicit_crosswalk=False):
    fixtures = list(CORE_SCAN_FIXTURES)
    if with_explicit_crosswalk:
        fixtures.append("explicit_crosswalk_patch_fixture.csv")
    for name in fixtures:
        shutil.copy(FIXTURE_DIR / name, Path(scan) / name)

    # Event-patch candidate registry also read directly from DATASET_DIR.
    shutil.copy(FIXTURE_DIR / "v1us_event_patch_candidate_registry.csv",
                Path(data) / "v1us_event_patch_candidate_registry.csv")
    # Add a candidate whose OWN patch recovered a same-namespace date.
    cands = read_csv(Path(data) / "v1us_event_patch_candidate_registry.csv")
    cands.append({
        "event_patch_candidate_id": "EPC3", "event_id": "REC_2022_05_24_30",
        "region": "REC", "patch_id": "REC_00009", "linkage_basis": "REGION_ONLY_CANDIDATE",
        "linkage_status": "CANDIDATE_NON_OPERATIONAL", "event_patch_candidate_only": "true",
        "patch_bound_truth": "false", "can_create_ground_reference": "false",
        "can_create_training_label": "false", "blocker": "SENTINEL_DATE_AND_GEOMETRY_MISSING",
        "notes": "x",
    })
    write_csv(Path(data) / "v1us_event_patch_candidate_registry.csv", list(cands[0].keys()), cands)

    epcs = ["EPC0", "EPC1", "EPC3"]
    write_csv(data / "v1us_phenomenon_status_attachment.csv",
              ["phenomenon_attachment_id", "event_patch_candidate_id", "event_id", "patch_id", "region", "phenomenon_class"],
              [{"phenomenon_attachment_id": f"PH{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "region": "REC", "phenomenon_class": "flood_context"} for i, e in enumerate(epcs)])
    write_csv(data / "v1us_geometry_blocker_attachment.csv",
              ["geometry_blocker_id", "event_patch_candidate_id", "event_id", "patch_id", "region", "coordinate_status", "geometry_status", "overlay_blocker"],
              [{"geometry_blocker_id": f"G{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "region": "REC", "coordinate_status": "NO_COORD", "geometry_status": "NO_OBSERVED_GEOMETRY", "overlay_blocker": "BLOCKED"} for i, e in enumerate(epcs)])
    write_csv(data / "v1us_external_evidence_attachment_registry.csv",
              ["attachment_id", "event_patch_candidate_id", "event_id", "patch_id", "evidence_source", "evidence_status"],
              [{"attachment_id": f"A{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "evidence_source": "official", "evidence_status": "CONTEXTUAL"} for i, e in enumerate(epcs)])
    write_csv(data / "v1us_dino_review_support_attachment.csv",
              ["dino_attachment_id", "event_patch_candidate_id", "event_id", "patch_id", "region", "dino_review_support_status", "dino_usage"],
              [{"dino_attachment_id": f"D{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "region": "REC", "dino_review_support_status": "DINO_REVIEW_SUPPORT_AVAILABLE", "dino_usage": "SUPPORT_ONLY"} for i, e in enumerate(epcs)])

    # v2aa recovered dates: parallel namespaces for REC/PET + same-patch REC_00009.
    write_csv(data / "v2aa_patch_date_candidate_consolidation.csv",
              ["patch_date_id", "patch_id", "region", "candidate_dates", "selected_sentinel_date", "selected_sentinel_datetime", "source_count", "agreeing_source_count", "conflict_status", "consolidation_status", "sentinel_date_recovered", "sentinel_date_inferred", "notes"],
              [
                  {"patch_date_id": "PD0", "patch_id": "REFPATCH_REC_001", "region": "REC", "candidate_dates": "2022-05-25", "selected_sentinel_date": "2022-05-25", "selected_sentinel_datetime": "", "source_count": "1", "agreeing_source_count": "1", "conflict_status": "NONE", "consolidation_status": "DATE_CONFIRMED_SINGLE_SOURCE", "sentinel_date_recovered": "true", "sentinel_date_inferred": "false", "notes": "x"},
                  {"patch_date_id": "PD1", "patch_id": "REC_PATCH_A", "region": "REC", "candidate_dates": "2022-05-26", "selected_sentinel_date": "2022-05-26", "selected_sentinel_datetime": "", "source_count": "1", "agreeing_source_count": "1", "conflict_status": "NONE", "consolidation_status": "DATE_CONFIRMED_SINGLE_SOURCE", "sentinel_date_recovered": "true", "sentinel_date_inferred": "false", "notes": "x"},
                  {"patch_date_id": "PD2", "patch_id": "REFPATCH_PET_001", "region": "PET", "candidate_dates": "2022-02-02", "selected_sentinel_date": "2022-02-02", "selected_sentinel_datetime": "", "source_count": "1", "agreeing_source_count": "1", "conflict_status": "NONE", "consolidation_status": "DATE_CONFIRMED_SINGLE_SOURCE", "sentinel_date_recovered": "true", "sentinel_date_inferred": "false", "notes": "x"},
                  {"patch_date_id": "PD3", "patch_id": "REC_00009", "region": "REC", "candidate_dates": "2022-05-25", "selected_sentinel_date": "2022-05-25", "selected_sentinel_datetime": "", "source_count": "1", "agreeing_source_count": "1", "conflict_status": "NONE", "consolidation_status": "DATE_CONFIRMED_SINGLE_SOURCE", "sentinel_date_recovered": "true", "sentinel_date_inferred": "false", "notes": "x"},
              ])
    write_csv(data / "v2aa_sentinel_date_confidence_audit.csv",
              ["confidence_audit_id", "patch_id", "selected_sentinel_date", "confidence_class", "confidence_score", "usable_for_temporal_linkage", "blocker", "notes"],
              [
                  {"confidence_audit_id": "CA0", "patch_id": "REFPATCH_REC_001", "selected_sentinel_date": "2022-05-25", "confidence_class": "HIGH_CONFIDENCE", "confidence_score": "90", "usable_for_temporal_linkage": "true", "blocker": "", "notes": "x"},
                  {"confidence_audit_id": "CA1", "patch_id": "REC_PATCH_A", "selected_sentinel_date": "2022-05-26", "confidence_class": "MEDIUM_CONFIDENCE", "confidence_score": "70", "usable_for_temporal_linkage": "true", "blocker": "", "notes": "x"},
                  {"confidence_audit_id": "CA2", "patch_id": "REFPATCH_PET_001", "selected_sentinel_date": "2022-02-02", "confidence_class": "HIGH_CONFIDENCE", "confidence_score": "90", "usable_for_temporal_linkage": "true", "blocker": "", "notes": "x"},
                  {"confidence_audit_id": "CA3", "patch_id": "REC_00009", "selected_sentinel_date": "2022-05-25", "confidence_class": "HIGH_CONFIDENCE", "confidence_score": "90", "usable_for_temporal_linkage": "true", "blocker": "", "notes": "x"},
              ])


def test_namespace_inventory_separates_namespaces(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    rows = common.run_patch_namespace_inventory(common.parse_args([]))
    classes = {r["namespace_class"] for r in rows}
    assert common.NS_DINO in classes
    assert common.NS_EVENT in classes
    assert common.NS_ANCHOR in classes
    assert common.NS_SCAFFOLD in classes
    # Equivalence between namespaces is never assumed.
    assert all(r["can_crosswalk_automatically"] == "false" for r in rows)
    # REFPATCH and numeric event ids are not collapsed into one namespace.
    anchor = next(r for r in rows if r["namespace_class"] == common.NS_ANCHOR)
    event = next(r for r in rows if r["namespace_class"] == common.NS_EVENT)
    assert anchor["patch_count"] != "0" and event["patch_count"] != "0"
