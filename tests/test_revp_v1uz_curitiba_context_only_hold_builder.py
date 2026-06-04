import csv
import os
from pathlib import Path

import scripts.protocolo_c.revp_v1uz_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v1uz"


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
    reports = tmp_path / "local_only" / "v1uz" / "reports"
    for p in (data, docs, cfg, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data


def install_base_inputs(data):
    write_csv(data / "v1uw_curitiba_event_candidate_status.csv", [
        "event_status_id", "candidate_event_id", "proposed_event_id", "status",
        "hydromet_support",
    ], [{
        "event_status_id": "EVS_v1uw_0000", "candidate_event_id": "CE_v1uv_0000",
        "proposed_event_id": "CUR_2022_01_15",
        "status": "CURITIBA_EVENT_CANDIDATE_HYDROMET_SUPPORTED",
        "hydromet_support": "AVAILABLE",
    }])
    # Curitiba context-only layer classification fixture (context layers only).
    write_csv(data / "v1uy_curitiba_context_layer_classification.csv", [
        "layer_classification_id", "candidate_event_id", "layer_class",
        "context_only_status", "can_create_ground_reference",
    ], [{
        "layer_classification_id": "LC_v1uy_0000", "candidate_event_id": "CE_v1uv_0000",
        "layer_class": "ADMINISTRATIVE_CONTEXT_LAYER", "context_only_status": "CONTEXT_ONLY",
        "can_create_ground_reference": "false",
    }, {
        "layer_classification_id": "LC_v1uy_0001", "candidate_event_id": "CE_v1uv_0000",
        "layer_class": "DRAINAGE_CONTEXT_LAYER", "context_only_status": "CONTEXT_ONLY",
        "can_create_ground_reference": "false",
    }])
    # No possible occurrence layer (header only).
    write_csv(data / "v1uy_curitiba_possible_occurrence_layer_audit.csv", [
        "occurrence_layer_audit_id", "candidate_event_id", "can_advance_to_controlled_download",
    ], [])
    # Controlled download plan: nothing recommended.
    write_csv(data / "v1uy_curitiba_controlled_feature_download_plan.csv", [
        "download_plan_id", "candidate_event_id", "layer_class", "plan_status",
    ], [{
        "download_plan_id": "DP_v1uy_0000", "candidate_event_id": "CE_v1uv_0000",
        "layer_class": "NO_OCCURRENCE_LAYER_READY", "plan_status": "NO_CONTROLLED_DOWNLOAD_RECOMMENDED",
    }])
    # Curitiba prelinks (event-patch readiness fixture).
    write_csv(data / "v1uw_curitiba_event_patch_prelink_update.csv", [
        "prelink_update_id", "proposed_event_id", "patch_id", "region",
        "linkage_basis", "linkage_status", "sentinel_date_status",
        "event_candidate_status",
    ], [{
        "prelink_update_id": f"PL_v1uw_{i:05d}", "proposed_event_id": "CUR_2022_01_15",
        "patch_id": f"CUR_{i:05d}", "region": "CUR",
        "linkage_basis": "REGION_ONLY_EVENT_CANDIDATE",
        "linkage_status": "CANDIDATE_ONLY_NO_OVERLAY",
        "sentinel_date_status": "SENTINEL_DATE_MISSING",
        "event_candidate_status": "CURITIBA_EVENT_CANDIDATE_HYDROMET_SUPPORTED",
    } for i in range(3)])
    # Multi-region event-patch candidate registry (sentinel-missing matrix).
    us_rows = [{
        "event_patch_candidate_id": f"EPC_v1us_{i:05d}", "event_id": "REC_2022_05_24_30",
        "region": "REC", "patch_id": f"REC_{i:05d}",
        "linkage_basis": "REGION_ONLY_CANDIDATE_NO_SPATIAL_DISTANCE",
        "blocker": "SENTINEL_DATE_AND_GEOMETRY_MISSING",
    } for i in range(5)]
    us_rows.append({
        "event_patch_candidate_id": "EPC_v1us_00099", "event_id": "CUR_EVENT_REGISTRY_MISSING",
        "region": "CUR", "patch_id": "", "linkage_basis": "EVENT_REGISTRY_MISSING",
        "blocker": "CURITIBA_EVENT_REGISTRY_MISSING",
    })
    write_csv(data / "v1us_event_patch_candidate_registry.csv", [
        "event_patch_candidate_id", "event_id", "region", "patch_id",
        "linkage_basis", "blocker",
    ], us_rows)
    write_csv(data / "v1us_dino_review_support_attachment.csv", [
        "dino_attachment_id", "event_patch_candidate_id", "event_id", "region",
        "dino_review_support_status", "dino_usage",
    ], [{
        "dino_attachment_id": f"DINO_v1us_{i:05d}", "event_patch_candidate_id": f"EPC_v1us_{i:05d}",
        "event_id": "REC_2022_05_24_30", "region": "REC",
        "dino_review_support_status": "DINO_REVIEW_SUPPORT_AVAILABLE", "dino_usage": "SUPPORT_ONLY",
    } for i in range(5)])
    # Multi-region event registry (Recife, Petropolis, Curitiba).
    write_csv(data / "v1uo_multiregion_event_registry.csv", [
        "event_id", "region", "city", "uf",
    ], [
        {"event_id": "REC_2022_05_24_30", "region": "REC", "city": "Recife", "uf": "PE"},
        {"event_id": "PET_2022_02_15", "region": "PET", "city": "Petropolis", "uf": "RJ"},
        {"event_id": "CUR_EVENT_REGISTRY_MISSING", "region": "CUR", "city": "Curitiba", "uf": "PR"},
    ])


def test_hold_builder_creates_context_only_hold_status(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_curitiba_context_only_hold_builder(common.parse_args([]))
    assert len(rows) == 1
    hold = rows[0]
    assert hold["hold_status"] == "CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL"
    assert hold["context_layer_status"] == "CONTEXT_LAYERS_PRESENT"
    assert hold["possible_occurrence_layer_status"] == "POSSIBLE_OCCURRENCE_LAYER_ABSENT"
    assert hold["controlled_feature_download_status"] == "NO_CONTROLLED_DOWNLOAD_RECOMMENDED"
    assert hold["overlay_status"] == "BLOCKED"
    assert hold["ground_reference_status"] == "BLOCKED"
    assert hold["can_create_ground_reference"] == "false"
    assert hold["can_create_training_label"] == "false"
