import csv
import io
import os
import zipfile

import pytest

import scripts.protocolo_c.revp_v2bj_common as common

CANDIDATE = "REC_2022_05_24_30"


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _inmet_csv(uf, code, name, precip_value):
    out = io.StringIO()
    out.write("\n".join([
        "REGIAO:;NE", f"UF:;{uf}", f"ESTACAO:;{name}", f"CODIGO (WMO):;{code}",
        "LATITUDE:;-8,0", "LONGITUDE:;-34,9", "ALTITUDE:;10", "DATA DE FUNDACAO:;01/01/00",
        "Data;Hora UTC;PRECIPITACAO TOTAL, HORARIO (mm)",
    ]) + "\n")
    for day in ("2022/05/20", "2022/05/28", "2022/06/01"):
        for hour in ("0000", "1200"):
            out.write(f"{day};{hour} UTC;{precip_value}\n")
    return out.getvalue().encode("latin-1")


def _build_inmet_zip(path, present=True, a301_value="", a320_value="31,8"):
    if not present:
        return
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("INMET_NE_PE_A301_RECIFE_01-01-2022_A_31-12-2022.CSV", _inmet_csv("PE", "A301", "RECIFE", a301_value))
        archive.writestr("INMET_NE_PE_A357_PALMARES_01-01-2022_A_31-12-2022.CSV", _inmet_csv("PE", "A357", "PALMARES", ""))
        archive.writestr("INMET_NE_PE_A328_SURUBIM_01-01-2022_A_31-12-2022.CSV", _inmet_csv("PE", "A328", "SURUBIM", "0,2"))
        archive.writestr("INMET_NE_PB_A320_JOAO PESSOA_01-01-2022_A_31-12-2022.CSV", _inmet_csv("PB", "A320", "JOAO PESSOA", a320_value))


CHARTER_HTML = (
    '<h3>Landslides after effects in Recife/PE - Brazil</h3>'
    '{\\"title\\":\\"Landslides after effects in Recife/PE - Brazil\\",'
    '\\"vapArticleSlug\\":\\"landslides-scars-in-recife-pe-brazil\\",'
    '\\"vapAcquired\\":1654134300000,\\"vapCopyright\\":\\"Map produced by CENAD\\",'
    '\\"vapSourcesCopyrights\\":\\"Includes Pleiades material (c) CNES (2022), Distribution Airbus DS.\\"}'
)


def seed(workspace, evidence=True, charter="map", inmet_present=True, a301_value=""):
    datasets = os.path.join(workspace, "datasets")
    docs = os.path.join(workspace, "docs_v2bj")
    charter_cache = os.path.join(workspace, "charter_cache")
    os.makedirs(datasets, exist_ok=True)
    os.makedirs(charter_cache, exist_ok=True)

    common.DATASET_DIR = datasets
    common.DOCS_DIR = docs
    common.CHARTER_CACHE = charter_cache
    common.INMET_ZIP = os.path.join(workspace, "inmet_2022.zip")
    common.REFRESH_V2BI = False

    _build_inmet_zip(common.INMET_ZIP, present=inmet_present, a301_value=a301_value)

    if charter in ("map", "vector"):
        with open(os.path.join(charter_cache, "charter_758_recife_20220602_metadata.html"), "w", encoding="utf-8") as h:
            h.write(CHARTER_HTML)

    _write(os.path.join(datasets, common.V2AZ_QUEUE),
           ["review_packet_id", "event_patch_package_id", "candidate_id", "region", "event_date",
            "window_start", "window_end"],
           [{"review_packet_id": "ARP_v2az_0005", "event_patch_package_id": "FACT_v2at_0005",
             "candidate_id": CANDIDATE, "region": "Recife", "event_date": "2022-05-24",
             "window_start": "2022-05-17", "window_end": "2022-06-02"},
            {"review_packet_id": "ARP_v2az_0009", "event_patch_package_id": "FACT_v2at_0009",
             "candidate_id": "REC_2099_01_01", "region": "Recife", "event_date": "2099-01-01",
             "window_start": "2099-01-01", "window_end": "2099-01-02"}])

    audit_status = {"map": "PREVIEW_ONLY_FOUND", "vector": "VECTOR_CANDIDATE_FOUND",
                    "none": "NO_MANUAL_CHARTER_FILE_FOUND"}[charter]
    _write(os.path.join(datasets, common.V2BI["charter_audit"]),
           ["audit_status", "vector_candidate_found"],
           [{"audit_status": audit_status, "vector_candidate_found": "true" if charter == "vector" else "false"}])
    readiness = {"map": "PREVIEW_ONLY_NOT_READY", "vector": "CANDIDATE_GEOMETRY_READY_FOR_HUMAN_REVIEW",
                 "none": "NO_FILE_AVAILABLE"}[charter]
    _write(os.path.join(datasets, common.V2BI["charter_readiness"]),
           ["product_id", "updated_candidate_status"],
           [{"product_id": common.PRODUCT_ID, "updated_candidate_status": readiness}])

    temporal_rows = []
    if evidence:
        temporal_rows = [{"source_candidate": "ANA_HIDROWEB", "file_name": "ana.xml"},
                         {"source_candidate": "APAC", "file_name": "apac.pdf"}]
    else:
        temporal_rows = [{"source_candidate": "UNKNOWN", "file_name": ""}]
    _write(os.path.join(datasets, common.V2BI["temporal_cache"]),
           ["source_candidate", "file_name"], temporal_rows)
    _write(os.path.join(datasets, common.V2BI["parse"]),
           ["parse_status"], [{"parse_status": "NO_TEMPORAL_SERIES_FOUND"}])
    _write(os.path.join(datasets, common.V2BI["metrics"]),
           ["temporal_status"], [{"temporal_status": "NO_SERIES_AVAILABLE"}])
    gate_status = {"C0_PROVENANCE": "PASS", "C1_TEMPORALITY": "PENDING",
                   "C2_VALID_SERIES_OR_STATION": "BLOCKED", "C3_SPATIAL_ANCHOR": "PASS",
                   "C4_CANDIDATE_GEOMETRY": "PENDING_VECTOR_CRS", "C5_HUMAN_REVIEW": "PENDING",
                   "C6_CANDIDATE_REFERENCE": "BLOCKED", "C7_FINAL_GROUND_TRUTH": "BLOCKED"}
    _write(os.path.join(datasets, common.V2BI["gates"]),
           ["candidate_id", "gate_id", "updated_status"],
           [{"candidate_id": CANDIDATE, "gate_id": g, "updated_status": s} for g, s in gate_status.items()])


@pytest.fixture
def workspace(tmp_path):
    saved = (common.DATASET_DIR, common.DOCS_DIR, common.CHARTER_CACHE, common.INMET_ZIP, common.REFRESH_V2BI)
    yield str(tmp_path)
    common.DATASET_DIR, common.DOCS_DIR, common.CHARTER_CACHE, common.INMET_ZIP, common.REFRESH_V2BI = saved


def test_invariants_all_false_forbidden():
    row = common.with_invariants({})
    for field in ("can_create_ground_truth", "can_create_label", "can_create_negative",
                  "can_train_model", "can_create_patch_truth", "raw_data_versioned"):
        assert row[field] == "false"


def test_inmet_audit_a301_gap_and_proxy_not_substituted(workspace):
    seed(workspace, a301_value="")
    rows = common.run_audit_inmet_proxy_availability()
    by = {r["station_code"]: r for r in rows}
    assert by["A301"]["coverage_status"] == "PRECIP_FULL_GAP"
    assert by["A320"]["coverage_status"] == "PRECIP_AVAILABLE"
    assert all(r["usable_as_recife_local_rainfall"] == "false" for r in rows)
    assert by["A320"]["station_role"] == "REGIONAL_PROXY"


def test_inmet_audit_raw_absent_fail_closed(workspace):
    seed(workspace, inmet_present=False)
    rows = common.run_audit_inmet_proxy_availability()
    assert all(r["raw_present"] == "false" for r in rows)
    assert all(r["coverage_status"] == "RAW_NOT_PRESENT" for r in rows)
    assert all(r["usable_as_recife_local_rainfall"] == "false" for r in rows)


def test_charter_facts_landslide_scars(workspace):
    seed(workspace, charter="map")
    facts = common.extract_charter_facts()
    assert facts["feature_type_candidate"] == "LANDSLIDE_SCARS"
    assert facts["product_date"] == "2022-06-02"
    assert "CNES" in facts["license_terms"]


def test_charter_facts_absent_unknown(workspace):
    seed(workspace, charter="none")
    facts = common.extract_charter_facts()
    assert facts["feature_type_candidate"] == "UNKNOWN"
    assert facts["source_html_present"] == "false"


def test_gates_with_evidence_review_only(workspace):
    seed(workspace, evidence=True, charter="map")
    common.run_audit_inmet_proxy_availability()
    common.run_build_recife_intake_result_summary()
    rows = {r["gate_id"]: r for r in common.run_reconcile_recife_candidate_gates()}
    assert rows["C1_TEMPORALITY"]["reconciled_status"] == "TEMPORALITY_SUPPORTED_FOR_HUMAN_REVIEW"
    assert rows["C2_VALID_SERIES_OR_STATION"]["reconciled_status"] == "PARTIAL_FOR_HUMAN_REVIEW"
    assert rows["C3_SPATIAL_ANCHOR"]["reconciled_status"] == "PASS"
    assert rows["C4_CANDIDATE_GEOMETRY"]["reconciled_status"] == "MAP_PRESENT_PENDING_VECTOR_CRS"
    assert rows["C6_CANDIDATE_REFERENCE"]["reconciled_status"] == "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW"
    assert rows["C7_FINAL_GROUND_TRUTH"]["reconciled_status"] == "BLOCKED"
    assert all(r["promotion_allowed"] == "false" for r in rows.values())


def test_gates_without_evidence_fail_closed(workspace):
    seed(workspace, evidence=False, charter="none", inmet_present=False)
    common.run_audit_inmet_proxy_availability()
    common.run_build_recife_intake_result_summary()
    rows = {r["gate_id"]: r for r in common.run_reconcile_recife_candidate_gates()}
    assert rows["C1_TEMPORALITY"]["reconciled_status"] == "PENDING"
    assert rows["C2_VALID_SERIES_OR_STATION"]["reconciled_status"] == "BLOCKED"
    assert rows["C4_CANDIDATE_GEOMETRY"]["reconciled_status"] == "PENDING_VECTOR_CRS"
    assert rows["C6_CANDIDATE_REFERENCE"]["reconciled_status"] == "BLOCKED"
    assert rows["C7_FINAL_GROUND_TRUTH"]["reconciled_status"] == "BLOCKED"


def test_vector_present_allows_geometry_review_only(workspace):
    seed(workspace, evidence=True, charter="vector")
    common.run_audit_inmet_proxy_availability()
    common.run_build_recife_intake_result_summary()
    rows = {r["gate_id"]: r for r in common.run_reconcile_recife_candidate_gates()}
    assert rows["C4_CANDIDATE_GEOMETRY"]["reconciled_status"] == "PASS_FOR_HUMAN_REVIEW_ONLY"
    assert rows["C7_FINAL_GROUND_TRUTH"]["reconciled_status"] == "BLOCKED"


def test_reference_queue_scopes_intake(workspace):
    seed(workspace, evidence=True, charter="map")
    common.run_audit_inmet_proxy_availability()
    common.run_build_recife_intake_result_summary()
    common.run_reconcile_recife_candidate_gates()
    rows = {r["candidate_id"]: r for r in common.run_build_recife_candidate_reference_queue()}
    assert rows[CANDIDATE]["reference_status"] == "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW"
    assert rows["REC_2099_01_01"]["reference_status"] == "BLOCKED_NO_INTAKE"
    assert rows[CANDIDATE]["c7"] == "BLOCKED"


def test_orchestrator_end_to_end_and_guardrails(workspace):
    seed(workspace, evidence=True, charter="map")
    manifest = common.run_orchestrator()
    assert manifest[0]["step_name"] == "refresh_v2bi_intake"
    assert manifest[0]["status"] == "SKIPPED"
    for name in common.OUTPUTS.values():
        assert os.path.exists(common.dataset_path(name))
    guard = common.load_csv(common.dataset_path(common.OUTPUTS["guardrail"]))
    assert guard and all(r["status"] == "PASS" for r in guard)
    assert os.path.exists(common.doc_path("README.md"))
    assert os.path.exists(common.doc_path("candidate_review_packets", f"{CANDIDATE}.md"))


def test_guardrail_raises_on_promotion(workspace):
    seed(workspace, evidence=True, charter="map")
    common.run_audit_inmet_proxy_availability()
    common.run_build_recife_intake_result_summary()
    common.run_reconcile_recife_candidate_gates()
    common.run_build_recife_candidate_reference_queue()
    # Inject a forbidden promotion and confirm the regression fails closed.
    gates = common.load_csv(common.dataset_path(common.OUTPUTS["gates"]))
    gates[0]["promotion_allowed"] = "true"
    common.write_csv(common.dataset_path(common.OUTPUTS["gates"]), gates)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()
