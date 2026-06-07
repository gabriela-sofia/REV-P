import csv
import shutil
from pathlib import Path

import pytest

import scripts.protocolo_c.revp_v2an_common as common

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "v2an"


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets"
    protocol = data / "protocolo_c"
    docs = tmp_path / "docs" / "protocolo_c" / "v2an_ground_reference_validation_sprint"
    dossiers = docs / "dossiers"
    cfg = tmp_path / "configs" / "protocolo_c"
    for path in (data, protocol, docs, dossiers, cfg):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "PROTOCOL_C_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "DOSSIER_DIR", str(dossiers))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "NETWORK_ENABLED", False)
    return data, protocol, docs, dossiers


def install_registry(data, fixture="nine_valid.csv"):
    shutil.copy(FIXTURES / fixture,
                data / "observed_event_reference_candidate_registry.csv")


def install_all(tmp_path, monkeypatch, fixture="nine_valid.csv"):
    data, protocol, docs, dossiers = set_env(tmp_path, monkeypatch)
    install_registry(data, fixture)
    return data, protocol, docs, dossiers


# --- common tests ----------------------------------------------------------
def test_loads_exactly_nine(tmp_path, monkeypatch):
    data, *_ = install_all(tmp_path, monkeypatch)
    rows = common.load_nine_observed_candidates()
    assert len(rows) == 9


def test_fails_on_less_than_nine(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch, "less_than_nine.csv")
    with pytest.raises(ValueError):
        common.load_nine_observed_candidates()


def test_fails_on_more_than_nine(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch, "more_than_nine.csv")
    with pytest.raises(ValueError):
        common.load_nine_observed_candidates()


def test_fails_when_registry_missing(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        common.load_nine_observed_candidates()


def test_is_true_fail_closed():
    assert common.is_true("true") and common.is_true("TRUE")
    for bad in ("", "false", "1", "yes", None):
        assert not common.is_true(bad)


def test_repo_relative_rejects_absolute():
    assert common.repo_relative_path("a\\b.csv") == "a/b.csv"
    with pytest.raises(ValueError):
        common.repo_relative_path("C:\\Users\\x.csv")


def test_normalize_helpers():
    assert common.normalize_date("2022-05-24") == "2022-05-24"
    assert common.normalize_region("", "REC_2022") == "Recife"
    assert common.normalize_region("", "CTB_x") == "Curitiba"
    hazard, amb = common.normalize_phenomenon("evento_misto")
    assert amb == "true"
    assert common.event_window_days("2022-05-24", "2022-05-30") == "7"
    assert common.safe_slug("Recife 2022") == "recife-2022"


def test_assert_no_absolute_paths_and_local_only():
    with pytest.raises(ValueError):
        common.assert_no_absolute_paths_in_content([{"p": "C:\\Users\\x"}])
    with pytest.raises(ValueError):
        common.assert_no_local_only([{"p": "local" + "_only"}])


def test_assert_no_operational_promotion():
    common.assert_no_operational_promotion([{"forbidden_decisions": "GROUND_TRUTH_VALIDATED|LABEL_READY"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"can_create_ground_truth": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_operational_promotion([{"note": "raw_data_versioned=true"}])


def test_assert_no_raw_data_versioned():
    common.assert_no_raw_data_versioned([{"raw_data_versioned": "false"}])
    with pytest.raises(ValueError):
        common.assert_no_raw_data_versioned([{"raw_data_versioned": "true"}])


def test_assert_no_fake_ground_truth():
    common.assert_no_fake_ground_truth([{"operational_ground_truth_status": "NOT_ESTABLISHED"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_ground_truth([{"operational_ground_truth_status": "GROUND_TRUTH_VALIDATED"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_ground_truth([{"can_create_ground_truth": "true"}])


def test_assert_no_fake_patch_overlay():
    common.assert_no_fake_patch_overlay([{"overlay_ready": "false"}])
    with pytest.raises(ValueError):
        common.assert_no_fake_patch_overlay([{
            "overlay_ready": "true", "has_event_geometry_available": "false",
            "has_patch_geometry_available": "false"}])


def test_assert_output_is_v2an():
    common.assert_output_is_v2an("datasets/protocolo_c/v2an_x.csv")
    with pytest.raises(ValueError):
        common.assert_output_is_v2an("datasets/protocolo_c/observed_event.csv")


def test_status_enum_not_flagged_with_safe_context():
    # uppercase forbidden enum only counts without a safe marker on the line
    assert common.scan_text_violations(
        "Estados proibidos a evitar: GROUND_TRUTH_VALIDATED")["forbidden_status"] == 0
    assert common.scan_text_violations(
        "status GROUND_TRUTH_VALIDATED agora")["forbidden_status"] == 1
