import csv
import os

import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    raw = tmp_path / "local_only" / "raw"
    for p in (data, docs, cfg, raw):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "RAW_DIR", str(raw))
    monkeypatch.setattr(common, "STAGING_DIR", str(raw / "staging"))
    monkeypatch.setattr(common, "QUARANTINE_DIR", str(raw / "quarantine"))
    monkeypatch.setattr(common, "REPORTS_DIR", str(raw / "reports"))
    return data


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def install_candidate_discovery(data, official=True, date="2022-01-15", hazard="alagamento"):
    write_csv(os.path.join(data, "v1uv_curitiba_public_event_discovery.csv"), common.DISCOVERY_COLUMNS, [{
        "discovery_id": "D1", "source_id": "curitiba_prefeitura_news",
        "result_url": "https://www.curitiba.pr.gov.br/noticias/x",
        "http_status": "200", "content_type": "text/html", "title_hash": "abc",
        "date_signal": date, "hazard_signal": hazard,
        "official_source_status": "OFFICIAL_PUBLIC_SOURCE" if official else "NON_OFFICIAL_SOURCE",
        "event_specificity": "DATED_HAZARD_CURITIBA_EVENT" if official and date and hazard else "INSUFFICIENT_EVENT_SIGNAL",
        "candidate_status": "PUBLIC_OFFICIAL_EVENT_CANDIDATE_SIGNAL" if official and date and hazard else "BLOCKED",
        "blocking_reason": "", "notes": "fixture",
    }])


def test_source_target_builder_creates_curitiba_sources(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    rows = common.run_source_target_builder(common.parse_args([]))
    assert len(rows) >= 6
    assert any(r["source_id"] == "geocuritiba" for r in rows)
    assert any("alagamento" in r["query_terms"] for r in rows)
