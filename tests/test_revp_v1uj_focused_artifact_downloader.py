"""Tests for v1uj — Focused Artifact Downloader."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_focused_artifact_downloader.py")
ALLOWED = os.path.join("configs", "protocolo_c", "v1ui_allowed_domains.yaml")


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_allowed(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("allowed_domains:\n  local:\n    - \"\"\n")


def _run(tmp_path, copernicus="", ckan="", extra=None, allowed=ALLOWED):
    out = os.path.join(tmp_path, "manifest.csv")
    collision = os.path.join(tmp_path, "collision.csv")
    cmd = [sys.executable, SCRIPT, "--out", out, "--collision-audit-out", collision,
           "--allowed-domains", allowed,
           "--local-only-dir", os.path.join(tmp_path, "raw")]
    if copernicus:
        cmd += ["--copernicus", copernicus]
    else:
        cmd += ["--copernicus", os.path.join(tmp_path, "none0.csv")]
    if ckan:
        cmd += ["--ckan", ckan]
    else:
        cmd += ["--ckan", os.path.join(tmp_path, "none3.csv")]
    # registries inexistentes para os demais
    cmd += ["--s2id", os.path.join(tmp_path, "none1.csv"),
            "--rigeo", os.path.join(tmp_path, "none2.csv")]
    cmd += (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(collision, "r", encoding="utf-8") as f:
        collision_rows = list(csv.DictReader(f))
    return rows, collision_rows


class TestFocusedArtifactDownloader:
    def test_empty(self, tmp_path):
        rows, collision = _run(str(tmp_path))
        assert rows == []
        assert collision == []

    def test_allowed_candidate_dry_run(self, tmp_path):
        cop = os.path.join(str(tmp_path), "cop.csv")
        _write(cop, ["event_id", "product_url", "format_hint", "download_allowed"], [
            {"event_id": "PET_2022_02_15",
             "product_url": "https://emergency.copernicus.eu/download/EMSR564_DEL.zip",
             "format_hint": ".zip", "download_allowed": "true"},
            {"event_id": "PET_2022_02_15",
             "product_url": "https://emergency.copernicus.eu/download/ql.png",
             "format_hint": ".png", "download_allowed": "false"},
        ])
        rows, _collision = _run(str(tmp_path), copernicus=cop)
        assert len(rows) == 1  # quicklook excluido pelo predicate
        assert rows[0]["download_status"] == "DRY_RUN"
        assert rows[0]["domain"] == "emergency.copernicus.eu"

    def test_blocked_domain(self, tmp_path):
        cop = os.path.join(str(tmp_path), "cop.csv")
        _write(cop, ["event_id", "product_url", "format_hint", "download_allowed"], [
            {"event_id": "X", "product_url": "https://evil.example.com/a.zip",
             "format_hint": ".zip", "download_allowed": "true"},
        ])
        rows, _collision = _run(str(tmp_path), copernicus=cop)
        assert rows[0]["download_status"] == "BLOCKED_DOMAIN"

    def test_build_candidates_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_focused_artifact_downloader import build_candidates
        regs = {
            "copernicus": [{"event_id": "E", "product_url": "u1",
                            "format_hint": ".zip", "download_allowed": "true"}],
            "ckan": [{"event_id": "E", "resource_url": "u2", "resource_format": "CSV",
                      "is_geospatial_candidate": "true", "is_contextual_only": "false"},
                     {"event_id": "E", "resource_url": "u3", "resource_format": "SHP",
                      "is_geospatial_candidate": "true", "is_contextual_only": "true"}],
            "s2id": [],
            "rigeo": [],
        }
        cands = build_candidates(regs)
        urls = {c["url"] for c in cands}
        assert "u1" in urls and "u2" in urls
        assert "u3" not in urls  # contextual_only excluido

    def test_distinct_urls_same_basename_get_distinct_safe_names(self, tmp_path):
        allowed = os.path.join(str(tmp_path), "allowed.yaml")
        _write_allowed(allowed)
        src_a = tmp_path / "a" / "same.csv"
        src_b = tmp_path / "b" / "same.csv"
        src_a.parent.mkdir()
        src_b.parent.mkdir()
        src_a.write_text("id\n1\n", encoding="utf-8")
        src_b.write_text("id\n2\n", encoding="utf-8")
        ckan = os.path.join(str(tmp_path), "ckan.csv")
        cols = ["event_id", "resource_url", "resource_format", "is_geospatial_candidate",
                "is_contextual_only", "resource_id", "ckan_record_id"]
        _write(ckan, cols, [
            {"event_id": "REC_2022_05_24_30", "resource_url": src_a.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "R1", "ckan_record_id": "C1"},
            {"event_id": "REC_2022_05_24_30", "resource_url": src_b.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "R2", "ckan_record_id": "C2"},
        ])
        rows, collision = _run(str(tmp_path), ckan=ckan, allowed=allowed,
                               extra=["--allow-web", "--download"])
        assert len({r["safe_filename"] for r in rows}) == 2
        assert all(r["download_status"] == "DOWNLOAD_OK" for r in rows)
        assert all(r["collision_status"] == "COLLISION_DETECTED" for r in rows)
        assert any(c["collision_status"] == "COLLISION_DETECTED" for c in collision)

    def test_same_basename_different_hashes_not_generic_already_exists(self, tmp_path):
        allowed = os.path.join(str(tmp_path), "allowed.yaml")
        _write_allowed(allowed)
        src_a = tmp_path / "a" / "asset.csv"
        src_b = tmp_path / "b" / "asset.csv"
        src_a.parent.mkdir()
        src_b.parent.mkdir()
        src_a.write_text("id\nalpha\n", encoding="utf-8")
        src_b.write_text("id\nbeta\n", encoding="utf-8")
        ckan = os.path.join(str(tmp_path), "ckan.csv")
        cols = ["event_id", "resource_url", "resource_format", "is_geospatial_candidate",
                "is_contextual_only", "resource_id"]
        _write(ckan, cols, [
            {"event_id": "REC_2022_05_24_30", "resource_url": src_a.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "A"},
            {"event_id": "REC_2022_05_24_30", "resource_url": src_b.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "B"},
        ])
        rows, _collision = _run(str(tmp_path), ckan=ckan, allowed=allowed,
                                extra=["--allow-web", "--download"])
        assert {r["download_status"] for r in rows} == {"DOWNLOAD_OK"}
        assert "ALREADY_EXISTS" not in {r["download_status"] for r in rows}

    def test_same_url_second_run_is_same_url_same_hash(self, tmp_path):
        allowed = os.path.join(str(tmp_path), "allowed.yaml")
        _write_allowed(allowed)
        src = tmp_path / "source" / "asset.csv"
        src.parent.mkdir()
        src.write_text("id\n1\n", encoding="utf-8")
        ckan = os.path.join(str(tmp_path), "ckan.csv")
        cols = ["event_id", "resource_url", "resource_format", "is_geospatial_candidate",
                "is_contextual_only", "resource_id"]
        _write(ckan, cols, [
            {"event_id": "REC_2022_05_24_30", "resource_url": src.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "A"},
        ])
        first, _collision = _run(str(tmp_path), ckan=ckan, allowed=allowed,
                                 extra=["--allow-web", "--download"])
        second, _collision = _run(str(tmp_path), ckan=ckan, allowed=allowed,
                                  extra=["--allow-web", "--download"])
        assert first[0]["download_status"] == "DOWNLOAD_OK"
        assert second[0]["download_status"] == "ALREADY_EXISTS_SAME_URL_SAME_HASH"

    def test_different_urls_same_content_are_duplicate_content(self, tmp_path):
        allowed = os.path.join(str(tmp_path), "allowed.yaml")
        _write_allowed(allowed)
        src_a = tmp_path / "a" / "one.csv"
        src_b = tmp_path / "b" / "two.csv"
        src_a.parent.mkdir()
        src_b.parent.mkdir()
        src_a.write_text("id\nsame\n", encoding="utf-8")
        src_b.write_text("id\nsame\n", encoding="utf-8")
        ckan = os.path.join(str(tmp_path), "ckan.csv")
        cols = ["event_id", "resource_url", "resource_format", "is_geospatial_candidate",
                "is_contextual_only", "resource_id"]
        _write(ckan, cols, [
            {"event_id": "REC_2022_05_24_30", "resource_url": src_a.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "A"},
            {"event_id": "REC_2022_05_24_30", "resource_url": src_b.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "B"},
        ])
        rows, _collision = _run(str(tmp_path), ckan=ckan, allowed=allowed,
                                extra=["--allow-web", "--download"])
        assert rows[0]["download_status"] == "DOWNLOAD_OK"
        assert rows[1]["download_status"] == "DUPLICATE_CONTENT_DIFFERENT_URL"

    def test_manifest_keeps_raw_local_and_public_paths_sanitized(self, tmp_path):
        allowed = os.path.join(str(tmp_path), "allowed.yaml")
        _write_allowed(allowed)
        src = tmp_path / "source" / "asset.csv"
        src.parent.mkdir()
        src.write_text("id\n1\n", encoding="utf-8")
        ckan = os.path.join(str(tmp_path), "ckan.csv")
        cols = ["event_id", "resource_url", "resource_format", "is_geospatial_candidate",
                "is_contextual_only", "resource_id"]
        _write(ckan, cols, [
            {"event_id": "REC_2022_05_24_30", "resource_url": src.as_uri(),
             "resource_format": "CSV", "is_geospatial_candidate": "true",
             "is_contextual_only": "false", "resource_id": "A"},
        ])
        rows, collision = _run(str(tmp_path), ckan=ckan, allowed=allowed,
                               extra=["--allow-web", "--download"])
        assert rows[0]["safe_filename"]
        assert not os.path.isabs(rows[0]["safe_filename"])
        assert not any(str(tmp_path) in r["current_local_target"] for r in collision)
        raw_files = list((tmp_path / "raw").rglob("*"))
        assert all("raw" in str(p) for p in raw_files if p.is_file())
