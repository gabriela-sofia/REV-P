from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import install_base_inputs, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_endpoint_probe_records_metadata_without_feature_download(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, v1ux_raw)

    def fake_fetch(url, timeout):
        return "200", "application/json", b'{"name":"layer","fields":[]}'

    monkeypatch.setattr(common, "fetch_head_or_sample", fake_fetch)
    rows = common.run_geodata_endpoint_probe(common.parse_args(["--allow-web"]))
    assert rows
    assert all(r["supports_feature_query"] == "false" for r in rows if r["endpoint_type"] not in {"ARCGIS_FEATURESERVER", "ARCGIS_MAPSERVER", "GEOSERVER_WFS"})
    assert all(r["raw_data_versioned"] == "false" for r in rows)
