#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_geojson_candidate_exporter
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_geojson_candidate_exporter


if __name__ == "__main__":
    run_geojson_candidate_exporter(parse_args())
