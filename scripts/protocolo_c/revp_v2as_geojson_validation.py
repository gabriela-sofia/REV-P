#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_geojson_validation
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_geojson_validation


if __name__ == "__main__":
    run_geojson_validation(parse_args())
