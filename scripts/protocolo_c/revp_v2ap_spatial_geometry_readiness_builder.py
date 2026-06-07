#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ap_common import parse_args, run_spatial_geometry_readiness_builder
except ModuleNotFoundError:
    from revp_v2ap_common import parse_args, run_spatial_geometry_readiness_builder


if __name__ == "__main__":
    run_spatial_geometry_readiness_builder(parse_args())
