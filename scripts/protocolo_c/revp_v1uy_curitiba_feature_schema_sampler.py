#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_feature_schema_sampler
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_feature_schema_sampler


if __name__ == "__main__":
    run_feature_schema_sampler(parse_args())
