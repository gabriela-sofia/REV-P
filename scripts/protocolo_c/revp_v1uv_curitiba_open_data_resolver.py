#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uv_curitiba_common import parse_args, run_open_data_resolver
except ModuleNotFoundError:
    from revp_v1uv_curitiba_common import parse_args, run_open_data_resolver


if __name__ == "__main__":
    run_open_data_resolver(parse_args())
