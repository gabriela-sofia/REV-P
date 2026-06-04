#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_download_target_builder
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_download_target_builder


if __name__ == "__main__":
    run_download_target_builder(parse_args())
