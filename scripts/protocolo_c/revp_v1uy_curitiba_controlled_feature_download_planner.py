#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_controlled_feature_download_planner
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_controlled_feature_download_planner


if __name__ == "__main__":
    run_controlled_feature_download_planner(parse_args())
