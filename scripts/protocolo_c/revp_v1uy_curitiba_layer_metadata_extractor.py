#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_layer_metadata_extractor
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_layer_metadata_extractor


if __name__ == "__main__":
    run_layer_metadata_extractor(parse_args())
