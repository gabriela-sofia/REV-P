#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uw_curitiba_common import parse_args, run_geocuritiba_layer_deepener
except ModuleNotFoundError:
    from revp_v1uw_curitiba_common import parse_args, run_geocuritiba_layer_deepener


if __name__ == "__main__":
    run_geocuritiba_layer_deepener(parse_args())
