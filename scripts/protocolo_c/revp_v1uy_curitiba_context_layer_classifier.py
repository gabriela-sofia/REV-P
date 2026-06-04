#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_context_layer_classifier
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_context_layer_classifier


if __name__ == "__main__":
    run_context_layer_classifier(parse_args())
