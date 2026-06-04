#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_possible_occurrence_layer_audit
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_possible_occurrence_layer_audit


if __name__ == "__main__":
    run_possible_occurrence_layer_audit(parse_args())
