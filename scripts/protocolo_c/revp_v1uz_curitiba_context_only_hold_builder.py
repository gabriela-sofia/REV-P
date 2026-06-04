#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uz_common import parse_args, run_curitiba_context_only_hold_builder
except ModuleNotFoundError:
    from revp_v1uz_common import parse_args, run_curitiba_context_only_hold_builder


if __name__ == "__main__":
    run_curitiba_context_only_hold_builder(parse_args())
