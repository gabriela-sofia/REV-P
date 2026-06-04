#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uz_common import parse_args, run_multiregion_closure_status_builder
except ModuleNotFoundError:
    from revp_v1uz_common import parse_args, run_multiregion_closure_status_builder


if __name__ == "__main__":
    run_multiregion_closure_status_builder(parse_args())
