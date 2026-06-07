#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ap_common import parse_args, run_temporal_window_builder
except ModuleNotFoundError:
    from revp_v2ap_common import parse_args, run_temporal_window_builder


if __name__ == "__main__":
    run_temporal_window_builder(parse_args())
