#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_event_table_detector
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_event_table_detector


if __name__ == "__main__":
    run_event_table_detector(parse_args())
