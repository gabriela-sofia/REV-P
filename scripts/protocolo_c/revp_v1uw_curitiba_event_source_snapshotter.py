#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uw_curitiba_common import parse_args, run_event_source_snapshotter
except ModuleNotFoundError:
    from revp_v1uw_curitiba_common import parse_args, run_event_source_snapshotter


if __name__ == "__main__":
    run_event_source_snapshotter(parse_args())
