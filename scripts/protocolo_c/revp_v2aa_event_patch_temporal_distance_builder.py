#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_event_patch_temporal_distance_builder
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_event_patch_temporal_distance_builder


if __name__ == "__main__":
    run_event_patch_temporal_distance_builder(parse_args())
