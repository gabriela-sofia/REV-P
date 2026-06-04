#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uv_curitiba_common import parse_args, run_candidate_event_builder
except ModuleNotFoundError:
    from revp_v1uv_curitiba_common import parse_args, run_candidate_event_builder


if __name__ == "__main__":
    run_candidate_event_builder(parse_args())
