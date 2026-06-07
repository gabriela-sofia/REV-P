#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_event_geometry_candidate_builder
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_event_geometry_candidate_builder


if __name__ == "__main__":
    run_event_geometry_candidate_builder(parse_args())
