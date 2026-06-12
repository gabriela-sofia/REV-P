#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_geometry_payload_detector
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_geometry_payload_detector


if __name__ == "__main__":
    run_geometry_payload_detector(parse_args())
