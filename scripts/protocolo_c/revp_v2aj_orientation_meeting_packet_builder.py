#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aj_common import parse_args, run_orientation_meeting_packet_builder
except ModuleNotFoundError:
    from revp_v2aj_common import parse_args, run_orientation_meeting_packet_builder


if __name__ == "__main__":
    run_orientation_meeting_packet_builder(parse_args())
