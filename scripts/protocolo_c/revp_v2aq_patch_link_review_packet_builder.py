#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_patch_link_review_packet_builder
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_patch_link_review_packet_builder


if __name__ == "__main__":
    run_patch_link_review_packet_builder(parse_args())
