#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2al_common import parse_args, run_orientador_review_packet_builder
except ModuleNotFoundError:
    from revp_v2al_common import parse_args, run_orientador_review_packet_builder


if __name__ == "__main__":
    run_orientador_review_packet_builder(parse_args())
