#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ap_common import parse_args, run_patch_event_link_readiness_builder
except ModuleNotFoundError:
    from revp_v2ap_common import parse_args, run_patch_event_link_readiness_builder


if __name__ == "__main__":
    run_patch_event_link_readiness_builder(parse_args())
