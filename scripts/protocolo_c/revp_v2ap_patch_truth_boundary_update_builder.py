#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ap_common import parse_args, run_patch_truth_boundary_update_builder
except ModuleNotFoundError:
    from revp_v2ap_common import parse_args, run_patch_truth_boundary_update_builder


if __name__ == "__main__":
    run_patch_truth_boundary_update_builder(parse_args())
